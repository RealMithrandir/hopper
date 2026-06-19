"""
engine.py — Orchestrates one inference across the layer-sharded pipeline.

Flow per generated token:
    activation enters stage 0's node -> hop -> stage 1's node -> ... -> stage N-1
    -> logits -> sample -> the new token re-enters the pipeline at stage 0.

KV cache stays put on each node; only the [n_tokens, d_model] activation hops.
Every hop is metered (bytes, latency, FLOPs), credited in the ledger, logged for
audit, and spot-checked by the verifier. If a node is slashed mid-stream the
pipeline is re-assembled around it and generation continues.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from model import ModelConfig, decode, encode
from transport import NetworkMonitor


@dataclass
class GenStats:
    tokens: int = 0
    audits: int = 0
    audit_fails: int = 0
    reroutes: int = 0
    network: dict = field(default_factory=dict)


class Engine:
    def __init__(self, cfg, nodes, router, ledger, verifier, monitor=None,
                 n_stages=None, rng=None):
        self.cfg = cfg
        self.nodes = nodes                  # node_id -> Node
        self.router = router
        self.ledger = ledger
        self.verifier = verifier
        self.mon = monitor or NetworkMonitor()
        self.n_stages = n_stages
        self.rng = rng or np.random.default_rng(0)

    def _pipeline(self, exclude):
        return self.router.assemble(self.n_stages, exclude=exclude)

    def _run_token(self, session, x, pipeline, client_id, stats, base_pos):
        """Push one activation through every stage; return logits."""
        excluded = set()
        s = 0
        while s < self.n_stages:
            node = self.nodes[pipeline[s]]
            # transport cost of the INCOMING activation (skip for the very first
            # producer, which receives token ids locally from the client)
            nbytes = int(np.asarray(x).astype(np.float32).nbytes)
            self.mon.transfer(nbytes, node.link)

            out, ex, flops = node.run_stage(s, session, x)
            self.ledger.record(provider=node.id, consumer=client_id, flops=flops)

            # counterfactual: KV cache we did NOT have to ship for this stage
            self.mon.note_kv_avoided(node.kv_footprint_bytes(s, session))

            audit = self.verifier.maybe_audit(ex)
            if audit.audited:
                stats.audits += 1
                if not audit.passed:
                    stats.audit_fails += 1
                    # node slashed by verifier; re-route this stage and replay
                    excluded.add(node.id)
                    stats.reroutes += 1
                    pipeline = self._pipeline(excluded)
                    # caches on the honest replacement are empty; rebuild this
                    # session by replaying is out of scope for the demo, so we
                    # restart the token on the fresh pipeline from this stage.
                    continue
            x = out
            s += 1
        return x, pipeline

    def generate(self, prompt: str, max_tokens: int = 16, client_id: str = "client",
                 temperature: float = 0.0):
        session = uuid.uuid4().hex[:8]
        stats = GenStats()
        pipeline = self._pipeline(exclude=set())

        ids = encode(prompt)
        logits, pipeline = self._run_token(session, ids, pipeline, client_id,
                                           stats, base_pos=0)
        out_ids = []
        for step in range(max_tokens):
            row = logits[-1]
            if temperature <= 0:
                nxt = int(row.argmax())
            else:
                p = np.exp((row - row.max()) / temperature)
                p /= p.sum()
                nxt = int(self.rng.choice(len(p), p=p))
            out_ids.append(nxt)
            logits, pipeline = self._run_token(
                session, np.array([nxt]), pipeline, client_id, stats,
                base_pos=len(ids) + step)

        stats.tokens = len(out_ids)
        stats.network = self.mon.report()
        return decode(out_ids), out_ids, stats