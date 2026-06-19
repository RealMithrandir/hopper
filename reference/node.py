"""
node.py — A peer in the swarm.

A node owns one or more contiguous Stages and holds the KV cache for those stages
*locally and persistently* across a session. The cache never leaves the node;
only the small hidden-state activation does.

For auditability the node also retains, per session, the full stream of inputs it
received and outputs it produced for each hosted stage. An auditor can demand
this stream and re-execute the stage statelessly to check the work (verify.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model import KVCache, Stage
from transport import LinkProfile
from verify import log_execution


@dataclass
class HostedStage:
    stage: Stage
    caches: dict = field(default_factory=dict)     # session -> [KVCache] (local!)
    inp_log: dict = field(default_factory=dict)    # session -> [activation in]
    out_log: dict = field(default_factory=dict)    # session -> [activation out]


class Node:
    def __init__(self, node_id: str, link: LinkProfile | None = None):
        self.id = node_id
        self.link = link or LinkProfile()
        self.hosted: dict[int, HostedStage] = {}
        self.flops_served = 0

    def host(self, stage_id: int, stage: Stage) -> None:
        self.hosted[stage_id] = HostedStage(stage)

    def _session_caches(self, stage_id: int, session: str) -> list[KVCache]:
        hs = self.hosted[stage_id]
        if session not in hs.caches:
            hs.caches[session] = [KVCache() for _ in range(hs.stage.hi - hs.stage.lo)]
            hs.inp_log[session] = []
            hs.out_log[session] = []
        return hs.caches[session]

    # Honest nodes leave the output untouched; a faulty/cheating node overrides
    # this hook. Tampering here corrupts both what is returned downstream AND
    # what gets committed/logged — exactly what an auditor must catch.
    def _tamper(self, out: np.ndarray) -> np.ndarray:
        return out

    def run_stage(self, stage_id: int, session: str, x: np.ndarray):
        """Execute the hosted stage online (local KV cache); log the cumulative
        I/O stream and emit an Execution for audit. Returns (out, Execution, flops)."""
        hs = self.hosted[stage_id]
        caches = self._session_caches(stage_id, session)
        base_pos = caches[0].length()
        out = self._tamper(hs.stage.forward(x, caches, base_pos))

        hs.inp_log[session].append(np.asarray(x).copy())
        hs.out_log[session].append(out.copy())
        flops = hs.stage.flops(np.asarray(x).shape[0])
        self.flops_served += flops

        cum_in = np.concatenate(hs.inp_log[session], axis=0)
        cum_out = np.concatenate(hs.out_log[session], axis=0)
        ex = log_execution(session, stage_id, self.id, cum_in, cum_out)
        return out, ex, flops

    def kv_footprint_bytes(self, stage_id: int, session: str) -> int:
        """Size of the cache we are NOT shipping (the original spec's payload)."""
        total = 0
        for c in self.hosted[stage_id].caches.get(session, []):
            if c.k is not None:
                total += c.k.nbytes + c.v.nbytes
        return total