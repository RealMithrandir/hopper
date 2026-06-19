"""
router.py — Stage discovery and pipeline assembly.

Abstracts the spec's geographically-clustered Kademlia DHT. The DHT answers one
question: "which live, non-throttled nodes serve stage S, and how close are
they?" The router then assembles the lowest-latency pipeline that covers every
stage 0..N-1, skipping banned/throttled providers and re-routing on failure.

Because the inter-stage payload is a tiny activation, a few extra WAN hops are
affordable — latency, not bandwidth, is what we optimize here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Provider:
    node_id: str
    stage_id: int
    rtt_ms: float


class Router:
    def __init__(self, ledger):
        self.ledger = ledger
        self.providers: dict[int, list[Provider]] = {}

    def announce(self, node_id: str, stage_id: int, rtt_ms: float) -> None:
        self.providers.setdefault(stage_id, []).append(
            Provider(node_id, stage_id, rtt_ms))

    def _eligible(self, p: Provider) -> bool:
        tier, _ = self.ledger.access(p.node_id)
        return tier not in ("blocked",)

    def assemble(self, n_stages: int, exclude: set[str] | None = None):
        """Pick one eligible provider per stage, minimizing RTT. Returns
        list[stage_id -> node_id] or raises if a stage is uncovered."""
        exclude = exclude or set()
        pipeline = []
        for s in range(n_stages):
            cands = [p for p in self.providers.get(s, [])
                     if p.node_id not in exclude and self._eligible(p)]
            if not cands:
                raise RuntimeError(f"no provider for stage {s}")
            best = min(cands, key=lambda p: p.rtt_ms +
                       self.ledger.access(p.node_id)[1])   # rtt + queue delay
            pipeline.append(best.node_id)
        return pipeline