"""
transport.py — Simulated peer-to-peer links with honest cost accounting.

In production this is `libp2p` over QUIC. Here it's an in-process model whose
only job is to make the *physics* visible: how many bytes cross the wire and how
long that takes given residential bandwidth. This is where the original spec's
prefill/decode disaggregation dies and layer-sharding wins — the difference is a
number this module reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinkProfile:
    rtt_ms: float = 18.0          # round-trip latency (geo-clustered target <20ms)
    up_mbps: float = 30.0         # residential uplink — the binding constraint


@dataclass
class NetworkMonitor:
    total_bytes: int = 0
    total_ms: float = 0.0
    hops: int = 0
    # counterfactual: what we'd have paid shipping KV cache instead of activations
    kv_ship_bytes_avoided: int = 0

    def transfer(self, n_bytes: int, link: LinkProfile) -> float:
        """Return simulated one-way wall time (ms) and record the cost."""
        serialize_ms = (n_bytes * 8) / (link.up_mbps * 1e6) * 1e3
        ms = link.rtt_ms / 2 + serialize_ms
        self.total_bytes += n_bytes
        self.total_ms += ms
        self.hops += 1
        return ms

    def note_kv_avoided(self, n_bytes: int) -> None:
        self.kv_ship_bytes_avoided += n_bytes

    def report(self) -> dict:
        return {
            "hops": self.hops,
            "activation_MB": round(self.total_bytes / 1e6, 4),
            "network_ms": round(self.total_ms, 1),
            "kv_ship_MB_avoided": round(self.kv_ship_bytes_avoided / 1e6, 2),
        }