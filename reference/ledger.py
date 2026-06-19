"""
ledger.py — The Tokenless Access Pact, made workable.

Keeps the spec's contribution-ratio idea but fixes its four real holes:

  * cold start      -> optimistic-unchoke grant for fresh, staked identities
  * whitewashing    -> identity has a one-time proof-of-work cost, so churning
                       to a new id to escape a bad ratio is not free
  * unit of account -> work is metered in FLOPs (compute), not raw token count,
                       so a heavy prefill cannot be farmed as cheaply as a decode
  * "global" ratio  -> reputation is advisory and gossiped, never consensus; the
                       authoritative signal is local tit-for-tat between peers

No blockchain, no token, no global agreement required.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field


def proof_of_work(identity: str, difficulty: int = 16) -> int:
    """Cost to *mint* an identity. Find a nonce s.t. H(id||nonce) has `difficulty`
    leading zero bits. Cheap for the network to check, costly to spam."""
    target = 1 << (256 - difficulty)
    nonce = 0
    while True:
        h = int(hashlib.sha256(f"{identity}:{nonce}".encode()).hexdigest(), 16)
        if h < target:
            return nonce
        nonce += 1


@dataclass
class Account:
    node_id: str
    minted_at: float
    pow_nonce: int
    stake: float = 1.0                       # slashable bond
    seeded: list = field(default_factory=list)   # (t, flops)
    leeched: list = field(default_factory=list)  # (t, flops)
    banned: bool = False


class Ledger:
    LAMBDA = math.log(2) / 3600.0            # ~1h half-life decay
    GRACE_FLOPS = 5e9                        # optimistic-unchoke allowance
    GRACE_WINDOW_S = 1800                    # new nodes get grace for 30 min

    def __init__(self, clock=time.time):
        self.clock = clock
        self.accounts: dict[str, Account] = {}

    # ---- identity --------------------------------------------------------
    def register(self, node_id: str, difficulty: int = 12) -> Account:
        nonce = proof_of_work(node_id, difficulty)
        acct = Account(node_id, self.clock(), nonce)
        self.accounts[node_id] = acct
        return acct

    # ---- work events -----------------------------------------------------
    def record(self, provider: str, consumer: str, flops: int) -> None:
        t = self.clock()
        if provider in self.accounts:
            self.accounts[provider].seeded.append((t, flops))
        if consumer in self.accounts:
            self.accounts[consumer].leeched.append((t, flops))

    def _decayed(self, events) -> float:
        now = self.clock()
        return sum(f * math.exp(-self.LAMBDA * (now - t)) for t, f in events)

    def contribution_ratio(self, node_id: str) -> float:
        a = self.accounts[node_id]
        s = self._decayed(a.seeded)
        l = self._decayed(a.leeched)
        if l <= 0:
            return math.inf
        return s / l

    # ---- access policy (spec §4.2 thresholds, plus grace) ---------------
    def access(self, node_id: str) -> tuple[str, float]:
        """Return (tier, queue_delay_ms). Ratio sets the tier; optimistic-unchoke
        is a FLOOR that rescues fresh nodes who'd otherwise be throttled, so they
        can earn a ratio — it never demotes a node that is already contributing."""
        a = self.accounts.get(node_id)
        if a is None or a.banned:
            return ("blocked", math.inf)

        r = self.contribution_ratio(node_id)
        if r >= 1.0:
            tier, delay = "priority", 0.0
        elif r >= 0.2:
            tier, delay = "delayed", 250.0 * (1.0 - r)
        else:
            tier, delay = "throttled", 2000.0

        if tier == "throttled":
            fresh = (self.clock() - a.minted_at) < self.GRACE_WINDOW_S
            if fresh and self._decayed(a.leeched) < self.GRACE_FLOPS:
                return ("optimistic_unchoke", 0.0)
        return (tier, delay)

    # ---- enforcement -----------------------------------------------------
    def slash(self, node_id: str) -> None:
        a = self.accounts.get(node_id)
        if a:
            a.stake = 0.0
            a.banned = True
            a.seeded.clear()                    # spec: cumulative credit erased

    def snapshot(self) -> dict:
        return {
            nid: {
                "ratio": round(self.contribution_ratio(nid), 2)
                if self.contribution_ratio(nid) != math.inf else "inf",
                "tier": self.access(nid)[0],
                "stake": a.stake,
                "banned": a.banned,
            }
            for nid, a in self.accounts.items()
        }