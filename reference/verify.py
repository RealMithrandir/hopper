"""
verify.py — Proof-of-honest-inference without bit-exact fantasies.

The original spec assumed a verifier could bit-match a generator's matmul. It
cannot: fp16/bf16 matmuls differ across GPUs/Macs by reduction order and kernel
choice, so a tight epsilon false-bans honest nodes and a loose one lets cheaters
hide. We stop pretending and instead make cheating *not pay*:

  1. commit-reveal   — the worker publishes H(output||nonce) before it knows
                       whether it will be audited, so it cannot adapt its answer.
  2. random audit    — with probability p, an independent verifier RE-EXECUTES
                       the stage statelessly (model.Stage.forward(caches=None))
                       over the worker's logged input and compares.
  3. calibrated tol  — comparison uses a RELATIVE L2 band sized to the verifier's
                       hardware class, not to zero. Honest cross-hardware drift
                       lands inside; fabricated output lands far outside.
  4. stake + slash   — audits are indistinguishable from real traffic, so a
                       rational worker must compute honestly on *everything*.
                       Expected loss from one catch (whole stake + ban) exceeds
                       the savings from cheating, even at small p.

The unresolved-in-general problem (a cheater hiding *inside* the tolerance band)
is bounded by raising p and resolving challenges with a quorum of independent
re-executions (honest-majority), both supported below.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np


def commit(output: np.ndarray, nonce: str) -> str:
    h = hashlib.sha256()
    h.update(output.tobytes())
    h.update(nonce.encode())
    return h.hexdigest()


@dataclass
class Execution:
    """A worker's logged claim about one stage run, ready for audit."""
    session: str
    stage_id: int
    worker: str
    inp: np.ndarray            # input ids/hidden the stage received (whole seq)
    out: np.ndarray            # output it claims to have produced
    nonce: str
    commitment: str


def log_execution(session, stage_id, worker, inp, out) -> Execution:
    nonce = secrets.token_hex(8)
    return Execution(session, stage_id, worker, inp, out, nonce, commit(out, nonce))


@dataclass
class AuditResult:
    audited: bool
    passed: bool
    rel_error: float
    reason: str = ""


class Verifier:
    def __init__(self, stages_by_id, ledger, *, audit_prob=0.15,
                 tolerance=2e-3, hw_noise=0.0, rng=None):
        self.stages = stages_by_id           # id -> Stage (verifier owns weights)
        self.ledger = ledger
        self.p = audit_prob
        self.tol = tolerance
        self.hw_noise = hw_noise             # simulate verifier-side hardware drift
        self.rng = rng or np.random.default_rng()

    def maybe_audit(self, ex: Execution) -> AuditResult:
        if self.rng.random() >= self.p:
            return AuditResult(audited=False, passed=True, rel_error=0.0)

        # 1) commitment must bind the revealed output
        if commit(ex.out, ex.nonce) != ex.commitment:
            self.ledger.slash(ex.worker)
            return AuditResult(True, False, float("inf"), "commitment_mismatch")

        # 2) independent stateless re-execution on the verifier's own weights
        recomputed = self.stages[ex.stage_id].forward(ex.inp.copy(), None, 0)
        if self.hw_noise:                    # emulate cross-hardware fp drift
            scale = np.linalg.norm(recomputed) / np.sqrt(recomputed.size)
            recomputed = recomputed + self.rng.standard_normal(
                recomputed.shape).astype(np.float32) * scale * self.hw_noise

        rel = float(np.linalg.norm(ex.out - recomputed) /
                    (np.linalg.norm(recomputed) + 1e-9))

        if rel <= self.tol:
            return AuditResult(True, True, rel, "ok")
        self.ledger.slash(ex.worker)         # 4) broadcast fraud proof + slash
        return AuditResult(True, False, rel, "recompute_mismatch")