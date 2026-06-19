//! `hopper-verify` — proof-of-honest-inference (mirrors `reference/verify.py`).
//!
//! We do not pretend cross-hardware matmuls are bit-reproducible. Instead we make
//! cheating *not pay* (Invariant 4):
//! 1. **commit-reveal** — the worker publishes `H(out || nonce)` before it knows
//!    whether it will be audited, so it cannot adapt its answer.
//! 2. **random re-execution** — with probability `p` an independent verifier
//!    re-runs the stage *statelessly* (the audit invariant of `hopper-model`).
//! 3. **calibrated tolerance** — comparison is a *relative-L2 band* sized to the
//!    verifier's hardware class, never zero. Honest drift lands inside; fabricated
//!    output lands far outside.
//! 4. **stake + slash** — a single catch forfeits the whole stake and bans the
//!    identity, making cheating negative-EV even at small `p`.

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

use hopper_ledger::Ledger;
use hopper_model::{shard, Activation, ModelConfig, Stage, Weights};

/// Errors surfaced while auditing.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error(transparent)]
    Model(#[from] hopper_model::ModelError),
}

/// `commit = sha256(output_bytes ‖ nonce)`, hex-encoded. Output bytes are the
/// f32 little-endian elements in row-major (C) order, matching `numpy.tobytes`.
pub fn commit(output: &Array2<f32>, nonce: &str) -> String {
    let mut hasher = Sha256::new();
    for &x in output.iter() {
        hasher.update(x.to_le_bytes());
    }
    hasher.update(nonce.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// A worker's logged, committed claim about one stage run, ready for audit. The
/// I/O is the *cumulative* session stream, so re-execution exercises the audit
/// invariant over the whole sequence.
#[derive(Debug, Clone)]
pub struct Execution {
    pub session: String,
    pub stage_id: usize,
    pub worker: String,
    pub inp: Activation,
    pub out: Array2<f32>,
    pub nonce: String,
    pub commitment: String,
}

/// Build an [`Execution`], binding `out` with `commit(out, nonce)`. The caller
/// (the worker) chooses a fresh, unpredictable `nonce`.
pub fn log_execution(
    session: &str,
    stage_id: usize,
    worker: &str,
    inp: Activation,
    out: Array2<f32>,
    nonce: String,
) -> Execution {
    let commitment = commit(&out, &nonce);
    Execution {
        session: session.to_string(),
        stage_id,
        worker: worker.to_string(),
        inp,
        out,
        nonce,
        commitment,
    }
}

/// Outcome of an audit decision.
#[derive(Debug, Clone)]
pub struct AuditResult {
    pub audited: bool,
    pub passed: bool,
    pub rel_error: f64,
    pub reason: String,
}

fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    let den: f64 = b
        .iter()
        .map(|&y| (y as f64) * (y as f64))
        .sum::<f64>()
        .sqrt()
        + 1e-9;
    num / den
}

/// RMS magnitude `||x|| / sqrt(size)`, used to scale synthetic hardware drift.
fn rms(a: &Array2<f32>) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let ss: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (ss / a.len() as f64).sqrt() as f32
}

/// One standard-normal sample via Box-Muller (avoids a distributions dependency).
fn next_normal(rng: &mut ChaCha8Rng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-9);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// An ephemeral verifier: it owns the (identical) weights and re-executes a
/// challenged stage statelessly. `maybe_audit` takes `&mut Ledger` so a caught
/// cheat is slashed without the verifier holding the ledger long-term.
pub struct Verifier<'w> {
    cfg: ModelConfig,
    weights: &'w Weights,
    bounds: Vec<(usize, usize)>,
    pub audit_prob: f64,
    pub tolerance: f64,
    pub hw_noise: f64,
    rng: ChaCha8Rng,
}

impl<'w> Verifier<'w> {
    /// Build a verifier over the same shard layout the workers use. `hw_noise`
    /// injects synthetic cross-hardware drift (0 to disable).
    pub fn new(
        cfg: ModelConfig,
        weights: &'w Weights,
        n_stages: usize,
        audit_prob: f64,
        tolerance: f64,
        hw_noise: f64,
        seed: u64,
    ) -> Self {
        let bounds = shard(&cfg, n_stages);
        Self {
            cfg,
            weights,
            bounds,
            audit_prob,
            tolerance,
            hw_noise,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// With probability `p`: verify the commitment binds the reveal, re-execute
    /// the stage statelessly, and compare relative-L2 to the tolerance band.
    /// Mismatch (or a broken commitment) slashes the worker.
    pub fn maybe_audit(
        &mut self,
        ex: &Execution,
        ledger: &mut Ledger,
    ) -> Result<AuditResult, VerifyError> {
        if self.rng.random::<f64>() >= self.audit_prob {
            return Ok(AuditResult {
                audited: false,
                passed: true,
                rel_error: 0.0,
                reason: String::new(),
            });
        }

        // 1) the commitment must bind the revealed output
        if commit(&ex.out, &ex.nonce) != ex.commitment {
            ledger.slash(&ex.worker);
            return Ok(AuditResult {
                audited: true,
                passed: false,
                rel_error: f64::INFINITY,
                reason: "commitment_mismatch".to_string(),
            });
        }

        // 2) independent stateless re-execution on the verifier's own weights
        let (lo, hi) = self.bounds[ex.stage_id];
        let stage = Stage::new(self.cfg.clone(), self.weights, lo, hi);
        let mut recomputed = stage.forward(ex.inp.clone(), None, 0)?;

        // 3) optionally emulate cross-hardware fp drift
        if self.hw_noise != 0.0 {
            let scale = rms(&recomputed);
            let noise = self.hw_noise as f32;
            for v in recomputed.iter_mut() {
                *v += next_normal(&mut self.rng) * scale * noise;
            }
        }

        // 4) relative-L2 vs the calibrated band
        let rel = rel_l2(&ex.out, &recomputed);
        if rel <= self.tolerance {
            Ok(AuditResult {
                audited: true,
                passed: true,
                rel_error: rel,
                reason: "ok".to_string(),
            })
        } else {
            ledger.slash(&ex.worker);
            Ok(AuditResult {
                audited: true,
                passed: false,
                rel_error: rel,
                reason: "recompute_mismatch".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hopper_ledger::{Ledger, ManualClock};
    use hopper_model::golden::Golden;
    use std::path::PathBuf;

    fn golden() -> Golden {
        Golden::load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/golden"))
            .expect("load golden fixture")
    }

    fn ledger_with_worker(worker: &str) -> Ledger {
        let mut l = Ledger::new(Box::new(ManualClock::new(0.0)));
        l.register(worker, 8);
        l
    }

    #[test]
    fn honest_node_with_hardware_drift_is_never_falsely_banned() {
        let g = golden();
        let cfg = g.config();
        let weights = g.weights().unwrap();
        // Stage 0 (first): exercise the embedding path too.
        let stage = Stage::new(cfg.clone(), &weights, 0, 2);
        let ids = Activation::Ids(vec![104, 111, 112, 112, 101, 114]);
        let honest = stage.forward(ids.clone(), None, 0).unwrap();

        let mut verifier = Verifier::new(cfg, &weights, 4, 1.0, 2e-3, 2e-4, 3);
        let mut ledger = ledger_with_worker("w");
        for i in 0..25 {
            let ex = log_execution(
                "sess",
                0,
                "w",
                ids.clone(),
                honest.clone(),
                format!("nonce{i}"),
            );
            let r = verifier.maybe_audit(&ex, &mut ledger).unwrap();
            assert!(r.audited && r.passed, "honest audit {i}: {r:?}");
        }
        assert!(!ledger.account("w").unwrap().banned, "zero false bans");
    }

    #[test]
    fn fabricated_output_is_caught_and_slashed() {
        let g = golden();
        let cfg = g.config();
        let weights = g.weights().unwrap();
        let stage = Stage::new(cfg.clone(), &weights, 0, 2);
        let ids = Activation::Ids(vec![104, 111, 112]);
        let honest = stage.forward(ids.clone(), None, 0).unwrap();
        // Fabricate work; the commitment binds the lie.
        let fake = honest.mapv(|x| x + 5.0);
        let ex = log_execution("sess", 0, "cheater", ids, fake, "n".to_string());

        let mut verifier = Verifier::new(cfg, &weights, 4, 1.0, 2e-3, 0.0, 5);
        let mut ledger = ledger_with_worker("cheater");
        let r = verifier.maybe_audit(&ex, &mut ledger).unwrap();
        assert!(r.audited && !r.passed);
        assert_eq!(r.reason, "recompute_mismatch");
        assert!(ledger.account("cheater").unwrap().banned);
        assert_eq!(ledger.account("cheater").unwrap().stake, 0.0);
    }

    #[test]
    fn commitment_tampering_is_caught() {
        let g = golden();
        let cfg = g.config();
        let weights = g.weights().unwrap();
        let stage = Stage::new(cfg.clone(), &weights, 0, 2);
        let ids = Activation::Ids(vec![104, 111, 112]);
        let honest = stage.forward(ids.clone(), None, 0).unwrap();
        // Commit to the honest output, then reveal a different one.
        let mut ex = log_execution("sess", 0, "liar", ids, honest.clone(), "n".to_string());
        ex.out = honest.mapv(|x| x + 1.0); // reveal != commitment

        let mut verifier = Verifier::new(cfg, &weights, 4, 1.0, 2e-3, 0.0, 1);
        let mut ledger = ledger_with_worker("liar");
        let r = verifier.maybe_audit(&ex, &mut ledger).unwrap();
        assert!(r.audited && !r.passed);
        assert_eq!(r.reason, "commitment_mismatch");
        assert!(ledger.account("liar").unwrap().banned);
    }
}
