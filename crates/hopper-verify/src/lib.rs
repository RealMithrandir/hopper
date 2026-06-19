//! `hopper-verify` — L1.5 proof-of-honest-inference.
//!
//! Phase 0 scaffold: intentionally empty. Phase 2 ports `reference/verify.py`
//! here — `commit(output, nonce)`, the logged `Execution`, and the random
//! re-execution audit that compares relative-L2 error to a hardware-calibrated
//! tolerance band (never zero) before slashing (Invariant 4).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 2 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
