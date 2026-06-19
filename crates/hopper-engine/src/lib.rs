//! `hopper-engine` — orchestration of one inference across the pipeline.
//!
//! Phase 0 scaffold: intentionally empty. Phase 2 ports `reference/engine.py`
//! here — prefill then decode, metering bytes/latency/FLOPs per hop, logging for
//! audit, spot-checking via the verifier, and re-assembling the pipeline around a
//! node slashed mid-stream (Invariants 1, 2, 4, 5, 7).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 2 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
