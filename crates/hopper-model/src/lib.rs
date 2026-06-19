//! `hopper-model` — L1 layer-sharded transformer with a node-resident KV cache.
//!
//! Phase 0 scaffold: intentionally empty. Phase 1 ports `reference/model.py`
//! here — deterministic weight materialization from `model_hash`, `Stage::forward`
//! with an `Option<&mut [KVCache]>` (online decode vs. stateless audit recompute),
//! the golden-parity test, and the cache↔stateless property test (Invariant 3).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 1 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
