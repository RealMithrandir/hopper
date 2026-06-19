//! `hopper-net` — L2 transport + router.
//!
//! Phase 0 scaffold: intentionally empty. Phase 2 ports the simulated transport
//! (`LinkProfile`, `NetworkMonitor` with the `kv_ship_bytes_avoided`
//! counterfactual) and the latency-aware `Router` (mirrors
//! `reference/transport.py` + `router.py`); Phase 3 swaps the sim for libp2p.
//! Routing optimizes latency, never bandwidth (Invariant 7), and the only
//! inter-stage payload is the activation (Invariant 1).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 2 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
