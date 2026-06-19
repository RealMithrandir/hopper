//! `hopper-ledger` — L2 Tokenless Access Pact accounting.
//!
//! Phase 0 scaffold: intentionally empty. Phase 2 ports `reference/ledger.py`
//! here — hashcash identity minting, decayed FLOP contribution ratios
//! (`λ = ln2/3600`), the four access tiers, optimistic-unchoke as a *floor*, and
//! stake-slash + identity-ban enforcement (Invariant 5).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 2 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
