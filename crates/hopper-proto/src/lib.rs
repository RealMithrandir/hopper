//! `hopper-proto` — protobuf wire types for the swarm.
//!
//! Phase 0 scaffold: intentionally empty. Phase 3 adds the `prost` messages from
//! CLAUDE.md's wire-protocol table (`InferenceRequest`, `ActivationStream`,
//! `AuditChallenge`, `AuditReveal`, `FraudProof`, `TokenPublish`) with codegen in
//! `build.rs`. The only inter-stage payload remains the activation (Invariant 1).

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until Phase 3 lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
