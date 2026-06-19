//! `hopper-daemon` — the node binary.
//!
//! Phase 0 scaffold: intentionally a no-op. Later phases grow this into the
//! tokio runtime that loads node identity/keypair + config, hosts stages via
//! `hopper-engine`, and exposes the OpenAI-compatible `POST /v1/chat/completions`
//! route (with a non-standard `hopper` telemetry block) over `axum`.

fn main() {
    // No runtime yet — see the crate docs for what lands here in Phases 2+.
}

#[cfg(test)]
mod tests {
    /// Placeholder so `cargo test` has a green target until the daemon lands.
    #[test]
    fn scaffold_builds() {
        assert_eq!(2 + 2, 4);
    }
}
