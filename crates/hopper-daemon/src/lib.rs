//! `hopper-daemon` library — the node's roles and the OpenAI facade, exposed as a
//! library so integration tests can drive them directly (the binary in `main.rs`
//! is a thin CLI wrapper).
//!
//! Roles (CLAUDE.md §"Phase 3 scope note"): `worker` hosts stages and serves
//! activations over libp2p QUIC; `coordinator` discovers providers via Kademlia and
//! drives the token loop with reroute; `serve` exposes the Phase-4 axum
//! OpenAI-compatible facade over an in-process engine.

pub mod convert;
pub mod coordinator;
pub mod facade;
pub mod net;
pub mod worker;

use clap::Args;
use libp2p::identity::Keypair;

/// Deterministic ed25519 keypair from a `u64` seed (reproducible PeerIds in tests).
pub fn keypair_from_seed(seed: u64) -> Keypair {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    Keypair::ed25519_from_bytes(bytes).expect("valid ed25519 key material")
}

/// Worker role arguments.
#[derive(Args)]
pub struct WorkerArgs {
    /// QUIC listen multiaddr (ephemeral port by default).
    #[arg(long, default_value = "/ip4/127.0.0.1/udp/0/quic-v1")]
    pub listen: String,
    /// Stage ids to host (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub stages: Vec<usize>,
    /// Total number of stages in the model.
    #[arg(long, default_value_t = 4)]
    pub n_stages: usize,
    /// Path to the exported golden weights directory.
    #[arg(long)]
    pub golden: String,
    /// Seed for the (deterministic) identity keypair + nonce RNG.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

/// Coordinator role arguments.
#[derive(Args)]
pub struct CoordinatorArgs {
    /// Bootstrap workers as `peerid@multiaddr` (repeatable).
    #[arg(long)]
    pub bootstrap: Vec<String>,
    /// Total number of stages in the model.
    #[arg(long, default_value_t = 4)]
    pub n_stages: usize,
    /// Prompt to generate from.
    #[arg(long)]
    pub prompt: String,
    /// Number of tokens to generate.
    #[arg(long, default_value_t = 8)]
    pub max_tokens: usize,
    /// Optional per-token delay (ms) — lets a test land a mid-stream kill.
    #[arg(long, default_value_t = 0)]
    pub step_delay_ms: u64,
    /// Seed for the (deterministic) identity keypair.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

/// Serve role arguments (the OpenAI-compatible HTTP facade).
#[derive(Args)]
pub struct ServeArgs {
    /// HTTP bind address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
    /// Path to the exported golden weights directory.
    #[arg(long)]
    pub golden: String,
    /// Total number of stages in the (in-process) model.
    #[arg(long, default_value_t = 4)]
    pub n_stages: usize,
}
