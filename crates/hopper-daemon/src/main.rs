//! `hopper-daemon` — the HOPPER node binary.
//!
//! One binary, two unprivileged roles (CLAUDE.md §"Phase 3 scope note"):
//! * `worker` hosts contiguous stages, pins their KV caches, and serves
//!   `ActivationStream` over a libp2p QUIC request-response protocol;
//! * `coordinator` discovers stage providers via Kademlia and drives the token
//!   loop, rerouting around a worker that dies mid-stream.
//!
//! The OpenAI-compatible HTTP facade + config-file loader arrive in Phase 4.

mod convert;
mod coordinator;
mod net;
mod worker;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use libp2p::identity::Keypair;

#[derive(Parser)]
#[command(
    name = "hopper-daemon",
    about = "HOPPER node daemon (worker / coordinator)"
)]
struct Cli {
    #[command(subcommand)]
    role: Role,
}

#[derive(Subcommand)]
enum Role {
    /// Host stages and serve activations.
    Worker(WorkerArgs),
    /// Discover providers and drive an inference.
    Coordinator(CoordinatorArgs),
}

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

/// Deterministic ed25519 keypair from a `u64` seed (reproducible PeerIds in tests).
pub fn keypair_from_seed(seed: u64) -> Keypair {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    Keypair::ed25519_from_bytes(bytes).expect("valid ed25519 key material")
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().role {
        Role::Worker(args) => worker::run(args).await,
        Role::Coordinator(args) => coordinator::run(args).await,
    }
}
