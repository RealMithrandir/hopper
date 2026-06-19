//! `hopper-daemon` binary — a thin CLI over the library roles.

use anyhow::Result;
use clap::{Parser, Subcommand};

use hopper_daemon::{coordinator, facade, worker, CoordinatorArgs, ServeArgs, WorkerArgs};

#[derive(Parser)]
#[command(
    name = "hopper-daemon",
    about = "HOPPER node daemon (worker / coordinator / serve)"
)]
struct Cli {
    #[command(subcommand)]
    role: Role,
}

#[derive(Subcommand)]
enum Role {
    /// Host stages and serve activations over libp2p.
    Worker(WorkerArgs),
    /// Discover providers and drive an inference.
    Coordinator(CoordinatorArgs),
    /// Serve the OpenAI-compatible HTTP facade over an in-process engine.
    Serve(ServeArgs),
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
        Role::Serve(args) => facade::run(args).await,
    }
}
