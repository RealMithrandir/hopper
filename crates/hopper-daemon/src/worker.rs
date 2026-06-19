//! Worker role: host contiguous stages, pin their KV caches locally, and serve
//! `ActivationStream` requests over libp2p. Reuses `hopper_engine::Node` for the
//! per-session cache + audit-commitment bookkeeping. Only the activation crosses
//! the wire (Invariant 1).

use std::io::Write;

use anyhow::{anyhow, Result};

use hopper_engine::Node;
use hopper_model::{golden::Golden, shard, Stage, Weights};
use hopper_net::{stage_key, LinkProfile};
use hopper_proto::StageResponse;

use crate::net;
use crate::{convert, keypair_from_seed, WorkerArgs};

pub async fn run(args: WorkerArgs) -> Result<()> {
    let keypair = keypair_from_seed(args.seed);
    let peer_id = keypair.public().to_peer_id();
    let swarm = hopper_net::build_swarm(keypair).map_err(|e| anyhow!(e.to_string()))?;
    let (client, event_loop, mut inbound_rx) = net::new(swarm);
    tokio::spawn(event_loop.run());

    // Load the canonical weights and host the assigned stages. Leaking the
    // weights gives the hosted `Stage`s a `'static` borrow for the daemon's life.
    let golden = Golden::load(&args.golden)?;
    let cfg = golden.config();
    let weights: &'static Weights = Box::leak(Box::new(golden.weights()?));
    let bounds = shard(&cfg, args.n_stages);

    let mut node = Node::new(
        &format!("worker-{peer_id}"),
        LinkProfile::default(),
        args.seed,
    );
    for &sid in &args.stages {
        let (lo, hi) = bounds[sid];
        node.host(sid, Stage::new(cfg.clone(), weights, lo, hi));
    }

    let bound = client.listen(args.listen.parse()?).await?;
    for &sid in &args.stages {
        client.start_providing(stage_key(sid)).await?;
    }
    // Machine-readable handshake line for the integration harness.
    println!("PEER {peer_id} ADDR {bound}");
    std::io::stdout().flush().ok();
    tracing::info!(%peer_id, %bound, stages = ?args.stages, "worker ready");

    while let Some(inbound) = inbound_rx.recv().await {
        let req = inbound.request;
        let sid = req.stage_id as usize;
        let model_in = match convert::request_input_to_model(&req) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("rejecting malformed request: {e}");
                continue;
            }
        };
        match node.run_stage(sid, &req.session, &model_in) {
            Ok((out, ex, flops)) => {
                let resp = StageResponse {
                    output: Some(convert::array2_to_proto(&out)),
                    is_logits: sid + 1 == args.n_stages,
                    commitment: ex.commitment,
                    nonce: ex.nonce,
                    flops,
                };
                let _ = client.respond(inbound.channel, resp);
            }
            Err(e) => tracing::warn!("stage {sid} execution failed: {e}"),
        }
    }
    Ok(())
}
