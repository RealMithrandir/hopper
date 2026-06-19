//! Coordinator role (an unprivileged daemon role): discover stage providers via
//! Kademlia, then drive the token loop — each stage hop is a QUIC request-response
//! to the chosen worker. On a failed request (peer gone), exclude it and re-select
//! a provider: coordinator-driven reroute, mirroring the reference engine.

use std::collections::HashSet;
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use libp2p::kad::RecordKey;
use libp2p::{Multiaddr, PeerId};
use ndarray::Array2;

use hopper_model::{decode, encode, Activation};
use hopper_net::stage_key;

use crate::net::{self, Client};
use crate::{convert, keypair_from_seed, CoordinatorArgs};

fn parse_bootstrap(s: &str) -> Result<(PeerId, Multiaddr)> {
    let (pid, addr) = s
        .split_once('@')
        .ok_or_else(|| anyhow!("expected `peerid@multiaddr`, got `{s}`"))?;
    Ok((pid.parse()?, addr.parse()?))
}

fn argmax_last(logits: &Array2<f32>) -> usize {
    let row = logits.row(logits.nrows() - 1);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Poll Kademlia for providers of `key` until some are found or `timeout` elapses.
async fn discover(client: &Client, key: RecordKey, timeout: Duration) -> Result<Vec<PeerId>> {
    let start = Instant::now();
    loop {
        let found = client.get_providers(key.clone()).await?;
        if !found.is_empty() {
            let mut peers: Vec<PeerId> = found.into_iter().collect();
            peers.sort();
            return Ok(peers);
        }
        if start.elapsed() > timeout {
            return Ok(Vec::new());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    client: &Client,
    providers: &[Vec<PeerId>],
    session: &str,
    input: Activation,
    dead: &mut HashSet<PeerId>,
    reroutes: &mut usize,
    seq: &mut u64,
) -> Result<Array2<f32>> {
    let n_in = input.n_tokens() as u64;
    let mut current = input;
    let mut logits: Option<Array2<f32>> = None;

    for (sid, stage_providers) in providers.iter().enumerate() {
        loop {
            let Some(peer) = stage_providers.iter().copied().find(|p| !dead.contains(p)) else {
                bail!("stage {sid} has no live provider");
            };
            let req = convert::stage_request(session, sid, *seq, &current);
            match client.send_stage(peer, req).await {
                Ok(resp) => {
                    let out = convert::proto_to_array2(
                        resp.output
                            .as_ref()
                            .ok_or_else(|| anyhow!("empty stage response"))?,
                    )?;
                    if resp.is_logits {
                        logits = Some(out);
                    } else {
                        current = Activation::Hidden(out);
                    }
                    break;
                }
                Err(e) => {
                    *reroutes += 1;
                    dead.insert(peer);
                    println!("REROUTE stage={sid} peer={peer}");
                    std::io::stdout().flush().ok();
                    tracing::warn!("stage {sid} via {peer} failed: {e}; rerouting");
                }
            }
        }
    }

    *seq += n_in;
    logits.ok_or_else(|| anyhow!("pipeline produced no logits"))
}

pub async fn run(args: CoordinatorArgs) -> Result<()> {
    let keypair = keypair_from_seed(args.seed.wrapping_add(1_000_000));
    let peer_id = keypair.public().to_peer_id();
    let swarm = hopper_net::build_swarm(keypair).map_err(|e| anyhow!(e.to_string()))?;
    let (client, event_loop, _inbound) = net::new(swarm);
    tokio::spawn(event_loop.run());

    // A local QUIC endpoint is needed before we can dial.
    client
        .listen("/ip4/127.0.0.1/udp/0/quic-v1".parse()?)
        .await?;

    for entry in &args.bootstrap {
        let (pid, addr) = parse_bootstrap(entry)?;
        client.add_address(pid, addr.clone())?;
        let _ = client.dial(addr).await; // best-effort warm-up
    }
    client.bootstrap()?;

    let mut providers: Vec<Vec<PeerId>> = Vec::with_capacity(args.n_stages);
    for sid in 0..args.n_stages {
        let found = discover(&client, stage_key(sid), Duration::from_secs(20)).await?;
        if found.is_empty() {
            bail!("no provider discovered for stage {sid}");
        }
        providers.push(found);
    }
    if let Some(primary) = providers[0].first() {
        println!("PRIMARY {primary}");
        std::io::stdout().flush().ok();
    }
    tracing::info!(?providers, "discovered stage providers");

    let session = format!("sess-{peer_id}");
    let mut dead: HashSet<PeerId> = HashSet::new();
    let mut reroutes = 0usize;
    let mut seq = 0u64;

    let mut logits = run_pipeline(
        &client,
        &providers,
        &session,
        Activation::Ids(encode(&args.prompt)),
        &mut dead,
        &mut reroutes,
        &mut seq,
    )
    .await?;

    let mut out_ids = Vec::with_capacity(args.max_tokens);
    for i in 0..args.max_tokens {
        let nxt = argmax_last(&logits);
        out_ids.push(nxt);
        println!("TOKEN {i}");
        std::io::stdout().flush().ok();
        if args.step_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(args.step_delay_ms)).await;
        }
        logits = run_pipeline(
            &client,
            &providers,
            &session,
            Activation::Ids(vec![nxt]),
            &mut dead,
            &mut reroutes,
            &mut seq,
        )
        .await?;
    }

    let text = decode(&out_ids);
    println!(
        "DONE tokens={} reroutes={reroutes} text={text:?}",
        out_ids.len()
    );
    std::io::stdout().flush().ok();
    Ok(())
}
