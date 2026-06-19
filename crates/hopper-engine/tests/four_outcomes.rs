//! Phase 2 gate: a single process simulates a swarm and reproduces `demo.py`'s
//! four outcomes — inference with constant per-hop payload (Inv 1), honest audit
//! with zero false bans (Inv 4), a fraud caught + slashed + rerouted (Inv 4/7),
//! and correct decayed ledger tiers (Inv 5).

use std::path::PathBuf;

use hopper_engine::{Engine, Node, NodeMap, Tamper};
use hopper_ledger::{Ledger, ManualClock, Tier};
use hopper_model::golden::Golden;
use hopper_model::{shard, ModelConfig, Stage, Weights};
use hopper_net::{LinkProfile, Router};
use hopper_verify::Verifier;

fn golden() -> Golden {
    Golden::load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/golden"))
        .expect("load golden fixture")
}

#[allow(clippy::too_many_arguments)]
fn add_node<'w>(
    nodes: &mut NodeMap<'w>,
    ledger: &mut Ledger,
    router: &mut Router,
    cfg: &ModelConfig,
    weights: &'w Weights,
    bounds: &[(usize, usize)],
    id: &str,
    stage_id: usize,
    rtt: f64,
    seed: u64,
) {
    ledger.register(id, 8);
    let mut node = Node::new(id, LinkProfile::new(rtt, 30.0), seed);
    let (lo, hi) = bounds[stage_id];
    node.host(stage_id, Stage::new(cfg.clone(), weights, lo, hi));
    router.announce(id, stage_id, rtt);
    nodes.insert(id.to_string(), node);
}

/// Build the demo's 4-stage swarm (with stage-1/2 backups for reroute) and an
/// engine wired to a quiet verifier. `weights` outlives the returned engine.
fn build_engine<'w>(cfg: &ModelConfig, weights: &'w Weights) -> Engine<'w> {
    let n_stages = 4;
    let bounds = shard(cfg, n_stages);
    let mut ledger = Ledger::new(Box::new(ManualClock::new(0.0)));
    let mut router = Router::new();
    let mut nodes: NodeMap = NodeMap::new();

    let rtts = [9.0, 14.0, 11.0, 17.0];
    for (i, &rtt) in rtts.iter().enumerate() {
        add_node(
            &mut nodes,
            &mut ledger,
            &mut router,
            cfg,
            weights,
            &bounds,
            &format!("node-{i}"),
            i,
            rtt,
            i as u64,
        );
    }
    // Redundant providers so a reroute always has somewhere to land.
    add_node(
        &mut nodes,
        &mut ledger,
        &mut router,
        cfg,
        weights,
        &bounds,
        "node-1b",
        1,
        22.0,
        101,
    );
    add_node(
        &mut nodes,
        &mut ledger,
        &mut router,
        cfg,
        weights,
        &bounds,
        "node-2b",
        2,
        25.0,
        102,
    );
    ledger.register("client", 8);

    let quiet = Verifier::new(cfg.clone(), weights, n_stages, 0.0, 2e-3, 0.0, 1);
    Engine::new(cfg.clone(), nodes, router, ledger, quiet, n_stages, 7)
}

#[test]
fn reproduces_demo_four_outcomes() {
    let cfg = ModelConfig::default();
    let weights = golden().weights().unwrap();
    let mut engine = build_engine(&cfg, &weights);

    // 1) INFERENCE: constant per-hop payload; KV cache pinned (Invariant 1).
    let (_text, _ids, stats) = engine
        .generate("explain the design", 24, "client", 0.0)
        .unwrap();
    assert_eq!(stats.tokens, 24);
    let net = stats.network.unwrap();
    assert_eq!(net.hops, (1 + 24) * 4, "prefill + 24 decode, 4 stages");
    // The big thing we did NOT ship dwarfs the small thing we did.
    assert!(
        net.kv_ship_mb_avoided > net.activation_mb,
        "kv avoided {} should exceed activation shipped {}",
        net.kv_ship_mb_avoided,
        net.activation_mb
    );

    // 2) HONEST VERIFICATION: every stage re-executed with ~2e-4 drift, 0 bans.
    engine.set_verifier(Verifier::new(cfg.clone(), &weights, 4, 1.0, 2e-3, 2e-4, 3));
    engine.reset_monitor();
    let (_t, _i, st) = engine.generate("audit me", 6, "client", 0.0).unwrap();
    assert!(st.audits > 0, "audits should happen at p=1.0");
    assert_eq!(st.audit_fails, 0, "honest drift stays inside the band");
    assert_eq!(st.reroutes, 0);
    for id in ["node-0", "node-1", "node-2", "node-3", "node-1b", "node-2b"] {
        assert!(
            !engine.ledger().account(id).unwrap().banned,
            "{id} falsely banned"
        );
    }

    // 3) FRAUD: stage-1 primary fabricates -> caught, slashed, routed around.
    engine.node_mut("node-1").unwrap().tamper = Tamper::AddConstant(5.0);
    engine.set_verifier(Verifier::new(cfg.clone(), &weights, 4, 1.0, 2e-3, 0.0, 5));
    engine.reset_monitor();
    let (_t, ids3, st3) = engine
        .generate("catch the cheater", 4, "client", 0.0)
        .unwrap();
    assert!(st3.audit_fails >= 1, "the cheater must be caught");
    assert!(st3.reroutes >= 1, "and routed around");
    assert_eq!(ids3.len(), 4, "generation still finishes on the backup");
    let cheater = engine.ledger().account("node-1").unwrap();
    assert!(cheater.banned, "cheater banned");
    assert_eq!(cheater.stake, 0.0, "stake slashed");

    // 4) LEDGER: providers priority; cheater blocked; pure consumer unchoked.
    let snap = engine.ledger().snapshot();
    for id in ["node-0", "node-2", "node-3", "node-1b"] {
        assert_eq!(snap.get(id).unwrap().tier, Tier::Priority, "{id} tier");
    }
    assert_eq!(snap.get("node-1").unwrap().tier, Tier::Blocked);
    assert!(snap.get("node-1").unwrap().banned);
    assert_eq!(
        snap.get("client").unwrap().tier,
        Tier::OptimisticUnchoke,
        "pure consumer rides its grace grant"
    );
}

#[test]
fn per_hop_hidden_activation_is_d_model_times_4() {
    // Invariant 1 guard: the inter-stage activation is `d_model * 4` bytes during
    // decode, regardless of how much context has accumulated.
    let cfg = ModelConfig::default();
    let weights = golden().weights().unwrap();
    let mut engine = build_engine(&cfg, &weights); // quiet verifier -> no reroutes
    let max_tokens = 12;

    let (_t, _i, _s) = engine
        .generate("a longer prompt to grow context", max_tokens, "client", 0.0)
        .unwrap();

    let expected = cfg.d_model * 4;
    let hop_bytes = &engine.monitor().hop_bytes;
    assert_eq!(hop_bytes.len(), (1 + max_tokens) * 4);

    // Hop i: pass = i/4 (0 = prefill), stage = i%4. Every non-first stage during
    // a decode pass carries exactly one [1, d_model] row.
    for (i, &bytes) in hop_bytes.iter().enumerate() {
        let (pass, stage) = (i / 4, i % 4);
        if pass >= 1 && stage != 0 {
            assert_eq!(
                bytes, expected,
                "decode hidden hop {i} (pass {pass}, stage {stage})"
            );
        }
    }
}
