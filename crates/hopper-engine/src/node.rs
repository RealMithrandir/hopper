//! A peer in the swarm. Mirrors `reference/node.py`.
//!
//! A node owns one or more contiguous [`Stage`]s and holds the KV cache for them
//! *locally and persistently* across a session — the cache never leaves the node
//! (Invariant 1); only the small activation does. For auditability the node also
//! retains, per session, the cumulative stream of inputs/outputs, so an auditor
//! can demand it and re-execute the stage statelessly (see `hopper-verify`).

use std::collections::HashMap;

use ndarray::{concatenate, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use hopper_model::{Activation, KVCache, ModelError, Stage};
use hopper_net::LinkProfile;
use hopper_verify::{log_execution, Execution};

use crate::error::EngineError;

/// How a node corrupts its output. Honest nodes use [`Tamper::None`]; a cheater
/// fabricates work, which the commitment then binds — exactly what an auditor
/// must catch.
#[derive(Debug, Clone, Copy, Default)]
pub enum Tamper {
    #[default]
    None,
    /// Add a constant to every output element (the demo's fabrication).
    AddConstant(f32),
}

impl Tamper {
    fn apply(self, out: Array2<f32>) -> Array2<f32> {
        match self {
            Tamper::None => out,
            Tamper::AddConstant(c) => out.mapv(|x| x + c),
        }
    }
}

/// One stage hosted on this node, with its per-session local cache and I/O logs.
struct HostedStage<'w> {
    stage: Stage<'w>,
    n_layers: usize,
    caches: HashMap<String, Vec<KVCache>>,
    inp_log: HashMap<String, Vec<Activation>>,
    out_log: HashMap<String, Vec<Array2<f32>>>,
}

/// The swarm's nodes, keyed by id (what the [`crate::Engine`] orchestrates over).
pub type NodeMap<'w> = HashMap<String, Node<'w>>;

/// A swarm peer.
pub struct Node<'w> {
    pub id: String,
    pub link: LinkProfile,
    pub flops_served: u64,
    /// Honest by default; set to fabricate output (used to exercise auditing).
    pub tamper: Tamper,
    hosted: HashMap<usize, HostedStage<'w>>,
    rng: ChaCha8Rng,
}

impl<'w> Node<'w> {
    /// A node identified by `id` with link profile `link`; `seed` seeds the
    /// (reproducible) nonce generator used for commitments.
    pub fn new(id: &str, link: LinkProfile, seed: u64) -> Self {
        Self {
            id: id.to_string(),
            link,
            flops_served: 0,
            tamper: Tamper::None,
            hosted: HashMap::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Host `stage` under `stage_id`.
    pub fn host(&mut self, stage_id: usize, stage: Stage<'w>) {
        let (lo, hi) = stage.layer_range();
        self.hosted.insert(
            stage_id,
            HostedStage {
                stage,
                n_layers: hi - lo,
                caches: HashMap::new(),
                inp_log: HashMap::new(),
                out_log: HashMap::new(),
            },
        );
    }

    /// Execute the hosted stage online (local KV cache), log the cumulative I/O
    /// stream, and emit an [`Execution`] for audit. Returns `(out, execution,
    /// flops)`. `x` is borrowed so the engine can re-send it on a reroute.
    pub fn run_stage(
        &mut self,
        stage_id: usize,
        session: &str,
        x: &Activation,
    ) -> Result<(Array2<f32>, Execution, u64), EngineError> {
        let hs = self
            .hosted
            .get_mut(&stage_id)
            .ok_or(EngineError::NoStage(stage_id))?;

        // Lazily create the per-session cache and logs.
        let n_layers = hs.n_layers;
        hs.caches
            .entry(session.to_string())
            .or_insert_with(|| (0..n_layers).map(|_| KVCache::new()).collect());
        hs.inp_log.entry(session.to_string()).or_default();
        hs.out_log.entry(session.to_string()).or_default();

        let caches = hs.caches.get_mut(session).expect("just inserted");
        let base_pos = caches[0].length();
        let n_tokens = x.n_tokens();

        // Tamper (if any) corrupts BOTH the returned output and what is logged /
        // committed — the cheater commits to its lie.
        let out = self.tamper.apply(hs.stage.forward(
            x.clone(),
            Some(caches.as_mut_slice()),
            base_pos,
        )?);

        hs.inp_log
            .get_mut(session)
            .expect("init above")
            .push(x.clone());
        hs.out_log
            .get_mut(session)
            .expect("init above")
            .push(out.clone());

        let flops = hs.stage.flops(n_tokens) as u64;
        self.flops_served += flops;

        // Cumulative session stream, exactly what an auditor re-executes.
        let cum_in = Activation::concat(&hs.inp_log[session])?;
        let out_views: Vec<_> = hs.out_log[session].iter().map(|a| a.view()).collect();
        let cum_out = concatenate(Axis(0), &out_views).map_err(ModelError::from)?;

        let nonce = format!("{:016x}", self.rng.random::<u64>());
        let ex = log_execution(session, stage_id, &self.id, cum_in, cum_out, nonce);
        Ok((out, ex, flops))
    }

    /// Size in bytes of the KV cache we are *not* shipping for `(stage, session)`
    /// — the counterfactual that quantifies the layer-sharding win (Invariant 1).
    pub fn kv_footprint_bytes(&self, stage_id: usize, session: &str) -> usize {
        let Some(hs) = self.hosted.get(&stage_id) else {
            return 0;
        };
        let Some(caches) = hs.caches.get(session) else {
            return 0;
        };
        caches
            .iter()
            .map(|c| {
                let k = c.k().map_or(0, |a| a.len() * 4);
                let v = c.v().map_or(0, |a| a.len() * 4);
                k + v
            })
            .sum()
    }
}
