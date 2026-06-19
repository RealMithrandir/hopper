//! Orchestrates one inference across the layer-sharded pipeline. Mirrors
//! `reference/engine.py`.
//!
//! Flow per generated token: an activation enters stage 0's node, hops stage→
//! stage to stage N-1 → logits → sample → the new token re-enters at stage 0
//! (Invariant 2: pipeline over layers on the same token, never a WAN round-trip
//! between tokens). The KV cache stays put on each node; only the tiny activation
//! hops (Invariant 1). Every hop is metered, FLOP-credited, logged, and
//! spot-checked; a node slashed mid-stream is routed around and generation
//! continues.

use std::collections::HashSet;

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use hopper_ledger::Ledger;
use hopper_model::{decode, encode, Activation, ModelConfig};
use hopper_net::{NetworkMonitor, NetworkReport, Router};
use hopper_verify::Verifier;

use crate::error::EngineError;
use crate::node::NodeMap;

/// Per-generation telemetry (mirrors `engine.GenStats`).
#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub tokens: usize,
    pub audits: usize,
    pub audit_fails: usize,
    pub reroutes: usize,
    pub network: Option<NetworkReport>,
}

/// Byte size of an activation as it would cross the wire (f32 elements).
fn transfer_bytes(x: &Activation) -> usize {
    match x {
        Activation::Ids(v) => v.len() * 4,
        Activation::Hidden(h) => h.len() * 4,
    }
}

/// Sample the next token id from the final-row logits.
fn sample(logits: &Array2<f32>, temperature: f64, rng: &mut ChaCha8Rng) -> usize {
    let row = logits.row(logits.nrows() - 1);
    if temperature <= 0.0 {
        // argmax
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    } else {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = row
            .iter()
            .map(|&v| (((v - max) as f64) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

/// The inference orchestrator. It owns the swarm collaborators so the same
/// nodes/ledger/router persist across generations; swap [`Engine::set_verifier`]
/// / [`Engine::reset_monitor`] between scenarios.
pub struct Engine<'w> {
    nodes: NodeMap<'w>,
    router: Router,
    ledger: Ledger,
    verifier: Verifier<'w>,
    mon: NetworkMonitor,
    n_stages: usize,
    rng: ChaCha8Rng,
    session_counter: u64,
}

impl<'w> Engine<'w> {
    /// Build an engine. `_cfg` is accepted for parity with the reference (the
    /// model dims live in the nodes' stages); `seed` seeds sampling.
    pub fn new(
        _cfg: ModelConfig,
        nodes: NodeMap<'w>,
        router: Router,
        ledger: Ledger,
        verifier: Verifier<'w>,
        n_stages: usize,
        seed: u64,
    ) -> Self {
        Self {
            nodes,
            router,
            ledger,
            verifier,
            mon: NetworkMonitor::new(),
            n_stages,
            rng: ChaCha8Rng::seed_from_u64(seed),
            session_counter: 0,
        }
    }

    /// Swap the active verifier (e.g. quiet → drift → strict between scenarios).
    pub fn set_verifier(&mut self, verifier: Verifier<'w>) {
        self.verifier = verifier;
    }

    /// Reset the network monitor (fresh byte/latency accounting per scenario).
    pub fn reset_monitor(&mut self) {
        self.mon = NetworkMonitor::new();
    }

    /// Read-only ledger access (for snapshots/assertions).
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Mutable ledger access (e.g. to register identities before a run).
    pub fn ledger_mut(&mut self) -> &mut Ledger {
        &mut self.ledger
    }

    /// Mutable access to a hosted node (e.g. to flip its tamper hook).
    pub fn node_mut(&mut self, id: &str) -> Option<&mut crate::node::Node<'w>> {
        self.nodes.get_mut(id)
    }

    /// The network monitor (for the transfer report / per-hop assertions).
    pub fn monitor(&self) -> &NetworkMonitor {
        &self.mon
    }

    fn next_session(&mut self) -> String {
        let s = format!("sess-{:08x}", self.session_counter);
        self.session_counter += 1;
        s
    }

    fn assemble(&self, exclude: &HashSet<String>) -> Result<Vec<String>, EngineError> {
        Ok(self.router.assemble(self.n_stages, exclude, &self.ledger)?)
    }

    /// Push one activation through every stage; return `(logits, pipeline)`. The
    /// pipeline may change if a node is slashed mid-stream (the reroute is
    /// returned so subsequent tokens reuse it).
    fn run_token(
        &mut self,
        session: &str,
        input: Activation,
        mut pipeline: Vec<String>,
        client_id: &str,
        stats: &mut GenStats,
    ) -> Result<(Array2<f32>, Vec<String>), EngineError> {
        let mut excluded: HashSet<String> = HashSet::new();
        let mut current = input;
        let mut logits: Option<Array2<f32>> = None;
        let mut s = 0usize;

        while s < self.n_stages {
            let node_id = pipeline[s].clone();

            // Transport cost of the incoming activation.
            let nbytes = transfer_bytes(&current);
            {
                let node = self
                    .nodes
                    .get_mut(&node_id)
                    .ok_or_else(|| EngineError::UnknownNode(node_id.clone()))?;
                self.mon.transfer(nbytes, &node.link);
            }

            // Run the stage; meter FLOPs and the KV-avoided counterfactual.
            let node = self
                .nodes
                .get_mut(&node_id)
                .ok_or_else(|| EngineError::UnknownNode(node_id.clone()))?;
            let (out, ex, flops) = node.run_stage(s, session, &current)?;
            let kv = node.kv_footprint_bytes(s, session);
            self.ledger.record(&node_id, client_id, flops as f64);
            self.mon.note_kv_avoided(kv);

            // Spot-check via the verifier (audits are indistinguishable from
            // real traffic).
            let audit = self.verifier.maybe_audit(&ex, &mut self.ledger)?;
            if audit.audited {
                stats.audits += 1;
                if !audit.passed {
                    stats.audit_fails += 1;
                    stats.reroutes += 1;
                    excluded.insert(node_id);
                    pipeline = self.assemble(&excluded)?;
                    // Replacement starts with an empty cache; restart this stage
                    // on the fresh pipeline (warm-cache replay is Phase-later).
                    continue;
                }
            }

            if s == self.n_stages - 1 {
                logits = Some(out);
            } else {
                current = Activation::Hidden(out);
            }
            s += 1;
        }

        Ok((logits.expect("last stage produced logits"), pipeline))
    }

    /// Generate up to `max_tokens` greedily (or sampled, if `temperature > 0`).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        client_id: &str,
        temperature: f64,
    ) -> Result<(String, Vec<usize>, GenStats), EngineError> {
        let session = self.next_session();
        let mut stats = GenStats::default();
        let mut pipeline = self.assemble(&HashSet::new())?;

        let ids = encode(prompt);
        let (mut logits, pl) = self.run_token(
            &session,
            Activation::Ids(ids),
            pipeline,
            client_id,
            &mut stats,
        )?;
        pipeline = pl;

        let mut out_ids = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let nxt = sample(&logits, temperature, &mut self.rng);
            out_ids.push(nxt);
            let (l, pl) = self.run_token(
                &session,
                Activation::Ids(vec![nxt]),
                pipeline,
                client_id,
                &mut stats,
            )?;
            logits = l;
            pipeline = pl;
        }

        stats.tokens = out_ids.len();
        stats.network = Some(self.mon.report());
        Ok((decode(&out_ids), out_ids, stats))
    }
}
