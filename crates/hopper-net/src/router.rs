//! Stage discovery + latency-aware pipeline assembly. Mirrors
//! `reference/router.py`.
//!
//! Abstracts the spec's geo-clustered Kademlia DHT down to its one question:
//! "which live, non-blocked nodes serve stage S, and how close are they?" Because
//! the inter-stage payload is tiny, we optimize **latency**, not bandwidth
//! (Invariant 7): pick the lowest `rtt + queue_delay` eligible provider per stage,
//! and re-assemble around banned/slashed/dropped nodes.

use std::collections::{HashMap, HashSet};

use hopper_ledger::{Ledger, Tier};

/// Errors from pipeline assembly.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NetError {
    #[error("no eligible provider for stage {0}")]
    NoProvider(usize),
}

/// A node advertising that it serves a given stage, with a measured RTT.
#[derive(Debug, Clone)]
pub struct Provider {
    pub node_id: String,
    pub stage_id: usize,
    pub rtt_ms: f64,
}

/// In-memory stand-in for the DHT: stage_id → advertised providers.
#[derive(Debug, Default)]
pub struct Router {
    providers: HashMap<usize, Vec<Provider>>,
}

impl Router {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advertise that `node_id` serves `stage_id` at `rtt_ms`.
    pub fn announce(&mut self, node_id: &str, stage_id: usize, rtt_ms: f64) {
        self.providers.entry(stage_id).or_default().push(Provider {
            node_id: node_id.to_string(),
            stage_id,
            rtt_ms,
        });
    }

    /// A provider is eligible if it is not banned/blocked by the ledger.
    fn eligible(ledger: &Ledger, node_id: &str) -> bool {
        ledger.access(node_id).tier != Tier::Blocked
    }

    /// Assemble the lowest-latency pipeline covering stages `0..n_stages`,
    /// skipping excluded and ineligible providers. Errors if a stage is uncovered.
    pub fn assemble(
        &self,
        n_stages: usize,
        exclude: &HashSet<String>,
        ledger: &Ledger,
    ) -> Result<Vec<String>, NetError> {
        let mut pipeline = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            let best = self
                .providers
                .get(&stage)
                .into_iter()
                .flatten()
                .filter(|p| !exclude.contains(&p.node_id) && Self::eligible(ledger, &p.node_id))
                .min_by(|a, b| {
                    let cost = |p: &Provider| p.rtt_ms + ledger.access(&p.node_id).queue_delay_ms;
                    cost(a).total_cmp(&cost(b))
                })
                .ok_or(NetError::NoProvider(stage))?;
            pipeline.push(best.node_id.clone());
        }
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hopper_ledger::{Ledger, ManualClock};

    fn ledger_with(nodes: &[&str]) -> Ledger {
        let mut l = Ledger::new(Box::new(ManualClock::new(0.0)));
        for n in nodes {
            l.register(n, 8);
        }
        l
    }

    #[test]
    fn assembles_lowest_latency_then_reroutes() {
        let ledger = ledger_with(&["fast", "slow", "s1"]);
        let mut r = Router::new();
        r.announce("fast", 0, 9.0);
        r.announce("slow", 0, 22.0);
        r.announce("s1", 1, 11.0);

        // lowest RTT wins
        let pipe = r.assemble(2, &HashSet::new(), &ledger).unwrap();
        assert_eq!(pipe, vec!["fast".to_string(), "s1".to_string()]);

        // exclude the primary -> reroute to the backup
        let excl: HashSet<String> = ["fast".to_string()].into_iter().collect();
        let pipe = r.assemble(2, &excl, &ledger).unwrap();
        assert_eq!(pipe, vec!["slow".to_string(), "s1".to_string()]);
    }

    #[test]
    fn skips_blocked_providers() {
        let mut ledger = ledger_with(&["fast", "slow", "s1"]);
        let mut r = Router::new();
        r.announce("fast", 0, 9.0);
        r.announce("slow", 0, 22.0);
        r.announce("s1", 1, 11.0);

        ledger.slash("fast"); // banned -> blocked
        let pipe = r.assemble(2, &HashSet::new(), &ledger).unwrap();
        assert_eq!(pipe, vec!["slow".to_string(), "s1".to_string()]);
    }

    #[test]
    fn uncovered_stage_errors() {
        let ledger = ledger_with(&["only0"]);
        let mut r = Router::new();
        r.announce("only0", 0, 9.0);
        assert_eq!(
            r.assemble(2, &HashSet::new(), &ledger),
            Err(NetError::NoProvider(1))
        );
    }
}
