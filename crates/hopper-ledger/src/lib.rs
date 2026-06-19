//! `hopper-ledger` — the Tokenless Access Pact (mirrors `reference/ledger.py`).
//!
//! Incentive alignment without a coin or a blockchain (Invariant 5). A node's
//! standing is a **time-decayed FLOP contribution ratio**; access tiers follow
//! from it; identity minting costs proof-of-work; enforcement is stake-slash +
//! identity-ban. Optimistic-unchoke is a *floor* that rescues fresh staked nodes
//! from the throttle — it never demotes a node that is already contributing.

pub mod clock;
pub mod pow;

use std::collections::{BTreeMap, HashMap};

use clock::{Clock, SystemClock};

pub use clock::{ManualClock, SystemClock as WallClock};
pub use pow::{proof_of_work, verify_pow};

/// Decay rate: `ln(2) / 3600`, i.e. a ~1 hour half-life.
pub const LAMBDA: f64 = std::f64::consts::LN_2 / 3600.0;
/// Optimistic-unchoke free FLOP allowance for fresh staked identities.
pub const GRACE_FLOPS: f64 = 5e9;
/// How long a fresh identity stays eligible for the unchoke grant (30 min).
pub const GRACE_WINDOW_S: f64 = 1800.0;

/// Access tier for a node, ordered best → worst. The string forms match the
/// reference snapshot exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Priority,
    Delayed,
    Throttled,
    OptimisticUnchoke,
    Blocked,
}

impl Tier {
    /// Snapshot string (matches `reference/ledger.py`).
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Priority => "priority",
            Tier::Delayed => "delayed",
            Tier::Throttled => "throttled",
            Tier::OptimisticUnchoke => "optimistic_unchoke",
            Tier::Blocked => "blocked",
        }
    }
}

/// The result of an access-policy decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Access {
    pub tier: Tier,
    pub queue_delay_ms: f64,
}

/// One node's account: its stake bond, decayed work history, and ban flag.
#[derive(Debug, Clone)]
pub struct Account {
    pub node_id: String,
    pub minted_at: f64,
    pub pow_nonce: u64,
    pub stake: f64,
    /// `(timestamp, flops)` of work this node *provided*.
    pub seeded: Vec<(f64, f64)>,
    /// `(timestamp, flops)` of work this node *consumed*.
    pub leeched: Vec<(f64, f64)>,
    pub banned: bool,
}

/// A compact per-node view for reporting (mirrors `ledger.snapshot`).
#[derive(Debug, Clone, Copy)]
pub struct NodeSnapshot {
    pub ratio: f64,
    pub tier: Tier,
    pub stake: f64,
    pub banned: bool,
}

/// The contribution ledger. No global consensus: this is local, advisory state.
pub struct Ledger {
    clock: Box<dyn Clock>,
    accounts: HashMap<String, Account>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new(Box::new(SystemClock))
    }
}

impl Ledger {
    /// Build a ledger with an explicit clock (use [`ManualClock`] in tests).
    pub fn new(clock: Box<dyn Clock>) -> Self {
        Self {
            clock,
            accounts: HashMap::new(),
        }
    }

    /// Mint and register an identity, paying `difficulty` bits of proof-of-work.
    /// Returns the PoW nonce. Re-registering overwrites (mirrors the reference).
    pub fn register(&mut self, node_id: &str, difficulty: u32) -> u64 {
        let nonce = proof_of_work(node_id, difficulty);
        let acct = Account {
            node_id: node_id.to_string(),
            minted_at: self.clock.now(),
            pow_nonce: nonce,
            stake: 1.0,
            seeded: Vec::new(),
            leeched: Vec::new(),
            banned: false,
        };
        self.accounts.insert(node_id.to_string(), acct);
        nonce
    }

    /// Read-only access to a registered account.
    pub fn account(&self, node_id: &str) -> Option<&Account> {
        self.accounts.get(node_id)
    }

    /// Credit `flops` of work to `provider` and debit it to `consumer`.
    pub fn record(&mut self, provider: &str, consumer: &str, flops: f64) {
        let t = self.clock.now();
        if let Some(a) = self.accounts.get_mut(provider) {
            a.seeded.push((t, flops));
        }
        if let Some(a) = self.accounts.get_mut(consumer) {
            a.leeched.push((t, flops));
        }
    }

    fn decayed(&self, events: &[(f64, f64)]) -> f64 {
        let now = self.clock.now();
        events
            .iter()
            .map(|&(t, f)| f * (-LAMBDA * (now - t)).exp())
            .sum()
    }

    /// Decayed sum of FLOPs this node has provided.
    pub fn decayed_seeded(&self, node_id: &str) -> Option<f64> {
        self.accounts.get(node_id).map(|a| self.decayed(&a.seeded))
    }

    /// Decayed sum of FLOPs this node has consumed.
    pub fn decayed_leeched(&self, node_id: &str) -> Option<f64> {
        self.accounts.get(node_id).map(|a| self.decayed(&a.leeched))
    }

    /// Decayed contribution ratio `Σseeded / Σleeched`; `inf` when never leeched.
    pub fn contribution_ratio(&self, node_id: &str) -> Option<f64> {
        let a = self.accounts.get(node_id)?;
        let seeded = self.decayed(&a.seeded);
        let leeched = self.decayed(&a.leeched);
        Some(if leeched <= 0.0 {
            f64::INFINITY
        } else {
            seeded / leeched
        })
    }

    /// Access tier + queue delay. The ratio sets the tier; optimistic-unchoke is
    /// a floor that only ever *upgrades* a throttled fresh node, never demotes.
    pub fn access(&self, node_id: &str) -> Access {
        let account = match self.accounts.get(node_id) {
            Some(a) if !a.banned => a,
            _ => {
                return Access {
                    tier: Tier::Blocked,
                    queue_delay_ms: f64::INFINITY,
                }
            }
        };

        let ratio = self.contribution_ratio(node_id).unwrap_or(f64::INFINITY);
        let (mut tier, mut delay) = if ratio >= 1.0 {
            (Tier::Priority, 0.0)
        } else if ratio >= 0.2 {
            (Tier::Delayed, 250.0 * (1.0 - ratio))
        } else {
            (Tier::Throttled, 2000.0)
        };

        if tier == Tier::Throttled {
            let fresh = (self.clock.now() - account.minted_at) < GRACE_WINDOW_S;
            if fresh && self.decayed(&account.leeched) < GRACE_FLOPS {
                tier = Tier::OptimisticUnchoke;
                delay = 0.0;
            }
        }

        Access {
            tier,
            queue_delay_ms: delay,
        }
    }

    /// Slash a node: zero its stake, ban it, and erase its seeded credit.
    pub fn slash(&mut self, node_id: &str) {
        if let Some(a) = self.accounts.get_mut(node_id) {
            a.stake = 0.0;
            a.banned = true;
            a.seeded.clear();
        }
    }

    /// Per-node snapshot for reporting, keyed (sorted) by node id.
    pub fn snapshot(&self) -> BTreeMap<String, NodeSnapshot> {
        self.accounts
            .keys()
            .map(|id| {
                let a = &self.accounts[id];
                (
                    id.clone(),
                    NodeSnapshot {
                        ratio: self.contribution_ratio(id).unwrap_or(f64::INFINITY),
                        tier: self.access(id).tier,
                        stake: a.stake,
                        banned: a.banned,
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_ledger(t0: f64) -> (Ledger, ManualClock) {
        let clk = ManualClock::new(t0);
        (Ledger::new(Box::new(clk.clone())), clk)
    }

    #[test]
    fn seeded_credit_decays_by_half_each_hour() {
        let (mut l, clk) = manual_ledger(0.0);
        l.register("a", 8);
        l.record("a", "consumer-unregistered", 1e9);
        let at0 = l.decayed_seeded("a").unwrap();
        assert!((at0 - 1e9).abs() < 1.0);
        clk.advance(3600.0);
        let at1 = l.decayed_seeded("a").unwrap();
        assert!((at1 - 0.5e9).abs() < 1e3, "one half-life halves it: {at1}");
        clk.advance(3600.0);
        let at2 = l.decayed_seeded("a").unwrap();
        assert!(at2 < at1, "decay is monotonic");
        assert!((at2 - 0.25e9).abs() < 1e3);
    }

    #[test]
    fn ratio_reflects_asymmetric_decay() {
        let (mut l, clk) = manual_ledger(0.0);
        l.register("a", 8);
        l.record("a", "x", 2e9); // seeded at t=0
        clk.advance(3600.0);
        l.record("y", "a", 1e9); // leeched at t=3600
                                 // seeded decayed to 1e9, leeched 1e9 -> ratio ~1.0
        let r = l.contribution_ratio("a").unwrap();
        assert!((r - 1.0).abs() < 1e-6, "ratio {r}");
    }

    #[test]
    fn four_tiers() {
        // priority: seeded, never leeched -> inf ratio
        let (mut l, _c) = manual_ledger(0.0);
        l.register("prio", 8);
        l.record("prio", "x", 1e9);
        assert_eq!(l.access("prio").tier, Tier::Priority);

        // delayed: ratio in [0.2, 1.0)
        l.register("del", 8);
        l.record("del", "x", 5e8);
        l.record("y", "del", 1e9); // ratio 0.5
        let acc = l.access("del");
        assert_eq!(acc.tier, Tier::Delayed);
        assert!((acc.queue_delay_ms - 125.0).abs() < 1e-6);

        // blocked: unknown id
        assert_eq!(l.access("ghost").tier, Tier::Blocked);
    }

    #[test]
    fn throttled_when_stale_and_low_ratio() {
        let (mut l, clk) = manual_ledger(0.0);
        l.register("old", 8);
        l.record("old", "x", 1e8);
        l.record("y", "old", 1e9); // ratio 0.1 < 0.2
        clk.advance(GRACE_WINDOW_S + 1.0); // no longer fresh
        let acc = l.access("old");
        assert_eq!(acc.tier, Tier::Throttled);
        assert!((acc.queue_delay_ms - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn optimistic_unchoke_rescues_fresh_low_ratio_node() {
        let (mut l, _c) = manual_ledger(0.0);
        l.register("newbie", 8);
        l.record("y", "newbie", 1e8); // pure consumer, ratio 0 < 0.2, leeched < grace
        assert_eq!(l.access("newbie").tier, Tier::OptimisticUnchoke);
    }

    #[test]
    fn unchoke_is_a_floor_never_a_ceiling() {
        // A *fresh* node with a high ratio must stay priority — the unchoke floor
        // must never demote a contributor (Invariant 5).
        let (mut l, _c) = manual_ledger(0.0);
        l.register("fresh_contributor", 8);
        l.record("fresh_contributor", "x", 1e9); // big seed, no leech -> inf
        assert_eq!(l.access("fresh_contributor").tier, Tier::Priority);
    }

    #[test]
    fn slash_is_idempotent() {
        let (mut l, _c) = manual_ledger(0.0);
        l.register("bad", 8);
        l.record("bad", "x", 1e9);
        l.slash("bad");
        assert!(l.account("bad").unwrap().banned);
        assert_eq!(l.account("bad").unwrap().stake, 0.0);
        assert!(l.account("bad").unwrap().seeded.is_empty());
        assert_eq!(l.access("bad").tier, Tier::Blocked);
        l.slash("bad"); // again
        assert!(l.account("bad").unwrap().banned);
        assert_eq!(l.access("bad").tier, Tier::Blocked);
    }
}
