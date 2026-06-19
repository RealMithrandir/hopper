//! `hopper-net` — L2 transport + router (simulated; libp2p arrives in Phase 3).
//!
//! Ports `reference/transport.py` (byte/latency accounting + the
//! KV-bytes-avoided counterfactual) and `reference/router.py` (latency-aware
//! pipeline assembly with reroute). Routing optimizes latency, not bandwidth
//! (Invariant 7); the only inter-stage payload is the activation (Invariant 1).

pub mod router;
pub mod transport;

pub use router::{NetError, Provider, Router};
pub use transport::{LinkProfile, NetworkMonitor, NetworkReport};
