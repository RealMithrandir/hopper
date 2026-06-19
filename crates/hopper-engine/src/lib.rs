//! `hopper-engine` — orchestration of one inference across the pipeline.
//!
//! Ports `reference/node.py` (the [`Node`] peer: hosts stages, pins KV caches
//! locally, logs cumulative I/O for audit, exposes a tamper hook) and
//! `reference/engine.py` (the [`Engine`]: prefill then decode, metering bytes /
//! latency / FLOPs per hop, FLOP-crediting the ledger, spot-checking via the
//! verifier, and re-assembling the pipeline around a node slashed mid-stream).
//!
//! It ties together every Phase-2 invariant: the KV cache stays resident and only
//! the activation hops (1), pipelining is over layers on one token (2), audits
//! gate on a tolerance band (4), work is FLOP-metered (5), and routing is
//! latency-first with live reroute (7).

pub mod engine;
pub mod error;
pub mod node;

pub use engine::{Engine, GenStats};
pub use error::EngineError;
pub use node::{Node, NodeMap, Tamper};
