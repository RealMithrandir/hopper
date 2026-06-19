//! `hopper-model` — L1 layer-sharded transformer with a node-resident KV cache.
//!
//! A faithful Rust port of `reference/model.py`. The model is split *by layer*
//! into [`Stage`]s that pass a small `[n_tokens, d_model]` activation between them
//! (Invariant 1); each stage keeps its KV cache locally ([`KVCache`]). The same
//! [`Stage::forward`] runs both online decode (with caches) and the stateless
//! audit recompute (without), and those agree within fp tolerance (Invariant 3).
//!
//! Phase 1 loads weights and golden I/O from the [`golden`] fixture and proves
//! numeric parity (<1e-4 rel-L2) plus the cache↔stateless invariant — see the
//! `tests/` directory.

pub mod cache;
pub mod config;
pub mod error;
pub mod golden;
pub mod stage;
pub mod tokenizer;
pub mod weights;

pub use cache::KVCache;
pub use config::ModelConfig;
pub use error::ModelError;
pub use stage::{shard, Activation, Stage};
pub use tokenizer::{decode, encode};
pub use weights::{LayerWeights, Weights};
