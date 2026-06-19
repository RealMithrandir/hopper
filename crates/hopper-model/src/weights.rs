//! Materialized model weights. Mirrors `reference/model.py`'s weight dict +
//! `LayerWeights`. Phase 1 loads these from the golden fixture (see
//! [`crate::golden`]); the values are keyed by `model_hash` so every node holds
//! byte-identical weights (Invariant 6).

use ndarray::{Array1, Array2};

/// Weights for one transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Fused query/key/value projection `[d_model, 3*d_model]`.
    pub wqkv: Array2<f32>,
    /// Attention output projection `[d_model, d_model]`.
    pub wo: Array2<f32>,
    /// RMSNorm gain before attention `[d_model]`.
    pub g1: Array1<f32>,
    /// RMSNorm gain before the MLP `[d_model]`.
    pub g2: Array1<f32>,
    /// MLP up projection `[d_model, d_ff]`.
    pub w1: Array2<f32>,
    /// MLP down projection `[d_ff, d_model]`.
    pub w2: Array2<f32>,
}

impl LayerWeights {
    /// Total scalar parameter count (used for FLOP metering).
    pub fn n_params(&self) -> usize {
        self.wqkv.len()
            + self.wo.len()
            + self.g1.len()
            + self.g2.len()
            + self.w1.len()
            + self.w2.len()
    }
}

/// All weights for a model: shared embeddings + per-layer blocks.
#[derive(Debug, Clone)]
pub struct Weights {
    /// Token embedding `[vocab, d_model]`; also the tied LM head on the last stage.
    pub tok_emb: Array2<f32>,
    /// Learned absolute position embedding `[max_seq, d_model]`.
    pub pos_emb: Array2<f32>,
    /// Final RMSNorm gain `[d_model]` (last stage only).
    pub final_g: Array1<f32>,
    /// Per-layer weights, indexed by absolute layer id.
    pub layers: Vec<LayerWeights>,
}
