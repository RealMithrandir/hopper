//! Model hyper-parameters and the deterministic `model_hash`.
//!
//! Mirrors `reference/model.py`'s `ModelConfig`. The `model_hash` is what pins
//! byte-identical weights across the swarm (Invariant 6) — every node that agrees
//! on the name agrees on the weights, which is what makes audits comparable.

use sha2::{Digest, Sha256};

/// Decoder-only transformer configuration. Defaults match `hopper-tiny`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub name: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub max_seq: usize,
    pub eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "hopper-tiny".to_string(),
            vocab_size: 256,
            d_model: 128,
            n_layer: 8,
            n_head: 4,
            d_ff: 512,
            max_seq: 512,
            eps: 1e-5,
        }
    }
}

impl ModelConfig {
    /// Per-head width. `d_model` is assumed divisible by `n_head`.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// `sha256(name)[..16]` — the deterministic weight key (Invariant 6).
    pub fn model_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            // Infallible: writing to a String never errors.
            let _ = write!(hex, "{byte:02x}");
        }
        hex.truncate(16);
        hex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_hash_matches_reference() {
        // reference/model.py: hashlib.sha256("hopper-tiny").hexdigest()[:16]
        assert_eq!(ModelConfig::default().model_hash(), "80fe0c38c9d5e237");
    }

    #[test]
    fn head_dim_is_d_model_over_n_head() {
        assert_eq!(ModelConfig::default().head_dim(), 32);
    }
}
