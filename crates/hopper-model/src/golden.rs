//! Loader for the `reference/golden/` fixture produced by
//! `reference/export_golden.py`.
//!
//! The fixture is the Phase 1 oracle: full weights plus per-stage golden I/O. The
//! format is deliberately boring — a `manifest.json` describing every tensor and
//! raw little-endian `.bin` blobs — so loading needs nothing beyond `std::fs`,
//! `serde_json`, and `f32::from_le_bytes`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::config::ModelConfig;
use crate::error::ModelError;
use crate::stage::Activation;
use crate::weights::{LayerWeights, Weights};

/// One tensor's manifest entry: name, shape, dtype, and path (relative to the dir).
#[derive(Debug, Deserialize)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub path: String,
    pub n_elements: usize,
}

/// Model metadata block from the manifest. Extra fields (seed, scale, …) are
/// ignored by serde.
#[derive(Debug, Deserialize)]
pub struct ModelMeta {
    pub name: String,
    pub model_hash: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub max_seq: usize,
    pub eps: f32,
}

/// One captured step (a prefill or a decode) for a stage.
#[derive(Debug, Deserialize)]
pub struct StepDesc {
    pub index: usize,
    pub kind: String,
    pub base_pos: usize,
    pub n_tokens: usize,
    pub input: TensorDesc,
    pub output: TensorDesc,
}

/// Cumulative session I/O for a stage: the concatenated input, the online
/// (cached) output, and the stateless recompute (the audit-invariant pair).
#[derive(Debug, Deserialize)]
pub struct Cumulative {
    pub input: TensorDesc,
    pub output_online: TensorDesc,
    pub output_stateless: TensorDesc,
    pub audit_rel_l2: f64,
}

/// A captured stage: its layer range, edge flags, per-step I/O, and cumulative I/O.
#[derive(Debug, Deserialize)]
pub struct StageDesc {
    pub label: String,
    pub stage_id: usize,
    pub layer_lo: usize,
    pub layer_hi: usize,
    pub n_layers: usize,
    pub is_first: bool,
    pub is_last: bool,
    pub input_kind: String,
    pub output_kind: String,
    pub n_params: usize,
    pub steps: Vec<StepDesc>,
    pub cumulative: Cumulative,
}

/// The whole manifest. Unneeded blocks (`pipeline`, `format`, …) are ignored.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub model: ModelMeta,
    pub weights: Vec<TensorDesc>,
    pub stages: Vec<StageDesc>,
}

/// A loaded golden fixture: its directory plus the parsed manifest.
pub struct Golden {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Golden {
    /// Load and parse `manifest.json` from `dir` (e.g. `reference/golden`).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        let dir = dir.as_ref().to_path_buf();
        let text = fs::read_to_string(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        Ok(Self { dir, manifest })
    }

    /// Reconstruct a [`ModelConfig`] from the manifest's model block.
    pub fn config(&self) -> ModelConfig {
        let m = &self.manifest.model;
        ModelConfig {
            name: m.name.clone(),
            vocab_size: m.vocab_size,
            d_model: m.d_model,
            n_layer: m.n_layer,
            n_head: m.n_head,
            d_ff: m.d_ff,
            max_seq: m.max_seq,
            eps: m.eps,
        }
    }

    fn read_f32(&self, desc: &TensorDesc) -> Result<Vec<f32>, ModelError> {
        if desc.dtype != "f32" {
            return Err(ModelError::BadDtype {
                name: desc.name.clone(),
                dtype: desc.dtype.clone(),
            });
        }
        let bytes = fs::read(self.dir.join(&desc.path))?;
        if bytes.len() != desc.n_elements * 4 {
            return Err(ModelError::TensorSize {
                name: desc.name.clone(),
                expected: desc.n_elements * 4,
                got: bytes.len(),
            });
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn read_i64(&self, desc: &TensorDesc) -> Result<Vec<i64>, ModelError> {
        if desc.dtype != "i64" {
            return Err(ModelError::BadDtype {
                name: desc.name.clone(),
                dtype: desc.dtype.clone(),
            });
        }
        let bytes = fs::read(self.dir.join(&desc.path))?;
        if bytes.len() != desc.n_elements * 8 {
            return Err(ModelError::TensorSize {
                name: desc.name.clone(),
                expected: desc.n_elements * 8,
                got: bytes.len(),
            });
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                i64::from_le_bytes(a)
            })
            .collect())
    }

    /// Load a 2-D f32 tensor (hidden state, logits, or a 2-D weight).
    pub fn array2(&self, desc: &TensorDesc) -> Result<Array2<f32>, ModelError> {
        let data = self.read_f32(desc)?;
        let (rows, cols) = (desc.shape[0], desc.shape[1]);
        Ok(Array2::from_shape_vec((rows, cols), data)?)
    }

    /// Load a 1-D f32 tensor (an RMSNorm gain).
    pub fn array1(&self, desc: &TensorDesc) -> Result<Array1<f32>, ModelError> {
        Ok(Array1::from_vec(self.read_f32(desc)?))
    }

    /// Load token ids (i64 in the fixture) as `usize`.
    pub fn ids(&self, desc: &TensorDesc) -> Result<Vec<usize>, ModelError> {
        Ok(self
            .read_i64(desc)?
            .into_iter()
            .map(|x| x as usize)
            .collect())
    }

    /// Load a tensor as a stage [`Activation`], routing by dtype (i64 → ids).
    pub fn activation(&self, desc: &TensorDesc) -> Result<Activation, ModelError> {
        if desc.dtype == "i64" {
            Ok(Activation::Ids(self.ids(desc)?))
        } else {
            Ok(Activation::Hidden(self.array2(desc)?))
        }
    }

    /// Reconstruct all model weights from the fixture, keyed by tensor name.
    pub fn weights(&self) -> Result<Weights, ModelError> {
        let by_name: HashMap<&str, &TensorDesc> = self
            .manifest
            .weights
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();
        let get = |name: &str| -> Result<&TensorDesc, ModelError> {
            by_name
                .get(name)
                .copied()
                .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
        };

        let mut layers = Vec::with_capacity(self.manifest.model.n_layer);
        for li in 0..self.manifest.model.n_layer {
            layers.push(LayerWeights {
                wqkv: self.array2(get(&format!("layer_{li:02}.wqkv"))?)?,
                wo: self.array2(get(&format!("layer_{li:02}.wo"))?)?,
                g1: self.array1(get(&format!("layer_{li:02}.g1"))?)?,
                g2: self.array1(get(&format!("layer_{li:02}.g2"))?)?,
                w1: self.array2(get(&format!("layer_{li:02}.w1"))?)?,
                w2: self.array2(get(&format!("layer_{li:02}.w2"))?)?,
            });
        }

        Ok(Weights {
            tok_emb: self.array2(get("tok_emb")?)?,
            pos_emb: self.array2(get("pos_emb")?)?,
            final_g: self.array1(get("final_g")?)?,
            layers,
        })
    }
}
