//! A contiguous block of transformer layers owned by one node.
//!
//! Mirrors `reference/model.py`'s `Stage`. A stage consumes and produces a small
//! `[n_tokens, d_model]` activation — the only thing that ever crosses the wire
//! (Invariant 1). The first stage also owns the embeddings (input = token ids);
//! the last owns the final norm + tied LM head (output = logits).
//!
//! [`Stage::forward`] takes `caches: Option<&mut [KVCache]>`:
//! * `Some` → online decode, mutating the per-layer caches;
//! * `None` → a stateless audit recompute over the whole input sequence.
//!
//! The audit invariant (Invariant 3) is that these two agree within fp tolerance,
//! which is exactly what [`crate::golden`] and the property tests assert.

use ndarray::{s, Array1, Array2, Array3, Axis};

use crate::cache::KVCache;
use crate::config::ModelConfig;
use crate::error::ModelError;
use crate::weights::{LayerWeights, Weights};

/// Input to a stage: token ids for the first stage, a hidden activation otherwise.
#[derive(Debug, Clone)]
pub enum Activation {
    /// Token ids `[n_tokens]` — only valid for the first stage.
    Ids(Vec<usize>),
    /// Hidden state `[n_tokens, d_model]` — for every non-first stage.
    Hidden(Array2<f32>),
}

impl Activation {
    /// Number of tokens this activation carries.
    pub fn n_tokens(&self) -> usize {
        match self {
            Activation::Ids(ids) => ids.len(),
            Activation::Hidden(h) => h.nrows(),
        }
    }
}

/// RMSNorm: `x / sqrt(mean(x^2) + eps) * gain`, row-wise. Matches `model._rmsnorm`.
fn rmsnorm(x: &Array2<f32>, gain: &Array1<f32>, eps: f32) -> Array2<f32> {
    let d = x.ncols() as f32;
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d;
        let denom = (ms + eps).sqrt();
        for (val, g) in row.iter_mut().zip(gain.iter()) {
            *val = *val / denom * *g;
        }
    }
    out
}

/// tanh-GELU, matching `model._gelu`. The sqrt(2/pi) literal is copied verbatim
/// from the reference oracle, so clippy's excessive-precision lint is silenced
/// here on purpose — the constant must stay bit-for-bit identical to the spec.
#[allow(clippy::excessive_precision)]
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + (0.7978845608_f32 * (v + 0.044715 * v * v * v)).tanh()))
}

/// Reshape `[n, d]` into per-head `[n_head, n, head_dim]`, matching numpy's
/// `t.reshape(n, h, hd).transpose(1, 0, 2)`.
fn to_heads(t: &Array2<f32>, n: usize, h: usize, hd: usize) -> Result<Array3<f32>, ModelError> {
    let reshaped = t.to_shape((n, h, hd))?.to_owned();
    Ok(reshaped
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .to_owned())
}

/// Inverse of [`to_heads`]: `[n_head, n, head_dim]` back to `[n, d]`.
fn from_heads(ctx: &Array3<f32>, n: usize, d: usize) -> Result<Array2<f32>, ModelError> {
    let merged = ctx.view().permuted_axes([1, 0, 2]);
    let contiguous = merged.as_standard_layout();
    Ok(contiguous.to_shape((n, d))?.to_owned())
}

/// A contiguous layer range `[lo, hi)` of the model, hosted by one node.
pub struct Stage<'w> {
    cfg: ModelConfig,
    w: &'w Weights,
    lo: usize,
    hi: usize,
    is_first: bool,
    is_last: bool,
}

impl<'w> Stage<'w> {
    /// Build a stage over layers `[lo, hi)`. `lo == 0` owns the embeddings;
    /// `hi == n_layer` owns the final norm + LM head.
    pub fn new(cfg: ModelConfig, w: &'w Weights, lo: usize, hi: usize) -> Self {
        let is_first = lo == 0;
        let is_last = hi == cfg.n_layer;
        Self {
            cfg,
            w,
            lo,
            hi,
            is_first,
            is_last,
        }
    }

    /// True if this stage owns the embeddings (input = token ids).
    pub fn is_first(&self) -> bool {
        self.is_first
    }

    /// True if this stage owns the final norm + LM head (output = logits).
    pub fn is_last(&self) -> bool {
        self.is_last
    }

    /// The `[lo, hi)` layer range this stage hosts.
    pub fn layer_range(&self) -> (usize, usize) {
        (self.lo, self.hi)
    }

    /// Scalar parameter count for this stage (layers + any owned embeddings/head).
    pub fn n_params(&self) -> usize {
        let mut p: usize = self.w.layers[self.lo..self.hi]
            .iter()
            .map(LayerWeights::n_params)
            .sum();
        if self.is_first {
            p += self.w.tok_emb.len() + self.w.pos_emb.len();
        }
        if self.is_last {
            p += self.w.final_g.len();
        }
        p
    }

    /// Standard `2 * params * tokens` forward-pass FLOP estimate.
    pub fn flops(&self, n_tokens: usize) -> usize {
        2 * self.n_params() * n_tokens
    }

    /// One pre-norm transformer layer. Mirrors `model.Stage._layer`.
    fn layer(
        &self,
        lw: &LayerWeights,
        x: &Array2<f32>,
        cache: Option<&mut KVCache>,
        base_pos: usize,
    ) -> Result<Array2<f32>, ModelError> {
        let cfg = &self.cfg;
        let (h, hd, n, d) = (cfg.n_head, cfg.head_dim(), x.nrows(), cfg.d_model);

        // --- attention ---
        let a = rmsnorm(x, &lw.g1, cfg.eps);
        let qkv = a.dot(&lw.wqkv); // [n, 3d]
        let q = to_heads(&qkv.slice(s![.., 0..d]).to_owned(), n, h, hd)?;
        let k = to_heads(&qkv.slice(s![.., d..2 * d]).to_owned(), n, h, hd)?;
        let v = to_heads(&qkv.slice(s![.., 2 * d..3 * d]).to_owned(), n, h, hd)?;

        // Online (cache present) appends and reads the full history; the stateless
        // audit path (cache absent) sees only this call's keys/values.
        let (k_all, v_all, past) = match cache {
            Some(c) => {
                let past = c.length();
                c.append(&k, &v)?;
                // Just appended, so both are present; this is a true invariant.
                let k_all = c.k().expect("kv present after append").clone();
                let v_all = c.v().expect("kv present after append").clone();
                (k_all, v_all, past)
            }
            None => (k, v, 0),
        };

        let seq = k_all.shape()[1];
        // Absolute key positions start here; `kpos_base` is 0 for both the online
        // and full-sequence stateless paths, but we keep the general form.
        let kpos_base = seq as i64 - (past + n) as i64;
        let scale = (hd as f32).sqrt();

        let mut ctx = Array3::<f32>::zeros((h, n, hd));
        for head in 0..h {
            let qh = q.index_axis(Axis(0), head); // [n, hd]
            let kh = k_all.index_axis(Axis(0), head); // [seq, hd]
            let vh = v_all.index_axis(Axis(0), head); // [seq, hd]

            let mut scores = qh.dot(&kh.t()); // [n, seq]
            scores.mapv_inplace(|s| s / scale);

            for i in 0..n {
                let qpos = base_pos as i64 + i as i64;
                let mut row = scores.row_mut(i);
                // causal mask by absolute position
                for (j, val) in row.iter_mut().enumerate() {
                    if kpos_base + j as i64 > qpos {
                        *val = -1e30;
                    }
                }
                // softmax over keys (subtract row max for stability)
                let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    let e = (*val - maxv).exp();
                    *val = e;
                    sum += e;
                }
                row.mapv_inplace(|e| e / sum);
            }

            let ctx_h = scores.dot(&vh); // [n, hd]
            ctx.index_axis_mut(Axis(0), head).assign(&ctx_h);
        }

        let ctx = from_heads(&ctx, n, d)?;
        let mut x = x + &ctx.dot(&lw.wo); // residual

        // --- mlp ---
        let m = rmsnorm(&x, &lw.g2, cfg.eps);
        x += &gelu(&m.dot(&lw.w1)).dot(&lw.w2);
        Ok(x)
    }

    /// Run the stage. With `caches = Some(..)` this is an online step that mutates
    /// the per-layer caches; with `None` it is a stateless recompute over the full
    /// input sequence (`base_pos` is the absolute position of the first row).
    pub fn forward(
        &self,
        input: Activation,
        mut caches: Option<&mut [KVCache]>,
        base_pos: usize,
    ) -> Result<Array2<f32>, ModelError> {
        let mut x: Array2<f32> = match input {
            Activation::Ids(ids) => {
                if !self.is_first {
                    return Err(ModelError::ExpectedHidden);
                }
                // tok_emb[ids] + pos_emb[base_pos .. base_pos + n]
                let mut emb = self.w.tok_emb.select(Axis(0), &ids);
                for (i, mut row) in emb.rows_mut().into_iter().enumerate() {
                    row += &self.w.pos_emb.row(base_pos + i);
                }
                emb
            }
            Activation::Hidden(h) => {
                if self.is_first {
                    return Err(ModelError::ExpectedIds);
                }
                h
            }
        };

        for (idx, lw) in self.w.layers[self.lo..self.hi].iter().enumerate() {
            let cache = caches.as_deref_mut().map(|cs| &mut cs[idx]);
            x = self.layer(lw, &x, cache, base_pos)?;
        }

        if self.is_last {
            x = rmsnorm(&x, &self.w.final_g, self.cfg.eps);
            x = x.dot(&self.w.tok_emb.t()); // tied LM head -> logits [n, vocab]
        }
        Ok(x)
    }
}

/// Python's round-half-to-even, so [`shard`] matches `model.shard` exactly.
fn round_half_even(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    // Round up on a clear majority, or on an exact half only when floor is odd.
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    let rounded = if round_up { floor + 1.0 } else { floor };
    rounded as usize
}

/// Split `n_layer` layers into `n_stages` roughly-equal contiguous blocks.
/// Mirrors `model.shard`.
pub fn shard(cfg: &ModelConfig, n_stages: usize) -> Vec<(usize, usize)> {
    let per = cfg.n_layer as f64 / n_stages as f64;
    (0..n_stages)
        .map(|s| {
            let lo = round_half_even(s as f64 * per);
            let hi = round_half_even((s as f64 + 1.0) * per);
            (lo, hi)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_matches_reference_bounds() {
        let cfg = ModelConfig::default();
        assert_eq!(shard(&cfg, 4), vec![(0, 2), (2, 4), (4, 6), (6, 8)]);
    }

    #[test]
    fn activation_token_count() {
        assert_eq!(Activation::Ids(vec![1, 2, 3]).n_tokens(), 3);
    }
}
