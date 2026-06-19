//! Per-(session, layer) KV cache. Mirrors `reference/model.py`'s `KVCache`.
//!
//! INVARIANT 1: this never crosses the network. It lives on the node that owns
//! the layer, grows with context, and is the *big* thing we deliberately pin in
//! place while only the small activation hops between stages. There is, by
//! design, no serialization for this type.

use ndarray::{concatenate, Array3, Axis};

use crate::error::ModelError;

/// Cached keys and values for one layer of one session.
///
/// `k`/`v` have shape `[n_head, seq, head_dim]`; `seq` grows as tokens are
/// appended during online decode.
#[derive(Debug, Default, Clone)]
pub struct KVCache {
    k: Option<Array3<f32>>,
    v: Option<Array3<f32>>,
}

impl KVCache {
    /// An empty cache (no cached positions yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached positions (the `seq` axis), `0` when empty.
    pub fn length(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.shape()[1])
    }

    /// Append this step's keys/values along the sequence axis.
    pub fn append(&mut self, k: &Array3<f32>, v: &Array3<f32>) -> Result<(), ModelError> {
        self.k = Some(match self.k.take() {
            None => k.clone(),
            Some(prev) => concatenate(Axis(1), &[prev.view(), k.view()])?,
        });
        self.v = Some(match self.v.take() {
            None => v.clone(),
            Some(prev) => concatenate(Axis(1), &[prev.view(), v.view()])?,
        });
        Ok(())
    }

    /// Cached keys `[n_head, seq, head_dim]`, if any have been appended.
    pub fn k(&self) -> Option<&Array3<f32>> {
        self.k.as_ref()
    }

    /// Cached values `[n_head, seq, head_dim]`, if any have been appended.
    pub fn v(&self) -> Option<&Array3<f32>> {
        self.v.as_ref()
    }
}
