//! Shared test helpers.

use std::path::PathBuf;

use ndarray::Array2;

/// Path to `reference/golden`, resolved relative to this crate (CWD-independent).
pub fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/golden")
}

/// Relative L2 error `||a - b|| / (||b|| + 1e-9)` — the audit metric from
/// `reference/verify.py` (`a` measured against baseline `b`).
pub fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let num = (a - b).mapv(|x| x * x).sum().sqrt();
    let den = b.mapv(|x| x * x).sum().sqrt() + 1e-9;
    num / den
}
