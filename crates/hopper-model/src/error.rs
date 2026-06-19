//! Error type for `hopper-model`. Libraries return `Result<_, ModelError>`
//! (CLAUDE.md error convention); only the daemon uses `anyhow`.

/// Anything that can go wrong loading or running a model stage.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("io error reading golden fixture: {0}")]
    Io(#[from] std::io::Error),

    #[error("manifest json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("first stage requires token ids as input")]
    ExpectedIds,

    #[error("non-first stage requires a hidden activation as input")]
    ExpectedHidden,

    #[error("golden tensor '{0}' missing from manifest")]
    MissingTensor(String),

    #[error("tensor '{name}': unexpected dtype '{dtype}'")]
    BadDtype { name: String, dtype: String },

    #[error("tensor '{name}': expected {expected} bytes, got {got}")]
    TensorSize {
        name: String,
        expected: usize,
        got: usize,
    },
}
