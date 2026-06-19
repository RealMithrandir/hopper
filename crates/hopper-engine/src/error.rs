//! Orchestration errors. Mirrors the failure modes the engine can hit while
//! driving one inference.

/// Errors raised while hosting a stage or orchestrating a generation.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("node does not host stage {0}")]
    NoStage(usize),

    #[error("pipeline references unknown node `{0}`")]
    UnknownNode(String),

    #[error(transparent)]
    Model(#[from] hopper_model::ModelError),

    #[error(transparent)]
    Net(#[from] hopper_net::NetError),

    #[error(transparent)]
    Verify(#[from] hopper_verify::VerifyError),
}
