//! Conversions between `hopper-model` tensors and `hopper-proto` wire types.
//! Lives in the daemon so `hopper-proto` stays a pure wire crate.

use anyhow::{anyhow, Result};
use ndarray::Array2;

use hopper_model::Activation as ModelActivation;
use hopper_proto::{activation_stream, Activation as ProtoActivation, ActivationStream, TokenIds};

/// `[n_tokens, d_model]` tensor → proto activation (row-major f32).
pub fn array2_to_proto(a: &Array2<f32>) -> ProtoActivation {
    ProtoActivation {
        n_tokens: a.nrows() as u32,
        d_model: a.ncols() as u32,
        data: a.iter().copied().collect(),
    }
}

/// Proto activation → `[n_tokens, d_model]` tensor.
pub fn proto_to_array2(a: &ProtoActivation) -> Result<Array2<f32>> {
    Array2::from_shape_vec((a.n_tokens as usize, a.d_model as usize), a.data.clone())
        .map_err(|e| anyhow!("malformed activation: {e}"))
}

/// Build a stage request carrying the model's current activation.
pub fn stage_request(
    session: &str,
    stage_id: usize,
    seq_pos: u64,
    input: &ModelActivation,
) -> ActivationStream {
    let input = match input {
        ModelActivation::Ids(ids) => activation_stream::Input::Ids(TokenIds {
            ids: ids.iter().map(|&i| i as u32).collect(),
        }),
        ModelActivation::Hidden(h) => activation_stream::Input::Activation(array2_to_proto(h)),
    };
    ActivationStream {
        session: session.to_string(),
        stage_id: stage_id as u32,
        seq_pos,
        input: Some(input),
    }
}

/// Worker side: decode a request's input into a model activation.
pub fn request_input_to_model(req: &ActivationStream) -> Result<ModelActivation> {
    match req
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("request missing input"))?
    {
        activation_stream::Input::Ids(t) => Ok(ModelActivation::Ids(
            t.ids.iter().map(|&i| i as usize).collect(),
        )),
        activation_stream::Input::Activation(a) => Ok(ModelActivation::Hidden(proto_to_array2(a)?)),
    }
}
