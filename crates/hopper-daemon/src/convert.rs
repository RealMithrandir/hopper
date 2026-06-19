//! Conversions between `hopper-model` tensors and `hopper-proto` wire types.
//! Lives in the daemon so `hopper-proto` stays a pure wire crate.

use anyhow::{anyhow, Result};
use half::f16;
use ndarray::Array2;

use hopper_model::Activation as ModelActivation;
use hopper_proto::{activation_stream, Activation as ProtoActivation, ActivationStream, TokenIds};

/// `[n_tokens, d_model]` tensor → proto activation, downcast to little-endian f16
/// for the wire (Phase 4: halve wire bytes, Invariant 1 unchanged).
pub fn array2_to_proto(a: &Array2<f32>) -> ProtoActivation {
    let mut data = Vec::with_capacity(a.len() * 2);
    for &x in a.iter() {
        data.extend_from_slice(&f16::from_f32(x).to_le_bytes());
    }
    ProtoActivation {
        n_tokens: a.nrows() as u32,
        d_model: a.ncols() as u32,
        data,
    }
}

/// Proto activation (little-endian f16) → `[n_tokens, d_model]` f32 tensor.
pub fn proto_to_array2(a: &ProtoActivation) -> Result<Array2<f32>> {
    let (rows, cols) = (a.n_tokens as usize, a.d_model as usize);
    if a.data.len() != rows * cols * 2 {
        return Err(anyhow!(
            "activation byte length {} != {rows}x{cols}x2",
            a.data.len()
        ));
    }
    let floats: Vec<f32> = a
        .data
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    Array2::from_shape_vec((rows, cols), floats).map_err(|e| anyhow!("malformed activation: {e}"))
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
