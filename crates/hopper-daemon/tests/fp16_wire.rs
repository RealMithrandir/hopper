//! fp16-on-the-wire golden parity: downcasting the inter-stage activation to f16
//! for transport (Phase 4) must keep the pipeline output within fp16 tolerance of
//! the f32 reference — and it must actually halve the wire bytes (Invariant 1's
//! payload, now in the model's working dtype).

use std::path::PathBuf;

use hopper_daemon::convert;
use hopper_model::golden::Golden;
use hopper_model::{shard, Activation, ModelConfig, Stage};
use ndarray::Array2;

fn golden() -> Golden {
    Golden::load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/golden"))
        .expect("load golden fixture")
}

fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let num = (a - b).mapv(|x| x * x).sum().sqrt();
    let den = b.mapv(|x| x * x).sum().sqrt() + 1e-9;
    num / den
}

/// Run a stateless prefill pass through all stages; if `wire`, round-trip each
/// inter-stage hidden through the proto f16 codec to simulate the wire.
fn run_pipeline(stages: &[Stage], ids: Vec<usize>, wire: bool) -> Array2<f32> {
    let n = stages.len();
    let mut x = Some(Activation::Ids(ids));
    let mut logits = None;
    for (s, stage) in stages.iter().enumerate() {
        let out = stage.forward(x.take().unwrap(), None, 0).expect("forward");
        if s + 1 == n {
            logits = Some(out);
        } else if wire {
            let hopped = convert::proto_to_array2(&convert::array2_to_proto(&out)).unwrap();
            x = Some(Activation::Hidden(hopped));
        } else {
            x = Some(Activation::Hidden(out));
        }
    }
    logits.unwrap()
}

#[test]
fn fp16_wire_handoff_stays_within_tolerance_and_halves_bytes() {
    let g = golden();
    let cfg = ModelConfig::default();
    assert_eq!(cfg.model_hash(), g.manifest.model.model_hash);
    let weights = g.weights().unwrap();
    let bounds = shard(&cfg, 4);
    let stages: Vec<Stage> = bounds
        .iter()
        .map(|&(lo, hi)| Stage::new(cfg.clone(), &weights, lo, hi))
        .collect();

    let ids: Vec<usize> = "explain the design".bytes().map(|b| b as usize).collect();
    let f32_logits = run_pipeline(&stages, ids.clone(), false);
    let f16_logits = run_pipeline(&stages, ids, true);

    let rel = rel_l2(&f16_logits, &f32_logits);
    println!("fp16 wire vs f32 logits rel-L2 = {rel:e}");
    // fp16 has ~1e-3 relative precision; a few stage handoffs stay well under 1e-2.
    assert!(rel < 1e-2, "fp16 handoff drift {rel} exceeds 1e-2");

    // The wire payload is f16: 2 bytes/elem, half of f32's 4.
    let hidden = stages[0]
        .forward(Activation::Ids(vec![104, 105]), None, 0)
        .unwrap();
    let proto = convert::array2_to_proto(&hidden);
    assert_eq!(
        proto.data.len(),
        hidden.len() * 2,
        "wire payload must be f16"
    );

    // A single hidden round-trip is at fp16 precision.
    let back = convert::proto_to_array2(&proto).unwrap();
    let round_trip = rel_l2(&back, &hidden);
    println!("single f16 round-trip rel-L2 = {round_trip:e}");
    assert!(
        round_trip < 2e-3,
        "f16 round-trip {round_trip} worse than expected"
    );
}
