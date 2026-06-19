//! Golden parity (Invariant 6): the Rust stage must reproduce the reference
//! oracle's exported stage outputs within 1e-4 rel-L2 — for every captured stage,
//! every prefill/decode step, and the cumulative stateless recompute.

mod common;

use common::{golden_dir, rel_l2};
use hopper_model::golden::Golden;
use hopper_model::{KVCache, Stage};

#[test]
fn stage_outputs_match_golden_within_1e_4() {
    let golden = Golden::load(golden_dir()).expect("load golden fixture");
    let cfg = golden.config();

    // Invariant 6 sanity: the name hashes to the same model_hash the export used.
    assert_eq!(cfg.model_hash(), golden.manifest.model.model_hash);

    let weights = golden.weights().expect("load weights");
    let mut worst = 0.0_f32;

    for sd in &golden.manifest.stages {
        let stage = Stage::new(cfg.clone(), &weights, sd.layer_lo, sd.layer_hi);
        assert_eq!(stage.is_first(), sd.is_first);
        assert_eq!(stage.is_last(), sd.is_last);
        assert_eq!(stage.n_params(), sd.n_params);

        // (a) online parity: feed each step's input through the cached stage.
        let mut caches: Vec<KVCache> = (0..sd.n_layers).map(|_| KVCache::new()).collect();
        for step in &sd.steps {
            assert_eq!(
                caches[0].length(),
                step.base_pos,
                "cache length must track base_pos"
            );
            let input = golden.activation(&step.input).expect("load step input");
            let out = stage
                .forward(input, Some(caches.as_mut_slice()), step.base_pos)
                .expect("forward online");
            let gold = golden.array2(&step.output).expect("load step output");
            let r = rel_l2(&out, &gold);
            worst = worst.max(r);
            assert!(
                r < 1e-4,
                "stage `{}` {} step {}: rel-L2 {r:e} >= 1e-4",
                sd.label,
                step.kind,
                step.index
            );
        }

        // (b) stateless recompute over the cumulative input matches the golden.
        let cum_in = golden
            .activation(&sd.cumulative.input)
            .expect("load cumulative input");
        let out = stage.forward(cum_in, None, 0).expect("forward stateless");
        let gold = golden
            .array2(&sd.cumulative.output_stateless)
            .expect("load cumulative stateless");
        let r = rel_l2(&out, &gold);
        worst = worst.max(r);
        assert!(
            r < 1e-4,
            "stage `{}` stateless: rel-L2 {r:e} >= 1e-4",
            sd.label
        );
    }

    println!("worst golden parity rel-L2 = {worst:e}");
    assert!(worst < 1e-4);
}
