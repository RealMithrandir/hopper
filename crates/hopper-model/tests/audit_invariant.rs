//! Invariant 3 (the audit invariant): for every stage, an online run *with* a KV
//! cache produces the same output as the same stage re-run *statelessly* over the
//! full input sequence, within fp tolerance. This is what makes re-execution
//! audits (verify.py) meaningful, so it gets a property test over random sessions.
//!
//! Weights come from the golden fixture (no RNG reproduction needed); the
//! *sessions* are randomized with an explicitly seeded `ChaCha8Rng` so failures
//! are reproducible (CLAUDE.md determinism convention).

mod common;

use common::{golden_dir, rel_l2};
use hopper_model::golden::Golden;
use hopper_model::{shard, Activation, KVCache, ModelConfig, Stage, Weights};
use ndarray::{concatenate, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A random stage input: token ids for the first stage, else a hidden state.
fn random_input(rng: &mut ChaCha8Rng, cfg: &ModelConfig, is_first: bool, n: usize) -> Activation {
    if is_first {
        Activation::Ids(
            (0..n)
                .map(|_| rng.random_range(0..cfg.vocab_size))
                .collect(),
        )
    } else {
        Activation::Hidden(Array2::from_shape_fn((n, cfg.d_model), |_| {
            rng.random::<f32>() * 2.0 - 1.0
        }))
    }
}

/// Concatenate a list of step inputs into the cumulative session input.
fn cumulative_input(steps: &[Activation], is_first: bool) -> Activation {
    if is_first {
        let mut ids = Vec::new();
        for s in steps {
            if let Activation::Ids(v) = s {
                ids.extend_from_slice(v);
            }
        }
        Activation::Ids(ids)
    } else {
        let mats: Vec<Array2<f32>> = steps
            .iter()
            .filter_map(|s| match s {
                Activation::Hidden(h) => Some(h.clone()),
                Activation::Ids(_) => None,
            })
            .collect();
        let views: Vec<_> = mats.iter().map(|m| m.view()).collect();
        Activation::Hidden(concatenate(Axis(0), &views).expect("concat hidden rows"))
    }
}

#[test]
fn cache_equals_stateless_over_random_sessions() {
    let golden = Golden::load(golden_dir()).expect("load golden fixture");
    let cfg = golden.config();
    let weights: Weights = golden.weights().expect("load weights");
    let bounds = shard(&cfg, 4);

    let mut worst = 0.0_f32;
    for seed in 0..16_u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // multi-token prefill + >= 2 decode steps, sizes vary per session
        let n_prompt = rng.random_range(2..=8);
        let n_decode = rng.random_range(2..=5);

        for &(lo, hi) in &bounds {
            let stage = Stage::new(cfg.clone(), &weights, lo, hi);
            let is_first = lo == 0;
            let n_layers = hi - lo;

            // Build the session's step inputs (one prefill block, then decodes).
            let mut steps: Vec<Activation> = Vec::with_capacity(1 + n_decode);
            steps.push(random_input(&mut rng, &cfg, is_first, n_prompt));
            for _ in 0..n_decode {
                steps.push(random_input(&mut rng, &cfg, is_first, 1));
            }

            // Online: run each step against a growing cache, base_pos = cache len.
            let mut caches: Vec<KVCache> = (0..n_layers).map(|_| KVCache::new()).collect();
            let mut online_rows: Vec<Array2<f32>> = Vec::with_capacity(steps.len());
            let mut base = 0_usize;
            for step in &steps {
                let n = step.n_tokens();
                let out = stage
                    .forward(step.clone(), Some(caches.as_mut_slice()), base)
                    .expect("forward online");
                online_rows.push(out);
                base += n;
            }
            let online_views: Vec<_> = online_rows.iter().map(|m| m.view()).collect();
            let online = concatenate(Axis(0), &online_views).expect("concat online out");

            // Stateless: re-run over the whole input sequence at once.
            let stateless = stage
                .forward(cumulative_input(&steps, is_first), None, 0)
                .expect("forward stateless");

            let r = rel_l2(&online, &stateless);
            worst = worst.max(r);
            assert!(
                r < 1e-4,
                "stage [{lo},{hi}) seed {seed}: cache<->stateless rel-L2 {r:e} >= 1e-4"
            );
        }
    }

    println!("worst cache<->stateless rel-L2 = {worst:e}");
    assert!(worst < 1e-4);
}
