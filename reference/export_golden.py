"""
export_golden.py — Freeze the reference oracle into a language-neutral fixture.

Phase 1 (hopper-model, Rust) must reproduce `reference/model.py` *exactly enough*:
load the same weights, match each stage's output, and preserve the cache↔stateless
audit invariant (CLAUDE.md Invariant 3). This script writes the data those tests
consume into `reference/golden/`:

  * the **full model weights** (every tensor build_weights() materializes), and
  * **golden (input -> stage output) vectors** for the FIRST, a MIDDLE, and the
    LAST stage, captured from a real pipeline run over a *multi-token prefill*
    plus several *decode* steps. For each captured stage we record both the
    per-step online (cached) I/O and the cumulative stateless recompute, so Phase 1
    can prove numeric parity AND the cache↔stateless equivalence.

Everything is a pure function of the model name (deterministic weights from
`model_hash`, greedy/argmax decode), so re-running regenerates byte-identical
output. The format is deliberately boring — raw little-endian binary blobs plus a
`manifest.json` describing every tensor's name, shape, dtype, and path — so a Rust
program can load it with nothing more exotic than `std::fs` + `f32::from_le_bytes`.

Run:  python3 reference/export_golden.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

# Import the oracle regardless of the caller's working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import (  # noqa: E402  (path shim must run first)
    KVCache,
    ModelConfig,
    Stage,
    build_weights,
    encode,
    shard,
)

GOLD = os.path.join(HERE, "golden")

# --- fixture parameters (fixed, so the export is deterministic) -------------
N_STAGES = 4              # how the model is sharded across the swarm
PROMPT = "hopper"         # multi-token prefill (6 byte-tokens)
N_DECODE = 3              # >= 2 subsequent decode steps (Invariant 3 needs >=2)
# Which stages to freeze: first, a genuine middle, and last. The full pipeline
# always runs so middle/last receive realistic upstream activations; we only
# write golden files for these three.
CAPTURE = {0: "first", N_STAGES // 2: "middle", N_STAGES - 1: "last"}


def _as_le(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a contiguous little-endian copy + its manifest dtype label.

    Floats are forced to f32 (the model's working dtype) and integer token ids to
    i64 (numpy's default int), both explicitly little-endian so the bytes are
    platform-independent."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.ascontiguousarray(arr.astype("<f4")), "f32"
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr.astype("<i8")), "i64"
    raise TypeError(f"unsupported dtype for golden export: {arr.dtype}")


def write_tensor(arr: np.ndarray, rel_path: str, name: str) -> dict:
    """Write one tensor as a raw LE blob and return its manifest descriptor."""
    le, dtype = _as_le(np.asarray(arr))
    full = os.path.join(GOLD, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(le.tobytes(order="C"))
    return {
        "name": name,
        "shape": list(le.shape),
        "dtype": dtype,
        "path": rel_path,
        "n_elements": int(le.size),
    }


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error, matching verify.py's audit metric (ref vs. baseline)."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main() -> None:
    cfg = ModelConfig()
    weights = build_weights(cfg)
    bounds = shard(cfg, weights, N_STAGES)
    stages = [Stage(cfg, weights, lo, hi) for (lo, hi) in bounds]

    # Fresh output dir so removed tensors never linger between runs.
    if os.path.isdir(GOLD):
        shutil.rmtree(GOLD)
    os.makedirs(GOLD)

    # --- 1) full model weights ------------------------------------------------
    weight_descs: list[dict] = []
    weight_descs.append(write_tensor(weights["tok_emb"], "weights/tok_emb.bin", "tok_emb"))
    weight_descs.append(write_tensor(weights["pos_emb"], "weights/pos_emb.bin", "pos_emb"))
    weight_descs.append(write_tensor(weights["final_g"], "weights/final_g.bin", "final_g"))
    for li, lw in enumerate(weights["layers"]):
        for attr in ("wqkv", "wo", "g1", "g2", "w1", "w2"):
            name = f"layer_{li:02d}.{attr}"
            weight_descs.append(
                write_tensor(getattr(lw, attr), f"weights/{name}.bin", name)
            )

    # --- 2) drive a real pipeline run, capturing per-stage I/O ----------------
    # Per stage: one KVCache per layer it owns. Mirrors node.py's local caches.
    caches = [[KVCache() for _ in range(hi - lo)] for (lo, hi) in bounds]
    # capture[s] = list of {kind, base_pos, input(np), output(np)}
    capture: dict[int, list[dict]] = {s: [] for s in CAPTURE}

    def run_pipeline(x: np.ndarray, kind: str) -> np.ndarray:
        """Push one activation through every stage online; return final logits.

        base_pos is read from the stage's own KV cache length, exactly as
        node.run_stage does, so positions line up with a stateless recompute."""
        for s in range(N_STAGES):
            base_pos = caches[s][0].length()
            inp = np.asarray(x).copy()
            out = stages[s].forward(x, caches[s], base_pos)
            if s in capture:
                capture[s].append(
                    {"kind": kind, "base_pos": int(base_pos),
                     "input": inp, "output": np.asarray(out).copy()}
                )
            x = out
        return x

    prompt_ids = encode(PROMPT)                       # int64 [n_prompt]
    logits = run_pipeline(prompt_ids, "prefill")      # multi-token prefill

    gen_ids: list[int] = []
    for _ in range(N_DECODE):                          # greedy decode, like engine.py
        nxt = int(logits[-1].argmax())
        gen_ids.append(nxt)
        logits = run_pipeline(np.array([nxt], dtype=np.int64), "decode")

    # --- 3) per-stage golden vectors + cumulative stateless recompute ---------
    stage_descs: list[dict] = []
    for s, label in CAPTURE.items():
        st = stages[s]
        steps = capture[s]

        step_descs = []
        for i, step in enumerate(steps):
            tag = f"step_{i:02d}_{step['kind']}"
            step_descs.append({
                "index": i,
                "kind": step["kind"],
                "base_pos": step["base_pos"],
                "n_tokens": int(np.asarray(step["input"]).shape[0]),
                "input": write_tensor(
                    step["input"], f"vectors/{label}/{tag}.input.bin",
                    f"{label}.{tag}.input"),
                "output": write_tensor(
                    step["output"], f"vectors/{label}/{tag}.output.bin",
                    f"{label}.{tag}.output"),
            })

        # Cumulative session stream (concat over steps), exactly what node.py logs
        # and verify.py re-executes. The audit invariant: stateless recompute over
        # cum_input == the online (cached) outputs concatenated.
        cum_input = np.concatenate([s_["input"] for s_ in steps], axis=0)
        cum_online = np.concatenate([s_["output"] for s_ in steps], axis=0)
        cum_stateless = st.forward(cum_input.copy(), None, 0)
        audit_rel = rel_l2(cum_online, cum_stateless)
        # Sanity: the invariant Phase 1 must preserve had better hold here too.
        assert audit_rel < 1e-4, f"audit invariant broken for stage {s}: {audit_rel}"

        stage_descs.append({
            "label": label,
            "stage_id": s,
            "layer_lo": st.lo,
            "layer_hi": st.hi,
            "n_layers": st.hi - st.lo,
            "is_first": st.is_first,
            "is_last": st.is_last,
            "input_kind": "token_ids" if st.is_first else "hidden",
            "output_kind": "logits" if st.is_last else "hidden",
            "n_params": int(st.n_params()),
            "steps": step_descs,
            "cumulative": {
                "input": write_tensor(
                    cum_input, f"vectors/{label}/cumulative.input.bin",
                    f"{label}.cumulative.input"),
                "output_online": write_tensor(
                    cum_online, f"vectors/{label}/cumulative.output_online.bin",
                    f"{label}.cumulative.output_online"),
                "output_stateless": write_tensor(
                    cum_stateless, f"vectors/{label}/cumulative.output_stateless.bin",
                    f"{label}.cumulative.output_stateless"),
                # Diagnostic only (reference measures ~1e-8 in fp32); the .bin
                # blobs above are the bit-exact artifacts Phase 1 consumes.
                "audit_rel_l2": round(audit_rel, 12),
            },
        })

    # --- 4) manifest ----------------------------------------------------------
    manifest = {
        "format": "hopper-golden/1",
        "description": (
            "Deterministic export of reference/model.py: full weights + golden "
            "stage I/O for first/middle/last stages over a prefill and decode "
            "steps. Raw little-endian blobs; see per-tensor dtype/shape/path."
        ),
        "byte_order": "little-endian",
        "model": {
            "name": cfg.name,
            "model_hash": cfg.model_hash,
            "weight_seed": int(cfg.model_hash, 16) % (2**32),
            "weight_scale": 0.02,
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "head_dim": cfg.head_dim,
            "d_ff": cfg.d_ff,
            "max_seq": cfg.max_seq,
            "eps": cfg.eps,
        },
        "pipeline": {
            "n_stages": N_STAGES,
            "shard_bounds": [list(b) for b in bounds],
            "prompt": PROMPT,
            "prompt_ids": [int(i) for i in prompt_ids],
            "n_decode_steps": N_DECODE,
            "generated_ids": gen_ids,
        },
        "weights": weight_descs,
        "stages": stage_descs,
    }
    with open(os.path.join(GOLD, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # --- report ---------------------------------------------------------------
    n_tensors = len(weight_descs) + sum(
        2 * len(s["steps"]) + 3 for s in stage_descs
    )
    print(f"wrote golden fixture to {GOLD}")
    print(f"  model={cfg.name}  hash={cfg.model_hash}  "
          f"d_model={cfg.d_model}  n_layer={cfg.n_layer}")
    print(f"  weights: {len(weight_descs)} tensors")
    print(f"  stages : {', '.join(s['label'] for s in stage_descs)} "
          f"(prefill + {N_DECODE} decode steps each)")
    for s in stage_descs:
        print(f"    {s['label']:6s} stage {s['stage_id']} "
              f"layers[{s['layer_lo']},{s['layer_hi']})  "
              f"audit rel-L2={s['cumulative']['audit_rel_l2']:.2e}")
    print(f"  total tensors written: {n_tensors} (+ manifest.json)")


if __name__ == "__main__":
    main()
