# CLAUDE.md — HOPPER

> This file is the **always-loaded contract** for building HOPPER. Read it every
> session. Deep detail lives in `docs/TECH_SPEC.md` and the executable spec in
> `reference/` (the working Python implementation). When this file and your
> instinct disagree, this file wins. When this file and `docs/TECH_SPEC.md`
> disagree, the spec wins and you flag the contradiction.

---

## What we are building

A peer-to-peer network that pools heterogeneous consumer hardware to serve LLMs
too large for one machine. No token, no blockchain, no gas. Incentives come from a
contribution ledger; trust comes from re-execution audits backed by stake.

The production target is **Rust** (tokio + libp2p + QUIC + protobuf), matching the
crate layout in `docs/TECH_SPEC.md §7`. A complete, runnable **Python reference
implementation** already exists in `reference/` and is the semantic oracle for
every behavior. Your job is to build the Rust system so that it reproduces the
reference's *behavior and invariants*, then extends it to a real network and real
models.

**Mental model in one line:** the model is split *by layer* across nodes; the
large KV cache stays pinned on the node that owns those layers; only a tiny
hidden-state activation crosses the wire. Move the small thing, pin the big thing.

---

## THE INVARIANTS (do not violate; each prevents a specific, fatal regression)

These are the spine of the design. They are the things the original v0.2 spec got
wrong. If a change would break one of these, stop and raise it — do not "improve"
your way around them.

1. **The KV cache never crosses the network.** The only inter-node payload is the
   activation `[n_tokens, d_model]` in the model's working dtype. The cache is
   pinned per-session on the node that owns those layers.
   *Prevents:* the v0.2 prefill/decode disaggregation that shipped gigabytes over
   residential uplinks. Bandwidth, not latency, was the killer.

2. **Pipeline parallelism is over LAYERS, on the same token — never over tokens
   across full-model nodes.** A token flows stage 0 → stage 1 → … → stage N-1 →
   sample → re-enter at stage 0.
   *Prevents:* the v0.2 scheme that put a WAN round-trip between every generated
   token (strictly slower than one node).

3. **The audit invariant must hold and stay tested.** A stage run *online with a KV
   cache* produces the same output as the same stage *re-run statelessly over the
   full input sequence*, within fp tolerance. Online decode is memoized
   full-sequence attention. **Every change to inference must keep a property test
   asserting this** (`reference` measures ≈1e-8 in fp32).
   *This is what makes verification possible.* If it breaks, verification is a lie.

4. **Verification is economic, not bit-exact.** Pipeline:
   `commit-reveal → random independent re-execution → relative-L2 error vs. a
   hardware-calibrated tolerance band → slash + ban on mismatch`. Never compare to
   zero tolerance. Never assume cross-hardware matmuls are bit-identical (they are
   not). The commitment is published *before* the node learns whether it is
   audited.
   *Prevents:* the v0.2 "IEEE-754 ε" hand-wave that would false-ban honest nodes
   and let cheaters hide.

5. **Accounting is FLOP-metered, decayed, and consensus-free.** Work is priced in
   compute (FLOPs), not raw tokens. Standing is a time-decayed contribution ratio.
   Optimistic-unchoke is a *floor* for fresh staked identities, never a ceiling
   that demotes contributors. Minting an identity costs proof-of-work. Reputation
   is gossiped and advisory; the authoritative signal is local tit-for-tat.
   Enforcement is **stake-slash + identity-ban**, never IP blacklisting.
   *Prevents:* token-farming, cold-start lockout, whitewashing, and the
   "no-blockchain-but-needs-global-state" contradiction.

6. **Determinism contract.** Every node materializes byte-identical weights for a
   given `model_hash`. Audits are only comparable because of this.

7. **Routing optimizes latency, not bandwidth.** Because activations are tiny, a
   few extra WAN hops are cheap. Pick the lowest-RTT eligible provider per stage;
   re-route around banned/slashed/dropped nodes mid-stream.

---

## Repository layout

```
.
├── CLAUDE.md                  # this file
├── docs/
│   └── TECH_SPEC.md           # full design rationale (authoritative for "why")
├── reference/                 # the Python reference impl — the semantic oracle
│   ├── model.py node.py engine.py verify.py ledger.py
│   ├── router.py transport.py api.py demo.py
│   └── golden/                # exported weights + golden I/O vectors (see Phase 0)
├── crates/
│   ├── hopper-model/          # L1  layer-sharded transformer + KV cache  (~ model.py)
│   ├── hopper-ledger/         # L2  contribution ratio, PoW identity, slash (~ ledger.py)
│   ├── hopper-verify/         # L1.5 commit-reveal + re-exec audit         (~ verify.py)
│   ├── hopper-net/            # L2  transport + router (sim first, libp2p later)
│   ├── hopper-engine/         # orchestration: pipeline, accounting, reroute (~ engine.py)
│   ├── hopper-proto/          # protobuf wire types (prost)
│   └── hopper-daemon/         # bin: tokio daemon + OpenAI-compatible HTTP    (~ api.py)
└── Cargo.toml                 # workspace
```

> **Setup the human does once:** drop the Python files into `reference/`, the spec
> into `docs/TECH_SPEC.md`. If `reference/` is missing, ask for it before guessing
> at semantics — it is the source of truth for behavior.

---

## Toolchain & commands

```bash
# build / lint / format — all must be clean before any task is "done"
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# run the reference oracle (Python) to compare behavior / regenerate goldens
python3 reference/demo.py
python3 reference/export_golden.py        # you will write this in Phase 0

# run the Rust daemon (Phase 4+)
cargo run -p hopper-daemon -- --config dev.toml
```

Rust edition 2021. Pin the toolchain in `rust-toolchain.toml`. CI must run the four
checks above on every PR.

**Dependencies (introduce only as the phase needs them; never add silently):**
`tokio`, `ndarray` (Phase 1 inference), `sha2`, `rand` + `rand_chacha`, `thiserror`,
`anyhow` (bins only), `serde`/`serde_json`, `tracing`. Later: `libp2p`
(kad, gossipsub, quic, request-response), `prost` + `prost-build`, `axum`,
`candle-core`/`candle-nn` (or llama.cpp bindings) for real models.

---

## Architecture (concise; see `docs/TECH_SPEC.md` for full treatment)

Three layers:

- **L1 Inference** — model partitioned into `S` contiguous **stages** (layer
  ranges). Stage 0 owns embeddings; stage `S-1` owns final norm + LM head. Each
  stage holds its own per-session KV cache locally.
- **L2 Routing & Ledger** — DHT maps `stage_id → {providers, RTT}`; router builds
  the lowest-latency pipeline and re-routes on failure. Ledger tracks decayed
  FLOP contribution ratios and access tiers.
- **L3 Application** — OpenAI-compatible `/v1/chat/completions` facade.

**Wire protocol** (`hopper-proto`, protobuf over QUIC):

| Message | Direction | Payload |
|---|---|---|
| `InferenceRequest` | client → stage 0 | prompt ids, `model_hash`, client signature, `session` |
| `ActivationStream` | stage k → stage k+1 | `[n_tokens, d_model]` activation (working dtype) + `session` + `seq_pos` |
| `AuditChallenge` | verifier → worker | `session`, `stage_id`, nonce request |
| `AuditReveal` | worker → verifier | logged input stream, claimed output stream, `nonce` (must satisfy prior commitment) |
| `FraudProof` | verifier → bucket (gossip) | offending `node_id`, recompute delta, evidence |
| `TokenPublish` | stage S-1 → client | sampled token + telemetry |

---

## Crate specs (what each must do, and what "done" means)

For each crate: mirror the named reference file's semantics, preserve the relevant
invariants, and ship the listed tests green.

### `hopper-model`  (mirrors `reference/model.py`)
- `ModelConfig` (vocab, d_model, n_layer, n_head, d_ff, max_seq, eps) with
  `head_dim` and `model_hash = sha256(name)[..16]`.
- Deterministic weight materialization from `model_hash`.
- `Stage { lo, hi, is_first, is_last }` with `forward(x, caches: Option<&mut [KVCache]>, base_pos)`.
  `caches: Some` = online decode (mutates cache); `None` = stateless audit recompute.
- `KVCache` per (session, layer); `append`, `length`. **Never serialized to the wire.**
- Pre-norm blocks: RMSNorm → causal MHA (with absolute-position causal mask) →
  residual → RMSNorm → GELU MLP → residual. Tied LM head on the last stage.
- `flops(n_tokens) = 2 * stage_params * n_tokens`.
- **Done when:** (a) loading `reference/golden/weights.*`, a stage's output matches
  the golden activation within `1e-4` rel-L2; (b) a **property test** asserts
  online-cached output == stateless recompute within `1e-4` over random sessions
  (Invariant 3).

### `hopper-ledger`  (mirrors `reference/ledger.py`)
- `proof_of_work(id, difficulty)` (hashcash) for identity minting.
- `Account { stake, seeded[], leeched[], banned }`.
- `record(provider, consumer, flops)`; decayed sums with `LAMBDA = ln(2)/3600`
  (≈1h half-life).
- `contribution_ratio`, `access -> (tier, queue_delay_ms)` with tiers
  `priority ≥1.0`, `delayed 0.2–1.0`, `throttled <0.2`; optimistic-unchoke **floor**
  for fresh staked ids under a FLOP grant inside a window.
- `slash(id)` → stake 0, banned, seeded credit erased.
- **Done when:** unit tests cover decay over time, the four tiers, optimistic-unchoke
  as a floor (high-ratio node still `priority`), PoW verify-cheap/mint-costly, and
  slash idempotence.

### `hopper-verify`  (mirrors `reference/verify.py`)
- `commit(output, nonce) = sha256(output_bytes ‖ nonce)`.
- `Execution { session, stage_id, worker, inp, out, nonce, commitment }` logged by
  the worker over the **cumulative** session I/O stream.
- `Verifier::maybe_audit` → with prob `p`: verify commitment binds the reveal, then
  stateless re-execute the stage and compare rel-L2 to `tolerance` (default `2e-3`),
  with optional injected `hw_noise` to model cross-hardware drift. Mismatch → slash.
- **Done when:** tests show (a) honest node with `hw_noise ≈ 2e-4` passes at `p=1.0`
  with **zero** false bans; (b) a fabricating node is caught and slashed; (c)
  commitment tampering is caught. (Invariant 4.)

### `hopper-net`  (transport + router; sim first, libp2p in Phase 3)
- Sim transport: `LinkProfile { rtt_ms, up_mbps }`, `NetworkMonitor` accumulating
  bytes + simulated latency and a `kv_ship_bytes_avoided` counterfactual.
- `Router`: `announce(node, stage, rtt)`, `assemble(n_stages, exclude)` picking
  min `(rtt + queue_delay)` eligible provider per stage; raise cleanly if a stage
  is uncovered.
- **Done when:** an integration test asserts per-hop activation bytes `== d_model*4`
  (Invariant 1) and that excluding a provider reroutes to a backup.

### `hopper-engine`  (mirrors `reference/engine.py`)
- Orchestrates one inference: prefill then decode; each hop meters bytes/latency,
  credits FLOPs to the ledger, logs for audit, spot-checks via the verifier, and
  **re-assembles the pipeline around a node slashed mid-stream**, continuing
  generation.
- **Done when:** an integration test reproduces `demo.py`'s four outcomes
  (inference with constant per-hop payload; honest audit 0 false bans; fraud caught
  + slashed + rerouted; ledger tiers correct).

### `hopper-proto` / `hopper-daemon`
- `hopper-proto`: prost messages above; codegen in `build.rs`.
- `hopper-daemon`: tokio runtime, node identity/keypair, config loader, and an
  `axum` server exposing `POST /v1/chat/completions` returning the OpenAI shape with
  a non-standard `hopper` telemetry block.
- **Done when:** the HTTP route returns `200` with `object: "chat.completion"` and
  valid `usage`, served by the in-process engine (Phase 2), then across processes
  (Phase 3+).

---

## Phased delivery (ship in order; each gate is green tests, not vibes)

**Phase 0 — Scaffold + golden oracle.** Create the workspace, empty crates, CI
(fmt/clippy/test). Write `reference/export_golden.py` that dumps deterministic
weights and a set of `(input → stage output)` golden vectors to `reference/golden/`.
*Gate:* `cargo test` runs; goldens exist; CI green.

**Phase 1 — Inference core.** Implement `hopper-model` in Rust (ndarray). Load the
reference weights, match golden stage outputs within `1e-4`, implement the KV cache,
and ship the cache↔stateless property test.
*Gate:* parity test + property test green (Invariants 1, 3, 6).

**Phase 2 — Single-process swarm.** Implement `hopper-ledger`, `hopper-verify`, sim
`hopper-net`, and `hopper-engine` so one process simulates N nodes and reproduces
`demo.py`. No real networking yet (this is the v0.2 "loopback first" discipline,
kept).
*Gate:* an integration test asserts all four `demo.py` outcomes; per-hop payload
assertion holds (Invariants 1, 4, 5, 7).

**Phase 3 — Real network.** Swap sim transport for `libp2p`: Kademlia stage
discovery, QUIC `ActivationStream` via request-response, gossipsub for reputation +
`FraudProof`. Two OS processes on localhost form a swarm and complete an inference;
killing one mid-stream triggers a reroute.
*Gate:* a multi-process test (spawn 2+ daemons) completes generation and survives a
mid-stream node kill.
> **Phase 3 scope note (decided 2026-06-18):** topology is **coordinator-mediated**
> — an *unprivileged daemon role* drives the token loop and each stage hop is a real
> libp2p QUIC request-response to the remote worker hosting that stage; discovery is
> Kademlia; reroute is coordinator-driven on request failure. Full **peer-relay**
> decentralization (`stage k → stage k+1` directly, per the wire table) is deferred
> to a later phase. The DoD is unchanged (2+ processes, inference completes,
> mid-stream kill reroutes). All wire-table message types land in `hopper-proto`, but
> `FraudProof`/gossipsub may be stubbed (defined, not actively gossiped) if wiring
> them threatens multi-process test stability. See `docs/TECH_SPEC.md §7`.

**Phase 4 — Real models + API.** Add a real inference backend (candle/llama.cpp,
GGUF) behind the same `Stage` interface; ship the `axum` OpenAI facade; use fp16 on
the wire. Serve a small real model across 2 processes.
*Gate:* `/v1/chat/completions` returns a coherent completion from a real model
served across processes.
> **Phase 4 split (decided 2026-06-19):** **4a** ships the `axum` OpenAI facade +
> fp16-on-the-wire, **fully CI-tested** against the deterministic `hopper-tiny`
> model (golden parity); the facade is first served by the in-process engine. **4b**
> adds the candle real-model backend behind a `Stage` backend trait; its
> *coherent-completion* gate is an **acceptance test** (real model download), so per
> the CI/acceptance split below it is feature-flagged / `#[ignore]`d, asserts a
> deterministic greedy golden-token sequence, and runs **locally before release +
> optional non-blocking nightly** — not on the blocking CI lane.

Later, non-blocking: quorum-based challenge resolution for in-band fraud; cold-cache
warm-up via input-stream replay on reroute; optional TEE attestation; a paid tier
layered on the ledger. (See `docs/TECH_SPEC.md §8`.)

---

## Coding conventions (Rust)

- **Errors:** libraries return `Result<_, ThisError>` (per-crate `thiserror` enums);
  only `hopper-daemon` uses `anyhow`. **No `unwrap`/`expect`/`panic!` on non-test
  paths** except documented invariants that are truly unreachable.
- **Async:** `tokio`; never block the runtime (no sync I/O or heavy compute on async
  tasks — offload model math to `spawn_blocking` or a dedicated pool).
- **Concurrency:** prefer message passing (`tokio::mpsc`) over shared locks. If a
  lock is unavoidable, hold it briefly; document why. Tensors crossing tasks must be
  `Send`.
- **Determinism:** seed RNGs explicitly (`rand_chacha`); no ambient entropy in the
  inference or weight paths. Tests must be reproducible.
- **No hidden state on the wire:** the only thing serialized between stages is the
  activation. Asserting this in code review is part of every networking PR.
- **Modules small and named by responsibility.** Public API minimal; keep the
  `Stage`/`InferenceEngine` trait boundary stable so backends are swappable.
- Document every public item; doc-comment the *why*, not the *what*.

## Testing strategy

Tests are how invariants survive a long agentic build. Required, by category:

- **Property:** cache↔stateless equivalence (Inv 3); ledger decay monotonicity;
  optimistic-unchoke never demotes a contributor.
- **Golden/parity:** Rust stage output vs. `reference/golden/` (Inv 6).
- **Security:** fabricating node caught + slashed; commitment-tamper caught; honest
  cross-hardware drift passes (Inv 4).
- **Integration:** `demo.py`-equivalent four-outcome test (Phase 2); multi-process
  swarm + mid-stream kill (Phase 3).
- **Assertion guards:** per-hop activation bytes `== d_model * dtype_size` (Inv 1).

Run `cargo test --workspace` before declaring any task done. A task with red or
skipped tests is not done.

**CI vs. acceptance split.** Two lanes:
- **CI (blocking):** `cargo test --workspace` — fast, deterministic, no network
  downloads, **no skips**. Everything provable against `hopper-tiny` and the golden
  fixture lives here and must stay green.
- **Acceptance (non-blocking):** heavy end-to-end checks that need a real model
  download (the Phase-4 *coherent-completion* gate). These are gated behind a Cargo
  feature and/or `#[ignore]`, assert a **deterministic greedy golden-token**
  sequence (temperature 0), and are run **locally before release** and optionally on
  a **non-blocking nightly** workflow. A skip here is allowed *only* for this class;
  the inability to run it in blocking CI must be because of the model download, not
  because the test is flaky or unfinished.

---

## DO NOT (regressions and traps — each maps to an invariant)

- ❌ Serialize or ship the KV cache between nodes. (Inv 1)
- ❌ Pipeline tokens across full-model nodes / put a round-trip between tokens. (Inv 2)
- ❌ Compare audit outputs with zero / near-zero tolerance, or assume bit-exact
  cross-hardware matmuls. (Inv 4)
- ❌ Add a blockchain, a coin/token, gas, or any global-consensus ledger. (Inv 5)
- ❌ Price work in raw tokens instead of FLOPs. (Inv 5)
- ❌ Ban or rate-limit by IP address. Use stake-slash + identity-ban. (Inv 5)
- ❌ Let optimistic-unchoke override a high contribution ratio. (Inv 5)
- ❌ Claim a task is complete with failing/skipped tests or clippy warnings. (The
  *only* sanctioned skip is the non-blocking real-model **acceptance** class above —
  feature-flagged/`#[ignore]`d because of a model download, never because it is
  flaky or unfinished.)
- ❌ Add a heavyweight dependency (libp2p, candle, etc.) outside its phase, or any
  dep, without calling it out in the PR description.
- ❌ Invent reference behavior — run `reference/` and match it, or ask.

## Agent operating rules

- Work in **small, reviewable PRs** scoped to one crate/feature; keep `main` green.
- Before coding a behavior, **read the mirrored `reference/*.py`** — it is the
  oracle. When unsure about a numeric or protocol detail, run the reference and
  compare rather than guessing.
- After each change: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`. Report
  what you ran and the results.
- When you hit an apparent conflict between an invariant and a requested change,
  **stop and surface it** — do not silently relax an invariant.
- Update `docs/TECH_SPEC.md` if you intentionally change a design decision, and note
  it in the PR. Keep CLAUDE.md current if commands or layout change.

---

## Glossary / key constants

- **Stage** — contiguous block of transformer layers owned by a node.
- **Activation** — `[n_tokens, d_model]` hidden state; the *only* inter-node payload.
- **`model_hash`** — `sha256(model_name)[..16]`; pins deterministic weights.
- **`C_ratio`** — decayed Σseeded·e^(−λΔt) / Σleeched·e^(−λΔt), `λ = ln2/3600`.
- **Audit tolerance** — default rel-L2 `2e-3`; sized per `(backend, dtype)`, never 0.
- **Audit invariant** — online-cached output == stateless recompute (≈1e-8 fp32).
- **Tiers** — `priority ≥1.0`, `delayed 0.2–1.0`, `throttled <0.2`,
  `optimistic_unchoke` (newcomer floor), `blocked` (banned).