# HOPPER

[![CI](https://github.com/RealMithrandir/hopper/actions/workflows/ci.yml/badge.svg)](https://github.com/RealMithrandir/hopper/actions/workflows/ci.yml)

**A peer-to-peer network that pools heterogeneous consumer hardware to serve LLMs
too large for any single machine — no token, no blockchain, no gas.** Incentives
come from a contribution ledger; trust comes from re-execution audits backed by
stake.

> **Mental model in one line:** the model is split *by layer* across nodes; the
> large KV cache stays pinned on the node that owns those layers, and only a tiny
> hidden-state activation crosses the wire. **Move the small thing, pin the big
> thing.**

```
per-hop activation payload = d_model × dtype   (constant; independent of context)
KV cache (NEVER shipped)   = 2 × n_layers × n_heads × head_dim × dtype × context  (grows)
```

That inversion — moving the small thing, pinning the big thing — is the whole
design. The original v0.2 design shipped the KV cache between nodes and died on
residential uplinks; v0.3 (this repo) fixes that.

---

## Table of contents

- [Why HOPPER](#why-hopper)
- [The seven invariants](#the-seven-invariants)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Project status](#project-status)
- [Quick start](#quick-start)
  - [Build, test, lint](#build-test-lint)
  - [Run the Python reference oracle](#run-the-python-reference-oracle)
  - [Run a multi-process swarm](#run-a-multi-process-swarm)
  - [Serve the OpenAI-compatible API](#serve-the-openai-compatible-api)
- [The golden fixture](#the-golden-fixture)
- [Wire protocol](#wire-protocol)
- [Testing strategy](#testing-strategy)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why HOPPER

Large models don't fit on one consumer GPU, and datacenter techniques for
splitting them assume NVLink/InfiniBand. Over a 30 Mbps residential uplink, the
naive approaches collapse:

| v0.2 mechanism that failed | Why | v0.3 replacement |
|---|---|---|
| Ship the **KV cache** between nodes (prefill/decode disaggregation) | The cache is huge and grows with context — minutes per request over a home uplink | **Layer-sharded pipeline:** the cache stays resident; only a `d_model`-sized activation hops, constant per token |
| **Per-token pipelining** across full-model nodes | Inserts a WAN round-trip between every generated token — strictly slower than one node | Pipeline parallelism over the **layer** dimension on the *same* token |
| **Bit-exact** matmul verification | LLM matmuls aren't bit-reproducible across GPUs/Macs; tight ε false-bans honest nodes, loose ε lets cheaters hide | **Economic + statistical** proof-of-inference: commit-reveal → random re-execution → calibrated relative-error band → stake slashing |

A complete, runnable **Python reference implementation** lives in
[`reference/`](reference/) and is the semantic oracle for every behavior. The
production target is **Rust** (tokio + libp2p + QUIC + protobuf), built to
reproduce the reference's behavior and invariants and then extend to a real
network and real models.

> ⚠️ This is a **systems reference, not a model release.** Weights are random, so
> generations are gibberish — the point is that the *distributed mechanics* are
> correct and measurable.

---

## The seven invariants

These are the spine of the design (full text in [`CLAUDE.md`](CLAUDE.md)):

1. **The KV cache never crosses the network.** The only inter-node payload is the
   `[n_tokens, d_model]` activation. The cache is pinned per-session on the node
   that owns those layers.
2. **Pipeline parallelism is over LAYERS, on the same token** — never over tokens
   across full-model nodes.
3. **The audit invariant must hold and stay tested.** A stage run *online with a
   KV cache* equals the same stage *re-run statelessly* over the full input
   sequence, within fp tolerance. This is what makes verification possible.
4. **Verification is economic, not bit-exact:** commit-reveal → random independent
   re-execution → relative-L2 error vs. a hardware-calibrated tolerance band →
   slash + ban on mismatch. Never compare to zero tolerance.
5. **Accounting is FLOP-metered, decayed, and consensus-free.** Standing is a
   time-decayed contribution ratio; optimistic-unchoke is a *floor* for fresh
   staked identities; minting an identity costs proof-of-work; enforcement is
   stake-slash + identity-ban, never IP blacklisting.
6. **Determinism contract.** Every node materializes byte-identical weights for a
   given `model_hash`. Audits are only comparable because of this.
7. **Routing optimizes latency, not bandwidth.** Activations are tiny, so a few
   extra WAN hops are cheap; pick the lowest-RTT eligible provider per stage and
   re-route around banned/slashed/dropped nodes mid-stream.

---

## Architecture

Three layers:

- **L1 Inference** — the model is partitioned into `S` contiguous **stages** (layer
  ranges). Stage 0 owns the embeddings; stage `S-1` owns the final norm + LM head.
  Each stage holds its own per-session KV cache locally.
- **L2 Routing & Ledger** — a Kademlia DHT maps `stage_id → {providers, RTT}`; the
  router builds the lowest-latency pipeline and re-routes on failure. The ledger
  tracks decayed FLOP contribution ratios and access tiers.
- **L3 Application** — an OpenAI-compatible `/v1/chat/completions` facade, so
  existing SDKs point straight at it.

```
client ─ids/activation▶ [stage 0] ─act▶ [stage 1] ─act▶ … ─▶ [stage S-1] ─▶ logits
                            │              │                       │
                       local KV cache  local KV cache         local KV cache
                       (never shipped) (never shipped)        (never shipped)
                                                               sample ▼
                                          next token re-enters at stage 0
```

The first real-network milestone is **coordinator-mediated**: an unprivileged
*coordinator* daemon role discovers stage providers via Kademlia and drives the
token loop, each stage hop being a libp2p QUIC request-response to the worker
hosting that stage; reroute is coordinator-driven on request failure. Full
peer-relay decentralization (`stage k → stage k+1` directly) is a later phase.
See [`docs/TECH_SPEC.md §7.1`](docs/TECH_SPEC.md).

---

## Repository layout

```
.
├── CLAUDE.md                  # the binding contract: invariants + phased plan
├── docs/TECH_SPEC.md          # full design rationale ("why")
├── reference/                 # the Python reference impl — the semantic oracle
│   ├── model.py node.py engine.py verify.py ledger.py router.py transport.py api.py demo.py
│   ├── export_golden.py       # exports deterministic weights + golden I/O vectors
│   └── golden/                # generated fixture (weights + golden vectors + manifest.json)
└── crates/
    ├── hopper-model/          # L1: layer-sharded transformer + KV cache   (~ model.py)
    ├── hopper-ledger/         # L2: decayed FLOP ratio, PoW identity, slash (~ ledger.py)
    ├── hopper-verify/         # L1.5: commit-reveal + re-exec audit         (~ verify.py)
    ├── hopper-net/            # L2: sim transport + router; libp2p QUIC swarm
    ├── hopper-engine/         # orchestration: Node + pipeline + reroute    (~ engine.py)
    ├── hopper-proto/          # protobuf wire types (prost)
    └── hopper-daemon/         # bin+lib: tokio daemon (worker/coordinator/serve)
```

| crate | mirrors | responsibility |
|---|---|---|
| `hopper-model` | `model.py` | `ModelConfig`, deterministic weights, `Stage::forward` (online vs. stateless), `KVCache`, byte tokenizer, golden loader |
| `hopper-ledger` | `ledger.py` | hashcash PoW, decayed contribution ratio (λ=ln2/3600), four access tiers, optimistic-unchoke floor, slash |
| `hopper-verify` | `verify.py` | `commit(out, nonce)`, `Execution`, re-execution audit vs. a calibrated tolerance band |
| `hopper-net` | `transport.py` + `router.py` | sim links + `NetworkMonitor`, latency-aware `Router`, **and** the libp2p QUIC swarm / Kademlia / request-response codec |
| `hopper-engine` | `node.py` + `engine.py` | `Node` (hosts stages, pins caches, logs I/O, tamper hook) + `Engine` (prefill/decode, metering, audit, mid-stream reroute) |
| `hopper-proto` | — | prost wire types: `InferenceRequest`, `ActivationStream`, `StageResponse`, `AuditChallenge`, `AuditReveal`, `FraudProof`, `TokenPublish` |
| `hopper-daemon` | `api.py` | tokio binary with `worker` / `coordinator` / `serve` roles + the axum OpenAI facade |

---

## Project status

Delivered in phases; each gate is green tests, not vibes.

| Phase | Scope | Status |
|---|---|---|
| **0** | Workspace + empty crates + CI + golden oracle exporter | ✅ done |
| **1** | `hopper-model`: load golden weights, match within `1e-4`, KV cache, cache↔stateless property test | ✅ done |
| **2** | Single-process swarm: ledger, verify, sim net, engine — reproduces `demo.py`'s four outcomes | ✅ done |
| **3** | Real network: libp2p QUIC + Kademlia + request-response; 2+ processes complete an inference and survive a mid-stream kill | ✅ done |
| **4a** | axum OpenAI facade (in-process) + fp16-on-the-wire, CI-tested on `hopper-tiny` | ✅ done |
| **4b** | candle real-model backend behind a `Stage` trait + coherent-completion acceptance gate | ⏳ planned |

The reference model `hopper-tiny` is tiny by design (vocab 256, `d_model` 128,
8 layers, 4 heads) so the whole swarm runs on one machine and tests stay fast and
deterministic.

---

## Quick start

Prerequisites: the pinned Rust toolchain installs automatically from
[`rust-toolchain.toml`](rust-toolchain.toml) (stable 1.93.1, with clippy +
rustfmt). Python 3 + NumPy for the reference oracle.

### Build, test, lint

The four gates CI runs on every push — all must be green:

```bash
cargo build  --workspace
cargo test   --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt    --all -- --check
```

### Run the Python reference oracle

```bash
pip install numpy
python3 reference/demo.py            # inference, verification, fraud, ledger
python3 reference/export_golden.py   # regenerate reference/golden/ deterministically
```

`demo.py` stands up a 4-node swarm and prints four measured outcomes: constant
per-hop payload, honest verification with zero false bans, a fraud caught +
slashed + rerouted, and correct decayed ledger tiers.

### Run a multi-process swarm

Build once (`cargo build`), then in separate terminals. Each **worker** hosts
stages and prints a `PEER <id> ADDR <multiaddr>` handshake line:

```bash
GOLDEN=$(pwd)/reference/golden

# worker 1 (hosts all four stages)
cargo run -p hopper-daemon -- worker --golden "$GOLDEN" --stages 0,1,2,3 --n-stages 4 --seed 1
# worker 2 (a redundant replica, so a kill can reroute)
cargo run -p hopper-daemon -- worker --golden "$GOLDEN" --stages 0,1,2,3 --n-stages 4 --seed 2
```

The **coordinator** discovers providers via Kademlia and drives the inference,
rerouting if a worker dies mid-stream:

```bash
cargo run -p hopper-daemon -- coordinator \
  --bootstrap <peer1>@<addr1> --bootstrap <peer2>@<addr2> \
  --n-stages 4 --prompt "explain the design" --max-tokens 16
```

(The `swarm_kill_reroute` integration test automates exactly this: spawn 2
workers + a coordinator, complete a generation, kill the primary mid-stream, and
assert the coordinator reroutes and finishes.)

### Serve the OpenAI-compatible API

```bash
cargo run -p hopper-daemon -- serve --bind 127.0.0.1:8080 --golden "$(pwd)/reference/golden" --n-stages 4
```

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"hopper-tiny","messages":[{"role":"user","content":"explain the design"}],"max_tokens":8}'
```

```json
{
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "length"}],
  "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32},
  "hopper": {"audits": 0, "audit_fails": 0, "reroutes": 0}
}
```

The `hopper` block is a non-standard telemetry extension; everything else matches
the OpenAI chat-completions shape. (Content is gibberish — random weights.)

---

## The golden fixture

`reference/export_golden.py` freezes the Python oracle into a language-neutral
fixture under `reference/golden/`, used as the Phase-1 parity oracle and by
several Rust tests:

- **Full model weights** — every tensor `build_weights()` materializes.
- **Golden (input → stage output) vectors** for the **first**, a **middle**, and
  the **last** stage, captured over a multi-token prefill and several decode steps
  — enough to prove both numeric parity (`<1e-4` rel-L2) and the cache↔stateless
  audit invariant.

The format is deliberately boring: raw little-endian `f32`/`i64` `.bin` blobs plus
a `manifest.json` describing each tensor's name, shape, dtype, and path — loadable
from `std::fs` + `f32::from_le_bytes` alone, no exotic crates. Output is a pure
function of the model name, so re-running regenerates byte-identical files.

---

## Wire protocol

`hopper-proto` defines the protobuf messages. The only inter-node payload is the
`Activation` — `[n_tokens, d_model]` in the model's working dtype (**fp16** on the
wire, halving bytes; the KV cache is never represented). Phase 3 actively uses
`ActivationStream` (request) + `StageResponse` (response) over libp2p
request-response; the remaining types are defined for the full protocol.

| Message | Direction | Payload |
|---|---|---|
| `InferenceRequest` | client → stage 0 | prompt ids, `model_hash`, session |
| `ActivationStream` | stage k → stage k+1 | `[n_tokens, d_model]` activation + `session` + `seq_pos` |
| `StageResponse` | worker → coordinator | output activation/logits + audit commitment + nonce + FLOPs |
| `AuditChallenge` | verifier → worker | `session`, `stage_id`, nonce request |
| `AuditReveal` | worker → verifier | logged I/O + `nonce` satisfying the prior commitment |
| `FraudProof` | verifier → bucket (gossip) | offending `node_id`, recompute delta, evidence |
| `TokenPublish` | stage S-1 → client | sampled token + telemetry |

---

## Testing strategy

Tests are how the invariants survive a long build. Two lanes:

- **CI (blocking):** `cargo test --workspace` — fast, deterministic, no network
  downloads, **no skips**. Property (cache↔stateless), golden/parity, security
  (fraud caught, commitment tamper, honest drift passes), and integration tests
  (the `demo.py` four outcomes; the multi-process swarm + mid-stream kill) all run
  here against `hopper-tiny` and the golden fixture.
- **Acceptance (non-blocking):** heavy end-to-end checks needing a real-model
  download (the Phase-4b coherent-completion gate) are feature-flagged / `#[ignore]`d,
  assert a deterministic greedy golden-token sequence, and run locally before
  release + optional nightly.

---

## Roadmap

Beyond Phase 4b: quorum-based challenge resolution for in-band fraud; cold-cache
warm-up via input-stream replay on reroute; gossipsub reputation + `FraudProof`
propagation (lands with peer-relay); optional TEE attestation; and a paid tier
layered on the ledger without redesign. See [`docs/TECH_SPEC.md §8`](docs/TECH_SPEC.md).

---

## Contributing

- Read [`CLAUDE.md`](CLAUDE.md) first — it is the binding contract (invariants,
  conventions, phased plan).
- Work in small, reviewable commits scoped to one crate/feature; keep `main`
  green. Run fmt + clippy + test before declaring anything done.
- The Python `reference/` is the oracle: match its behavior (run it and compare)
  rather than inventing semantics.
- Never add a dependency without calling it out.

---

## License

Apache-2.0 (declared in the workspace `Cargo.toml`).
