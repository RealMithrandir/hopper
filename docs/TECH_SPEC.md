# HOPPER — Technical Specification (v0.3, reference implementation)

**Status:** working reference + architecture review
**Supersedes:** Project HOPPER v0.2.0-alpha

A peer-to-peer network that lets heterogeneous consumer hardware pool together to
serve LLMs too large for any single machine — with no token, no blockchain, and a
contribution ledger instead of gas fees. This document describes the design as
**re-architected to survive contact with wide-area residential networking**, and
maps it to the accompanying runnable reference implementation.

---

## 0. What changed from v0.2, and why

The v0.2 vision (pooled consumer inference, BitTorrent-style incentives,
proof-of-inference) is sound. Three of its *mechanisms* are not, and they were
load-bearing. v0.3 keeps the vision and replaces the mechanisms.

| v0.2 mechanism | Why it fails on residential P2P | v0.3 replacement |
|---|---|---|
| Disaggregated **prefill/decode**, shipping the **KV cache** between nodes | The KV cache is huge and grows with context (GBs at long context). Datacenter disaggregation works only because prefill↔decode sit on NVLink/InfiniBand. Over a 30 Mbps asymmetric uplink this is minutes per request. Low RTT does not buy bandwidth. | **Layer-sharded pipeline.** The model is split *by layer* across nodes; the KV cache stays **resident** on the node that owns those layers. Only a tiny hidden-state activation (`d_model × dtype`) crosses the wire, and it is **constant per token regardless of context length**. |
| **Per-token pipelining** across full-model nodes (Node A does token *n*, Node B token *n+1*) | Autoregressive decode is sequential; each full-model node needs all weights **and** the full growing cache, so this inserts a WAN round-trip *between every token* — the worst place to add latency. It is strictly slower than one node. | Pipeline parallelism over the **layer** dimension on the *same* token. Tokens stream through stages; KV locality means each stage only ever processes the new token(s). |
| **Bit-exact matmul verification** with "IEEE-754 tolerance ε" | LLM matmuls are **not** bit-reproducible across GPUs/Macs (reduction order, kernels, FMA). Tight ε false-bans honest nodes; loose ε lets cheaters hide. ε *is* the unsolved problem, not a footnote. | **Economic + statistical proof-of-inference:** commit-reveal, random independent **re-execution** audits, a **hardware-calibrated relative-error band**, and **stake slashing** that makes cheating negative-EV even when ε is loose. |

The single most important number in this spec, measured by the reference impl:

```
per-hop activation payload = d_model × 4 bytes  (constant; independent of context)
KV cache (NOT shipped)      = 2 × n_layers × n_heads × head_dim × 4 × context  (grows)
```

That inversion — moving the small thing, pinning the big thing — is the whole design.

---

## 1. Layered architecture

```
Layer 3  Application / API     OpenAI-compatible /v1/chat/completions facade
                               (clients keep their existing SDKs)
Layer 2  Routing & Ledger      latency-aware DHT stage discovery; decayed
                               contribution-ratio accounting (no consensus)
Layer 1  Sharded Inference     layer-partitioned transformer; KV cache pinned
                               per node; activations stream stage→stage
```

### Node roles (now uniform)

There is no Heavy/Light split. A node advertises which **stage(s)** (contiguous
layer ranges) it can host given its memory; a 64 GB box may host several stages, a
16 GB box one. Any node may also be transiently selected as a **Verifier**.

---

## 2. Layer 1 — Sharded inference pipeline

The model is partitioned into `S` contiguous **stages**, each a block of
transformer layers. Stage 0 also owns the embeddings; stage `S−1` owns the final
norm + LM head.

**Execution per generated token**

```
ids/activation ─▶ [stage 0 node] ─act▶ [stage 1 node] ─act▶ … ─▶ [stage S-1] ─▶ logits
                       │                     │                          │
                  local KV cache        local KV cache             local KV cache
                  (never shipped)       (never shipped)            (never shipped)
                                                                    sample ▼
                                              next token re-enters at stage 0
```

* The activation handed between stages is `[n_tokens, d_model]`: the full prompt
  during prefill, a **single row** during decode.
* Each stage keeps a per-session KV cache **for its own layers only**, so a decode
  step processes one token against locally-cached history — no context re-transfer.
* Determinism contract: weights are derived deterministically from the model name
  (`model_hash`), so every node materializes byte-identical weights. This is what
  makes audit re-execution comparable.

**The audit invariant** (proven in the reference impl to ~1e-8): a stage run online
*with* a KV cache produces the same output as the same stage re-run *statelessly*
over the full input sequence. Online decode is just memoized full-sequence
attention. Verification leans entirely on this equivalence.

---

## 3. Layer 2a — Routing

A Kademlia-style DHT maps `stage_id → {providers, measured RTT}`. The router
assembles the lowest-latency pipeline covering stages `0..S−1`, skipping
banned/throttled providers, and **re-assembles around a node that drops or is
slashed mid-stream**. Because inter-stage payloads are tiny, extra WAN hops are
cheap; the router optimizes **latency**, not bandwidth. (Production keeps the
v0.2 idea of clustering peers into low-RTT regional buckets — that part was fine.)

---

## 4. Layer 2b — The Tokenless Access Pact (accounting)

Incentive alignment without a coin. A node's standing is a **decayed contribution
ratio**:

```
C_ratio(t) = Σ T_seeded · e^(−λΔt)  /  Σ T_leeched · e^(−λΔt)
```

with the v0.2 access tiers (`≥1.0` priority, `0.2–1.0` progressive delay, `<0.2`
throttle). v0.3 closes the four gaps that made the v0.2 ledger inoperable:

* **Unit of account = FLOPs, not tokens.** Work is metered in compute, so a heavy
  64 GB prefill cannot be farmed at the price of a cheap decode step.
* **Cold start = optimistic unchoke.** A fresh, staked identity gets a bounded
  free FLOP grant inside a time window — a *floor* that rescues newcomers from the
  throttle so they can earn a ratio. It never demotes a contributing node.
* **Whitewashing = identity has a cost.** Minting an identity requires a one-time
  proof-of-work (hashcash). Churning to a new id to escape a bad ratio is no
  longer free, so the throttle has teeth.
* **No global consensus.** Reputation is gossiped and *advisory*; the
  authoritative signal is **local tit-for-tat** between direct peers. There is no
  ledger to agree on, consistent with "no blockchain."

`IP blacklisting` from v0.2 is dropped — CGNAT/dynamic IPs punish innocents and
don't stop a router reboot. Enforcement is **stake slashing + identity ban**.

---

## 5. Layer 1.5 — Verification (proof-of-honest-inference)

The Byzantine question: did the node actually do the math, or fabricate plausible
numbers to farm credit? v0.3 answers it without bit-exactness:

1. **Commit-reveal.** For each stage run the worker publishes
   `commitment = H(output ‖ nonce)` *before* learning whether it will be audited.
   It cannot adapt its answer to the challenge.
2. **Random independent re-execution.** With probability `p` an ephemeral verifier
   demands the worker's logged input stream and **re-executes the stage
   statelessly** on its own (identical) weights — using the audit invariant of §2.
3. **Hardware-calibrated tolerance.** Comparison is a **relative L2 band** sized to
   the verifier's hardware class, not to zero. Honest cross-hardware fp drift
   (≈1e-4–1e-3) lands inside; fabricated output lands far outside. *(The reference
   impl injects synthetic drift and shows honest nodes pass with zero false bans.)*
4. **Stake + slash.** Audits are indistinguishable from real traffic, so a rational
   worker must compute honestly on **everything**. A single catch forfeits the
   whole stake and bans the identity, making cheating negative-EV even at small `p`.

```
worker:   run stage → commit H(out‖nonce) → reveal (out, nonce) on challenge
verifier: H matches?  ──no──▶ slash + ban
              │yes
          re-execute stage(input) statelessly
          ‖out − out'‖ / ‖out'‖ ≤ tol ? ──no──▶ slash + ban + fraud-proof gossip
              │yes
          ledger: credit FLOPs
```

**What this does NOT fully solve (stated honestly):** a cheater hiding *inside* the
tolerance band. That residual is bounded by (a) raising `p`, (b) resolving
challenges via a **quorum** of independent re-executions under an honest-majority
assumption, and (c) optional TEE attestation where hardware supports it. There is
no known cheap scheme that makes wide-area FP inference cryptographically
bit-verifiable; v0.3 makes it *economically* unprofitable instead, which is the
right tool for an untrusted-but-rational swarm.

---

## 6. Reference implementation

Pure-Python + NumPy, ~900 LOC, runs on one machine simulating the whole swarm.
It is a **systems reference, not a model release** — weights are random, so output
is gibberish; the point is that the *distributed mechanics* are correct and
measurable.

| File | Layer | Responsibility |
|---|---|---|
| `model.py` | L1 | Layer-shardable transformer (`Stage`, `KVCache`); deterministic weights; proves the cache↔stateless audit invariant |
| `node.py` | L1 | A peer: hosts stages, pins KV cache locally, logs I/O for audit, exposes a tamper hook |
| `transport.py` | L2 | Simulated links; **byte + latency accounting**, including KV-bytes-avoided counterfactual |
| `router.py` | L2 | DHT-style stage discovery, latency-aware pipeline assembly, re-routing |
| `ledger.py` | L2 | Decayed contribution ratio, FLOP metering, optimistic unchoke, PoW identity, slashing |
| `verify.py` | L1.5 | Commit-reveal, random re-execution audit, calibrated tolerance, slashing |
| `engine.py` | — | Orchestrates a full inference: hops, accounting, audit, mid-stream re-route |
| `api.py` | L3 | OpenAI-compatible facade (`chat_completion` + optional FastAPI route) |
| `demo.py` | — | Stands up a 4-node swarm and exercises all of the above |

**Run it**

```bash
pip install numpy                 # core demo
python3 demo.py                   # inference, verification, fraud, ledger
pip install fastapi httpx         # optional: exercise the Layer-3 HTTP route
```

**What the demo demonstrates (measured):**

1. **Inference** through the OpenAI facade: per-hop payload `d_model×4 = 512 B`,
   constant, vs. a KV cache that would cost hundreds of KB *and grow with context*.
2. **Honest verification:** every stage re-executed with ~2e-4 synthetic
   cross-hardware drift → **0 false bans**.
3. **Fraud:** a node fabricates output → caught on audit → stake slashed → banned →
   pipeline **re-routed to a backup** and finishes.
4. **Ledger:** providers reach `priority`; a pure consumer rides its
   optimistic-unchoke grant; the cheater is `blocked`.

---

## 7. Production port (Rust / libp2p)

The Python modules map 1:1 onto the v0.2 Rust layout, so the reference is a
specification, not a throwaway:

```
src/
  inference/     <- model.py      candle / llama.cpp stage execution; the
                                  hidden-state activation is the only wire payload
  network/       <- transport.py  libp2p swarm + Kademlia over QUIC; protobuf
                   + router.py     frames: ActivationStream, AuditChallenge, FraudProof
  ledger/        <- ledger.py      decayed tit-for-tat; gossipsub reputation
  verification/  <- verify.py      commit-reveal + re-exec audit + slashing
  main.rs        <- engine/api     tokio daemon + OpenAI-compatible HTTP
```

Hot-path notes for the port: run stages on the native ML backend (candle/llama.cpp)
and keep the cross-stage activation in the model's working dtype (fp16/bf16) to
halve wire bytes again; pin the KV cache in the inference process; make the audit
log a ring buffer of commitments with on-demand reveal rather than retaining full
I/O; size the tolerance band per `(backend, dtype)` pair from an offline
calibration sweep.

---

## 8. Open problems (unchanged honesty)

* **Weight distribution & licensing.** Where 40 GB+ of weights come from, how
  they're sharded to nodes, and under what license, is out of scope here and is a
  real prerequisite.
* **Sustained-cost incentive.** Running a stage burns electricity with real
  opportunity cost; a contribution ratio is a weaker pull than money. Whether
  ratio-credit alone sustains supply is an empirical open question — the ledger is
  designed so a paid tier could sit on top without redesign.
* **Cold-cache re-routing.** When a stage is re-routed mid-session the replacement
  starts with an empty KV cache; production needs input-stream replay or periodic
  cache checkpointing to warm it. The reference impl re-routes correctly but does
  not replay.
* **In-band fraud.** See §5 — bounded economically and by quorum, not eliminated.