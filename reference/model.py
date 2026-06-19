"""
model.py — A minimal, layer-shardable decoder-only transformer.

This is the `InferenceEngine` of the spec. The only thing that matters for the
network design is that it can be *split by layer* into independent `Stage`s that
talk to each other by passing a small hidden-state activation [n_tokens, d_model]
— never the (large) KV cache, which stays resident on the node that owns those
layers.

Weights are derived deterministically from the model name, so every node in the
swarm materializes byte-identical weights for a given `model_hash`. That is what
makes re-execution audits (verify.py) meaningful. Weights are random, so
generations are gibberish — this is a systems reference, not a model release.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    name: str = "hopper-tiny"
    vocab_size: int = 256          # byte-level tokenizer, zero downloads
    d_model: int = 128
    n_layer: int = 8
    n_head: int = 4
    d_ff: int = 512
    max_seq: int = 512
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def model_hash(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()[:16]


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _rmsnorm(x: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * gain


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class LayerWeights:
    wqkv: np.ndarray   # [d, 3d]
    wo: np.ndarray     # [d, d]
    g1: np.ndarray     # [d]  (rmsnorm gain, attn)
    g2: np.ndarray     # [d]  (rmsnorm gain, mlp)
    w1: np.ndarray     # [d, d_ff]
    w2: np.ndarray     # [d_ff, d]

    def n_params(self) -> int:
        return sum(w.size for w in (self.wqkv, self.wo, self.g1, self.g2, self.w1, self.w2))


def build_weights(cfg: ModelConfig) -> dict:
    """Deterministically materialize all weights from the model name."""
    seed = int(cfg.model_hash, 16) % (2**32)
    rng = np.random.default_rng(seed)
    s = 0.02

    def rnd(*shape):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    layers = []
    for _ in range(cfg.n_layer):
        layers.append(LayerWeights(
            wqkv=rnd(cfg.d_model, 3 * cfg.d_model),
            wo=rnd(cfg.d_model, cfg.d_model),
            g1=np.ones(cfg.d_model, dtype=np.float32),
            g2=np.ones(cfg.d_model, dtype=np.float32),
            w1=rnd(cfg.d_model, cfg.d_ff),
            w2=rnd(cfg.d_ff, cfg.d_model),
        ))
    return {
        "tok_emb": rnd(cfg.vocab_size, cfg.d_model),
        "pos_emb": rnd(cfg.max_seq, cfg.d_model),
        "final_g": np.ones(cfg.d_model, dtype=np.float32),
        "layers": layers,
    }


@dataclass
class KVCache:
    """Per-(session, layer) cache. Lives on the node that owns the layer and
    NEVER travels the network. k/v: [n_head, seq, head_dim]."""
    k: np.ndarray | None = None
    v: np.ndarray | None = None

    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[1]

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        self.k = k if self.k is None else np.concatenate([self.k, k], axis=1)
        self.v = v if self.v is None else np.concatenate([self.v, v], axis=1)


class Stage:
    """A contiguous block of transformer layers, owned by one node.

    A Stage consumes and produces a small activation of shape [n_tokens, d_model].
    The first stage additionally owns the embeddings (input = token ids); the last
    stage owns the final norm + LM head (output = logits).
    """

    def __init__(self, cfg: ModelConfig, weights: dict, layer_lo: int, layer_hi: int):
        self.cfg = cfg
        self.w = weights
        self.lo, self.hi = layer_lo, layer_hi
        self.is_first = layer_lo == 0
        self.is_last = layer_hi == cfg.n_layer
        self.layers = weights["layers"][layer_lo:layer_hi]

    # ---- cost accounting -------------------------------------------------
    def n_params(self) -> int:
        p = sum(l.n_params() for l in self.layers)
        if self.is_first:
            p += self.w["tok_emb"].size + self.w["pos_emb"].size
        if self.is_last:
            p += self.w["final_g"].size
        return p

    def flops(self, n_tokens: int) -> int:
        # Standard 2*N*tokens approximation for a forward pass.
        return 2 * self.n_params() * n_tokens

    # ---- one transformer layer ------------------------------------------
    def _layer(self, lw: LayerWeights, x: np.ndarray, cache: KVCache | None,
               base_pos: int) -> np.ndarray:
        cfg = self.cfg
        h, hd, n = cfg.n_head, cfg.head_dim, x.shape[0]

        # --- attention ---
        a = _rmsnorm(x, lw.g1, cfg.eps)
        qkv = a @ lw.wqkv                                  # [n, 3d]
        q, k, v = np.split(qkv, 3, axis=-1)               # each [n, d]
        reshape = lambda t: t.reshape(n, h, hd).transpose(1, 0, 2)  # -> [h, n, hd]
        q, k, v = reshape(q), reshape(k), reshape(v)

        if cache is not None:
            past = cache.length()
            cache.append(k, v)
            k_all, v_all = cache.k, cache.v
        else:
            past, k_all, v_all = 0, k, v                  # stateless (audit) path

        seq = k_all.shape[1]
        q_pos = np.arange(base_pos, base_pos + n)
        k_pos = np.arange(seq - (past + n), seq)          # absolute key positions
        mask = k_pos[None, :] <= q_pos[:, None]           # [n, seq] causal

        scores = np.einsum("hnd,hsd->hns", q, k_all) / np.sqrt(hd)
        scores = np.where(mask[None], scores, -1e30)
        att = _softmax(scores)
        ctx = np.einsum("hns,hsd->hnd", att, v_all)       # [h, n, hd]
        ctx = ctx.transpose(1, 0, 2).reshape(n, cfg.d_model)
        x = x + ctx @ lw.wo

        # --- mlp ---
        m = _rmsnorm(x, lw.g2, cfg.eps)
        x = x + _gelu(m @ lw.w1) @ lw.w2
        return x

    # ---- stage forward ---------------------------------------------------
    def forward(self, x: np.ndarray, caches: list[KVCache] | None, base_pos: int):
        """x: token ids [n] if first stage, else hidden [n, d_model].
        caches: per-layer KVCache for online decode, or None for a stateless
        audit recompute. Returns hidden [n, d] (or logits [n, vocab] if last)."""
        if self.is_first:
            ids = x.astype(np.int64)
            n = ids.shape[0]
            pos = np.arange(base_pos, base_pos + n)
            x = self.w["tok_emb"][ids] + self.w["pos_emb"][pos]

        for i, lw in enumerate(self.layers):
            cache = None if caches is None else caches[i]
            x = self._layer(lw, x, cache, base_pos)

        if self.is_last:
            x = _rmsnorm(x, self.w["final_g"], self.cfg.eps)
            x = x @ self.w["tok_emb"].T                   # tied LM head -> logits
        return x


def shard(cfg: ModelConfig, weights: dict, n_stages: int) -> list[tuple[int, int]]:
    """Split n_layer layers into n_stages roughly-equal contiguous blocks."""
    bounds, per = [], cfg.n_layer / n_stages
    for s in range(n_stages):
        lo = round(s * per)
        hi = round((s + 1) * per)
        bounds.append((lo, hi))
    return bounds


# byte-level tokenizer ----------------------------------------------------
def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def decode(ids: list[int]) -> str:
    return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")