"""
demo.py — Stand up a small swarm and exercise every layer.
Run: python3 demo.py
"""
from __future__ import annotations

import numpy as np

from api import chat_completion
from engine import Engine
from ledger import Ledger
from model import ModelConfig, build_weights, shard, Stage
from node import Node
from router import Router
from transport import LinkProfile, NetworkMonitor
from verify import Verifier


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def kv_ship_cost(cfg, context_len):
    """What the ORIGINAL spec would pay to ship the whole-model KV cache once."""
    per_token = 2 * cfg.n_layer * cfg.n_head * cfg.head_dim * 4  # k+v, fp32
    return per_token * context_len


def main():
    cfg = ModelConfig()
    weights = build_weights(cfg)
    N = 4
    bounds = shard(cfg, weights, N)
    stages = {i: Stage(cfg, weights, lo, hi) for i, (lo, hi) in enumerate(bounds)}

    rule("SWARM TOPOLOGY")
    print(f"model={cfg.name}  hash={cfg.model_hash}  "
          f"layers={cfg.n_layer}  d_model={cfg.d_model}")
    for i, (lo, hi) in enumerate(bounds):
        s = stages[i]
        tag = ("first " if s.is_first else "") + ("last" if s.is_last else "")
        print(f"  stage {i}: layers [{lo},{hi})  params={s.n_params():>8,}  {tag}")

    ledger = Ledger()
    router = Router(ledger)
    nodes = {}
    rtts = [9.0, 14.0, 11.0, 17.0]

    def add_node(nid, stage_id, rtt, cls=Node):
        ledger.register(nid, difficulty=8)
        n = cls(nid, LinkProfile(rtt_ms=rtt, up_mbps=30.0))
        n.host(stage_id, stages[stage_id])
        nodes[nid] = n
        router.announce(nid, stage_id, rtt)
        return n

    for i in range(N):
        add_node(f"node-{i}", i, rtts[i])
    # redundant providers so re-routing always has somewhere to land
    add_node("node-1b", 1, 22.0)
    add_node("node-2b", 2, 25.0)
    ledger.register("client", difficulty=8)

    mon = NetworkMonitor()
    quiet = Verifier(stages, ledger, audit_prob=0.0, rng=np.random.default_rng(1))
    engine = Engine(cfg, nodes, router, ledger, quiet, monitor=mon,
                    n_stages=N, rng=np.random.default_rng(7))

    # 1) normal inference through the OpenAI-compatible facade ----------------
    rule("1) INFERENCE  (OpenAI-compatible facade -> layer-sharded pipeline)")
    resp = chat_completion(engine, cfg.name,
                           [{"role": "user", "content": "explain the design"}],
                           max_tokens=24, client_id="client")
    net = resp["hopper"]["network"]
    ctx_len = len("user: explain the design") + 24
    print(f"completion : {resp['choices'][0]['message']['content']!r}")
    print(f"tokens={resp['usage']['completion_tokens']}  hops={net['hops']}")
    print(f"per-hop activation : {cfg.d_model * 4} bytes "
          f"(= d_model x 4, INDEPENDENT of context length)")
    print(f"total activation shipped : {net['activation_MB'] * 1e3:.1f} KB")
    print(f"KV cache if shipped (orig spec, once @ ctx={ctx_len}) : "
          f"{kv_ship_cost(cfg, ctx_len) / 1e3:.1f} KB  and it grows with context")
    print("  -> HOPPER keeps the KV cache RESIDENT; WAN payload is constant per token.")

    # 2) honest cross-hardware node passes within tolerance -------------------
    rule("2) VERIFICATION  (honest node, ~2e-4 cross-hardware fp drift)")
    v_drift = Verifier(stages, ledger, audit_prob=1.0, tolerance=2e-3,
                       hw_noise=2e-4, rng=np.random.default_rng(3))
    eng2 = Engine(cfg, nodes, router, ledger, v_drift, NetworkMonitor(),
                  n_stages=N, rng=np.random.default_rng(7))
    _, _, st = eng2.generate("audit me", max_tokens=6, client_id="client")
    print(f"audits={st.audits}  failed={st.audit_fails}  reroutes={st.reroutes}")
    print("  -> every stage re-executed; honest drift stays inside the band, zero false bans.")

    # 3) a cheating node is caught, slashed, and routed around ----------------
    rule("3) FRAUD  (node fabricates output -> caught, slashed, re-routed)")

    class CheatingNode(Node):
        def _tamper(self, out):
            return out + 5.0          # fabricate work; commitment will bind the lie

    cheater = CheatingNode("node-1", LinkProfile(rtt_ms=14.0))
    cheater.host(1, stages[1])
    nodes["node-1"] = cheater         # stage-1 primary is now dishonest

    v_strict = Verifier(stages, ledger, audit_prob=1.0, tolerance=2e-3,
                        rng=np.random.default_rng(5))
    eng3 = Engine(cfg, nodes, router, ledger, v_strict, NetworkMonitor(),
                  n_stages=N, rng=np.random.default_rng(7))
    _, _, st3 = eng3.generate("catch the cheater", max_tokens=4, client_id="client")
    print(f"audit_fails={st3.audit_fails}  reroutes={st3.reroutes}")
    print(f"node-1 ledger -> {ledger.snapshot()['node-1']}")
    print("  -> stake slashed, banned, dropped from routing; pipeline finished on the backup.")

    # 4) ledger state ---------------------------------------------------------
    rule("4) LEDGER  (decayed contribution ratio + access tiers)")
    for nid in ["node-0", "node-1", "node-1b", "node-2", "node-3", "client"]:
        snap = ledger.snapshot()[nid]
        print(f"  {nid:9s}  ratio={str(snap['ratio']):>5}  "
              f"tier={snap['tier']:<18}  banned={snap['banned']}")
    print("\n  providers seed >> leech -> ratio=inf -> priority; pure-consumer client")
    print("  rides its 30-min optimistic-unchoke grant; node-1 is blocked.")


if __name__ == "__main__":
    main()