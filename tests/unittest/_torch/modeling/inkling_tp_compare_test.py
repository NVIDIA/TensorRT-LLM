#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic STEP B: TP=1 reference replay of STEP A's dumped TP=4 activations,
to localize the full-model garbage bug to the first divergent *transform*.

STEP A (inkling_tp_dump_test.py) dumps the TP=4 runtime's PREFILL activations for
one fixed prompt: ``embed_norm`` and per-layer hidden ``0..7`` (one file per rank).
This replays the SAME input through the VALIDATED reduced-model reference at TP=1
(the path the focused replays proved: cos>=0.9999 per layer) and reports three
independent signals so the next fault is unambiguous:

  * XRANK -- cross-rank consistency of the TP=4 dump itself: after every layer's
    all-reduce the residual stream must be identical on all 4 ranks. A layer whose
    ranks DISAGREE has a missing/partial all-reduce (the routed-expert TP reduce).

  * CUMULATIVE (Pass A) -- the reference builds its OWN trajectory from
    ``embed_norm`` and compares each layer's cumulative hidden to the TP=4 dump.
    This is what the model actually produces end to end, but a tiny early diff is
    amplified by MoE routing and compounds, so a low cumulative cos does NOT by
    itself localize the bug.

  * ISOLATED (Pass B) -- for each layer i the reference is fed the TP=4 dump's
    ACTUAL input to that layer (``dump[i-1]``, or ``embed_norm`` for i==0) and its
    output is compared to ``dump[i]``. Identical input => identical routing => this
    isolates layer i's TP transform from upstream compounding. The FIRST isolated
    layer with cos < TOL is the buggy transform to fix; if every isolated layer
    matches, the divergence is pure routing amplification of a tiny reduce-order
    diff (a numerical-stability issue, not a per-layer bug).

Layers 0..7 cover the first of every kind: 0/1 dense, 2 bf16-MoE, 3/4 NVFP4-MoE
(local), 5 NVFP4-MoE + GLOBAL attention (8 kv-heads, different TP head-sharding),
6/7 NVFP4-MoE (local). The reference cache uses the real per-layer hybrid KV
geometry (local 16 / global 8 kv-heads) so layer 5 is exercised faithfully.

Run (single GPU, same container as STEP A):
    INKLING_DUMP_PREFILL=/abs/path/prefill.pt \
    python tests/unittest/_torch/modeling/inkling_tp_compare_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

N_CMP_LAYERS = 8  # 0/1 dense, 2 bf16-MoE, 3/4 NVFP4-MoE local, 5 global, 6/7 local
TOL = 0.99
# The residual stream is identical on all ranks after each all-reduce; bf16 rounding
# of the (correct) reduce leaves at most a couple of ULPs, so anything above this is
# a genuine missing/partial reduce rather than rounding.
XRANK_EPS = 0.5


def _metrics(ref, got):
    import torch
    a = ref.flatten().float()
    b = got.flatten().float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    return cos, max_abs


def _per_token_cos(ref, got, n):
    """Per-token cosine [n] -- to see if a layer's error is concentrated in a few
    tokens (routing flip) or spread across all tokens (systematic transform bug)."""
    import torch
    a = ref.reshape(n, -1).float()
    b = got.reshape(n, -1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=1).tolist()


def _cross_rank_check(dump_base, n_layers, n):
    """Load all TP=4 rank dumps and report, per layer, the max abs diff between
    rank 0 and any other rank. After a correct all-reduce every rank holds the same
    residual stream, so a non-trivial diff localizes a missing routed-expert reduce.
    Returns the first layer whose ranks disagree (or None)."""
    import torch
    recs = []
    r = 0
    while True:
        p = f"{dump_base}.rank{r}"
        if not os.path.exists(p):
            break
        recs.append(torch.load(p, map_location="cpu"))
        r += 1
    print(f"[xrank] loaded {len(recs)} rank dumps", flush=True)
    if len(recs) < 2:
        print("[xrank] <2 ranks -- skipping cross-rank check", flush=True)
        return None
    first_bad = None
    for i in range(n_layers):
        base = recs[0]["layers"][i].float()
        mx = 0.0
        for rr in range(1, len(recs)):
            mx = max(mx, (recs[rr]["layers"][i].float() - base).abs().max().item())
        flag = "  <-- RANKS DISAGREE" if mx > XRANK_EPS else ""
        print(f"[xrank] layer{i} max_cross_rank_abs={mx:.6f}{flag}", flush=True)
        if mx > XRANK_EPS and first_bad is None:
            first_bad = i
    if first_bad is None:
        print("INKLING_TP_XRANK_ALL_CONSISTENT (all-reduce complete every layer)",
              flush=True)
    else:
        print(f"INKLING_TP_XRANK_FIRSTBAD layer={first_bad} (missing/partial "
              f"all-reduce: this layer's routed-expert output is not reduced)",
              flush=True)
    return first_bad


def main() -> int:
    import inkling_moe_replay_test as moe
    import inkling_runtime_state_test as rt
    import torch

    assert torch.cuda.is_available(), "this compare needs a CUDA device"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dump_base = os.environ["INKLING_DUMP_PREFILL"]
    rec = torch.load(f"{dump_base}.rank0", map_location="cpu")
    ids = rec["input_ids"].to(device).view(-1)
    N = ids.numel()
    pos_dump = rec.get("position_ids")
    have_layers = sorted(int(k) for k in rec["layers"].keys())
    print(f"[cmp] loaded {dump_base}.rank0: N={N} input_ids={ids.tolist()} "
          f"dumped_layers={have_layers}",
          flush=True)
    n_layers = min(N_CMP_LAYERS, len(have_layers))

    # (1) Cross-rank consistency of the TP=4 dump -- needs no model, run first.
    xrank_bad = _cross_rank_check(dump_base, n_layers, N)

    # Reduced n_layers production model on the real NVFP4 checkpoint at TP=1 -- the
    # reference the focused replays validated. Layers 0..7 span every kind incl.
    # the first global-attention layer (5), so build with the real hybrid KV
    # geometry (local 16 / global 8 kv-heads per layer).
    moe.N_LAYERS = n_layers
    model, config = moe.build_reduced_model(CKPT, device)
    tc = config.pretrained_config.text_config
    inner = model.model
    kv_list = tc.num_kv_heads_per_layer()[:n_layers]
    head_dim = tc.head_dim
    for i in range(n_layers):
        inner.layers[i].attn.attn.local_layer_idx = i
    kinds = [("dense" if tc.is_dense_layer(i) else
              ("local" if tc.is_local_layer(i) else "GLOBAL"))
             for i in range(n_layers)]
    print(f"[cmp] n_layers={n_layers} kv_heads={kv_list} kinds={kinds}", flush=True)

    if pos_dump is not None:
        pos = pos_dump.to(device).view(-1).to(torch.int32)[:N]
    else:
        pos = torch.arange(N, device=device, dtype=torch.int32)

    # Stage 0: embedding + embed_norm (TP sharding of embed_tokens).
    with torch.no_grad():
        emb = inner.embed_tokens(ids)
        ref_embed_norm = inner.embed_norm(emb)
    cos, mx = _metrics(rec["embed_norm"].to(device).view(N, -1), ref_embed_norm)
    print(f"[cmp] stage=embed_norm cos={cos:.6f} max_abs={mx:.6f}", flush=True)

    def dump_in(i):
        """The TP=4 dump's actual INPUT to layer i (dump[i-1], or embed_norm)."""
        t = rec["embed_norm"] if i == 0 else rec["layers"][i - 1]
        return t.to(device).view(N, -1).to(torch.bfloat16)

    def dump_out(i):
        return rec["layers"][i].to(device).view(N, -1)

    def dump_hattn(i):
        """The TP=4 dump's post-attention residual for layer i (or None if the
        dump predates the sub-block instrumentation)."""
        t = rec.get("h_attn", {}).get(i)
        return None if t is None else t.to(device).view(N, -1)

    def dump_moeout(i):
        """The TP=4 dump's pure MLP/MoE transform output for layer i (pre-sconv,
        pre-residual) -- or None for a pre-instrumentation dump."""
        t = rec.get("moe_out", {}).get(i)
        return None if t is None else t.to(device).view(N, -1)

    have_split = bool(rec.get("h_attn")) and bool(rec.get("moe_out"))
    print(f"[cmp] sub-block split available (h_attn+moe_out per layer): "
          f"{have_split}", flush=True)

    # Pass A -- CUMULATIVE: reference builds its own trajectory from embed_norm.
    cumulative = []
    mgrA = rt._make_ml_manager(kv_list, head_dim, [N], device)
    mdA = rt._md(mgrA, num_contexts=1, seq_lens=[N], num_cached=[0],
                 request_ids=[0], N=N)
    def _topk_experts(mlp, xin):
        """Top-k routed expert ids selected by the gate for input ``xin``."""
        rl = mlp.gate(xin)
        _, idx = torch.topk(
            (rl[..., :mlp.num_routed].sigmoid() + mlp.gate.bias),
            mlp.top_k, dim=-1)
        return idx

    try:
        with torch.no_grad():
            hidden = ref_embed_norm.to(torch.bfloat16)
            for i in range(n_layers):
                # Routing-flip probe: compare the top-k experts the gate selects on
                # the reference's cumulative input vs the TP=4 dump's input to this
                # layer. A shrinking overlap as depth grows IS the amplification --
                # a tiny cumulative hidden diff flips expert selection, and a
                # flipped expert changes that token's output entirely.
                if not tc.is_dense_layer(i):
                    mlp = inner.layers[i].mlp
                    ref_sel = _topk_experts(mlp, inner.layers[i].mlp_norm(hidden))
                    tp_sel = _topk_experts(mlp,
                                           inner.layers[i].mlp_norm(dump_in(i)))
                    same = (ref_sel.sort(-1)[0] == tp_sel.sort(-1)[0]).sum().item()
                    tot = ref_sel.numel()
                    print(f"[cmpA-routing] layer{i} topk_overlap={same}/{tot}",
                          flush=True)
                hidden = inner.layers[i](pos, hidden, mdA)
                cos, mx = _metrics(dump_out(i), hidden)
                cumulative.append((i, cos, mx))
                print(f"[cmpA-cumulative] layer{i}({kinds[i]}) cos={cos:.6f} "
                      f"max_abs={mx:.6f}", flush=True)
    finally:
        mgrA.shutdown()

    # Pass B -- ISOLATED: feed each layer the TP=4 dump's real input, compare its
    # output to the TP=4 dump. Same input => same routing => isolates the transform.
    isolated = []
    # SPLIT (decisive): a layer's isolated divergence is attention TP + MoE TP,
    # and the isolated probe only fixes the *layer* input -- so a tiny attention-TP
    # error re-routes the MoE and masquerades as "MoE divergence". Separate them:
    #   * attn sub-block: compare the reference's post-attention residual (built
    #     from the SAME dump_in(i)) to the TP=4 dump's h_attn -> pure attention TP.
    #   * moe sub-block: run the reference MoE on the TP=4 dump's OWN h_attn
    #     (identical MoE input => identical routing) and compare to the TP=4 dump's
    #     moe_out -> pure routed/shared expert TP transform, no attention seeding.
    # Whichever sub-block carries the ~0.3% seed is the transform to fix.
    split = []  # (layer, attn_cos, moe_cos)
    mgrB = rt._make_ml_manager(kv_list, head_dim, [N], device)
    mdB = rt._md(mgrB, num_contexts=1, seq_lens=[N], num_cached=[0],
                 request_ids=[0], N=N)
    mgrS = rt._make_ml_manager(kv_list, head_dim, [N], device) if have_split \
        else None
    mdS = rt._md(mgrS, num_contexts=1, seq_lens=[N], num_cached=[0],
                 request_ids=[0], N=N) if have_split else None
    if have_split:
        for i in range(n_layers):
            inner.layers[i].attn.attn.local_layer_idx = i  # (re)prime for mdS
    try:
        with torch.no_grad():
            for i in range(n_layers):
                out = inner.layers[i](pos, dump_in(i), mdB)
                cos, mx = _metrics(dump_out(i), out)
                isolated.append((i, cos, mx))
                print(f"[cmpB-isolated] layer{i}({kinds[i]}) cos={cos:.6f} "
                      f"max_abs={mx:.6f}", flush=True)
                # Always print per-token cos: concentration in a few tokens =>
                # routing/edge sensitivity; spread across all tokens => systematic
                # transform/precision difference. Decisive for the fix direction.
                pt = _per_token_cos(dump_out(i), out, N)
                print(f"[cmpB-isolated]   layer{i} per_token_cos="
                      f"{[round(c, 5) for c in pt]}", flush=True)
                # For MoE layers also decompose the reference into routed-only and
                # shared-only so their relative magnitude is visible (the seed is
                # in whichever dominates the divergence).
                if not tc.is_dense_layer(i):
                    from tensorrt_llm._torch.models.modeling_inkling import \
                        inkling_joint_renorm
                    mlp = inner.layers[i].mlp
                    xin = inner.layers[i].mlp_norm(dump_in(i))
                    rl = mlp.gate(xin)
                    routed = mlp.experts(xin, rl)
                    _, _, sg = inkling_joint_renorm(
                        rl, gate_bias=mlp.gate.bias,
                        global_scale=mlp.gate.global_scale,
                        route_scale=mlp.route_scale, top_k=mlp.top_k,
                        num_routed=mlp.num_routed, n_shared=mlp.n_shared)
                    shared = mlp.shared_experts(xin, sg)
                    print(f"[cmpB-isolated]   layer{i} routed_norm="
                          f"{routed.float().norm().item():.3f} shared_norm="
                          f"{shared.float().norm().item():.3f}", flush=True)

                # --- Attention-vs-MoE sub-block split (the decisive isolation). ---
                if have_split and dump_hattn(i) is not None \
                        and dump_moeout(i) is not None:
                    layer = inner.layers[i]
                    xin_layer = dump_in(i)
                    # Reference attention sub-block from the SAME layer input
                    # (mirrors InklingDecoderLayer stateless attention path).
                    ha = layer.attn_norm(xin_layer)
                    ha = layer.attn(pos, ha, mdS)
                    ha = layer.attn_sconv(ha)
                    h_attn_ref = xin_layer + ha
                    a_cos, a_mx = _metrics(dump_hattn(i), h_attn_ref)
                    # Reference MoE sub-block on the TP=4 dump's OWN h_attn -->
                    # identical MoE input, so routing is identical and only the
                    # routed/shared expert TP transform can differ.
                    hattn_tp4 = dump_hattn(i).to(torch.bfloat16)
                    moe_out_ref = layer.mlp(layer.mlp_norm(hattn_tp4))
                    m_cos, m_mx = _metrics(dump_moeout(i), moe_out_ref)
                    split.append((i, a_cos, m_cos))
                    worse = "ATTN" if a_cos <= m_cos else "MoE"
                    print(f"[cmpB-split]   layer{i}({kinds[i]}) "
                          f"attn_cos={a_cos:.6f} attn_max_abs={a_mx:.6f} | "
                          f"moe_cos={m_cos:.6f} moe_max_abs={m_mx:.6f} "
                          f"-> seed={worse}", flush=True)
    finally:
        mgrB.shutdown()
        if mgrS is not None:
            mgrS.shutdown()

    # Split verdict: which sub-block carries the seed. For each MoE layer we have
    # attn_cos (attention TP transform) and moe_cos (routed/shared TP transform on
    # identical input). The seed is in whichever is consistently the lower cosine
    # on the layers that actually diverge (worst isolated layers).
    if split:
        worst_iso = sorted(isolated, key=lambda r: r[1])[:3]
        worst_ids = {i for i, _, _ in worst_iso}
        focus = [(i, a, m) for i, a, m in split if i in worst_ids] or split
        attn_min = min(a for _, a, _ in focus)
        moe_min = min(m for _, _, m in focus)
        seed = "ATTENTION" if attn_min <= moe_min else "MoE"
        detail = ", ".join(f"L{i}:attn={a:.5f}/moe={m:.5f}" for i, a, m in split)
        print(f"[split-verdict] worst_isolated_layers={sorted(worst_ids)} "
              f"attn_min_cos={attn_min:.6f} moe_min_cos={moe_min:.6f}", flush=True)
        print(f"INKLING_TP_SPLIT_SEED={seed} (lower cosine on the diverging "
              f"layers is the sub-block to fix) [{detail}]", flush=True)
    else:
        print("INKLING_TP_SPLIT_SEED=UNAVAILABLE (dump predates h_attn/moe_out "
              "sub-block instrumentation; re-dump to localize)", flush=True)

    # Verdict: the FIRST isolated layer below TOL is the buggy transform.
    iso_bad = next((i for i, c, _ in isolated if c < TOL), None)
    cum_bad = next((i for i, c, _ in cumulative if c < TOL), None)
    print(f"[verdict] xrank_first_bad={xrank_bad} "
          f"cumulative_first_bad={cum_bad} isolated_first_bad={iso_bad}",
          flush=True)
    if iso_bad is not None:
        w = min(isolated, key=lambda r: r[1])
        print(f"INKLING_TP_COMPARE_ISOLATED_FIRSTBAD stage=layer{iso_bad} "
              f"kind={kinds[iso_bad]} worst=layer{w[0]}:cos={w[1]:.6f} "
              f"(this transform diverges at TP with identical input+routing)",
              flush=True)
    elif xrank_bad is not None:
        print(f"INKLING_TP_COMPARE_XRANK_ONLY layer{xrank_bad} "
              f"(ranks disagree but isolated transforms match -- missing reduce)",
              flush=True)
    else:
        print(f"INKLING_TP_COMPARE_ISOLATED_ALL_MATCH layers=0..{n_layers-1} "
              f"(every per-layer transform is TP-correct; residual cumulative "
              f"drift is routing amplification of reduce-order rounding)",
              flush=True)
    print("INKLING_TP_COMPARE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
