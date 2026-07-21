#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2 probe #2: the RUNTIME meta.ready in-graph-KV-write attention decode under CUDA graph.

iter73 exonerated the short-conv op as B2 (inkling_conv_graph_test.py). B2 (the
cuda-graph enabled-config decode corruption that hits even batch=1) therefore lives
in the attention decode. crit4 (inkling_attention_replay_test) validates attention
decode under CUDA graph ONLY through the ``skip_kv_write`` static-tensor path (KV
pre-written by an eager host loop; graph reads static seq_lens/page_table). The REAL
runtime decode path is the meta.ready branch (modeling_inkling.py _run_generation:
1398-1414): it derives the KV write slot ON-GPU from the eagerly-refreshed
InklingDecodeMeta.seq_lens buffer (``pos = seq_lens - 1``) and does an IN-GRAPH
scatter write ``k_cache[pages,:,offs,:] = k``, then the paged decode kernel. That
path is UNTESTED under CUDA graph -- neither eager nor captured. The decisive B2
question: does the in-graph write slot ADVANCE across graph replays (re-derived from
the refreshed buffer every replay) or bake the capture-time position, so every replay
writes the SAME slot and the KV cache corrupts as decode proceeds?

Focus: the LOCAL (SWA / sliding-window) layer -- the human's 'sconv/SWA' hint, and the
layer whose window logic is most graph-sensitive.

  PART 1: meta.ready EAGER decode == crit4 host-write eager decode  (path correct eagerly?)
  PART 2: meta.ready GRAPH (capture-once, replay once) == meta.ready EAGER  (single-step graph ok?)
  PART 3: meta.ready GRAPH multi-step (capture-once, replay K, KV accumulates) == eager multi-step
          (does the in-graph KV write slot ADVANCE across replays? -- the B2 gap)

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    python tests/unittest/_torch/modeling/inkling_attn_graph_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _cos_max(a, b):
    import torch
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
    mx = float((a - b).abs().max().item())
    return cos, mx


def main() -> int:
    import copy

    import torch

    import inkling_attention_replay_test as a4
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401 (registers auto-model)
    from tensorrt_llm.mapping import Mapping

    assert torch.cuda.is_available(), "attn-graph probe needs a CUDA GPU"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    CKPT = a4.CKPT
    K = int(os.environ.get("INKLING_AG_STEPS", "8"))  # multi-step decode count

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    model_config = ModelConfig.from_pretrained(CKPT, trust_remote_code=True,
                                               mapping=mapping,
                                               attn_backend="TRTLLM",
                                               moe_backend="CUTLASS")
    text_config = model_config.pretrained_config.text_config
    tmc = copy.copy(model_config)
    tmc.pretrained_config = text_config

    # Layer under test: local (SWA) by default, global (full attention) via env.
    layer_idx = (a4.LAYER_GLOBAL
                 if os.environ.get("INKLING_AG_LAYER") == "global"
                 else a4.LAYER_LOCAL)
    x, N, _ = a4._compute_input(CKPT, a4.N_TARGET, device)
    nkv = text_config.layer_num_kv_heads(layer_idx)
    hd = text_config.layer_head_dim(layer_idx)
    ksize = text_config.sconv_kernel_size
    is_local = text_config.is_local_layer(layer_idx)
    print(f"[ag] layer={layer_idx} kind={'local' if is_local else 'global'} N={N} "
          f"nkv={nkv} head_dim={hd} ksize={ksize} sliding_window="
          f"{text_config.sliding_window_size} steps={K}", flush=True)

    from tensorrt_llm._torch.attention_backend.utils import \
        get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    AttentionCls = get_attention_backend("TRTLLM")

    def build_decode_md(mgr, num_cached, max_num_tokens):
        md = AttentionCls.Metadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[num_cached]),
            seq_lens=torch.tensor([1], dtype=torch.int),
            max_num_requests=1, max_num_tokens=max_num_tokens,
            kv_cache_manager=mgr, request_ids=[0], prompt_lens=[N],
            kv_layout="HND")
        md.prepare()
        return md

    attn_w = a4._read_attn_weights(CKPT, layer_idx, device)

    # ---- helper: build a fresh attention+cache, prefill P tokens --------------
    def fresh(P):
        attn = a4._build_trtllm_attention(tmc, layer_idx, attn_w, device)
        # 1-layer cache lives at index 0; pin BOTH the backend op layer and the
        # decode-meta layer so get_batch_cache_indices(.,layer_idx) hits it.
        attn.attn.local_layer_idx = 0
        attn._decode_meta.layer_idx = 0
        mgr, prefill_md, _ = a4._build_cache_and_metadatas(nkv, hd, N, P, device)
        with torch.no_grad():
            pos_prefill = torch.arange(P, device=device, dtype=torch.int32)
            attn.forward(position_ids=pos_prefill,
                         hidden_states=x[:P].contiguous(),
                         attn_metadata=prefill_md)
        return attn, mgr

    max_num_tokens = max(8192, N)

    def meta_ready_decode(attn, mgr, pos):
        """One meta.ready decode step at absolute position ``pos`` (num_cached=pos).
        Refreshes the stable decode buffers, then runs the forward on the meta.ready
        in-graph-KV-write branch (decode_seq_lens/page_table = None)."""
        md = build_decode_md(mgr, pos, max_num_tokens)
        ck, cv = a4._compute_conv_states(x, attn_w, nkv, hd, ksize, pos)
        posv = torch.tensor([pos], device=device, dtype=torch.int32)
        xd = x[pos:pos + 1].contiguous()
        ok = attn._decode_meta.refresh(md, device)
        assert ok and attn._decode_meta.ready, "meta.ready not set by refresh()"
        out = attn.forward(position_ids=posv, hidden_states=xd, attn_metadata=md,
                           conv_states=(ck, cv), decode_seq_lens=None,
                           decode_page_table=None)[:1].contiguous()
        return out, md, (ck, cv, posv, xd)

    P = N - 1

    # ---- PART 1: meta.ready EAGER == crit4 host-write EAGER (path correct eagerly?)
    attn, mgr = fresh(P)
    try:
        with torch.no_grad():
            md0 = build_decode_md(mgr, P, max_num_tokens)
            ck0, cv0 = a4._compute_conv_states(x, attn_w, nkv, hd, ksize, P)
            posv0 = torch.tensor([P], device=device, dtype=torch.int32)
            xd0 = x[P:P + 1].contiguous()
            # crit4 host-write eager path (meta NOT refreshed): writes KV[P] via host loop.
            ref_dec = attn.forward(position_ids=posv0, hidden_states=xd0,
                                   attn_metadata=md0,
                                   conv_states=(ck0, cv0))[:1].contiguous()
            # meta.ready eager path (refresh -> in-graph scatter write, eager).
            mr_eager, _, _ = meta_ready_decode(attn, mgr, P)
    finally:
        mgr.shutdown()
    c1, m1 = _cos_max(ref_dec, mr_eager)
    p1_ok = c1 > 0.9995
    print(f"  [PART 1] meta.ready EAGER vs host-write EAGER cos={c1:.6f} max={m1:.4f} "
          f"{'PASS' if p1_ok else 'FAIL'}", flush=True)

    # ---- PART 2: meta.ready GRAPH (single-step) == meta.ready EAGER --------------
    attn, mgr = fresh(P)
    try:
        with torch.no_grad():
            mr_eager2, _, _ = meta_ready_decode(attn, mgr, P)
            # Capture the meta.ready forward and replay ONCE. Static buffers.
            md = build_decode_md(mgr, P, max_num_tokens)
            ck, cv = a4._compute_conv_states(x, attn_w, nkv, hd, ksize, P)
            ck_b, cv_b = ck.clone(), cv.clone()
            pos_b = torch.tensor([P], device=device, dtype=torch.int32)
            x_b = x[P:P + 1].contiguous().clone()
            attn._decode_meta.refresh(md, device)

            def run():
                return attn.forward(position_ids=pos_b, hidden_states=x_b,
                                    attn_metadata=md, conv_states=(ck_b, cv_b),
                                    decode_seq_lens=None, decode_page_table=None)

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g_out = run()
            graph.replay()
            torch.cuda.synchronize()
            mr_graph = g_out[:1].contiguous().clone()
    finally:
        mgr.shutdown()
    c2, m2 = _cos_max(mr_eager2, mr_graph)
    p2_ok = c2 > 0.9995
    print(f"  [PART 2] meta.ready GRAPH(1step) vs EAGER cos={c2:.6f} max={m2:.4f} "
          f"{'PASS' if p2_ok else 'FAIL'}", flush=True)

    # ---- PART 3: meta.ready GRAPH multi-step == eager multi-step (the B2 gap) ----
    # EAGER reference: decode positions P0..P0+K-1, KV accumulates each step.
    P0 = N - K
    attn_e, mgr_e = fresh(P0)
    eager_steps = []
    try:
        with torch.no_grad():
            for j in range(K):
                out, _, _ = meta_ready_decode(attn_e, mgr_e, P0 + j)
                eager_steps.append(out.float().cpu().clone())
    finally:
        mgr_e.shutdown()

    # GRAPH: capture ONCE at P0, then replay K times, refreshing the stable decode
    # buffers + updating the static inputs each step so the KV write slot must
    # advance (P0, P0+1, ...) purely from the refreshed seq_lens buffer.
    attn_g, mgr_g = fresh(P0)
    graph_steps = []
    try:
        with torch.no_grad():
            md_g = build_decode_md(mgr_g, P0, max_num_tokens)
            ckg, cvg = a4._compute_conv_states(x, attn_w, nkv, hd, ksize, P0)
            ck_g, cv_g = ckg.clone(), cvg.clone()
            pos_g = torch.tensor([P0], device=device, dtype=torch.int32)
            x_g = x[P0:P0 + 1].contiguous().clone()

            def run_g():
                return attn_g.forward(position_ids=pos_g, hidden_states=x_g,
                                      attn_metadata=md_g, conv_states=(ck_g, cv_g),
                                      decode_seq_lens=None,
                                      decode_page_table=None)

            attn_g._decode_meta.refresh(md_g, device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    run_g()
            torch.cuda.current_stream().wait_stream(side)
            graph_g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_g):
                g_out_ms = run_g()
            # Replay K steps: refresh advances the KV write slot via seq_lens.
            for j in range(K):
                pos = P0 + j
                mdj = build_decode_md(mgr_g, pos, max_num_tokens)
                ckj, cvj = a4._compute_conv_states(x, attn_w, nkv, hd, ksize, pos)
                ck_g.copy_(ckj)
                cv_g.copy_(cvj)
                pos_g.copy_(torch.tensor([pos], device=device, dtype=torch.int32))
                x_g.copy_(x[pos:pos + 1].contiguous())
                attn_g._decode_meta.refresh(mdj, device)
                graph_g.replay()
                torch.cuda.synchronize()
                graph_steps.append(g_out_ms[:1].contiguous().float().cpu().clone())
    finally:
        mgr_g.shutdown()

    p3_ok = True
    worst = (1.0, 0.0, -1)
    for j in range(K):
        cj, mj = _cos_max(eager_steps[j], graph_steps[j])
        step_ok = cj > 0.9995
        p3_ok &= step_ok
        if cj < worst[0]:
            worst = (cj, mj, j)
        if not step_ok or j < 2 or j == K - 1:
            print(f"    [PART 3 step {j:2d} pos={P0 + j}] cos={cj:.6f} max={mj:.4f} "
                  f"{'ok' if step_ok else '<== GRAPH DIVERGES'}", flush=True)
    print(f"  [PART 3] meta.ready GRAPH multi-step vs EAGER "
          f"{'PASS' if p3_ok else 'FAIL'} worst_step={worst[2]}(cos={worst[0]:.6f} "
          f"max={worst[1]:.4f})", flush=True)

    ok = p1_ok and p2_ok and p3_ok
    print(f"INKLING_ATTN_GRAPH_{'OK' if ok else 'FAIL'} "
          f"part1_metaready_eager={'ok' if p1_ok else 'FAIL'} "
          f"part2_graph_1step={'ok' if p2_ok else 'FAIL'} "
          f"part3_graph_multistep={'ok' if p3_ok else 'FAIL'} "
          f"layer={layer_idx}({'local' if is_local else 'global'}) steps={K}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
