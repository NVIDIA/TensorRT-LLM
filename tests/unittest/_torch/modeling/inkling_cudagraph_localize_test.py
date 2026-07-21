#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2 localizer: WHERE does the assembled multi-layer DECODE diverge between the
EAGER pool path and the CUDA-GRAPH (capture-once, replay-K) pool path, per layer.

Why the isolated probes could not find B2
-----------------------------------------
iter73 (conv op) and iter74 (local/SWA attention meta.ready in-graph-KV-write)
both proved their component is graph-faithful under multi-step replay -- but each
isolated probe REBUILDS the attention metadata fresh every step and re-passes
explicit tensors, so it sidesteps the exact production failure mode: the assembled
66-layer model captures ONE decode graph whose kernels alias PERSISTENT buffers
(``_decode_meta.seq_lens``/``page_table`` per layer, the conv pool's
``state_indices``) that must be refreshed IN-PLACE every replay. Inspection
(iter75) confirmed every such buffer is in-place-refreshed and every isolated
component advances correctly, yet the full model still corrupts under cuda_graph
(served ENABLED smoke '!!!!' at batch=1; crit7 cg=on tf_mismatch~157 vs 38
baseline; crit6 pos1 decode diag 7/10). So B2 is an ASSEMBLED multi-layer decode
integration effect, not any single op -- exactly what an eager-vs-graph per-layer
localizer on the real reduced model can pin.

Method
------
Build the real reduced NVFP4 model (6 layers: 0/1 dense-local, 2-4 MoE-local, 5
MoE-global) at TP=1. Drive the SAME production decode path (``prepare_inkling_attn
_decode`` -> per-layer ``_decode_meta.ready`` in-graph-KV-write branch + the
``InklingConvRuntime`` pool) two ways over the SAME fixed input, batch=1 (the B2
batch):

  * EAGER: pool-prefill P0 tokens, then K real ``inner.forward`` decode steps.
  * GRAPH: pool-prefill P0, snapshot the conv pool, warm up + CAPTURE one decode
    forward under ``torch.cuda.CUDAGraph``, restore the pool, then REPLAY K steps
    -- refreshing the persistent decode buffers in-place before each replay so the
    KV write slot + conv slot must advance purely from the aliased buffers.

Per-layer capture is CUDA-graph-safe: forward hooks issue a pure device->device
``copy_`` of each layer's decode output into a PERSISTENT gpu buffer. During
capture that copy is recorded into the graph, so it RE-RUNS on every replay (the
hook itself does not fire on replay); after each replay we read the buffers to
host. The same hooks read the eager path. We then compare eager-vs-graph per layer
per step and report the FIRST divergent (step, layer, sub-block).

Outcomes:
  * a divergent layer  -> B2 localized to that layer/sub-block under cuda_graph.
  * graph==eager everywhere -> B2 does NOT reproduce in the reduced assembled
    model; it lives in the full-model-only integration or the production runner's
    buffer management (escalate to full-model / production instrumentation).

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    INKLING_MOE_BACKEND=TRTLLM python tests/unittest/_torch/modeling/inkling_cudagraph_localize_test.py
Env: INKLING_CHECKPOINT, INKLING_MOE_BACKEND (CUTLASS|TRTLLM), INKLING_CGLOC_{LAYERS,N,P,STEPS}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

N_LAYERS = int(os.environ.get("INKLING_CGLOC_LAYERS", "6"))  # 0/1 dense, 2-5 MoE, 5 global
N_TOKENS = int(os.environ.get("INKLING_CGLOC_N", "16"))
P_PREFILL = int(os.environ.get("INKLING_CGLOC_P", "8"))       # first decode position
K_STEPS = int(os.environ.get("INKLING_CGLOC_STEPS", "8"))     # decode steps replayed
COS_GATE = float(os.environ.get("INKLING_CGLOC_GATE", "0.9995"))


def _cos_max(a, b):
    import torch
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
    mx = float((a - b).abs().max().item())
    return cos, mx


def main() -> int:
    import inkling_moe_replay_test as moe
    import inkling_runtime_state_test as rs
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingConvRuntime, InklingConvStateCache)
    from tensorrt_llm.mapping import Mapping

    assert torch.cuda.is_available(), "cuda-graph localizer needs a CUDA GPU"
    # TP-awareness (iter100 B2 repro): under MPI (srun --mpi=pmix, world>1) build
    # THIS rank's shard of the reduced model so the SAME eager-vs-graph decode
    # localizer exercises the TP-collective decode path -- the sole remaining B2
    # suspect after iter99 (TP=1 multipage clean) and the iter100 inspection that
    # ruled out cuda-graph batch padding. The single-GPU default (no MPI /
    # world==1) is byte-identical to the prior TP=1 behavior.
    try:
        from tensorrt_llm._utils import (local_mpi_rank, mpi_barrier, mpi_rank,
                                         mpi_world_size)
        world, rank, local_rank = mpi_world_size(), mpi_rank(), local_mpi_rank()
    except Exception:  # noqa: BLE001
        world, rank, local_rank = 1, 0, 0

        def mpi_barrier():  # single-process no-op
            return None

    mapping = Mapping(world_size=world, tp_size=world,
                      rank=rank) if world > 1 else None
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(0)
    tag = f"[cgloc r{rank}/{world}]"
    print(f"{tag} tp_size={world} device=cuda:{local_rank}", flush=True)

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "(default CUTLASS)")
    moe.N_LAYERS = N_LAYERS
    model, config = moe.build_reduced_model(CKPT, device, mapping=mapping)
    tc = config.pretrained_config.text_config
    inner = model.model
    dense_mlp_idx = tc.dense_mlp_idx
    local_ids = set(tc.local_layer_ids) if hasattr(tc, "local_layer_ids") else set()
    kinds = [
        f"L{i}:{'dense' if i < dense_mlp_idx else 'moe'}/"
        f"{'local' if i in local_ids else 'global'}" for i in range(N_LAYERS)
    ]
    head_dim = tc.head_dim
    kv_list = tc.num_kv_heads_per_layer()[:N_LAYERS]
    H = tc.hidden_size
    N, P0, K = N_TOKENS, P_PREFILL, K_STEPS
    assert P0 + K <= N, f"need N({N}) >= P0({P0}) + K({K})"
    print(f"[cgloc] moe_backend={moe_backend} N={N} P0={P0} K={K} gate={COS_GATE} "
          f"layers={kinds} kv_heads={kv_list} head_dim={head_dim}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(3)
    x_embeds = torch.randn(N, H, generator=g).to(device).bfloat16()
    pos_all = torch.arange(N, device=device, dtype=torch.int32)

    # --- Persistent per-layer capture buffers (CUDA-graph safe). A forward hook
    #     copies (device->device) each decode layer's output into these; under
    #     capture the copy is recorded and re-runs on every replay. ---
    buf_layer = [torch.zeros(H, device=device, dtype=torch.float32)
                 for _ in range(N_LAYERS)]
    buf_hattn = [torch.zeros(H, device=device, dtype=torch.float32)
                 for _ in range(N_LAYERS)]
    buf_moe = [torch.zeros(H, device=device, dtype=torch.float32)
               for _ in range(N_LAYERS)]
    handles = []

    def _install_hooks():
        for i, layer in enumerate(inner.layers):
            def pre(_m, args, _i=i):
                t = args[0]
                if t.shape[0] == 1:  # decode token only (skip P0-token prefill)
                    buf_hattn[_i].copy_(t.detach().reshape(-1).float())
                return None

            def moe_hook(_m, _in, out, _i=i):
                o = out[0] if isinstance(out, tuple) else out
                if o.shape[0] == 1:
                    buf_moe[_i].copy_(o.detach().reshape(-1).float())

            def layer_hook(_m, _in, out, _i=i):
                o = out[0] if isinstance(out, tuple) else out
                if o.shape[0] == 1:
                    buf_layer[_i].copy_(o.detach().reshape(-1).float())

            handles.append(layer.mlp_norm.register_forward_pre_hook(pre))
            handles.append(layer.mlp.register_forward_hook(moe_hook))
            handles.append(layer.register_forward_hook(layer_hook))

    _install_hooks()

    def _read_bufs():
        torch.cuda.synchronize()
        return {
            "layer": [buf_layer[i].cpu().clone() for i in range(N_LAYERS)],
            "hattn": [buf_hattn[i].cpu().clone() for i in range(N_LAYERS)],
            "moe": [buf_moe[i].cpu().clone() for i in range(N_LAYERS)],
        }

    def _pool_prefill(dec_cache, dec_mgr):
        md_p = rs._md(dec_mgr, num_contexts=1, seq_lens=[P0], num_cached=[0],
                      request_ids=[0], N=N)
        model.prepare_inkling_attn_decode(md_p)  # no-op for context (num_gen<=0)
        rt_p = InklingConvRuntime.build(md_p, dec_cache)
        inner.forward(md_p, inputs_embeds=x_embeds[:P0],
                      position_ids=pos_all[:P0], conv_cache=dec_cache,
                      conv_rt=rt_p)

    def _prep_step(dec_mgr, dec_cache, pos):
        """Refresh every per-layer decode buffer + conv slot IN-PLACE for `pos`."""
        md_d = rs._md(dec_mgr, num_contexts=0, seq_lens=[1], num_cached=[pos],
                      request_ids=[0], N=N)
        model.prepare_inkling_attn_decode(md_d)          # _decode_meta.* in place
        rt_d = InklingConvRuntime.build(md_d, dec_cache)  # state_indices in place
        return md_d, rt_d

    def _snap_pool(dec_cache):
        return [[t.clone() for t in dec_cache._layers[i]] for i in range(N_LAYERS)]

    def _restore_pool(dec_cache, snap):
        for i in range(N_LAYERS):
            for t, s in zip(dec_cache._layers[i], snap[i]):
                t.copy_(s)

    # ================= EAGER reference (production pool decode) ==================
    eager = []
    dec_cache_e = InklingConvStateCache(config, max_batch_size=2, device=device)
    dec_mgr_e = rs._make_ml_manager(kv_list, head_dim, [N], device,
                                    mapping=mapping)
    rs._set_layer_offsets(inner)
    try:
        with torch.no_grad():
            _pool_prefill(dec_cache_e, dec_mgr_e)
            for j in range(K):
                pos = P0 + j
                md_d, rt_d = _prep_step(dec_mgr_e, dec_cache_e, pos)
                inner.forward(md_d, inputs_embeds=x_embeds[pos:pos + 1],
                              position_ids=pos_all[pos:pos + 1],
                              conv_cache=dec_cache_e, conv_rt=rt_d)
                eager.append(_read_bufs())
    finally:
        dec_mgr_e.shutdown()

    # ============ GRAPH: capture one decode forward, replay K steps =============
    graph_steps = []
    dec_cache_g = InklingConvStateCache(config, max_batch_size=2, device=device)
    dec_mgr_g = rs._make_ml_manager(kv_list, head_dim, [N], device,
                                    mapping=mapping)
    rs._set_layer_offsets(inner)
    try:
        with torch.no_grad():
            _pool_prefill(dec_cache_g, dec_mgr_g)
            snap = _snap_pool(dec_cache_g)             # conv pool state at P0
            x_buf = x_embeds[P0:P0 + 1].contiguous().clone()
            pos_buf = pos_all[P0:P0 + 1].contiguous().clone()

            def _run(md, rt):
                return inner.forward(md, inputs_embeds=x_buf,
                                     position_ids=pos_buf,
                                     conv_cache=dec_cache_g, conv_rt=rt)

            # Warm up on a side stream (mutates the pool; restored after).
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    m, r = _prep_step(dec_mgr_g, dec_cache_g, P0)
                    _run(m, r)
            torch.cuda.current_stream().wait_stream(side)
            _restore_pool(dec_cache_g, snap)           # pool back to P0

            # Capture one decode forward (its execution mutates pool P0->P0+1).
            m_cap, r_cap = _prep_step(dec_mgr_g, dec_cache_g, P0)
            mpi_barrier()  # TP: all ranks enter capture together (NCCL lockstep)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g_out = _run(m_cap, r_cap)
            _restore_pool(dec_cache_g, snap)           # undo the capture execution
            mpi_barrier()  # TP: all ranks finished capture before replay

            # Replay K steps: in-place refresh advances the KV write slot + conv
            # slot from P0, P0+1, ... purely through the aliased buffers.
            for j in range(K):
                pos = P0 + j
                x_buf.copy_(x_embeds[pos:pos + 1])
                pos_buf.copy_(pos_all[pos:pos + 1])
                _prep_step(dec_mgr_g, dec_cache_g, pos)
                graph.replay()
                rec = _read_bufs()
                rec["out"] = g_out.detach().reshape(-1).float().cpu().clone()
                graph_steps.append(rec)
    finally:
        dec_mgr_g.shutdown()
        for h in handles:
            h.remove()

    # ============================== compare ====================================
    print(f"\n[cgloc] per-layer EAGER-vs-GRAPH decode (step j -> position {P0}+j):",
          flush=True)
    first_div = None
    worst = (1.0, -1, -1)
    n_bad_layers = set()
    for j in range(K):
        print(f"  --- step {j} (pos {P0 + j}) ---", flush=True)
        for i in range(N_LAYERS):
            hc, hm = _cos_max(eager[j]["hattn"][i], graph_steps[j]["hattn"][i])
            mc, mm = _cos_max(eager[j]["moe"][i], graph_steps[j]["moe"][i])
            oc, om = _cos_max(eager[j]["layer"][i], graph_steps[j]["layer"][i])
            flag = ""
            if oc < COS_GATE:
                n_bad_layers.add(i)
                if first_div is None:
                    block = ("attn" if hc < COS_GATE
                             else ("moe/mlp" if mc < COS_GATE else "post/residual"))
                    first_div = (j, i, kinds[i], block)
                    flag = "  <== FIRST GRAPH DIVERGENCE"
            if oc < worst[0]:
                worst = (oc, j, i)
            print(f"    {kinds[i]:16s} h_attn(cos={hc:.6f} max={hm:.4f}) "
                  f"moe(cos={mc:.6f} max={mm:.4f}) "
                  f"layer(cos={oc:.6f} max={om:.4f}){flag}", flush=True)

    step0 = [_cos_max(eager[0]["layer"][i], graph_steps[0]["layer"][i])[0]
             for i in range(N_LAYERS)]
    reproduced = first_div is not None
    print(f"\n[cgloc] FIRST_GRAPH_DIVERGENCE={first_div} "
          f"worst(cos={worst[0]:.6f} step={worst[1]} layer={worst[2]}) "
          f"diverged_layers={sorted(n_bad_layers)}", flush=True)
    print("INKLING_CGLOC_DONE "
          f"rank={rank} tp={world} "
          f"moe_backend={moe_backend} reproduced_B2={reproduced} "
          f"first_div={first_div} worst_cos={worst[0]:.5f} "
          f"diverged_layers={sorted(n_bad_layers)} "
          f"step0_layer_cos=[" + ",".join(f"{c:.5f}" for c in step0) + "]",
          flush=True)
    if reproduced:
        print(f"INKLING_CGLOC_LOCALIZED B2 reproduced in reduced assembled model "
              f"at step={first_div[0]} layer={first_div[1]}({first_div[2]}) "
              f"block={first_div[3]}", flush=True)
    else:
        print("INKLING_CGLOC_NOREPRO reduced-model graph==eager everywhere; B2 is "
              "full-model-only or production-runner buffer management", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
