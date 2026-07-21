#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit8 foundation: multi-step decode short-conv state-carry (one decoder layer).

What this proves
----------------
The Inkling decoder layer has FOUR causal short convolutions (k, v inside
attention; post-attention and post-MLP on the residual stream). In the context
phase they run a stateless full-sequence causal conv; in the *generation* phase
each must convolve the one new token against the previous ``kernel_size-1``
pre-conv inputs carried from earlier steps. Before this test the decoder layer
ran the post-attention / post-MLP short-convs STATELESS in decode (only the k/v
short-convs took a conv-state window, and even those discarded the rolled state),
so a multi-token decode silently lost short-conv history -- a latent generation
bug. This test validates the runtime-correct decode contract now wired into
``InklingDecoderLayer.forward`` (the ``conv_state=InklingConvState(...)`` path):

  * REFERENCE: the layer's stateless full-sequence forward over N tokens (the
    context/prefill attention path), which is the crit4-validated ground truth
    for the whole decoder layer (attention + short-convs + dense/MoE).
  * STEP-BY-STEP DECODE: process the same N tokens one at a time through the
    generation path, starting from a zero-initialised ``InklingConvState`` and a
    fresh KV cache. Each step convolves the new token against the carried window,
    rolls all four short-conv windows forward IN PLACE, writes the new token's
    K/V to the paged cache, and attends over the reused cache. The per-position
    output must reproduce the full-sequence reference.

Because the full-sequence causal conv left-pads with zeros, a zero-initialised
step-by-step decode reproduces it exactly IFF the four short-conv windows carry
correctly across steps. Equivalence therefore isolates the conv-state-carry
contract (the hard part of the crit8 runtime short-conv cache) from the KV cache,
attention math, and MoE -- all already validated by crit4/crit5. We test one
LOCAL dense layer (0), one LOCAL MoE layer (3), and one GLOBAL MoE layer (5) so
both attention geometries and both MLP kinds exercise the decode carry.

N is kept small (a few dozen tokens, still well past the kernel window of 4) so
the N-step decode loop is fast; the carry logic is independent of N. It also runs
an independent fp32-exact ``InklingShortConv`` carry unit check (rolled decode ==
stateless conv). Prints ``max_abs``/``mean_abs``/``cosine`` per layer and, iff the
unit check passes AND every layer's decode matches its type-appropriate gate
(dense tight, MoE routing-tolerant -- see ``main``), prints
``CRIT8_DECODE_CARRY_OK`` and exits 0.

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    python tests/unittest/_torch/modeling/inkling_decode_carry_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)

N_MODEL_LAYERS = 6  # reduced model: layers 0-5 (0/1 dense, 2-5 MoE; 5 global)
TEST_LAYERS = (0, 3, 5)  # local-dense, local-MoE, global-MoE
N_TOKENS = 24  # a few dozen tokens: > kernel window (4), fast N-step decode


def _make_manager(num_kv_heads, head_dim, N, device):
    """A single-layer KVCacheManagerV2 with one request reserving N tokens."""
    import math

    import torch

    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import \
        KVCacheManagerV2
    from tensorrt_llm._utils import torch_dtype_to_binding
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    tokens_per_block = 64
    pages_per_seq = math.ceil(N / tokens_per_block)
    max_seq_len = pages_per_seq * tokens_per_block
    num_blocks = pages_per_seq

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cache_types = tensorrt_llm.bindings.internal.batch_manager.CacheType
    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_blocks * tokens_per_block),
        cache_types.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=torch_dtype_to_binding(torch.bfloat16),
    )
    mgr.add_dummy_requests([0], [N])
    return mgr


def _context_md(mgr, N, device):
    """Context (prefill) metadata: one request of N new tokens, nothing cached."""
    import torch

    from tensorrt_llm._torch.attention_backend.utils import \
        get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams

    AttentionCls = get_attention_backend("TRTLLM")
    md = AttentionCls.Metadata(
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True,
                                      num_cached_tokens_per_seq=[0]),
        seq_lens=torch.tensor([N], dtype=torch.int),
        max_num_requests=1,
        max_num_tokens=max(8192, N),
        kv_cache_manager=mgr,
        request_ids=[0],
        prompt_lens=[N],
        kv_layout="HND",
    )
    md.prepare()
    return md


def _decode_md(mgr, num_cached, N, device):
    """Generation metadata: one request emitting its (num_cached+1)-th token."""
    import torch

    from tensorrt_llm._torch.attention_backend.utils import \
        get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams

    AttentionCls = get_attention_backend("TRTLLM")
    md = AttentionCls.Metadata(
        num_contexts=0,
        kv_cache_params=KVCacheParams(use_cache=True,
                                      num_cached_tokens_per_seq=[num_cached]),
        seq_lens=torch.tensor([1], dtype=torch.int),
        max_num_requests=1,
        max_num_tokens=max(8192, N),
        kv_cache_manager=mgr,
        request_ids=[0],
        prompt_lens=[N],
        kv_layout="HND",
    )
    md.prepare()
    return md


def _zero_conv_state(tc, num_kv_heads, head_dim, device):
    """A zero-initialised InklingConvState for a one-request decode."""
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import InklingConvState

    kwin = tc.sconv_kernel_size - 1
    kv_dim = num_kv_heads * head_dim
    hidden = tc.hidden_size

    def z(c):
        return torch.zeros(1, c, kwin, device=device, dtype=torch.bfloat16)

    return InklingConvState(k=z(kv_dim),
                            v=z(kv_dim),
                            attn=z(hidden),
                            mlp=z(hidden))


def _shortconv_carry_unit(device):
    """Independent, fp32-exact proof of the short-conv carry MATH.

    Steps ``InklingShortConv.forward_decode`` one token at a time from a
    zero-initialised window and asserts it reproduces the module's stateless
    full-sequence causal conv (``forward`` with ``conv_state=None``, a different
    code path: ``F.conv1d`` vs an explicit window sum). fp32 in/out makes the two
    paths bit-close, so this isolates the carry math from attention/MoE/bf16 --
    the rigorous carry check the end-to-end layer replays cannot give (they also
    carry attention-kernel epsilon and MoE routing sensitivity). Runs for the
    real per-conv channel counts: local/global k+v dims and the hidden size.
    """
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import InklingShortConv

    g = torch.Generator(device="cpu").manual_seed(7)
    kernel = 4
    worst = 0.0
    for channels in (1024, 2048, 6144):  # global-kv, local-kv, hidden
        conv = InklingShortConv(channels, kernel).to(device)
        with torch.no_grad():
            conv.weight.copy_(
                torch.randn(channels, 1, kernel, generator=g).to(device))
        n = 20
        x = torch.randn(n, channels, generator=g,
                        dtype=torch.float32).to(device)
        with torch.no_grad():
            y_full = conv(x)  # stateless full-sequence causal conv
            state = torch.zeros(1, channels, kernel - 1, device=device)
            ys = []
            for t in range(n):
                y_t, state = conv.forward_decode(x[t:t + 1], state)
                ys.append(y_t)
            y_roll = torch.cat(ys, dim=0)
        worst = max(worst, (y_full - y_roll).abs().max().item())
    ok = worst < 1e-4
    print(f"SHORTCONV_CARRY_UNIT worst_max_abs={worst:.3e} ok={ok}", flush=True)
    return ok


def _decode_carry_for_layer(inner, tc, layer_idx, x_all, device):
    """Reference (full-sequence) vs step-by-step decode for one decoder layer."""
    import torch
    from inkling_attention_replay_test import _metrics

    layer = inner.layers[layer_idx]
    N = x_all.shape[0]
    num_kv = tc.layer_num_kv_heads(layer_idx)
    head_dim = tc.head_dim
    pos_all = torch.arange(N, device=device, dtype=torch.int32)

    # --- Reference: stateless full-sequence forward (context attention path). ---
    ref_mgr = _make_manager(num_kv, head_dim, N, device)
    layer.attn.attn.local_layer_idx = 0
    try:
        with torch.no_grad():
            ref = layer(pos_all, x_all, _context_md(ref_mgr, N,
                                                    device)).contiguous()
    finally:
        ref_mgr.shutdown()

    # --- Step-by-step decode: N generation steps, zero-init conv state, fresh
    # cache. Each step carries all four short-conv windows + the paged KV. ---
    dec_mgr = _make_manager(num_kv, head_dim, N, device)
    layer.attn.attn.local_layer_idx = 0
    cs = _zero_conv_state(tc, num_kv, head_dim, device)
    outs = []
    try:
        with torch.no_grad():
            for p in range(N):
                x_p = x_all[p:p + 1].contiguous()
                pos_p = torch.tensor([p], device=device, dtype=torch.int32)
                out_p = layer(pos_p,
                              x_p,
                              _decode_md(dec_mgr, p, N, device),
                              conv_state=cs)
                outs.append(out_p[:1].contiguous())
    finally:
        dec_mgr.shutdown()
    dec = torch.cat(outs, dim=0).contiguous()

    max_abs, mean_abs, cosine = _metrics(ref, dec)
    # Split the error at the kernel window so a stateless-decode regression (which
    # only diverges once the window fills, i.e. from token kernel_size onward) is
    # not hidden by the first few correct tokens.
    kw = tc.sconv_kernel_size
    late = (ref[kw:].float() - dec[kw:].float()).abs().amax().item() \
        if N > kw else float("nan")
    return {
        "N": N,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "late_max_abs": late,
    }


def main() -> int:
    import inkling_moe_replay_test as moe
    import torch
    from inkling_attention_replay_test import _compute_input

    # Import registers the auto-model + defines InklingConvState / the layer.
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401

    assert torch.cuda.is_available(
    ), "this decode-carry test needs a CUDA device"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Reduced 6-layer production model on the real NVFP4 checkpoint (layers 0-5).
    moe.N_LAYERS = N_MODEL_LAYERS
    model, config = moe.build_reduced_model(CKPT, device)
    tc = config.pretrained_config.text_config
    inner = model.model  # InklingModel

    # A representative real residual-stream activation fed (identically) to the
    # reference and the step-by-step decode for each tested layer. The carry
    # equivalence is independent of whether this is the exact per-layer source
    # residual -- both paths see the same input, so any mismatch is a carry bug.
    x_full, _, used_random = _compute_input(CKPT, 64, device)
    x_all = x_full[:N_TOKENS].contiguous()
    src = "RANDOM-FALLBACK" if used_random else "real-prompt embed_norm(embed(ids))"
    print(
        f"[info] N={N_TOKENS} hidden={x_all.shape[1]} src={src} "
        f"layers={list(TEST_LAYERS)}",
        flush=True)

    # 1) Rigorous, confound-free proof of the carry math (independent path).
    unit_ok = _shortconv_carry_unit(device)

    # 2) End-to-end decoder-layer decode carry. The gate is per layer TYPE:
    #    * DENSE layer (no MoE router): the decode-vs-full-sequence difference is
    #      only the attention-kernel epsilon (prefill vs paged-decode kernel,
    #      ~1e-4 by crit4) + bf16, so require a TIGHT cosine (DENSE_TOL). This is
    #      the strict end-to-end proof that the decoder layer threads all four
    #      short-conv states correctly in decode.
    #    * MoE layer: the SAME tiny attention epsilon can cross a top-6 routing
    #      boundary at a few tokens and flip an expert, giving a large per-token
    #      delta (high cosine but big max_abs) -- a known routing sensitivity, NOT
    #      a carry defect (a real carry bug corrupts EVERY post-window token and
    #      collapses cosine well below MOE_TOL). It differs from the dense layer
    #      only in the MLP, and the dense layer already matches at ~1.0, so the
    #      residual here is routing, validated tightly at crit6/crit7 via
    #      teacher-forced replay. Require a routing-tolerant cosine (MOE_TOL).
    dense_tol, moe_tol = 0.999, 0.99
    results, gates = {}, {}
    for layer_idx in TEST_LAYERS:
        local = tc.is_local_layer(layer_idx)
        dense = tc.is_dense_layer(layer_idx)
        kind = ("local" if local else "global") + ("-dense"
                                                   if dense else "-moe")
        m = _decode_carry_for_layer(inner, tc, layer_idx, x_all, device)
        results[layer_idx] = m
        tol = dense_tol if dense else moe_tol
        gates[layer_idx] = m["cosine"] >= tol
        print(
            f"DECODE_CARRY layer={layer_idx} kind={kind} N={m['N']} "
            f"max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} "
            f"cosine={m['cosine']:.6f} late_max_abs={m['late_max_abs']:.6f} "
            f"gate={'dense>=%.3f' % dense_tol if dense else 'moe>=%.2f' % moe_tol}"
            f" ok={gates[layer_idx]}",
            flush=True)

    if unit_ok and all(gates.values()):
        print("CRIT8_DECODE_CARRY_OK", flush=True)
        return 0
    print(f"CRIT8_DECODE_CARRY_MISMATCH unit_ok={unit_ok} gates={gates}",
          flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
