# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM 10.7) for kernels/m3_drafter_attn: the Eagle3 drafter layer's generation pass on the full-context
PrimTS decode, driven through ``M3DrafterFullCtxFmha.run_generation`` (in-tree preprocess with the step's FP8 K/V
append, the K / V sub-page views of the drafter cache, the pinned plan) at every recipe the target pins: the 4-row
first draft pass and the 1-row later passes of 1-2 requests.

Lengths sweep the KV tile edges each query row in turn ends on (the rows of a Q CTA with no key in its trailing
splits), with mixed per-request lengths; every call is held to the fp32 reference over the cache after the append
(E4M3 query, unit KV scales, no RoPE) within the FP8-KV band, finite, for both split-KV reducers.
"""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_drafter_attn as drafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_fullctx_decode as fullctx,
)

assert torch.cuda.is_available(), "m3_drafter_attn requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the tp8 target is certified on SM 10.7 only", allow_module_level=True)

H, KVH, D, PAGE = fullctx.HEADS, fullctx.KV_HEADS, fullctx.HEAD_DIM, 128
MAX_LEN = 1 << 20
FP8_KV_QUANT_MODE = 128
FP8_TOL = 6e-2
# Persistent like the FMHA manager's workspace and the spec-decoding metadata of a real step: the in-tree preprocess
# kernels must not see these move between calls.
_PERSISTENT = {}


def _persistent(name: str, make):
    if name not in _PERSISTENT:
        _PERSISTENT[name] = make()
    return _PERSISTENT[name]


def _edge_lens(plan, qlen: int):
    tile, insts, splits = plan.tile_kv, plan.insts_kv, int(plan.recipe["splits"])
    grow = splits * insts
    tiles = sorted({1, 2, 8, 9, grow - 1, grow, grow + 1, grow + 2 * insts, 2 * grow})
    return [k * tile + d for k in tiles for d in range(-1, qlen)]


def _library(qlen: int, nreq: int, own_reducer: bool):
    lib = drafter.M3DrafterFullCtxFmha.__new__(drafter.M3DrafterFullCtxFmha)
    lib._recipes = {(qlen, nreq): modeling.FULLCTX_DECODE_RECIPES[(qlen, nreq)]}
    lib._own_reducer = own_reducer
    lib._max_kv_len = MAX_LEN
    lib._multi_processor_count = torch.cuda.get_device_properties(0).multi_processor_count
    lib._on_taken = lambda rows: None
    # The in-tree preprocess pass (its RoPE-free cache append); the one-launch preprocess has its own bitwise test.
    lib._fused_rope_mode = None
    lib._pdl = False
    lib._on_rope_taken = None
    return lib


def _attn(qlen: int):
    return SimpleNamespace(
        num_heads=H,
        num_kv_heads=KVH,
        head_dim=D,
        q_scaling=1.0,
        attention_chunk_size=None,
        quant_mode=FP8_KV_QUANT_MODE,
        local_layer_idx=0,
        position_embedding_type=0,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        predicted_tokens_per_seq=qlen,
        rope_params=SimpleNamespace(
            dim=0, theta=10000.0, scale_type=0, scale=1.0, max_positions=1024
        ),
    )


def _run(lib, qlen: int, kv_lens, seed: int, spec: bool = True):
    """One drafter generation pass over a fresh FP8 cache; returns (output, fp32 reference). ``spec`` passes the
    spec-decoding lengths/positions (a multi-token generation step); without them a multi-row pass is the drafter's
    plain generation step of input_seq_length rows, as it runs in-model."""
    dev = torch.device("cuda")
    nreq = len(kv_lens)
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = math.ceil(max(kv_lens) / PAGE)
    nblocks = nreq * pages + 2
    pool = torch.randn(nblocks, 2, KVH, PAGE, D, device=dev, generator=g).clamp_(-4, 4)
    perm = torch.randperm(nblocks, device=dev, generator=g)[: nreq * pages].view(nreq, pages)
    for r, n in enumerate(
        kv_lens
    ):  # shift the tail third of V, so a missed trailing split fails the band
        start = 2 * n // 3
        for pg in range(start // PAGE, math.ceil(n / PAGE)):
            pool[int(perm[r, pg]), 1, :, (start % PAGE if pg == start // PAGE else 0) :] += 1.0
    pool = pool.to(torch.float8_e4m3fn)
    offsets = torch.zeros(1, nreq, 2, pages, dtype=torch.int32, device=dev)
    offsets[0, :, 0] = 2 * perm
    offsets[0, :, 1] = 2 * perm + 1
    pointers = torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64)
    ntok = nreq * qlen
    qkv = torch.randn(ntok, (H + 2 * KVH) * D, dtype=torch.bfloat16, device=dev, generator=g)
    seq = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    multi = qlen > 1 and spec
    out = torch.zeros(ntok, H * D, dtype=torch.bfloat16, device=dev)
    params = SimpleNamespace(
        qkv_input=qkv,
        query_input=None,
        attn=_attn(qlen),
        meta=SimpleNamespace(
            kv_cache_block_offsets=offsets,
            host_kv_cache_pool_pointers=pointers,
            host_kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        ),
        fwd=SimpleNamespace(
            kv_scale_orig_quant=one,
            kv_scale_quant_orig=one,
            out_scale=None,
            mrope_position_deltas=None,
        ),
        workspace=_persistent(
            "workspace", lambda: torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        ),
        sequence_lengths=seq,
        spec_decoding_generation_lengths=(
            _persistent(
                f"gen_lengths{qlen}x{nreq}",
                lambda: torch.full((nreq,), qlen, dtype=torch.int32, device=dev),
            )
            if multi
            else None
        ),
        spec_decoding_position_offsets=(
            _persistent(
                f"position_offsets{qlen}x{nreq}",
                lambda: (
                    torch.arange(qlen, dtype=torch.int32, device=dev).repeat(nreq, 1).contiguous()
                ),
            )
            if multi
            else None
        ),
        seq_offset=0,
        tokens_per_block=PAGE,
        max_attention_window_size=MAX_LEN,
        cyclic_attention_window_size=MAX_LEN,
        num_tokens=ntok,
        batch_size=nreq,
        input_seq_length=qlen,
        max_past_kv_length=max(kv_lens),
        total_num_blocks=nblocks,
        kv_factor=2,
        is_cross=False,
        output=out,
    )
    lib.run_generation(params)
    torch.cuda.synchronize()
    # fp32 reference over the cache after the append: E4M3 query, unit scales, causal over the step's rows.
    q = qkv[:, : H * D].float().view(nreq, qlen, H, D).to(torch.float8_e4m3fn).float()
    flat = pool.float()
    ref = torch.empty(nreq, qlen, H, D, device=dev)
    for r, n in enumerate(kv_lens):
        blocks = perm[r, : math.ceil(n / PAGE)]
        k = flat[blocks, 0].permute(1, 0, 2, 3).reshape(KVH, -1, D)[:, :n]
        v = flat[blocks, 1].permute(1, 0, 2, 3).reshape(KVH, -1, D)[:, :n]
        k = k.repeat_interleave(H // KVH, dim=0)
        v = v.repeat_interleave(H // KVH, dim=0)
        s = torch.einsum("qhd,hkd->hqk", q[r], k) * D**-0.5
        visible = (
            torch.arange(n, device=dev)[None, :]
            < (n - qlen + 1 + torch.arange(qlen, device=dev))[:, None]
        )
        s = s.masked_fill(~visible[None], float("-inf"))
        ref[r] = torch.einsum("hqk,hkd->qhd", torch.softmax(s, dim=-1), v)
    return out.float().view(nreq, qlen, H, D), ref


@pytest.mark.parametrize("qlen,nreq", sorted(modeling.FULLCTX_DECODE_RECIPES))
def test_the_drafter_pass_at_tile_edges(qlen, nreq) -> None:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    recipe = modeling.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    reducers = (True, False) if fullctx.has_own_reducer() else (False,)
    for own in reducers:
        fullctx.prebuild(
            torch.device("cuda"), {(qlen, nreq): recipe}, MAX_LEN, own, get_env_enable_pdl()
        )
    lens = _edge_lens(fullctx.plan_for(torch.device("cuda"), qlen, nreq, False), qlen)
    lens = [n for n in lens if n > qlen]
    batches = [[n] for n in lens] if nreq == 1 else [[a, b] for a, b in zip(lens, lens[::-1])]
    batches.append([1300, 1281][:nreq])
    failures = []
    for i, kv_lens in enumerate(batches):
        # The first draft pass runs in-model as a plain multi-row generation step; every other batch also takes the
        # spec-decoding form.
        spec = qlen > 1 and i % 2 == 1
        for own in reducers:
            got, ref = _run(_library(qlen, nreq, own), qlen, kv_lens, sum(kv_lens), spec=spec)
            if not bool(torch.isfinite(got).all()) or not torch.allclose(
                got, ref, rtol=FP8_TOL, atol=FP8_TOL
            ):
                failures.append((kv_lens, "bf16-merge" if own else "primts-reducer"))
    assert not failures, failures


def _drafter_model(scale: float):
    """A drafter whose one attention module holds a TrtllmAttention backend as ``attn`` (the backend is no module)."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    backend = TrtllmAttention.__new__(TrtllmAttention)
    backend.kv_cache_scaling_factor = torch.full((1,), scale, device="cuda")
    backend.is_mla_enable, backend.kv_lora_rank, backend.v_head_dim, backend.head_dim = (
        False,
        None,
        None,
        D,
    )
    backend._fmha_manager = SimpleNamespace(fmha_libs=[])
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    model.self_attn = torch.nn.Module()
    model.self_attn.attn = backend
    return model, backend


def test_install_declines_a_scaled_kv_cache() -> None:
    """At unit KV scales the arithmetic is the stock decode's; a scaled cache keeps the stock path."""
    reasons = []
    model, layer = _drafter_model(0.5)
    assert drafter.install(model, {}, MAX_LEN, False, False, None, reasons.append, None) == 0
    assert layer._fmha_manager.fmha_libs == [] and reasons == ["a scaled FP8 KV cache"]
    model, layer = _drafter_model(1.0)
    assert drafter.install(model, {}, MAX_LEN, False, False, None, reasons.append, None) == 1
    assert isinstance(layer._fmha_manager.fmha_libs[0], drafter.M3DrafterFullCtxFmha)


def test_a_cache_bound_past_the_plan_capacity_keeps_the_stock_path() -> None:
    """The plans are compiled for ``max_kv_len`` of K/V: a step whose cache length bound is longer falls through."""

    from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase

    class _Backend:
        num_heads, num_kv_heads, head_dim, quant_mode = H, KVH, D, FP8_KV_QUANT_MODE
        kv_scale_orig_quant = kv_scale_quant_orig = None

    backend = _Backend()
    backend.kv_scale_orig_quant, backend.kv_scale_quant_orig = object(), object()
    reasons = []
    lib = _library(1, 1, False)
    lib._on_uncovered = reasons.append
    lib._attn_ref = weakref.ref(backend)
    fwd = SimpleNamespace(
        attention_input_type=None,
        kv_scale_orig_quant=backend.kv_scale_orig_quant,
        kv_scale_quant_orig=backend.kv_scale_quant_orig,
    )
    meta = SimpleNamespace(num_contexts=0, num_generations=1, max_seq_len=MAX_LEN + 8)
    q = torch.empty(1, H * D, device="cuda")
    assert not lib._is_supported(q, None, None, meta, fwd, phase=FmhaPhase.GENERATION)
    assert reasons == ["the cache length bound is beyond the plans' K/V capacity"]
