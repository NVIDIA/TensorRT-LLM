# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM 10.7) for minimax_m3_sm107tp4_drafter_rope_fp8_kv, the TP4 drafter layer's one-launch generation
preprocess: bitwise against the in-tree ``thop.trtllm_gen_generation_preprocess`` followed by the torch E4M3 cast
of its BF16 query, on the query bytes and on every byte of the paged cache, with GPT-NeoX RoPE from a cos/sin table
(full and partial rotary width), multi-token (spec-decoding positions) and plain passes, one and two requests, and
query values past the E4M3 range. The library's fused pass is also held bitwise to its preprocess pass end to end.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
    m3_drafter_attn as drafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
    m3_fullctx_decode as fullctx,
)
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
)
from tensorrt_llm.bindings.internal import thop

assert torch.cuda.is_available(), "the drafter preprocess op requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the tp4 target is certified on SM 10.7 only", allow_module_level=True)
if not hasattr(torch.ops.trtllm, "minimax_m3_sm107tp4_drafter_rope_fp8_kv"):
    pytest.skip("minimax_m3_sm107tp4_drafter_rope_fp8_kv is not built", allow_module_level=True)

H, KVH, D, PAGE = fullctx.HEADS, fullctx.KV_HEADS, fullctx.HEAD_DIM, 128
MAX_LEN = 1 << 20
FP8_KV_QUANT_MODE = 128
ROPE_GPT_NEOX = 2
_PERSISTENT = {}


def _persistent(name, make):
    if name not in _PERSISTENT:
        _PERSISTENT[name] = make()
    return _PERSISTENT[name]


def _case(qlen, kv_lens, rotary_dim, multi, seed):
    dev = torch.device("cuda")
    nreq = len(kv_lens)
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = math.ceil(max(kv_lens) / PAGE)
    nblocks = nreq * pages + 2
    pool = (torch.randn(nblocks, 2, KVH, PAGE, D, device=dev, generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(nblocks, device=dev, generator=g)[: nreq * pages].view(nreq, pages)
    offsets = torch.zeros(1, nreq, 2, pages, dtype=torch.int32, device=dev)
    offsets[0, :, 0] = 2 * perm
    offsets[0, :, 1] = 2 * perm + 1
    ntok = nreq * qlen
    qkv = torch.randn(ntok, (H + 2 * KVH) * D, device=dev, generator=g) * 4
    # A few query / key values near and past the E4M3 range (both casts saturate at 448).
    hot = torch.rand(qkv.shape, device=dev, generator=g) < 0.01
    qkv = torch.where(
        hot, qkv.sign() * (430 + 80 * torch.rand(qkv.shape, device=dev, generator=g)), qkv
    )
    qkv = qkv.to(torch.bfloat16).contiguous()
    max_pos = max(kv_lens) + 64
    cos_sin = torch.randn(max_pos, rotary_dim // 2, 2, device=dev, generator=g).float().contiguous()
    attn = SimpleNamespace(
        num_heads=H,
        num_kv_heads=KVH,
        head_dim=D,
        q_scaling=1.0,
        attention_chunk_size=None,
        quant_mode=FP8_KV_QUANT_MODE,
        local_layer_idx=0,
        position_embedding_type=ROPE_GPT_NEOX,
        rotary_inv_freq=None,
        rotary_cos_sin=cos_sin,
        predicted_tokens_per_seq=qlen,
        rope_params=SimpleNamespace(
            dim=rotary_dim, theta=10000.0, scale_type=0, scale=1.0, max_positions=max_pos
        ),
    )
    one = torch.ones(1, dtype=torch.float32, device=dev)
    spec_offsets = None
    if multi:
        spec_offsets = _persistent(
            f"pos{qlen}x{nreq}",
            lambda: torch.arange(qlen, dtype=torch.int32, device=dev).repeat(nreq, 1).contiguous(),
        )
        # Spec positions are not the plain row index: shift them per request.
        spec_offsets.copy_(
            torch.arange(qlen, dtype=torch.int32, device=dev).repeat(nreq, 1)
            + torch.arange(nreq, dtype=torch.int32, device=dev)[:, None]
        )
    params = SimpleNamespace(
        qkv_input=qkv,
        query_input=None,
        attn=attn,
        meta=SimpleNamespace(
            kv_cache_block_offsets=offsets,
            host_kv_cache_pool_pointers=torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
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
        sequence_lengths=torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        spec_decoding_generation_lengths=(
            _persistent(
                f"len{qlen}x{nreq}",
                lambda: torch.full((nreq,), qlen, dtype=torch.int32, device=dev),
            )
            if multi
            else None
        ),
        spec_decoding_position_offsets=spec_offsets,
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
        output=torch.zeros(ntok, H * D, dtype=torch.bfloat16, device=dev),
    )
    return params, pool


def _preprocess(params, mpc):
    """run_generation's in-tree preprocess call and the E4M3 cast of its query."""
    attn, meta, fwd, rope = params.attn, params.meta, params.fwd, params.attn.rope_params
    q_processed = thop.trtllm_gen_generation_preprocess(
        params.qkv_input,
        params.workspace,
        params.sequence_lengths,
        params.spec_decoding_generation_lengths,
        params.spec_decoding_position_offsets,
        meta.kv_cache_block_offsets,
        meta.host_kv_cache_pool_pointers,
        meta.host_kv_cache_pool_mapping,
        fwd.kv_scale_orig_quant,
        fwd.kv_scale_quant_orig,
        fwd.out_scale,
        attn.rotary_inv_freq,
        attn.rotary_cos_sin,
        fwd.mrope_position_deltas,
        attn.local_layer_idx,
        params.seq_offset,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        params.tokens_per_block,
        attn.quant_mode,
        params.max_attention_window_size,
        params.cyclic_attention_window_size,
        params.num_tokens,
        params.batch_size,
        params.input_seq_length,
        params.max_past_kv_length,
        rope.dim,
        rope.theta,
        int(rope.scale_type),
        rope.scale,
        rope.max_positions,
        attn.position_embedding_type,
        get_bmm1_scale(attn),
        1.0,
        False,
        attn.predicted_tokens_per_seq,
        get_attention_chunk_size(attn),
        mpc,
        params.total_num_blocks,
        params.kv_factor,
        True,
        params.is_cross,
    )[0]
    rows = params.num_tokens * attn.num_heads * attn.head_dim
    return (
        q_processed.flatten()[:rows]
        .to(torch.float8_e4m3fn)
        .view(params.num_tokens, attn.num_heads, attn.head_dim)
    )


def _op(params, pool, mode):
    attn = params.attn
    nreq, qlen = params.batch_size, params.input_seq_length
    kv_pages = pool.view(-1, KVH, PAGE, D)
    tables = params.meta.kv_cache_block_offsets[0, :nreq]
    q = torch.empty(
        params.num_tokens,
        attn.num_heads,
        attn.head_dim,
        dtype=torch.float8_e4m3fn,
        device=pool.device,
    )
    multi = params.spec_decoding_generation_lengths is not None and qlen > 1
    torch.ops.trtllm.minimax_m3_sm107tp4_drafter_rope_fp8_kv(
        params.qkv_input,
        attn.rotary_cos_sin,
        params.sequence_lengths,
        params.spec_decoding_position_offsets if multi else None,
        kv_pages,
        tables,
        q,
        qlen,
        attn.num_heads,
        attn.num_kv_heads,
        attn.rope_params.dim,
        PAGE,
        mode,
        True,
    )
    return q


# One-request multi-token passes keep the in-tree preprocess (it reads an unbuilt token map), so the spec cases hold
# two requests.
CASES = [
    (qlen, kv_lens, rotary, multi)
    for qlen, multi in ((1, False), (4, True), (4, False))
    for kv_lens in (((4, 130) if multi else (4,)) if qlen > 1 else (130,), (8191, 131073))
    for rotary in (128, 64)
]


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _matches(case, seed):
    """Per rope mode: (query bytes equal, cache bytes equal) against the in-tree preprocess + cast."""
    qlen, kv_lens, rotary, multi = case
    kv_lens = [max(n, qlen) for n in kv_lens]
    mpc = torch.cuda.get_device_properties(0).multi_processor_count
    params, pool = _case(qlen, kv_lens, rotary, multi, seed)
    start = pool.clone()
    q_ref = _preprocess(params, mpc)
    torch.cuda.synchronize()
    pool_ref = pool.clone()
    out = {}
    for mode in (0, 1, 2):
        pool.copy_(start)
        q = _op(params, pool, mode)
        torch.cuda.synchronize()
        out[mode] = (
            bool(torch.equal(_bytes(q), _bytes(q_ref))),
            bool(torch.equal(_bytes(pool), _bytes(pool_ref))),
        )
    return out


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda c: f"q{c[0]}-{'x'.join(map(str, c[1]))}-r{c[2]}-{'spec' if c[3] else 'plain'}",
)
def test_the_pinned_rope_mode_is_bitwise(case) -> None:
    result = _matches(case, seed=7)
    print(f"[rope modes] {case}: {result}", flush=True)
    assert result[dispatch.DRAFTER_ROPE_MODE] == (True, True), result


def test_the_fused_pass_matches_the_preprocess_pass() -> None:
    mpc = torch.cuda.get_device_properties(0).multi_processor_count
    for qlen, nreq in sorted(dispatch.FULLCTX_DECODE_RECIPES):
        for multi in (False, True) if qlen > 1 else (False,):
            kv_lens = [70001 + 517 * r for r in range(nreq)]
            outs = []
            for fused in (False, True):
                params, pool = _case(qlen, kv_lens, 128, multi, seed=11)
                lib = drafter.M3DrafterFullCtxFmha.__new__(drafter.M3DrafterFullCtxFmha)
                lib._recipes = {(qlen, nreq): dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]}
                lib._own_reducer = True
                lib._max_kv_len = MAX_LEN
                lib._multi_processor_count = mpc
                lib._fused_rope_mode = dispatch.DRAFTER_ROPE_MODE if fused else None
                lib._pdl = True
                taken = []
                lib._on_rope_taken = lambda rows, what: taken.append(what)
                lib._on_taken = lambda rows: None
                fullctx.prebuild(torch.device("cuda"), lib._recipes, MAX_LEN, True, True)
                lib.run_generation(params)
                torch.cuda.synchronize()
                assert bool(taken) == (fused and not (multi and nreq == 1))
                outs.append((params.output.clone(), pool.clone()))
            assert torch.equal(_bytes(outs[0][1]), _bytes(outs[1][1])), (qlen, nreq, multi, "cache")
            assert torch.equal(_bytes(outs[0][0]), _bytes(outs[1][0])), (
                qlen,
                nreq,
                multi,
                "output",
            )


def _round_bf16(x):
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.to(torch.int64).to(torch.int32).view(torch.float32)


def _discriminating(n, seed):
    """(cos, sin, x, pair) quadruples whose rotated first-half output's E4M3 byte differs between the rope modes
    (fp32 CPU emulation; the GPU comparison is the verdict)."""
    g = torch.Generator().manual_seed(seed)
    keep = []
    while sum(k.shape[0] for k in keep) < n:
        m = 1 << 20
        c, s = torch.randn(m, generator=g), torch.randn(m, generator=g)
        x = (torch.randn(m, generator=g) * 4).to(torch.bfloat16).float()
        xp = (torch.randn(m, generator=g) * 4).to(torch.bfloat16).float()
        sn = -s
        r0 = (c * x) + (sn * xp)
        r1 = ((c.double() * x.double()) + (sn * xp).double()).float()
        r2 = ((sn.double() * xp.double()) + (c * x).double()).float()
        b = [
            _round_bf16(r.contiguous()).to(torch.float8_e4m3fn).view(torch.uint8)
            for r in (r0, r1, r2)
        ]
        diff = (b[0] != b[1]) | (b[0] != b[2]) | (b[1] != b[2])
        diff &= (r0.abs() < 400) & (r1.abs() < 400)
        keep.append(torch.stack([c, s, x, xp], 1)[diff])
    return torch.cat(keep)[:n]


def test_crafted_inputs_pin_the_rope_mode() -> None:
    """Rotations whose E4M3 byte depends on how the two products are summed: the in-tree preprocess and exactly the
    pinned mode agree on every one."""
    mpc = torch.cuda.get_device_properties(0).multi_processor_count
    params, pool = _case(1, [130], 128, False, seed=3)
    half = 64
    quads = _discriminating(H * half, seed=5).cuda()
    pos = 129  # the one row's rope position: cache length 130, one query row
    table = params.attn.rotary_cos_sin.view(-1, half, 2)
    # One (cos, sin) per dim: every head shares the position's table row, so the heads take the same dims' quads with
    # their own (x, pair).
    table[pos, :, 0] = quads[:half, 0]
    table[pos, :, 1] = quads[:half, 1]
    qkv = params.qkv_input.view(-1, H + 2 * KVH, D)
    for h in range(H):
        qkv[0, h, :half] = quads[:half, 2].to(torch.bfloat16)
        qkv[0, h, half:] = quads[:half, 3].to(torch.bfloat16)
    start = pool.clone()
    q_ref = _preprocess(params, mpc)
    torch.cuda.synchronize()
    agree = {}
    for mode in (0, 1, 2):
        pool.copy_(start)
        q = _op(params, pool, mode)
        torch.cuda.synchronize()
        agree[mode] = int((_bytes(q)[0, :, :half] == _bytes(q_ref)[0, :, :half]).sum())
    print(
        f"[rope modes crafted] agreeing first-half query bytes of {H * half}: {agree}", flush=True
    )
    assert agree[dispatch.DRAFTER_ROPE_MODE] == H * half, agree
