# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PrimTS native sparse MLA FMHA library (DeepSeek-V4, SM100/SM103)."""

import pytest
import torch
import triton

from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    PrimsTSSparseMlaFmha,
    _pack_sparse_mla_metadata,
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._utils import get_sm_version

from ..test_sparse_mla_forward import HAS_FLASH_MLA
from ..test_sparse_mla_forward import (
    test_forward_sparse_mla_unified as _run_unified_sparse_mla_forward,
)

LIB_NAME = "prims_ts_sparse_mla"
LATENT = 512
ROPE = 64
BLOCK = 128

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="PrimTS sparse MLA requires SM100 or SM103"
)


def _skip_unless_toolchain() -> None:
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)


def test_registry_entry_is_opt_in() -> None:
    assert registry.FMHA_LIBS[LIB_NAME] is PrimsTSSparseMlaFmha
    assert LIB_NAME not in registry.DEFAULT_FMHA_LIBS
    names = tuple(registry.FMHA_LIBS)
    assert names.index(LIB_NAME) < names.index("fallback")


def _rope_table(max_positions: int, device: torch.device) -> torch.Tensor:
    """Forward [cos(32), sin(32)] rows, the layout both kernels consume."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE)
    )
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32, device=device), inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


def _random_indices(rows: int, capacity: int, pool_rows: int, device: torch.device):
    lengths = torch.randint(1, capacity + 1, (rows,), dtype=torch.int32, device=device)
    indices = torch.randint(0, pool_rows, (rows, capacity), dtype=torch.int32, device=device)
    inactive = torch.arange(capacity, device=device)[None, :] >= lengths[:, None]
    indices[inactive] = -1
    return indices, lengths


def _prepare_metadata(wrapper, indices, lengths, extra_indices, extra_lengths, heads, bmm1, bmm2):
    """Build SparseMLAPreparedMetadata exactly like the FMHA adapter does."""
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import SparseMLAPreparedMetadata

    rows, ks = indices.shape
    kc = 0 if extra_indices is None else extra_indices.shape[1]
    capacity = int(wrapper._state["capacity"])
    device = indices.device

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    prepared = SparseMLAPreparedMetadata(
        indices=indices.contiguous(),
        lengths=lengths.contiguous(),
        extra_indices=None if kc == 0 else extra_indices.contiguous(),
        extra_lengths=None if kc == 0 else extra_lengths.contiguous(),
        routes=empty((1, rows, capacity)),
        execution_lengths=empty((1, rows)),
        valid_counts=empty((1, rows)),
        scale_params=empty((1, 2 + heads + rows), torch.float32),
    )
    sinks = torch.full((heads,), -torch.inf, dtype=torch.float32, device=device)
    _pack_sparse_mla_metadata[(rows,)](
        prepared.indices,
        prepared.extra_indices if kc else prepared.indices,
        prepared.lengths,
        prepared.extra_lengths if kc else prepared.lengths,
        prepared.routes,
        prepared.execution_lengths,
        prepared.valid_counts,
        prepared.scale_params,
        sinks,
        bmm1,
        bmm2,
        ks,
        kc,
        rows,
        heads,
        capacity,
        256,
        triton.next_power_of_2(heads),
        num_warps=4,
    )
    return prepared


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "rows,heads,groups,topk,extra_topk",
    [
        (1, 128, 16, 128, 0),
        (5, 128, 16, 128, 512),
        (64, 128, 16, 128, 512),
        (16, 64, 8, 128, 256),
        (3, 64, 8, 128, 0),
    ],
)
def test_fused_output_matches_trtllm_inverse_rope_quant(
    dtype, rows, heads, groups, topk, extra_topk
):
    """The kernel's fused epilogue must reproduce TRT-LLM's inverse-RoPE + FP8 quant.

    Reference: the same attention with a BF16 output, followed by
    ``trtllm.fused_inv_rope_fp8_quant_vllm_port`` (the op the unfused DSv4
    path runs). The fused kernel quantizes from FP32 accumulators instead of
    the BF16 output, so compare dequantized values up to FP8 rounding.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    pool_rows, extra_rows, max_positions = 4096, 2048, 4096

    def make_pool(n):
        return (torch.randn(n, 1, LATENT, dtype=torch.bfloat16, device=device)).to(dtype)

    pool = make_pool(pool_rows)
    extra_pool = make_pool(extra_rows) if extra_topk else None
    indices, lengths = _random_indices(rows, topk, pool_rows, device)
    extra_indices, extra_lengths = (
        (None, None) if extra_topk == 0 else _random_indices(rows, extra_topk, extra_rows, device)
    )
    query = (torch.randn(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device) * 0.5).to(
        dtype
    )
    positions = torch.randint(0, max_positions, (rows,), dtype=torch.int32, device=device)
    cos_sin = _rope_table(max_positions, device)

    softmax_scale = LATENT**-0.5
    descale = 0.75 if dtype == torch.float8_e4m3fn else 1.0
    bmm1 = softmax_scale * descale * descale
    bmm2 = descale
    run_scales = dict(
        softmax_scale=softmax_scale, q_scale=descale, kv_scale=descale, extra_kv_scale=descale
    )
    plan_args = dict(max_topk=topk, max_extra_topk=extra_topk, max_seq_len_q=1, q_data_type=dtype)

    reference = BatchSparseMLADecodePagedTSWrapper()
    reference.plan(device, rows, heads, **plan_args)
    metadata = _prepare_metadata(
        reference, indices, lengths, extra_indices, extra_lengths, heads, bmm1, bmm2
    )
    out_bf16 = reference.run(query, pool, metadata, extra_pool, **run_scales)
    heads_per_group = heads // groups
    ref_fp8, ref_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        out_bf16.view(rows, heads, LATENT),
        positions,
        cos_sin,
        groups,
        heads_per_group,
        LATENT - ROPE,
        ROPE,
        BLOCK,
        False,
    )

    fused = BatchSparseMLADecodePagedTSWrapper()
    fused.plan(
        device,
        rows,
        heads,
        output_quant_block_size=BLOCK,
        out_scale_format="fp32",
        num_output_groups=groups,
        **plan_args,
    )
    metadata = _prepare_metadata(
        fused, indices, lengths, extra_indices, extra_lengths, heads, bmm1, bmm2
    )
    channels = heads_per_group * LATENT
    padded_rows = (rows + 3) // 4 * 4
    # The exact buffers TRT-LLM's fused DSv4 epilogue allocates.
    out_buf = torch.empty((groups, rows, channels), dtype=torch.float8_e4m3fn, device=device)
    scale_buf = torch.empty(
        (groups, channels // BLOCK, padded_rows), dtype=torch.float32, device=device
    )
    fused_out, fused_scale = fused.run(
        query,
        pool,
        metadata,
        extra_pool,
        out=out_buf.transpose(0, 1),
        out_scale=scale_buf.permute(2, 0, 1)[:rows],
        token_positions=positions.view(rows, 1),
        cos_sin_cache=cos_sin,
        **run_scales,
    )
    assert fused_out.data_ptr() == out_buf.data_ptr()
    assert fused_scale.data_ptr() == scale_buf.data_ptr()
    assert ref_fp8.shape == out_buf.shape and ref_scale.shape == scale_buf.shape

    ref_scales = ref_scale[..., :rows].permute(0, 2, 1)  # [G, T, blocks]
    got_scales = scale_buf[..., :rows].permute(0, 2, 1)
    torch.testing.assert_close(got_scales, ref_scales, rtol=2e-2, atol=1e-6)

    def dequant(values, scales):
        blocks = values.float().view(groups, rows, channels // BLOCK, BLOCK)
        return (blocks * scales[..., None]).flatten(-2)

    ref = dequant(ref_fp8, ref_scales)
    got = dequant(out_buf, got_scales)
    # One E4M3 ulp is 1/8 of the magnitude. The reference rotates the
    # BF16-rounded output, so near-cancelling RoPE channels (D[448:512]) carry
    # an absolute error of a few BF16 ulps of their O(1) inputs, which the
    # kernel (rotating FP32 accumulators) does not have; that shows up as up to
    # about one quantization step of the block. Allow two steps there and half
    # a step elsewhere, on top of two relative ulps.
    block_scales = ref_scales.repeat_interleave(BLOCK, dim=-1)
    is_rope = torch.zeros(channels, dtype=torch.bool, device=device)
    is_rope.view(-1, LATENT)[:, LATENT - ROPE :] = True
    step_allowance = torch.where(is_rope, 2.0, 0.5) * block_scales
    ulp = ref.abs() * 0.125 + block_scales * 2.0**-6
    diff = (got - ref).abs()
    assert (diff <= 2.0 * ulp + step_allowance + 1e-6).all(), (
        f"max fused-vs-reference gap {diff.max().item()}"
    )
    assert diff.mean().item() <= 0.02 * ref.abs().mean().item() + 1e-6


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "batch_name",
    [
        "small_decode",
        "medium_decode",
        "single_decode",
        "small_mixed",
        "mixed_heavy_decode",
    ],
)
def test_forward_sparse_mla_unified_prims_ts(monkeypatch, batch_name, kv_cache_dtype):
    """Run the DeepSeek-V4 module test with the PrimTS library serving generation."""
    _skip_unless_toolchain()
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    calls = []
    original_forward = PrimsTSSparseMlaFmha.forward

    def counting_forward(self, *args, **kwargs):
        calls.append(1)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", counting_forward)
    _run_unified_sparse_mla_forward(batch_name, kv_cache_dtype, "deepseek_v4")
    assert calls, "the PrimTS sparse MLA FMHA did not serve the generation phase"
