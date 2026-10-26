# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Interface-level sparse attention tests and benchmark harness.

This file intentionally keeps correctness coverage and ad-hoc benchmarking in
one module. Tests only exercise the public `sparse_atten_func` interface:
  - sparse attention: forward numerical checks
  - page attention: forward numerical checks

Examples:
  pytest -q test_sparse_atten.py
  python test_sparse_atten.py benchmark
  python test_sparse_atten.py benchmark --customer-case both --backend both --q2k-pattern both --causal
  python test_sparse_atten.py benchmark --paged --causal --page-size 128 --seqused-trim 17
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import torch

import interface as sparse_interface
from interface import sparse_atten_func, sparse_atten_nvfp4_kv_func
from sparse_index_utils import build_k2q_csr, build_k2q_csr_torch_reference
from src.sm100.prepare_scheduler import (
    prepare_sparse_fwd_schedule_and_split,
)

# The Triton reference backend was removed for the open-source release.
# Tests below validate the CuTe-DSL kernels against PyTorch references only.
TRITON_REFERENCE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEFAULT_B = 1
DEFAULT_SQ = 32768
DEFAULT_SKV = 32768
DEFAULT_TOPK = 16
DEFAULT_HEAD_KV = 4
DEFAULT_QHEAD_PER_KV = 16
DEFAULT_DIM = 128
DEFAULT_BLK_KV = 128
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_SEED = 42
BLK_KV = DEFAULT_BLK_KV
DECODE_BATCH = 32
DECODE_SEQLEN_Q = 8
DECODE_HEAD_KV = 4
DECODE_QHEAD_PER_KV = 16
DECODE_DIM = 128
DECODE_KV_TOKEN_SWEEP = tuple(2**exp for exp in range(3, 21))


def test_prepare_paged_hnd_input_keeps_aligned_outer_page_stride() -> None:
    pages, roles, heads, page_size, head_dim = 5, 3, 2, 128, 128
    pool = torch.empty(
        pages,
        roles,
        heads,
        page_size,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    view = pool[:, 1]

    assert not view.is_contiguous()
    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)
    assert prepared.data_ptr() == view.data_ptr()
    assert prepared.stride() == view.stride()


def test_prepare_paged_hnd_input_materializes_unpacked_tokens() -> None:
    pages, heads, page_size, head_dim = 5, 2, 128, 128
    storage = torch.arange(
        pages * heads * page_size * 2 * head_dim,
        dtype=torch.int32,
        device="cuda",
    ).remainder(16).to(torch.float8_e4m3fn).reshape(pages, heads, page_size * 2, head_dim)
    view = storage[:, :, ::2, :]

    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)
    assert prepared.is_contiguous()
    assert prepared.data_ptr() != view.data_ptr()
    torch.testing.assert_close(prepared, view, rtol=0, atol=0)


def test_prepare_paged_hnd_input_materializes_unaligned_outer_stride() -> None:
    pages, heads, page_size, head_dim = 5, 2, 128, 128
    outer_stride = heads * page_size * head_dim + 1
    storage = torch.arange(
        pages * outer_stride,
        dtype=torch.int32,
        device="cuda",
    ).remainder(16).to(torch.float8_e4m3fn)
    view = storage.as_strided(
        (pages, heads, page_size, head_dim),
        (outer_stride, page_size * head_dim, head_dim, 1),
    )

    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)
    assert prepared.is_contiguous()
    assert prepared.data_ptr() != view.data_ptr()
    torch.testing.assert_close(prepared, view, rtol=0, atol=0)


def test_prepare_paged_hnd_input_defers_page_size_validation() -> None:
    pages, roles, heads, page_size, head_dim = 5, 2, 2, 128, 128
    pool = torch.empty(
        pages,
        roles,
        heads,
        page_size,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    view = pool[:, 0]

    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size // 2)
    assert prepared.data_ptr() == view.data_ptr()
    with pytest.raises(ValueError, match="page_size == blk_kv"):
        sparse_interface._prepare_paged_kv_for_tma(prepared, prepared, page_size // 2)


@contextmanager
def _nvtx_range(message: str):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _nvtx_token(text: object) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))


class Q2KPattern:
    SINK = "sink"
    UNIFORM = "uniform"
    BOTH = "both"


class BenchmarkBackend:
    CUTE = "cute"
    TRITON = "triton"
    BOTH = "both"


CUSTOMER_BENCHMARK_CASES = {
    "ring48k": {
        "name": "bs1_nhq16_hkv1_seq48k_ring_attn",
        "batch": 1,
        "seqlen_q": 48 * 1024,
        "seqlen_k": 48 * 1024,
        "head_kv": 1,
        "qhead_per_kv": 16,
    },
    "ulysses384k": {
        "name": "bs1_nhq2_hkv1_seq384k_ulysses",
        "batch": 1,
        "seqlen_q": 384 * 1024,
        "seqlen_k": 384 * 1024,
        "head_kv": 1,
        "qhead_per_kv": 2,
    },
}
CUSTOMER_QHEAD_SWEEP_CASES = {
    f"qhead{qhead}": {
        "name": f"bs1_nhq{qhead}_hkv1_seq{(768 // qhead)}k_qhead_sweep",
        "batch": 1,
        "seqlen_q": 768 * 1024 // qhead,
        "seqlen_k": 768 * 1024 // qhead,
        "head_kv": 1,
        "qhead_per_kv": qhead,
    }
    for qhead in (1, 2, 4, 8, 16)
}


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _bottom_right_causal_visible_blocks(
    q_pos: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    blk_kv: int,
) -> torch.Tensor:
    num_kv_blocks = (seqlen_k + blk_kv - 1) // blk_kv
    causal_k_limit = q_pos + (seqlen_k - seqlen_q)
    return (causal_k_limit // blk_kv + 1).clamp(min=0, max=num_kv_blocks).to(torch.int32)


def _randn_qkv(shape: tuple[int, ...], *, dtype: torch.dtype, device: str | torch.device = "cuda") -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def generate_test_data(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_q: int,
    head_kv: int,
    dim: int,
    topk: int,
    *,
    blk_kv: int = BLK_KV,
    causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dense Q/K/V plus sparse q->kv block indices."""
    assert seqlen_k % blk_kv == 0, f"seqlen_k ({seqlen_k}) must be divisible by blk_kv ({blk_kv})"
    assert head_q % head_kv == 0, f"head_q ({head_q}) must be divisible by head_kv ({head_kv})"
    num_kv_blocks = seqlen_k // blk_kv
    assert topk <= num_kv_blocks, f"topk ({topk}) must be <= num_kv_blocks ({num_kv_blocks})"

    q = _randn_qkv((batch, seqlen_q, head_q, dim), dtype=dtype, device=device)
    k = _randn_qkv((batch, seqlen_k, head_kv, dim), dtype=dtype, device=device)
    v = _randn_qkv((batch, seqlen_k, head_kv, dim), dtype=dtype, device=device)

    if not causal:
        q2k_indices = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [torch.randperm(num_kv_blocks, device=device)[:topk] for _ in range(seqlen_q)]
                        )
                        for _ in range(head_kv)
                    ]
                )
                for _ in range(batch)
            ]
        ).to(torch.int32)
        return q, k, v, q2k_indices

    q2k_indices = torch.full((batch, head_kv, seqlen_q, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(head_kv):
            for q_idx in range(seqlen_q):
                visible_blocks = max(
                    0,
                    min((q_idx + seqlen_k - seqlen_q) // blk_kv + 1, num_kv_blocks),
                )
                if visible_blocks == 0:
                    continue
                if visible_blocks <= topk:
                    chosen = torch.arange(visible_blocks, device=device, dtype=torch.int32)
                else:
                    chosen = torch.randperm(visible_blocks, device=device, dtype=torch.int32)[:topk]
                    chosen, _ = chosen.sort()
                q2k_indices[b, h, q_idx, : chosen.numel()] = chosen
    return q, k, v, q2k_indices


def sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    blk_kv: int = BLK_KV,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: Optional[float] = None,
    upcast: bool = True,
    seqused_k: Optional[torch.Tensor] = None,
    p_dtype: Optional[torch.dtype] = None,
    o_partial_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Golden sparse attention reference via explicit mask construction."""
    batch, seqlen_q, head_q, dim = q.shape
    seqlen_k = k.shape[1]
    head_kv = k.shape[2]
    qhead_per_kv = head_q // head_kv
    num_kv_blocks = seqlen_k // blk_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            raise TypeError("seqused_k must be torch.int32")
        if seqused_k.shape != (batch,):
            raise ValueError("seqused_k must have shape [batch]")

    q_t = q.float() if upcast else q
    k_t = k.float() if upcast else k
    v_t = v.float() if upcast else v

    block_mask = torch.zeros(batch, head_kv, seqlen_q, num_kv_blocks, dtype=torch.bool, device=q.device)
    valid_q2k = q2k_indices >= 0
    safe_q2k = torch.where(valid_q2k, q2k_indices, torch.zeros_like(q2k_indices))
    flat_mask = block_mask.view(-1, num_kv_blocks)
    flat_valid = valid_q2k.view(-1, valid_q2k.shape[-1])
    flat_idx = safe_q2k.view(-1, safe_q2k.shape[-1]).long()
    row_idx = torch.arange(flat_mask.shape[0], device=q.device).unsqueeze(1)
    flat_mask[row_idx.expand_as(flat_idx)[flat_valid], flat_idx[flat_valid]] = True

    token_mask = block_mask.repeat_interleave(blk_kv, dim=3)
    if qhead_per_kv > 1:
        token_mask = token_mask.repeat_interleave(qhead_per_kv, dim=1)
        k_t = k_t.repeat_interleave(qhead_per_kv, dim=2)
        v_t = v_t.repeat_interleave(qhead_per_kv, dim=2)

    scores = torch.einsum("bshd,bthd->bhst", q_t * softmax_scale, k_t)
    if seqused_k is not None:
        kv_pos = torch.arange(seqlen_k, device=q.device, dtype=torch.int32)
        valid_k_mask = kv_pos.unsqueeze(0) < seqused_k.view(batch, 1)
        token_mask = token_mask & valid_k_mask[:, None, None, :]
    if causal:
        q_pos = torch.arange(seqlen_q, device=q.device, dtype=torch.int32)
        kv_pos = torch.arange(seqlen_k, device=q.device, dtype=torch.int32)
        causal_seqlen_k = (
            seqused_k
            if seqused_k is not None
            else torch.full((batch,), seqlen_k, device=q.device, dtype=torch.int32)
        )
        causal_limit = q_pos.unsqueeze(0) + (causal_seqlen_k.view(batch, 1) - seqlen_q)
        causal_mask = kv_pos.view(1, 1, seqlen_k) <= causal_limit.unsqueeze(-1)
        token_mask = token_mask & causal_mask.unsqueeze(1)
    scores = scores.masked_fill(~token_mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1).transpose(1, 2).contiguous()
    lse_temperature_out = (
        torch.logsumexp(scores * (1.0 / float(lse_temperature_scale)), dim=-1)
        .transpose(1, 2)
        .contiguous()
        if lse_temperature_scale is not None
        else None
    )
    row_has_value = token_mask.any(dim=-1, keepdim=True)
    scores_for_softmax = torch.where(row_has_value, scores, torch.zeros_like(scores))
    if o_partial_dtype is not None:
        # Split-K reference that mirrors the kernel: each topK split runs PV
        # independently, quantizes the partial output, and combine accumulates
        # scaled partials. This is what the kernel actually does when
        # O_partial dtype < fp32. Re-use scratch buffers across splits to
        # avoid topK x O(batch*head*sq*sk) peak memory.
        # Wrap in no_grad: this split-K reference is forward-only and should
        # avoid retaining per-split graphs.
        with torch.no_grad():
            out = torch.zeros(
                batch, seqlen_q, head_q, dim, dtype=torch.float32, device=q.device
            )
            split_block_mask_buf = torch.zeros_like(block_mask)
            split_flat_mask_buf = split_block_mask_buf.view(-1, num_kv_blocks)
            flat_rows = torch.arange(split_flat_mask_buf.shape[0], device=q.device)
            for split in range(q2k_indices.shape[-1]):
                split_block_mask_buf.zero_()
                split_indices = q2k_indices[..., split]
                split_valid = split_indices >= 0
                split_flat_idx = torch.where(
                    split_valid, split_indices, torch.zeros_like(split_indices)
                ).view(-1).long()
                split_flat_valid = split_valid.view(-1)
                split_flat_mask_buf[flat_rows[split_flat_valid], split_flat_idx[split_flat_valid]] = True

                # Build per-split token mask without keeping repeat_interleave
                # intermediates alive past one statement.
                split_token_mask = split_block_mask_buf.repeat_interleave(blk_kv, dim=3)
                if qhead_per_kv > 1:
                    split_token_mask = split_token_mask.repeat_interleave(qhead_per_kv, dim=1)
                split_token_mask &= token_mask

                # split_scores is a transient 6.4 GB-class tensor; reuse by
                # copying scores once and applying mask in-place.
                split_scores = scores.clone()
                split_scores.masked_fill_(~split_token_mask, float("-inf"))
                split_lse = torch.logsumexp(split_scores, dim=-1)
                split_has_value = split_token_mask.any(dim=-1, keepdim=True)
                del split_token_mask
                split_scores.masked_fill_(~split_has_value, 0.0)
                split_attn = torch.softmax(split_scores, dim=-1)
                split_attn.masked_fill_(~split_has_value, 0.0)
                del split_scores, split_has_value
                if p_dtype is not None:
                    split_attn = split_attn.to(p_dtype).float()
                out_partial = torch.einsum("bhst,bthd->bshd", split_attn, v_t)
                del split_attn
                out_partial = out_partial.to(o_partial_dtype).float()
                split_lse_t = split_lse.transpose(1, 2).contiguous()
                del split_lse
                scale = torch.exp(split_lse_t - lse)
                scale = torch.where(
                    torch.isfinite(split_lse_t), scale, torch.zeros_like(scale)
                )
                del split_lse_t
                out.add_(scale.unsqueeze(-1) * out_partial)
                del scale, out_partial
            del split_block_mask_buf, flat_rows
    elif p_dtype == torch.float8_e4m3fn:
        # The SM100 fp8 forward path quantizes unnormalized P to e4m3 before
        # PV, then applies 1 / row_sum in the epilogue.
        row_max = scores_for_softmax.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores_for_softmax - row_max)
        exp_scores = torch.where(
            token_mask & row_has_value,
            exp_scores,
            torch.zeros_like(exp_scores),
        )
        row_sum = exp_scores.sum(dim=-1, keepdim=True)
        p_unnorm = exp_scores.to(p_dtype).float()
        out = torch.einsum("bhst,bthd->bshd", p_unnorm, v_t)
        out = out / torch.where(
            row_sum.transpose(1, 2) > 0,
            row_sum.transpose(1, 2),
            torch.ones_like(row_sum.transpose(1, 2)),
        )
    else:
        attn = torch.where(
            row_has_value,
            torch.softmax(scores_for_softmax, dim=-1),
            torch.zeros_like(scores),
        )
        if p_dtype is not None:
            attn = attn.to(p_dtype).float()
        out = torch.einsum("bhst,bthd->bshd", attn, v_t)
    if lse_temperature_out is not None:
        return out, lse, lse_temperature_out
    return out, lse


def pack_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    page_size: int,
    page_table_mode: str = "identity",
    max_pages_per_seq: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dense [B, Skv, H, D] KV into paged [num_pages, H, page_size, D]."""
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if page_table_mode not in {"identity", "shuffle"}:
        raise ValueError(f"Unsupported page_table_mode={page_table_mode!r}")

    batch, seqlen_k, head_kv, dim = k.shape
    logical_pages = (seqlen_k + page_size - 1) // page_size
    max_pages_per_seq = logical_pages if max_pages_per_seq is None else max_pages_per_seq
    if max_pages_per_seq < logical_pages:
        raise ValueError("max_pages_per_seq must cover the logical KV length")
    total_pages = batch * max_pages_per_seq
    device = k.device

    page_table = torch.arange(total_pages, device=device, dtype=torch.int32).view(batch, max_pages_per_seq)
    if page_table_mode == "shuffle":
        perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
        page_table = perm.to(torch.int32).view(batch, max_pages_per_seq)

    k_paged = torch.zeros(total_pages, head_kv, page_size, dim, dtype=k.dtype, device=device)
    v_paged = torch.zeros_like(k_paged)
    for b in range(batch):
        for logical_page in range(logical_pages):
            physical_page = int(page_table[b, logical_page].item())
            src_start = logical_page * page_size
            src_end = min(src_start + page_size, seqlen_k)
            page_len = src_end - src_start
            if page_len > 0:
                k_paged[physical_page, :, :page_len].copy_(
                    k[b, src_start:src_end].transpose(0, 1)
                )
                v_paged[physical_page, :, :page_len].copy_(
                    v[b, src_start:src_end].transpose(0, 1)
                )
    return k_paged, v_paged, page_table


def _install_flash_attn3_stub_for_te_import() -> None:
    """Install a minimal import stub for TE builds that import FA3 attention."""

    package = sys.modules.get("flash_attn_3", types.ModuleType("flash_attn_3"))
    interface = sys.modules.get(
        "flash_attn_3.flash_attn_interface",
        types.ModuleType("flash_attn_3.flash_attn_interface"),
    )

    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash_attn_3 attention stub is import-only")

    for name in (
        "flash_attn_func",
        "flash_attn_varlen_func",
        "flash_attn_with_kvcache",
        "_flash_attn_forward",
        "_flash_attn_backward",
    ):
        setattr(interface, name, _unavailable)
    package.flash_attn_interface = interface
    sys.modules.setdefault("flash_attn_3", package)
    sys.modules.setdefault("flash_attn_3.flash_attn_interface", interface)


# The kernel reads exactly one scale-factor layout pair, unconditionally: K
# linear (`t*S + s`) and V in the 4x4 token-quad swizzle
# (`(t//4)*(4*S) + 4*s + (t%4)`).  There is no third option, no runtime flag and
# no 128x4 path left anywhere in the branch.
NVFP4_K_SCALE_LAYOUT = "linear"
NVFP4_V_SCALE_LAYOUT = "swizzle4x4"


def _pack_synthetic_nvfp4_scale(
    logical: torch.Tensor,
    *,
    layout: str,
) -> torch.Tensor:
    """Pack a *logical* block-scale array into the bytes the kernel reads.

    ``logical`` is indexed exactly like the interface's scale argument:
    ``[num_pages, Hkv, page_size, S]`` when paged and ``[total_k, Hkv, S]``
    when flat.  The result has the same shape and the same bytes, permuted
    within each region.

    Only the *within-region* permutation comes from
    ``quantize.pack_nvfp4_scale``.  The region decomposition is re-derived here
    from the kernel on purpose, so that a silent change of region on either
    side surfaces as a failure rather than as agreement:

      * paged -- a region is one ``(page, head)`` pair covering ``page_size``
        tokens (``src/sm100/fwd/atten_fwd_nvfp4_kv.py::_scale_region_base``).
      * flat -- a region is ``R`` consecutive global tokens of a single KV
        head, laid out head-minor, with ``R = 1`` for ``"linear"`` and
        ``R = 4`` for ``"swizzle4x4"``
        (``..._scale_region_base_flat``).  ``R = 4`` requires
        ``total_k % 4 == 0``.  Flat is a test-only geometry with no production
        producer, so it is not an ABI.

    ``"linear"`` is ``t*S + s``, i.e. plain row-major over ``[tokens, S]``, and
    is therefore the identity for both geometries.
    """

    if layout == NVFP4_K_SCALE_LAYOUT:
        return logical.contiguous()
    if layout != NVFP4_V_SCALE_LAYOUT:
        raise ValueError(f"unknown NVFP4 scale layout {layout!r}")

    from quantize import pack_nvfp4_scale

    if logical.ndim == 4:
        num_pages, head_kv, page_size, scale_cols = (int(d) for d in logical.shape)
        regions = logical.reshape(num_pages * head_kv, page_size, scale_cols)
        packed = pack_nvfp4_scale(
            regions.contiguous(),
            page_size=page_size,
            scale_cols=scale_cols,
            layout=layout,
        )
        return packed.reshape(num_pages, head_kv, page_size, scale_cols).contiguous()

    if logical.ndim != 3:
        raise ValueError(
            "logical NVFP4 scales must be rank-4 [num_pages, Hkv, page_size, S] "
            f"or rank-3 [total_k, Hkv, S], got shape {tuple(logical.shape)}"
        )
    total_k, head_kv, scale_cols = (int(d) for d in logical.shape)
    if total_k % 4 != 0:
        raise ValueError(
            "the flat V swizzle tiles tokens in quads, so total_k must be a "
            f"multiple of 4, got {total_k}"
        )
    regions = (
        logical.reshape(total_k // 4, 4, head_kv, scale_cols)
        .permute(0, 2, 1, 3)
        .reshape(total_k // 4 * head_kv, 4, scale_cols)
        .contiguous()
    )
    packed = pack_nvfp4_scale(
        regions,
        page_size=4,
        scale_cols=scale_cols,
        layout=layout,
    )
    return packed.reshape(total_k, head_kv, scale_cols).contiguous()


def _make_synthetic_nvfp4_tensor(
    shape: tuple[int, ...],
    *,
    layout: str,
    global_scale_value: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> object:
    """Create deterministic packed NVFP4 data with *varying* block scales.

    The block scales must vary for these tests to mean anything: with every
    scale byte set to the same value, permuting the whole scale array leaves
    the dequantized result bit-identical, so a wrong scale-layout decode in the
    kernel is undetectable. Varying scales make the layout observable.

    Bytes 0x30..0x3E are positive E4M3 values in [0.5, 1.875] -- no NaN, zero
    or denormal -- so the reference stays well conditioned.

    The scales are drawn in *logical* ``[..., S]`` order and only then packed,
    so the same seed produces the same logical scale for a given
    ``(token, scale_col)`` under every layout.  Filling the *physical* array
    with random bytes instead -- which is what the 128x4 version of this helper
    did -- would make the logical scales a permutation of each other and no two
    layouts would ever agree.

    Pass ``generator=`` when the draw must be independent of how many global
    RNG draws the surrounding input builder happened to make; the golden gate
    below depends on that.
    """

    from quantize import Nvfp4QuantizedTensor

    if shape[-1] % 16 != 0:
        raise ValueError("NVFP4 synthetic shape requires D divisible by 16")
    rows = math.prod(int(dim) for dim in shape[:-1])
    scale_cols = int(shape[-1]) // 16
    draw = {} if generator is None else {"generator": generator}
    data = torch.randint(
        0,
        256,
        (*shape[:-1], shape[-1] // 2),
        device="cuda",
        dtype=torch.uint8,
        **draw,
    )
    # Drawn as rank-2 [rows, S] and reshaped, not drawn in the final rank: the
    # golden capture script draws it that way, and only an identical draw makes
    # the two comparable.
    logical_scale = torch.randint(
        0x30,
        0x3F,
        (rows, scale_cols),
        device="cuda",
        dtype=torch.uint8,
        **draw,
    ).reshape(*shape[:-1], scale_cols)
    global_scale = torch.full(
        (1,),
        float(global_scale_value),
        device="cuda",
        dtype=torch.float32,
    )
    return Nvfp4QuantizedTensor(
        data=data,
        scale=_pack_synthetic_nvfp4_scale(logical_scale, layout=layout),
        global_scale=global_scale,
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(shape),
        layout=layout,
    )


def _quantize_bf16_to_nvfp4_with_layout(x: torch.Tensor, layout: str) -> object:
    """Quantize one tensor to NVFP4 under an explicitly chosen SF layout.

    Used both for the correct layout and -- by the negative controls -- for the
    wrong one.  The kernel has no layout flag, so packing on the host with the
    wrong ``layout=`` is the only way to build a mismatched input.
    """

    _install_flash_attn3_stub_for_te_import()
    from quantize import quantize_bf16_to_nvfp4

    return quantize_bf16_to_nvfp4(x, layout=layout)


def _quantize_kv_bf16_to_nvfp4(k: torch.Tensor, v: torch.Tensor) -> tuple[object, object]:
    """Quantize a BF16 K/V pair for the NVFP4 KV path.

    K is packed linear and V with the 4x4 token-quad swizzle, which is the one
    layout pair the kernel reads.

    This deliberately does not skip on error. It used to swallow any
    ``RuntimeError`` from the quantizer and call ``pytest.skip``, which turned a
    missing Transformer Engine into 141 green-looking skips and silently removed
    *all* NVFP4 accuracy coverage on machines that have the hardware -- and would
    equally have hidden a genuine quantizer bug. The quantizer now has a
    dependency-free PyTorch backend, so there is nothing left to be unavailable
    and a raised error here is a real defect.
    """

    _install_flash_attn3_stub_for_te_import()
    from quantize import quantize_kv_bf16_to_nvfp4

    return quantize_kv_bf16_to_nvfp4(k, v)


def _dequant_nvfp4_to_bf16(qx: object, *, include_global_scale: bool = True) -> torch.Tensor:
    """Reference dequantization.

    Layout-independent by construction: it reads ``qx.layout`` and undoes
    whatever packing produced ``qx.scale``, so the same logical block scales
    give the same BF16 K/V no matter how the FP4 kernel addresses the bytes.
    That is what lets the BF16 reference stand as an absolute gate across the
    layout change.
    """

    from quantize import dequantize_nvfp4_to_bf16

    return dequantize_nvfp4_to_bf16(
        qx,
        include_global_scale=include_global_scale,
    )


def _get_sparse_decode_atten_func_for_test():
    wrapper_cls = getattr(sparse_interface, "SparseDecodePagedAttentionWrapper", None)
    if wrapper_cls is None:
        pytest.skip("SparseDecodePagedAttentionWrapper is not implemented yet")
    return wrapper_cls(blk_kv=BLK_KV, causal=True)


def _get_sparse_decode_atten_func_for_benchmark():
    wrapper_cls = getattr(sparse_interface, "SparseDecodePagedAttentionWrapper", None)
    if wrapper_cls is None:
        raise RuntimeError("SparseDecodePagedAttentionWrapper is not implemented yet")
    return wrapper_cls(blk_kv=BLK_KV, causal=True)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _cat_cu_seqlens(lengths: tuple[int, ...], *, device: str = "cuda") -> torch.Tensor:
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, device=device, dtype=torch.int32)


def _sample_q_lens(batch: int, max_seqlen_q: int, *, seed: int) -> tuple[int, ...]:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if max_seqlen_q < 1:
        raise ValueError("max_seqlen_q must be >= 1")
    if batch == 1:
        return (max_seqlen_q,)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    lengths = torch.randint(1, max_seqlen_q + 1, (batch,), generator=generator, dtype=torch.int64)
    lengths[0] = max_seqlen_q
    if torch.all(lengths == max_seqlen_q):
        lengths[-1] = max(1, max_seqlen_q - 1)
    return tuple(int(length.item()) for length in lengths)


def _sample_k_lens(
    batch: int,
    max_seqlen_kv: int,
    *,
    blk_kv: int,
    seed: int,
) -> tuple[int, ...]:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if max_seqlen_kv < blk_kv or max_seqlen_kv % blk_kv != 0:
        raise ValueError("max_seqlen_kv must be a positive multiple of blk_kv")
    if batch == 1:
        return (max_seqlen_kv,)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    max_blocks = max_seqlen_kv // blk_kv
    num_blocks = torch.randint(1, max_blocks + 1, (batch,), generator=generator, dtype=torch.int64)
    num_blocks[0] = max_blocks
    if torch.all(num_blocks == max_blocks):
        num_blocks[-1] = max(1, max_blocks - 1)
    lengths = num_blocks * blk_kv
    return tuple(int(length.item()) for length in lengths)


def _parse_dtype(raw: str) -> torch.dtype:
    if raw == "bf16":
        return torch.bfloat16
    if raw == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported dtype {raw!r}")


def _parse_partial_dtype(raw: str) -> torch.dtype:
    if raw == "fp32":
        return torch.float32
    if raw == "bf16":
        return torch.bfloat16
    if raw == "fp16":
        return torch.float16
    if raw == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported partial dtype {raw!r}")


def _build_pattern_q2k(
    *,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    head_kv: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    pattern: str,
    device: str = "cuda",
) -> torch.Tensor:
    if pattern not in {Q2KPattern.SINK, Q2KPattern.UNIFORM}:
        raise ValueError(f"unsupported q2k pattern: {pattern}")
    total_q = sum(q_lens)
    q2k = torch.full((head_kv, total_q, topk), -1, dtype=torch.int32, device=device)
    slots = torch.arange(topk, device=device, dtype=torch.int32)
    head_hash = torch.arange(head_kv, device=device, dtype=torch.int64).view(-1, 1)
    q_cursor = 0

    for seqlen_q, seqlen_k in zip(q_lens, k_lens):
        num_kv_blocks = (seqlen_k + blk_kv - 1) // blk_kv
        if num_kv_blocks < 1:
            raise ValueError("each sequence needs at least one KV block")
        q_local = torch.arange(seqlen_q, device=device, dtype=torch.int32)
        if causal:
            visible_blocks = _bottom_right_causal_visible_blocks(
                q_local,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                blk_kv=blk_kv,
            )
        else:
            visible_blocks = torch.full((seqlen_q,), num_kv_blocks, dtype=torch.int32, device=device)
        budget = visible_blocks.clamp(max=topk)
        latest = visible_blocks - 1

        if pattern == Q2KPattern.UNIFORM:
            if causal:
                first = latest - budget + 1
                idx = first[:, None] + slots[None, :]
            else:
                idx = torch.remainder(q_local[:, None] // blk_kv + slots[None, :], num_kv_blocks)
            idx = torch.where(slots[None, :] < budget[:, None], idx, torch.full_like(idx, -1))
            q2k[:, q_cursor : q_cursor + seqlen_q, :] = idx[None, :, :]
            q_cursor += seqlen_q
            continue

        sink = torch.zeros((head_kv, seqlen_q), dtype=torch.int32, device=device)
        q2k[:, q_cursor : q_cursor + seqlen_q, 0] = torch.where(
            budget.view(1, -1) > 0,
            sink,
            torch.full_like(sink, -1),
        )
        if topk > 1:
            current = latest.view(1, -1).expand(head_kv, -1)
            q2k[:, q_cursor : q_cursor + seqlen_q, 1] = torch.where(
                budget.view(1, -1) > 1,
                current,
                torch.full_like(current, -1),
            )
        if topk > 2:
            q_hash = q_local.to(torch.int64).view(1, -1)
            n_prev = (latest.to(torch.int64) - 1).clamp(min=1).view(1, -1)
            valid_prev = (latest > 1).view(1, -1)
            for slot in range(2, topk):
                valid = (budget > slot).view(1, -1) & valid_prev
                base = torch.remainder(
                    q_hash * 1103515245 + head_hash * 12345 + slot * 2654435761,
                    2147483647,
                )
                candidate = 1 + torch.remainder(base, n_prev)
                previous = q2k[:, q_cursor : q_cursor + seqlen_q, :slot].to(torch.int64)
                for _ in range(slot):
                    duplicate = (previous == candidate.unsqueeze(-1)).any(dim=-1) & valid
                    if not duplicate.any():
                        break
                    candidate = torch.where(
                        duplicate,
                        1 + torch.remainder(candidate, n_prev),
                        candidate,
                    )
                q2k[:, q_cursor : q_cursor + seqlen_q, slot] = torch.where(
                    valid,
                    candidate.to(torch.int32),
                    torch.full_like(candidate, -1, dtype=torch.int32),
                )
        q_cursor += seqlen_q
    return q2k


def _format_q2k_fanout(q2k: torch.Tensor) -> str:
    valid = q2k[q2k >= 0]
    if valid.numel() == 0:
        return "valid=0"
    fanout = torch.bincount(valid.flatten(), minlength=int(valid.max().item()) + 1)
    fanout_f = fanout.float()
    return (
        f"valid={valid.numel()} "
        f"kv_block0={int(fanout[0].item())} "
        f"max={int(fanout.max().item())} "
        f"mean={fanout_f.mean().item():.1f} "
        f"p50={fanout_f.median().item():.1f} "
        f"sink_share={fanout[0].item() / valid.numel() * 100:.2f}% "
        f"max/mean={fanout_f.max().item() / fanout_f.mean().item():.2f}x"
    )


def _format_csr_pattern(
    k2q_row_ptr: torch.Tensor,
    *,
    k_lens: tuple[int, ...],
    blk_kv: int,
    topk: int,
) -> str:
    counts = (k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]).to(torch.float32)
    valid_rows_per_batch = tuple((k_len + blk_kv - 1) // blk_kv for k_len in k_lens)
    sink_rows = [
        batch_idx
        for batch_idx, rows in enumerate(valid_rows_per_batch)
        if rows > 0
    ]
    if counts.numel() == 0 or counts.sum().item() == 0:
        return "inferred=empty valid=0"

    total = counts.sum().item()
    sink_entries = counts[:, sink_rows].sum().item() if sink_rows else 0.0
    sink_share = sink_entries / total
    uniform_share = len(sink_rows) / counts.shape[1] if counts.shape[1] > 0 else 0.0
    sink_threshold = max(2.0 * uniform_share, 0.5 / max(topk, 1))
    inferred = Q2KPattern.SINK if sink_share >= sink_threshold else Q2KPattern.UNIFORM
    p50 = counts.flatten().median().item()
    mean = counts.mean().item()
    max_count = counts.max().item()
    return (
        f"inferred={inferred} "
        f"sink_entries={int(sink_entries)} "
        f"sink_share={sink_share * 100:.2f}% "
        f"uniform_row_share={uniform_share * 100:.2f}% "
        f"max={int(max_count)} "
        f"mean={mean:.1f} "
        f"p50={p50:.1f} "
        f"max/mean={max_count / mean:.2f}x"
    )


def _build_sparse_inputs(
    *,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    dtype: torch.dtype,
    q2k_pattern: Optional[str] = None,
) -> dict[str, torch.Tensor | float]:
    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    q2ks: list[torch.Tensor] = []

    for seqlen_q, seqlen_k in zip(q_lens, k_lens):
        if q2k_pattern is None:
            q, k, v, q2k = generate_test_data(
                batch=1,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                head_q=head_kv * qhead_per_kv,
                head_kv=head_kv,
                dim=dim,
                topk=topk,
                blk_kv=blk_kv,
                causal=causal,
                dtype=dtype,
            )
            qs.append(q.squeeze(0))
            ks.append(k.squeeze(0))
            vs.append(v.squeeze(0))
            q2ks.append(q2k.squeeze(0))
        else:
            qs.append(_randn_qkv((seqlen_q, head_kv * qhead_per_kv, dim), dtype=dtype, device="cuda"))
            ks.append(_randn_qkv((seqlen_k, head_kv, dim), dtype=dtype, device="cuda"))
            vs.append(_randn_qkv((seqlen_k, head_kv, dim), dtype=dtype, device="cuda"))

    q = torch.cat(qs, dim=0)
    k = torch.cat(ks, dim=0)
    v = torch.cat(vs, dim=0)
    q2k = (
        torch.cat(q2ks, dim=1)
        if q2k_pattern is None
        else _build_pattern_q2k(
            q_lens=q_lens,
            k_lens=k_lens,
            head_kv=head_kv,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            pattern=q2k_pattern,
        )
    )
    cu_seqlens_q = _cat_cu_seqlens(q_lens)
    cu_seqlens_k = _cat_cu_seqlens(k_lens)
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(k_lens)
    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        blk_kv,
        total_k=k.shape[0],
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=sum((length + blk_kv - 1) // blk_kv for length in k_lens),
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "q2k": q2k,
        "k2q_row_ptr": k2q_row_ptr,
        "k2q_q_indices": k2q_q_indices,
        "schedule": schedule,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "q_lens": q_lens,
        "k_lens": k_lens,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "blk_kv": blk_kv,
        "softmax_scale": 1.0 / math.sqrt(dim),
    }


def _force_empty_kv_tile(
    q2k: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    head_kv = q2k.shape[0]
    q_cursor = 0
    k_cursor = 0
    empty_blocks: list[slice] = []
    active_blocks: list[slice] = []
    for seqlen_q, seqlen_k in zip(q_lens, k_lens):
        num_kv_blocks = seqlen_k // blk_kv
        if num_kv_blocks < 2:
            raise ValueError("check_empty=True requires at least 2 KV blocks per sequence")

        slots = torch.arange(topk, device=q2k.device, dtype=torch.int32).unsqueeze(0)
        chosen = torch.where(slots == 0, torch.zeros_like(slots), slots + 1)
        if causal:
            visible_blocks = _bottom_right_causal_visible_blocks(
                torch.arange(seqlen_q, device=q2k.device, dtype=torch.int32),
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                blk_kv=blk_kv,
            )
        else:
            visible_blocks = torch.full(
                (seqlen_q,),
                num_kv_blocks,
                device=q2k.device,
                dtype=torch.int32,
            )
        valid = chosen < visible_blocks.unsqueeze(1)
        q2k_seq = torch.where(
            valid.unsqueeze(0),
            chosen.unsqueeze(0).expand(head_kv, -1, -1),
            torch.full((head_kv, seqlen_q, topk), -1, dtype=torch.int32, device=q2k.device),
        )

        q2k[:, q_cursor : q_cursor + seqlen_q] = q2k_seq
        active_blocks.append(slice(k_cursor, k_cursor + blk_kv))
        empty_blocks.append(slice(k_cursor + blk_kv, k_cursor + 2 * blk_kv))
        q_cursor += seqlen_q
        k_cursor += seqlen_k
    return tuple(empty_blocks), tuple(active_blocks)


def _build_paged_inputs(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    page_size: int,
    seqused_trim: int,
    dtype: torch.dtype,
    page_table_mode: str = "shuffle",
    q2k_pattern: Optional[str] = None,
) -> dict[str, torch.Tensor | float]:
    q_lens = _sample_q_lens(batch, seqlen_q, seed=29)
    k_lens = _sample_k_lens(batch, seqlen_kv, blk_kv=blk_kv, seed=31)

    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    q2ks: list[torch.Tensor] = []

    for q_len, k_len in zip(q_lens, k_lens):
        if q2k_pattern is None:
            q, k, v, q2k = generate_test_data(
                batch=1,
                seqlen_q=q_len,
                seqlen_k=k_len,
                head_q=head_kv * qhead_per_kv,
                head_kv=head_kv,
                dim=dim,
                topk=topk,
                blk_kv=blk_kv,
                causal=causal,
                dtype=dtype,
            )
            qs.append(q.squeeze(0))
            ks.append(k.squeeze(0))
            vs.append(v.squeeze(0))
            q2ks.append(q2k.squeeze(0))
        else:
            qs.append(_randn_qkv((q_len, head_kv * qhead_per_kv, dim), dtype=dtype, device="cuda"))
            ks.append(_randn_qkv((k_len, head_kv, dim), dtype=dtype, device="cuda"))
            vs.append(_randn_qkv((k_len, head_kv, dim), dtype=dtype, device="cuda"))

    q = torch.cat(qs, dim=0)
    k_ref = torch.cat(ks, dim=0)
    v_ref = torch.cat(vs, dim=0)
    q2k = (
        torch.cat(q2ks, dim=1)
        if q2k_pattern is None
        else _build_pattern_q2k(
            q_lens=q_lens,
            k_lens=k_lens,
            head_kv=head_kv,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            pattern=q2k_pattern,
        )
    )
    cu_seqlens_q = _cat_cu_seqlens(q_lens)

    k_dense = torch.zeros(batch, seqlen_kv, head_kv, dim, dtype=dtype, device=q.device)
    v_dense = torch.zeros_like(k_dense)
    for batch_idx, (k_seq, v_seq, k_len) in enumerate(zip(ks, vs, k_lens)):
        k_dense[batch_idx, :k_len].copy_(k_seq)
        v_dense[batch_idx, :k_len].copy_(v_seq)

    k_paged, v_paged, page_table = pack_paged_kv(
        k_dense,
        v_dense,
        page_size=page_size,
        page_table_mode=page_table_mode,
        max_pages_per_seq=(seqlen_kv + page_size - 1) // page_size,
    )

    effective_k_lens = tuple(
        max(1, k_len - min(seqused_trim, k_len - 1))
        for k_len in k_lens
    )
    q2k_csr = q2k
    if effective_k_lens != k_lens:
        q2k_csr = q2k.clone()
        q_cursor = 0
        for q_len, effective_k_len in zip(q_lens, effective_k_lens):
            max_effective_block = (effective_k_len + blk_kv - 1) // blk_kv
            q_slice = slice(q_cursor, q_cursor + q_len)
            invalid = q2k_csr[:, q_slice, :] >= max_effective_block
            q2k_csr[:, q_slice, :] = torch.where(
                invalid,
                torch.full_like(q2k_csr[:, q_slice, :], -1),
                q2k_csr[:, q_slice, :],
            )
            q_cursor += q_len
    cu_seqlens_k = _cat_cu_seqlens(effective_k_lens)
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(effective_k_lens)
    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
        q2k_csr,
        cu_seqlens_q,
        cu_seqlens_k,
        blk_kv,
        total_k=sum(effective_k_lens),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=sum(
            (length + blk_kv - 1) // blk_kv for length in effective_k_lens
        ),
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    paged_capacity = int(page_table.shape[1]) * page_size
    seqused_k = None
    if any(k_len != paged_capacity for k_len in effective_k_lens):
        seqused_k = torch.tensor(
            effective_k_lens,
            device=q.device,
            dtype=torch.int32,
        )
    return {
        "q": q,
        "k_paged": k_paged,
        "v_paged": v_paged,
        "k_ref": k_ref,
        "v_ref": v_ref,
        "q2k": q2k,
        "k2q_row_ptr": k2q_row_ptr,
        "k2q_q_indices": k2q_q_indices,
        "schedule": schedule,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "q_lens": q_lens,
        "k_lens": k_lens,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "blk_kv": blk_kv,
        "softmax_scale": 1.0 / math.sqrt(dim),
    }


def _build_decode_paged_dense_inputs(
    *,
    kv_tokens: int,
    batch: int = DECODE_BATCH,
    seqlen_q: int = DECODE_SEQLEN_Q,
    head_kv: int = DECODE_HEAD_KV,
    qhead_per_kv: int = DECODE_QHEAD_PER_KV,
    dim: int = DECODE_DIM,
    blk_kv: int = BLK_KV,
    page_size: int = BLK_KV,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> dict[str, object]:
    if kv_tokens < 1:
        raise ValueError("kv_tokens must be >= 1")
    if page_size != blk_kv:
        raise ValueError("decode fp8 page test requires page_size == blk_kv")
    if dtype != torch.float8_e4m3fn:
        raise ValueError("decode fp8 page test requires torch.float8_e4m3fn")
    if qhead_per_kv != DECODE_QHEAD_PER_KV:
        raise ValueError("decode target only supports qhead_per_kv=16")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if seqlen_q < 1:
        raise ValueError("seqlen_q must be >= 1")

    head_q = head_kv * qhead_per_kv
    total_q = batch * seqlen_q
    page_count = (kv_tokens + page_size - 1) // page_size
    physical_kv_tokens = page_count * page_size
    device = "cuda"

    q = _randn_qkv((total_q, head_q, dim), dtype=dtype, device=device)
    k_pages = _randn_qkv(
        (batch, page_count, head_kv, page_size, dim),
        dtype=dtype,
        device=device,
    )
    v_pages = _randn_qkv(
        (batch, page_count, head_kv, page_size, dim),
        dtype=dtype,
        device=device,
    )
    tail_tokens = kv_tokens - (page_count - 1) * page_size
    if tail_tokens < page_size:
        k_pages[:, -1, :, tail_tokens:, :].zero_()
        v_pages[:, -1, :, tail_tokens:, :].zero_()

    k_paged = k_pages.reshape(batch * page_count, head_kv, page_size, dim).contiguous()
    v_paged = v_pages.reshape(batch * page_count, head_kv, page_size, dim).contiguous()
    page_table = torch.arange(
        batch * page_count,
        device=device,
        dtype=torch.int32,
    ).view(batch, page_count)
    physical_k_lens = (physical_kv_tokens,) * batch
    seqused_k = torch.full((batch,), kv_tokens, dtype=torch.int32, device=device)

    result: dict[str, object] = {
        "q": q,
        "k_paged": k_paged,
        "v_paged": v_paged,
        "q2k": None,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "k_lens": physical_k_lens,
        "seqlen_q": seqlen_q,
        "kv_tokens": kv_tokens,
        "physical_kv_tokens": physical_kv_tokens,
        "max_seqlen_k": kv_tokens,
        "topk": page_count,
        "blk_kv": blk_kv,
        "softmax_scale": 1.0 / math.sqrt(dim),
    }
    return result


def _run_sparse_decode_page_fp8(
    wrapper,
    inputs: dict[str, object],
    *,
    skip_not_implemented: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        wrapper.plan(
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            seqlen_q=inputs["seqlen_q"],
            max_seqlen_k=inputs["max_seqlen_k"],
            q2k_indices=inputs["q2k"],
            num_qo_heads=inputs["q"].shape[1],
            num_kv_heads=inputs["k_paged"].shape[1],
            head_dim=inputs["q"].shape[2],
        )
        result = wrapper.run(
            inputs["q"],
            inputs["k_paged"],
            inputs["v_paged"],
            softmax_scale=inputs["softmax_scale"],
            return_softmax_lse=True,
        )
    except NotImplementedError as exc:
        if skip_not_implemented:
            pytest.skip(str(exc))
        raise
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("sparse_decode_atten_func must return (out, lse)")
    return result


def _decode_paged_dense_reference(
    inputs: dict[str, object],
    *,
    chunk_tokens: int = 8192,
    p_dtype: torch.dtype | None = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode attention.

    ``p_dtype`` selects the rounding applied to unnormalized P.  The default
    (e4m3) models the kernel.  ``p_dtype=None`` skips that rounding entirely
    and yields the same reference with its own P-quantization noise removed;
    ``_assert_fp8_forward_close`` uses the difference of the two as a measured
    noise floor.  See the derivation in that function.
    """
    q = inputs["q"].float()
    k_paged = inputs["k_paged"]
    v_paged = inputs["v_paged"]
    page_table = inputs["page_table"]
    kv_tokens = int(inputs["kv_tokens"])
    softmax_scale = float(inputs["softmax_scale"])
    page_size = int(inputs["blk_kv"])
    seqlen_q = int(inputs["seqlen_q"])
    batch = int(page_table.shape[0])
    head_kv = int(k_paged.shape[1])
    qhead_per_kv = q.shape[1] // head_kv
    dim = int(q.shape[2])
    chunk_tokens = max(page_size, (int(chunk_tokens) // page_size) * page_size)

    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)

    for batch_idx in range(batch):
        q_begin = batch_idx * seqlen_q
        q_end = q_begin + seqlen_q
        q_batch = q[q_begin:q_end]
        q_pos = torch.arange(seqlen_q, device=q.device, dtype=torch.int64)
        causal_limit = q_pos + (kv_tokens - seqlen_q)
        for head_kv_idx in range(head_kv):
            h_begin = head_kv_idx * qhead_per_kv
            h_end = h_begin + qhead_per_kv
            q_cur = q_batch[:, h_begin:h_end, :]
            row_max = torch.full(
                (qhead_per_kv, seqlen_q),
                -float("inf"),
                dtype=torch.float32,
                device=q.device,
            )
            row_sum = torch.zeros_like(row_max)
            for start in range(0, kv_tokens, chunk_tokens):
                end = min(start + chunk_tokens, kv_tokens)
                k_chunk = _decode_paged_kv_chunk(
                    k_paged,
                    page_table,
                    batch_idx=batch_idx,
                    head_kv_idx=head_kv_idx,
                    start=start,
                    end=end,
                    page_size=page_size,
                ).float()
                scores = torch.einsum("shd,td->hst", q_cur * softmax_scale, k_chunk)
                kv_pos = torch.arange(start, end, device=q.device, dtype=torch.int64)
                scores = scores.masked_fill(
                    kv_pos.view(1, 1, -1) > causal_limit.view(1, -1, 1),
                    -float("inf"),
                )
                chunk_max = scores.max(dim=-1).values
                new_max = torch.maximum(row_max, chunk_max)
                old_scale = torch.exp(row_max - new_max)
                old_scale = torch.where(torch.isfinite(row_max), old_scale, torch.zeros_like(old_scale))
                exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
                exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
                row_sum = row_sum * old_scale + exp_scores.sum(dim=-1)
                row_max = new_max

            out_cur = torch.zeros(
                (seqlen_q, qhead_per_kv, dim),
                dtype=torch.float32,
                device=q.device,
            )
            safe_row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
            for start in range(0, kv_tokens, chunk_tokens):
                end = min(start + chunk_tokens, kv_tokens)
                k_chunk = _decode_paged_kv_chunk(
                    k_paged,
                    page_table,
                    batch_idx=batch_idx,
                    head_kv_idx=head_kv_idx,
                    start=start,
                    end=end,
                    page_size=page_size,
                ).float()
                v_chunk = _decode_paged_kv_chunk(
                    v_paged,
                    page_table,
                    batch_idx=batch_idx,
                    head_kv_idx=head_kv_idx,
                    start=start,
                    end=end,
                    page_size=page_size,
                ).float()
                scores = torch.einsum("shd,td->hst", q_cur * softmax_scale, k_chunk)
                kv_pos = torch.arange(start, end, device=q.device, dtype=torch.int64)
                valid = kv_pos.view(1, 1, -1) <= causal_limit.view(1, -1, 1)
                exp_scores = torch.exp(scores - row_max.unsqueeze(-1))
                exp_scores = torch.where(
                    valid & (row_sum.unsqueeze(-1) > 0),
                    exp_scores,
                    torch.zeros_like(exp_scores),
                )
                # The SM100 fp8 path quantizes unnormalized P to e4m3 before
                # PV, then applies 1 / row_sum in the epilogue.  Quantizing
                # normalized probabilities here would model a different
                # rounding point and overstate the kernel error for short-KV
                # decode rows.
                p_unnorm = (
                    exp_scores
                    if p_dtype is None
                    else exp_scores.to(p_dtype).float()
                )
                out_cur += torch.einsum("hst,td->shd", p_unnorm, v_chunk)
            out_cur = out_cur / safe_row_sum.transpose(0, 1).unsqueeze(-1)

            out[q_begin:q_end, h_begin:h_end, :] = out_cur
            lse_cur = torch.where(
                row_sum > 0,
                row_max + torch.log(row_sum),
                torch.full_like(row_sum, -float("inf")),
            )
            lse[q_begin:q_end, h_begin:h_end] = lse_cur.transpose(0, 1)
    return out, lse


def _decode_paged_kv_chunk(
    paged: torch.Tensor,
    page_table: torch.Tensor,
    *,
    batch_idx: int,
    head_kv_idx: int,
    start: int,
    end: int,
    page_size: int,
) -> torch.Tensor:
    start_page = start // page_size
    end_page = (end + page_size - 1) // page_size
    physical_pages = page_table[batch_idx, start_page:end_page].long()
    chunk = paged[physical_pages, head_kv_idx].reshape(-1, paged.shape[-1])
    offset = start - start_page * page_size
    return chunk[offset : offset + (end - start)]


def _reference_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    *,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    upcast: bool,
    lse_temperature_scale: Optional[float] = None,
    seqused_k: Optional[torch.Tensor] = None,
    p_dtype: Optional[torch.dtype] = None,
    o_partial_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_chunks = []
    lse_chunks = []
    lse_temperature_chunks = []
    q_cursor = 0
    k_cursor = 0
    for batch_idx, (seqlen_q, seqlen_k) in enumerate(zip(q_lens, k_lens)):
        seq_used_k = None if seqused_k is None else seqused_k[batch_idx : batch_idx + 1]
        ref_result = sparse_attention_ref(
            q[q_cursor : q_cursor + seqlen_q].unsqueeze(0),
            k[k_cursor : k_cursor + seqlen_k].unsqueeze(0),
            v[k_cursor : k_cursor + seqlen_k].unsqueeze(0),
            q2k[:, q_cursor : q_cursor + seqlen_q].unsqueeze(0),
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            lse_temperature_scale=lse_temperature_scale,
            upcast=upcast,
            seqused_k=seq_used_k,
            p_dtype=p_dtype,
            o_partial_dtype=o_partial_dtype,
        )
        if lse_temperature_scale is None:
            out_b, lse_b = ref_result
        else:
            out_b, lse_b, lse_temperature_b = ref_result
            lse_temperature_chunks.append(lse_temperature_b.squeeze(0))
        out_chunks.append(out_b.squeeze(0))
        lse_chunks.append(lse_b.squeeze(0))
        q_cursor += seqlen_q
        k_cursor += seqlen_k
    if lse_temperature_scale is not None:
        return (
            torch.cat(out_chunks, dim=0),
            torch.cat(lse_chunks, dim=0),
            torch.cat(lse_temperature_chunks, dim=0),
        )
    return torch.cat(out_chunks, dim=0), torch.cat(lse_chunks, dim=0)


def _assert_forward_close(
    out: torch.Tensor,
    out_ref: torch.Tensor,
    out_pt: torch.Tensor,
    lse: torch.Tensor,
    lse_ref: torch.Tensor,
) -> None:
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    pt_diff = (out_pt - out_ref).abs().max().item()
    kernel_diff = (out.float() - out_ref).abs().max().item()
    assert kernel_diff <= 2 * pt_diff + fwd_atol

    finite = lse_ref.isfinite()
    if finite.any():
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)


def _assert_fp8_forward_close(
    out: torch.Tensor,
    out_ref: torch.Tensor,
    lse: torch.Tensor,
    lse_ref: torch.Tensor,
    partial_is_fp8: bool = False,
    out_ref_unquantized_p: torch.Tensor | None = None,
) -> None:
    out_ref_f = out_ref.float()
    out_ref_bf16 = out_ref.to(torch.bfloat16).float()
    bf16_err = out_ref_bf16 - out_ref_f
    bf16_floor = bf16_err.abs().max().item()
    bf16_rms_floor = bf16_err.pow(2).mean().sqrt().item()
    diff = out.float() - out_ref_bf16
    kernel_diff = diff.abs().max().item()
    diff_rms = diff.pow(2).mean().sqrt().item()
    out_scale = out_ref_bf16.abs().max().item()
    out_rms = out_ref_bf16.pow(2).mean().sqrt().item()
    # QK/LSE should be tight. O additionally includes P(e4m3) @ V(e4m3) with
    # hardware FP8 MMA accumulation, so it needs a slacker bound -- but that
    # bound has to be RELATIVE to |O|. It used to be the bare constant 0.125
    # (0.5 with fp8 O_partial), which is not scale-invariant: it silently gets
    # stricter as the data grows. Making the synthetic NVFP4 block scales vary
    # over [0.5, 1.875] instead of being pinned at 1.0 raised max|O| by 1.75x
    # and raised |diff| by the same factor at *constant* relative error
    # (measured 1.4-1.9% of max|O| in every scale window from constant-0x38 to
    # 0x30-0x3E), and that alone flipped the test to failing. A relative bound
    # would not have moved.
    #
    # Error model for the constant. P is rounded to E4M3, which carries 3
    # explicit mantissa bits, so its round-to-nearest relative error is at most
    # 2**-4. The kernel and the reference arrive at P by different roundings,
    # so any P entry may land one E4M3 step away; O = sum_j P_j V_j / sum_j P_j
    # is a convex combination of V rows, so that carries into O as the same
    # relative perturbation. With fp8 O_partial every split additionally
    # re-quantizes the partial output to E4M3 before the combine, which
    # empirically costs ~4x more (measured max 12.4% vs 4.3% relative) -- the
    # same 4x the two old constants encoded, now expressed relatively.
    #
    # Calibration: 561 measurements over the fp8 and NVFP4 test set give a
    # worst observed relative error of 4.35% (441 bf16-O_partial cases) and
    # 12.38% (120 fp8-O_partial cases), i.e. 1.4x and 2.0x of headroom below
    # the bounds used here. The 4 * bf16_floor term is the reference's own
    # bf16 storage floor; bf16 keeps 8 significand bits, so it is bounded by
    # 4 * 2**-8 * max|O| = 1.6% * max|O| and is scale-invariant too. The
    # headroom figures above count only the rtol term, so they are lower
    # bounds on the real margin.
    #
    # The rtol term above is *only* valid while the reference's own P entries
    # stay inside E4M3's normal range.  They do not, for long-KV decode.  The
    # reference normalizes P by the GLOBAL row max, so p_j = exp(s_j - max_j
    # s_j) covers (0, 1]; as the KV length N grows the row max grows like
    # sqrt(2 ln N) and an ever larger share of p_j falls below E4M3's smallest
    # normal 2**-6, where the rounding error is an ABSOLUTE half-ulp of 2**-10
    # rather than a relative 2**-4.  Measured on the decode sweep, the
    # subnormal share of the reference's P goes 0.9% (N=8) -> 34% (N=8192) ->
    # 74% (N=1048576).  Meanwhile max|O| shrinks like 1/sqrt(N) (O is a convex
    # combination of random V rows), so that absolute noise shows up as a
    # relative error that climbs with N -- entirely on the reference's side.
    #
    # The kernel does not have this problem: it normalizes P against a
    # per-tile running max (`rescale_threshold = 4.0` in
    # src/sm100/fwd_decode/atten_fwd.py), never against a 1M-token global max.
    # Measured against the same reference evaluated with an *unquantized* P,
    # rms|kernel - exact| / rms|exact| is FLAT at 2.7% over N = 8192 ..
    # 1048576, while rms|ref - exact| / rms|exact| climbs 2.68% -> 3.79% and
    # max|ref - exact| / max|O| climbs 3.97% -> 12.74%.  At N = 1048576 the
    # kernel is closer to the unquantized answer (6.9%) than the reference is
    # (12.7%).  A bound that charges the reference's own quantization noise to
    # the kernel is mis-modelled, not a kernel defect.
    #
    # So: when the caller can supply the same reference computed without the
    # E4M3 P rounding, subtract the two to get that noise directly and admit
    # it into the tolerance.  This is measured from the *inputs*, not from the
    # kernel, so it cannot be inflated by a broken kernel.  Callers that do
    # not supply it (all prefill/NVFP4 sites) keep exactly the previous bound.
    e4m3_relative_ulp = 2.0**-4
    rtol = 4.0 * e4m3_relative_ulp if partial_is_fp8 else e4m3_relative_ulp
    if out_ref_unquantized_p is None:
        ref_noise_max = 0.0
        ref_noise_rms = 0.0
    else:
        ref_noise = out_ref_f - out_ref_unquantized_p.float()
        ref_noise_max = ref_noise.abs().max().item()
        ref_noise_rms = ref_noise.pow(2).mean().sqrt().item()

    # Triangle inequality through the unquantized-P reference:
    #   |out - out_ref| <= |out - exact| + |exact - out_ref|.
    # The second leg is ref_noise.  The first leg is the kernel's own E4M3 P
    # rounding, which is of the same class and (per the measurements above)
    # never coarser, so it is charged the same quantity -- hence the factor 2.
    tol = 4.0 * bf16_floor + rtol * out_scale + 2.0 * ref_noise_max
    assert kernel_diff <= tol, (
        f"max|out - out_ref| = {kernel_diff:.6g} exceeds {tol:.6g} "
        f"(= 4 * bf16_floor {bf16_floor:.6g} + rtol {rtol:.6g} "
        f"* max|out_ref| {out_scale:.6g} + 2 * ref_quant_noise "
        f"{ref_noise_max:.6g}); relative error "
        f"{kernel_diff / max(out_scale, 1e-30):.4%}"
    )

    # The same model evaluated in RMS instead of at the single worst element.
    # rms <= max elementwise, so the same relative constants are conservative
    # here; this is the coarse net for "the kernel got noisier everywhere",
    # which a max over millions of elements reports only weakly.
    rms_tol = 4.0 * bf16_rms_floor + rtol * out_rms + 2.0 * ref_noise_rms
    assert diff_rms <= rms_tol, (
        f"rms(out - out_ref) = {diff_rms:.6g} exceeds {rms_tol:.6g} "
        f"(= 4 * bf16_rms_floor {bf16_rms_floor:.6g} + rtol {rtol:.6g} "
        f"* rms|out_ref| {out_rms:.6g} + 2 * ref_quant_noise_rms "
        f"{ref_noise_rms:.6g}); ratio {diff_rms / max(out_rms, 1e-30):.4%}"
    )

    # Both bounds above are set by the quantization NOISE, which is a few
    # percent for any E4M3 attention kernel and grows with N on the reference
    # side.  On their own they would accept a several-percent systematic
    # corruption at large N, i.e. they would be close to vacuous exactly where
    # the old absolute constant already was.  The two projections below are
    # what keep the criterion sharp: they are insensitive to zero-mean noise
    # (it averages down by sqrt(n_elem)) but respond linearly to a systematic
    # error.
    #
    #   gain_err = <out - out_ref, out_ref> / <out_ref, out_ref>
    #
    # is exactly eps for a corruption out -> (1 + eps) * out.  For a correct
    # kernel it is not zero, because round-to-nearest on the E4M3 grid is only
    # unbiased to FIRST order in the relative ulp; writing the residual as
    # sum_j e_j p_j / sum_j p_j^2 with |e_j| <= rtol * p_j, the surviving
    # systematic part is second order, O(rtol**2).  Measured: the kernel's own
    # projection (against the unquantized-P reference) is flat at -6.2e-4 ..
    # -7.2e-4 across N = 1024 .. 1048576 = 0.16 .. 0.18 * rtol**2, and over
    # 141 prefill/NVFP4 measurements |gain_err| <= 9.7e-5 = 0.025 * rtol**2.
    # The reference's own projection is what grows with N (+4.6e-5 at N=1024
    # -> +9.1e-4 at N=1048576), so it enters through ref_noise like the terms
    # above.  The 8 * (statistical null sd) term covers small tensors, where
    # the projection of a genuinely zero-mean residual fluctuates.
    n_elem = diff.numel()
    ref_energy = (out_ref_bf16 * out_ref_bf16).sum().item()
    gain_err = (diff * out_ref_bf16).sum().item() / ref_energy if ref_energy > 0 else 0.0
    gain_null_sd = diff_rms / (math.sqrt(n_elem) * out_rms) if out_rms > 0 else 0.0
    ref_gain_noise = 0.0
    if out_ref_unquantized_p is not None and ref_energy > 0:
        ref_gain_noise = abs(
            (ref_noise * out_ref_bf16).sum().item() / ref_energy
        )
    gain_tol = rtol * rtol + 2.0 * ref_gain_noise + 8.0 * gain_null_sd
    assert abs(gain_err) <= gain_tol, (
        f"<out - out_ref, out_ref> / <out_ref, out_ref> = {gain_err:.6g} "
        f"exceeds {gain_tol:.6g} (= rtol**2 {rtol * rtol:.6g} + 2 * "
        f"ref_gain_noise {ref_gain_noise:.6g} + 8 * null_sd "
        f"{gain_null_sd:.6g}); the kernel output is systematically scaled"
    )

    # Companion projection onto the constant vector: catches an additive
    # offset, which the gain statistic cannot see when out_ref is zero-mean.
    # Its null is diff_rms / sqrt(n_elem); over the 160 fp8/NVFP4 measurements
    # the observed |mean| never exceeded 2.38 of those sd, so 8 is >3x the
    # worst observation.  The gain_tol * |mean(out_ref)| term keeps the two
    # checks from double-counting when out_ref does have a nonzero mean.
    mean_err = diff.mean().item()
    mean_ref = out_ref_bf16.mean().item()
    mean_tol = 8.0 * diff_rms / math.sqrt(n_elem) + gain_tol * abs(mean_ref)
    assert abs(mean_err) <= mean_tol, (
        f"mean(out - out_ref) = {mean_err:.6g} exceeds {mean_tol:.6g} "
        f"(= 8 * null_sd {diff_rms / math.sqrt(n_elem):.6g} + gain_tol "
        f"{gain_tol:.6g} * |mean(out_ref)| {abs(mean_ref):.6g}); the kernel "
        f"output carries a systematic offset"
    )

    finite = lse_ref.isfinite()
    if finite.any():
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# Pytest coverage
# ---------------------------------------------------------------------------

def test_scheduler_builds_packed_qsplit_metadata() -> None:
    seqlen_q = 256
    seqlen_k = 512
    blk_kv = 128
    topk = 4
    q2k = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk)
    q2k = q2k.expand(1, seqlen_q, topk).contiguous()
    cu_seqlens_q = torch.tensor([0, seqlen_q], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_k], device="cuda", dtype=torch.int32)
    k2q_row_ptr, k2q_q_indices = build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        blk_kv,
        total_k=seqlen_k,
        max_seqlen_k=seqlen_k,
        total_rows=seqlen_k // blk_kv,
    )
    k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
    split_counts = torch.empty((seqlen_q, 1), dtype=torch.int32, device=q2k.device)
    schedule = prepare_sparse_fwd_schedule_and_split(
        k2q_row_ptr=k2q_row_ptr,
        k2q_q_indices=k2q_q_indices,
        k2q_qsplit_indices=k2q_qsplit_indices,
        split_counts=split_counts,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        total_q=seqlen_q,
        max_seqlen_q=seqlen_q,
        topk=topk,
        head_kv=1,
        qhead_per_kv=1,
        blk_kv=blk_kv,
        device=q2k.device,
        enabled=True,
    )
    assert schedule.enabled
    torch.cuda.synchronize()
    work_count = int(schedule.work_count.item())
    metadata = schedule.scheduler_metadata[:work_count].cpu()
    assert metadata.shape[1] == 6
    assert int(metadata[:, 3].max().item()) <= schedule.target_q_per_cta
    assert int(metadata[:, 3].sum().item()) == int(k2q_row_ptr[0, -1].item())
    assert torch.equal(metadata[:, 4], torch.zeros_like(metadata[:, 4]))
    assert set(metadata[:, 5].tolist()) == set(range(seqlen_k // blk_kv))

    seen: dict[int, list[int]] = {q: [] for q in range(seqlen_q)}
    nnz = int(k2q_row_ptr[0, -1].item())
    for offset in range(nnz):
        packed = int(k2q_qsplit_indices[0, offset].item()) & 0xFFFF_FFFF
        q_idx = packed & 0x00FF_FFFF
        split_idx = (packed >> 24) & 0xFF
        assert q_idx == int(k2q_q_indices[0, offset].item())
        seen[q_idx].append(split_idx)

    for q_idx, slots in seen.items():
        assert sorted(slots) == list(range(int(split_counts[q_idx, 0].item())))


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("topk", [4, 8, 16, 32])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_k2q_csr_builder_matches_torch_reference(
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_kv: int,
    topk: int,
    causal: bool,
) -> None:
    q_lens = _sample_q_lens(batch, seqlen_q, seed=17)
    k_lens = _sample_k_lens(batch, seqlen_kv, blk_kv=BLK_KV, seed=23)
    k_lens = tuple(max(length, topk * BLK_KV) for length in k_lens)
    inputs = _build_sparse_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        head_kv=head_kv,
        qhead_per_kv=1,
        dim=128,
        topk=topk,
        blk_kv=BLK_KV,
        causal=causal,
        dtype=torch.bfloat16,
    )
    ref_row_ptr, ref_q_indices = build_k2q_csr_torch_reference(
        inputs["q2k"],
        inputs["cu_seqlens_q"],
        inputs["cu_seqlens_k"],
        BLK_KV,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(inputs["k2q_row_ptr"], ref_row_ptr, atol=0, rtol=0)
    torch.testing.assert_close(inputs["k2q_q_indices"], ref_q_indices, atol=0, rtol=0)


@pytest.mark.parametrize("check_empty", [False, True])
@pytest.mark.parametrize("lse_temperature_scale", [None, 2.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_sparse_atten(
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    topk: int,
    causal: bool,
    dtype: torch.dtype,
    lse_temperature_scale: Optional[float],
    check_empty: bool,
) -> None:
    torch.random.manual_seed(0)
    q_lens = _sample_q_lens(batch, seqlen_q, seed=17)
    k_lens = _sample_k_lens(batch, seqlen_kv, blk_kv=BLK_KV, seed=23)
    k_min_len = topk * BLK_KV
    if check_empty:
        k_min_len = max(k_min_len, 2 * BLK_KV)
    k_lens = tuple(max(length, k_min_len) for length in k_lens)
    inputs = _build_sparse_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        dim=128,
        topk=topk,
        blk_kv=BLK_KV,
        causal=causal,
        dtype=dtype,
    )

    q_src = inputs["q"]
    k_src = inputs["k"]
    v_src = inputs["v"]
    q2k = inputs["q2k"]
    k2q_row_ptr = inputs["k2q_row_ptr"]
    k2q_q_indices = inputs["k2q_q_indices"]
    schedule = inputs["schedule"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    blk_kv = inputs["blk_kv"]
    softmax_scale = inputs["softmax_scale"]
    empty_blocks: tuple[slice, ...] = ()
    active_blocks: tuple[slice, ...] = ()
    partial_dtype = torch.float32
    if dtype == torch.float8_e4m3fn:
        partial_dtype = {
            None: torch.float32,
            1.0: torch.bfloat16,
            2.0: torch.float8_e4m3fn,
        }[lse_temperature_scale]
    if check_empty:
        q2k = q2k.clone()
        empty_blocks, active_blocks = _force_empty_kv_tile(
            q2k,
            q_lens,
            k_lens,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
        )
        k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            total_k=sum(k_lens),
            max_seqlen_k=max(k_lens),
            max_seqlen_q=max(q_lens),
            total_rows=sum((length + blk_kv - 1) // blk_kv for length in k_lens),
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )

    q = q_src.detach().clone()
    k = k_src.detach().clone()
    v = v_src.detach().clone()
    return_temperature_lse = lse_temperature_scale is not None
    result = sparse_atten_func(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        lse_temperature_scale=1.0 if lse_temperature_scale is None else lse_temperature_scale,
        return_temperature_lse=return_temperature_lse,
        partial_dtype=partial_dtype,
        return_softmax_lse=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        schedule=schedule,
    )
    if return_temperature_lse:
        out, lse, lse_temperature_out = result
    else:
        out, lse = result
        lse_temperature_out = None

    q_ref = q_src.detach().clone().float()
    k_ref = k_src.detach().clone().float()
    v_ref = v_src.detach().clone().float()
    ref_result = _reference_varlen_forward(
        q_ref,
        k_ref,
        v_ref,
        q2k,
        q_lens=q_lens,
        k_lens=k_lens,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        upcast=True,
        lse_temperature_scale=lse_temperature_scale,
        p_dtype=torch.float8_e4m3fn if dtype == torch.float8_e4m3fn else None,
        o_partial_dtype=partial_dtype if partial_dtype is torch.float8_e4m3fn else None,
    )
    if return_temperature_lse:
        out_ref, lse_ref, lse_temperature_ref = ref_result
    else:
        out_ref, lse_ref = ref_result

    if dtype == torch.float8_e4m3fn:
        assert out.dtype == torch.bfloat16
        _assert_fp8_forward_close(
            out, out_ref, lse, lse_ref,
            partial_is_fp8=partial_dtype is torch.float8_e4m3fn,
        )
    else:
        q_pt = q_src.detach().clone()
        k_pt = k_src.detach().clone()
        v_pt = v_src.detach().clone()
        out_pt, _ = _reference_varlen_forward(
            q_pt,
            k_pt,
            v_pt,
            q2k,
            q_lens=q_lens,
            k_lens=k_lens,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            upcast=False,
        )
        _assert_forward_close(out, out_ref, out_pt.float(), lse, lse_ref)
    if return_temperature_lse:
        assert lse_temperature_out is not None
        if math.isclose(
            float(lse_temperature_scale),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            assert lse_temperature_out.data_ptr() == lse.data_ptr()
            torch.testing.assert_close(lse_temperature_out, lse, atol=0.0, rtol=0.0)
        else:
            finite = lse_temperature_ref.isfinite()
            torch.testing.assert_close(
                lse_temperature_out[finite],
                lse_temperature_ref[finite],
                atol=2e-3 if dtype == torch.float8_e4m3fn else 1e-3,
                rtol=2e-3 if dtype == torch.float8_e4m3fn else 1e-3,
            )



@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [4, 8, 16])
@pytest.mark.parametrize("page_size", [128])
@pytest.mark.parametrize("seqused_trim", [0])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_sparse_page_atten(
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    topk: int,
    page_size: int,
    seqused_trim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    torch.random.manual_seed(0)
    inputs = _build_paged_inputs(
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        dim=128,
        topk=topk,
        blk_kv=BLK_KV,
        causal=causal,
        page_size=page_size,
        seqused_trim=seqused_trim,
        dtype=dtype,
    )

    q_src = inputs["q"]
    k_paged_src = inputs["k_paged"]
    v_paged_src = inputs["v_paged"]
    k_ref = inputs["k_ref"]
    v_ref = inputs["v_ref"]
    q2k = inputs["q2k"]
    k2q_row_ptr = inputs["k2q_row_ptr"]
    k2q_q_indices = inputs["k2q_q_indices"]
    schedule = inputs["schedule"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    q_lens = inputs["q_lens"]
    k_lens = inputs["k_lens"]
    page_table = inputs["page_table"]
    seqused_k = inputs["seqused_k"]
    blk_kv = inputs["blk_kv"]
    softmax_scale = inputs["softmax_scale"]

    q = q_src.detach().clone()
    k_paged = k_paged_src.detach().clone()
    v_paged = v_paged_src.detach().clone()
    out, lse = sparse_atten_func(
        q,
        k_paged,
        v_paged,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=page_table,
        seqused_k=seqused_k,
        schedule=schedule,
    )

    out_ref, lse_ref = _reference_varlen_forward(
        q_src.float(),
        k_ref.float(),
        v_ref.float(),
        q2k,
        q_lens=q_lens,
        k_lens=k_lens,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        upcast=True,
        seqused_k=seqused_k,
        p_dtype=torch.float8_e4m3fn if dtype == torch.float8_e4m3fn else None,
    )
    if dtype == torch.float8_e4m3fn:
        assert out.dtype == torch.bfloat16
        _assert_fp8_forward_close(out, out_ref, lse, lse_ref)
        return

    out_pt, _ = _reference_varlen_forward(
        q_src,
        k_ref,
        v_ref,
        q2k,
        q_lens=q_lens,
        k_lens=k_lens,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        upcast=False,
        seqused_k=seqused_k,
    )

    _assert_forward_close(out, out_ref, out_pt.float(), lse, lse_ref)


def test_sparse_page_atten_strided_outer_page_matches_packed() -> None:
    inputs = _build_paged_inputs(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        head_kv=2,
        qhead_per_kv=16,
        dim=128,
        topk=16,
        blk_kv=128,
        causal=True,
        page_size=128,
        seqused_trim=0,
        dtype=torch.float8_e4m3fn,
    )
    k_packed = inputs["k_paged"].detach().clone()
    v_packed = inputs["v_paged"].detach().clone()
    pool = torch.empty(
        k_packed.shape[0],
        4,
        *k_packed.shape[1:],
        dtype=k_packed.dtype,
        device=k_packed.device,
    )
    k_strided = pool[:, 1]
    v_strided = pool[:, 3]
    k_strided.copy_(k_packed)
    v_strided.copy_(v_packed)

    assert not k_strided.is_contiguous()
    assert not v_strided.is_contiguous()
    assert (
        sparse_interface._prepare_paged_hnd_input(k_strided, inputs["blk_kv"]).data_ptr()
        == k_strided.data_ptr()
    )
    assert (
        sparse_interface._prepare_paged_hnd_input(v_strided, inputs["blk_kv"]).data_ptr()
        == v_strided.data_ptr()
    )

    def run(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sparse_atten_func(
            inputs["q"],
            k,
            v,
            inputs["k2q_row_ptr"],
            inputs["k2q_q_indices"],
            16,
            blk_kv=inputs["blk_kv"],
            causal=True,
            softmax_scale=inputs["softmax_scale"],
            return_softmax_lse=True,
            cu_seqlens_q=inputs["cu_seqlens_q"],
            cu_seqlens_k=inputs["cu_seqlens_k"],
            max_seqlen_q=inputs["max_seqlen_q"],
            max_seqlen_k=inputs["max_seqlen_k"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            schedule=inputs["schedule"],
        )

    packed_out, packed_lse = run(k_packed, v_packed)
    strided_out, strided_lse = run(k_strided, v_strided)
    torch.testing.assert_close(strided_out, packed_out, rtol=0, atol=0)
    torch.testing.assert_close(strided_lse, packed_lse, rtol=0, atol=0)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_sparse_atten_nvfp4_kv_matches_dequantized_bf16(
    seqlen_q: int,
    seqlen_kv: int,
    topk: int,
    qhead_per_kv: int,
    head_kv: int,
    batch: int,
    causal: bool,
    paged: bool,
) -> None:
    seed = (
        1000
        + int(paged) * 100
        + seqlen_q
        + seqlen_kv
        + topk
        + qhead_per_kv
    )
    torch.random.manual_seed(seed)
    if paged:
        causal = True
        inputs = _build_paged_inputs(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            page_size=BLK_KV,
            seqused_trim=17,
            dtype=torch.bfloat16,
            page_table_mode="shuffle",
        )
        k_source = inputs["k_paged"]
        v_source = inputs["v_paged"]
    else:
        inputs = _build_sparse_inputs(
            q_lens=(seqlen_q,) * batch,
            k_lens=(seqlen_kv,) * batch,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            dtype=torch.bfloat16,
            q2k_pattern=Q2KPattern.UNIFORM,
        )
        k_source = inputs["k"]
        v_source = inputs["v"]

    k_q, v_q = _quantize_kv_bf16_to_nvfp4(k_source, v_source)
    k_deq = _dequant_nvfp4_to_bf16(k_q)
    v_deq = _dequant_nvfp4_to_bf16(v_q)

    out_ref, lse_ref = sparse_atten_func(
        inputs["q"],
        k_deq,
        v_deq,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
    )
    out, lse = sparse_atten_nvfp4_kv_func(
        inputs["q"],
        k_q.data,
        v_q.data,
        k_q.scale,
        v_q.scale,
        k_q.global_scale,
        v_q.global_scale,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
    )

    torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_sparse_atten_fp8_kv_matches_dequantized_bf16(
    seqlen_q: int,
    seqlen_kv: int,
    topk: int,
    qhead_per_kv: int,
    head_kv: int,
    batch: int,
    causal: bool,
    paged: bool,
) -> None:
    seed = (
        2000
        + int(paged) * 100
        + seqlen_q
        + seqlen_kv
        + topk
        + qhead_per_kv
    )
    torch.random.manual_seed(seed)
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    if paged:
        causal = True
        inputs = _build_paged_inputs(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            page_size=BLK_KV,
            seqused_trim=17,
            dtype=torch.bfloat16,
            page_table_mode="shuffle",
        )
        k_source = inputs["k_paged"]
        v_source = inputs["v_paged"]
    else:
        inputs = _build_sparse_inputs(
            q_lens=(seqlen_q,) * batch,
            k_lens=(seqlen_kv,) * batch,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            dtype=torch.bfloat16,
            q2k_pattern=Q2KPattern.UNIFORM,
        )
        k_source = inputs["k"]
        v_source = inputs["v"]

    k_fp8 = k_source.to(torch.float8_e4m3fn)
    v_fp8 = v_source.to(torch.float8_e4m3fn)
    k_deq = k_fp8.to(torch.bfloat16)
    v_deq = v_fp8.to(torch.bfloat16)

    out_ref, lse_ref = sparse_atten_func(
        inputs["q"],
        k_deq,
        v_deq,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
    )
    out, lse = sparse_atten_func(
        inputs["q"],
        k_fp8,
        v_fp8,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
    )

    torch.testing.assert_close(out.float(), out_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(lse, lse_ref, atol=0, rtol=0)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("batch", [3])
@pytest.mark.parametrize("head_kv", [2])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_kv"),
    [
        (8192, 8192),
        (4096, 8192),
        (2048, 8192),
        (1024, 8192),
        (8192, 4096),
        (4096, 4096),
    ],
)
def test_sparse_atten_fp8_qkv_qk_fp8_pv_bf16_matches_reference(
    seqlen_q: int,
    seqlen_kv: int,
    topk: int,
    qhead_per_kv: int,
    head_kv: int,
    batch: int,
    causal: bool,
    paged: bool,
) -> None:
    seed = (
        3000
        + int(paged) * 100
        + seqlen_q
        + seqlen_kv
        + topk
        + qhead_per_kv
    )
    torch.random.manual_seed(seed)
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    if paged:
        causal = True
        inputs = _build_paged_inputs(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            page_size=BLK_KV,
            seqused_trim=17,
            dtype=torch.bfloat16,
            page_table_mode="shuffle",
        )
        k_source = inputs["k_paged"]
        v_source = inputs["v_paged"]
    else:
        inputs = _build_sparse_inputs(
            q_lens=(seqlen_q,) * batch,
            k_lens=(seqlen_kv,) * batch,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            dtype=torch.bfloat16,
            q2k_pattern=Q2KPattern.UNIFORM,
        )
        k_source = inputs["k"]
        v_source = inputs["v"]

    q_fp8 = inputs["q"].to(torch.float8_e4m3fn)
    k_fp8 = k_source.to(torch.float8_e4m3fn)
    v_fp8 = v_source.to(torch.float8_e4m3fn)
    if paged:
        k_ref_chunks = []
        v_ref_chunks = []
        for batch_idx, k_len in enumerate(inputs["k_lens"]):
            logical_pages = (int(k_len) + BLK_KV - 1) // BLK_KV
            for logical_page in range(logical_pages):
                physical_page = int(inputs["page_table"][batch_idx, logical_page].item())
                start = logical_page * BLK_KV
                page_len = min(BLK_KV, int(k_len) - start)
                k_ref_chunks.append(k_fp8[physical_page, :, :page_len].transpose(0, 1))
                v_ref_chunks.append(v_fp8[physical_page, :, :page_len].transpose(0, 1))
        k_ref = torch.cat(k_ref_chunks, dim=0)
        v_ref = torch.cat(v_ref_chunks, dim=0).to(torch.bfloat16)
    else:
        k_ref = k_fp8
        v_ref = v_fp8.to(torch.bfloat16)

    out_ref, lse_ref = _reference_varlen_forward(
        q_fp8,
        k_ref,
        v_ref,
        inputs["q2k"],
        q_lens=inputs["q_lens"],
        k_lens=inputs["k_lens"],
        blk_kv=inputs["blk_kv"],
        softmax_scale=inputs["softmax_scale"],
        causal=causal,
        upcast=True,
        seqused_k=inputs["seqused_k"] if paged else None,
        p_dtype=torch.bfloat16,
    )
    out, lse = sparse_atten_func(
        q_fp8,
        k_fp8,
        v_fp8,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
        qk_dtype=torch.float8_e4m3fn,
        pv_dtype=torch.bfloat16,
    )

    assert out.dtype == torch.bfloat16
    _assert_fp8_forward_close(out, out_ref, lse, lse_ref)


def test_sparse_atten_nvfp4_kv_te_quantized_flat_smoke() -> None:
    torch.random.manual_seed(101)
    inputs = _build_sparse_inputs(
        q_lens=(64,),
        k_lens=(512,),
        head_kv=1,
        qhead_per_kv=4,
        dim=128,
        topk=4,
        blk_kv=BLK_KV,
        causal=False,
        dtype=torch.bfloat16,
        q2k_pattern=Q2KPattern.UNIFORM,
    )
    k_q, v_q = _quantize_kv_bf16_to_nvfp4(inputs["k"], inputs["v"])
    k_deq = _dequant_nvfp4_to_bf16(k_q)
    v_deq = _dequant_nvfp4_to_bf16(v_q)

    out_ref, lse_ref = sparse_atten_func(
        inputs["q"],
        k_deq,
        v_deq,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        4,
        blk_kv=inputs["blk_kv"],
        causal=False,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        schedule=inputs["schedule"],
    )
    out, lse = sparse_atten_nvfp4_kv_func(
        inputs["q"],
        k_q.data,
        v_q.data,
        k_q.scale,
        v_q.scale,
        k_q.global_scale,
        v_q.global_scale,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        4,
        blk_kv=inputs["blk_kv"],
        causal=False,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        schedule=inputs["schedule"],
    )

    torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("qhead_per_kv", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [4, 8])
def test_sparse_atten_nvfp4_kv_fp8_q_matches_calibrated_fp8_reference(
    topk: int,
    qhead_per_kv: int,
    paged: bool,
) -> None:
    torch.random.manual_seed(211 + int(paged) * 100 + qhead_per_kv * 10 + topk)
    if paged:
        causal = True
        inputs = _build_paged_inputs(
            batch=1,
            seqlen_q=64,
            seqlen_kv=1024,
            head_kv=1,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            page_size=BLK_KV,
            seqused_trim=0,
            dtype=torch.bfloat16,
            page_table_mode="shuffle",
        )
        k_source = inputs["k_paged"]
        v_source = inputs["v_paged"]
    else:
        causal = False
        inputs = _build_sparse_inputs(
            q_lens=(64,),
            k_lens=(1024,),
            head_kv=1,
            qhead_per_kv=qhead_per_kv,
            dim=128,
            topk=topk,
            blk_kv=BLK_KV,
            causal=causal,
            dtype=torch.bfloat16,
            q2k_pattern=Q2KPattern.UNIFORM,
        )
        k_source = inputs["k"]
        v_source = inputs["v"]

    q_fp8 = inputs["q"].to(torch.float8_e4m3fn)
    k_q, v_q = _quantize_kv_bf16_to_nvfp4(k_source, v_source)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    # The block-scaled FP4 product is exactly representable in BF16. Apply
    # the global scale in FP32, avoiding an extra BF16 rounding before FP8.
    k_stage = _dequant_nvfp4_to_bf16(k_q, include_global_scale=False).float() * k_q.global_scale
    v_stage = _dequant_nvfp4_to_bf16(v_q, include_global_scale=False).float() * v_q.global_scale
    k_stage = k_stage.float().clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    v_stage = v_stage.float().clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)

    if paged:
        k_ref_chunks = []
        v_ref_chunks = []
        for batch_idx, k_len in enumerate(inputs["k_lens"]):
            logical_pages = (int(k_len) + BLK_KV - 1) // BLK_KV
            for logical_page in range(logical_pages):
                physical_page = int(inputs["page_table"][batch_idx, logical_page].item())
                start = logical_page * BLK_KV
                page_len = min(BLK_KV, int(k_len) - start)
                k_ref_chunks.append(k_stage[physical_page, :, :page_len].transpose(0, 1))
                v_ref_chunks.append(v_stage[physical_page, :, :page_len].transpose(0, 1))
        k_ref = torch.cat(k_ref_chunks, dim=0)
        v_ref = torch.cat(v_ref_chunks, dim=0)
    else:
        k_ref = k_stage
        v_ref = v_stage

    out_ref, lse_ref = _reference_varlen_forward(
        q_fp8,
        k_ref,
        v_ref,
        inputs["q2k"],
        q_lens=inputs["q_lens"],
        k_lens=inputs["k_lens"],
        blk_kv=inputs["blk_kv"],
        softmax_scale=inputs["softmax_scale"],
        causal=causal,
        upcast=True,
        seqused_k=inputs["seqused_k"] if paged else None,
        p_dtype=torch.float8_e4m3fn,
    )

    out, lse = sparse_atten_nvfp4_kv_func(
        q_fp8,
        k_q.data,
        v_q.data,
        k_q.scale,
        v_q.scale,
        k_q.global_scale,
        v_q.global_scale,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        topk,
        blk_kv=inputs["blk_kv"],
        causal=causal,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        page_table=inputs["page_table"] if paged else None,
        seqused_k=inputs["seqused_k"] if paged else None,
        schedule=inputs["schedule"],
    )

    _assert_fp8_forward_close(out, out_ref, lse, lse_ref)


def test_sparse_atten_nvfp4_kv_fp8_q_without_global_scale_matches_reference() -> None:
    torch.random.manual_seed(503)
    inputs = _build_sparse_inputs(
        q_lens=(64,),
        k_lens=(1024,),
        head_kv=1,
        qhead_per_kv=4,
        dim=128,
        topk=4,
        blk_kv=BLK_KV,
        causal=False,
        dtype=torch.bfloat16,
        q2k_pattern=Q2KPattern.UNIFORM,
    )
    q_fp8 = inputs["q"].to(torch.float8_e4m3fn)
    k_q = _make_synthetic_nvfp4_tensor(
        tuple(inputs["k"].shape),
        layout=NVFP4_K_SCALE_LAYOUT,
    )
    v_q = _make_synthetic_nvfp4_tensor(
        tuple(inputs["v"].shape),
        layout=NVFP4_V_SCALE_LAYOUT,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    k_ref = _dequant_nvfp4_to_bf16(
        k_q,
        include_global_scale=False,
    ).float().clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    v_ref = _dequant_nvfp4_to_bf16(
        v_q,
        include_global_scale=False,
    ).float().clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)

    out_ref, lse_ref = _reference_varlen_forward(
        q_fp8,
        k_ref,
        v_ref,
        inputs["q2k"],
        q_lens=inputs["q_lens"],
        k_lens=inputs["k_lens"],
        blk_kv=inputs["blk_kv"],
        softmax_scale=inputs["softmax_scale"],
        causal=False,
        upcast=True,
        p_dtype=torch.float8_e4m3fn,
    )
    out, lse = sparse_atten_nvfp4_kv_func(
        q_fp8,
        k_q.data,
        v_q.data,
        k_q.scale,
        v_q.scale,
        None,
        None,
        inputs["k2q_row_ptr"],
        inputs["k2q_q_indices"],
        4,
        blk_kv=inputs["blk_kv"],
        causal=False,
        softmax_scale=inputs["softmax_scale"],
        partial_dtype=torch.bfloat16,
        return_softmax_lse=True,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        schedule=inputs["schedule"],
    )

    _assert_fp8_forward_close(out, out_ref, lse, lse_ref)


@pytest.mark.parametrize("kv_tokens", DECODE_KV_TOKEN_SWEEP)
def test_sparse_decode_page_fp8_schedule_sweep(kv_tokens: int) -> None:
    fn = _get_sparse_decode_atten_func_for_test()
    page_count = (kv_tokens + BLK_KV - 1) // BLK_KV
    page_table = torch.arange(
        DECODE_BATCH * page_count,
        device="cuda",
        dtype=torch.int32,
    ).view(DECODE_BATCH, page_count)
    seqused_k = torch.full((DECODE_BATCH,), kv_tokens, dtype=torch.int32, device="cuda")

    fn.plan(
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=DECODE_SEQLEN_Q,
        max_seqlen_k=kv_tokens,
        num_qo_heads=DECODE_HEAD_KV * DECODE_QHEAD_PER_KV,
        num_kv_heads=DECODE_HEAD_KV,
        head_dim=DECODE_DIM,
    )
    schedule = fn.decode_schedule
    assert schedule is not None
    assert schedule.cta_tile_q == 128
    assert schedule.num_q_tiles == 1
    assert schedule.padded_work_count >= schedule.work_count
    expected_chunks = (
        (page_count + schedule.kv_chunk_size_pages - 1) // schedule.kv_chunk_size_pages
        if schedule.split_kv
        else 1
    )
    assert schedule.work_count == DECODE_BATCH * expected_chunks
    assert torch.equal(
        schedule.kv_pages.cpu(),
        torch.full((DECODE_BATCH,), page_count, dtype=torch.int32),
    )
    assert torch.equal(
        schedule.split_counts.cpu(),
        torch.full((DECODE_BATCH,), expected_chunks, dtype=torch.int32),
    )
    o_indptr = schedule.o_indptr.cpu()
    q_tokens_per_group = 128 // DECODE_QHEAD_PER_KV
    q_stride_partial = (
        (DECODE_SEQLEN_Q + q_tokens_per_group - 1) // q_tokens_per_group
    ) * q_tokens_per_group
    assert int(o_indptr[-1]) == DECODE_BATCH * q_stride_partial * expected_chunks
    assert torch.equal(
        o_indptr[1:] - o_indptr[:-1],
        torch.full((DECODE_BATCH,), q_stride_partial * expected_chunks, dtype=torch.int32),
    )


def test_sparse_decode_page_fp8_schedule_partial_rows_are_tile_aligned() -> None:
    fn = _get_sparse_decode_atten_func_for_test()
    seqlen_q = DECODE_SEQLEN_Q
    kv_tokens = BLK_KV * 3
    page_count = (kv_tokens + BLK_KV - 1) // BLK_KV
    page_table = torch.arange(
        DECODE_BATCH * page_count,
        device="cuda",
        dtype=torch.int32,
    ).view(DECODE_BATCH, page_count)
    seqused_k = torch.full((DECODE_BATCH,), kv_tokens, dtype=torch.int32, device="cuda")

    fn.plan(
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=seqlen_q,
        max_seqlen_k=kv_tokens,
        num_qo_heads=DECODE_HEAD_KV * DECODE_QHEAD_PER_KV,
        num_kv_heads=DECODE_HEAD_KV,
        head_dim=DECODE_DIM,
        fixed_split_size=1,
    )
    schedule = fn.decode_schedule
    assert schedule is not None
    assert schedule.split_kv
    q_tokens_per_group = 128 // DECODE_QHEAD_PER_KV
    q_stride_partial = (
        (seqlen_q + q_tokens_per_group - 1) // q_tokens_per_group
    ) * q_tokens_per_group
    expected_chunks = page_count
    o_indptr = schedule.o_indptr.cpu()
    assert int(o_indptr[-1]) == DECODE_BATCH * q_stride_partial * expected_chunks
    assert torch.equal(
        o_indptr[1:] - o_indptr[:-1],
        torch.full((DECODE_BATCH,), q_stride_partial * expected_chunks, dtype=torch.int32),
    )


def test_sparse_decode_page_fp8_schedule_varlen_kv_balance() -> None:
    fn = _get_sparse_decode_atten_func_for_test()
    lens = torch.tensor(
        [8, 128, 1024, 1 << 20] * (DECODE_BATCH // 4),
        device="cuda",
        dtype=torch.int32,
    )
    max_kv = int(lens.max().item())
    max_pages = (max_kv + BLK_KV - 1) // BLK_KV
    page_table = torch.arange(
        DECODE_BATCH * max_pages,
        device="cuda",
        dtype=torch.int32,
    ).view(DECODE_BATCH, max_pages)

    fn.plan(
        page_table=page_table,
        seqused_k=lens,
        seqlen_q=DECODE_SEQLEN_Q,
        max_seqlen_k=max_kv,
        num_qo_heads=DECODE_HEAD_KV * DECODE_QHEAD_PER_KV,
        num_kv_heads=DECODE_HEAD_KV,
        head_dim=DECODE_DIM,
        max_grid_size=256,
    )
    schedule = fn.decode_schedule
    assert schedule is not None
    assert schedule.split_kv
    kv_pages = schedule.kv_pages.cpu()
    split_counts = schedule.split_counts.cpu()
    assert int(kv_pages[0]) == 1
    assert int(kv_pages[3]) == max_pages
    assert int(split_counts[3]) > int(split_counts[0])
    assert schedule.work_count == int(split_counts.sum().item()) * schedule.num_q_tiles
    merge_indptr = schedule.merge_indptr.cpu()
    row_counts = merge_indptr[1:] - merge_indptr[:-1]
    for batch_idx in range(DECODE_BATCH):
        q_slice = row_counts[
            batch_idx * DECODE_SEQLEN_Q : (batch_idx + 1) * DECODE_SEQLEN_Q
        ]
        assert torch.equal(
            q_slice,
            torch.full((DECODE_SEQLEN_Q,), int(split_counts[batch_idx]), dtype=torch.int32),
        )


@pytest.mark.parametrize("kv_tokens", DECODE_KV_TOKEN_SWEEP)
def test_sparse_decode_page_fp8_forward(kv_tokens: int) -> None:
    torch.random.manual_seed(0)
    fn = _get_sparse_decode_atten_func_for_test()
    inputs = _build_decode_paged_dense_inputs(kv_tokens=kv_tokens)

    out, lse = _run_sparse_decode_page_fp8(fn, inputs)
    out_ref, lse_ref = _decode_paged_dense_reference(inputs)
    out_ref_exact, _ = _decode_paged_dense_reference(inputs, p_dtype=None)
    assert out.dtype == torch.bfloat16
    assert out.shape == inputs["q"].shape
    assert lse.shape == inputs["q"].shape[:2]
    _assert_fp8_forward_close(
        out, out_ref, lse, lse_ref, out_ref_unquantized_p=out_ref_exact
    )


def test_sparse_decode_page_fp8_forward_forced_split_kv_tile_aligned_partial() -> None:
    torch.random.manual_seed(0)
    fn = _get_sparse_decode_atten_func_for_test()
    inputs = _build_decode_paged_dense_inputs(
        kv_tokens=BLK_KV * 2,
        seqlen_q=DECODE_SEQLEN_Q,
    )

    fn.plan(
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        seqlen_q=inputs["seqlen_q"],
        max_seqlen_k=inputs["max_seqlen_k"],
        q2k_indices=inputs["q2k"],
        num_qo_heads=inputs["q"].shape[1],
        num_kv_heads=inputs["k_paged"].shape[1],
        head_dim=inputs["q"].shape[2],
        fixed_split_size=1,
    )
    assert fn.decode_schedule is not None
    assert fn.decode_schedule.split_kv

    out, lse = fn.run(
        inputs["q"],
        inputs["k_paged"],
        inputs["v_paged"],
        softmax_scale=inputs["softmax_scale"],
        return_softmax_lse=True,
    )
    out_ref, lse_ref = _decode_paged_dense_reference(inputs)
    out_ref_exact, _ = _decode_paged_dense_reference(inputs, p_dtype=None)
    _assert_fp8_forward_close(
        out, out_ref, lse, lse_ref, out_ref_unquantized_p=out_ref_exact
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _sparse_effective_tokens_per_q(topk: int, blk_kv: int) -> int:
    return topk * blk_kv


def _sparse_fwd_flops(total_q: int, head_q: int, topk: int, blk_kv: int, dim: int) -> int:
    visible_tokens = _sparse_effective_tokens_per_q(topk, blk_kv)
    return 2 * total_q * head_q * visible_tokens * (dim + dim)


def _decode_fwd_flops(total_q: int, head_q: int, kv_tokens: int, dim: int) -> int:
    return 2 * total_q * head_q * kv_tokens * (dim + dim)


def _tflops_from_ms(flops: int, time_ms: float) -> float:
    return flops / time_ms / 1e9


def _cuda_time_ms(
    fn,
    *,
    warmup: int,
    repeat: int,
    cuda_profiler_capture: bool = False,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cuda_profiler_capture:
        torch.cuda.profiler.start()
    try:
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeat
    finally:
        if cuda_profiler_capture:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()


def _resolve_bench_loop_counts(warmup: int, iters: int, profile: bool) -> tuple[int, int]:
    if profile:
        return 0, 1
    return warmup, iters


def _cuda_timed_loop_ms(
    fn,
    *,
    repeat: int,
    range_name: str,
    sync_nvtx: bool,
    cuda_profiler_capture: bool = False,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cuda_profiler_capture:
        torch.cuda.profiler.start()
    try:
        with _nvtx_range(range_name):
            start.record()
            for _ in range(repeat):
                fn()
                if sync_nvtx:
                    torch.cuda.synchronize()
            end.record()
            end.synchronize()
        return start.elapsed_time(end) / repeat
    finally:
        if cuda_profiler_capture:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()


def _build_sparse_benchmark_context(
    *,
    case_name: str,
    backend: str,
    q2k_pattern: str,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    dtype: torch.dtype,
    partial_dtype: torch.dtype,
    seed: int,
    fp8_kv: bool = False,
    mixed_fp8_qkv_pv_bf16: bool = False,
) -> dict[str, object]:
    torch.random.manual_seed(seed)
    input_dtype = torch.bfloat16 if (fp8_kv or mixed_fp8_qkv_pv_bf16) else dtype
    inputs = _build_sparse_inputs(
        q_lens=(seqlen_q,) * batch,
        k_lens=(seqlen_k,) * batch,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        dim=dim,
        topk=topk,
        blk_kv=blk_kv,
        causal=causal,
        dtype=input_dtype,
        q2k_pattern=q2k_pattern,
    )

    q_src = inputs["q"]
    k_src = inputs["k"]
    v_src = inputs["v"]
    q2k = inputs["q2k"]
    k2q_row_ptr = inputs["k2q_row_ptr"]
    k2q_q_indices = inputs["k2q_q_indices"]
    schedule = inputs["schedule"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    max_seqlen_q = int(inputs["max_seqlen_q"])
    max_seqlen_k = int(inputs["max_seqlen_k"])
    blk_kv = inputs["blk_kv"]
    softmax_scale = inputs["softmax_scale"]

    q_bf16 = q_src.detach().clone()
    q = q_bf16.to(torch.float8_e4m3fn) if mixed_fp8_qkv_pv_bf16 else q_bf16
    k_bf16 = k_src.detach().clone()
    v_bf16 = v_src.detach().clone()
    use_fp8_kv_storage = fp8_kv or mixed_fp8_qkv_pv_bf16
    k = k_bf16.to(torch.float8_e4m3fn) if use_fp8_kv_storage else k_bf16
    v = v_bf16.to(torch.float8_e4m3fn) if use_fp8_kv_storage else v_bf16
    needs_cute = backend in {BenchmarkBackend.CUTE, BenchmarkBackend.BOTH}
    needs_triton = backend in {BenchmarkBackend.TRITON, BenchmarkBackend.BOTH}
    if (fp8_kv or mixed_fp8_qkv_pv_bf16) and not needs_cute:
        raise ValueError("fp8 KV/mixed precision benchmark currently supports only --backend cute")
    if dtype == torch.float8_e4m3fn and needs_triton:
        raise ValueError("fp8 benchmark currently supports only --backend cute")

    triton_metadata = None
    if needs_triton:
        if not TRITON_REFERENCE_AVAILABLE:
            raise RuntimeError("Triton reference backend is not available")
        triton_metadata = build_triton_sparse_metadata(
            q2k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            block_size_q=1,
            block_size_k=blk_kv,
        )
        torch.cuda.synchronize()

    def run_cute_fwd() -> tuple[torch.Tensor, ...]:
        return sparse_atten_func(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            partial_dtype=partial_dtype,
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            schedule=schedule,
            qk_dtype=torch.float8_e4m3fn if mixed_fp8_qkv_pv_bf16 else None,
            pv_dtype=torch.bfloat16 if mixed_fp8_qkv_pv_bf16 else None,
        )

    def run_bf16_fwd() -> tuple[torch.Tensor, ...]:
        return sparse_atten_func(
            q_bf16,
            k_bf16,
            v_bf16,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            partial_dtype=partial_dtype,
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            schedule=schedule,
        )

    def run_triton_fwd() -> tuple[torch.Tensor, torch.Tensor]:
        return run_triton_sparse_attention_forward(
            q,
            k,
            v,
            q2k,
            triton_metadata,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            block_size_q=1,
            block_size_k=blk_kv,
            softmax_scale=softmax_scale,
        )

    backend_fns = []
    if mixed_fp8_qkv_pv_bf16:
        backend_fns.append(("bf16_prefill", run_bf16_fwd))
        backend_fns.append(("mixed_fp8_qkv_pv_bf16_prefill", run_cute_fwd))
    elif fp8_kv:
        backend_fns.append(("bf16_prefill", run_bf16_fwd))
        backend_fns.append(("fp8_kv_prefill", run_cute_fwd))
    elif needs_cute:
        backend_fns.append((BenchmarkBackend.CUTE, run_cute_fwd))
    if needs_triton:
        backend_fns.append((BenchmarkBackend.TRITON, run_triton_fwd))

    # CSR rebuild closure for benchmark timing. Each call rebuilds the
    # full k2q CSR from q2k + cu_seqlens. The output tensors are
    # discarded — we only care about kernel latency, not capturing the
    # result (the live k2q_row_ptr / k2q_q_indices used by forward were built
    # once during context setup).
    csr_total_k = int(k_src.shape[0])

    def run_csr() -> None:
        build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            total_k=csr_total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )

    return {
        "case_name": case_name,
        "backend": (
            "bf16_vs_mixed_fp8_qkv_pv_bf16"
            if mixed_fp8_qkv_pv_bf16
            else "bf16_vs_fp8_kv" if fp8_kv else backend
        ),
        "q2k_pattern": q2k_pattern,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "dim": dim,
        "topk": topk,
        "blk_kv": blk_kv,
        "causal": causal,
        "dtype": (
            "fp8_qkv_qk_fp8_pv_bf16"
            if mixed_fp8_qkv_pv_bf16
            else "bf16_q_fp8_kv" if fp8_kv else dtype
        ),
        "partial_dtype": partial_dtype,
        "seed": seed,
        "inputs": inputs,
        "q2k": q2k,
        "backend_fns": backend_fns,
        "run_csr": run_csr,
    }


def _build_sparse_nvfp4_kv_benchmark_context(
    *,
    case_name: str,
    q2k_pattern: str,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    seed: int,
    q_dtype: torch.dtype = torch.bfloat16,
    paged: bool = False,
    page_size: int = BLK_KV,
    seqused_trim: int = 0,
) -> dict[str, object]:
    torch.random.manual_seed(seed)
    if paged:
        inputs = _build_paged_inputs(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_k,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=dim,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            page_size=page_size,
            seqused_trim=seqused_trim,
            dtype=torch.bfloat16,
            q2k_pattern=q2k_pattern,
        )
        k_source = inputs["k_paged"]
        v_source = inputs["v_paged"]
    else:
        inputs = _build_sparse_inputs(
            q_lens=(seqlen_q,) * batch,
            k_lens=(seqlen_k,) * batch,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            dim=dim,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            dtype=torch.bfloat16,
            q2k_pattern=q2k_pattern,
        )
        k_source = inputs["k"]
        v_source = inputs["v"]

    q = inputs["q"].detach().clone()
    if q_dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    elif q_dtype != torch.bfloat16:
        raise ValueError(f"Unsupported NVFP4 KV benchmark Q dtype: {q_dtype}")
    k_q = _make_synthetic_nvfp4_tensor(
        tuple(k_source.shape),
        layout=NVFP4_K_SCALE_LAYOUT,
        global_scale_value=0.75,
    )
    v_q = _make_synthetic_nvfp4_tensor(
        tuple(v_source.shape),
        layout=NVFP4_V_SCALE_LAYOUT,
        global_scale_value=1.25,
    )
    k_bf16 = _dequant_nvfp4_to_bf16(k_q)
    v_bf16 = _dequant_nvfp4_to_bf16(v_q)

    k2q_row_ptr = inputs["k2q_row_ptr"]
    k2q_q_indices = inputs["k2q_q_indices"]
    schedule = inputs["schedule"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    page_table = inputs["page_table"] if paged else None
    seqused_k = inputs["seqused_k"] if paged else None
    max_seqlen_q = int(inputs["max_seqlen_q"])
    max_seqlen_k = int(inputs["max_seqlen_k"])
    softmax_scale = inputs["softmax_scale"]

    def run_bf16_fwd() -> tuple[torch.Tensor, ...]:
        return sparse_atten_func(
            q,
            k_bf16,
            v_bf16,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            partial_dtype=torch.bfloat16,
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
        )

    def run_nvfp4_kv_fwd() -> tuple[torch.Tensor, ...]:
        return sparse_atten_nvfp4_kv_func(
            q,
            k_q.data,
            v_q.data,
            k_q.scale,
            v_q.scale,
            k_q.global_scale,
            v_q.global_scale,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            partial_dtype=torch.bfloat16,
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
        )

    def run_csr() -> None:
        build_k2q_csr(
            inputs["q2k"],
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            total_k=(
                int(cu_seqlens_k[-1].item())
                if paged
                else int(inputs["k"].shape[0])
            ),
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )

    backend_fns = [("nvfp4_kv_prefill", run_nvfp4_kv_fwd)]
    if q_dtype == torch.bfloat16:
        backend_fns.insert(0, ("bf16_prefill", run_bf16_fwd))

    dtype_label = (
        "fp8_q_nvfp4_kv"
        if q_dtype == torch.float8_e4m3fn
        else "bf16_q_nvfp4_kv"
    )

    return {
        "case_name": case_name,
        "backend": "bf16_vs_nvfp4_kv",
        "q2k_pattern": q2k_pattern,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "dim": dim,
        "topk": topk,
        "blk_kv": blk_kv,
        "causal": causal,
        "dtype": dtype_label,
        "partial_dtype": torch.bfloat16,
        "seed": seed,
        "inputs": inputs,
        "q2k": inputs["q2k"],
        "backend_fns": backend_fns,
        "run_csr": run_csr,
        "paged": paged,
    }


def _run_sparse_benchmark_warmup(ctx: dict[str, object], *, warmup: int, sync_nvtx: bool) -> None:
    if warmup <= 0:
        return

    case_token = _nvtx_token(ctx["case_name"])
    pattern_token = _nvtx_token(ctx["q2k_pattern"])
    for backend_name, run_fwd in ctx["backend_fns"]:
        backend_token = _nvtx_token(backend_name)
        nvtx_prefix = f"case_{case_token}__pattern_{pattern_token}__backend_{backend_token}"
        with _nvtx_range(f"{nvtx_prefix}__compile_warmup"):
            for i in range(warmup):
                with _nvtx_range(f"{nvtx_prefix}__warmup_iter_{i}__fwd"):
                    run_fwd()
                    if sync_nvtx:
                        torch.cuda.synchronize()
            torch.cuda.synchronize()


def _run_sparse_benchmark_autotune(ctx: dict[str, object], *, sync_nvtx: bool) -> None:
    case_token = _nvtx_token(ctx["case_name"])
    pattern_token = _nvtx_token(ctx["q2k_pattern"])
    for backend_name, run_fwd in ctx["backend_fns"]:
        backend_token = _nvtx_token(backend_name)
        nvtx_prefix = f"case_{case_token}__pattern_{pattern_token}__backend_{backend_token}"
        with _nvtx_range(f"{nvtx_prefix}__autotune_preflight"):
            run_fwd()
            if sync_nvtx:
                torch.cuda.synchronize()
            torch.cuda.synchronize()


def _run_sparse_benchmark_timed(
    ctx: dict[str, object],
    *,
    iters: int,
    sync_nvtx: bool,
    cuda_profiler_capture_labels: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    timings: dict[str, dict[str, float]] = {}

    with _nvtx_range(
        f"case {ctx['case_name']} pattern {ctx['q2k_pattern']}"
    ):
        # CSR build is backend-independent (both cute and triton consume
        # the same k2q_row_ptr + k2q_q_indices). Time it once and attach
        # to every backend's timing dict.
        run_csr = ctx.get("run_csr")
        csr_ms = 0.0
        if run_csr is not None:
            with _nvtx_range("csr_build"):
                if iters > 1:
                    with _nvtx_range("csr prewarm"):
                        run_csr()
                        torch.cuda.synchronize()
                csr_ms = _cuda_timed_loop_ms(
                    run_csr,
                    repeat=iters,
                    range_name="csr_build",
                    sync_nvtx=sync_nvtx,
                )
        for backend_name, run_fwd in ctx["backend_fns"]:
            backend_label = str(backend_name)
            with _nvtx_range(backend_label):
                if iters > 1:
                    with _nvtx_range(f"{backend_label} fwd prewarm"):
                        run_fwd()
                        torch.cuda.synchronize()
                fwd_ms = _cuda_timed_loop_ms(
                    run_fwd,
                    repeat=iters,
                    range_name=f"{backend_label} fwd",
                    sync_nvtx=sync_nvtx,
                    cuda_profiler_capture=(
                        cuda_profiler_capture_labels is not None
                        and backend_label in cuda_profiler_capture_labels
                    ),
                )
            timings[backend_name] = {
                "csr_ms": csr_ms,
                "fwd_ms": fwd_ms,
            }

    return timings


def _print_sparse_benchmark_results(
    ctx: dict[str, object],
    timings: dict[str, dict[str, float]],
) -> None:
    inputs = ctx["inputs"]
    case_name = str(ctx["case_name"])
    backend = str(ctx["backend"])
    q2k_pattern = str(ctx["q2k_pattern"])
    batch = int(ctx["batch"])
    seqlen_q = int(ctx["seqlen_q"])
    seqlen_k = int(ctx["seqlen_k"])
    head_kv = int(ctx["head_kv"])
    qhead_per_kv = int(ctx["qhead_per_kv"])
    dim = int(ctx["dim"])
    topk = int(ctx["topk"])
    blk_kv = int(ctx["blk_kv"])
    causal = bool(ctx["causal"])
    dtype = ctx["dtype"]
    partial_dtype = ctx["partial_dtype"]
    seed = int(ctx["seed"])
    q2k = ctx["q2k"]
    k2q_row_ptr = inputs["k2q_row_ptr"]
    k_lens = inputs["k_lens"]
    paged = bool(ctx.get("paged", False))

    total_q = int(inputs["q"].shape[0])
    head_q = head_kv * qhead_per_kv
    visible_tokens = _sparse_effective_tokens_per_q(topk, blk_kv)
    fwd_flops = _sparse_fwd_flops(total_q, head_q, topk, blk_kv, dim)
    print(f"case: {case_name}")
    mode = "sparse_page_atten" if paged else "sparse_atten"
    print(f"mode: {mode} causal={causal} batch={batch} q={(seqlen_q,) * batch} k={(seqlen_k,) * batch}")
    print(f"backend: {backend} q2k_pattern: {q2k_pattern}")
    print("q_load_warps: 4")
    print(f"q2k_fanout: {_format_q2k_fanout(q2k)}")
    print(
        "csr_pattern: "
        f"{_format_csr_pattern(k2q_row_ptr, k_lens=k_lens, blk_kv=blk_kv, topk=topk)}"
    )
    for backend_name, timing in timings.items():
        csr_ms = timing.get("csr_ms", 0.0)
        fwd_ms = timing["fwd_ms"]
        fwd_tflops = _tflops_from_ms(fwd_flops, fwd_ms)
        full_e2e_ms = csr_ms + fwd_ms
        print(
            f"{backend_name}_ms: csr={csr_ms:.3f} fwd={fwd_ms:.3f} "
            f"csr+fwd={full_e2e_ms:.3f}"
        )
        print(f"{backend_name}_tflops: fwd={fwd_tflops:.3f}")
    if BenchmarkBackend.CUTE in timings and BenchmarkBackend.TRITON in timings:
        print(
            "speedup_triton_over_cute: "
            f"fwd={timings[BenchmarkBackend.TRITON]['fwd_ms'] / timings[BenchmarkBackend.CUTE]['fwd_ms']:.2f}x"
        )
    if "bf16_prefill" in timings and "nvfp4_kv_prefill" in timings:
        bf16_ms = timings["bf16_prefill"]["fwd_ms"]
        nvfp4_kv_ms = timings["nvfp4_kv_prefill"]["fwd_ms"]
        print(
            "nvfp4_kv_vs_bf16_fwd_speedup: "
            f"speedup={bf16_ms / nvfp4_kv_ms:.2f}x "
            f"bf16_fwd_ms={bf16_ms:.3f} nvfp4_kv_fwd_ms={nvfp4_kv_ms:.3f}"
        )
    if "bf16_prefill" in timings and "fp8_kv_prefill" in timings:
        bf16_ms = timings["bf16_prefill"]["fwd_ms"]
        fp8_kv_ms = timings["fp8_kv_prefill"]["fwd_ms"]
        print(
            "fp8_kv_vs_bf16_fwd_speedup: "
            f"speedup={bf16_ms / fp8_kv_ms:.2f}x "
            f"bf16_fwd_ms={bf16_ms:.3f} fp8_kv_fwd_ms={fp8_kv_ms:.3f}"
        )
    if "bf16_prefill" in timings and "mixed_fp8_qkv_pv_bf16_prefill" in timings:
        bf16_ms = timings["bf16_prefill"]["fwd_ms"]
        mixed_ms = timings["mixed_fp8_qkv_pv_bf16_prefill"]["fwd_ms"]
        print(
            "mixed_fp8_qkv_pv_bf16_vs_bf16_fwd_speedup: "
            f"speedup={bf16_ms / mixed_ms:.2f}x "
            f"bf16_fwd_ms={bf16_ms:.3f} mixed_fwd_ms={mixed_ms:.3f}"
        )
    print(f"tokens_per_q: {visible_tokens}")
    print(f"total_q: {total_q} head_q: {head_q} dim: {dim}")
    print(
        f"seed: {seed} dtype: {dtype} partial_dtype: {partial_dtype} "
        f"blk_kv: {blk_kv}"
    )


def _benchmark_sparse_atten(
    *,
    case_name: str,
    backend: str,
    q2k_pattern: str,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    dtype: torch.dtype,
    partial_dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
    profile: bool,
    mixed_fp8_qkv_pv_bf16: bool = False,
) -> None:
    warmup, iters = _resolve_bench_loop_counts(warmup, iters, profile)
    ctx = _build_sparse_benchmark_context(
        case_name=case_name,
        backend=backend,
        q2k_pattern=q2k_pattern,
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        dim=dim,
        topk=topk,
        blk_kv=blk_kv,
        causal=causal,
        dtype=dtype,
        partial_dtype=partial_dtype,
        seed=seed,
        mixed_fp8_qkv_pv_bf16=mixed_fp8_qkv_pv_bf16,
    )
    _run_sparse_benchmark_warmup(ctx, warmup=warmup, sync_nvtx=False)
    timings = _run_sparse_benchmark_timed(ctx, iters=iters, sync_nvtx=False)
    _print_sparse_benchmark_results(ctx, timings)


def _benchmark_sparse_page_atten(
    *,
    q2k_pattern: str,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    dim: int,
    topk: int,
    blk_kv: int,
    causal: bool,
    dtype: torch.dtype,
    partial_dtype: torch.dtype,
    page_size: int,
    seqused_trim: int,
    warmup: int,
    iters: int,
    seed: int,
    profile: bool,
    fp8_kv: bool = False,
    cuda_profiler_capture: bool = False,
) -> None:
    warmup, iters = _resolve_bench_loop_counts(warmup, iters, profile)
    torch.random.manual_seed(seed)
    input_dtype = torch.bfloat16 if fp8_kv else dtype
    inputs = _build_paged_inputs(
        batch=1,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_k,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        dim=dim,
        topk=topk,
        blk_kv=blk_kv,
        causal=causal,
        page_size=page_size,
        seqused_trim=seqused_trim,
        dtype=input_dtype,
        q2k_pattern=q2k_pattern,
    )

    q = inputs["q"]
    k_paged_bf16 = inputs["k_paged"]
    v_paged_bf16 = inputs["v_paged"]
    k_paged = k_paged_bf16.to(torch.float8_e4m3fn) if fp8_kv else k_paged_bf16
    v_paged = v_paged_bf16.to(torch.float8_e4m3fn) if fp8_kv else v_paged_bf16
    k2q_row_ptr = inputs["k2q_row_ptr"]
    k2q_q_indices = inputs["k2q_q_indices"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    page_table = inputs["page_table"]
    seqused_k = inputs["seqused_k"]
    schedule = inputs["schedule"]
    blk_kv = inputs["blk_kv"]
    softmax_scale = inputs["softmax_scale"]

    def run_fwd(k_arg: torch.Tensor, v_arg: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return sparse_atten_func(
            q,
            k_arg,
            v_arg,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            partial_dtype=partial_dtype,
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=inputs["max_seqlen_q"],
            max_seqlen_k=inputs["max_seqlen_k"],
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
        )

    def run_bf16_fwd() -> tuple[torch.Tensor, ...]:
        return run_fwd(k_paged_bf16, v_paged_bf16)

    def run_fp8_kv_fwd() -> tuple[torch.Tensor, ...]:
        return run_fwd(k_paged, v_paged)

    timings = {}
    if fp8_kv:
        timings["bf16_prefill"] = _cuda_time_ms(
            run_bf16_fwd,
            warmup=warmup,
            repeat=iters,
        )
        timings["fp8_kv_prefill"] = _cuda_time_ms(
            run_fp8_kv_fwd,
            warmup=warmup,
            repeat=iters,
            cuda_profiler_capture=cuda_profiler_capture,
        )
    else:
        timings["paged_prefill"] = _cuda_time_ms(
            run_fp8_kv_fwd,
            warmup=warmup,
            repeat=iters,
            cuda_profiler_capture=cuda_profiler_capture,
        )
    logical_seqused_k = int(seqused_k.item()) if seqused_k is not None else int(page_table.shape[1]) * page_size
    visible_tokens = _sparse_effective_tokens_per_q(topk, blk_kv)
    total_q = seqlen_q
    head_q = head_kv * qhead_per_kv
    fwd_flops = _sparse_fwd_flops(total_q, head_q, topk, blk_kv, dim)
    print("case: customer_sparse_page_atten")
    print(f"mode: sparse_page_atten causal={causal} page_size={page_size} seqused_k={logical_seqused_k}")
    print(f"q2k_pattern: {q2k_pattern}")
    for label, fwd_ms in timings.items():
        fwd_tflops = _tflops_from_ms(fwd_flops, fwd_ms)
        print(f"{label}_ms: fwd={fwd_ms:.3f}")
        print(f"{label}_tflops: fwd={fwd_tflops:.3f}")
    if "bf16_prefill" in timings and "fp8_kv_prefill" in timings:
        bf16_ms = timings["bf16_prefill"]
        fp8_kv_ms = timings["fp8_kv_prefill"]
        print(
            "fp8_kv_vs_bf16_fwd_speedup: "
            f"speedup={bf16_ms / fp8_kv_ms:.2f}x "
            f"bf16_fwd_ms={bf16_ms:.3f} fp8_kv_fwd_ms={fp8_kv_ms:.3f}"
        )
    print(f"tokens_per_q: {visible_tokens}")
    print(f"sq: {seqlen_q} skv: {seqlen_k} head_kv: {head_kv} qhead_per_kv: {qhead_per_kv} dim: {dim}")
    dtype_label = "bf16_q_fp8_kv" if fp8_kv else dtype
    print(f"topk: {topk} seed: {seed} dtype: {dtype_label} partial_dtype: {partial_dtype} blk_kv: {blk_kv}")


def _parse_decode_kv_tokens(raw: str) -> tuple[int, ...]:
    if raw in {"sweep", "all"}:
        return DECODE_KV_TOKEN_SWEEP
    values = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item.endswith("m"):
            value = int(item[:-1]) * 1024 * 1024
        elif item.endswith("k"):
            value = int(item[:-1]) * 1024
        else:
            value = int(item)
        values.append(value)
    if not values:
        raise ValueError("decode kv token list must not be empty")
    return tuple(values)


def _benchmark_sparse_decode_page_atten(
    *,
    kv_token_values: tuple[int, ...],
    batch: int,
    seqlen_q: int,
    warmup: int,
    iters: int,
    seed: int,
    profile: bool,
    sync_nvtx: bool,
) -> None:
    fn = _get_sparse_decode_atten_func_for_benchmark()
    warmup, iters = _resolve_bench_loop_counts(warmup, iters, profile)
    for kv_tokens in kv_token_values:
        torch.random.manual_seed(seed)
        inputs = _build_decode_paged_dense_inputs(
            kv_tokens=kv_tokens,
            batch=batch,
            seqlen_q=seqlen_q,
        )

        def run_fwd() -> tuple[torch.Tensor, torch.Tensor]:
            return _run_sparse_decode_page_fp8(fn, inputs, skip_not_implemented=False)

        if warmup > 0:
            with _nvtx_range(f"decode_page_fp8_kv{kv_tokens}_warmup"):
                for _ in range(warmup):
                    run_fwd()
                    if sync_nvtx:
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
        fwd_ms = _cuda_timed_loop_ms(
            run_fwd,
            repeat=iters,
            range_name=f"decode_page_fp8_kv{kv_tokens}_fwd",
            sync_nvtx=sync_nvtx,
        )
        total_q = batch * seqlen_q
        head_q = DECODE_HEAD_KV * DECODE_QHEAD_PER_KV
        fwd_tflops = _tflops_from_ms(
            _decode_fwd_flops(total_q, head_q, kv_tokens, DECODE_DIM),
            fwd_ms,
        )
        print("case: decode_page_fp8_dense_sparse")
        print(
            "mode: sparse_decode_page_atten "
            f"causal=True page_size={BLK_KV} seqused_k={kv_tokens}"
        )
        print(
            f"config: batch={batch} seqlen_q={seqlen_q} total_q={total_q} "
            f"head_q={head_q} head_kv={DECODE_HEAD_KV} "
            f"qhead_per_kv={DECODE_QHEAD_PER_KV} dim={DECODE_DIM}"
        )
        print(
            f"kv_tokens={kv_tokens} physical_kv_tokens={inputs['physical_kv_tokens']} "
            f"topk_blocks={inputs['topk']} dtype=torch.float8_e4m3fn"
        )
        print(f"decode_fp8_ms: fwd={fwd_ms:.3f}")
        print(f"decode_fp8_tflops: fwd={fwd_tflops:.3f}")


def _resolve_q2k_patterns(raw: str) -> tuple[str, ...]:
    if raw == Q2KPattern.BOTH:
        return (Q2KPattern.SINK, Q2KPattern.UNIFORM)
    return (raw,)


def _resolve_sparse_benchmark_cases(args: argparse.Namespace) -> tuple[dict[str, int | str], ...]:
    if args.customer_case == "manual":
        return (
            {
                "name": "manual_sparse_atten",
                "batch": args.b,
                "seqlen_q": args.sq,
                "seqlen_k": args.skv,
                "head_kv": args.head_kv,
                "qhead_per_kv": args.qhead_per_kv,
            },
        )
    if args.customer_case == "both":
        return tuple(dict(CUSTOMER_BENCHMARK_CASES[name]) for name in ("ring48k", "ulysses384k"))
    if args.customer_case == "qhead_sweep":
        return tuple(dict(CUSTOMER_QHEAD_SWEEP_CASES[name]) for name in ("qhead1", "qhead2", "qhead4", "qhead8", "qhead16"))
    return (dict(CUSTOMER_BENCHMARK_CASES[args.customer_case]),)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run interface-level sparse attention benchmarks")
    bench.add_argument("--paged", action="store_true")
    bench.add_argument(
        "--nvfp4-kv",
        dest="nvfp4_kv",
        action="store_true",
        help=(
            "Benchmark NVFP4 KV prefill against bf16 prefill using the same "
            "dequantized K/V values. Forces O_partial dtype to bf16."
        ),
    )
    bench.add_argument(
        "--fp8-kv",
        dest="fp8_kv",
        action="store_true",
        help=(
            "Benchmark Q=bf16 with K/V=fp8_e4m3 dequantized to bf16 "
            "inside the attention kernel, alongside a bf16 K/V baseline."
        ),
    )
    bench.add_argument(
        "--mixed-fp8-qkv-pv-bf16",
        dest="mixed_fp8_qkv_pv_bf16",
        action="store_true",
        help=(
            "Benchmark Q/K/V=fp8_e4m3 with QK fp8 and PV bf16 "
            "against a bf16 prefill baseline."
        ),
    )
    bench.add_argument(
        "--decode",
        action="store_true",
        help=(
            "Run decode fp8 paged dense-sparse benchmark: "
            "default B=32, seqlen_q=8, Hq=64, Hkv=4, D=128, causal=True."
        ),
    )
    bench.add_argument(
        "--backend",
        choices=[BenchmarkBackend.CUTE],
        default=BenchmarkBackend.CUTE,
    )
    bench.add_argument(
        "--customer-case",
        choices=["manual", "ring48k", "ulysses384k", "both", "qhead_sweep"],
        default="manual",
    )
    bench.add_argument(
        "--q2k-pattern",
        choices=[Q2KPattern.SINK, Q2KPattern.UNIFORM, Q2KPattern.BOTH],
        default=Q2KPattern.SINK,
    )
    bench.add_argument("--b", type=int, default=DEFAULT_B)
    bench.add_argument("--sq", type=int, default=DEFAULT_SQ)
    bench.add_argument("--skv", type=int, default=DEFAULT_SKV)
    bench.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    bench.add_argument("--head-kv", type=int, default=DEFAULT_HEAD_KV)
    bench.add_argument("--qhead-per-kv", type=int, default=DEFAULT_QHEAD_PER_KV)
    bench.add_argument("--dim", type=int, default=DEFAULT_DIM)
    bench.add_argument("--blk-kv", type=int, default=DEFAULT_BLK_KV)
    bench.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    bench.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    bench.add_argument("--seed", type=int, default=DEFAULT_SEED)
    bench.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    bench.add_argument("--partial-dtype", choices=["fp32", "bf16", "fp16", "fp8"], default="bf16")
    bench.add_argument("--causal", action="store_true")
    bench.add_argument("--profile", action="store_true")
    bench.add_argument(
        "--profile-skip-autotune",
        action="store_true",
        help=(
            "Run one unprofiled fwd preflight before timed profile ranges so "
            "JIT/autotune work is excluded from NCU NVTX captures."
        ),
    )
    bench.add_argument(
        "--sync-nvtx",
        action="store_true",
        help="Synchronize at benchmark NVTX boundaries for cleaner Nsight Systems timelines.",
    )
    bench.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help=(
            "Call cudaProfilerStart/Stop once around all timed sparse benchmark loops. "
            "Use with nsys --capture-range=cudaProfilerApi to exclude warmup/autotune."
        ),
    )
    bench.add_argument("--page-size", type=int, default=DEFAULT_BLK_KV)
    bench.add_argument("--seqused-trim", type=int, default=0)
    bench.add_argument(
        "--decode-kv-tokens",
        default="sweep",
        help="Decode KV token counts: 'sweep' for 8..1M powers of two, or comma list such as 8,128,1m.",
    )
    bench.add_argument("--decode-batch", type=int, default=DECODE_BATCH)
    bench.add_argument("--decode-seqlen-q", type=int, default=DECODE_SEQLEN_Q)

    args = parser.parse_args()
    dtype = _parse_dtype(args.dtype)
    partial_dtype = _parse_partial_dtype(args.partial_dtype)
    if args.decode:
        if args.nvfp4_kv or args.fp8_kv or args.mixed_fp8_qkv_pv_bf16:
            raise ValueError("--nvfp4-kv/--fp8-kv/--mixed-fp8-qkv-pv-bf16 cannot be combined with --decode")
        if args.paged:
            raise ValueError("--decode already uses paged KV; do not pass --paged")
        if args.backend != BenchmarkBackend.CUTE:
            raise ValueError("decode benchmark currently supports only --backend cute")
        _benchmark_sparse_decode_page_atten(
            kv_token_values=_parse_decode_kv_tokens(args.decode_kv_tokens),
            batch=args.decode_batch,
            seqlen_q=args.decode_seqlen_q,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            profile=args.profile,
            sync_nvtx=args.sync_nvtx,
        )
        return
    if args.nvfp4_kv:
        if args.fp8_kv or args.mixed_fp8_qkv_pv_bf16:
            raise ValueError("--nvfp4-kv cannot be combined with --fp8-kv or --mixed-fp8-qkv-pv-bf16")
        if args.backend != BenchmarkBackend.CUTE:
            raise ValueError("NVFP4 KV benchmark currently supports only --backend cute")
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("NVFP4 KV benchmark supports only --dtype bf16 or --dtype fp8")
        if args.paged and args.b != 1:
            raise ValueError("paged NVFP4 KV benchmark currently supports only batch=1")
        warmup, iters = _resolve_bench_loop_counts(args.warmup, args.iters, args.profile)
        benchmark_contexts = []
        nvfp4_kv_cases = (
            (
                {
                    "name": "manual_sparse_page_atten_nvfp4_kv",
                    "batch": args.b,
                    "seqlen_q": args.sq,
                    "seqlen_k": args.skv,
                    "head_kv": args.head_kv,
                    "qhead_per_kv": args.qhead_per_kv,
                },
            )
            if args.paged and args.customer_case == "manual"
            else _resolve_sparse_benchmark_cases(args)
        )
        for case in nvfp4_kv_cases:
            if args.paged and int(case["batch"]) != 1:
                raise ValueError("paged NVFP4 KV benchmark currently supports only batch=1")
            for q2k_pattern in _resolve_q2k_patterns(args.q2k_pattern):
                ctx = _build_sparse_nvfp4_kv_benchmark_context(
                    case_name=f"{case['name']}_nvfp4_kv",
                    q2k_pattern=q2k_pattern,
                    batch=int(case["batch"]),
                    seqlen_q=int(case["seqlen_q"]),
                    seqlen_k=int(case["seqlen_k"]),
                    head_kv=int(case["head_kv"]),
                    qhead_per_kv=int(case["qhead_per_kv"]),
                    dim=args.dim,
                    topk=args.topk,
                    blk_kv=args.blk_kv,
                    causal=args.causal,
                    seed=args.seed,
                    q_dtype=dtype,
                    paged=args.paged,
                    page_size=args.page_size,
                    seqused_trim=args.seqused_trim,
                )
                benchmark_contexts.append(ctx)

        for ctx in benchmark_contexts:
            print(
                f"warmup_autotune: case={ctx['case_name']} "
                f"q2k_pattern={ctx['q2k_pattern']} "
                f"warmup={warmup}",
                flush=True,
            )
            _run_sparse_benchmark_warmup(ctx, warmup=warmup, sync_nvtx=args.sync_nvtx)
        torch.cuda.synchronize()
        capture_labels = {"nvfp4_kv_prefill"} if args.cuda_profiler_capture else None
        benchmark_results = [
            (
                ctx,
                _run_sparse_benchmark_timed(
                    ctx,
                    iters=iters,
                    sync_nvtx=args.sync_nvtx,
                    cuda_profiler_capture_labels=capture_labels,
                ),
            )
            for ctx in benchmark_contexts
        ]
        for ctx, timings in benchmark_results:
            _print_sparse_benchmark_results(ctx, timings)
        return
    if args.fp8_kv:
        if args.mixed_fp8_qkv_pv_bf16:
            raise ValueError("--fp8-kv cannot be combined with --mixed-fp8-qkv-pv-bf16")
        if args.backend != BenchmarkBackend.CUTE:
            raise ValueError("fp8 KV benchmark currently supports only --backend cute")
        if dtype != torch.bfloat16:
            raise ValueError("fp8 KV benchmark requires --dtype bf16")
    if args.mixed_fp8_qkv_pv_bf16:
        if args.backend != BenchmarkBackend.CUTE:
            raise ValueError("mixed precision benchmark currently supports only --backend cute")
        if dtype != torch.bfloat16:
            raise ValueError("mixed precision benchmark requires --dtype bf16 for the bf16 baseline")
        if args.paged:
            raise ValueError("mixed precision benchmark currently supports non-paged sparse attention only")
    if not args.paged:
        if dtype == torch.float8_e4m3fn and args.backend != BenchmarkBackend.CUTE:
            raise ValueError("fp8 benchmark currently supports only --backend cute")
        warmup, iters = _resolve_bench_loop_counts(args.warmup, args.iters, args.profile)
        benchmark_contexts = []
        benchmark_dtypes = (
            (("bf16", torch.bfloat16), ("fp8", torch.float8_e4m3fn))
            if dtype == torch.float8_e4m3fn
            else ((args.dtype, dtype),)
        )
        for case in _resolve_sparse_benchmark_cases(args):
            for q2k_pattern in _resolve_q2k_patterns(args.q2k_pattern):
                for dtype_label, benchmark_dtype in benchmark_dtypes:
                    ctx = _build_sparse_benchmark_context(
                        case_name=f"{case['name']}_{dtype_label}",
                        backend=args.backend,
                        q2k_pattern=q2k_pattern,
                        batch=int(case["batch"]),
                        seqlen_q=int(case["seqlen_q"]),
                        seqlen_k=int(case["seqlen_k"]),
                        head_kv=int(case["head_kv"]),
                        qhead_per_kv=int(case["qhead_per_kv"]),
                        dim=args.dim,
                        topk=args.topk,
                        blk_kv=args.blk_kv,
                        causal=args.causal,
                        dtype=benchmark_dtype,
                        partial_dtype=partial_dtype,
                        seed=args.seed,
                        fp8_kv=args.fp8_kv,
                        mixed_fp8_qkv_pv_bf16=args.mixed_fp8_qkv_pv_bf16,
                    )
                    ctx["case_base_name"] = str(case["name"])
                    ctx["bench_dtype_label"] = dtype_label
                    benchmark_contexts.append(ctx)

        for ctx in benchmark_contexts:
            print(
                f"warmup_autotune: case={ctx['case_name']} "
                f"q2k_pattern={ctx['q2k_pattern']} "
                f"warmup={warmup}",
                flush=True,
            )
            _run_sparse_benchmark_warmup(ctx, warmup=warmup, sync_nvtx=args.sync_nvtx)

        if args.profile_skip_autotune:
            for ctx in benchmark_contexts:
                print(
                    f"profile_autotune_preflight: case={ctx['case_name']} "
                    f"q2k_pattern={ctx['q2k_pattern']}",
                    flush=True,
                )
                _run_sparse_benchmark_autotune(ctx, sync_nvtx=args.sync_nvtx)

        torch.cuda.synchronize()
        capture_labels = None
        if args.cuda_profiler_capture:
            if args.mixed_fp8_qkv_pv_bf16:
                capture_labels = {"mixed_fp8_qkv_pv_bf16_prefill"}
            elif args.fp8_kv:
                capture_labels = {"fp8_kv_prefill"}
            else:
                capture_labels = {args.backend}
        benchmark_results = []
        for ctx in benchmark_contexts:
            benchmark_results.append(
                (
                    ctx,
                    _run_sparse_benchmark_timed(
                        ctx,
                        iters=iters,
                        sync_nvtx=args.sync_nvtx,
                        cuda_profiler_capture_labels=capture_labels,
                    ),
                )
            )

        for ctx, timings in benchmark_results:
            _print_sparse_benchmark_results(ctx, timings)
        if dtype == torch.float8_e4m3fn:
            by_case: dict[tuple[str, str], dict[str, float]] = {}
            for ctx, timings in benchmark_results:
                if BenchmarkBackend.CUTE not in timings:
                    continue
                key = (str(ctx["case_base_name"]), str(ctx["q2k_pattern"]))
                by_case.setdefault(key, {})[str(ctx["bench_dtype_label"])] = timings[
                    BenchmarkBackend.CUTE
                ]["fwd_ms"]
            for (case_name, q2k_pattern), values in by_case.items():
                if "bf16" in values and "fp8" in values:
                    print(
                        "fp8_vs_bf16_fwd_speedup: "
                        f"case={case_name} q2k_pattern={q2k_pattern} "
                        f"speedup={values['bf16'] / values['fp8']:.2f}x "
                        f"bf16_fwd_ms={values['bf16']:.3f} fp8_fwd_ms={values['fp8']:.3f}"
                    )
    else:
        if args.backend != BenchmarkBackend.CUTE:
            raise ValueError("paged benchmark currently supports only --backend cute")
        paged_cases = (
            (
                {
                    "name": "manual_sparse_page_atten",
                    "batch": args.b,
                    "seqlen_q": args.sq,
                    "seqlen_k": args.skv,
                    "head_kv": args.head_kv,
                    "qhead_per_kv": args.qhead_per_kv,
                },
            )
            if args.customer_case == "manual"
            else _resolve_sparse_benchmark_cases(args)
        )
        for case in paged_cases:
            if int(case["batch"]) != 1:
                raise ValueError("paged benchmark currently supports only batch=1")
            for q2k_pattern in _resolve_q2k_patterns(args.q2k_pattern):
                print(
                    f"warmup_autotune: case={case['name']} "
                    f"q2k_pattern={q2k_pattern} warmup={args.warmup}",
                    flush=True,
                )
                _benchmark_sparse_page_atten(
                    q2k_pattern=q2k_pattern,
                    seqlen_q=int(case["seqlen_q"]),
                    seqlen_k=int(case["seqlen_k"]),
                    head_kv=int(case["head_kv"]),
                    qhead_per_kv=int(case["qhead_per_kv"]),
                    dim=args.dim,
                    topk=args.topk,
                    blk_kv=args.blk_kv,
                    causal=args.causal,
                    dtype=dtype,
                    partial_dtype=partial_dtype,
                    page_size=args.page_size,
                    seqused_trim=args.seqused_trim,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    profile=args.profile,
                    fp8_kv=args.fp8_kv,
                    cuda_profiler_capture=args.cuda_profiler_capture,
                )


if __name__ == "__main__":
    main()
