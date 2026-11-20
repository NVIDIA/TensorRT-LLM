# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@dataclass(frozen=True)
class PrimsSageQuantization:
    qk_dtype: torch.dtype
    q_block_size: int
    k_block_size: int
    smooth_k: bool
    smooth_v: bool


def supports_dense_sage_profile(
    q: torch.Tensor, k: torch.Tensor, sage: PrimsSageQuantization
) -> bool:
    """Static support for a dense mask with 128-token routing blocks."""
    from tensorrt_llm._torch.attention.backends.prims_ts.sage import SAGE_K_BLOCK_SIZES

    batch, q_len, heads, dim = q.shape
    kv_len, kv_heads = k.shape[1:3]
    if kv_heads <= 0 or heads % kv_heads:
        return False
    group = heads // kv_heads
    return (
        0 < batch <= 65535
        and kv_heads <= 65535
        and dim == 128
        and 0 < q_len < 2**31
        and 0 < kv_len < 2**31
        and 0 < group <= 32
        and group & (group - 1) == 0
        and q.dtype in (torch.float16, torch.bfloat16)
        and sage.qk_dtype in (torch.float8_e4m3fn, torch.int8)
        and sage.q_block_size > 0
        and sage.q_block_size & (sage.q_block_size - 1) == 0
        and sage.k_block_size in SAGE_K_BLOCK_SIZES
    )


def _sage_operands(q, k, v, k_summary, v_summary, spec):
    from tensorrt_llm._torch.attention.backends.prims_ts import SageAttentionParams
    from tensorrt_llm._torch.attention.kernels.sage_quant import sage_quantize

    batch, q_len = q.shape[:2]
    kv_len = k.shape[1]
    qo_offsets = torch.arange(0, (batch + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
    kv_offsets = (
        qo_offsets
        if q_len == kv_len
        else torch.arange(0, (batch + 1) * kv_len, kv_len, device=q.device, dtype=torch.int32)
    )
    operands = sage_quantize(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        qo_offsets,
        kv_offsets,
        q_block_size=spec.q_block_size,
        k_block_size=spec.k_block_size,
        qk_quant_dtype=spec.qk_dtype,
        smooth_k=spec.smooth_k,
        smooth_v=spec.smooth_v,
    )
    summary_scale = None
    if k_summary is not None:
        count = k_summary.shape[1]
        summary_offsets = torch.arange(
            0, (batch + 1) * count, count, device=q.device, dtype=torch.int32
        )
        summary = sage_quantize(
            None,
            k_summary.flatten(0, 1),
            v_summary.flatten(0, 1),
            None,
            summary_offsets,
            k_block_size=1,
            qk_quant_dtype=spec.qk_dtype,
            stats=operands,
        )
        k_summary, v_summary = summary.k.view_as(k_summary), summary.v.view_as(v_summary)
        summary_scale = summary.k_scale
    params = SageAttentionParams(
        q_scale=operands.q_scale,
        k_scale=operands.k_scale,
        k_summary_scale=summary_scale,
        v_scale=operands.v_scale,
        v_mean=operands.v_mean,
    )
    return (
        operands.q.view_as(q),
        operands.k.view_as(k),
        operands.v.view_as(v),
        k_summary,
        v_summary,
        params,
    )


class PrimsRoutedAttention(OpWrapper):
    """Attention over explicit per-batch, per-KV-head routes on compact BSHD tensors.

    BSR lists or packed exact-block bits select full KV blocks. Optional summary
    K/V rows represent each omitted block once per real token in that block.
    Packed valid-token bits mask exact tokens. Omit routes for dense attention.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("distinct sparse rows for each KV head with token padding", dict(route="bsr")),
        Cell("bitmask routes with a partial final proxy block", dict(route="proxy")),
        Cell("dense block-scaled Q/K and mean-restored V", dict(route="sage")),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        q_block_size: int = 128,
        kv_block_size: int = 128,
        max_blocks_per_row: int | None = None,
        block_indptr: torch.Tensor | None = None,
        block_indices: torch.Tensor | None = None,
        exact_block_bits: torch.Tensor | None = None,
        kv_valid_bits: torch.Tensor | None = None,
        k_summary: torch.Tensor | None = None,
        v_summary: torch.Tensor | None = None,
        quantization: PrimsSageQuantization | None = None,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.attention.backends.prims_ts import (
            BlockSparseTSWrapper,
            SageAttentionConfig,
        )

        batch, q_len, heads, dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        sparse = block_indptr is not None or exact_block_bits is not None
        proxy = k_summary is not None
        if sparse and max_blocks_per_row is None:
            max_blocks_per_row = (kv_len + kv_block_size - 1) // kv_block_size
        dtype = q.dtype
        qk_dtype, value_dtype, sage = dtype, dtype, None
        if quantization is not None:
            sage = SageAttentionConfig(
                q_block_size=quantization.q_block_size,
                k_block_size=quantization.k_block_size,
                v_mean=quantization.smooth_v,
                k_summary_block_size=1,
            )
            qk_dtype, value_dtype = quantization.qk_dtype, torch.float8_e4m3fn
        geometry = (batch, q_len, kv_len, heads, kv_heads, dim, q_block_size, kv_block_size)
        profile = dict(
            device=q.device,
            max_blocks_per_row=max_blocks_per_row,
            q_data_type=qk_dtype,
            k_data_type=qk_dtype,
            v_data_type=value_dtype,
            o_data_type=dtype,
            use_kv_valid_bits=kv_valid_bits is not None,
            sparse_format="bitmask" if exact_block_bits is not None else "bsr",
            use_proxy_routes=proxy,
            sage_config=sage,
            use_block_sparse=sparse,
        )
        key = (geometry, tuple(profile.values()))
        if key not in plans:
            wrapper = BlockSparseTSWrapper()
            wrapper.plan(*geometry, **profile)
            plans[key] = wrapper
        sage_params = None
        if quantization is not None:
            q, k, v, k_summary, v_summary, sage_params = _sage_operands(
                q, k, v, k_summary, v_summary, quantization
            )
        return plans[key].run(
            q,
            k,
            v,
            sm_scale=1.0 / math.sqrt(dim),
            out=torch.empty_like(q, dtype=dtype),
            sage=sage_params,
            block_indptr=block_indptr,
            block_indices=block_indices,
            exact_block_bits=exact_block_bits,
            kv_valid_bits=kv_valid_bits,
            k_summary=k_summary,
            v_summary=v_summary,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        q_block_size: int = 128,
        kv_block_size: int = 128,
        max_blocks_per_row: int | None = None,
        block_indptr: torch.Tensor | None = None,
        block_indices: torch.Tensor | None = None,
        exact_block_bits: torch.Tensor | None = None,
        kv_valid_bits: torch.Tensor | None = None,
        k_summary: torch.Tensor | None = None,
        v_summary: torch.Tensor | None = None,
        quantization: PrimsSageQuantization | None = None,
    ) -> torch.Tensor:
        batch, q_len, heads, dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        group = heads // kv_heads
        tokens = torch.arange(kv_len, device=q.device)
        kv_blocks = tokens // kv_block_size
        out = torch.empty_like(q)
        for b in range(batch):
            valid = torch.ones_like(tokens, dtype=torch.bool)
            if kv_valid_bits is not None:
                valid = ((kv_valid_bits[b].long()[tokens // 32] >> (tokens % 32)) & 1).bool()
            for h in range(heads):
                kvh = h // group
                for row, start in enumerate(range(0, q_len, q_block_size)):
                    exact = torch.ones_like(valid)
                    if block_indptr is not None:
                        lo, hi = block_indptr[b, kvh, row : row + 2].tolist()
                        exact = torch.isin(kv_blocks, block_indices[lo:hi].long())
                    elif exact_block_bits is not None:
                        bits = exact_block_bits[b, kvh, row].long()
                        exact = ((bits[kv_blocks // 32] >> (kv_blocks % 32)) & 1).bool()
                    keys, values = k[b, :, kvh].float(), v[b, :, kvh].float()
                    allowed = valid & exact
                    if k_summary is not None:
                        keys = torch.where(
                            exact[:, None], keys, k_summary[b, kv_blocks, kvh].float()
                        )
                        values = torch.where(
                            exact[:, None], values, v_summary[b, kv_blocks, kvh].float()
                        )
                        allowed |= ~exact
                    rows = slice(start, start + q_block_size)
                    logits = (q[b, rows, h].float() @ keys.T) / math.sqrt(dim)
                    probs = logits.masked_fill(~allowed, -torch.inf).softmax(dim=-1).nan_to_num()
                    out[b, rows, h] = (probs @ values).to(q.dtype)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        q_block_size: int = 128,
        kv_block_size: int = 128,
        max_blocks_per_row: int | None = None,
        block_indptr: torch.Tensor | None = None,
        block_indices: torch.Tensor | None = None,
        exact_block_bits: torch.Tensor | None = None,
        kv_valid_bits: torch.Tensor | None = None,
        k_summary: torch.Tensor | None = None,
        v_summary: torch.Tensor | None = None,
        quantization: PrimsSageQuantization | None = None,
    ) -> None:
        def check_tensor(tensor, shape, dtype):
            assert tensor.shape == shape and tensor.dtype == dtype and tensor.device == q.device
            assert tensor.stride() == tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))

        assert q.ndim == k.ndim == v.ndim == 4
        batch, q_len, heads, dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        assert min(batch, q_len, kv_len, heads, kv_heads) > 0
        assert dim == 128 and heads % kv_heads == 0 and q.dtype in (torch.float16, torch.bfloat16)
        for tensor, shape in ((q, q.shape), (k, (batch, kv_len, kv_heads, dim)), (v, k.shape)):
            check_tensor(tensor, shape, q.dtype)
            assert tensor.data_ptr() % 16 == 0
        assert q_block_size > 0 and kv_block_size > 0
        assert (block_indices is None) == (block_indptr is None)
        assert (k_summary is None) == (v_summary is None)
        if block_indptr is None and exact_block_bits is None:
            assert k_summary is None and kv_valid_bits is None and max_blocks_per_row is None
            return
        q_blocks = math.ceil(q_len / q_block_size)
        kv_blocks = math.ceil(kv_len / kv_block_size)
        capacity = kv_blocks if max_blocks_per_row is None else max_blocks_per_row
        assert type(capacity) is int and 0 <= capacity <= kv_blocks
        if kv_valid_bits is not None:
            check_tensor(kv_valid_bits, (batch, math.ceil(kv_len / 32)), torch.uint32)
        if k_summary is not None:
            for summary in (k_summary, v_summary):
                check_tensor(summary, (batch, kv_blocks, kv_heads, dim), q.dtype)
                assert summary.data_ptr() % 16 == 0
        if exact_block_bits is not None:
            assert block_indptr is None
            words = math.ceil(kv_blocks / 32)
            check_tensor(exact_block_bits, (batch, kv_heads, q_blocks, words), torch.uint32)
            if capacity < kv_blocks:
                tail_mask = (1 << (kv_blocks - 32 * (words - 1))) - 1
                for row in exact_block_bits.reshape(-1, words).tolist():
                    count = sum(word.bit_count() for word in row[:-1])
                    assert count + (row[-1] & tail_mask).bit_count() <= capacity
            return
        check_tensor(block_indptr, (batch, kv_heads, q_blocks + 1), torch.int32)
        check_tensor(block_indices, (block_indices.numel(),), torch.int32)
        indices = block_indices.tolist()
        for offsets in block_indptr.reshape(-1, q_blocks + 1).tolist():
            for lo, hi in zip(offsets, offsets[1:]):
                assert 0 <= lo <= hi <= len(indices) and hi - lo <= capacity
                row = indices[lo:hi]
                assert all(0 <= block < kv_blocks for block in row)
                assert all(left < right for left, right in zip(row, row[1:]))


prims_routed_attention = PrimsRoutedAttention()
