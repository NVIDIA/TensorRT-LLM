# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The key-shard decode logits: ``cute_dsl_fp8_paged_mqa_logits_into`` with a PDL launch.

The kernel is the tree's ``FP8MQALogitsKernel`` with the arguments
``CuteDSLPagedMQALogitsRunner`` compiles for ``cute_dsl_fp8_paged_mqa_logits_into`` (FP32
accumulator, epilogue and output, one atom, the K cache with the evict-first hint), plus ``pdl``:
the launch carries programmatic stream serialization, the TMA warps wait for the earlier grids
before the first predecessor-produced load, and every CTA triggers its dependents after its last
logits store. Only the launch changes; the logits are bit-identical (the kernel's ``pdl`` contract).
The select/pack kernel that follows waits for this grid before it reads the logits.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch

COMPUTE_BLOCK_KV = 128
SPLIT_KV = 2 * COMPUTE_BLOCK_KV


@functools.lru_cache(maxsize=None)
def _compile(phys_block_kv: int, num_heads: int, head_dim: int, next_n: int, num_sms: int):
    from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.fp8_paged_mqa_logits import (  # noqa: E501
        FP8MQALogitsKernel,
    )

    n = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (cute.sym_int(), block_bytes), stride=(cute.sym_int64(), 1)
    )
    sym_b = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n, head_dim, sym_b), stride_order=(1, 0, 2)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n, sym_b), stride_order=(0, 1))
    logits_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    sym_batch = cute.sym_int()
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_batch, cute.sym_int()), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_batch,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(), 2), stride_order=(1, 0)
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = FP8MQALogitsKernel(
        block_kv=COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_next_n_atoms=1,
        atom=None,
        num_sms=num_sms,
        num_epi_subtiles=1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        kv_evict_first=True,
        pdl=True,
    )
    return cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )


def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def paged_logits_pdl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    out: torch.Tensor,
) -> None:
    """``cute_dsl_fp8_paged_mqa_logits_into(q, kv_cache, weights, context_lens, block_table,
    schedule_meta, max_context_len, out, kv_evict_first=True)`` launched with PDL.

    q [B, next_n, H, D] FP8; kv_cache [blocks, block_kv, 1, D + 4] uint8; weights [B*next_n, H]
    FP32; out FP32 [>= B*next_n, >= align256(max_context_len)] with unit column stride.
    """
    b, next_n, h, d = q.shape
    n = next_n * h
    aligned = -(-max_context_len // SPLIT_KV) * SPLIT_KV
    assert (
        out.dtype == torch.float32
        and out.shape[0] >= b * next_n
        and out.shape[1] >= aligned
        and out.stride(1) == 1
    )
    assert weights.dtype == torch.float32 and q.dtype == torch.float8_e4m3fn
    compiled = _compile(kv_cache.shape[1], h, d, next_n, num_sms(q.device))
    q_3d = q.reshape(-1).as_strided((n, d, b), (d, 1, n * d)).view(torch.uint8)
    w_2d = weights.reshape(-1).as_strided((n, b), (1, n))
    kv_flat = kv_cache.reshape(kv_cache.shape[0], -1)
    compiled(
        kv_flat,
        q_3d,
        w_2d,
        out[: b * next_n, :max_context_len],
        block_table,
        context_lens,
        schedule_meta,
        kv_cache.shape[0],
        b,
    )
