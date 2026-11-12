# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RMSNorm + RoPE launchers of the Kimi K3 MLA DSpark drafter.

* :func:`k3r_dspark_rmsnorm_rope`: main's ``trtllm::cute_dsl_dspark_rmsnorm_rope`` kernel and
  support check (the family's copy, ``dspark/kernels/dspark_rmsnorm_rope.py``), compiled for
  regular row-strided input, so the drafter normalizes column slices of its fused GEMM outputs
  in place instead of staging them in a contiguous copy. Main's op is compiled for compact
  rows.
* :func:`dspark_mla_ctx_norm_rope` / :func:`dspark_mla_block_rope` (the ``dspark_tail_v2`` constant):
  the fused context-latent and block kernels of ``kernels/k3r_spec/dspark_mla_rope.py``; bitwise the
  per-layer launches they replace.

Every operand is detached before the DLPack export, which refuses tensors that require grad
(module parameters do, also under inference mode).
"""

import functools

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._utils import get_sm_version

from ...dspark.kernels.dspark_rmsnorm_rope import (
    DSparkRMSNormRoPEKernel,
    is_fused_dspark_rmsnorm_rope_supported,
)
from ...kernels.k3r_spec.dspark_mla_rope import DSparkMlaBlockRoPEKernel, DSparkMlaCtxNormRoPEKernel


@functools.cache
def _compile_rmsnorm_rope_strided(
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int,
):
    """``DSparkRMSNormRoPEKernel`` for ``[rows, hidden_dim]`` input of any row stride (unit inner
    stride), writing compact ``[rows, hidden_dim]`` output."""
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    # Regular row-strided input, as is_fused_dspark_rmsnorm_rope_supported admits: the MLA
    # drafter normalizes column slices of fused GEMM outputs.
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    # None, not a fake tensor, when the kernel does not scale by a weight: the operand then does
    # not exist in the compiled signature.
    weight_fake = (
        cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (norm_dim,), stride_order=(0,))
        if apply_weight
        else None
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (freq_rows, cute.sym_int(), 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEKernel(
        hidden_dim,
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
        norm_dim=norm_dim,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def k3r_dspark_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int | None = None,
) -> torch.Tensor:
    """``cute_dsl_dspark_rmsnorm_rope`` on regular row-strided BF16 rows, read in place.

    ``weight`` may be None when ``apply_weight`` is False. The result is compact.
    """
    if apply_weight and weight is None:
        raise ValueError("k3r_dspark_rmsnorm_rope needs a weight when apply_weight is set")
    weight = weight.detach() if apply_weight else None
    if not is_fused_dspark_rmsnorm_rope_supported(x, weight, freqs, num_heads, rope_dim, norm_dim):
        raise ValueError(
            "k3r_dspark_rmsnorm_rope requires regular row-strided BF16 tensors on an SM100, "
            f"SM103, or SM107 GPU with a valid FP32 frequency view; got SM {get_sm_version()}"
        )
    original_shape = x.shape
    x_flat = x.detach().view(-1, x.shape[-1])
    output = torch.empty(x_flat.shape, dtype=x.dtype, device=x.device)
    compiled = _compile_rmsnorm_rope_strided(
        x.shape[-1],
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
        x.shape[-1] if norm_dim is None else norm_dim,
    )
    compiled(x_flat, weight, freqs.detach(), output)
    return output.view(original_shape)


@functools.cache
def _compile_dspark_mla_ctx_norm_rope(
    num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float
):
    head_dim = kv_lora_rank + rope_dim
    tokens = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, num_layers * head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_layers, kv_lora_rank), stride_order=(1, 0)
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), head_dim), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkMlaCtxNormRoPEKernel(num_layers, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_mla_ctx_norm_rope(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm + RoPE of the context latents of every drafter layer, one launch.

    Args:
        latent: ``[N, L, kv_lora_rank + rope_dim]`` BF16 with contiguous
            ``(L, head_dim)`` rows (any token stride).
        norm_weight: ``[L, kv_lora_rank]`` BF16 latent RMSNorm weights.
        positions: ``[N]`` int64 RoPE positions.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position.
        eps: RMSNorm epsilon.
    Returns:
        ``[L, N, head_dim]``: for each layer the values of
        ``cute_dsl_dspark_rmsnorm_rope(latent[:, l], norm_weight[l], freqs[positions],
        1, rope_dim, eps, True, True, False, kv_lora_rank)``, bitwise.
    """
    num_tokens, num_layers, head_dim = latent.shape
    kv_lora_rank = norm_weight.shape[1]
    rope_dim = head_dim - kv_lora_rank
    assert latent.dtype == torch.bfloat16 and norm_weight.dtype == torch.bfloat16
    assert latent.stride(2) == 1 and latent.stride(1) == head_dim
    assert norm_weight.shape[0] == num_layers and norm_weight.is_contiguous()
    assert positions.dtype == torch.int64 and positions.shape == (num_tokens,)
    assert freqs.dtype == torch.float32 and freqs.is_contiguous()
    assert freqs.shape[1:] == (rope_dim // 2, 2)
    out = torch.empty((num_layers, num_tokens, head_dim), dtype=latent.dtype, device=latent.device)
    if num_tokens == 0:
        return out
    compiled = _compile_dspark_mla_ctx_norm_rope(num_layers, kv_lora_rank, rope_dim, float(eps))
    compiled(
        latent.detach().as_strided((num_tokens, num_layers * head_dim), (latent.stride(0), 1)),
        norm_weight.detach(),
        positions.contiguous(),
        freqs.detach(),
        out.view(num_layers * num_tokens, head_dim),
    )
    return out


@functools.cache
def _compile_dspark_mla_block_rope(
    num_heads: int, qk_nope_dim: int, kv_lora_rank: int, rope_dim: int, eps: float
):
    head_dim = kv_lora_rank + rope_dim
    q_rows = cute.sym_int()
    tokens = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, qk_nope_dim + rope_dim), stride=(cute.sym_int64(), 1)
    )
    query_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, head_dim), stride=(cute.sym_int64(), 1)
    )
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (kv_lora_rank,), stride_order=(0,)
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    pool_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), cute.sym_int(), head_dim),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    pages_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (tokens,), stride_order=(0,))
    offsets_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkMlaBlockRoPEKernel(num_heads, qk_nope_dim, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        q_fake,
        query_fake,
        kv_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        pool_fake,
        pages_fake,
        offsets_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_mla_block_rope(
    q: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> None:
    """The block RoPE work of one MLA drafter layer in one launch.

    Args:
        q: ``[T, H, qk_nope_dim + rope_dim]`` BF16 query rows (unit inner stride,
            ``H`` heads per token).
        query: ``[T, H, kv_lora_rank + rope_dim]`` BF16 absorbed query; only its
            rope columns are written.
        kv: ``[T, kv_lora_rank + rope_dim]`` BF16 block latents (row-strided).
        norm_weight: ``[kv_lora_rank]`` BF16 latent RMSNorm weight.
        positions: ``[T]`` int64 RoPE positions of the block tokens.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position.
        pool: ``[pages, page_size, kv_lora_rank + rope_dim]`` paged latent pool.
        pages, offsets: ``[T]`` int64 destination page and in-page offset.
        eps: RMSNorm epsilon.

    Writes what ``cute_dsl_dspark_rope_into(q, freqs[positions], query, H, rope_dim,
    kv_lora_rank)`` writes into ``query`` and what
    ``scatter_rows_to_pool(cute_dsl_dspark_rmsnorm_rope(kv, norm_weight,
    freqs[positions], 1, rope_dim, eps, True, True, False, kv_lora_rank), pool,
    pages, offsets)`` writes into ``pool``, bitwise.
    """
    num_tokens, num_heads, q_dim = q.shape
    head_dim = query.shape[-1]
    kv_lora_rank = norm_weight.shape[0]
    rope_dim = head_dim - kv_lora_rank
    assert q.dtype == query.dtype == kv.dtype == pool.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    assert query.shape[:2] == (num_tokens, num_heads) and kv.shape == (num_tokens, head_dim)
    assert q.stride(2) == 1 and q.stride(0) == num_heads * q.stride(1)
    assert query.stride(2) == 1 and query.stride(0) == num_heads * query.stride(1)
    assert kv.stride(1) == 1 and pool.stride(2) == 1 and pool.shape[2] == head_dim
    assert positions.dtype == pages.dtype == offsets.dtype == torch.int64
    assert positions.numel() == pages.numel() == offsets.numel() == num_tokens
    assert freqs.dtype == torch.float32 and freqs.is_contiguous()
    assert freqs.shape[1:] == (rope_dim // 2, 2)
    if num_tokens == 0:
        return
    compiled = _compile_dspark_mla_block_rope(
        num_heads, q_dim - rope_dim, kv_lora_rank, rope_dim, float(eps)
    )
    compiled(
        q.detach().as_strided((num_tokens * num_heads, q_dim), (q.stride(1), 1)),
        query.detach().as_strided((num_tokens * num_heads, head_dim), (query.stride(1), 1)),
        kv.detach(),
        norm_weight.detach(),
        positions.reshape(-1).contiguous(),
        freqs.detach(),
        pool.detach(),
        pages.reshape(-1).contiguous(),
        offsets.reshape(-1).contiguous(),
    )


__all__ = [
    "dspark_mla_block_rope",
    "dspark_mla_ctx_norm_rope",
    "k3r_dspark_rmsnorm_rope",
]
