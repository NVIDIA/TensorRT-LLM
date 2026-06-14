# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom ops for linear layers."""

from typing import List, Optional

import torch

from ..._compat import get_sm_version

# DeepSeek-V3 MLA fused a-projection shape (q_a 1536 + kv_a_with_mqa 576 = 2112,
# in 7168, bf16).  torch_linear_simple routes this exact GEMM to the trtllm
# dsv3 min-latency kernel; the kernel is only instantiated for these dims.
_DSV3_A_IN = 7168
_DSV3_A_FUSED_OUT = 2112


@torch.library.custom_op("auto_deploy::torch_linear_simple", mutates_args=())
def simple(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """A wrapper for the linear functional to control how it is exposed.

    By default ``F.linear`` (used in linear layers) is represented as a call to
    ``torch.ops.aten.linear.default`` wrapped with two ``view`` ops to flatten/unflatten
    multiple batch dimensions into one batch dimension. This wrapper avoids exposing
    that reshape pattern during export.

    Args:
        input: Input activations passed to ``torch.nn.functional.linear``
            (``input @ weight.T + bias``). Shape is typically ``(..., in_features)``.
        weight: Weight matrix of shape ``(out_features, in_features)``.
        bias: Optional bias vector of shape ``(out_features,)``. If ``None``, no bias
            is applied.
        tp_mode: TP sharding mode hint (see "Sharding hint arguments" below).
        output_sizes: Fused-weight group sizes hint (see below).
        tp_min_local_shape: Minimum per-rank output width hint (see below).
        layer_type: Layer classification hint for selective sharding (see below).

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``tp_mode``: TP sharding mode. ``"colwise"`` shards weight dim 0,
        ``"rowwise"`` shards weight dim 1, ``"none"`` skips sharding.
        ``output_sizes``: Group sizes for fused-weight proportional column sharding
        (e.g., ``[q_dim, kv_dim, kv_dim]`` for fused QKV).
        ``tp_min_local_shape``: Minimum output size per rank after sharding. Used for
        GQA where ``num_kv_heads < tp_size`` (set to ``head_dim``).
        ``layer_type``: Layer classification for selective sharding via
        ``shard_layers`` config. Values: ``"mha"``, ``"mla"``, ``"mlp"``,
        ``"moe"``, ``"ssm"``, ``"delta"``, ``"unknown"``.

    These hint arguments do not change the numeric result of the linear; they only
    guide graph transforms when tensor-parallel sharding is applied.

    Returns:
        Output tensor of shape ``(..., out_features)``.
    """
    # Backend selection (most-specific first): the DeepSeek-V3 MLA a-projection
    # (q_a + kv_a_with_mqa fused by fuse_gemms into a 7168->2112 bf16 GEMM with no
    # bias) dispatches to the trtllm dsv3 min-latency kernel (cuBLAS fallback for
    # num_tokens > 16, handled inside the op).  Strictly shape+dtype gated.
    if (
        bias is None
        and weight.dtype == torch.bfloat16
        and weight.shape[0] == _DSV3_A_FUSED_OUT
        and weight.shape[1] == _DSV3_A_IN
    ):
        return torch.ops.auto_deploy.dsv3_fused_a_gemm(input, weight)
    # Blackwell (sm>=100) + bf16: route any other bf16 linear to trtllm::cublas_mm.
    # Selects single-pass cluster-mode cubins instead of cuBLAS-default
    # split-K + reduce + zero-fill for small-M (decode) projection GEMMs.
    # (Same trick PT introduced for GPT-OSS via use_custom_cublas_mm in
    # modeling_gpt_oss.py; we apply it model-agnostically based on dtype + SM.)
    if get_sm_version() >= 100 and input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        # cublas_mm requires 2D mat_a/mat_b. Flatten leading dims and unflatten on exit.
        in_shape = input.shape
        input_2d = input.reshape(-1, in_shape[-1])
        out_2d = torch.ops.trtllm.cublas_mm(
            input_2d,
            weight.t(),
            bias,
            None,  # out_dtype
            0,  # output_buffer_kind = DEFAULT
            None,  # group (no TP)
        )
        return out_2d.view(*in_shape[:-1], out_2d.shape[-1])
    return torch.ops.aten.linear(input, weight, bias)


@simple.register_fake
def simple_fake(
    input,
    weight,
    bias,
    tp_mode="none",
    output_sizes=None,
    tp_min_local_shape=1,
    layer_type="unknown",
):
    """Fake implementation of simple_linear."""
    return torch.ops.aten.linear(input, weight, bias)
