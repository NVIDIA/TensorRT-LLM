# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ..._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::k3_mnnvl_fusion_allreduce")
def _(
    input,
    gamma,
    residual,
    epsilon,
    buffer,
    buffer_flags,
    rmsnorm_fusion,
    scale=None,
    fusion_op: int = 0,
    one_shot_max_bytes: int = 1048576,
    early_trigger: bool = False,
):
    from tensorrt_llm.functional import AllReduceFusionOp

    op = AllReduceFusionOp(fusion_op)
    if op == AllReduceFusionOp.NONE and rmsnorm_fusion:
        op = AllReduceFusionOp.RESIDUAL_RMS_NORM

    if op == AllReduceFusionOp.NONE:
        return [torch.empty_like(input)]
    if op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
        return [torch.empty_like(input), torch.empty_like(residual)]
    if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        return [quant_out, torch.empty_like(residual)]
    if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        norm_out = torch.empty_like(input)
        quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        return [norm_out, quant_out, torch.empty_like(residual)]
    if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
        quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
        scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
        return [quant_fp4, scale_fp4, torch.empty_like(residual)]
    if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
        norm_out = torch.empty_like(input)
        quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
        scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
        return [norm_out, quant_fp4, scale_fp4, torch.empty_like(residual)]
    return [torch.empty_like(input)]


@torch.library.register_fake(f"{_NS}::k3_attn_res_rmsnorm_fwd")
def _(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    early_trigger: bool = False,
) -> torch.Tensor:
    return torch.empty_like(layer_residual)


@torch.library.register_fake(f"{_NS}::k3_attn_res_add_rmsnorm_fwd")
def _(
    layer_residual: torch.Tensor,
    layer_residual_add: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    early_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(layer_residual), torch.empty_like(layer_residual)


@torch.library.register_fake(f"{_NS}::mnnvl_allgather_split")
def _(input, bf16_columns, world_size, comm_buffer, buffer_flags):
    num_tokens, columns = input.shape
    return [
        input.new_empty((num_tokens, world_size * bf16_columns), dtype=torch.bfloat16),
        input.new_empty((num_tokens, world_size * (columns - bf16_columns))),
    ]


@torch.library.register_fake(f"{_NS}::mnnvl_allreduce_attn_res")
def _(input, prefix_sum, block_residual, res_weight, rms_weight, output_rms_weight, rms_eps, output_rms_eps,
      comm_buffer, buffer_flags):  # fmt: skip
    return [torch.empty_like(input), torch.empty_like(input)]
