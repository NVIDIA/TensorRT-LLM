# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version

from ....catalog.gemm.gate_up_swiglu_quack import (
    gate_up_swiglu_quack_available,
    gate_up_swiglu_quack_bf16,
    gate_up_swiglu_quack_shape_ok,
)
from ..ops.activation import swiglu
from .linear import Linear, projection_shard_bounds


class GatedMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        intermediate_size,
        bias,
        dtype,
        config,
        reduce_output=True,
        layer_idx=None,
        fuse_bf16_gate_up_swiglu=False,
    ):
        super().__init__()
        mapping = config.mapping
        quant = config.get_quant_config()
        lo, hi = projection_shard_bounds(
            intermediate_size,
            mapping.tp_size,
            mapping.tp_rank,
            quant.quant_algo if quant is not None else None,
            config.skip_create_weights_in_init,
        )
        size = hi - lo
        kwargs = dict(
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant,
            force_dynamic_quantization=config.force_dynamic_quantization,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            use_cute_dsl_bf16_gemm=getattr(config, "use_cute_dsl_bf16_gemm", False),
            allreduce_strategy=config.allreduce_strategy,
        )
        self.gate_up_proj = Linear(
            hidden_size,
            2 * intermediate_size,
            **kwargs,
            tensor_parallel_mode="column",
            reduce_output=False,
            fused_weight_shard_indices_mapping={"gate": (0, size), "up": (size, size)},
            override_tp_sharding={"gate": (lo, hi), "up": (lo, hi)},
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            **kwargs,
            tensor_parallel_mode="row",
            reduce_output=reduce_output,
            override_tp_sharding=(lo, hi),
        )
        self.fuse_bf16_gate_up_swiglu = (
            fuse_bf16_gate_up_swiglu and gate_up_swiglu_quack_available()
        )

    def _can_fuse_gate_up_swiglu_bf16(self, x):
        proj = self.gate_up_proj
        return (
            self.fuse_bf16_gate_up_swiglu
            and gate_up_swiglu_quack_available()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim >= 2
            and not torch.is_grad_enabled()
            and not proj.has_any_quant
            and proj.tp_size == 1
            and proj.bias is None
            and proj.weight.dtype == torch.bfloat16
            and not proj.use_cute_dsl_bf16_gemm
            and not proj.use_custom_cublas_mm
            and gate_up_swiglu_quack_shape_ok(proj.in_features, proj.out_features // 2)
        )

    def forward(self, x):
        if self._can_fuse_gate_up_swiglu_bf16(x):
            h = gate_up_swiglu_quack_bf16(x.reshape(-1, x.shape[-1]), self.gate_up_proj.weight)
            return self.down_proj(h.reshape(*x.shape[:-1], h.shape[-1]))
        h = self.gate_up_proj(x)
        if (
            self.down_proj.has_fp8_block_scales
            and self.down_proj.packed_sm100
            and get_sm_version() != 107
        ):
            packed = torch.ops.trtllm.silu_and_mul_fp8_quantize_1x128_packed_ue8m0(
                h.reshape(-1, h.shape[-1]), None, False
            )
            out = self.down_proj(packed)
            return out.reshape(*h.shape[:-1], out.shape[-1])
        scale = self.down_proj.input_scale if self.down_proj.has_fp8_qdq else None
        return self.down_proj(swiglu(h, scale))
