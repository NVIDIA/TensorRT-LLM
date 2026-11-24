# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.runtime import get_sm_version, logger

from ....catalog.activation.swiglu import swiglu
from ....catalog.activation.swiglu_block_fp8 import swiglu_block_fp8
from ....catalog.gemm.gate_up_swiglu_quack import gate_up_swiglu_quack_available as quack_available
from ....catalog.gemm.gate_up_swiglu_quack import gate_up_swiglu_quack_bf16 as quack_gate_up_swiglu
from ....catalog.gemm.gate_up_swiglu_quack import gate_up_swiglu_quack_shape_ok
from .linear import Linear, projection_shard_bounds

_quack_failed = False


def gate_up_swiglu_quack_available() -> bool:
    return not _quack_failed and quack_available()


# Keep the BF16 fallback inside the compiled op boundary.
@torch.library.custom_op(
    "trtllm::minimax_h3_v2_gate_up_swiglu", mutates_args=(), device_types="cuda"
)
def gate_up_swiglu_quack_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _quack_failed
    if not _quack_failed:
        try:
            return quack_gate_up_swiglu(x, weight)
        except Exception as exc:
            _quack_failed = True
            logger.warning(f"QuACK gemm_act failed. Using unfused BF16 SwiGLU: {exc}")
    gate, up = F.linear(x, weight).chunk(2, dim=-1)
    return F.silu(gate) * up


@gate_up_swiglu_quack_bf16.register_fake
def _(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


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
            packed = swiglu_block_fp8(h.reshape(-1, h.shape[-1]))
            out = self.down_proj(packed)
            return out.reshape(*h.shape[:-1], out.shape[-1])
        scale = self.down_proj.input_scale if self.down_proj.has_fp8_qdq else None
        return self.down_proj(swiglu(h, scale))
