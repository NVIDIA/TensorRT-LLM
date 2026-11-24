# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.visual_gen.runtime import get_sm_version, logger
from tensorrt_llm._torch.visual_gen.runtime.cutlass_compat import install_cutlass_dsl_compatibility


@lru_cache(maxsize=1)
def _quack_gemm_act():
    try:
        install_cutlass_dsl_compatibility()
        from quack.gemm_interface import gemm_act
    except (ImportError, AttributeError) as exc:
        logger.warning(f"QuACK gemm_act is unavailable: {exc}")
        return None
    return gemm_act


def gate_up_swiglu_quack_available() -> bool:
    return (
        torch.cuda.is_available()
        and get_sm_version() in (100, 103)
        and _quack_gemm_act() is not None
    )


def gate_up_swiglu_quack_shape_ok(hidden_size: int, intermediate_size: int) -> bool:
    # TMA requires 16-byte alignment in each operand's contiguous dimension.
    return hidden_size % 8 == 0 and intermediate_size % 8 == 0


@torch.library.custom_op(
    "trtllm::visual_gen_v2_gate_up_swiglu_bf16", mutates_args=(), device_types="cuda"
)
def _gate_up_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or x.ndim != 2
        or weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.shape[0] % 2
        or not gate_up_swiglu_quack_shape_ok(x.shape[1], weight.shape[0] // 2)
    ):
        raise ValueError(
            "Expected BF16 x [M,K] and weight [2I,K] on one device, with K and I divisible by 8."
        )
    gemm_act = _quack_gemm_act()
    if gemm_act is None:
        raise RuntimeError("QuACK gemm_act is unavailable.")
    _, out = gemm_act(
        x.contiguous(),
        weight.t(),
        activation="swiglu",
        store_preact=False,
        tuned=False,
        concat_layout=("B",),
    )
    return out


@_gate_up_swiglu.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


class GateUpSwiGLU(OpWrapper):
    """Apply the QuACK BF16 GEMM SwiGLU epilogue to packed gate/up weight rows."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "Compiled projection with partial token and output tiles",
            dict(tokens=133, hidden=256, intermediate=200),
        ),
    )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _gate_up_swiglu(x, weight)

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        gate_weights, up_weights = weight.float().chunk(2, dim=0)
        with true_fp32_matmul():
            gate = F.linear(x.float(), gate_weights)
            up = F.linear(x.float(), up_weights)
        return (F.silu(gate) * up).to(x.dtype)


gate_up_swiglu_quack_bf16 = GateUpSwiGLU()
