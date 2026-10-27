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
"""BF16 gate/up GEMM with SwiGLU applied in the GEMM epilogue (QuACK ``gemm_act``, SM100 family).

``GatedMLP`` runs the gate/up GEMM into a ``[M, 2I]`` BF16 buffer and then a SwiGLU kernel that
re-reads it and writes ``silu(gate) * up`` as ``[M, I]``. QuACK's SM100 GEMM applies the
activation to the FP32 accumulators in its epilogue and stores only the ``[M, I]`` product, so the
``[M, 2I]`` intermediate is never written or read back and the output is rounded to BF16 once.

The fused ``GatedMLP`` weight keeps TRT-LLM's ``[gate; up]`` row layout; ``concat_layout=("B",)``
tells the kernel that the N dimension of ``B = W^T`` is that concatenation, so no weight copy is
needed. The tile configuration is QuACK's SM100 default (``tuned=False``): its autotuner keys on
exact shapes and would re-tune for every new token count.
"""

import functools

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

# Cleared the first time the QuACK kernel raises; every later call takes the unfused path.
_kernel_state = {"ok": True}


@functools.lru_cache(maxsize=1)
def _quack_gemm_act():
    """Import QuACK's gemm_act once; None when the pinned release does not import.

    A QuACK release built against another CUTLASS DSL fails at import with an
    ``AttributeError`` on a missing CuTe symbol, so that counts as unavailable too.
    """
    try:
        from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility

        install_cutlass_dsl_compatibility()  # CuTe aliases the pinned QuACK release still imports
        from quack.gemm_interface import gemm_act
    except (ImportError, AttributeError) as exc:
        logger.warning(
            f"QuACK gemm_act unavailable; GatedMLP keeps the unfused SwiGLU path: {exc!r}"
        )
        return None
    return gemm_act


def gate_up_swiglu_quack_available() -> bool:
    """True on SM100/SM103 GPUs with QuACK importable and its kernel not yet failed.

    The kernel is not validated on SM107. QuACK is JIT-compiled against the installed CUTLASS
    DSL, so a release that imports can still fail when the kernel is first built; after that
    ``gate_up_swiglu_quack_bf16`` computes the unfused result and this returns False.
    """
    return (
        _kernel_state["ok"]
        and torch.cuda.is_available()
        and get_sm_version() in (100, 103)
        and _quack_gemm_act() is not None
    )


def _disable_kernel(exc: BaseException) -> None:
    if _kernel_state["ok"]:
        _kernel_state["ok"] = False
        logger.warning(
            "QuACK gemm_act failed; GatedMLP uses the unfused SwiGLU path from now on. The "
            f"pinned QuACK release and the installed CUTLASS DSL may not match: {exc!r}"
        )


def _unfused_gate_up_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x, weight).chunk(2, dim=-1)
    return F.silu(gate) * up


def gate_up_swiglu_quack_shape_ok(hidden_size: int, intermediate_size: int) -> bool:
    """Whether ``[M, K] x [2I, K]`` is a shape the kernel accepts.

    TMA moves 16-byte rows, so the contiguous dimension of every operand must be a
    multiple of eight BF16 elements: K for the activations and the weight, I for the
    ``[M, I]`` output. Partial tiles along M and N are handled by the kernel.
    """
    return hidden_size % 8 == 0 and intermediate_size % 8 == 0


@torch.library.custom_op("trtllm::gate_up_swiglu_quack_bf16", mutates_args=(), device_types="cuda")
def gate_up_swiglu_quack_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``silu(x @ W_gate^T) * (x @ W_up^T)`` for the fused ``weight = [W_gate; W_up]``.

    Args:
        x: ``[M, K]`` BF16 activations.
        weight: ``[2I, K]`` BF16 ``GatedMLP.gate_up_proj.weight`` (gate rows first).

    Returns:
        Contiguous ``[M, I]`` BF16. Computed without the kernel (gate/up GEMM, then SwiGLU)
        when QuACK is unavailable or its kernel has failed once in this process.
    """
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.shape[0] % 2
        or not gate_up_swiglu_quack_shape_ok(x.shape[1], weight.shape[0] // 2)
    ):
        raise ValueError(
            "gate_up_swiglu_quack_bf16 needs BF16 x [M, K] and weight [2I, K] with K and I "
            f"multiples of 8; got x {tuple(x.shape)} {x.dtype} / weight {tuple(weight.shape)} "
            f"{weight.dtype}"
        )
    gemm_act = _quack_gemm_act()
    if gemm_act is None or not _kernel_state["ok"]:
        return _unfused_gate_up_swiglu(x, weight)
    try:
        _, out = gemm_act(
            x.contiguous(),
            weight.t(),
            activation="swiglu",
            store_preact=False,
            tuned=False,
            concat_layout=("B",),
        )
    except Exception as exc:  # noqa: BLE001 - third-party JIT; any failure means "no kernel"
        _disable_kernel(exc)
        return _unfused_gate_up_swiglu(x, weight)
    return out


@gate_up_swiglu_quack_bf16.register_fake
def _(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))
