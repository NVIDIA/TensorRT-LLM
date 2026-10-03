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

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger


@functools.lru_cache(maxsize=1)
def _quack_gemm_act():
    """Import QuACK's gemm_act and compile it once on a probe problem.

    The pinned QuACK release is compiled against a range of CUTLASS DSL builds; when
    the installed DSL does not match, the failure surfaces at kernel compile time
    (for example a tile-scheduler tuple mismatch), not at import. Probing here keeps
    GatedMLP on the unfused path in that environment instead of failing the model's
    first forward.
    """
    try:
        from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility

        install_cutlass_dsl_compatibility()  # CuTe aliases the pinned QuACK release still imports
        from quack.gemm_interface import gemm_act

        x = torch.randn((64, 256), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((512, 256), device="cuda", dtype=torch.bfloat16)
        _, out = gemm_act(
            x,
            weight.t(),
            activation="swiglu",
            store_preact=False,
            tuned=False,
            concat_layout=("B",),
        )
        torch.cuda.synchronize()
        if out.shape != (64, 256):
            raise RuntimeError(f"unexpected probe output shape {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001  (any import or compile failure means: stay unfused)
        logger.warning(
            f"QuACK gemm_act unavailable; GatedMLP keeps the unfused SwiGLU path: {exc!r}"
        )
        return None
    return gemm_act


def gate_up_swiglu_quack_available() -> bool:
    """True on SM100/SM103 GPUs with QuACK importable (the kernel is not validated on SM107)."""
    return (
        torch.cuda.is_available()
        and get_sm_version() in (100, 103)
        and _quack_gemm_act() is not None
    )


@torch.library.custom_op("trtllm::gate_up_swiglu_quack_bf16", mutates_args=(), device_types="cuda")
def gate_up_swiglu_quack_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``silu(x @ W_gate^T) * (x @ W_up^T)`` for the fused ``weight = [W_gate; W_up]``.

    Args:
        x: ``[M, K]`` BF16 activations.
        weight: ``[2I, K]`` BF16 ``GatedMLP.gate_up_proj.weight`` (gate rows first).

    Returns:
        Contiguous ``[M, I]`` BF16.
    """
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.shape[0] % 2
    ):
        raise ValueError(
            f"unsupported x {tuple(x.shape)} {x.dtype} / weight {tuple(weight.shape)} {weight.dtype}"
        )
    gemm_act = _quack_gemm_act()
    if gemm_act is None:
        raise RuntimeError("QuACK gemm_act is not available")
    _, out = gemm_act(
        x.contiguous(),
        weight.t(),
        activation="swiglu",
        store_preact=False,
        tuned=False,
        concat_layout=("B",),
    )
    return out


@gate_up_swiglu_quack_bf16.register_fake
def _(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))
