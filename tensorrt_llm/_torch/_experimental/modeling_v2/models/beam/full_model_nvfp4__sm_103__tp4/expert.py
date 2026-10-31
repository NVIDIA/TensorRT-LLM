# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A4 SwiGLU expert on one rank's intermediate slice: the MoE expert provider.

``nvfp4_swiglu_expert`` is the provider ``beam_dispatch_combine`` calls once per selected expert,
in ascending expert order, with that expert's routed rows ``x`` ``[n, 5120]`` bf16. With the
rank's packed codes ``W``, 128x4-swizzled block scales ``S`` and FP32 scalars of expert ``e``::

    fc1, sf1 = fp4_quantize(x, g_gate[e])                         [n, 2560] u8, swizzled scales
    gate     = nvfp4_gemm(fc1, W_gate[e], sf1, S_gate[e], alpha_gate[e])     [n, 640] bf16
    up       = nvfp4_gemm(fc1, W_up[e],   sf1, S_up[e],   alpha_up[e])       [n, 640] bf16
    act      = bf16(silu(gate.float()) * up.float())                         swiglu_fp32
    fc2, sf2 = fp4_quantize(act, g_down[e])                       [n, 320] u8, swizzled scales
    y        = nvfp4_gemm(fc2, W_down[e], sf2, S_down[e], alpha_down[e])     [n, 5120] bf16

``g = 1 / input_scale`` and ``alpha = input_scale * weight_scale_2`` per projection. Gate and up
share one activation quantization only when their stored input scales are bitwise equal;
otherwise up quantizes ``x`` with its own ``g``. Every GEMM is the CUTLASS NVFP4 GEMM
(``allowed_backends="cutlass"``) with BF16 output; there is no BF16 expert path. The output is
the expert's unweighted rank partial: mixing weights, the routed scaling factor and the
tensor-parallel sum are applied by the caller.

Per expert this is two quantizer calls (three when the gate/up input scales differ), three
GEMMs and one elementwise SwiGLU stage. Expert enumeration happens on the host in
``beam_dispatch_combine``, so the path synchronizes with the device and is not CUDA-graph
capturable; a batched pre-routed backend can replace this provider without changing routing,
combination or the reduction.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import swiglu_fp32
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from .layout import GROUP

#: The only NVFP4 GEMM backend this provider runs.
GEMM_BACKENDS = "cutlass"

#: The kernel contract, as recorded in load and run records.
CONTRACT = {
    "activation_quantizer": "torch.ops.trtllm.fp4_quantize(x, g, sf_vec_size=16, "
    "sf_use_ue8m0=False, is_sf_swizzled_layout=True)",
    "gemm": "torch.ops.trtllm.nvfp4_gemm(..., output_dtype=torch.bfloat16, "
    f"allowed_backends={GEMM_BACKENDS!r})",
    "activation": "swiglu_fp32: bf16(silu(gate.float()) * up.float())",
    "g": "float32(1 / input_scale)",
    "alpha": "float32(input_scale * weight_scale_2)",
    "weight_scales": "E4M3 bytes, rank slice first, then torch.ops.trtllm.block_scale_interleave",
}


def _gemm(a: torch.Tensor, a_sf: torch.Tensor, proj: Mapping[str, torch.Tensor], e: int):
    return nvfp4_gemm(
        a,
        proj["weight"][e],
        a_sf,
        proj["weight_scale"][e],
        proj["alpha"][e : e + 1],
        torch.bfloat16,
        0,
        GEMM_BACKENDS,
    )


def _quantize(x: torch.Tensor, proj: Mapping[str, torch.Tensor], e: int):
    return fp4_quantize(x, proj["g"][e : e + 1], GROUP, False, True)


@torch.no_grad()
def nvfp4_swiglu_expert(
    rows: torch.Tensor,
    bank: Mapping[str, Mapping[str, torch.Tensor]],
    expert: int,
    shared_fc1: bool,
    record: Optional[list] = None,
) -> torch.Tensor:
    """Expert ``expert``'s rank partial for ``rows`` ``[n, 5120]`` bf16 -> ``[n, 5120]`` bf16.

    ``bank`` maps ``"gate"``, ``"up"``, ``"down"`` to the layer's banks (``weight``,
    ``weight_scale``, ``g``, ``alpha``, each indexed by global expert id); ``shared_fc1`` says
    the gate and up input scales of this expert are bitwise equal. With ``record`` (a list) the
    call's operands and intermediates are appended as one dict.
    """
    gate_p, up_p, down_p = bank["gate"], bank["up"], bank["down"]
    fc1, sf1 = _quantize(rows, gate_p, expert)
    fc1_up, sf1_up = (fc1, sf1) if shared_fc1 else _quantize(rows, up_p, expert)
    gate = _gemm(fc1, sf1, gate_p, expert)
    up = _gemm(fc1_up, sf1_up, up_p, expert)
    act = swiglu_fp32(gate, up)
    fc2, sf2 = _quantize(act["act"], down_p, expert)
    y = _gemm(fc2, sf2, down_p, expert)
    if record is not None:
        record.append(
            {
                "expert": expert,
                "rows": rows,
                "fc1": fc1,
                "sf1": sf1,
                "fc1_up": None if shared_fc1 else fc1_up,
                "sf1_up": None if shared_fc1 else sf1_up,
                "gate": gate,
                "up": up,
                "silu": act["silu"],
                "product": act["product"],
                "act": act["act"],
                "fc2": fc2,
                "sf2": sf2,
                "y": y,
                "scalars": torch.stack(
                    [
                        gate_p["g"][expert],
                        up_p["g"][expert],
                        gate_p["alpha"][expert],
                        up_p["alpha"][expert],
                        down_p["g"][expert],
                        down_p["alpha"][expert],
                    ]
                ),
            }
        )
    return y
