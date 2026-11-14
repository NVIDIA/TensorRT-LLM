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
"""``trtllm::k3_route_quant``: the CuTe DSL form of ``trtllm::kimi_k3_noaux_tc_mxfp8_quant``.

Same arguments, same outputs bit for bit: top-16 expert ids (int32 [M, 16]), routing weights
(bf16 [M, 16]), the MXFP8 latent (e4m3 [M, 3584]) and its UE8M0 scales (uint8 [M, 112], linear).
The kernel is compiled on the first call for each (early trigger, PDL) pair, which must happen
outside CUDA-graph capture (the model's warmup does it).
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

import torch

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584
SF_VEC_SIZE = 32
MAX_TOKENS = 64

_lock = threading.Lock()
_compiled: Dict[Tuple[bool, bool], object] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad, e.g. the routing bias parameter.
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _use_pdl() -> bool:
    # Same switch as the C++ launches (tensorrt_llm::common::getEnvEnablePDL).
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _check(scores: torch.Tensor, bias: torch.Tensor, hidden_states: torch.Tensor) -> None:
    if not (scores.is_cuda and bias.is_cuda and hidden_states.is_cuda):
        raise ValueError("k3_route_quant: all inputs must be CUDA tensors")
    if scores.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("k3_route_quant: scores and bias must be float32")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("k3_route_quant: hidden_states must be bfloat16")
    if not (scores.is_contiguous() and bias.is_contiguous() and hidden_states.is_contiguous()):
        raise ValueError("k3_route_quant: all inputs must be contiguous")
    if scores.dim() != 2 or scores.shape[1] != NUM_EXPERTS or bias.numel() != NUM_EXPERTS:
        raise ValueError(
            f"k3_route_quant: scores must be [M, {NUM_EXPERTS}] and bias [{NUM_EXPERTS}]"
        )
    if hidden_states.shape != (scores.shape[0], HIDDEN_SIZE):
        raise ValueError(
            f"k3_route_quant: hidden_states must be [M, {HIDDEN_SIZE}] with the M of scores"
        )
    if not 0 < scores.shape[0] <= MAX_TOKENS:
        raise ValueError(f"k3_route_quant: M must be in [1, {MAX_TOKENS}]")


def _kernel(early_trigger: bool, use_pdl: bool, args, num_tokens: int, scale: float, stream):
    key = (early_trigger, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_route_quant must run once outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        from . import k3_route_quant_kernel as kernel

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_route_quant, *args, num_tokens, scale, early_trigger, use_pdl, stream
                )
    return fn


@torch.library.custom_op("trtllm::k3_route_quant", mutates_args=())
def k3_route_quant(
    scores: torch.Tensor,
    bias: torch.Tensor,
    hidden_states: torch.Tensor,
    routed_scaling_factor: float,
    early_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kimi K3 top-16 routing of ``scores`` (fp32 [M, 896]) with ``bias`` (fp32 [896]) and MXFP8
    quantization of ``hidden_states`` (bf16 [M, 3584]); M <= 64.

    Returns ``(topk_ids, topk_weights, quantized, scales)`` as ``kimi_k3_noaux_tc_mxfp8_quant``.
    ``early_trigger`` lets the dependent grid launch once every CTA has passed its own dependency
    wait (for dependents that wait for this whole grid before reading its outputs).
    """
    import cuda.bindings.driver as cuda_driver

    _check(scores, bias, hidden_states)
    num_tokens = scores.shape[0]
    device = scores.device
    topk_ids = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
    quantized = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(num_tokens, HIDDEN_SIZE // SF_VEC_SIZE, dtype=torch.uint8, device=device)
    args = (
        _arg(scores.view(-1)),
        _arg(bias.view(-1)),
        _arg(hidden_states.view(-1).view(torch.int32)),
        _arg(topk_ids.view(-1)),
        _arg(topk_weights.view(-1).view(torch.int16)),
        _arg(quantized.view(-1).view(torch.int32)),
        _arg(scales.view(-1)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    use_pdl = _use_pdl()
    scale = float(routed_scaling_factor)
    fn = _kernel(early_trigger, use_pdl, args, num_tokens, scale, stream)
    # The compiled function takes the runtime arguments only (the Constexpr ones are baked in).
    fn(*args, num_tokens, scale, stream)
    return topk_ids, topk_weights, quantized, scales


@k3_route_quant.register_fake
def _(scores, bias, hidden_states, routed_scaling_factor, early_trigger=False):
    num_tokens = scores.shape[0]
    return (
        scores.new_empty((num_tokens, TOP_K), dtype=torch.int32),
        scores.new_empty((num_tokens, TOP_K), dtype=torch.bfloat16),
        hidden_states.new_empty((num_tokens, HIDDEN_SIZE), dtype=torch.float8_e4m3fn),
        hidden_states.new_empty((num_tokens, HIDDEN_SIZE // SF_VEC_SIZE), dtype=torch.uint8),
    )
