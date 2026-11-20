# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE engine on the BF16 trtllm-gen path (declared by ``MoEDecodeOptions.bf16_decode_engine``): an
un-finalized logits-routed SwiGLU run of up to ``M_MAX`` rows takes one launch (``moe_bf16_engine.py``)
instead of ``bf16_moe_runner``'s routing + two GEMMs and returns its ``do_finalize=False`` outputs: per (token, k) a
gemm2 row and the BF16 top-k weight, through the engine's own expanded -> permuted index (rows packed).
"""

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import (
    M_MAX,
    moe_bf16_engine,
    supports_local_experts,
    supports_routing,
)
from tensorrt_llm._torch.utils import ActivationType

from ..quantization import TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K
from ..routing import RoutingMethodType

if TYPE_CHECKING:
    from .kernel_inputs import KernelInputs
    from .trtllm_bf16 import TrtllmTrtllmGenBf16Impl


def applies(impl: "TrtllmTrtllmGenBf16Impl", k: "KernelInputs") -> bool:
    """Whether this run fits the engine (host-side checks only)."""
    if k.do_finalize or k.router_logits is None or k.routing_bias is not None:
        return False
    if impl.activation_type != ActivationType.Swiglu or impl.bias:
        return False
    if impl.act_alpha is not None or impl.act_beta is not None or impl.act_clamp is not None:
        return False
    if impl.routing_method.routing_method_type != RoutingMethodType.Renormalize:
        return False
    if not supports_routing(impl.num_slots, k.top_k) or not supports_local_experts(
        impl.num_slots, impl.expert_size_per_partition
    ):
        return False
    qm = impl.quant_method
    if not qm.use_shuffled_weight or qm.weight_layout != TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K:
        return False
    x, lg = k.x, k.router_logits
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    m = x.shape[0]
    # The kernel reads x rows with 16-byte cp.async and logits rows with 16-byte vector loads.
    if not 1 <= m <= M_MAX or x.stride(1) != 1 or (x.stride(0) * 2) % 16 or x.data_ptr() % 16:
        return False
    if (
        lg.dtype != torch.bfloat16
        or tuple(lg.shape) != (m, impl.num_slots)
        or not lg.is_contiguous()
        or lg.data_ptr() % 16
    ):
        return False
    w13, w2 = impl.w3_w1_weight, impl.w2_weight
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or w13.dim() != 4 or w2.dim() != 4:
        return False
    e, hb, two_i, kb = w13.shape
    hidden, inter = hb * kb, two_i // 2
    if (
        kb != 64
        or e != impl.expert_size_per_partition
        or tuple(w2.shape) != (e, inter // 64, hidden, 64)
        or x.shape[1] != hidden
    ):
        return False
    return hidden % 128 == 0 and inter % 128 == 0


def run(
    impl: "TrtllmTrtllmGenBf16Impl", k: "KernelInputs", state: torch.Tensor
) -> list[torch.Tensor]:
    x = k.x
    m = x.shape[0]
    top_k = k.top_k
    hidden, inter = impl.w2_weight.shape[2], impl.w3_w1_weight.shape[2] // 2
    dev = x.device
    h = torch.empty((m * top_k, inter), dtype=torch.bfloat16, device=dev)
    out = torch.empty((m * top_k, hidden), dtype=torch.bfloat16, device=dev)
    wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
    e2p = torch.empty((m, top_k), dtype=torch.int32, device=dev)
    moe_bf16_engine(
        x,
        k.router_logits,
        impl.w3_w1_weight,
        impl.w2_weight,
        h,
        out,
        wts,
        e2p,
        state,
        top_k=top_k,
        local_start=impl.slot_start,
    )
    # bf16_moe_runner's un-finalized ABI: gemm2 rows, top-k weights [m, top_k], expanded -> permuted index [m, top_k]
    return [out, wts, e2p]
