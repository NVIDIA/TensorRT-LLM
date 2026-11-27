# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Qwen3Next MoE blocks quantize a context-sized input once for both expert paths.

The routed experts must take an NVFP4 input quantized ahead of them (``ConfigurableMoE.nvfp4_input_scale``)
and the shared expert's fused gate_up must be an FP8 projection with a static input scale. The blocks and
backends are built without ``__init__``; no GPU is needed.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.cutedsl.grouped_gemm.nvfp4 import (
    TrtllmCutedslGroupedGemmNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.impl_base import MoEImplBase

pytestmark = pytest.mark.cpu_only


def _bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def _nvfp4_grouped_gemm_backend() -> TrtllmCutedslGroupedGemmNvfp4Impl:
    backend = _bare(TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.fc31_input_scale = torch.tensor([448.0 * 6.0], dtype=torch.float32)
    return backend


def _block(backend, *, declared: bool = True, comm=None) -> Qwen3NextSparseMoeBlock:
    experts = _bare(ConfigurableMoE)
    experts.backend = backend
    experts.comm = comm
    experts.apply_router_weight_on_input = False
    experts.layer_load_balancer = None
    experts.moe_max_num_tokens = 16384
    gate_up = SimpleNamespace(
        quant_method=FP8QDQLinearMethod(),
        dtype=torch.bfloat16,
        input_scale=torch.tensor([0.5], dtype=torch.float32),
        force_dynamic_quantization=False,
        _locality_domain_weight_shards=None,
    )
    block = _bare(Qwen3NextSparseMoeBlock)
    block.moe_input_dual_quant = declared
    block.experts = experts
    block.shared_expert = SimpleNamespace(
        split_gate_up=False, _can_fuse_gate_up_swiglu=lambda: False, gate_up_proj=gate_up
    )
    return block


def test_dual_quant_with_nvfp4_grouped_gemm_experts():
    assert _block(_nvfp4_grouped_gemm_backend())._moe_input_dual_quant_supported()


@pytest.mark.parametrize("blocker", ["not_declared", "backend_quantizes_itself", "comm"])
def test_dual_quant_declined(blocker):
    backend = _nvfp4_grouped_gemm_backend()
    if blocker == "backend_quantizes_itself":
        # MoEImplBase's default: the backend takes no pre-quantized NVFP4 input.
        backend.prequantized_nvfp4_input_scale = lambda: MoEImplBase.prequantized_nvfp4_input_scale(
            backend
        )
    block = _block(
        backend,
        declared=blocker != "not_declared",
        comm=object() if blocker == "comm" else None,
    )
    assert not block._moe_input_dual_quant_supported()


def test_prefill_kernels_declared_by_qwen3_5_moe_only():
    """Only the Qwen3.5 MoE class declares the prefill-sized kernels (dual quant included); dense Qwen3.5 (also
    MiniCPM-V 4.6's inner LM) and Qwen3Next keep the stock paths."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )
    from tensorrt_llm._torch.models.modeling_qwen3_next import (
        Qwen3NextForCausalLM,
        _uses_prefill_kernels,
    )

    assert Qwen3_5MoeForCausalLM.prefill_kernels
    assert not Qwen3_5ForCausalLM.prefill_kernels
    assert not Qwen3NextForCausalLM.prefill_kernels
    config = ModelConfig()
    assert not _uses_prefill_kernels(config)
    config.extra_attrs["qwen3_next_prefill_kernels"] = True
    assert _uses_prefill_kernels(config)
