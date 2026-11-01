# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pre-expert norm fused with the experts' NVFP4 input quantization: gating, layout, graphs."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant_available
from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenNvfp4Base
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

fused_available = pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_fp4_quant_available(),
    reason="CUDA and FlashInfer's fused norm and quantization required",
)


def _module(device, comm=None):
    module = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(module)
    module.register_buffer("_pre_expert_quant_gain", None, persistent=False)
    module.pre_expert_norm = RMSNorm(
        hidden_size=4096, eps=1e-5, dtype=torch.bfloat16, device=device, use_gemma=True
    )
    backend = SimpleNamespace(
        op_backend=TRTLLMOpBackend(),
        has_nvfp4=True,
        fc31_input_scale=torch.tensor([448.0], dtype=torch.float32, device=device),
    )
    module.experts = SimpleNamespace(backend=backend, comm=comm)
    return module


@pytest.mark.parametrize("comm", [None, object()])
def test_cpu_inputs_take_the_norm(comm):
    module = _module("cpu", comm)
    x = torch.randn(2, 4096, dtype=torch.float32)
    torch.testing.assert_close(module.prepare_expert_input(x), module.pre_expert_norm(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reason", ["comm", "awq", "chunks", "backend", "quant", "scale"])
def test_unsupported_consumers_take_the_norm(reason):
    module = _module("cuda")
    backend = module.experts.backend
    if reason == "comm":
        module.experts.comm = object()
    elif reason == "awq":
        backend.fc31_act_scale = torch.ones(4096, device="cuda")
    elif reason == "backend":
        backend.op_backend = object()
    elif reason == "quant":
        backend.has_nvfp4 = False
    elif reason == "scale":
        backend.fc31_input_scale = torch.ones(2, device="cuda")
    else:
        module.experts.moe_max_num_tokens = 1
    x = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(module.prepare_expert_input(x), module.pre_expert_norm(x))


@fused_available
def test_without_the_fused_kernel_the_norm_is_kept(monkeypatch):
    with torch.inference_mode():
        module = _module("cuda")
        x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
        assert isinstance(module.prepare_expert_input(x), Fp4QuantizedTensor)
        monkeypatch.setattr(modeling_musespark_open, "rmsnorm_fp4_quant_available", lambda: False)
        torch.testing.assert_close(module.prepare_expert_input(x), module.pre_expert_norm(x))


@fused_available
def test_the_moe_takes_the_layout_and_graphs_replay_it():
    torch.manual_seed(812)
    with torch.inference_mode():
        module = _module("cuda")
        x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
        output = module.prepare_expert_input(x)
        assert isinstance(output, Fp4QuantizedTensor)
        assert not output.is_sf_swizzled
        # The quantized-input branch reads nothing off the module, so the
        # namespace stands in for the NVFP4 experts.
        packed, scales = TRTLLMGenNvfp4Base.quantize_input(module.experts.backend, output)
        assert packed.data_ptr() == output.fp4_tensor.data_ptr()
        assert scales.data_ptr() == output.scaling_factor.data_ptr()
        assert packed.shape == (2, 2048)
        assert scales.shape == (2, 256)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module.prepare_expert_input(x)
        x.normal_()
        expected = module.prepare_expert_input(x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.fp4_tensor, expected.fp4_tensor, rtol=0, atol=0)
        torch.testing.assert_close(captured.scaling_factor, expected.scaling_factor, rtol=0, atol=0)


@fused_available
def test_the_cached_gain_matches_the_one_computed_per_call():
    torch.manual_seed(813)
    with torch.inference_mode():
        module = _module("cuda")
        module.pre_expert_norm.weight.normal_(0, 0.1)
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        uncached = module.prepare_expert_input(x)
        module.cache_derived_state()
        weight = module.pre_expert_norm.weight
        torch.testing.assert_close(
            module._pre_expert_quant_gain, (weight.float() + 1).to(weight.dtype), rtol=0, atol=0
        )
        cached = module.prepare_expert_input(x)
        torch.testing.assert_close(cached.fp4_tensor, uncached.fp4_tensor, rtol=0, atol=0)
        torch.testing.assert_close(cached.scaling_factor, uncached.scaling_factor, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["post_load_weights", "cache_derived_state"])
def test_the_gain_cache_follows_the_weight(stage):
    """Filled by either loader walk, skipped while meta, dropped by a dtype move."""
    with torch.device("meta"):
        module = _module("meta")
    getattr(module, stage)()
    assert module._pre_expert_quant_gain is None

    module = _module("cpu")
    getattr(module, stage)()
    assert module._pre_expert_quant_gain is not None
    module.to(torch.float32)
    assert module._pre_expert_quant_gain is None
