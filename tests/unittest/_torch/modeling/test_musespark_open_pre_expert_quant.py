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


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """The E2M1 codes of packed NVFP4 values, two per byte, low nibble first."""
    packed = packed.view(torch.uint8).to(torch.int64)
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def _dequantize(packed: torch.Tensor, scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    codes = _codes(packed)
    magnitudes = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)[codes & 7]
    values = torch.where(codes >= 8, -magnitudes, magnitudes)
    blocks = scales.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, dim=-1)
    return values * blocks / global_scale


@fused_available
@pytest.mark.parametrize("gains", ["zero_centered", "small"])
def test_the_fused_path_agrees_with_the_norm_then_the_experts_quantization(gains):
    """Within one rounding step of `pre_expert_norm` followed by the experts' own quantization.

    The two round differently (see `prepare_expert_input`), so a value near an
    FP4 rounding boundary, or a block maximum near an E4M3 one, may land one
    step away. `small` gains (|w| < 2^-8) lose most of `w` to the BF16 `1 + w`.
    """
    torch.manual_seed(845)
    with torch.inference_mode():
        module = _module("cuda")
        weight = module.pre_expert_norm.weight
        if gains == "zero_centered":
            weight.normal_(0, 0.1)
        else:
            weight.uniform_(-0.99 * 2**-8, 0.99 * 2**-8)
        x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
        eps = module.pre_expert_norm.variance_epsilon
        rms = x.double().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        exact = x.double() / rms * (1 + weight.double())
        backend = module.experts.backend
        # A calibrated input scale: the largest block maximum maps to the E4M3 maximum.
        global_scale = 448.0 * 6.0 / exact.abs().max().item()
        backend.fc31_input_scale.fill_(global_scale)
        # What the experts' BF16 input branch reads: their packed width (no padding here).
        backend.w3_w1_weight = torch.empty(0, 2048, dtype=torch.uint8, device="cuda")
        backend.scaling_vector_size = 16

        fused = module.prepare_expert_input(x)
        assert isinstance(fused, Fp4QuantizedTensor)
        packed, scales = TRTLLMGenNvfp4Base.quantize_input(backend, module.pre_expert_norm(x))

        scale_steps = (
            fused.scaling_factor.view(torch.uint8).int() - scales.view(torch.uint8).int()
        ).abs()
        assert scale_steps.max().item() <= 1
        assert (scale_steps != 0).double().mean().item() <= 0.05
        codes, reference_codes = _codes(fused.fp4_tensor), _codes(packed)
        assert ((codes & 7) - (reference_codes & 7)).abs().max().item() <= 1
        nonzero = ((codes & 7) != 0) & ((reference_codes & 7) != 0)
        assert not (((codes ^ reference_codes) & 8) != 0)[nonzero].any(), "a sign flipped"
        assert (codes != reference_codes).double().mean().item() <= 0.02

        error = (_dequantize(fused.fp4_tensor, fused.scaling_factor, global_scale) - exact).norm()
        reference_error = (_dequantize(packed, scales, global_scale) - exact).norm()
        assert error.item() <= 1.01 * reference_error.item()


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
