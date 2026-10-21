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
"""MoE input-layout and CUDA graph contracts for fused pre-expert quantization."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant_available
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenNvfp4Base
from tensorrt_llm._torch.utils import Fp4QuantizedTensor


def _module(device, comm=None):
    module = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(module)
    module.register_buffer("_pre_expert_quant_gain", None, persistent=False)
    module.pre_expert_norm = RMSNorm(
        hidden_size=4096, eps=1e-5, dtype=torch.bfloat16, device=device, use_gemma=True
    )
    backend = SimpleNamespace(
        op_backend=TRTLLMOpBackend(),
        has_any_quant=True,
        has_w4a8_mxfp4_fp8=False,
        has_nvfp4=True,
        fc31_input_scale=torch.tensor([448.0], dtype=torch.float32, device=device),
    )
    module.experts = SimpleNamespace(backend=backend, comm=comm)
    return module


@pytest.mark.parametrize("comm", [None, object()])
def test_pre_expert_quant_preserves_fallback(comm):
    module = _module("cpu", comm)
    x = torch.randn(2, 4096, dtype=torch.float32)
    torch.testing.assert_close(module.prepare_expert_input(x), module.pre_expert_norm(x))
    assert module._pre_expert_quant_gain is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reason", ["comm", "awq", "chunks", "backend"])
def test_pre_expert_quant_unsupported_consumers(reason):
    module = _module("cuda")
    if reason == "comm":
        module.experts.comm = object()
    elif reason == "awq":
        module.experts.backend.fc31_act_scale = torch.ones(4096, device="cuda")
    elif reason == "backend":
        module.experts.backend.op_backend = object()
    else:
        module.experts.moe_max_num_tokens = 1
    x = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(module.prepare_expert_input(x), module.pre_expert_norm(x))
    assert module._pre_expert_quant_gain is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_fp4_quant_available(),
    reason="CUDA and FlashInfer fused norm/quantization required",
)
def test_pre_expert_quant_without_fused_kernel_keeps_the_norm(monkeypatch):
    """Without the fused norm/quantization kernel, a configuration where the
    fused path would otherwise engage keeps the unfused norm."""
    with torch.inference_mode():
        module = _module("cuda")
        x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
        assert isinstance(module.prepare_expert_input(x), Fp4QuantizedTensor)
        monkeypatch.setattr(modeling_musespark_open, "rmsnorm_fp4_quant_available", lambda: False)
        unfused = _module("cuda")
        torch.testing.assert_close(unfused.prepare_expert_input(x), unfused.pre_expert_norm(x))
        assert unfused._pre_expert_quant_gain is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_fp4_quant_available(),
    reason="CUDA and FlashInfer fused norm/quantization required",
)
def test_pre_expert_quant_moe_layout_and_graph_replay():
    torch.manual_seed(812)
    with torch.inference_mode():
        module = _module("cuda")
        x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
        output = module.prepare_expert_input(x)
        assert isinstance(output, Fp4QuantizedTensor)
        assert not output.is_sf_swizzled
        # Unbound on the NVFP4 leaf: since the TRTLLMGen split, the family
        # name is an alias of the abstract base, whose quantize_input is a
        # typed stub. The Fp4QuantizedTensor branch reads nothing off self,
        # so the duck-typed backend namespace still works here.
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
