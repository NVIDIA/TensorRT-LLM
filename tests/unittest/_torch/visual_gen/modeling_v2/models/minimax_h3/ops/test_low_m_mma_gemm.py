# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules import low_m_gemm
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.linear import Linear
from tensorrt_llm._torch.visual_gen.runtime import QuantAlgo, QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="warp-MMA projections require SM100 or SM103",
)


@pytest.mark.parametrize("backend", ["warp_mma", "warp_mma_fp8", "cublas_fp8"])
@torch.inference_mode()
def test_h3_low_m_projection(backend, monkeypatch):
    quantized = backend != "warp_mma"
    config = QuantConfig(quant_algo=QuantAlgo.FP8) if quantized else None
    projection = Linear(256, 128, bias=False, dtype=torch.bfloat16, quant_config=config).cuda()
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    projection._low_m_gemm_dispatcher = low_m_gemm.LowMGemmDispatcher()

    selected = []

    def choose_one(name, runners, tuning_config, inputs, **kwargs):
        assert name == ("trtllm::low_m_fp8_gemm" if quantized else "trtllm::low_m_bf16_gemm")
        assert runners[0].unique_id()[0] == ("cublas_fp8" if quantized else "cublas")
        runner = next(candidate for candidate in runners if candidate.unique_id()[0] == backend)
        tactics = runner.get_valid_tactics(inputs, None, **kwargs)
        assert tactics
        selected.append(runner.unique_id())
        return runner, tactics[0]

    monkeypatch.setattr(low_m_gemm.AutoTuner.get(), "choose_one", choose_one)
    rng = torch.Generator(device="cuda").manual_seed(19)
    x = torch.randint(-512, 513, (3, 256), generator=rng, device="cuda").bfloat16() / 16
    weight = torch.randint(-8, 9, (128, 256), generator=rng, device="cuda").bfloat16()
    projection.weight.copy_(weight)
    ref_x, ref_w = x.float(), weight.float()
    if quantized:
        projection.input_scale.fill_(1 / 16)
        projection.weight_scale.fill_(1 / 4)
        ref_x = (ref_x * 16).clamp(-448, 448).to(torch.float8_e4m3fn).float() / 16
        ref_w = ref_w / 4

    out = projection(x)
    assert selected == ([(backend, True)] if quantized else [(backend,)])
    torch.testing.assert_close(out, (ref_x @ ref_w.T).bfloat16(), rtol=0, atol=0)
