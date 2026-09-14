# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear-scale opt-in must never change a swizzled consumer's contract."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("swizzled", [None, True, False])
def test_linear_scales_require_explicit_consumer_and_environment_opt_in(
    flag, swizzled, monkeypatch
):
    module = importlib.import_module("tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant")
    if flag is None:
        monkeypatch.delenv("TRTLLM_RMSNORM_FP4_LINEAR_SCALES", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_RMSNORM_FP4_LINEAR_SCALES", flag)
    calls = []

    def provider(x, w, **kwargs):
        calls.append(kwargs["is_sf_swizzled_layout"])
        return torch.zeros(2, 64, dtype=torch.uint8), torch.zeros(2, 8, dtype=torch.uint8)

    monkeypatch.setattr(module, "_rmsnorm_fp4quant", provider)
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    w = torch.ones(128, dtype=torch.bfloat16)
    kwargs = {} if swizzled is None else {"is_sf_swizzled_layout": swizzled}
    if swizzled is False and flag != "1":
        with pytest.raises(ValueError, match="TRTLLM_RMSNORM_FP4_LINEAR_SCALES=1"):
            module.rmsnorm_fp4_quant(x, w, 1e-6, torch.ones(1), **kwargs)
        assert calls == []
    else:
        module.rmsnorm_fp4_quant(x, w, 1e-6, torch.ones(1), **kwargs)
        assert calls == [swizzled is not False]
