# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The imported TP8 MoE engine rejects an unsupported phase lag before touching CUDA."""

import importlib
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- register ops before importing the target
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"


@pytest.mark.cpu_only
@pytest.mark.parametrize("lag", [4, 5, 6])
@pytest.mark.parametrize("tokens", [1, 4, 5, 8])
def test_imported_tp8_constructor_rejects_unsafe_lag(monkeypatch, lag, tokens):
    try:
        op = importlib.import_module(f"{PKG}.kernels.m3_moe.op")
    except NoTarget as no_target:
        pytest.skip(str(no_target))
    monkeypatch.setattr(op, "M3_MOE_LAG", lag)
    probe = Mock(side_effect=AssertionError("lag admission must precede CUDA access"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", probe)
    with pytest.raises(ValueError, match="fc1_done"):
        op.M3MoeState(torch.device("cpu"), tokens)
    probe.assert_not_called()
