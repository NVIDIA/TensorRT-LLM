# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the optional Marlin + FlashInfer RMSNorm path."""

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import (
    allow_flashinfer_fused_add_rmsnorm_with_nvfp4_marlin,
    is_nvfp4_marlin_enabled,
    model_extra_attrs,
)
from tensorrt_llm._utils import get_sm_version

skip_unless_marlin_flashinfer = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_FLASHINFER_AVAILABLE
    or not hasattr(torch.ops.trtllm, "marlin_nvfp4_gemm")
    or not (89 <= get_sm_version() < 100),
    reason="Requires FlashInfer and the Marlin operator on Ada or Hopper",
)


def test_marlin_flashinfer_rmsnorm_gate_is_enabled_by_default():
    with model_extra_attrs({"nvfp4_gemm_allowed_backends": ["marlin"]}):
        assert allow_flashinfer_fused_add_rmsnorm_with_nvfp4_marlin()


def test_marlin_flashinfer_rmsnorm_gate_can_be_explicitly_disabled():
    with model_extra_attrs(
        {
            "nvfp4_gemm_allowed_backends": ["marlin"],
            "enable_flashinfer_fused_add_rmsnorm_with_nvfp4_marlin": False,
        }
    ):
        assert not allow_flashinfer_fused_add_rmsnorm_with_nvfp4_marlin()


@torch.inference_mode()
@skip_unless_marlin_flashinfer
def test_marlin_flashinfer_fused_add_rmsnorm_matches_aten_fallback():
    """Validate O1 on both Marlin-supported GPU architecture families.

    Marlin supports SM89 Ada (including L40S) and SM90-SM99 Hopper. The same
    test executes on either family, compares the default-on FlashInfer path
    with the explicit-false ATen fallback, and validates both in-place outputs.
    """
    torch.manual_seed(0)
    hidden_size = 2688  # Nemotron Nano 30B layer-boundary hidden dimension.
    norm = RMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16, device="cuda")
    norm.weight.copy_((1 + 0.1 * torch.randn(hidden_size, device="cuda")).to(torch.bfloat16))
    hidden_states = torch.randn((4, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(hidden_states)
    marlin_attrs = {"nvfp4_gemm_allowed_backends": ["marlin"]}

    with model_extra_attrs(marlin_attrs):
        assert is_nvfp4_marlin_enabled()

    with model_extra_attrs(
        {
            **marlin_attrs,
            "enable_flashinfer_fused_add_rmsnorm_with_nvfp4_marlin": False,
        }
    ):
        expected, expected_residual = norm(hidden_states.clone(), residual.clone())

    with model_extra_attrs(marlin_attrs):
        actual, actual_residual = norm(hidden_states.clone(), residual.clone())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
