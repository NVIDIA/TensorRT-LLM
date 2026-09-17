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
"""The FP4 GEMM ops reject an architecture they have no kernel for before they
reach the autotuner -- and, for NVFP4, only after reading which backends the
caller allowed, since Marlin is what puts Ada/Hopper in range."""

from unittest.mock import patch

import pytest
import torch

import tensorrt_llm  # noqa: F401  # registers torch.ops.trtllm
from tensorrt_llm._torch.custom_ops import torch_custom_ops

pytestmark = pytest.mark.cpu_only

MODULE = "tensorrt_llm._torch.custom_ops.torch_custom_ops"


def nvfp4_gemm_kwargs():
    """Shapes are irrelevant: every case here is stopped before a tensor is read."""
    return dict(
        act_fp4=torch.zeros((2, 16), dtype=torch.uint8),
        weight=torch.zeros((8, 16), dtype=torch.uint8),
        act_sf=torch.zeros((128,), dtype=torch.uint8),
        weight_scale=torch.zeros((512,), dtype=torch.uint8),
        alpha=torch.ones((1,), dtype=torch.float32),
        output_dtype=torch.bfloat16,
    )


def w4a8_mxfp4_fp8_gemm_args():
    """Likewise, for the op whose activations are FP8 rather than FP4."""
    return (
        torch.zeros((2, 16), dtype=torch.float8_e4m3fn),
        torch.zeros((8, 16), dtype=torch.uint8),
        torch.zeros((128,), dtype=torch.uint8),
        torch.zeros((512,), dtype=torch.uint8),
        torch.ones((1,), dtype=torch.float32),
        torch.bfloat16,
    )


def test_w4a8_mxfp4_fp8_gemm_rejects_hopper_before_autotuning():
    with (
        patch(f"{MODULE}.get_sm_version", return_value=90),
        patch(f"{MODULE}.AutoTuner.get") as autotuner_get,
        pytest.raises(RuntimeError, match="SM90"),
    ):
        torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(*w4a8_mxfp4_fp8_gemm_args())

    autotuner_get.assert_not_called()


def test_w4a8_mxfp4_fp8_gemm_error_names_its_architectures():
    """SM120 is newer than the SM100/SM103 this mode needs, so "newer
    architectures only" was the wrong thing to tell the operator."""
    with (
        patch(f"{MODULE}.get_sm_version", return_value=120),
        patch(f"{MODULE}.AutoTuner.get"),
        pytest.raises(RuntimeError, match="SM100, SM103"),
    ):
        torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(*w4a8_mxfp4_fp8_gemm_args())


@pytest.mark.parametrize(
    ("allowed_backends", "reaches_backend_selection"),
    [
        # Marlin is the one NVFP4 backend Hopper has, so allowing it has to
        # carry the op past the architecture check and into runner selection.
        ("marlin", True),
        ("cutlass,marlin", True),
        ("cutlass,cublaslt,cuda_core", False),
        ("cutlass", False),
    ],
)
def test_nvfp4_gemm_arch_check_reads_the_allowed_backends(
    allowed_backends, reaches_backend_selection
):
    """``nvfp4_gemm`` is registered for CUDA only, so drive its Python
    implementation directly and let a sentinel from the autotuner stand in for
    "selection was reached"."""
    sentinel = RuntimeError("reached backend selection")
    inputs = nvfp4_gemm_kwargs()

    with (
        patch(f"{MODULE}.get_sm_version", return_value=90),
        patch(f"{MODULE}.AutoTuner.get", side_effect=sentinel),
        pytest.raises(RuntimeError) as excinfo,
    ):
        torch_custom_ops.nvfp4_gemm.python_impl(**inputs, allowed_backends=allowed_backends)

    if reaches_backend_selection:
        assert excinfo.value is sentinel
    else:
        assert "SM90" in str(excinfo.value)


def test_nvfp4_gemm_rejects_an_invalid_backend_before_the_arch_check():
    """A typo in ``allowed_backends`` is the caller's mistake either way, and
    naming it beats blaming the GPU."""
    inputs = nvfp4_gemm_kwargs()
    with (
        patch(f"{MODULE}.get_sm_version", return_value=90),
        pytest.raises(ValueError, match="nonesuch"),
    ):
        torch_custom_ops.nvfp4_gemm.python_impl(**inputs, allowed_backends="nonesuch")
