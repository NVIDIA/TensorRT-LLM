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
"""Host-only contracts for Push-Combine's common TensorRT-LLM PDL behavior."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
COMM_DIR = REPO_ROOT / "tensorrt_llm/_torch/modules/fused_moe/communication"
PUSH_SELECTOR = COMM_DIR / "nvlink_one_sided_tile_nvfp4_push.py"
PUSH_LOADER = COMM_DIR / "tile_tma_forced_loader.py"
PUSH_CPP_DIR = REPO_ROOT / "cpp/tensorrt_llm/kernels/communicationKernels/tileNvfp4"
PUSH_OP = PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushOp.cpp"
PDL_KERNELS = (
    PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushD0.cu",
    PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushK1.cu",
    PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushK4.cu",
)


def test_push_has_no_private_pdl_policy():
    loader_source = PUSH_LOADER.read_text(encoding="utf-8")
    selector_source = PUSH_SELECTOR.read_text(encoding="utf-8")
    op_source = PUSH_OP.read_text(encoding="utf-8")

    assert "validate_push_combine_pdl_mode" not in loader_source
    assert "PUSH_COMBINE_PDL_ENV" not in loader_source
    assert "validate_push_combine_pdl_mode" not in selector_source
    assert 'os.environ.get("TRTLLM_ENABLE_PDL")' not in selector_source
    assert "validateExplicitPdlMode" not in op_source
    assert 'std::getenv("TRTLLM_ENABLE_PDL")' not in op_source


@pytest.mark.parametrize("kernel_source", PDL_KERNELS, ids=lambda path: path.stem)
def test_push_data_kernels_have_pdl_launch_and_device_dependency(kernel_source):
    source = kernel_source.read_text(encoding="utf-8")

    assert "launchWithPdlWhenEnabled" in source
    assert "cudaGridDependencySynchronize" in source
