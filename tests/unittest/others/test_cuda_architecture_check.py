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
"""The build records its CUDA architectures and refuses GPUs outside them."""

import pytest
import torch

from tensorrt_llm.bindings.BuildInfo import (
    CUDA_ARCHITECTURES,
    SUPPORTED_CUDA_ARCHITECTURES,
    check_cuda_architecture_supported,
    is_cuda_architecture_built,
    is_cuda_architecture_supported,
)


@pytest.mark.cpu_only
def test_build_records_its_cuda_architectures():
    assert len(CUDA_ARCHITECTURES) > 0
    assert all(isinstance(sm, int) and sm >= 80 for sm in CUDA_ARCHITECTURES)


@pytest.mark.cpu_only
def test_only_listed_architectures_are_built():
    for sm in CUDA_ARCHITECTURES:
        assert is_cuda_architecture_built(sm)
    assert not is_cuda_architecture_built(999)
    unlisted = [
        sm for sm in SUPPORTED_CUDA_ARCHITECTURES if sm not in CUDA_ARCHITECTURES and sm != 120
    ]
    for sm in unlisted:
        assert not is_cuda_architecture_built(sm)


@pytest.mark.cpu_only
def test_built_architectures_are_supported():
    for sm in SUPPORTED_CUDA_ARCHITECTURES:
        assert is_cuda_architecture_supported(sm)
    for sm in CUDA_ARCHITECTURES:
        assert is_cuda_architecture_supported(sm)
    assert not is_cuda_architecture_supported(999)


@pytest.mark.cpu_only
def test_sm120_and_sm121_are_interchangeable():
    built = 120 in CUDA_ARCHITECTURES or 121 in CUDA_ARCHITECTURES
    assert is_cuda_architecture_built(120) == built
    assert is_cuda_architecture_built(121) == built
    assert is_cuda_architecture_supported(121)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_current_gpu_passes_the_check():
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    if not is_cuda_architecture_built(major * 10 + minor):
        pytest.skip(f"this build does not include SM {major * 10 + minor}")
    check_cuda_architecture_supported(device)
