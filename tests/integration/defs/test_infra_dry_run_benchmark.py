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

"""Small, model-free benchmark for the infrastructure dry-run test context."""

from __future__ import annotations

import os

import torch

_MATRIX_SIZE = 32


def _validate_matmul(device: torch.device, dtype: torch.dtype) -> None:
    left = torch.full((_MATRIX_SIZE, _MATRIX_SIZE), 0.25, dtype=dtype, device=device)
    right = torch.full((_MATRIX_SIZE, _MATRIX_SIZE), 0.5, dtype=dtype, device=device)
    output = torch.matmul(left, right)
    expected = torch.full_like(output, _MATRIX_SIZE * 0.25 * 0.5)
    assert output.device.type == device.type
    assert output.dtype == dtype
    assert torch.isfinite(output).all().item()
    assert torch.equal(output, expected)


def _run_cpu() -> None:
    _validate_matmul(torch.device("cpu"), torch.float32)


def _run_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA is required for this infrastructure dry-run stage"
    device_count = torch.cuda.device_count()
    assert device_count > 0, "no CUDA devices are visible to the infrastructure dry run"
    for device_index in range(device_count):
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        _validate_matmul(device, torch.float16)
        torch.cuda.synchronize(device)


def test_infra_dry_run_benchmark() -> None:
    """Exercise the CPU or every CUDA device visible to the pytest runner."""
    if os.environ.get("stageName", "").startswith("CPU-"):
        _run_cpu()
    else:
        _run_cuda()
