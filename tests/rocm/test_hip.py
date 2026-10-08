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
"""Local RDNA4 tests. Native compilation occurs only when these tests actually run."""

import pytest
import torch

from tensorrt_llm.rocm.ops import rms_norm
from tensorrt_llm.rocm.runtime import architecture_name
from tensorrt_llm.rocm.validation import validate

_available = bool(
    torch.version.hip
    and torch.cuda.is_available()
    and architecture_name(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    in ("gfx1200", "gfx1201")
)
pytestmark = [
    pytest.mark.rdna4,
    pytest.mark.skipif(not _available, reason="Requires a real RDNA4 ROCm GPU and SDK"),
]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_native_hip_parity(dtype) -> None:
    report = validate("cuda:0", "hip", (dtype,))
    assert report["native_kernels_executed"]
    assert report["native_model_norms"] > 0
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]


def test_native_launches_on_pytorch_current_stream() -> None:
    with torch.inference_mode():
        x = torch.randn(3, 257, device="cuda:0")
        weight = torch.randn(257, device="cuda:0")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = rms_norm(x, weight, backend="hip")
            expected = rms_norm(x, weight, backend="torch")
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
