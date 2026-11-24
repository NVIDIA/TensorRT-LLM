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
"""NVFP4_PLUS checkpoint scales: the E8M0 to UE5M3 conversion."""

import pytest
import torch

from tensorrt_llm.quantization.utils.fp4_utils import e8m0_to_ue5m3, ue5m3_to_float

pytestmark = pytest.mark.cpu_only


def test_e8m0_to_ue5m3_is_exact_in_range() -> None:
    e8m0 = torch.arange(127 - 17, 127 + 17, dtype=torch.uint8)
    converted = ue5m3_to_float(e8m0_to_ue5m3(e8m0)).double()
    assert torch.equal(converted, torch.exp2(e8m0.double() - 127))
    # Below 2**-17 the block is negligible and becomes zero.
    assert e8m0_to_ue5m3(torch.tensor([0, 100], dtype=torch.uint8)).tolist() == [0, 0]
    with pytest.raises(ValueError):
        e8m0_to_ue5m3(torch.tensor([127 + 17], dtype=torch.uint8))
