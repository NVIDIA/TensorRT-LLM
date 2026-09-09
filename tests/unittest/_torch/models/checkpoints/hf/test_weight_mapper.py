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

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight


@pytest.mark.parametrize(
    "duplicate_kv",
    [HfWeightMapper()._duplicate_kv, duplicate_kv_weight],
    ids=["hf_weight_mapper", "legacy_modeling_utils"],
)
def test_duplicate_kv_bias_preserves_head_boundaries(duplicate_kv) -> None:
    bias = torch.tensor([10, 11, 20, 21])

    duplicated_bias = duplicate_kv(bias, num_kv_heads=2, tensor_parallel_size=4)

    expected_bias = torch.tensor([10, 11, 10, 11, 20, 21, 20, 21])
    torch.testing.assert_close(duplicated_bias, expected_bias)
