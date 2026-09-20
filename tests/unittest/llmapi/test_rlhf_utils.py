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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension


@torch.no_grad()
def test_refit_refreshes_fused_norm_cache_in_place() -> None:
    model = nn.Module()
    model.norm = nn.Module()
    original = torch.tensor([0.5, 1.0], dtype=torch.bfloat16)
    model.norm._fused_norm_weight = original
    original_ptr = original.data_ptr()

    def post_load_weights() -> None:
        model.norm._fused_norm_weight = torch.tensor([1.5, 2.0], dtype=torch.bfloat16)

    model.norm.post_load_weights = post_load_weights
    model_loader = MagicMock()
    extension = WorkerExtension.__new__(WorkerExtension)
    extension.engine = SimpleNamespace(
        model_engine=SimpleNamespace(model=model, model_loader=model_loader)
    )

    extension.finalize_weight_update()

    model_loader.finalize_update_weights.assert_called_once_with()

    assert model.norm._fused_norm_weight.data_ptr() == original_ptr
    torch.testing.assert_close(
        model.norm._fused_norm_weight,
        torch.tensor([1.5, 2.0], dtype=torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )
