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
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension


@torch.no_grad()
def test_begin_weight_update_releases_capture_and_prepares_modules() -> None:
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.pre_reload_weights = MagicMock()
    model_engine = SimpleNamespace(
        model=model,
        model_loader=MagicMock(),
        unwrap_compiled_model_for_refit=MagicMock(),
    )
    extension = WorkerExtension.__new__(WorkerExtension)
    extension.engine = SimpleNamespace(model_engine=model_engine)

    extension.begin_weight_update()

    model_engine.unwrap_compiled_model_for_refit.assert_called_once_with()
    model_engine.model_loader.begin_update_weights.assert_called_once_with()
    model.layer.pre_reload_weights.assert_called_once_with()


def test_finish_weight_update_invalidates_cache_and_recaptures() -> None:
    resource_manager = object()
    model_engine = SimpleNamespace(
        restore_compiled_model_after_refit=MagicMock(),
    )
    engine = SimpleNamespace(
        model_engine=model_engine,
        resource_manager=resource_manager,
        reset_prefix_cache=MagicMock(),
    )
    extension = WorkerExtension.__new__(WorkerExtension)
    extension.engine = engine

    with patch("torch.cuda.synchronize") as synchronize:
        extension.finish_weight_update()

    engine.reset_prefix_cache.assert_called_once_with()
    synchronize.assert_called_once_with()
    model_engine.restore_compiled_model_after_refit.assert_called_once_with(
        resource_manager
    )


@torch.no_grad()
def test_finalize_weight_update_refreshes_post_load_state() -> None:
    model = nn.Module()
    model.norm = nn.Module()
    model.norm._fused_norm_weight = torch.tensor([0.5, 1.0], dtype=torch.bfloat16)

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

    torch.testing.assert_close(
        model.norm._fused_norm_weight,
        torch.tensor([1.5, 2.0], dtype=torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )
