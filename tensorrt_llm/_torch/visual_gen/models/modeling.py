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
"""Base classes for VisualGen model components."""

from typing import TYPE_CHECKING

import torch.nn as nn

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxScheduler
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


class BaseDiffusionModel(nn.Module):
    """Base class for TRT-LLM VisualGen model components."""

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        self.model_config = model_config
        self.component_name = model_config.component_name
        self.pretrained_config = model_config.pretrained_config

    def register_cuda_graph_extra_key_fns(self, runner: "CUDAGraphRunner") -> None:
        """Register non-shape CUDA graph key contributors for this model."""
        sparse_config = self.model_config.attention.sparse_attention_config
        if not isinstance(sparse_config, SkipSoftmaxAttentionConfig):
            return

        disabled_until_timestep = sparse_config.disabled_until_timestep
        if disabled_until_timestep is None:
            return

        runner.register_extra_key_fn(
            "skip_softmax_phase",
            lambda *args, **kwargs: SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                kwargs.get("timestep"),
                disabled_until_timestep=disabled_until_timestep,
            ),
        )
