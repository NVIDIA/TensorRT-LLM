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
"""Unit tests for TeaCache (CPU-only, no model weights needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.visual_gen.cache.teacache import TeaCacheBackend
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, TeaCacheConfig
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


class TestSetupTeacache:
    """Tests for _setup_cache_acceleration TeaCache coefficient matching and fail-early behavior."""

    def _make_pipeline_mock(self, checkpoint_name, use_ret_steps=False):
        pipeline = MagicMock()
        pipeline.cache_accelerator = None
        pipeline.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(_name_or_path=f"/path/to/{checkpoint_name}/snapshot"),
            cache=TeaCacheConfig(
                teacache_thresh=0.3,
                use_ret_steps=use_ret_steps,
            ),
            skip_create_weights_in_init=True,
        )
        return pipeline

    def test_matching_variant_selects_coefficients(self):
        """Picks coefficients whose key appears in checkpoint path."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        coefficients = {
            "dev": {"standard": [1.0, 2.0, 3.0], "ret_steps": [4.0, 5.0]},
            "schnell": {"standard": [10.0, 20.0]},
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), coefficients)

        assert pipeline.model_config.teacache.coefficients == [1.0, 2.0, 3.0]

    def test_no_match_raises_valueerror(self):
        """Raises ValueError (fail-early) when no variant matches checkpoint."""
        pipeline = self._make_pipeline_mock("FLUX.1-unknown-variant")
        coefficients = {
            "dev": {"standard": [1.0, 2.0]},
            "schnell": {"standard": [10.0, 20.0]},
        }
        with pytest.raises(ValueError, match="No coefficients found"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), coefficients)

    def test_disabled_teacache_is_noop(self):
        """No-op when cache is None (TeaCache not selected)."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        pipeline.model_config = pipeline.model_config.model_copy(update={"cache": None})

        BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), {"dev": [1.0]})
        assert pipeline.cache_accelerator is None
