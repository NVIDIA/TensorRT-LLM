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
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.visual_gen.cache.teacache import TeaCacheBackend
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm.visual_gen.args import TeaCacheConfig


class _PipelineConfigShim:
    """Minimal pipeline_config shim that delegates all reads back to a DiffusionModelConfig.

    Shares the same TeaCacheConfig object so mutations from _apply_teacache_coefficients
    are visible via both pipeline_config.teacache and model_config.teacache.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        self._mc = model_config

    @property
    def cache_backend(self):
        return self._mc.cache_backend

    @property
    def teacache(self):
        return self._mc.teacache

    @property
    def cache_dit(self):
        return self._mc.cache_dit

    @property
    def primary_pretrained_config(self):
        return self._mc.pretrained_config

    def model_copy(self, update=None):
        if update:
            for k, v in update.items():
                setattr(self._mc, k, v)
        return self


class _PipelineTeacacheTestDouble:
    """Minimal object for BasePipeline._setup_cache_acceleration / _apply_teacache_coefficients tests."""

    def __init__(self, model_config: DiffusionModelConfig):
        self.model_config = model_config
        self.cache_accelerator = None
        self.cache_backend = None
        self.pipeline_config = _PipelineConfigShim(model_config)

    def _apply_teacache_coefficients(self, coefficients: Optional[dict] = None):
        return BasePipeline._apply_teacache_coefficients(self, coefficients)


class TestSetupTeacache:
    """Tests for _setup_cache_acceleration TeaCache coefficient matching and fail-early behavior."""

    def _make_pipeline_mock(self, checkpoint_name, use_ret_steps=False):
        return _PipelineTeacacheTestDouble(
            DiffusionModelConfig(
                pretrained_config=SimpleNamespace(
                    _name_or_path=f"/path/to/{checkpoint_name}/snapshot"
                ),
                cache=TeaCacheConfig(
                    teacache_thresh=0.3,
                    use_ret_steps=use_ret_steps,
                    coefficients=None,
                ),
                skip_create_weights_in_init=True,
            )
        )

    def _make_pipeline_teacache_enable_only(self, checkpoint_name, use_ret_steps=False):
        """Only use_ret_steps (+ defaults) so teacache_thresh may be omitted from explicit set."""
        return _PipelineTeacacheTestDouble(
            DiffusionModelConfig(
                pretrained_config=SimpleNamespace(
                    _name_or_path=f"/path/to/{checkpoint_name}/snapshot"
                ),
                cache=TeaCacheConfig(
                    use_ret_steps=use_ret_steps,
                    coefficients=None,
                ),
                skip_create_weights_in_init=True,
            )
        )

    def test_setup_cache_acceleration_raises_when_no_table_and_no_user_coefficients(self):
        """Fails if TeaCache is on but the pipeline passes no coefficient table."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        with pytest.raises(ValueError, match="no polynomial coefficients were resolved"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), None)

    def test_setup_cache_acceleration_raises_when_empty_table(self):
        """Same as no table: nothing to resolve from."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        with pytest.raises(ValueError, match="no polynomial coefficients were resolved"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), {})

    def test_matching_variant_selects_ret_steps_mode(self):
        """Nested table: use_ret_steps=True selects ret_steps coefficients."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev", use_ret_steps=True)
        coefficients = {
            "dev": {"standard": [1.0, 2.0, 3.0], "ret_steps": [4.0, 5.0]},
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), coefficients)

        assert pipeline.model_config.teacache.coefficients == [4.0, 5.0]

    def test_flat_list_table_entry(self):
        """Table value may be a plain list (no standard/ret_steps nesting)."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        coefficients = {"dev": [11.0, 22.0, 33.0]}
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), coefficients)

        assert pipeline.model_config.teacache.coefficients == [11.0, 22.0, 33.0]

    def test_default_thresh_from_table_when_user_did_not_set_teacache_thresh(self):
        """default_thresh applies when teacache_thresh was not explicitly set (exclude_unset)."""
        pipeline = self._make_pipeline_teacache_enable_only("FLUX.1-dev")
        builtin = {
            "dev": {
                "standard": [1.0, 2.0],
                "ret_steps": [3.0, 4.0],
                "default_thresh": 0.42,
            },
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), builtin)

        assert pipeline.model_config.teacache.coefficients == [1.0, 2.0]
        assert pipeline.model_config.teacache.teacache_thresh == 0.42

    def test_explicit_identity_coefficients_still_skip_table(self):
        """[1.0, 0.0] is a user override: no variant lookup, no ValueError on unknown path."""
        pipeline = self._make_pipeline_mock("FLUX.1-unknown-variant")
        pipeline.model_config.cache = TeaCacheConfig(
            teacache_thresh=0.3,
            coefficients=[1.0, 0.0],
        )
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(
                pipeline,
                MagicMock(),
                {"dev": {"standard": [99.0, 99.0]}},
            )

        assert pipeline.model_config.teacache.coefficients == [1.0, 0.0]

    def test_matching_variant_selects_coefficients(self):
        """Picks coefficients whose key appears in checkpoint path."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        coefficients = {
            "dev": {"standard": [1.0, 2.0, 3.0], "ret_steps": [4.0, 5.0]},
            "schnell": {"standard": [10.0, 20.0]},
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), coefficients)

        assert pipeline.pipeline_config.teacache.coefficients == [1.0, 2.0, 3.0]

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
        pipeline.pipeline_config = pipeline.pipeline_config.model_copy(update={"cache": None})

        BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), {"dev": [1.0]})
        assert pipeline.cache_accelerator is None

    def test_user_configured_coefficients_skip_variant_matching(self):
        """Explicit TeaCacheConfig.coefficients skips dict lookup (no ValueError)."""
        pipeline = self._make_pipeline_mock("FLUX.1-unknown-variant")
        pipeline.model_config.cache = TeaCacheConfig(
            teacache_thresh=0.3,
            coefficients=[0.25, 0.5, 0.75],
        )
        builtin = {
            "dev": {"standard": [1.0, 2.0, 3.0], "ret_steps": [4.0, 5.0]},
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), builtin)

        assert pipeline.model_config.teacache.coefficients == [0.25, 0.5, 0.75]

    def test_user_configured_coefficients_take_precedence_over_builtin_table(self):
        """User coefficients are not overwritten when a built-in variant would also match."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        pipeline.model_config.cache = TeaCacheConfig(
            teacache_thresh=0.3,
            coefficients=[9.0, 8.0, 7.0],
        )
        builtin = {
            "dev": {"standard": [1.0, 2.0, 3.0], "ret_steps": [4.0, 5.0]},
        }
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(pipeline, MagicMock(), builtin)

        assert pipeline.model_config.teacache.coefficients == [9.0, 8.0, 7.0]

    def test_apply_teacache_coefficients_only(self):
        """_apply_teacache_coefficients updates config without enabling backend."""
        pipeline = self._make_pipeline_mock("FLUX.1-unknown-variant")
        pipeline.model_config.cache = TeaCacheConfig(
            coefficients=[0.1, 0.2],
        )
        BasePipeline._apply_teacache_coefficients(pipeline, {"dev": {"standard": [99.0]}})
        assert pipeline.model_config.teacache.coefficients == [0.1, 0.2]

    def test_nested_table_missing_requested_mode_warns_then_raises_if_no_fallback(self):
        """Variant matches path but nested dict lacks 'standard' / 'ret_steps' entry for mode."""
        pipeline = self._make_pipeline_mock("FLUX.1-dev")
        # Only ret_steps; standard mode requested (use_ret_steps=False)
        coefficients = {"dev": {"ret_steps": [9.0, 8.0]}}
        # tensorrt_llm.logger does not propagate to the root logger, so pytest caplog
        # does not see these records; assert via the pipeline module's logger.warning.
        with patch("tensorrt_llm._torch.visual_gen.pipeline.logger.warning") as mock_warning:
            with pytest.raises(ValueError, match="No coefficients found"):
                BasePipeline._apply_teacache_coefficients(pipeline, coefficients)
        mock_warning.assert_called_once()
        joined = " ".join(str(a) for a in mock_warning.call_args[0])
        assert "matched variant" in joined


class TestTeaCacheConfigValidation:
    """TeaCacheConfig validation (no pipeline)."""

    def test_empty_coefficients_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            TeaCacheConfig(coefficients=[])

    def test_empty_coefficients_2_rejected(self):
        with pytest.raises(ValueError, match="coefficients_2"):
            TeaCacheConfig(coefficients_2=[])


class TestRefreshTeacacheBackends:
    """BasePipeline._refresh_teacache_backends walks _teacache_backends only."""

    def test_single_backend_refresh(self):
        shared = MagicMock()
        shared.is_enabled.return_value = True
        pipe = SimpleNamespace(_teacache_backends=[shared])
        BasePipeline._refresh_teacache_backends(pipe, 50)
        shared.refresh.assert_called_once_with(50)

    def test_refreshes_two_distinct_backends(self):
        b1 = MagicMock()
        b1.is_enabled.return_value = True
        b2 = MagicMock()
        b2.is_enabled.return_value = True
        pipe = SimpleNamespace(_teacache_backends=[b1, b2])
        BasePipeline._refresh_teacache_backends(pipe, 30)
        b1.refresh.assert_called_once_with(30)
        b2.refresh.assert_called_once_with(30)

    def test_skips_disabled_backends(self):
        active = MagicMock()
        active.is_enabled.return_value = True
        disabled = MagicMock()
        disabled.is_enabled.return_value = False
        pipe = SimpleNamespace(_teacache_backends=[active, disabled])
        BasePipeline._refresh_teacache_backends(pipe, 10)
        active.refresh.assert_called_once_with(10)
        disabled.refresh.assert_not_called()


class TestFlux2TeacacheTable:
    """FLUX.2 built-in coefficient table (dev variant)."""

    def test_flux2_dev_variant_resolves_from_checkpoint_path(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux2 import (
            FLUX2_TEACACHE_COEFFICIENTS,
        )

        pipeline = _PipelineTeacacheTestDouble(
            DiffusionModelConfig(
                pretrained_config=SimpleNamespace(
                    _name_or_path="/weights/black-forest-labs/FLUX.2-dev/snapshot"
                ),
                cache=TeaCacheConfig(teacache_thresh=0.2),
                skip_create_weights_in_init=True,
            )
        )
        with patch.object(TeaCacheBackend, "enable"):
            BasePipeline._setup_cache_acceleration(
                pipeline, MagicMock(), FLUX2_TEACACHE_COEFFICIENTS
            )
        assert pipeline.model_config.teacache.coefficients is not None
        assert len(pipeline.model_config.teacache.coefficients) >= 2


class TestExplicitTeaCacheCoefficientsRequired:
    """LTX-2 and Wan 2.2 have no built-in TeaCache tables; teacache requires user coefficients."""

    def test_ltx2_raises_when_teacache_enabled_without_coefficients(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        pipe.transformer = MagicMock()
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            cache=TeaCacheConfig(coefficients=None),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with pytest.raises(ValueError, match="LTX-2 requires explicit teacache.coefficients"):
                LTX2Pipeline.post_load_weights(pipe)

    def test_ltx2_succeeds_with_explicit_coefficients(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        pipe.transformer = MagicMock()
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            cache=TeaCacheConfig(
                coefficients=[1.0, 2.0, 3.0],
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2.register_extractor"
            ):
                with patch.object(TeaCacheBackend, "enable"):
                    LTX2Pipeline.post_load_weights(pipe)
        assert pipe.cache_accelerator is not None

    def test_wan22_raises_when_teacache_enabled_without_both_coefficient_lists(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        pipe = object.__new__(WanPipeline)
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        pipe.is_wan22 = True
        pipe.is_wan22_14b = True
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(
                _name_or_path="/models/wan14b/snapshot", boundary_ratio=0.2
            ),
            cache=TeaCacheConfig(
                coefficients=None,
                coefficients_2=None,
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.register_extractor_from_config"
            ):
                with pytest.raises(ValueError, match="Wan 2.2 TeaCache requires explicit"):
                    WanPipeline.post_load_weights(pipe)

    def test_wan22_i2v_raises_when_teacache_enabled_without_both_coefficient_lists(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        pipe = object.__new__(WanImageToVideoPipeline)
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        pipe.is_wan22 = True
        pipe.is_wan22_14b = True
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(
                _name_or_path="/models/wan720p/snapshot", boundary_ratio=0.2
            ),
            cache=TeaCacheConfig(
                coefficients=None,
                coefficients_2=None,
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v.register_extractor_from_config"
            ):
                with pytest.raises(ValueError, match="Wan 2.2 TeaCache requires explicit"):
                    WanImageToVideoPipeline.post_load_weights(pipe)

    def test_wan22_t2v_installs_two_teacache_backends_when_coefficients_provided(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        pipe = object.__new__(WanPipeline)
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        pipe.is_wan22 = True
        pipe.is_wan22_14b = True
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(_name_or_path="/wan/snapshot", boundary_ratio=0.2),
            cache=TeaCacheConfig(
                coefficients=[1.0, 2.0],
                coefficients_2=[3.0, 4.0],
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        mock_enable = MagicMock()
        backend_a = MagicMock()
        backend_a.enable = mock_enable
        backend_b = MagicMock()
        backend_b.enable = mock_enable
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.register_extractor_from_config"
            ):
                with patch(
                    "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.TeaCacheBackend"
                ) as TB:
                    TB.side_effect = [backend_a, backend_b]
                    WanPipeline.post_load_weights(pipe)
        assert TB.call_count == 2
        assert mock_enable.call_count == 2
        assert pipe.transformer_cache_backend is pipe.cache_backend
        assert pipe.transformer_2_cache_backend is not None
        assert pipe.transformer_2_cache_backend is not pipe.cache_backend

    def test_wan22_t2v_transformer_gets_coefficients_and_transformer_2_gets_coefficients_2(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        pipe = object.__new__(WanPipeline)
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        pipe.is_wan22 = True
        pipe.is_wan22_14b = True
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(_name_or_path="/wan/snapshot", boundary_ratio=0.2),
            cache=TeaCacheConfig(
                coefficients=[1.0, 2.0],
                coefficients_2=[3.0, 4.0],
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.register_extractor_from_config"
            ):
                with patch(
                    "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan.TeaCacheBackend"
                ) as TB:
                    TB.return_value = MagicMock()
                    WanPipeline.post_load_weights(pipe)

        assert TB.call_count == 2
        cfg_high = TB.call_args_list[0][0][0]
        cfg_low = TB.call_args_list[1][0][0]
        assert cfg_high.coefficients == [1.0, 2.0]
        assert cfg_low.coefficients == [3.0, 4.0]

    def test_wan22_i2v_transformer_gets_coefficients_and_transformer_2_gets_coefficients_2(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        pipe = object.__new__(WanImageToVideoPipeline)
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        pipe.is_wan22 = True
        pipe.is_wan22_14b = True
        pipe.model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(_name_or_path="/wan/snapshot", boundary_ratio=0.2),
            cache=TeaCacheConfig(
                coefficients=[5.0, 6.0],
                coefficients_2=[7.0, 8.0],
            ),
            skip_create_weights_in_init=True,
        )
        pipe.pipeline_config = _PipelineConfigShim(pipe.model_config)
        with patch.object(BasePipeline, "post_load_weights", lambda self: None):
            with patch(
                "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v.register_extractor_from_config"
            ):
                with patch(
                    "tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v.TeaCacheBackend"
                ) as TB:
                    TB.return_value = MagicMock()
                    WanImageToVideoPipeline.post_load_weights(pipe)

        assert TB.call_count == 2
        cfg_high = TB.call_args_list[0][0][0]
        cfg_low = TB.call_args_list[1][0][0]
        assert cfg_high.coefficients == [5.0, 6.0]
        assert cfg_low.coefficients == [7.0, 8.0]
