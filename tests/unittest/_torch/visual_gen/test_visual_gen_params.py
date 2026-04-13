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
"""Tests for VisualGenParams, ExtraParamSchema, pipeline defaults, default merging, and validation."""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# VisualGenParams — Pydantic validation
# =============================================================================


class TestVisualGenParamsValidation:
    """VisualGenParams is a Pydantic StrictBaseModel with correct defaults."""

    def test_default_construction(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams()
        # Universal fields default to None
        assert params.height is None
        assert params.width is None
        assert params.num_inference_steps is None
        assert params.guidance_scale is None
        assert params.max_sequence_length is None
        assert params.num_frames is None
        assert params.frame_rate is None
        assert params.negative_prompt is None
        assert params.image is None
        assert params.mask is None
        assert params.image_cond_strength is None
        # Concrete defaults
        assert params.seed == 42
        assert params.num_images_per_prompt == 1
        # Extra params
        assert params.extra_params is None

    def test_explicit_values(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(
            height=720,
            width=1280,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=123,
        )
        assert params.height == 720
        assert params.width == 1280
        assert params.num_inference_steps == 50
        assert params.guidance_scale == 5.0
        assert params.seed == 123

    def test_unknown_field_rejected(self):
        from pydantic import ValidationError

        from tensorrt_llm.visual_gen import VisualGenParams

        with pytest.raises(ValidationError):
            VisualGenParams(stg_scale=0.5)

    def test_extra_params_accepted(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(extra_params={"stg_scale": 0.5, "enhance_prompt": True})
        assert params.extra_params["stg_scale"] == 0.5
        assert params.extra_params["enhance_prompt"] is True

    def test_image_accepts_str(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image="/path/to/image.png")
        assert params.image == "/path/to/image.png"

    def test_image_accepts_bytes(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image=b"\x89PNG")
        assert params.image == b"\x89PNG"

    def test_image_accepts_list(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(image=["/path/a.png", b"\x89PNG"])
        assert len(params.image) == 2

    def test_model_dump(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(height=512, seed=1)
        d = params.model_dump()
        assert d["height"] == 512
        assert d["seed"] == 1
        assert d["width"] is None

    def test_negative_prompt_on_params(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        params = VisualGenParams(negative_prompt="blurry, low quality")
        assert params.negative_prompt == "blurry, low quality"


# =============================================================================
# ExtraParamSchema
# =============================================================================


class TestExtraParamSchema:
    """ExtraParamSchema construction and field access."""

    def test_basic_construction(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="float", default=0.0, description="test param")
        assert spec.type == "float"
        assert spec.default == 0.0
        assert spec.description == "test param"
        assert spec.range is None

    def test_with_range(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="float", range=(0.0, 1.0))
        assert spec.range == (0.0, 1.0)

    def test_none_default(self):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        spec = ExtraParamSchema(type="str")
        assert spec.default is None

    def test_public_import(self):
        from tensorrt_llm import ExtraParamSchema

        spec = ExtraParamSchema(type="int", default=42)
        assert spec.default == 42


# =============================================================================
# Pipeline DEFAULT_GENERATION_PARAMS and EXTRA_PARAM_SPECS
# =============================================================================


class TestPipelineDefaults:
    """Each pipeline declares correct default generation params."""

    def test_wan_defaults(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        d = WanPipeline.DEFAULT_GENERATION_PARAMS
        assert d["height"] == 480
        assert d["width"] == 832
        assert d["num_inference_steps"] == 50
        assert d["guidance_scale"] == 5.0
        assert d["num_frames"] == 81

    def test_flux_defaults(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        d = FluxPipeline.DEFAULT_GENERATION_PARAMS
        assert d["height"] == 1024
        assert d["width"] == 1024
        assert d["guidance_scale"] == 3.5

    def test_ltx2_defaults(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        d = LTX2Pipeline.DEFAULT_GENERATION_PARAMS
        assert d["height"] == 512
        assert d["width"] == 768
        assert d["num_inference_steps"] == 40
        assert d["guidance_scale"] == 4.0
        assert d["max_sequence_length"] == 1024
        assert d["num_frames"] == 121

    def test_base_pipeline_empty_defaults(self):
        from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

        assert BasePipeline.DEFAULT_GENERATION_PARAMS == {}
        assert BasePipeline.EXTRA_PARAM_SPECS == {}


class TestPipelineExtraParamSpecs:
    """Each pipeline declares correct extra param specs."""

    def test_wan_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        specs = WanPipeline.EXTRA_PARAM_SPECS
        assert "guidance_scale_2" in specs
        assert "boundary_ratio" in specs
        assert specs["guidance_scale_2"].type == "float"
        assert specs["boundary_ratio"].range == (0.0, 1.0)

    def test_wan_i2v_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        specs = WanImageToVideoPipeline.EXTRA_PARAM_SPECS
        assert "last_image" in specs
        assert "guidance_scale_2" in specs
        assert specs["last_image"].type == "str"

    def test_flux_no_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        assert FluxPipeline.EXTRA_PARAM_SPECS == {}

    def test_ltx2_extra_specs(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        specs = LTX2Pipeline.EXTRA_PARAM_SPECS
        expected_keys = {
            "output_type",
            "guidance_rescale",
            "stg_scale",
            "stg_blocks",
            "modality_scale",
            "rescale_scale",
            "guidance_skip_step",
            "enhance_prompt",
        }
        assert set(specs.keys()) == expected_keys
        assert specs["stg_scale"].default == 0.0
        assert specs["enhance_prompt"].default is False
        assert specs["stg_blocks"].default is None

    def test_ltx2_extra_specs_attribute_access(self):
        """Direct attribute-style access works: Pipeline.EXTRA_PARAM_SPECS['key']."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        assert LTX2Pipeline.EXTRA_PARAM_SPECS["modality_scale"].type == "float"
        assert LTX2Pipeline.EXTRA_PARAM_SPECS["modality_scale"].default == 1.0


# =============================================================================
# Executor default merging
# =============================================================================


class TestDefaultMerging:
    """DiffusionExecutor._merge_defaults fills None fields correctly."""

    def _make_mock_executor(self, pipeline_cls):
        """Create a mock DiffusionExecutor with the given pipeline class's specs."""
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.DEFAULT_GENERATION_PARAMS = pipeline_cls.DEFAULT_GENERATION_PARAMS
        executor.pipeline.EXTRA_PARAM_SPECS = pipeline_cls.EXTRA_PARAM_SPECS
        return executor

    def _make_request(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

        return DiffusionRequest(request_id=0, prompt=["test"], **kwargs)

    def _merge(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._merge_defaults(executor, req)

    def test_universal_defaults_merged(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request()
        assert req.height is None

        self._merge(executor, req)
        assert req.height == 480
        assert req.width == 832
        assert req.num_inference_steps == 50

    def test_user_values_not_overwritten(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(height=1080, width=1920)

        self._merge(executor, req)
        assert req.height == 1080  # User value preserved
        assert req.width == 1920
        assert req.num_inference_steps == 50  # Default filled

    def test_extra_params_defaults_merged(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request()

        self._merge(executor, req)
        assert req.extra_params is not None
        assert req.extra_params["stg_scale"] == 0.0
        assert req.extra_params["output_type"] == "pt"
        assert req.extra_params["enhance_prompt"] is False
        # None defaults are also filled
        assert req.extra_params["stg_blocks"] is None

    def test_user_extra_params_not_overwritten(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})

        self._merge(executor, req)
        assert req.extra_params["stg_scale"] == 0.5  # User value preserved
        assert req.extra_params["output_type"] == "pt"  # Default filled

    def test_no_extra_params_for_flux(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request()

        self._merge(executor, req)
        assert req.extra_params is None  # Flux has no extra specs

    def test_all_declared_keys_present_after_merge(self):
        """After merge, all EXTRA_PARAM_SPECS keys are in extra_params."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})

        self._merge(executor, req)
        for key in LTX2Pipeline.EXTRA_PARAM_SPECS:
            assert key in req.extra_params, f"Missing key: {key}"


# =============================================================================
# VisualGen.default_params and extra_param_specs
# =============================================================================


class TestVisualGenDefaultParams:
    """VisualGen.default_params returns correctly merged params per pipeline.

    VisualGen delegates to executor.default_generation_params and
    executor.extra_param_specs (populated from the READY signal).
    """

    def _make_visual_gen(self, pipeline_cls):
        """Create VisualGen with mocked init and executor carrying pipeline metadata."""
        from tensorrt_llm.visual_gen import VisualGen

        with patch.object(VisualGen, "__init__", lambda self, *a, **kw: None):
            vg = VisualGen.__new__(VisualGen)
            vg.executor = MagicMock()
            if pipeline_cls is not None:
                vg.executor.default_generation_params = pipeline_cls.DEFAULT_GENERATION_PARAMS
                vg.executor.extra_param_specs = pipeline_cls.EXTRA_PARAM_SPECS
            else:
                vg.executor.default_generation_params = {}
                vg.executor.extra_param_specs = {}
            return vg

    def test_ltx2_default_params(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        vg = self._make_visual_gen(LTX2Pipeline)
        params = vg.default_params
        assert params.height == 512
        assert params.width == 768
        assert params.num_inference_steps == 40
        assert params.seed == 42
        assert params.extra_params is not None
        assert params.extra_params["stg_scale"] == 0.0
        assert params.extra_params["output_type"] == "pt"
        # None-default keys are present
        assert "stg_blocks" in params.extra_params

    def test_wan_default_params(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        vg = self._make_visual_gen(WanPipeline)
        params = vg.default_params
        assert params.height == 480
        assert params.width == 832
        assert params.extra_params is not None
        assert "guidance_scale_2" in params.extra_params
        assert "boundary_ratio" in params.extra_params

    def test_flux_default_params_no_extra(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        vg = self._make_visual_gen(FluxPipeline)
        params = vg.default_params
        assert params.height == 1024
        assert params.width == 1024
        assert params.extra_params is None

    def test_no_pipeline_returns_bare_params(self):
        vg = self._make_visual_gen(None)
        params = vg.default_params
        assert params.height is None
        assert params.extra_params is None

    def test_extra_param_specs(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        vg = self._make_visual_gen(LTX2Pipeline)
        specs = vg.extra_param_specs
        assert "stg_scale" in specs
        assert specs["stg_scale"].type == "float"

    def test_extra_param_specs_empty_for_flux(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        vg = self._make_visual_gen(FluxPipeline)
        assert vg.extra_param_specs == {}


# =============================================================================
# Pipeline metadata bridging (executor → client)
# =============================================================================


class TestPipelineMetadataBridging:
    """Verify DEFAULT_GENERATION_PARAMS and EXTRA_PARAM_SPECS survive
    the pickle round-trip from DiffusionExecutor READY signal to the client."""

    def _build_ready_response(self, pipeline_cls):
        """Build a DiffusionResponse matching what DiffusionExecutor sends."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse

        return DiffusionResponse(
            request_id=-1,
            output={
                "status": "READY",
                "default_generation_params": pipeline_cls.DEFAULT_GENERATION_PARAMS,
                "extra_param_specs": pipeline_cls.EXTRA_PARAM_SPECS,
            },
        )

    def _roundtrip(self, response):
        """Pickle/unpickle to simulate ZMQ transport."""
        import pickle

        return pickle.loads(pickle.dumps(response))

    def test_ready_payload_pickle_roundtrip(self):
        """The READY dict survives pickle (the ZMQ transport layer)."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        assert isinstance(restored, DiffusionResponse)
        assert restored.request_id == -1
        payload = restored.output
        assert isinstance(payload, dict)
        assert payload["status"] == "READY"
        assert payload["default_generation_params"] == LTX2Pipeline.DEFAULT_GENERATION_PARAMS
        assert set(payload["extra_param_specs"].keys()) == set(
            LTX2Pipeline.EXTRA_PARAM_SPECS.keys()
        )

    def test_extra_param_schema_type_preserved(self):
        """ExtraParamSchema instances keep their type through pickle."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        specs = restored.output["extra_param_specs"]
        for key, spec in specs.items():
            assert isinstance(spec, ExtraParamSchema), (
                f"spec '{key}' lost its type: got {type(spec).__name__}"
            )

    def test_extra_param_schema_fields_preserved(self):
        """ExtraParamSchema field values survive the round-trip."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        specs = restored.output["extra_param_specs"]
        original = LTX2Pipeline.EXTRA_PARAM_SPECS
        for key in original:
            assert specs[key].type == original[key].type
            assert specs[key].default == original[key].default
            assert specs[key].range == original[key].range
            assert specs[key].description == original[key].description

    def test_wan_pipeline_roundtrip(self):
        """Wan pipeline metadata survives the round-trip."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        resp = self._build_ready_response(WanPipeline)
        restored = self._roundtrip(resp)

        payload = restored.output
        assert payload["default_generation_params"]["height"] == 480
        assert payload["default_generation_params"]["num_frames"] == 81
        assert "guidance_scale_2" in payload["extra_param_specs"]
        assert "boundary_ratio" in payload["extra_param_specs"]

    def test_flux_empty_specs_roundtrip(self):
        """Pipeline with no EXTRA_PARAM_SPECS round-trips as empty dict."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        resp = self._build_ready_response(FluxPipeline)
        restored = self._roundtrip(resp)

        assert restored.output["extra_param_specs"] == {}
        assert restored.output["default_generation_params"]["height"] == 1024

    def test_client_extracts_metadata_from_ready(self):
        """DiffusionRemoteClient stores metadata when processing a READY response."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
        from tensorrt_llm.visual_gen.visual_gen import DiffusionRemoteClient

        resp = self._build_ready_response(LTX2Pipeline)
        restored = self._roundtrip(resp)

        # Simulate what _wait_ready_async does: extract from the response payload
        client = MagicMock(spec=DiffusionRemoteClient)
        client.default_generation_params = {}
        client.extra_param_specs = {}

        payload = restored.output
        if isinstance(payload, dict):
            client.default_generation_params = payload.get("default_generation_params", {})
            client.extra_param_specs = payload.get("extra_param_specs", {})

        assert client.default_generation_params == LTX2Pipeline.DEFAULT_GENERATION_PARAMS
        assert set(client.extra_param_specs.keys()) == set(LTX2Pipeline.EXTRA_PARAM_SPECS.keys())
        for spec in client.extra_param_specs.values():
            assert isinstance(spec, ExtraParamSchema)


# =============================================================================
# VisualGenParamsError — error class
# =============================================================================


class TestVisualGenParamsError:
    """VisualGenParamsError is importable and is a subclass of ValueError."""

    def test_import_from_top_level(self):
        from tensorrt_llm import VisualGenParamsError

        assert issubclass(VisualGenParamsError, ValueError)

    def test_import_from_visual_gen(self):
        from tensorrt_llm.visual_gen import VisualGenParamsError

        assert VisualGenParamsError is not None

    def test_is_subclass_of_value_error(self):
        from tensorrt_llm.visual_gen import VisualGenParamsError

        assert issubclass(VisualGenParamsError, ValueError)
        assert not issubclass(VisualGenParamsError, RuntimeError)

    def test_raise_and_catch_as_value_error(self):
        from tensorrt_llm.visual_gen import VisualGenParamsError

        with pytest.raises(ValueError):
            raise VisualGenParamsError("bad param")

    def test_message_preserved(self):
        from tensorrt_llm.visual_gen import VisualGenParamsError

        with pytest.raises(VisualGenParamsError, match="height.*out of range"):
            raise VisualGenParamsError("height is out of range")


# =============================================================================
# Request validation — _validate_request
# =============================================================================


class TestRequestValidation:
    """DiffusionExecutor._validate_request raises VisualGenParamsError on bad params."""

    def _make_mock_executor(self, pipeline_cls):
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__ = pipeline_cls
        executor.pipeline.DEFAULT_GENERATION_PARAMS = pipeline_cls.DEFAULT_GENERATION_PARAMS
        executor.pipeline.EXTRA_PARAM_SPECS = pipeline_cls.EXTRA_PARAM_SPECS
        return executor

    def _make_request(self, **kwargs):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

        return DiffusionRequest(request_id=0, prompt=["test"], **kwargs)

    def _validate(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._validate_request(executor, req)

    def _merge_and_validate(self, executor, req):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        DiffusionExecutor._merge_defaults(executor, req)
        DiffusionExecutor._validate_request(executor, req)

    # --- unknown extra_params ---

    def test_unknown_extra_params_raises(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(extra_params={"nonexistent_key": 42})
        with pytest.raises(VisualGenParamsError, match="Unknown extra_params"):
            self._validate(executor, req)

    def test_unknown_extra_params_lists_supported_keys(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"bad_key": 1})
        with pytest.raises(VisualGenParamsError, match="Supported"):
            self._validate(executor, req)

    def test_valid_extra_params_accepted(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 0.5})
        self._merge_and_validate(executor, req)  # should not raise

    # --- unsupported universal fields ---

    def test_num_frames_on_image_pipeline_raises(self):
        """num_frames=81 to FLUX (image-only) should raise."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(num_frames=81)
        with pytest.raises(VisualGenParamsError, match="num_frames.*not use it"):
            self._validate(executor, req)

    def test_frame_rate_on_image_pipeline_raises(self):
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(frame_rate=24.0)
        with pytest.raises(VisualGenParamsError, match="frame_rate.*not use it"):
            self._validate(executor, req)

    def test_image_not_checked_by_validator(self):
        """image is a conditioning input — validated at runtime by infer(), not here."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(image="/path/to/img.png")
        # Should not raise — image validation is the pipeline's responsibility
        self._merge_and_validate(executor, req)

    def test_num_frames_on_video_pipeline_ok(self):
        """num_frames is declared by WanPipeline, should not raise."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(num_frames=81)
        self._merge_and_validate(executor, req)

    def test_image_on_i2v_pipeline_ok(self):
        """image is declared by WanImageToVideoPipeline, should not raise."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )

        executor = self._make_mock_executor(WanImageToVideoPipeline)
        req = self._make_request(image="/path/to/img.png")
        self._merge_and_validate(executor, req)

    def test_none_fields_not_flagged(self):
        """Fields left as None should never trigger unsupported-field errors."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request()  # all None
        self._merge_and_validate(executor, req)

    # --- type validation on extra_params ---

    def test_wrong_type_extra_param_raises(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": "not_a_number"})
        with pytest.raises(VisualGenParamsError, match="expected type 'float'"):
            self._merge_and_validate(executor, req)

    def test_int_accepted_for_float_spec(self):
        """An int value should be accepted for a 'float'-typed spec."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": 1})
        self._merge_and_validate(executor, req)

    def test_bool_rejected_for_float_spec(self):
        """A bool should not be accepted for a 'float' spec (bool is-a int in Python,
        but semantically wrong for floats)."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        executor = self._make_mock_executor(LTX2Pipeline)
        req = self._make_request(extra_params={"stg_scale": True})
        # bool is instance of int which is accepted for float, so this passes type check.
        # This is intentional — Python's type hierarchy makes bool a subclass of int.
        self._merge_and_validate(executor, req)

    def test_wrong_type_str_extra_param(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_i2v import (
            WanImageToVideoPipeline,
        )
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(WanImageToVideoPipeline)
        req = self._make_request(
            image="/img.png",
            extra_params={"last_image": 123},
        )
        with pytest.raises(VisualGenParamsError, match="expected type 'str'"):
            self._merge_and_validate(executor, req)

    # --- range validation on extra_params ---

    def test_out_of_range_extra_param_raises(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(WanPipeline)
        # boundary_ratio has range (0.0, 1.0)
        req = self._make_request(extra_params={"boundary_ratio": 2.0})
        with pytest.raises(VisualGenParamsError, match="out of range"):
            self._merge_and_validate(executor, req)

    def test_negative_boundary_ratio_raises(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(extra_params={"boundary_ratio": -0.5})
        with pytest.raises(VisualGenParamsError, match="out of range"):
            self._merge_and_validate(executor, req)

    def test_boundary_value_at_range_edge_ok(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(extra_params={"boundary_ratio": 0.0})
        self._merge_and_validate(executor, req)

    def test_boundary_value_at_range_max_ok(self):
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(extra_params={"boundary_ratio": 1.0})
        self._merge_and_validate(executor, req)

    # --- multiple errors collected ---

    def test_multiple_errors_in_single_message(self):
        """Multiple validation failures should be collected into one error."""
        from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux import FluxPipeline
        from tensorrt_llm.visual_gen import VisualGenParamsError

        executor = self._make_mock_executor(FluxPipeline)
        req = self._make_request(
            num_frames=81,
            frame_rate=24.0,
            extra_params={"bogus": 1},
        )
        with pytest.raises(VisualGenParamsError) as exc_info:
            self._validate(executor, req)
        msg = str(exc_info.value)
        assert "num_frames" in msg
        assert "frame_rate" in msg
        assert "bogus" in msg

    # --- None extra_params values skip type/range check ---

    def test_none_extra_param_value_skipped(self):
        """None values for extra_params with range specs should not fail validation."""
        from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline

        executor = self._make_mock_executor(WanPipeline)
        req = self._make_request(extra_params={"boundary_ratio": None})
        self._merge_and_validate(executor, req)

    # --- process_request returns error response instead of crashing ---

    def test_process_request_returns_error_on_validation_failure(self):
        """Validation errors become error responses, not server crashes."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor, DiffusionResponse

        # Build a mock with real method bindings for the three methods
        # that process_request chains through.
        executor = MagicMock()
        executor.pipeline = MagicMock()
        executor.pipeline.__class__.__name__ = "FluxPipeline"
        executor.pipeline.DEFAULT_GENERATION_PARAMS = {"height": 1024, "width": 1024}
        executor.pipeline.EXTRA_PARAM_SPECS = {}
        executor.pipeline._warmed_up_shapes = set()
        executor.pipeline.warmup_cache_key = MagicMock(return_value=(1024, 1024, None))
        executor.rank = 0
        executor.device_id = 0
        executor.response_queue = MagicMock()

        # Wire real methods onto the mock so process_request uses them
        executor._merge_defaults = lambda req: DiffusionExecutor._merge_defaults(executor, req)
        executor._validate_request = lambda req: DiffusionExecutor._validate_request(executor, req)

        req = self._make_request(num_frames=81, extra_params={"bad": 1})

        # Call the real process_request
        DiffusionExecutor.process_request(executor, req)

        # Should have put an error response, not crashed
        executor.response_queue.put.assert_called_once()
        resp = executor.response_queue.put.call_args[0][0]
        assert isinstance(resp, DiffusionResponse)
        assert resp.error_msg is not None
        assert "validation failed" in resp.error_msg.lower()
