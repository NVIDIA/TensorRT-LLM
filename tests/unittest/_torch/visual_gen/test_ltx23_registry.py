# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU registration and workflow routing tests for LTX-2.3."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
    LTX2_FORCE_ONE_STAGE_ENV,
    LTX2Pipeline,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import LTX23RetakePipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_two_stages import (
    LTX23TwoStagesPipeline,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

pytestmark = pytest.mark.cpu_only


def test_ltx23_pipeline_is_registered() -> None:
    entry = PIPELINE_REGISTRY["LTX23Pipeline"]
    assert entry.pipeline_cls is LTX23Pipeline
    assert entry.hf_ids == ["Lightricks/LTX-2.3"]


def test_ltx23_retake_is_registered_as_a_workflow_variant() -> None:
    entry = PIPELINE_REGISTRY["LTX23RetakePipeline"]
    assert entry.pipeline_cls is LTX23RetakePipeline
    assert entry.hf_ids == []
    assert entry.defaults == {}
    assert issubclass(LTX23RetakePipeline, LTX23Pipeline)


def test_ltx23_has_a_separate_pipeline_boundary() -> None:
    assert issubclass(LTX23Pipeline, BasePipeline)
    assert not issubclass(LTX23Pipeline, LTX2Pipeline)


@pytest.mark.parametrize("class_name", ["LTX23Pipeline", "LTX23RetakePipeline"])
def test_auto_pipeline_detects_registered_ltx23_class_name(tmp_path: Path, class_name: str) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": class_name}), encoding="utf-8"
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == class_name


@pytest.mark.parametrize(
    "extra_attrs, expected",
    [
        ({}, LTX23Pipeline),
        ({"workflow": "generation"}, LTX23Pipeline),
        ({"workflow": "retake"}, LTX23RetakePipeline),
    ],
)
def test_ltx23_routes_workflow_variants(extra_attrs: dict, expected: type) -> None:
    config = SimpleNamespace(
        extra_attrs=extra_attrs,
        primary_pretrained_config=SimpleNamespace(_name_or_path=""),
    )
    assert LTX23Pipeline.resolve_variant(config) is expected


def test_ltx23_rejects_unknown_workflow() -> None:
    with pytest.raises(ValueError, match="Unsupported LTX-2.3 workflow"):
        LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "unknown"}))


@pytest.mark.parametrize("cache_backend", ["teacache", "cache_dit"])
def test_ltx23_cache_selects_single_stage_with_auxiliary_paths(monkeypatch, cache_backend) -> None:
    """Cache routing must select single-stage without relying on the environment override."""
    monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
    config = SimpleNamespace(
        cache_backend=cache_backend,
        extra_attrs={"spatial_upsampler_path": "/upscaler", "distilled_lora_path": "/lora"},
    )
    assert LTX23Pipeline.resolve_variant(config) is LTX23Pipeline


@pytest.mark.parametrize("path_key", ["spatial_upsampler_path", "distilled_lora_path"])
def test_ltx23_retake_rejects_two_stage_paths(monkeypatch, path_key) -> None:
    """Either explicitly supplied two-stage component is invalid for retake."""
    monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
    config = SimpleNamespace(extra_attrs={"workflow": "retake", path_key: "/auxiliary"})
    with pytest.raises(ValueError, match="not supported for retake"):
        LTX23Pipeline.resolve_variant(config)


@pytest.mark.parametrize("quant_algo", [None, "FP8_BLOCK_SCALES", "NVFP4"])
@pytest.mark.parametrize("force_one_stage", [False, True])
def test_ltx23_discovers_two_stage_components(
    tmp_path, monkeypatch, quant_algo, force_one_stage
) -> None:
    """Auxiliary discovery promotes every supported precision unless explicitly forced off."""
    monkeypatch.setenv(LTX2_FORCE_ONE_STAGE_ENV, "1" if force_one_stage else "0")
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "ltx-2.3-spatial-upscaler-x2.safetensors").touch()
    (tmp_path / "ltx-2.3-distilled-lora.safetensors").touch()
    config = SimpleNamespace(
        cache_backend=None,
        extra_attrs={},
        primary_pretrained_config=SimpleNamespace(
            _name_or_path=str(tmp_path / "model.safetensors")
        ),
        quant_config=SimpleNamespace(quant_algo=quant_algo),
    )
    expected = LTX23Pipeline if force_one_stage else LTX23TwoStagesPipeline
    assert LTX23Pipeline.resolve_variant(config) is expected


@pytest.mark.parametrize("explicit", [False, True])
def test_ltx23_partial_two_stage_components(tmp_path, monkeypatch, explicit) -> None:
    """Discovery alone falls back for incomplete assets; an explicit request fails clearly."""
    monkeypatch.delenv(LTX2_FORCE_ONE_STAGE_ENV, raising=False)
    upscaler = tmp_path / "ltx-2.3-spatial-upscaler-x2.safetensors"
    upscaler.touch()
    config = SimpleNamespace(
        cache_backend=None,
        extra_attrs={"spatial_upsampler_path": str(upscaler)} if explicit else {},
        primary_pretrained_config=SimpleNamespace(_name_or_path=str(tmp_path)),
    )
    if explicit:
        with pytest.raises(ValueError, match="requires both"):
            LTX23Pipeline.resolve_variant(config)
    else:
        assert LTX23Pipeline.resolve_variant(config) is LTX23Pipeline
