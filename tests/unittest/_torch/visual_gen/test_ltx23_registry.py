# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU registration and workflow routing tests for LTX-2.3."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import LTX23RetakePipeline
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
    assert LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs=extra_attrs)) is expected


def test_ltx23_rejects_unknown_workflow() -> None:
    with pytest.raises(ValueError, match="Unsupported LTX-2.3 workflow"):
        LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "unknown"}))
