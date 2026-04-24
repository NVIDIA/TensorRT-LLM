# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 registry smoke tests for Qwen-Image.

These tests do NOT exercise the transformer forward pass -- the point of
Phase 0 is to make sure Qwen-Image checkpoints route through VisualGen's
``AutoPipeline`` and emit a clear NotImplementedError from the stub
instead of ``Unknown pipeline: ''`` from registry detection.
"""

import json

import pytest

# Importing the models package side-effects the ``@register_pipeline``
# decorator on ``QwenImagePipeline`` being applied, which is what we are
# testing here.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import (
    PIPELINE_REGISTRY,
    AutoPipeline,
)


def test_qwen_image_pipeline_is_registered():
    """@register_pipeline("QwenImagePipeline") must have been applied."""
    assert "QwenImagePipeline" in PIPELINE_REGISTRY
    assert PIPELINE_REGISTRY["QwenImagePipeline"] is QwenImagePipeline


def test_auto_pipeline_detects_qwen_image_class_name(tmp_path):
    """model_index.json with _class_name=QwenImagePipeline resolves."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline"})
    )
    assert (
        AutoPipeline._detect_from_checkpoint(str(tmp_path))
        == "QwenImagePipeline"
    )


@pytest.mark.parametrize(
    "variant_class_name",
    [
        "QwenImageImg2ImgPipeline",
        "QwenImageEditPipeline",
        "QwenImageControlNetPipeline",
    ],
)
def test_auto_pipeline_routes_qwen_image_variants(tmp_path, variant_class_name):
    """Unknown Qwen-Image variants route to the base QwenImagePipeline.

    Matches the behaviour of the Wan and Flux branches in
    ``pipeline_registry.py``: anything containing ``Qwen`` + ``Image`` in
    its class name falls through to the base pipeline, letting Phase 1
    decide whether to implement a dedicated subclass or share the base.
    """
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": variant_class_name})
    )
    assert (
        AutoPipeline._detect_from_checkpoint(str(tmp_path))
        == "QwenImagePipeline"
    )


def test_transformer_constructs_with_defaults():
    """M6: the full transformer instantiates with the documented defaults.

    Doesn't touch real weights -- just verifies that the class graph is
    buildable and exposes the expected public attributes.
    """
    model = QwenImageTransformer2DModel(
        model_config=None,
        num_layers=2,  # tiny to keep CPU instantiation fast
    )
    assert len(model.transformer_blocks) == 2
    assert model.inner_dim == 24 * 128
    assert hasattr(model, "img_in")
    assert hasattr(model, "txt_in")
    assert hasattr(model, "txt_norm")
    assert hasattr(model, "norm_out")
    assert hasattr(model, "proj_out")
    assert hasattr(model, "pos_embed")


def test_transformer_load_weights_detects_mismatch():
    """load_weights should surface a clear RuntimeError on a bad dict."""
    model = QwenImageTransformer2DModel(model_config=None, num_layers=2)
    with pytest.raises(RuntimeError, match="Missing keys|Unexpected keys"):
        model.load_weights({})
