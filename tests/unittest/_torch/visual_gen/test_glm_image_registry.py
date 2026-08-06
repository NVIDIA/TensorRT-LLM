# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry and adapter tests for GLM-Image VisualGen support."""

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.glm_image import GlmImagePipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs


def _write_minimal_glm_checkpoint(tmp_path: Path) -> Path:
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "GlmImagePipeline",
                "transformer": ["diffusers", "GlmImageTransformer2DModel"],
            }
        )
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "GlmImageTransformer2DModel"})
    )
    return tmp_path


def _make_pipeline() -> GlmImagePipeline:
    config = DiffusionPipelineConfig(
        model_configs={
            "transformer": DiffusionModelConfig(
                pretrained_config=SimpleNamespace(_name_or_path="zai-org/GLM-Image")
            )
        }
    )
    return GlmImagePipeline(config)


def test_glm_image_pipeline_is_registered() -> None:
    assert "GlmImagePipeline" in PIPELINE_REGISTRY
    entry = PIPELINE_REGISTRY["GlmImagePipeline"]
    assert entry.pipeline_cls is GlmImagePipeline
    assert "zai-org/GLM-Image" in entry.hf_ids


def test_auto_pipeline_detects_glm_image_class_name(tmp_path: Path) -> None:
    checkpoint_dir = _write_minimal_glm_checkpoint(tmp_path)
    assert AutoPipeline._detect_from_checkpoint(str(checkpoint_dir)) == "GlmImagePipeline"


def test_glm_image_supported_model_and_pipeline_config() -> None:
    from tensorrt_llm import VisualGen

    assert "zai-org/GLM-Image" in VisualGen.supported_models()
    assert VisualGen.pipeline_config("zai-org/GLM-Image") == {}


def test_glm_image_pipeline_config_rejects_unknown_keys(tmp_path: Path) -> None:
    checkpoint_dir = _write_minimal_glm_checkpoint(tmp_path)
    args = VisualGenArgs(model=str(checkpoint_dir), pipeline_config={"native_transformer": True})

    with pytest.raises(ValueError, match="Unknown pipeline_config keys for GlmImagePipeline"):
        PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))


def test_glm_image_adapter_has_no_native_transformer_weights() -> None:
    pipeline = _make_pipeline()

    assert pipeline.transformer is None
    assert pipeline.load_transformer_weights("/does/not/matter") == {}
    assert pipeline.load_weights({}) is None
    assert pipeline.default_generation_params == {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 1.5,
        "max_sequence_length": 2048,
    }
    assert pipeline.resolution_multiple_of == (32, 32)


def test_glm_image_pil_images_to_tensor() -> None:
    images = [
        Image.new("RGB", (2, 3), color=(255, 0, 0)),
        Image.new("RGB", (2, 3), color=(0, 255, 0)),
    ]

    tensor = GlmImagePipeline._pil_images_to_tensor(images)

    assert tensor.shape == (2, 3, 2, 3)
    assert tensor.dtype == torch.uint8
    assert tensor[0, 0, 0].tolist() == [255, 0, 0]
    assert tensor[1, 0, 0].tolist() == [0, 255, 0]


def test_glm_image_condition_tensor_scales_unit_float_image() -> None:
    tensor = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.5, 0.25]],
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.0, 0.0], [0.5, 0.25]],
            ]
        ]
    )

    images = GlmImagePipeline._load_condition_images(tensor)

    assert len(images) == 1
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[0].getpixel((1, 0)) == (0, 255, 0)


def test_glm_image_batch_prompt_multiple_images_limit() -> None:
    pipeline = _make_pipeline()
    req = SimpleNamespace(
        prompt=["one", "two"],
        params=SimpleNamespace(
            num_images_per_prompt=2,
            negative_prompt=None,
            image=None,
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            max_sequence_length=2048,
        ),
    )

    with pytest.raises(NotImplementedError, match="batched prompts"):
        pipeline.infer(req)


def test_glm_image_rejects_string_negative_prompt() -> None:
    pipeline = _make_pipeline()
    req = SimpleNamespace(
        prompt=["one"],
        params=SimpleNamespace(
            num_images_per_prompt=1,
            negative_prompt="blur",
            image=None,
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            max_sequence_length=2048,
        ),
    )

    with pytest.raises(ValueError, match="negative_prompt"):
        pipeline.infer(req)


def test_glm_image_e2e_scripts_are_checkpoint_aware() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scripts = [
        repo_root / "examples/glm_image/reference_e2e.sh",
        repo_root / "examples/glm_image/candidate_e2e.sh",
        repo_root / "examples/glm_image/lpips_e2e.sh",
    ]
    for script in scripts:
        assert script.exists()
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR

    reference_text = scripts[0].read_text(encoding="utf-8")
    candidate_text = scripts[1].read_text(encoding="utf-8")
    lpips_text = scripts[2].read_text(encoding="utf-8")

    for text in (reference_text, candidate_text):
        assert "GLM_IMAGE_MODEL" in text
        assert "GLM_IMAGE_REVISION" in text
        assert "GLM_IMAGE_HF_CACHE_DIR" in text
        assert "GLM_IMAGE_LOCAL_FILES_ONLY" in text
        assert "GLM_IMAGE_INPUT_IMAGE" in text

    assert "--cache-dir" in reference_text
    assert "--hf-cache-dir" in candidate_text
    assert "reference_report.json" in lpips_text
    assert "candidate_report.json" in lpips_text
