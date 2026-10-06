# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import json

import pytest

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.checkpoint import (
    checkpoint_has_markers,
    resolve_checkpoint_source,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image_21.config import (
    DEFAULT_NUM_INFERENCE_STEPS,
    QwenImage21PipelineConfig,
    QwenImage21TransformerConfig,
    resolve_num_inference_steps,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image_21.manifest import (
    candidate_metadata,
    flatten_generation_manifest,
)


def test_qwen_image_21_step_resolution_prefers_frozen_manifest():
    env = {"NUM_INFERENCE_STEPS": "7", "VISUALGEN_NUM_INFERENCE_STEPS": "9"}
    assert resolve_num_inference_steps({"num_inference_steps": 40}, env) == 40
    assert resolve_num_inference_steps({"denoising_steps": "32"}, env) == 32


def test_qwen_image_21_step_resolution_honors_env_then_sdk_default():
    assert resolve_num_inference_steps({}, {"VISUALGEN_NUM_INFERENCE_STEPS": "11"}) == 11
    assert resolve_num_inference_steps({}, {}) == DEFAULT_NUM_INFERENCE_STEPS
    with pytest.raises(ValueError):
        resolve_num_inference_steps({"num_inference_steps": 0}, {})


def test_qwen_image_21_shape_defaults_match_native_pipeline():
    transformer = QwenImage21TransformerConfig()
    assert transformer.latent_channels == 64
    assert transformer.patch_size == 1
    assert transformer.vae_scale_factor == 16
    assert transformer.latent_shape(1024, 1024) == (64, 64, 64)
    config = QwenImage21PipelineConfig.from_manifest_and_env({"height": 1024, "width": 1024, "num_inference_steps": 40})
    assert config.max_sequence_length == 4096
    assert config.guidance_scale == 1.0
    assert config.num_inference_steps == 40


def test_qwen_image_21_checkpoint_marker_resolution(tmp_path):
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    assert not checkpoint_has_markers(checkpoint)
    (checkpoint / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}))
    assert checkpoint_has_markers(checkpoint)
    resolved = resolve_checkpoint_source({"local_path": str(checkpoint), "revision": "abc"})
    assert resolved.source == "local_path"
    assert resolved.local_files_only
    assert resolved.revision == "abc"


def test_qwen_image_21_manifest_flatten_and_metadata(tmp_path):
    image = tmp_path / "candidate.png"
    manifest = {
        "generation": {"prompt": "a test", "height": 1024},
        "checkpoint": {"hf_model_id": "Qwen/Qwen-Image-2.1", "revision": "main"},
        "width": 1024,
        "num_inference_steps": 40,
    }
    flat = flatten_generation_manifest(manifest)
    assert flat["prompt"] == "a test"
    assert flat["hf_model_id"] == "Qwen/Qwen-Image-2.1"
    metadata = candidate_metadata(manifest, output_path=image, image_sha256="0" * 64)
    assert metadata["full_e2e"] is True
    assert metadata["num_inference_steps"] == 40
    assert metadata["pipeline_class"] == "QwenImage21Pipeline"
