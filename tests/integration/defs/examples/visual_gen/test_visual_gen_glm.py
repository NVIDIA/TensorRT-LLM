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
"""Single-device GLM-Image visual-quality regression tests."""

from dataclasses import dataclass

import pytest
import torch
from defs.examples.visual_gen.visual_gen_test_utils import (
    FeatureConfigState,
    _assert_feature_quantization_installed,
    _assert_lpips_below_threshold,
    _assert_resolved_single_device_feature_config,
    _assert_single_device_feature_executed,
    _build_single_device_feature_args,
    _cleanup_cuda,
    _cleanup_single_device_feature_pipeline,
    _disable_inductor_compile_worker_quiesce,
    _fixed_nvfp4_quantization_backend,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_image_lpips_eval,
    _run_single_device_feature_generator,
    _skip_if_missing,
    _validate_single_feature_config,
)

GLM_IMAGE_CHECKPOINT_SUBDIR = "GLM-Image"
GLM_IMAGE_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
# GLM's native resolution; the AR prior stage degrades badly below it.
GLM_IMAGE_LPIPS_HEIGHT = 1024
GLM_IMAGE_LPIPS_WIDTH = 1024
GLM_IMAGE_LPIPS_NUM_INFERENCE_STEPS = 30
GLM_IMAGE_LPIPS_GUIDANCE_SCALE = 1.5
GLM_IMAGE_LPIPS_SEED = 42
GLM_IMAGE_LPIPS_THRESHOLD = 0.05
GLM_IMAGE_FEATURE_LPIPS_THRESHOLD = 0.05
GLM_IMAGE_SUPPORTED_FEATURES = frozenset({"fp8-blockwise", "nvfp4"})


@dataclass(frozen=True)
class GlmImageFeatureProfile:
    id: str
    features: FeatureConfigState


@dataclass(frozen=True)
class GlmImageAccuracyCase:
    id: str
    checkpoint_subdir: str
    golden_file: str
    features: FeatureConfigState
    lpips_threshold: float


GLM_IMAGE_FEATURE_PROFILES = (
    GlmImageFeatureProfile(
        id="fp8-blockwise",
        features=FeatureConfigState(quantization="FP8_BLOCK_SCALES"),
    ),
    GlmImageFeatureProfile(
        id="nvfp4",
        features=FeatureConfigState(quantization="NVFP4"),
    ),
)


def _build_glm_image_accuracy_cases():
    cases = []
    for profile in GLM_IMAGE_FEATURE_PROFILES:
        _validate_single_feature_config(
            profile.features,
            GLM_IMAGE_SUPPORTED_FEATURES,
            "GLM-Image",
        )
        case_id = profile.id
        cases.append(
            pytest.param(
                GlmImageAccuracyCase(
                    id=case_id,
                    checkpoint_subdir=GLM_IMAGE_CHECKPOINT_SUBDIR,
                    golden_file=f"glm_image_{profile.id.replace('-', '_')}_lpips_golden.png",
                    features=profile.features,
                    lpips_threshold=GLM_IMAGE_FEATURE_LPIPS_THRESHOLD,
                ),
                id=case_id,
            )
        )
    return cases


GLM_IMAGE_ACCURACY_CASES = _build_glm_image_accuracy_cases()


def _glm_image_generator(device):
    # GlmImagePipeline.forward takes a generator rather than a seed
    return torch.Generator(device=device).manual_seed(GLM_IMAGE_LPIPS_SEED)


def _generate_glm_image_lpips_image(model_path, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "GLM-Image checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    with _lpips_deterministic_algorithms():
        args = VisualGenArgs(
            model=model_path,
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            result = pipeline.forward(
                prompt=GLM_IMAGE_LPIPS_PROMPT,
                height=GLM_IMAGE_LPIPS_HEIGHT,
                width=GLM_IMAGE_LPIPS_WIDTH,
                num_inference_steps=GLM_IMAGE_LPIPS_NUM_INFERENCE_STEPS,
                guidance_scale=GLM_IMAGE_LPIPS_GUIDANCE_SCALE,
                generator=_glm_image_generator(pipeline.device),
            )
            generated_image = result.image[0].detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()

    save_image(generated_image, output_path)


def _generate_glm_image_feature_image(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image

    model_path = _lpips_model_path(case.checkpoint_subdir)
    _skip_if_missing(model_path, f"{case.checkpoint_subdir} checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    with _lpips_deterministic_algorithms(), _fixed_nvfp4_quantization_backend(case.features):
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(GLM_IMAGE_LPIPS_HEIGHT, GLM_IMAGE_LPIPS_WIDTH),
            num_frames=1,
        )
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=False)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(GLM_IMAGE_LPIPS_HEIGHT, GLM_IMAGE_LPIPS_WIDTH),
                num_frames=1,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=GLM_IMAGE_LPIPS_PROMPT,
                height=GLM_IMAGE_LPIPS_HEIGHT,
                width=GLM_IMAGE_LPIPS_WIDTH,
                num_inference_steps=GLM_IMAGE_LPIPS_NUM_INFERENCE_STEPS,
                guidance_scale=GLM_IMAGE_LPIPS_GUIDANCE_SCALE,
                generator=_glm_image_generator(pipeline.device),
            )
            _assert_single_device_feature_executed(pipeline, case.features)
            generated_image = result.image[0].detach().cpu()
        finally:
            try:
                if pipeline is not None:
                    _cleanup_single_device_feature_pipeline(pipeline)
                    del pipeline
            finally:
                _cleanup_cuda()

    save_image(generated_image, output_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_glm_image_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "glm_image_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "glm_image_lpips_golden.png", "GLM-Image LPIPS golden image"
    )
    _generate_glm_image_lpips_image(_lpips_model_path(GLM_IMAGE_CHECKPOINT_SUBDIR), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "glm_image",
        "image",
        GLM_IMAGE_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, GLM_IMAGE_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", GLM_IMAGE_ACCURACY_CASES)
def test_glm_image_feature_accuracy_against_golden(
    request,
    tmp_path,
    case,
    _visual_gen_lpips_scorer,
):
    generated_path = tmp_path / f"glm_image_{case.id}_generated.png"
    reference_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"{case.checkpoint_subdir} {case.id} LPIPS golden image",
    )

    _run_single_device_feature_generator(
        case.features, _generate_glm_image_feature_image, case, generated_path
    )
    score = _run_reusable_image_lpips_eval(
        f"glm_image_{case.id}",
        reference_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"glm_image_{case.id}_generated.png",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)
