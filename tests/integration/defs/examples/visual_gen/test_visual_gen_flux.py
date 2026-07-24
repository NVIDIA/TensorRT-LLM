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
"""Single-device FLUX visual-quality and example regression tests.

The case matrix intentionally covers the all-default baseline plus exactly one
feature delta at a time. All supported feature cases run in B200 post-merge L0;
the cache cases use fixed 50-step eager goldens.
"""

import os
from dataclasses import dataclass

import pytest
import torch
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    FeatureConfigState,
    _assert_feature_quantization_installed,
    _assert_feature_torch_compile_installed,
    _assert_lpips_below_threshold,
    _assert_resolved_single_device_feature_config,
    _assert_single_device_feature_executed,
    _build_single_device_feature_args,
    _cleanup_cuda,
    _cleanup_single_device_feature_pipeline,
    _disable_inductor_compile_worker_quiesce,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_reusable_image_lpips_eval,
    _skip_if_missing,
    _validate_single_feature_config,
)

FLUX_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
FLUX_LPIPS_HEIGHT = 256
FLUX_LPIPS_WIDTH = 256
FLUX_LPIPS_NUM_INFERENCE_STEPS = 4
FLUX_CACHE_LPIPS_NUM_INFERENCE_STEPS = 50
FLUX_LPIPS_GUIDANCE_SCALE = 3.5
FLUX_LPIPS_SEED = 42

# The eager BF16 baselines are the established accuracy gates. Non-baseline
# envelopes below are provisional, rounded-up limits from the initial B200
# calibration run; all supported single-feature profiles run in post-merge L0
# to collect regression coverage while those limits are refined. The 50-step
# cache profiles use fixed feature-specific goldens.
FLUX_INITIAL_OBSERVED_LPIPS_ENVELOPES = {
    "flux1": 0.05,
    "flux2": 0.05,
}
FLUX_SUPPORTED_FEATURES = frozenset(
    {
        "fp8-blockwise",
        "nvfp4",
        "teacache",
        "cache-dit",
        "cuda-graph",
        "torch-compile",
    }
)
FLUX_CASE_OBSERVED_LPIPS_ENVELOPES = {
    ("flux1", "baseline"): 0.05,
    ("flux1", "fp8-blockwise"): 0.20,
    ("flux1", "nvfp4"): 0.26,
    ("flux1", "teacache"): 0.10,
    ("flux1", "cache-dit"): 0.10,
    ("flux2", "baseline"): 0.05,
    ("flux2", "nvfp4"): 0.07,
    ("flux2", "teacache"): 0.05,
    ("flux2", "cache-dit"): 0.05,
}


@dataclass(frozen=True)
class FluxModelSpec:
    id: str
    checkpoint_subdir: str
    golden_file: str
    cache_golden_file: str


@dataclass(frozen=True)
class FluxFeatureProfile:
    id: str
    features: FeatureConfigState
    num_inference_steps: int = FLUX_LPIPS_NUM_INFERENCE_STEPS


@dataclass(frozen=True)
class FluxAccuracyCase:
    id: str
    checkpoint_subdir: str
    golden_file: str
    features: FeatureConfigState
    num_inference_steps: int
    observed_lpips_envelope: float


FLUX_MODEL_SPECS = (
    FluxModelSpec(
        id="flux1",
        checkpoint_subdir="FLUX.1-dev",
        golden_file="flux1_lpips_golden.png",
        cache_golden_file="flux1_50step_eager_lpips_golden.png",
    ),
    FluxModelSpec(
        id="flux2",
        checkpoint_subdir="FLUX.2-dev",
        golden_file="flux2_lpips_golden.png",
        cache_golden_file="flux2_50step_eager_lpips_golden.png",
    ),
)

# Each explicit profile differs from the baseline on exactly one
# FeatureConfigState field. Cache profiles use a fixed 50-step eager BF16
# golden so they exercise the production-default workload without generating
# an expensive live eager reference in every test run.
FLUX_FEATURE_PROFILES = (
    FluxFeatureProfile(id="baseline", features=FeatureConfigState()),
    FluxFeatureProfile(
        id="fp8-blockwise",
        features=FeatureConfigState(quantization="FP8_BLOCK_SCALES"),
    ),
    FluxFeatureProfile(
        id="nvfp4",
        features=FeatureConfigState(quantization="NVFP4"),
    ),
    FluxFeatureProfile(
        id="teacache",
        features=FeatureConfigState(cache_backend="teacache"),
        num_inference_steps=FLUX_CACHE_LPIPS_NUM_INFERENCE_STEPS,
    ),
    FluxFeatureProfile(
        id="cache-dit",
        features=FeatureConfigState(cache_backend="cache_dit"),
        num_inference_steps=FLUX_CACHE_LPIPS_NUM_INFERENCE_STEPS,
    ),
    FluxFeatureProfile(
        id="cuda-graph",
        features=FeatureConfigState(cuda_graph=True),
    ),
    FluxFeatureProfile(
        id="torch-compile",
        features=FeatureConfigState(torch_compile=True),
    ),
)


def _build_flux_accuracy_cases():
    cases = []
    for model in FLUX_MODEL_SPECS:
        for profile in FLUX_FEATURE_PROFILES:
            enabled_feature_count = _validate_single_feature_config(
                profile.features,
                FLUX_SUPPORTED_FEATURES,
                "FLUX",
            )
            expected_feature_count = 0 if profile.id == "baseline" else 1
            if enabled_feature_count != expected_feature_count:
                raise ValueError(
                    f"FLUX profile {profile.id!r} must enable "
                    f"{expected_feature_count} feature(s), got {enabled_feature_count}"
                )
            case_id = f"{model.id}-{profile.id}"
            golden_file = (
                model.cache_golden_file
                if profile.features.cache_backend != "off"
                else model.golden_file
            )
            observed_lpips_envelope = FLUX_CASE_OBSERVED_LPIPS_ENVELOPES.get(
                (model.id, profile.id),
                FLUX_INITIAL_OBSERVED_LPIPS_ENVELOPES[model.id],
            )
            cases.append(
                pytest.param(
                    FluxAccuracyCase(
                        id=case_id,
                        checkpoint_subdir=model.checkpoint_subdir,
                        golden_file=golden_file,
                        features=profile.features,
                        num_inference_steps=profile.num_inference_steps,
                        observed_lpips_envelope=observed_lpips_envelope,
                    ),
                    id=case_id,
                )
            )
    return cases


FLUX_ACCURACY_CASES = _build_flux_accuracy_cases()


def _generate_flux_image(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image

    model_path = _lpips_model_path(case.checkpoint_subdir)
    _skip_if_missing(model_path, f"{case.checkpoint_subdir} checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    with _lpips_deterministic_algorithms():
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(FLUX_LPIPS_HEIGHT, FLUX_LPIPS_WIDTH),
            num_frames=1,
        )
        skip_warmup = not (case.features.cuda_graph or case.features.torch_compile)
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=skip_warmup)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(FLUX_LPIPS_HEIGHT, FLUX_LPIPS_WIDTH),
                num_frames=1,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            _assert_feature_torch_compile_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=FLUX_LPIPS_PROMPT,
                height=FLUX_LPIPS_HEIGHT,
                width=FLUX_LPIPS_WIDTH,
                num_inference_steps=case.num_inference_steps,
                guidance_scale=FLUX_LPIPS_GUIDANCE_SCALE,
                seed=FLUX_LPIPS_SEED,
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
@pytest.mark.parametrize("case", FLUX_ACCURACY_CASES)
def test_flux_accuracy_against_golden(request, tmp_path, case, _visual_gen_lpips_scorer):
    generated_path = tmp_path / f"{case.id}_generated.png"
    reference_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"{case.checkpoint_subdir} LPIPS golden image",
    )

    _generate_flux_image(case, generated_path)
    score = _run_reusable_image_lpips_eval(
        case.id,
        reference_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.observed_lpips_envelope,
        generated_path,
        f"{case.id}_generated.png",
    )
    _assert_lpips_below_threshold(score, case.observed_lpips_envelope)


def test_flux1_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the FLUX.1 example with the supported single-GPU NVFP4 config."""
    model_path = _lpips_model_path("FLUX.1-dev")
    _skip_if_missing(model_path, "FLUX.1-dev checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "flux1_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "flux1_output.png")
    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "flux1.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "flux1-dev-fp4-1gpu.yaml"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"
    assert os.path.isfile(config_path), f"Config not found: {config_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"


def test_flux2_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the FLUX.2 example with the supported single-GPU NVFP4 config."""
    model_path = _lpips_model_path("FLUX.2-dev")
    _skip_if_missing(model_path, "FLUX.2-dev checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "flux2_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "flux2_output.png")
    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "flux2.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "flux2-dev-fp4-1gpu.yaml"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"
    assert os.path.isfile(config_path), f"Config not found: {config_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
