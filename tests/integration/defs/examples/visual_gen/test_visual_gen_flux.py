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
"""Single-device FLUX visual-quality and example regression tests."""

import os
from dataclasses import dataclass

import pytest
import torch
from defs.common import venv_check_call
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

FLUX_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
FLUX_LPIPS_HEIGHT = 256
FLUX_LPIPS_WIDTH = 256
FLUX_LPIPS_NUM_INFERENCE_STEPS = 4
FLUX_LPIPS_GUIDANCE_SCALE = 3.5
FLUX_LPIPS_SEED = 42
FLUX_LPIPS_THRESHOLD = 0.05
FLUX_FEATURE_LPIPS_THRESHOLD = 0.05
FLUX_SUPPORTED_FEATURES = frozenset({"fp8", "fp8-blockwise", "nvfp4", "cuda-graph"})


@dataclass(frozen=True)
class FluxModelSpec:
    id: str
    checkpoint_subdir: str


@dataclass(frozen=True)
class FluxFeatureProfile:
    id: str
    features: FeatureConfigState


@dataclass(frozen=True)
class FluxAccuracyCase:
    id: str
    checkpoint_subdir: str
    golden_file: str
    features: FeatureConfigState
    lpips_threshold: float


FLUX_MODEL_SPECS = (
    FluxModelSpec(
        id="flux1",
        checkpoint_subdir="FLUX.1-dev",
    ),
    FluxModelSpec(
        id="flux2",
        checkpoint_subdir="FLUX.2-dev",
    ),
)

FLUX_FEATURE_PROFILES = (
    FluxFeatureProfile(
        id="fp8-blockwise",
        features=FeatureConfigState(quantization="FP8_BLOCK_SCALES"),
    ),
    FluxFeatureProfile(
        id="nvfp4",
        features=FeatureConfigState(quantization="NVFP4"),
    ),
    FluxFeatureProfile(
        id="cuda-graph",
        features=FeatureConfigState(cuda_graph=True),
    ),
)

FLUX_STATIC_QUANT_ACCURACY_CASES = (
    FluxAccuracyCase(
        id="flux1-fp8-static",
        checkpoint_subdir="FLUX.1-dev-FP8",
        golden_file="flux1_fp8_static_lpips_golden.png",
        features=FeatureConfigState(
            quantization="FP8",
            quantization_source="static",
        ),
        lpips_threshold=FLUX_FEATURE_LPIPS_THRESHOLD,
    ),
    FluxAccuracyCase(
        id="flux1-fp8-static-mha-quantize",
        checkpoint_subdir="FLUX.1-dev-FP8",
        golden_file="flux1_fp8_static_mha_quantize_lpips_golden.png",
        features=FeatureConfigState(
            quantization="FP8",
            quantization_source="static",
            mha_quantize=True,
        ),
        lpips_threshold=FLUX_FEATURE_LPIPS_THRESHOLD,
    ),
    FluxAccuracyCase(
        id="flux1-nvfp4-static",
        checkpoint_subdir="FLUX.1-dev-NVFP4",
        golden_file="flux1_nvfp4_static_lpips_golden.png",
        features=FeatureConfigState(
            quantization="NVFP4",
            quantization_source="static",
        ),
        lpips_threshold=FLUX_FEATURE_LPIPS_THRESHOLD,
    ),
    FluxAccuracyCase(
        id="flux1-nvfp4-static-mha-quantize",
        checkpoint_subdir="FLUX.1-dev-NVFP4",
        golden_file="flux1_nvfp4_static_mha_quantize_lpips_golden.png",
        features=FeatureConfigState(
            quantization="NVFP4",
            quantization_source="static",
            mha_quantize=True,
        ),
        lpips_threshold=FLUX_FEATURE_LPIPS_THRESHOLD,
    ),
)


def _build_flux_accuracy_cases():
    cases = []
    for model in FLUX_MODEL_SPECS:
        for profile in FLUX_FEATURE_PROFILES:
            _validate_single_feature_config(
                profile.features,
                FLUX_SUPPORTED_FEATURES,
                "FLUX",
            )
            case_id = f"{model.id}-{profile.id}"
            cases.append(
                pytest.param(
                    FluxAccuracyCase(
                        id=case_id,
                        checkpoint_subdir=model.checkpoint_subdir,
                        golden_file=(f"{model.id}_{profile.id.replace('-', '_')}_lpips_golden.png"),
                        features=profile.features,
                        lpips_threshold=FLUX_FEATURE_LPIPS_THRESHOLD,
                    ),
                    id=case_id,
                )
            )
    for case in FLUX_STATIC_QUANT_ACCURACY_CASES:
        _validate_single_feature_config(case.features, FLUX_SUPPORTED_FEATURES, "FLUX")
        cases.append(pytest.param(case, id=case.id))
    return cases


FLUX_ACCURACY_CASES = _build_flux_accuracy_cases()


def _generate_flux_lpips_image(model_path, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "FLUX checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    with _lpips_deterministic_algorithms():
        args = VisualGenArgs(
            model=model_path,
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            result = pipeline.forward(
                prompt=FLUX_LPIPS_PROMPT,
                height=FLUX_LPIPS_HEIGHT,
                width=FLUX_LPIPS_WIDTH,
                num_inference_steps=FLUX_LPIPS_NUM_INFERENCE_STEPS,
                guidance_scale=FLUX_LPIPS_GUIDANCE_SCALE,
                seed=FLUX_LPIPS_SEED,
            )
            generated_image = result.image[0].detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()

    save_image(generated_image, output_path)


def _generate_flux_image(case, output_path):
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
            resolution=(FLUX_LPIPS_HEIGHT, FLUX_LPIPS_WIDTH),
            num_frames=1,
        )
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=False)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(FLUX_LPIPS_HEIGHT, FLUX_LPIPS_WIDTH),
                num_frames=1,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=FLUX_LPIPS_PROMPT,
                height=FLUX_LPIPS_HEIGHT,
                width=FLUX_LPIPS_WIDTH,
                num_inference_steps=FLUX_LPIPS_NUM_INFERENCE_STEPS,
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
def test_flux1_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "flux1_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "flux1_lpips_golden.png", "FLUX.1 LPIPS golden image"
    )
    _generate_flux_lpips_image(_lpips_model_path("FLUX.1-dev"), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "flux1",
        "image",
        FLUX_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, FLUX_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flux2_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "flux2_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "flux2_lpips_golden.png", "FLUX.2 LPIPS golden image"
    )
    _generate_flux_lpips_image(_lpips_model_path("FLUX.2-dev"), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "flux2",
        "image",
        FLUX_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, FLUX_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", FLUX_ACCURACY_CASES)
def test_flux_accuracy_against_golden(request, tmp_path, case, _visual_gen_lpips_scorer):
    generated_path = tmp_path / f"{case.id}_generated.png"
    reference_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"{case.checkpoint_subdir} LPIPS golden image",
    )

    _run_single_device_feature_generator(case.features, _generate_flux_image, case, generated_path)
    score = _run_reusable_image_lpips_eval(
        case.id,
        reference_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"{case.id}_generated.png",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)


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


def test_flux2_reference_image_example(_visual_gen_deps, llm_root, llm_venv, tmp_path):
    """Run the FLUX.2 example with the existing reference-image request argument."""
    model_path = _lpips_model_path("FLUX.2-dev")
    _skip_if_missing(model_path, "FLUX.2-dev checkpoint", is_dir=True)
    reference_path = _golden_media_path(
        tmp_path, "flux2_lpips_golden.png", "FLUX.2 reference image"
    )

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "flux2_reference_image_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "flux2_reference_image_output.png")
    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "flux2.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "flux2-dev-fp4-1gpu.yaml"
    )

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--image",
            str(reference_path),
            "--height",
            "256",
            "--width",
            "256",
            "--num_inference_steps",
            "4",
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
