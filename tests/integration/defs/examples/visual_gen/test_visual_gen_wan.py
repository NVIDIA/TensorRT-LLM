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

"""Single-GPU integration and accuracy tests for Wan models and examples."""

import os
from dataclasses import dataclass

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    FASTWAN_LPIPS_FRAME_RATE,
    FASTWAN_LPIPS_GUIDANCE_SCALE,
    FASTWAN_LPIPS_HEIGHT,
    FASTWAN_LPIPS_NEGATIVE_PROMPT,
    FASTWAN_LPIPS_NUM_FRAMES,
    FASTWAN_LPIPS_NUM_INFERENCE_STEPS,
    FASTWAN_LPIPS_PROMPT,
    FASTWAN_LPIPS_SEED,
    FASTWAN_LPIPS_WIDTH,
    WAN21_LPIPS_GUIDANCE_SCALE,
    WAN21_LPIPS_HEIGHT,
    WAN21_LPIPS_NEGATIVE_PROMPT,
    WAN21_LPIPS_NUM_FRAMES,
    WAN21_LPIPS_NUM_INFERENCE_STEPS,
    WAN21_LPIPS_PROMPT,
    WAN21_LPIPS_SEED,
    WAN21_LPIPS_WIDTH,
    WAN22_LPIPS_FRAME_RATE,
    WAN22_LPIPS_GUIDANCE_SCALE,
    WAN22_LPIPS_HEIGHT,
    WAN22_LPIPS_NEGATIVE_PROMPT,
    WAN22_LPIPS_NUM_FRAMES,
    WAN22_LPIPS_NUM_INFERENCE_STEPS,
    WAN22_LPIPS_PROMPT,
    WAN22_LPIPS_SEED,
    WAN22_LPIPS_WIDTH,
    WAN_LPIPS_FRAME_RATE,
    WAN_LPIPS_THRESHOLD,
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
    _generate_wan_lpips_video,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_video_lpips_eval,
    _run_single_device_feature_generator,
    _save_lpips_video_mp4,
    _skip_if_missing,
    _validate_single_feature_config,
    _visual_gen_output_path,
)

WAN_T2V_MODEL_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
WAN22_T2V_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers"
WAN22_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-NVFP4"
FASTWAN_MODEL_SUBPATH = "FastWan2.2-TI2V-5B-FullAttn-Diffusers"
WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-I2V-A14B-Diffusers-NVFP4"
WAN_FEATURE_LPIPS_THRESHOLD = 0.05
WAN_STANDARD_SUPPORTED_FEATURES = frozenset({"fp8-blockwise", "nvfp4", "cuda-graph"})


@dataclass(frozen=True)
class WanModelSpec:
    id: str
    checkpoint_subdir: str
    prompt: str
    negative_prompt: str | None
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    seed: int


@dataclass(frozen=True)
class WanAccuracyCase:
    id: str
    model_id: str
    checkpoint_subdir: str
    golden_file: str
    prompt: str
    negative_prompt: str | None
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    features: FeatureConfigState
    lpips_threshold: float


WAN_MODEL_SPECS = (
    WanModelSpec(
        id="wan21",
        checkpoint_subdir=WAN_T2V_MODEL_SUBPATH,
        prompt=WAN21_LPIPS_PROMPT,
        negative_prompt=WAN21_LPIPS_NEGATIVE_PROMPT,
        height=WAN21_LPIPS_HEIGHT,
        width=WAN21_LPIPS_WIDTH,
        num_frames=WAN21_LPIPS_NUM_FRAMES,
        num_inference_steps=WAN21_LPIPS_NUM_INFERENCE_STEPS,
        guidance_scale=WAN21_LPIPS_GUIDANCE_SCALE,
        seed=WAN21_LPIPS_SEED,
    ),
    WanModelSpec(
        id="wan22",
        checkpoint_subdir=WAN22_T2V_MODEL_SUBPATH,
        prompt=WAN22_LPIPS_PROMPT,
        negative_prompt=WAN22_LPIPS_NEGATIVE_PROMPT,
        height=WAN22_LPIPS_HEIGHT,
        width=WAN22_LPIPS_WIDTH,
        num_frames=WAN22_LPIPS_NUM_FRAMES,
        num_inference_steps=WAN22_LPIPS_NUM_INFERENCE_STEPS,
        guidance_scale=WAN22_LPIPS_GUIDANCE_SCALE,
        seed=WAN22_LPIPS_SEED,
    ),
)

WAN_FEATURE_PROFILES = (
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("cuda-graph", FeatureConfigState(cuda_graph=True)),
)


def _build_wan_accuracy_cases():
    cases = []
    for model in WAN_MODEL_SPECS:
        for profile_id, features in WAN_FEATURE_PROFILES:
            _validate_single_feature_config(
                features,
                WAN_STANDARD_SUPPORTED_FEATURES,
                model.id,
            )
            case_id = f"{model.id}-{profile_id}"
            cases.append(
                pytest.param(
                    WanAccuracyCase(
                        id=case_id,
                        model_id=model.id,
                        checkpoint_subdir=model.checkpoint_subdir,
                        golden_file=(
                            f"{model.id}_{profile_id.replace('-', '_')}_lpips_golden_video.mp4"
                        ),
                        prompt=model.prompt,
                        negative_prompt=model.negative_prompt,
                        height=model.height,
                        width=model.width,
                        num_frames=model.num_frames,
                        num_inference_steps=model.num_inference_steps,
                        guidance_scale=model.guidance_scale,
                        seed=model.seed,
                        features=features,
                        lpips_threshold=WAN_FEATURE_LPIPS_THRESHOLD,
                    ),
                    id=case_id,
                )
            )

    return cases


WAN_ACCURACY_CASES = _build_wan_accuracy_cases()


@pytest.fixture(scope="session")
def wan21_bf16_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "wan21_bf16")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path(WAN_T2V_MODEL_SUBPATH),
            output_path,
            WAN21_LPIPS_PROMPT,
            WAN21_LPIPS_NEGATIVE_PROMPT,
            WAN21_LPIPS_HEIGHT,
            WAN21_LPIPS_WIDTH,
            WAN21_LPIPS_NUM_FRAMES,
            WAN21_LPIPS_NUM_INFERENCE_STEPS,
            WAN21_LPIPS_GUIDANCE_SCALE,
            WAN21_LPIPS_SEED,
            WAN_LPIPS_FRAME_RATE,
        )
    return output_path


@pytest.fixture(scope="session")
def wan22_bf16_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "wan22_bf16")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path(WAN22_T2V_MODEL_SUBPATH),
            output_path,
            WAN22_LPIPS_PROMPT,
            WAN22_LPIPS_NEGATIVE_PROMPT,
            WAN22_LPIPS_HEIGHT,
            WAN22_LPIPS_WIDTH,
            WAN22_LPIPS_NUM_FRAMES,
            WAN22_LPIPS_NUM_INFERENCE_STEPS,
            WAN22_LPIPS_GUIDANCE_SCALE,
            WAN22_LPIPS_SEED,
            WAN22_LPIPS_FRAME_RATE,
        )
    return output_path


@pytest.fixture(scope="session")
def fastwan_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "fastwan")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path(FASTWAN_MODEL_SUBPATH),
            output_path,
            FASTWAN_LPIPS_PROMPT,
            FASTWAN_LPIPS_NEGATIVE_PROMPT,
            FASTWAN_LPIPS_HEIGHT,
            FASTWAN_LPIPS_WIDTH,
            FASTWAN_LPIPS_NUM_FRAMES,
            FASTWAN_LPIPS_NUM_INFERENCE_STEPS,
            FASTWAN_LPIPS_GUIDANCE_SCALE,
            FASTWAN_LPIPS_SEED,
            FASTWAN_LPIPS_FRAME_RATE,
        )
    return output_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wan21_t2v_lpips_against_golden(request, tmp_path, wan21_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "wan21_t2v_lpips_golden_video.mp4", "Wan 2.1 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "wan21_t2v",
        "video",
        WAN21_LPIPS_PROMPT,
        golden_path,
        wan21_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        WAN_LPIPS_THRESHOLD,
        wan21_bf16_video_path,
        "wan21_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, WAN_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wan22_t2v_lpips_against_golden(request, tmp_path, wan22_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "wan22_t2v_lpips_golden_video.mp4", "Wan 2.2 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "wan22_t2v",
        "video",
        WAN22_LPIPS_PROMPT,
        golden_path,
        wan22_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        WAN_LPIPS_THRESHOLD,
        wan22_bf16_video_path,
        "wan22_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, WAN_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fastwan_lpips_against_golden(request, tmp_path, fastwan_video_path):
    golden_path = _golden_media_path(
        tmp_path, "fastwan_lpips_golden_video.mp4", "FastWan LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "fastwan",
        "video",
        FASTWAN_LPIPS_PROMPT,
        golden_path,
        fastwan_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        WAN_LPIPS_THRESHOLD,
        fastwan_video_path,
        "fastwan_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, WAN_LPIPS_THRESHOLD)


def _generate_wan_feature_video(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    model_path = _lpips_model_path(case.checkpoint_subdir)
    _skip_if_missing(model_path, f"{case.checkpoint_subdir} checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    with (
        _lpips_deterministic_algorithms(),
        torch.compiler.set_stance("force_eager"),
        _fixed_nvfp4_quantization_backend(case.features),
    ):
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(case.height, case.width),
            num_frames=case.num_frames,
        )
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=False)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(case.height, case.width),
                num_frames=case.num_frames,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=case.prompt,
                negative_prompt=case.negative_prompt,
                height=case.height,
                width=case.width,
                num_frames=case.num_frames,
                num_inference_steps=case.num_inference_steps,
                guidance_scale=case.guidance_scale,
                seed=case.seed,
            )
            assert result.video is not None, f"{case.id} produced no video"
            _assert_single_device_feature_executed(pipeline, case.features)
            generated_video = result.video.detach().cpu()
        finally:
            try:
                if pipeline is not None:
                    _cleanup_single_device_feature_pipeline(pipeline)
                    del pipeline
            finally:
                _cleanup_cuda()

    _save_lpips_video_mp4(generated_video, output_path, frame_rate=WAN_LPIPS_FRAME_RATE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", WAN_ACCURACY_CASES)
def test_wan_feature_accuracy_against_golden(
    request,
    tmp_path,
    case,
    _visual_gen_deps,
    _visual_gen_lpips_scorer,
):
    generated_path = tmp_path / f"{case.id}_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"{case.id} LPIPS golden video",
    )
    _run_single_device_feature_generator(
        case.features, _generate_wan_feature_video, case, generated_path
    )
    score = _run_reusable_video_lpips_eval(
        case.id,
        golden_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"{case.id}_generated.mp4",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)


def test_visual_gen_quickstart(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/quickstart_example.py end-to-end."""
    scratch_space = conftest.llm_models_root()
    model_src = os.path.join(scratch_space, WAN_T2V_MODEL_SUBPATH)
    if not os.path.isdir(model_src):
        pytest.skip(
            f"Model not found: {model_src} "
            f"(set LLM_MODELS_ROOT or place {WAN_T2V_MODEL_SUBPATH} under scratch)"
        )

    model_dst = os.path.join(llm_venv.get_working_directory(), "Wan-AI", WAN_T2V_MODEL_SUBPATH)
    if not os.path.islink(model_dst):
        os.makedirs(os.path.dirname(model_dst), exist_ok=True)
        os.symlink(model_src, model_dst, target_is_directory=True)

    script_path = os.path.join(llm_root, "examples", "visual_gen", "quickstart_example.py")
    venv_check_call(llm_venv, [script_path])

    output_path = os.path.join(llm_venv.get_working_directory(), "output.avi")
    assert os.path.isfile(output_path), f"Quickstart did not produce output.avi at {output_path}"


def test_visual_gen_api_walkthrough(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/api_walkthrough.py end-to-end."""
    scratch_space = conftest.llm_models_root()
    model_src = os.path.join(scratch_space, WAN_T2V_MODEL_SUBPATH)
    if not os.path.isdir(model_src):
        pytest.skip(
            f"Model not found: {model_src} "
            f"(set LLM_MODELS_ROOT or place {WAN_T2V_MODEL_SUBPATH} under scratch)"
        )

    model_dst = os.path.join(llm_venv.get_working_directory(), "Wan-AI", WAN_T2V_MODEL_SUBPATH)
    if not os.path.islink(model_dst):
        os.makedirs(os.path.dirname(model_dst), exist_ok=True)
        os.symlink(model_src, model_dst, target_is_directory=True)

    script_path = os.path.join(llm_root, "examples", "visual_gen", "api_walkthrough.py")
    venv_check_call(llm_venv, [script_path])

    output_path = os.path.join(llm_venv.get_working_directory(), "api_walkthrough_output.avi")
    assert os.path.isfile(output_path), f"API walkthrough did not produce {output_path}"


# =============================================================================
# Core example tests — run per-model scripts from examples/visual_gen/models/
# with shared YAML configs from examples/visual_gen/configs/.
# =============================================================================


def test_wan_t2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/wan_t2v.py with NVFP4 config end-to-end.

    This is a core example test: it validates that the per-model example script
    and the shared YAML config work together as documented in the README.
    Uses the pre-quantized Wan 2.2 T2V A14B NVFP4 checkpoint and the shared
    ``configs/wan2.2-t2v-fp4-1gpu.yaml`` (NVFP4 dynamic quant).
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, WAN22_A14B_NVFP4_MODEL_SUBPATH)
    assert os.path.isdir(model_path), (
        f"Model not found: {model_path} "
        f"(set LLM_MODELS_ROOT or place {WAN22_A14B_NVFP4_MODEL_SUBPATH} under models root)"
    )

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "wan_t2v_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "wan_t2v_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "wan_t2v.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "wan2.2-t2v-fp4-1gpu.yaml"
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


def test_wan_i2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/wan_i2v.py with NVFP4 config end-to-end.

    Validates that the Wan I2V example script and ``configs/wan2.2-i2v-fp4-1gpu.yaml``
    work together as documented. Uses the pre-quantized Wan 2.2 I2V A14B NVFP4
    checkpoint and the default input image (cat_piano.png) bundled with the examples.
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH} under models root)"
        )

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "wan_i2v_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "wan_i2v_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "wan_i2v.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "wan2.2-i2v-fp4-1gpu.yaml"
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
