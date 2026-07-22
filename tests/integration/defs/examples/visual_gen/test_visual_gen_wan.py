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

import contextlib
import os
from dataclasses import dataclass

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    VISUAL_GEN_OUTPUT_VIDEO,
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
    _assert_feature_torch_compile_installed,
    _assert_lpips_below_threshold,
    _assert_resolved_single_device_feature_config,
    _assert_single_device_feature_executed,
    _build_single_device_feature_args,
    _cleanup_cuda,
    _cleanup_single_device_feature_pipeline,
    _disable_inductor_compile_worker_quiesce,
    _generate_wan_lpips_video,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_video_lpips_eval,
    _run_vbench_and_report,
    _save_lpips_video_mp4,
    _skip_if_missing,
    _validate_single_feature_config,
    _visual_gen_output_path,
)

WAN_T2V_MODEL_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
WAN22_A14B_FP8_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-FP8"
WAN22_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-NVFP4"
WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-I2V-A14B-Diffusers-NVFP4"
WAN21_VSA_MODEL_SUBPATH = "Wan2.1-VSA-T2V-14B-720P-Diffusers"

WAN_FEATURE_CACHE_NUM_INFERENCE_STEPS = 12
WAN21_CACHE_GOLDEN_FILE = "wan21_12step_eager_lpips_golden_video.mp4"
WAN22_CACHE_GOLDEN_FILE = "wan22_12step_eager_lpips_golden_video.mp4"
WAN21_VSA_DENSE_GOLDEN_FILE = "wan21_vsa_dense_lpips_golden_video.mp4"
WAN21_VSA_HEIGHT = 720
WAN21_VSA_WIDTH = 1280
WAN21_VSA_NUM_FRAMES = 9
WAN21_VSA_NUM_INFERENCE_STEPS = 4
WAN21_VSA_SPARSITY = 0.9

WAN22_T2V_HIGH_NOISE_COEFFICIENTS = [
    -5784.54975374,
    5449.50911966,
    -1811.16591783,
    256.27178429,
    -13.02301147,
]
WAN22_T2V_LOW_NOISE_COEFFICIENTS = [
    2.39676752e03,
    -1.31110545e03,
    2.01331979e02,
    -8.29855975e00,
    1.37887774e-01,
]

WAN_STANDARD_SUPPORTED_FEATURES = frozenset(
    {
        "fp8-blockwise",
        "nvfp4",
        "teacache",
        "cache-dit",
        "cuda-graph",
        "torch-compile",
    }
)
# Rounded-up current-behavior envelopes from B200 calibration. Cache limits
# use the worst of three deterministic 12-step runs plus 0.05, rounded up to
# the next 0.05. These are regression limits, not claims that lossy profiles
# preserve eager visual quality closely.
WAN_FEATURE_THRESHOLDS = {
    ("wan21", "baseline"): 0.05,
    ("wan21", "fp8-blockwise"): 0.45,
    ("wan21", "nvfp4"): 0.75,
    ("wan21", "teacache"): 0.15,
    ("wan21", "cache-dit"): 0.15,
    ("wan21", "cuda-graph"): 0.05,
    ("wan21", "torch-compile"): 0.10,
    ("wan22", "baseline"): 0.05,
    ("wan22", "fp8-blockwise"): 0.55,
    ("wan22", "nvfp4"): 0.55,
    ("wan22", "teacache"): 0.40,
    ("wan22", "cache-dit"): 0.40,
    ("wan22", "cuda-graph"): 0.05,
    ("wan22", "torch-compile"): 0.25,
    ("wan21-vsa", "vsa"): 0.75,
}


@dataclass(frozen=True)
class WanModelSpec:
    id: str
    checkpoint_subdir: str
    golden_file: str
    cache_golden_file: str
    prompt: str
    negative_prompt: str | None
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float


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
    features: FeatureConfigState
    lpips_threshold: float
    vsa_sparsity: float = WAN21_VSA_SPARSITY


WAN_MODEL_SPECS = (
    WanModelSpec(
        id="wan21",
        checkpoint_subdir="Wan2.1-T2V-1.3B-Diffusers",
        golden_file="wan21_t2v_lpips_golden_video.mp4",
        cache_golden_file=WAN21_CACHE_GOLDEN_FILE,
        prompt=WAN21_LPIPS_PROMPT,
        negative_prompt=WAN21_LPIPS_NEGATIVE_PROMPT,
        height=WAN21_LPIPS_HEIGHT,
        width=WAN21_LPIPS_WIDTH,
        num_frames=WAN21_LPIPS_NUM_FRAMES,
        num_inference_steps=WAN21_LPIPS_NUM_INFERENCE_STEPS,
        guidance_scale=WAN21_LPIPS_GUIDANCE_SCALE,
    ),
    WanModelSpec(
        id="wan22",
        checkpoint_subdir="Wan2.2-T2V-A14B-Diffusers",
        golden_file="wan22_t2v_lpips_golden_video.mp4",
        cache_golden_file=WAN22_CACHE_GOLDEN_FILE,
        prompt=WAN22_LPIPS_PROMPT,
        negative_prompt=WAN22_LPIPS_NEGATIVE_PROMPT,
        height=WAN22_LPIPS_HEIGHT,
        width=WAN22_LPIPS_WIDTH,
        num_frames=WAN22_LPIPS_NUM_FRAMES,
        num_inference_steps=WAN22_LPIPS_NUM_INFERENCE_STEPS,
        guidance_scale=WAN22_LPIPS_GUIDANCE_SCALE,
    ),
)

WAN_FEATURE_PROFILES = (
    ("baseline", FeatureConfigState()),
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("teacache", FeatureConfigState(cache_backend="teacache")),
    ("cache-dit", FeatureConfigState(cache_backend="cache_dit")),
    ("cuda-graph", FeatureConfigState(cuda_graph=True)),
    ("torch-compile", FeatureConfigState(torch_compile=True)),
)


def _build_wan_accuracy_cases():
    cases = []
    for model in WAN_MODEL_SPECS:
        for profile_id, features in WAN_FEATURE_PROFILES:
            enabled_count = _validate_single_feature_config(
                features,
                WAN_STANDARD_SUPPORTED_FEATURES,
                model.id,
            )
            expected_count = 0 if profile_id == "baseline" else 1
            if enabled_count != expected_count:
                raise ValueError(
                    f"{model.id} profile {profile_id!r} must enable "
                    f"{expected_count} feature(s), got {enabled_count}"
                )
            is_cache_case = features.cache_backend != "off"
            case_id = f"{model.id}-{profile_id}"
            cases.append(
                pytest.param(
                    WanAccuracyCase(
                        id=case_id,
                        model_id=model.id,
                        checkpoint_subdir=model.checkpoint_subdir,
                        golden_file=(
                            model.cache_golden_file if is_cache_case else model.golden_file
                        ),
                        prompt=model.prompt,
                        negative_prompt=model.negative_prompt,
                        height=model.height,
                        width=model.width,
                        num_frames=model.num_frames,
                        num_inference_steps=(
                            WAN_FEATURE_CACHE_NUM_INFERENCE_STEPS
                            if is_cache_case
                            else model.num_inference_steps
                        ),
                        guidance_scale=model.guidance_scale,
                        features=features,
                        lpips_threshold=WAN_FEATURE_THRESHOLDS[(model.id, profile_id)],
                    ),
                    id=case_id,
                )
            )

    vsa_features = FeatureConfigState(sparse_attention="vsa")
    _validate_single_feature_config(vsa_features, {"vsa"}, "Wan 2.1 VSA")
    cases.append(
        pytest.param(
            WanAccuracyCase(
                id="wan21-vsa",
                model_id="wan21-vsa",
                checkpoint_subdir=WAN21_VSA_MODEL_SUBPATH,
                golden_file=WAN21_VSA_DENSE_GOLDEN_FILE,
                prompt=WAN22_LPIPS_PROMPT,
                negative_prompt=WAN22_LPIPS_NEGATIVE_PROMPT,
                height=WAN21_VSA_HEIGHT,
                width=WAN21_VSA_WIDTH,
                num_frames=WAN21_VSA_NUM_FRAMES,
                num_inference_steps=WAN21_VSA_NUM_INFERENCE_STEPS,
                guidance_scale=5.0,
                features=vsa_features,
                lpips_threshold=WAN_FEATURE_THRESHOLDS[("wan21-vsa", "vsa")],
            ),
            id="wan21-vsa",
        )
    )
    return cases


WAN_ACCURACY_CASES = _build_wan_accuracy_cases()


WAN_T2V_PROMPT = "A cute cat playing piano"
WAN_T2V_HEIGHT = 480
WAN_T2V_WIDTH = 832
WAN_T2V_NUM_FRAMES = 165


# Golden VBench scores from HF reference video (WAN 2.1 1.3B); TRT-LLM is compared against these.
VBENCH_WAN_GOLDEN_SCORES = {
    "subject_consistency": 0.8907,
    "background_consistency": 0.9274,
    "motion_smoothness": 0.9818,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.2928,
    "imaging_quality": 0.3812,
}


# TODO: Reference scores from bf16 baseline runs
VBENCH_WAN22_BF16_GOLDEN_SCORES = {
    "subject_consistency": 0.9103,
    "background_consistency": 0.9516,
    "motion_smoothness": 0.9693,
    "dynamic_degree": 0.0000,
    "aesthetic_quality": 0.6821,
    "imaging_quality": 0.3993,
}
VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES = {
    "subject_consistency": 0.9173,
    "background_consistency": 0.9717,
    "motion_smoothness": 0.9865,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5465,
    "imaging_quality": 0.7142,
}
VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES = {
    "subject_consistency": 0.9173,
    "background_consistency": 0.9717,
    "motion_smoothness": 0.9865,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5465,
    "imaging_quality": 0.7142,
}


@pytest.fixture(scope="session")
def wan21_bf16_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "wan21_bf16")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path("Wan2.1-T2V-1.3B-Diffusers"),
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
            _lpips_model_path("Wan2.2-T2V-A14B-Diffusers"),
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


def _wan_teacache_kwargs(case):
    if case.model_id != "wan22":
        return None
    return {
        "teacache_thresh": 0.15,
        "coefficients": WAN22_T2V_HIGH_NOISE_COEFFICIENTS,
        "coefficients_2": WAN22_T2V_LOW_NOISE_COEFFICIENTS,
    }


def _generate_wan_feature_video(case, output_path, *, vsa_sparsity=None):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    model_path = _lpips_model_path(case.checkpoint_subdir)
    _skip_if_missing(model_path, f"{case.checkpoint_subdir} checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    compiler_context = (
        contextlib.nullcontext()
        if case.features.torch_compile
        else torch.compiler.set_stance("force_eager")
    )
    resolved_vsa_sparsity = case.vsa_sparsity if vsa_sparsity is None else vsa_sparsity
    with _lpips_deterministic_algorithms(), compiler_context:
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(case.height, case.width),
            num_frames=case.num_frames,
            teacache_kwargs=_wan_teacache_kwargs(case),
            vsa_sparsity=resolved_vsa_sparsity,
        )
        skip_warmup = not (case.features.cuda_graph or case.features.torch_compile)
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=skip_warmup)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(case.height, case.width),
                num_frames=case.num_frames,
                vsa_sparsity=resolved_vsa_sparsity,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            _assert_feature_torch_compile_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=case.prompt,
                negative_prompt=case.negative_prompt,
                height=case.height,
                width=case.width,
                num_frames=case.num_frames,
                num_inference_steps=case.num_inference_steps,
                guidance_scale=case.guidance_scale,
                seed=WAN21_LPIPS_SEED,
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
        f"{case.id} eager LPIPS golden video",
    )
    _generate_wan_feature_video(case, generated_path)
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


def _generate_wan_video(llm_venv, model_subpath, output_subdir):
    """Generate a WAN video for a given model checkpoint.

    Returns the path to the generated .mp4, or calls pytest.skip if the model
    is not found under LLM_MODELS_ROOT.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, model_subpath)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {model_subpath} under scratch)"
        )
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)
    if os.path.isfile(output_path):
        return output_path

    visual_gen_args = VisualGenArgs(
        attention_config={"backend": "VANILLA"},
        parallel_config={"cfg_size": 2 if torch.cuda.device_count() >= 2 else 1},
        torch_compile_config={"enable": False},
        compilation_config={"skip_warmup": True},
    )
    visual_gen = VisualGen(model=model_path, args=visual_gen_args)
    try:
        frame_rate = visual_gen.default_params.frame_rate
        output = visual_gen.generate(
            inputs=WAN_T2V_PROMPT,
            params=VisualGenParams(
                height=WAN_T2V_HEIGHT,
                width=WAN_T2V_WIDTH,
                seed=42,
                num_frames=WAN_T2V_NUM_FRAMES,
                frame_rate=frame_rate,
            ),
        )
        assert output.error is None, f"unexpected error on WAN run: {output.error}"
        assert output.video is not None
        _save_lpips_video_mp4(output.video, output_path, frame_rate=frame_rate)
    finally:
        visual_gen.shutdown()

    assert os.path.isfile(output_path), f"Visual gen did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def wan22_a14b_fp8_video_path(_visual_gen_deps, llm_venv):
    """Generate video with Wan 2.2 A14B FP8 checkpoint."""
    return _generate_wan_video(llm_venv, WAN22_A14B_FP8_MODEL_SUBPATH, "wan22_fp8")


@pytest.fixture(scope="session")
def wan22_a14b_nvfp4_video_path(_visual_gen_deps, llm_venv):
    """Generate video with Wan 2.2 A14B NVFP4 checkpoint."""
    return _generate_wan_video(llm_venv, WAN22_A14B_NVFP4_MODEL_SUBPATH, "wan22_nvfp4")


def test_vbench_dimension_score_wan(vbench_repo_root, wan21_bf16_video_path, llm_venv):
    """Run VBench on WAN 2.1 BF16 video generated with the LPIPS config."""
    videos_dir = os.path.dirname(wan21_bf16_video_path)
    assert os.path.isfile(wan21_bf16_video_path), "WAN 2.1 BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.1 BF16",
        golden_scores=VBENCH_WAN_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_wan22_bf16(vbench_repo_root, wan22_bf16_video_path, llm_venv):
    """VBench accuracy for Wan 2.2 A14B BF16 generated with the LPIPS config."""
    videos_dir = os.path.dirname(wan22_bf16_video_path)
    assert os.path.isfile(wan22_bf16_video_path), "WAN 2.2 BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B BF16",
        golden_scores=VBENCH_WAN22_BF16_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_wan22_a14b_fp8(
    vbench_repo_root, wan22_a14b_fp8_video_path, llm_venv
):
    """VBench accuracy for Wan 2.2 A14B FP8 — full generate + evaluate."""
    videos_dir = os.path.dirname(wan22_a14b_fp8_video_path)
    assert os.path.isfile(wan22_a14b_fp8_video_path), "FP8 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B FP8",
        golden_scores=VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES,
        max_score_diff=0.06,
    )


def test_vbench_dimension_score_wan22_a14b_nvfp4(
    vbench_repo_root, wan22_a14b_nvfp4_video_path, llm_venv
):
    """VBench accuracy for Wan 2.2 A14B NVFP4 — full generate + evaluate."""
    videos_dir = os.path.dirname(wan22_a14b_nvfp4_video_path)
    assert os.path.isfile(wan22_a14b_nvfp4_video_path), "NVFP4 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B NVFP4",
        golden_scores=VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


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
    ``configs/wan2.2-t2v-fp4-1gpu.yaml`` (NVFP4 dynamic quant). The closest
    overlapping test is ``test_vbench_dimension_score_wan22_a14b_nvfp4``,
    which runs the same script but with a no-quant YAML synthesized at
    runtime and additionally evaluates VBench scores.
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
