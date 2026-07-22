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

"""Single-GPU integration and accuracy tests for Cosmos3."""

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
    _run_lpips_eval,
    _run_reusable_image_lpips_eval,
    _save_lpips_video_mp4,
    _skip_if_missing,
    _validate_single_feature_config,
)

# Cosmos3-Nano (text-to-video + text-to-image) — default-setting LPIPS golden.
# Params are the Cosmos3 720P defaults (cosmos3/defaults.py:COSMOS3_720P_PARAMS).
# Cosmos3 requires VANILLA attention and guardrails disabled in CI.
COSMOS3_NANO_MODEL_SUBPATH = "Cosmos3-Nano"
COSMOS3_LPIPS_PROMPT = "A serene mountain landscape with snow-capped peaks and a flowing river"
COSMOS3_LPIPS_HEIGHT = 720
COSMOS3_LPIPS_WIDTH = 1280
COSMOS3_LPIPS_T2V_NUM_FRAMES = 189
COSMOS3_LPIPS_T2I_NUM_FRAMES = 1
COSMOS3_LPIPS_NUM_INFERENCE_STEPS = 35
COSMOS3_LPIPS_GUIDANCE_SCALE = 6.0
COSMOS3_LPIPS_SEED = 42
COSMOS3_LPIPS_FRAME_RATE = 24.0
COSMOS3_LPIPS_THRESHOLD = 0.05

COSMOS3_FEATURE_HEIGHT = 256
COSMOS3_FEATURE_WIDTH = 256
COSMOS3_FEATURE_NUM_INFERENCE_STEPS = 4
COSMOS3_FEATURE_GOLDEN_FILE = "cosmos3_nano_feature_eager_lpips_golden.png"
COSMOS3_QUANTIZATION_IGNORE = [
    "language_model.*",
    "vae2llm",
    "llm2vae",
    "time_embedder.*",
]
# Rounded-up regression envelopes from the initial B200 calibration.
COSMOS3_FEATURE_THRESHOLDS = {
    "baseline": 0.05,
    "fp8-blockwise": 0.25,
    "nvfp4": 0.40,
    "torch-compile": 0.10,
}
COSMOS3_SUPPORTED_FEATURES = frozenset(COSMOS3_FEATURE_THRESHOLDS).difference({"baseline"})


@dataclass(frozen=True)
class Cosmos3AccuracyCase:
    id: str
    features: FeatureConfigState
    lpips_threshold: float


# CUDA graph is not included yet: Cosmos3 reads a CUDA scalar with ``.item()``
# inside the captured transformer forward, which CUDA rejects during capture.
COSMOS3_FEATURE_PROFILES = (
    ("baseline", FeatureConfigState()),
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("torch-compile", FeatureConfigState(torch_compile=True)),
)


def _build_cosmos3_accuracy_cases():
    cases = []
    for profile_id, features in COSMOS3_FEATURE_PROFILES:
        enabled_count = _validate_single_feature_config(
            features,
            COSMOS3_SUPPORTED_FEATURES,
            "Cosmos3",
        )
        expected_count = 0 if profile_id == "baseline" else 1
        if enabled_count != expected_count:
            raise ValueError(
                f"Cosmos3 profile {profile_id!r} must enable "
                f"{expected_count} feature(s), got {enabled_count}"
            )
        cases.append(
            pytest.param(
                Cosmos3AccuracyCase(
                    id=profile_id,
                    features=features,
                    lpips_threshold=COSMOS3_FEATURE_THRESHOLDS[profile_id],
                ),
                id=profile_id,
            )
        )
    return cases


COSMOS3_ACCURACY_CASES = _build_cosmos3_accuracy_cases()


def _run_cosmos3_lpips_pipeline(num_frames):
    """Run the Cosmos3-Nano pipeline (default setting, VANILLA attn, compile-off).

    Returns the generated video tensor ``(B, T, H, W, C)`` (T == ``num_frames``),
    or ``None`` if generation produced no video.  ``num_frames=1`` yields the
    single-frame text-to-image path.
    """
    # Cosmos3 re-reads the guardrail flag in __init__; set it before the pipeline loads.
    guardrails_env_key = "TRTLLM_DISABLE_COSMOS3_GUARDRAILS"
    previous_guardrails_env = os.environ.get(guardrails_env_key)
    os.environ[guardrails_env_key] = "1"
    try:
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
        from tensorrt_llm.visual_gen.args import (
            AttentionConfig,
            CompilationConfig,
            TorchCompileConfig,
            VisualGenArgs,
        )

        model_path = _lpips_model_path(COSMOS3_NANO_MODEL_SUBPATH)
        _skip_if_missing(model_path, "Cosmos3-Nano checkpoint", is_dir=True)
        _disable_inductor_compile_worker_quiesce()
        args = VisualGenArgs(
            model=model_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=False),
            attention_config=AttentionConfig(backend="VANILLA"),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=COSMOS3_LPIPS_PROMPT,
                    seed=COSMOS3_LPIPS_SEED,
                    height=COSMOS3_LPIPS_HEIGHT,
                    width=COSMOS3_LPIPS_WIDTH,
                    num_frames=num_frames,
                    num_inference_steps=COSMOS3_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=COSMOS3_LPIPS_GUIDANCE_SCALE,
                    frame_rate=COSMOS3_LPIPS_FRAME_RATE,
                    use_guardrails=False,
                )
            if result is None or result.video is None:
                return None
            return result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()
    finally:
        if previous_guardrails_env is None:
            os.environ.pop(guardrails_env_key, None)
        else:
            os.environ[guardrails_env_key] = previous_guardrails_env


def _generate_cosmos3_lpips_video(output_path):
    """Generate the Cosmos3-Nano text-to-video LPIPS sample."""
    video = _run_cosmos3_lpips_pipeline(COSMOS3_LPIPS_T2V_NUM_FRAMES)
    assert video is not None, "Cosmos3-Nano T2V LPIPS run produced no video"
    _save_lpips_video_mp4(video, output_path, frame_rate=COSMOS3_LPIPS_FRAME_RATE)


def _generate_cosmos3_lpips_image(output_path):
    """Generate the Cosmos3-Nano text-to-image LPIPS sample (single frame)."""
    from tensorrt_llm.media.encoding import save_image

    video = _run_cosmos3_lpips_pipeline(COSMOS3_LPIPS_T2I_NUM_FRAMES)
    assert video is not None, "Cosmos3-Nano T2I LPIPS run produced no frame"
    # video is (B, T, H, W, C); take the single frame -> (H, W, C) for save_image.
    save_image(video[0, 0], output_path)


def _generate_cosmos3_feature_image(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image

    guardrails_env_key = "TRTLLM_DISABLE_COSMOS3_GUARDRAILS"
    previous_guardrails_env = os.environ.get(guardrails_env_key)
    os.environ[guardrails_env_key] = "1"
    pipeline = None
    try:
        model_path = _lpips_model_path(COSMOS3_NANO_MODEL_SUBPATH)
        _skip_if_missing(model_path, "Cosmos3-Nano checkpoint", is_dir=True)
        _disable_inductor_compile_worker_quiesce()
        with _lpips_deterministic_algorithms():
            args = _build_single_device_feature_args(
                model_path,
                case.features,
                resolution=(COSMOS3_FEATURE_HEIGHT, COSMOS3_FEATURE_WIDTH),
                num_frames=1,
                quantization_kwargs={"ignore": COSMOS3_QUANTIZATION_IGNORE},
            )
            skip_warmup = not (case.features.cuda_graph or case.features.torch_compile)
            try:
                pipeline = PipelineLoader(args).load(skip_warmup=skip_warmup)
                _assert_resolved_single_device_feature_config(
                    pipeline,
                    case.features,
                    resolution=(COSMOS3_FEATURE_HEIGHT, COSMOS3_FEATURE_WIDTH),
                    num_frames=1,
                )
                _assert_feature_quantization_installed(pipeline, case.features)
                _assert_feature_torch_compile_installed(pipeline, case.features)
                result = pipeline.forward(
                    prompt=COSMOS3_LPIPS_PROMPT,
                    seed=COSMOS3_LPIPS_SEED,
                    height=COSMOS3_FEATURE_HEIGHT,
                    width=COSMOS3_FEATURE_WIDTH,
                    num_frames=1,
                    num_inference_steps=COSMOS3_FEATURE_NUM_INFERENCE_STEPS,
                    guidance_scale=COSMOS3_LPIPS_GUIDANCE_SCALE,
                    frame_rate=COSMOS3_LPIPS_FRAME_RATE,
                    use_guardrails=False,
                )
                _assert_single_device_feature_executed(pipeline, case.features)
                generated_image = result.video[0, 0].detach().cpu()
            finally:
                try:
                    if pipeline is not None:
                        _cleanup_single_device_feature_pipeline(pipeline)
                        del pipeline
                finally:
                    _cleanup_cuda()
        save_image(generated_image, output_path)
    finally:
        if previous_guardrails_env is None:
            os.environ.pop(guardrails_env_key, None)
        else:
            os.environ[guardrails_env_key] = previous_guardrails_env


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", COSMOS3_ACCURACY_CASES)
def test_cosmos3_feature_accuracy_against_golden(
    request,
    tmp_path,
    case,
    _visual_gen_lpips_scorer,
):
    generated_path = tmp_path / f"cosmos3_{case.id}_generated.png"
    golden_path = _golden_media_path(
        tmp_path,
        COSMOS3_FEATURE_GOLDEN_FILE,
        "Cosmos3-Nano compact eager LPIPS golden image",
    )
    _generate_cosmos3_feature_image(case, generated_path)
    score = _run_reusable_image_lpips_eval(
        f"cosmos3-{case.id}",
        golden_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"cosmos3_{case.id}_generated.png",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_nano_t2v_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "cosmos3_nano_t2v_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        "cosmos3_nano_t2v_lpips_golden_video.mp4",
        "Cosmos3-Nano T2V LPIPS golden video",
    )
    _generate_cosmos3_lpips_video(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_nano_t2v",
        "video",
        COSMOS3_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, COSMOS3_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_nano_t2i_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "cosmos3_nano_t2i_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "cosmos3_nano_t2i_lpips_golden.png", "Cosmos3-Nano T2I LPIPS golden image"
    )
    _generate_cosmos3_lpips_image(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_nano_t2i",
        "image",
        COSMOS3_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, COSMOS3_LPIPS_THRESHOLD)


def test_cosmos3_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/cosmos3/cosmos3.py with FP8 config end-to-end.

    Validates that the Cosmos3-Nano example script and ``configs/cosmos3-nano-1gpu.yaml``
    work together as documented. Uses the local Cosmos3-Nano checkpoint and
    the shared FP8 dynamic-quant config.
    """
    model_path = _lpips_model_path("Cosmos3-Nano")
    _skip_if_missing(model_path, "Cosmos3-Nano checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "cosmos3_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "cosmos3_output.mp4")

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "cosmos3-nano-1gpu.yaml"
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
            "--prompt",
            "A serene mountain landscape with snow-capped peaks and a flowing river",
            "--output_path",
            output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
