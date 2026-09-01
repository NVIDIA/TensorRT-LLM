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

import contextlib
import json
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
    _lpips_pinned_fp32_matmul_precision,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_image_lpips_eval,
    _run_single_device_feature_generator,
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
# 9 frames = 3 latent frames: latents (0, 1) are pinned to the V2V reference,
# latent 2 (pixel frames 5-8) is generated. Frame 8 is the golden-compared frame.
COSMOS3_LPIPS_V2V_NUM_FRAMES = 9
COSMOS3_LPIPS_V2V_FREE_FRAME_INDEX = 8
COSMOS3_LPIPS_NUM_INFERENCE_STEPS = 35
COSMOS3_LPIPS_GUIDANCE_SCALE = 6.0
COSMOS3_LPIPS_SEED = 42
COSMOS3_LPIPS_FRAME_RATE = 24.0
COSMOS3_LPIPS_THRESHOLD = 0.05
COSMOS3_I2V_4STEP_MODEL_SUBPATH = "Cosmos3-Super-Image2Video-4Step"
COSMOS3_I2V_4STEP_LPIPS_PROMPT = (
    "The orange sphere slowly rises while the camera pans right across the scene"
)
COSMOS3_I2V_4STEP_LPIPS_NUM_FRAMES = 29
# Fixed by the distilled checkpoint (scheduler t_list / CFG baked into weights).
COSMOS3_I2V_4STEP_LPIPS_NUM_INFERENCE_STEPS = 4
COSMOS3_I2V_4STEP_LPIPS_GUIDANCE_SCALE = 1.0
# Golden is diffusers-produced (cross-stack), not a TRT-LLM self-golden:
# 0.0563 measured at creation + headroom for ~0.04 cross-host kernel drift
# (see _preserve_lpips_candidate_on_failure). Provenance:
# golden/visual_gen_lpips/cosmos3_i2v_4step_lpips_golden_video.json.
COSMOS3_I2V_4STEP_LPIPS_THRESHOLD = 0.10


COSMOS3_FEATURE_LPIPS_THRESHOLD = 0.05
COSMOS3_QUANTIZATION_IGNORE = [
    "language_model.*",
    "vae2llm",
    "llm2vae",
    "time_embedder.*",
]
COSMOS3_SUPPORTED_FEATURES = frozenset({"fp8-blockwise", "nvfp4"})


@dataclass(frozen=True)
class Cosmos3AccuracyCase:
    id: str
    golden_file: str
    features: FeatureConfigState
    lpips_threshold: float


# CUDA graph is not included yet: Cosmos3 reads a CUDA scalar with ``.item()``
# inside the captured transformer forward, which CUDA rejects during capture.
COSMOS3_FEATURE_PROFILES = (
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
)


def _build_cosmos3_accuracy_cases():
    cases = []
    for profile_id, features in COSMOS3_FEATURE_PROFILES:
        _validate_single_feature_config(
            features,
            COSMOS3_SUPPORTED_FEATURES,
            "Cosmos3",
        )
        cases.append(
            pytest.param(
                Cosmos3AccuracyCase(
                    id=profile_id,
                    golden_file=(f"cosmos3_nano_{profile_id.replace('-', '_')}_lpips_golden.png"),
                    features=features,
                    lpips_threshold=COSMOS3_FEATURE_LPIPS_THRESHOLD,
                ),
                id=profile_id,
            )
        )
    return cases


COSMOS3_ACCURACY_CASES = _build_cosmos3_accuracy_cases()


def _run_cosmos3_lpips_pipeline(num_frames, video=None):
    """Run the Cosmos3-Nano pipeline (default setting, VANILLA attn, compile-off).

    Returns the generated video tensor ``(B, T, H, W, C)`` (T == ``num_frames``),
    or ``None`` if generation produced no video.  ``num_frames=1`` yields the
    single-frame text-to-image path; passing ``video`` (encoded MP4 bytes,
    decoded on the worker's NVDEC) yields the video-to-video path.
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
            # Pin fp32-matmul arithmetic: the goldens are cut and compared
            # under "highest" so they reproduce on both PyPI and NGC torch.
            with torch.no_grad(), _lpips_pinned_fp32_matmul_precision():
                result = pipeline.forward(
                    prompt=COSMOS3_LPIPS_PROMPT,
                    # The goldens were generated against an empty uncond branch,
                    # so pin it rather than inheriting the video-mode default.
                    negative_prompt="",
                    seed=COSMOS3_LPIPS_SEED,
                    height=COSMOS3_LPIPS_HEIGHT,
                    width=COSMOS3_LPIPS_WIDTH,
                    num_frames=num_frames,
                    num_inference_steps=COSMOS3_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=COSMOS3_LPIPS_GUIDANCE_SCALE,
                    frame_rate=COSMOS3_LPIPS_FRAME_RATE,
                    use_guardrails=False,
                    video=video,
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


# 5-frame 720p conditioning window (the default
# ``max(condition_video_latent_indexes) * 4 + 1``): gray background with a
# block moving 40 px/frame -- a real structure signal. Encoded once offline
# with ffmpeg/libx264 (H.264 decode is bit-exact by spec, so NVDEC output is
# deterministic across machines); provenance in test_data/README.md.
COSMOS3_LPIPS_V2V_REFERENCE_MP4 = os.path.join(
    os.path.dirname(__file__), "test_data", "cosmos3_v2v_lpips_reference.mp4"
)


def _cosmos3_v2v_lpips_reference_bytes():
    _skip_if_missing(COSMOS3_LPIPS_V2V_REFERENCE_MP4, "Cosmos3 V2V LPIPS reference fixture")
    with open(COSMOS3_LPIPS_V2V_REFERENCE_MP4, "rb") as f:
        return f.read()


def _generate_cosmos3_v2v_lpips_frame(output_path):
    """Generate the Cosmos3-Nano video-to-video LPIPS sample (free frame only)."""
    from tensorrt_llm.media.encoding import save_image

    video = _run_cosmos3_lpips_pipeline(
        COSMOS3_LPIPS_V2V_NUM_FRAMES, video=_cosmos3_v2v_lpips_reference_bytes()
    )
    assert video is not None, "Cosmos3-Nano V2V LPIPS run produced no video"
    # video is (B, T, H, W, C); take the free frame -> (H, W, C) for save_image.
    save_image(video[0, COSMOS3_LPIPS_V2V_FREE_FRAME_INDEX], output_path)


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
        # Pin fp32-matmul arithmetic only for the profiles whose goldens are
        # re-baselined under it. NVFP4's golden is waived (nvbugs/6572800) and
        # not re-cut here, so it keeps generating under the host default; pin
        # it when that golden is re-cut. Note the NVFP4 profile reaches this
        # same function inside a spawned process, so the guard has to be here
        # rather than at the call site.
        precision_context = (
            contextlib.nullcontext()
            if case.features.quantization == "NVFP4"
            else _lpips_pinned_fp32_matmul_precision()
        )
        with (
            _lpips_deterministic_algorithms(),
            precision_context,
            _fixed_nvfp4_quantization_backend(case.features),
        ):
            args = _build_single_device_feature_args(
                model_path,
                case.features,
                resolution=(COSMOS3_LPIPS_HEIGHT, COSMOS3_LPIPS_WIDTH),
                num_frames=1,
                quantization_kwargs={"ignore": COSMOS3_QUANTIZATION_IGNORE},
            )
            try:
                pipeline = PipelineLoader(args).load(skip_warmup=False)
                _assert_resolved_single_device_feature_config(
                    pipeline,
                    case.features,
                    resolution=(COSMOS3_LPIPS_HEIGHT, COSMOS3_LPIPS_WIDTH),
                    num_frames=1,
                )
                _assert_feature_quantization_installed(pipeline, case.features)
                result = pipeline.forward(
                    prompt=COSMOS3_LPIPS_PROMPT,
                    # The goldens were generated against an empty uncond branch,
                    # so pin it rather than inheriting the video-mode default.
                    negative_prompt="",
                    seed=COSMOS3_LPIPS_SEED,
                    height=COSMOS3_LPIPS_HEIGHT,
                    width=COSMOS3_LPIPS_WIDTH,
                    num_frames=1,
                    num_inference_steps=COSMOS3_LPIPS_NUM_INFERENCE_STEPS,
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
        case.golden_file,
        f"Cosmos3-Nano {case.id} LPIPS golden image",
    )
    _run_single_device_feature_generator(
        case.features, _generate_cosmos3_feature_image, case, generated_path
    )
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
def test_cosmos3_nano_v2v_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "cosmos3_nano_v2v_generated_frame.png"
    golden_path = _golden_media_path(
        tmp_path,
        "cosmos3_nano_v2v_lpips_golden_frame.png",
        "Cosmos3-Nano V2V LPIPS golden frame",
    )
    _generate_cosmos3_v2v_lpips_frame(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_nano_v2v",
        "image",
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


def test_cosmos3_t2i_4step_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the distilled T2I checkpoint through the recommended invocation.

    Validates the documented deployment for ``Cosmos3-Super-Text2Image-4Step``:
    the example script with ``configs/cosmos3-t2i-1gpu.yaml`` (T2I warmup
    shapes) and ``--output_type image``. Steps/guidance come from the
    checkpoint's fixed distilled schedule; the run must produce an image.
    """
    model_path = _lpips_model_path("Cosmos3-Super-Text2Image-4Step")
    _skip_if_missing(model_path, "Cosmos3-Super-Text2Image-4Step checkpoint", is_dir=True)

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "cosmos3_t2i_4step_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "cosmos3_t2i_4step_output.png")
    if os.path.exists(output_path):
        os.remove(output_path)

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "cosmos3-t2i-1gpu.yaml"
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
            "A ceramic teapot pouring steaming tea into a cup, morning window light",
            "--output_type",
            "image",
            "--output_path",
            output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
    assert os.path.getsize(output_path) > 0, f"Example produced an empty image at {output_path}"


def _write_cosmos3_i2v_conditioning_image(path):
    """Deterministic 1280x720 conditioning image for the I2V smoke test.

    Gradient sky plus simple shapes, so I2V has real structure to animate
    without shipping an asset file.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 720))
    draw = ImageDraw.Draw(image)
    for y in range(720):
        draw.line([(0, y), (1280, y)], fill=(30, 60 + y // 8, 140))
    draw.ellipse([480, 200, 800, 520], fill=(230, 120, 40), outline=(255, 255, 255), width=6)
    draw.rectangle([100, 500, 400, 680], fill=(40, 160, 90))
    draw.polygon([(1000, 600), (1120, 380), (1240, 600)], fill=(200, 200, 60))
    image.save(path)


def test_cosmos3_i2v_4step_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the distilled I2V checkpoint through the recommended invocation.

    Validates the documented deployment for ``Cosmos3-Super-Image2Video-4Step``:
    the example script with a conditioning image and no config override (the
    omni defaults — 720p x 189 frames — are the deployed shape). Steps,
    guidance, and the system-prompt default come from the checkpoint; the run
    must produce a video.
    """
    model_path = _lpips_model_path("Cosmos3-Super-Image2Video-4Step")
    _skip_if_missing(model_path, "Cosmos3-Super-Image2Video-4Step checkpoint", is_dir=True)

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "cosmos3_i2v_4step_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, "conditioning.png")
    _write_cosmos3_i2v_conditioning_image(image_path)
    output_path = os.path.join(out_dir, "cosmos3_i2v_4step_output.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--prompt",
            "The orange sphere slowly rises while the camera pans right across the scene",
            "--image_path",
            image_path,
            "--output_path",
            output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
    assert os.path.getsize(output_path) > 0, f"Example produced an empty video at {output_path}"


def _run_cosmos3_i2v_4step_lpips_pipeline(image_path):
    """Run the distilled I2V pipeline on the deterministic conditioning image.

    VANILLA attention, compile-off. Returns the generated video tensor
    ``(B, T, H, W, C)``, or ``None`` if generation produced no video.
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

        model_path = _lpips_model_path(COSMOS3_I2V_4STEP_MODEL_SUBPATH)
        _skip_if_missing(model_path, "Cosmos3-Super-Image2Video-4Step checkpoint", is_dir=True)
        _disable_inductor_compile_worker_quiesce()
        args = VisualGenArgs(
            model=model_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=False),
            attention_config=AttentionConfig(backend="VANILLA"),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            # Deliberately NOT pinned with _lpips_pinned_fp32_matmul_precision:
            # this golden is a diffusers cross-stack reference whose provenance
            # records no fp32-matmul state, so whether pinning improves or
            # degrades agreement is unknown. The test is skipped in CI anyway
            # (checkpoint absent), and its golden is not re-cut here. Pin this
            # path when the golden is re-cut and the flag can be recorded.
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=COSMOS3_I2V_4STEP_LPIPS_PROMPT,
                    # The goldens were generated against an empty uncond branch,
                    # so pin it rather than inheriting the video-mode default.
                    negative_prompt="",
                    seed=COSMOS3_LPIPS_SEED,
                    image=image_path,
                    height=COSMOS3_LPIPS_HEIGHT,
                    width=COSMOS3_LPIPS_WIDTH,
                    num_frames=COSMOS3_I2V_4STEP_LPIPS_NUM_FRAMES,
                    # Direct forward() calls must pass checkpoint-valid sampling
                    # values (the signature defaults are the base-checkpoint
                    # video table, which a distilled checkpoint rejects).
                    num_inference_steps=COSMOS3_I2V_4STEP_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=COSMOS3_I2V_4STEP_LPIPS_GUIDANCE_SCALE,
                    frame_rate=COSMOS3_LPIPS_FRAME_RATE,
                    # The checkpoint declares default_use_system_prompt=true and
                    # the golden was generated with it; forward()'s signature
                    # default is the historical False, so pass it explicitly.
                    use_system_prompt=True,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_i2v_4step_lpips_against_golden(_visual_gen_deps, request, tmp_path):
    """Quality gate for the distilled I2V checkpoint against a diffusers golden.

    Unlike the self-goldens of the other models, the golden video here was
    produced by the reference implementation (diffusers modular pipeline,
    PR #14177, with its per-step SDE noise made generator-seeded) — so this
    gate checks the denoising trajectory against the reference, not just
    regression against a past TRT-LLM run. Full provenance:
    ``golden/visual_gen_lpips/cosmos3_i2v_4step_lpips_golden_video.json``.
    """
    image_path = str(tmp_path / "cosmos3_i2v_4step_conditioning.png")
    _write_cosmos3_i2v_conditioning_image(image_path)
    generated_path = tmp_path / "cosmos3_i2v_4step_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        "cosmos3_i2v_4step_lpips_golden_video.mp4",
        "Cosmos3 I2V-4Step LPIPS golden video",
    )

    video = _run_cosmos3_i2v_4step_lpips_pipeline(image_path)
    assert video is not None, "Cosmos3 I2V-4Step LPIPS run produced no video"
    _save_lpips_video_mp4(video, generated_path, frame_rate=COSMOS3_LPIPS_FRAME_RATE)

    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_i2v_4step",
        "video",
        COSMOS3_I2V_4STEP_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        COSMOS3_I2V_4STEP_LPIPS_THRESHOLD,
        generated_path,
        "cosmos3_i2v_4step_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, COSMOS3_I2V_4STEP_LPIPS_THRESHOLD)


def _write_cosmos3_edge_conditioning_image(path):
    """Deterministic 832x480 conditioning image (Edge's native 480p 16:9)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (832, 480))
    draw = ImageDraw.Draw(image)
    for y in range(480):
        draw.line([(0, y), (832, y)], fill=(30, 60 + y // 6, 140))
    draw.ellipse([320, 130, 520, 330], fill=(230, 120, 40), outline=(255, 255, 255), width=4)
    draw.rectangle([60, 330, 260, 450], fill=(40, 160, 90))
    image.save(path)


def test_cosmos3_edge_i2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the Edge checkpoint through the recommended invocation.

    Validates the documented deployment for ``Cosmos3-Edge``: the example
    script with a conditioning image and no config override (the Edge
    defaults — 480p x 121 frames, 50 UniPC steps on the native flow schedule,
    guidance 5.0, shift 3.0 — are the deployed shape). The run must produce a
    video.
    """
    model_path = _lpips_model_path("Cosmos3-Edge")
    _skip_if_missing(model_path, "Cosmos3-Edge checkpoint", is_dir=True)

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "cosmos3_edge_i2v_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, "conditioning.png")
    _write_cosmos3_edge_conditioning_image(image_path)
    output_path = os.path.join(out_dir, "cosmos3_edge_i2v_output.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--prompt",
            "The orange sphere slowly rises while the camera pans right across the scene",
            "--image_path",
            image_path,
            "--output_path",
            output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
    assert os.path.getsize(output_path) > 0, f"Example produced an empty video at {output_path}"


def _write_cosmos3_edge_policy_observation(path):
    """Deterministic 736x544 three-view observation in the DROID layout."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (736, 544), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 735, 271], fill=(75, 115, 170))
    draw.rectangle([0, 272, 367, 543], fill=(150, 90, 55))
    draw.rectangle([368, 272, 735, 543], fill=(55, 140, 90))
    draw.ellipse([305, 180, 430, 305], fill=(230, 190, 45))
    image.save(path)


def test_cosmos3_edge_policy_droid_example(_visual_gen_deps, llm_root, llm_venv):
    """Run the released Policy-DROID checkpoint through its documented API."""
    from safetensors.torch import load_file

    model_path = _lpips_model_path("Cosmos3-Edge-Policy-DROID")
    _skip_if_missing(model_path, "Cosmos3-Edge-Policy-DROID checkpoint", is_dir=True)

    out_dir = os.path.join(
        llm_venv.get_working_directory(),
        "visual_gen_output",
        "cosmos3_edge_policy_droid_example",
    )
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, "droid_observation.png")
    state_path = os.path.join(out_dir, "current_state.json")
    output_path = os.path.join(out_dir, "droid_policy.safetensors")
    action_output_path = os.path.join(out_dir, "droid_policy.action.json")
    _write_cosmos3_edge_policy_observation(image_path)
    with open(state_path, "w", encoding="utf-8") as state_file:
        json.dump([0.0] * 8, state_file)
    for stale_path in (output_path, action_output_path):
        if os.path.exists(stale_path):
            os.remove(stale_path)

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    prompt_path = os.path.join(
        llm_root,
        "examples",
        "visual_gen",
        "models",
        "cosmos3",
        "prompts",
        "action_edge_policy_droid.json",
    )
    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--prompt_file",
            prompt_path,
            "--image_path",
            image_path,
            "--action_json",
            state_path,
            "--output_path",
            output_path,
            "--action_output_path",
            action_output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )

    payload = load_file(output_path)
    assert payload["video"].shape == (33, 544, 736, 3)
    assert payload["action"].shape == (32, 8)
    assert payload["frame_rate"].item() == pytest.approx(15.0)
    with open(action_output_path, encoding="utf-8") as action_file:
        action_output = json.load(action_file)
    assert action_output["action_mode"] == "policy"
    assert action_output["shape"] == [32, 8]


# Edge LPIPS gates compare against diffusers-main reference goldens with the
# scheduler patched to the cosmos-framework native flow schedule; full
# provenance in golden/visual_gen_lpips/cosmos3_edge_*.json. The I2V gate runs
# 10 steps (cross-stack drift accumulates per step; the deployed 50-step shape
# is covered by test_cosmos3_edge_i2v_example).
COSMOS3_EDGE_LPIPS_SEED = 42
COSMOS3_EDGE_LPIPS_FRAME_RATE = 24.0
COSMOS3_EDGE_LPIPS_NUM_FRAMES = 29
COSMOS3_EDGE_T2V_LPIPS_PROMPT = "A red ball rolls across a wooden floor, casting a soft shadow."
COSMOS3_EDGE_T2V_LPIPS_STEPS = 50
COSMOS3_EDGE_T2V_LPIPS_THRESHOLD = 0.1
COSMOS3_EDGE_I2V_LPIPS_PROMPT = (
    "The orange sphere slowly rises while the camera pans right across the scene"
)
COSMOS3_EDGE_I2V_LPIPS_STEPS = 10
COSMOS3_EDGE_I2V_LPIPS_THRESHOLD = 0.13
COSMOS3_EDGE_T2I_LPIPS_PROMPT = (
    "A ceramic teapot pouring steaming tea into a cup, morning window light"
)
COSMOS3_EDGE_T2I_LPIPS_STEPS = 50
COSMOS3_EDGE_T2I_LPIPS_THRESHOLD = 0.05


def _run_cosmos3_edge_lpips_pipeline(**forward_kwargs):
    """Run the Cosmos3-Edge pipeline and return the PipelineOutput.

    VANILLA attention, compile-off; guardrails disabled for the run.
    """
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

        model_path = _lpips_model_path("Cosmos3-Edge")
        _skip_if_missing(model_path, "Cosmos3-Edge checkpoint", is_dir=True)
        _disable_inductor_compile_worker_quiesce()
        args = VisualGenArgs(
            model=model_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=False),
            attention_config=AttentionConfig(backend="VANILLA"),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            # The goldens were generated against an empty uncond branch, so pin it
            # here rather than inheriting the video-mode default negative prompt.
            forward_kwargs.setdefault("negative_prompt", "")
            with torch.no_grad(), _lpips_pinned_fp32_matmul_precision():
                result = pipeline.forward(
                    seed=COSMOS3_EDGE_LPIPS_SEED,
                    use_guardrails=False,
                    **forward_kwargs,
                )
            if result is not None:
                if result.video is not None:
                    result.video = result.video.detach().cpu()
                if result.image is not None:
                    result.image = result.image.detach().cpu()
            return result
        finally:
            del pipeline
            _cleanup_cuda()
    finally:
        if previous_guardrails_env is None:
            os.environ.pop(guardrails_env_key, None)
        else:
            os.environ[guardrails_env_key] = previous_guardrails_env


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_edge_t2v_lpips_against_golden(_visual_gen_deps, request, tmp_path):
    generated_path = tmp_path / "cosmos3_edge_t2v_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path, "cosmos3_edge_t2v_lpips_golden_video.mp4", "Cosmos3-Edge T2V LPIPS golden video"
    )
    result = _run_cosmos3_edge_lpips_pipeline(
        prompt=COSMOS3_EDGE_T2V_LPIPS_PROMPT,
        height=480,
        width=832,
        num_frames=COSMOS3_EDGE_LPIPS_NUM_FRAMES,
        num_inference_steps=COSMOS3_EDGE_T2V_LPIPS_STEPS,
        guidance_scale=5.0,
        frame_rate=COSMOS3_EDGE_LPIPS_FRAME_RATE,
    )
    assert result is not None and result.video is not None, "Edge T2V produced no video"
    _save_lpips_video_mp4(result.video, generated_path, frame_rate=COSMOS3_EDGE_LPIPS_FRAME_RATE)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_edge_t2v",
        "video",
        COSMOS3_EDGE_T2V_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        COSMOS3_EDGE_T2V_LPIPS_THRESHOLD,
        generated_path,
        "cosmos3_edge_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, COSMOS3_EDGE_T2V_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_edge_i2v_lpips_against_golden(_visual_gen_deps, request, tmp_path):
    generated_path = tmp_path / "cosmos3_edge_i2v_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path, "cosmos3_edge_i2v_lpips_golden_video.mp4", "Cosmos3-Edge I2V LPIPS golden video"
    )
    image_path = tmp_path / "cosmos3_edge_i2v_conditioning.png"
    _write_cosmos3_edge_conditioning_image(str(image_path))
    result = _run_cosmos3_edge_lpips_pipeline(
        prompt=COSMOS3_EDGE_I2V_LPIPS_PROMPT,
        image=str(image_path),
        height=480,
        width=832,
        num_frames=COSMOS3_EDGE_LPIPS_NUM_FRAMES,
        num_inference_steps=COSMOS3_EDGE_I2V_LPIPS_STEPS,
        guidance_scale=5.0,
        frame_rate=COSMOS3_EDGE_LPIPS_FRAME_RATE,
    )
    assert result is not None and result.video is not None, "Edge I2V produced no video"
    _save_lpips_video_mp4(result.video, generated_path, frame_rate=COSMOS3_EDGE_LPIPS_FRAME_RATE)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_edge_i2v",
        "video",
        COSMOS3_EDGE_I2V_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        COSMOS3_EDGE_I2V_LPIPS_THRESHOLD,
        generated_path,
        "cosmos3_edge_i2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, COSMOS3_EDGE_I2V_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_edge_t2i_lpips_against_golden(request, tmp_path):
    from tensorrt_llm.media.encoding import save_image

    generated_path = tmp_path / "cosmos3_edge_t2i_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "cosmos3_edge_t2i_lpips_golden.png", "Cosmos3-Edge T2I LPIPS golden image"
    )
    result = _run_cosmos3_edge_lpips_pipeline(
        prompt=COSMOS3_EDGE_T2I_LPIPS_PROMPT,
        height=640,
        width=640,
        num_inference_steps=COSMOS3_EDGE_T2I_LPIPS_STEPS,
        guidance_scale=4.0,
        output_type="image",
    )
    assert result is not None and result.image is not None, "Edge T2I produced no image"
    save_image(result.image[0], str(generated_path))
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_edge_t2i",
        "image",
        COSMOS3_EDGE_T2I_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        COSMOS3_EDGE_T2I_LPIPS_THRESHOLD,
        generated_path,
        "cosmos3_edge_t2i_lpips_golden.png",
    )
    _assert_lpips_below_threshold(score, COSMOS3_EDGE_T2I_LPIPS_THRESHOLD)
