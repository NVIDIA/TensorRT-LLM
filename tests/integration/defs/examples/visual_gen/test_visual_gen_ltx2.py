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

"""Single-GPU integration and accuracy tests for LTX-2."""

import os
from dataclasses import dataclass

import pytest
import torch
from defs import conftest
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
    _run_reusable_video_lpips_eval,
    _run_single_device_feature_generator,
    _save_lpips_video_mp4,
    _skip_if_missing,
    _validate_single_feature_config,
    _visual_gen_output_path,
)

LTX2_LPIPS_NUM_FRAMES = 49
LTX2_LPIPS_NUM_INFERENCE_STEPS = 8
LTX2_LPIPS_THRESHOLD = 0.15
LTX2_CUDA_GRAPH_LPIPS_THRESHOLD = 0.01

LTX2_FEATURE_LPIPS_THRESHOLD = 0.15
LTX2_SUPPORTED_FEATURES = frozenset({"fp8-blockwise", "nvfp4", "cuda-graph"})


@dataclass(frozen=True)
class LTX2AccuracyCase:
    id: str
    golden_file: str
    features: FeatureConfigState
    lpips_threshold: float


LTX2_FEATURE_PROFILES = (
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("cuda-graph", FeatureConfigState(cuda_graph=True)),
)


def _build_ltx2_accuracy_cases():
    cases = []
    for profile_id, features in LTX2_FEATURE_PROFILES:
        _validate_single_feature_config(
            features,
            LTX2_SUPPORTED_FEATURES,
            "LTX-2",
        )
        cases.append(
            pytest.param(
                LTX2AccuracyCase(
                    id=profile_id,
                    golden_file=(f"ltx2_{profile_id.replace('-', '_')}_lpips_golden_video.mp4"),
                    features=features,
                    lpips_threshold=LTX2_FEATURE_LPIPS_THRESHOLD,
                ),
                id=profile_id,
            )
        )
    return cases


LTX2_ACCURACY_CASES = _build_ltx2_accuracy_cases()


# LTX-2 configuration
LTX2_MODEL_CHECKPOINT_PATH = "LTX-2/ltx-2-19b-dev.safetensors"
LTX2_TEXT_ENCODER_SUBPATH = "gemma-3-12b-it"
LTX2_T2V_PROMPT = (
    "A woman with long brown hair and light skin smiles at the camera while "
    "standing in a sunlit park, her hair gently blowing in the breeze as she "
    "tilts her head slightly to the side."
)
LTX2_T2V_HEIGHT = 512
LTX2_T2V_WIDTH = 768
LTX2_T2V_NUM_FRAMES = 121
LTX2_T2V_STEPS = 40
LTX2_T2V_GUIDANCE_SCALE = 4.0
LTX2_T2V_MAX_SEQ_LEN = 1024
LTX2_T2V_FRAME_RATE = 24.0
LTX2_T2V_SEED = 42
LTX2_T2V_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


# LTX-2 Two-Stage configuration
LTX2_UPSAMPLER_SUBPATH = "LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_DISTILLED_LORA_SUBPATH = "LTX-2/ltx-2-19b-distilled-lora-384.safetensors"


def _ltx2_lpips_text_encoder_path():
    scratch_space = conftest.llm_models_root()
    candidates = [
        os.path.join(scratch_space, LTX2_TEXT_ENCODER_SUBPATH),
        os.path.join(scratch_space, "gemma", LTX2_TEXT_ENCODER_SUBPATH),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _generate_ltx2_feature_video(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    checkpoint_path = _lpips_model_path("LTX-2", "ltx-2-19b-dev.safetensors")
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    spatial_upsampler_path = _lpips_model_path("LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_path = _lpips_model_path("LTX-2", "ltx-2-19b-distilled-lora-384.safetensors")
    _skip_if_missing(checkpoint_path, "LTX-2 checkpoint")
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder", is_dir=True)
    _skip_if_missing(spatial_upsampler_path, "LTX-2 spatial upsampler")
    _skip_if_missing(distilled_lora_path, "LTX-2 distilled LoRA")
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    with (
        _lpips_deterministic_algorithms(),
        torch.compiler.set_stance("force_eager"),
        _fixed_nvfp4_quantization_backend(case.features),
    ):
        args = _build_single_device_feature_args(
            checkpoint_path,
            case.features,
            resolution=(LTX2_T2V_HEIGHT, LTX2_T2V_WIDTH),
            num_frames=LTX2_LPIPS_NUM_FRAMES,
            pipeline_config={
                "text_encoder_path": text_encoder_path,
                "spatial_upsampler_path": spatial_upsampler_path,
                "distilled_lora_path": distilled_lora_path,
            },
        )
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=False)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(LTX2_T2V_HEIGHT, LTX2_T2V_WIDTH),
                num_frames=LTX2_LPIPS_NUM_FRAMES,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=LTX2_T2V_PROMPT,
                negative_prompt=LTX2_T2V_NEGATIVE_PROMPT,
                height=LTX2_T2V_HEIGHT,
                width=LTX2_T2V_WIDTH,
                num_frames=LTX2_LPIPS_NUM_FRAMES,
                num_inference_steps=LTX2_LPIPS_NUM_INFERENCE_STEPS,
                guidance_scale=LTX2_T2V_GUIDANCE_SCALE,
                seed=LTX2_T2V_SEED,
            )
            assert result.video is not None, "LTX-2 feature run produced no video"
            _assert_single_device_feature_executed(pipeline, case.features)
            generated_video = result.video.detach().cpu()
        finally:
            try:
                if pipeline is not None:
                    _cleanup_single_device_feature_pipeline(pipeline)
                    del pipeline
            finally:
                _cleanup_cuda()

    _save_lpips_video_mp4(generated_video, output_path, frame_rate=LTX2_T2V_FRAME_RATE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", LTX2_ACCURACY_CASES)
def test_ltx2_feature_accuracy_against_golden(
    request,
    tmp_path,
    case,
    _visual_gen_deps,
    _visual_gen_lpips_scorer,
):
    generated_path = tmp_path / f"ltx2_{case.id}_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"LTX-2 {case.id} LPIPS golden video",
    )
    _run_single_device_feature_generator(
        case.features, _generate_ltx2_feature_video, case, generated_path
    )
    score = _run_reusable_video_lpips_eval(
        f"ltx2-{case.id}",
        golden_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"ltx2_{case.id}_generated.mp4",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)


def _generate_ltx2_lpips_video(output_path, *, enable_cuda_graph=False):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import CudaGraphConfig, TorchCompileConfig, VisualGenArgs

    checkpoint_path = _lpips_model_path("LTX-2", "ltx-2-19b-dev.safetensors")
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    spatial_upsampler_path = _lpips_model_path("LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_path = _lpips_model_path("LTX-2", "ltx-2-19b-distilled-lora-384.safetensors")
    _skip_if_missing(checkpoint_path, "LTX-2 checkpoint")
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder", is_dir=True)
    _skip_if_missing(spatial_upsampler_path, "LTX-2 spatial upsampler")
    _skip_if_missing(distilled_lora_path, "LTX-2 distilled LoRA")
    _disable_inductor_compile_worker_quiesce()

    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    # Wrapped here (not in the fixture) so the golden fixture and both sides of
    # test_ltx2_cuda_graph_lpips_matches_eager run the same eager numerics.
    with _lpips_deterministic_algorithms(), torch.compiler.set_stance("force_eager"):
        args = VisualGenArgs(
            model=checkpoint_path,
            pipeline_config={
                "text_encoder_path": text_encoder_path,
                "spatial_upsampler_path": spatial_upsampler_path,
                "distilled_lora_path": distilled_lora_path,
            },
            torch_compile_config=TorchCompileConfig(enable=False),
            cuda_graph_config=CudaGraphConfig(enable=enable_cuda_graph),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=LTX2_T2V_PROMPT,
                    negative_prompt=LTX2_T2V_NEGATIVE_PROMPT,
                    height=LTX2_T2V_HEIGHT,
                    width=LTX2_T2V_WIDTH,
                    num_frames=LTX2_LPIPS_NUM_FRAMES,
                    num_inference_steps=LTX2_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=LTX2_T2V_GUIDANCE_SCALE,
                    seed=LTX2_T2V_SEED,
                )
            generated_video = result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()

    _save_lpips_video_mp4(generated_video, output_path, frame_rate=LTX2_T2V_FRAME_RATE)


def _generate_ltx2_cuda_graph_trtllm_backend_video(output_path):
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        CompilationConfig,
        CudaGraphConfig,
        ParallelConfig,
        TorchCompileConfig,
    )

    scratch_space = conftest.llm_models_root()
    checkpoint_path = os.path.join(scratch_space, LTX2_MODEL_CHECKPOINT_PATH)
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    spatial_upsampler_path = os.path.join(scratch_space, LTX2_UPSAMPLER_SUBPATH)
    distilled_lora_path = os.path.join(scratch_space, LTX2_DISTILLED_LORA_SUBPATH)
    _skip_if_missing(checkpoint_path, "LTX-2 checkpoint")
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder", is_dir=True)
    _skip_if_missing(spatial_upsampler_path, "LTX-2 spatial upsampler")
    _skip_if_missing(distilled_lora_path, "LTX-2 distilled LoRA")
    _disable_inductor_compile_worker_quiesce()

    visual_gen_args = VisualGenArgs(
        model=checkpoint_path,
        quant_config={"quant_algo": "NVFP4", "dynamic": True},
        attention_config=AttentionConfig(backend="TRTLLM"),
        parallel_config=ParallelConfig(
            cfg_size=1,
            ulysses_size=1,
            parallel_vae_size=1,
        ),
        compilation_config=CompilationConfig(
            resolutions=[
                (
                    LTX2_T2V_HEIGHT,
                    LTX2_T2V_WIDTH,
                )
            ],
            num_frames=[LTX2_LPIPS_NUM_FRAMES],
        ),
        cuda_graph_config=CudaGraphConfig(enable=True),
        torch_compile_config=TorchCompileConfig(
            enable=True,
            enable_fullgraph=False,
            enable_autotune=True,
        ),
        pipeline_config={
            "text_encoder_path": text_encoder_path,
            "spatial_upsampler_path": spatial_upsampler_path,
            "distilled_lora_path": distilled_lora_path,
        },
    )

    visual_gen = VisualGen(model=checkpoint_path, args=visual_gen_args)
    try:
        params = VisualGenParams(
            height=LTX2_T2V_HEIGHT,
            width=LTX2_T2V_WIDTH,
            num_frames=LTX2_LPIPS_NUM_FRAMES,
            num_inference_steps=LTX2_LPIPS_NUM_INFERENCE_STEPS,
            guidance_scale=LTX2_T2V_GUIDANCE_SCALE,
            max_sequence_length=LTX2_T2V_MAX_SEQ_LEN,
            seed=LTX2_T2V_SEED,
            frame_rate=LTX2_T2V_FRAME_RATE,
            negative_prompt=LTX2_T2V_NEGATIVE_PROMPT,
        )
        output = visual_gen.generate(inputs=LTX2_T2V_PROMPT, params=params)
        _save_lpips_video_mp4(output.video, output_path, frame_rate=LTX2_T2V_FRAME_RATE)
    finally:
        visual_gen.shutdown()
        del visual_gen
        _cleanup_cuda()

    assert os.path.isfile(output_path), f"LTX-2 TRTLLM backend did not produce {output_path}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx2_lpips_against_golden(request, tmp_path, ltx2_two_stage_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "ltx2_lpips_golden_video.mp4", "LTX-2 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "ltx2",
        "video",
        LTX2_T2V_PROMPT,
        golden_path,
        ltx2_two_stage_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        LTX2_LPIPS_THRESHOLD,
        ltx2_two_stage_bf16_video_path,
        "ltx2_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, LTX2_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx2_cuda_graph_lpips_matches_eager(_visual_gen_deps, tmp_path):
    eager_path = tmp_path / "ltx2_eager_generated.mp4"
    cuda_graph_path = tmp_path / "ltx2_cuda_graph_generated.mp4"

    _generate_ltx2_lpips_video(eager_path, enable_cuda_graph=False)
    _generate_ltx2_lpips_video(cuda_graph_path, enable_cuda_graph=True)
    score = _run_lpips_eval(
        tmp_path,
        "ltx2_cuda_graph",
        "video",
        LTX2_T2V_PROMPT,
        eager_path,
        cuda_graph_path,
    )
    _assert_lpips_below_threshold(score, LTX2_CUDA_GRAPH_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx2_cuda_graph_trtllm_backend(request, _visual_gen_deps, tmp_path):
    generated_path = tmp_path / "ltx2_cuda_graph_trtllm_backend_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path, "ltx2_lpips_golden_video.mp4", "LTX-2 LPIPS golden video"
    )
    _generate_ltx2_cuda_graph_trtllm_backend_video(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "ltx2_cuda_graph_trtllm_backend",
        "video",
        LTX2_T2V_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        LTX2_LPIPS_THRESHOLD,
        generated_path,
        "ltx2_cuda_graph_trtllm_backend_generated.mp4",
    )
    _assert_lpips_below_threshold(score, LTX2_LPIPS_THRESHOLD)


@pytest.fixture(scope="session")
def ltx2_two_stage_bf16_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 two-stage BF16 video with the LPIPS config and return path."""
    output_path = _visual_gen_output_path(llm_venv, "ltx2_two_stage_bf16")
    if os.path.isfile(output_path):
        return output_path
    _generate_ltx2_lpips_video(output_path)
    return output_path


def test_ltx2_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/ltx2.py with NVFP4 config end-to-end.

    Validates that the LTX-2 example script and ``configs/ltx2-fp4-1gpu.yaml``
    work together as documented. The Gemma3 text encoder is passed separately via
    ``--text_encoder_path`` because the shared YAML intentionally omits it to keep
    the config model-path-agnostic.
    """
    model_path = _lpips_model_path("LTX-2", "ltx-2-19b-dev.safetensors")
    _skip_if_missing(model_path, "LTX-2 checkpoint")
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder (gemma-3-12b-it)", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "ltx2_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "ltx2_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "ltx2.py")
    config_path = os.path.join(llm_root, "examples", "visual_gen", "configs", "ltx2-fp4-1gpu.yaml")
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
            "--text_encoder_path",
            text_encoder_path,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
