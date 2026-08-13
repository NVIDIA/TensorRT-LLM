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

"""Single-GPU integration and accuracy tests for HunyuanVideo 1.5."""

import os

import pytest
import torch
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    _assert_lpips_below_threshold,
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _save_lpips_video_mp4,
    _skip_if_missing,
)

HUNYUAN_T2V_MODEL_SUBPATH = "HunyuanVideo-1.5-Diffusers-480p_t2v"

HUNYUAN_LPIPS_PROMPT = "A cat sitting on a windowsill"
HUNYUAN_LPIPS_NEGATIVE_PROMPT = ""
# 256x256 is a multiple of resolution_multiple_of (32 = vae_scale_factor_spatial
# * patch_size) and 9 frames = 3 latent frames at temporal compression 4, so
# nothing is padded.
HUNYUAN_LPIPS_HEIGHT = 256
HUNYUAN_LPIPS_WIDTH = 256
HUNYUAN_LPIPS_NUM_FRAMES = 9
HUNYUAN_LPIPS_NUM_INFERENCE_STEPS = 4
HUNYUAN_LPIPS_SEED = 42
# Golden is encoded at the pipeline's default frame rate.
HUNYUAN_LPIPS_FRAME_RATE = 24.0
HUNYUAN_LPIPS_THRESHOLD = 0.05


def _run_hunyuan_lpips_pipeline():
    """Run HunyuanVideo 1.5 T2V at the reduced LPIPS setting (VANILLA attn, compile-off).

    Returns the generated video tensor ``(B, T, H, W, C)``, or ``None`` if
    generation produced no video. Guidance is owned by the pipeline's guider
    component, so ``forward()`` takes no ``guidance_scale``.
    """
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        CompilationConfig,
        TorchCompileConfig,
        VisualGenArgs,
    )

    model_path = _lpips_model_path(HUNYUAN_T2V_MODEL_SUBPATH)
    _skip_if_missing(model_path, "HunyuanVideo 1.5 480p T2V checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    with _lpips_deterministic_algorithms():
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
                    prompt=HUNYUAN_LPIPS_PROMPT,
                    negative_prompt=HUNYUAN_LPIPS_NEGATIVE_PROMPT,
                    seed=HUNYUAN_LPIPS_SEED,
                    height=HUNYUAN_LPIPS_HEIGHT,
                    width=HUNYUAN_LPIPS_WIDTH,
                    num_frames=HUNYUAN_LPIPS_NUM_FRAMES,
                    num_inference_steps=HUNYUAN_LPIPS_NUM_INFERENCE_STEPS,
                )
            if result is None or result.video is None:
                return None
            return result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()


def _generate_hunyuan_lpips_video(output_path):
    """Generate the HunyuanVideo 1.5 text-to-video LPIPS sample."""
    video = _run_hunyuan_lpips_pipeline()
    assert video is not None, "HunyuanVideo 1.5 T2V LPIPS run produced no video"
    _save_lpips_video_mp4(video, output_path, frame_rate=HUNYUAN_LPIPS_FRAME_RATE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hunyuan_t2v_lpips_against_golden(request, _visual_gen_deps, tmp_path):
    generated_path = tmp_path / "hunyuan_t2v_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        "hunyuan_t2v_lpips_golden_video.mp4",
        "HunyuanVideo 1.5 T2V LPIPS golden video",
    )
    _generate_hunyuan_lpips_video(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "hunyuan_t2v",
        "video",
        HUNYUAN_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        HUNYUAN_LPIPS_THRESHOLD,
        generated_path,
        "hunyuan_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, HUNYUAN_LPIPS_THRESHOLD)


def test_hunyuan_t2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/hunyuan_t2v.py with the FP8 config end-to-end.

    Validates that the HunyuanVideo 1.5 example script and
    ``configs/hunyuan-t2v-fp8-1gpu.yaml`` work together as documented in the
    README, at the example's own 480p defaults.
    """
    model_path = _lpips_model_path(HUNYUAN_T2V_MODEL_SUBPATH)
    _skip_if_missing(model_path, "HunyuanVideo 1.5 480p T2V checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "hunyuan_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "hunyuan_t2v_output.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "hunyuan_t2v.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "hunyuan-t2v-fp8-1gpu.yaml"
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
