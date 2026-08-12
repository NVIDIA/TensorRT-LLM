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

"""Single-GPU integration and accuracy tests for MiniMax-H3.

MiniMax-H3 is a guidance-distilled omni-modal text-to-video+audio model. Its
transformer (~62 GB bf16) is quantized to NVFP4 on the fly at load time for
the accuracy test, so the FP4 and bf16 runs can be compared on a single
96 GB accelerator; the Qwen3-VL conditioner is swapped in per request
(``conditioner_offload=auto``) in both.

The accuracy gate compares NVFP4 against the bf16 run of the same request
(no registry golden required) with LPIPS below an explicit threshold.
"""

import os

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    VISUAL_GEN_OUTPUT_VIDEO,
    _assert_lpips_below_threshold,
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _lpips_deterministic_algorithms,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _save_lpips_video_mp4,
    _skip_if_missing,
)

# MiniMax-H3 request configuration. The released checkpoint generates at
# 24 fps for 5-15 s; 124 frames is the minimum duration (5 s, num_frames
# congruent to 5 mod 17 for the video VAE) at a reduced canvas.
H3_PROMPT = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
H3_HEIGHT = 384
H3_WIDTH = 640
H3_NUM_FRAMES = 124
H3_STEPS = 10
H3_SEED = 0
H3_FRAME_RATE = 24.0

H3_NVFP4_VS_BF16_LPIPS_THRESHOLD = 0.10


def _h3_checkpoint_path():
    """Resolve the MiniMax-H3 checkpoint under the shared model registry."""
    scratch_space = conftest.llm_models_root()
    candidates = [
        os.path.join(scratch_space, "MiniMax-H3"),
        os.path.join(scratch_space, "MiniMaxAI", "MiniMax-H3"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _generate_h3_video(output_path, *, nvfp4: bool):
    """Generate a MiniMax-H3 video+audio sample through the public API."""
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm.visual_gen.args import TorchCompileConfig

    checkpoint_path = _h3_checkpoint_path()
    _skip_if_missing(checkpoint_path, "MiniMax-H3 checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()

    quant_config = {"quant_algo": "NVFP4"} if nvfp4 else None
    with _lpips_deterministic_algorithms(), torch.compiler.set_stance("force_eager"):
        visual_gen = VisualGen(
            model=checkpoint_path,
            args=VisualGenArgs(
                quant_config=quant_config,
                torch_compile_config=TorchCompileConfig(enable=False),
            ),
        )
        try:
            params = VisualGenParams(
                height=H3_HEIGHT,
                width=H3_WIDTH,
                num_frames=H3_NUM_FRAMES,
                num_inference_steps=H3_STEPS,
                seed=H3_SEED,
            )
            output = visual_gen.generate(inputs=H3_PROMPT, params=params)
            assert output.error is None, f"MiniMax-H3 generation failed: {output.error}"
            assert output.video is not None, "MiniMax-H3 produced no video"
            assert output.audio is not None, "MiniMax-H3 produced no audio"
            _save_lpips_video_mp4(output.video, output_path, frame_rate=H3_FRAME_RATE)
        finally:
            visual_gen.shutdown()
            del visual_gen
            _cleanup_cuda()

    assert os.path.isfile(output_path), f"MiniMax-H3 did not produce {output_path}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_t2va_nvfp4_matches_bf16(request, tmp_path, _visual_gen_deps):
    """NVFP4 online quantization must stay close to the bf16 reference.

    Lossy-work gate for the dynamic-quantization path: both runs use the same
    request, so no registry golden is required. Needs >= 96 GB of accelerator
    memory for the bf16 reference run (the conditioner is offloaded in both).
    """
    bf16_path = tmp_path / "minimax_h3_bf16.mp4"
    nvfp4_path = tmp_path / "minimax_h3_nvfp4.mp4"

    _generate_h3_video(bf16_path, nvfp4=False)
    _generate_h3_video(nvfp4_path, nvfp4=True)

    score = _run_lpips_eval(
        tmp_path,
        "minimax_h3_nvfp4_vs_bf16",
        "video",
        H3_PROMPT,
        bf16_path,
        nvfp4_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        H3_NVFP4_VS_BF16_LPIPS_THRESHOLD,
        nvfp4_path,
        "minimax_h3_nvfp4_vs_bf16_generated.mp4",
    )
    _assert_lpips_below_threshold(score, H3_NVFP4_VS_BF16_LPIPS_THRESHOLD)


def test_minimax_h3_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/minimax_h3.py with --fp4 end-to-end.

    Exercises the example's NVFP4 flag, which shrinks the transformer to
    about 35 GiB so the whole pipeline fits on a single 96 GiB accelerator.
    """
    checkpoint_path = _h3_checkpoint_path()
    _skip_if_missing(checkpoint_path, "MiniMax-H3 checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "minimax_h3")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "minimax_h3.py")
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            checkpoint_path,
            "--fp4",
            "--num_frames",
            str(H3_NUM_FRAMES),
            "--num_inference_steps",
            str(H3_STEPS),
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
