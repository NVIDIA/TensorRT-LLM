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
"""End-to-end golden-media tests for LTX-2.3.

Video LPIPS and audio log-mel L1 against goldens. Override paths with
LTX23_MODEL_PATH, LTX23_TEXT_ENCODER_PATH, and LTX23_GOLDEN_DIR.
"""

import os
import subprocess
import sys

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from defs.examples.visual_gen.visual_gen_test_utils import (
    REPO_ROOT,
    _assert_lpips_below_threshold,
    _cleanup_cuda,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _run_lpips_eval,
    _save_lpips_video_mp4,
    _skip_if_missing,
)

LTX23_MODEL_PATH = os.environ.get("LTX23_MODEL_PATH", _lpips_model_path("LTX-2.3"))
LTX23_TEXT_ENCODER_PATH = os.environ.get(
    "LTX23_TEXT_ENCODER_PATH", _lpips_model_path("gemma", "gemma-3-12b-it")
)

LTX23_GOLDEN_DIR = os.environ.get(
    "LTX23_GOLDEN_DIR",
    os.path.join(os.path.dirname(__file__), "golden", "visual_gen_lpips"),
)
LTX23_GOLDEN_VIDEO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_lpips_golden_video.mp4")
LTX23_GOLDEN_AUDIO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_audio_golden.wav")

LTX23_LPIPS_PROMPT = (
    "A cinematic close-up of a golden retriever puppy running through a sunlit "
    "meadow of wildflowers, gentle breeze, birds chirping"
)
LTX23_LPIPS_HEIGHT = 512
LTX23_LPIPS_WIDTH = 768
LTX23_LPIPS_NUM_FRAMES = 49
LTX23_LPIPS_NUM_INFERENCE_STEPS = 8
LTX23_LPIPS_GUIDANCE_SCALE = 5.0
LTX23_LPIPS_FRAME_RATE = 24.0
LTX23_LPIPS_SEED = 42

LTX23_LPIPS_THRESHOLD = 0.05
LTX23_AUDIO_MEL_L1_THRESHOLD = float(os.environ.get("LTX23_AUDIO_MEL_L1_THRESHOLD", "0.45"))


def _save_wav(audio, path, sample_rate):
    tensor = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    mono = librosa.to_mono(tensor.reshape(-1, tensor.shape[-1]).numpy())
    sf.write(str(path), np.clip(mono, -1.0, 1.0), int(sample_rate))
    assert os.path.isfile(path), f"Failed to write WAV {path}"


def _audio_mel_l1_distance(generated_path, reference_path):
    """Mean absolute log-mel dB difference between two WAVs."""
    generated, gen_rate = librosa.load(str(generated_path), sr=None, mono=True)
    reference, ref_rate = librosa.load(str(reference_path), sr=None, mono=True)
    if gen_rate != ref_rate:
        raise ValueError(f"Sample-rate mismatch: generated {gen_rate} vs golden {ref_rate}")

    def _log_mel(wav, rate):
        mel = librosa.feature.melspectrogram(
            y=wav, sr=rate, n_fft=1024, hop_length=256, n_mels=64, htk=True, norm=None
        )
        return librosa.power_to_db(mel, top_db=80.0)

    gen_mel, ref_mel = _log_mel(generated, gen_rate), _log_mel(reference, ref_rate)
    if gen_mel.shape[1] != ref_mel.shape[1]:
        raise ValueError(
            f"Audio length mismatch: generated {gen_mel.shape[1]} mel frames vs "
            f"golden {ref_mel.shape[1]}"
        )
    if gen_mel.shape[1] == 0:
        raise ValueError("Empty audio: both clips decoded to zero frames")
    distance = float(np.abs(gen_mel - ref_mel).mean())
    print(f"[E2E audio log-mel L1] distance: {distance:.6f} dB (frames: {gen_mel.shape[1]})")
    return distance


def _generate_ltx23_av(video_out_path, audio_out_path):
    """Render the LPIPS clip deterministically into an MP4 plus a WAV."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True)

    # Force eager: nested @torch.compile is not turned off by TorchCompileConfig.
    with _lpips_deterministic_algorithms(fully_eager=True):
        args = VisualGenArgs(
            model=LTX23_MODEL_PATH,
            pipeline_config={"text_encoder_path": LTX23_TEXT_ENCODER_PATH},
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                out = pipeline.forward(
                    prompt=LTX23_LPIPS_PROMPT,
                    seed=LTX23_LPIPS_SEED,
                    height=LTX23_LPIPS_HEIGHT,
                    width=LTX23_LPIPS_WIDTH,
                    num_frames=LTX23_LPIPS_NUM_FRAMES,
                    frame_rate=LTX23_LPIPS_FRAME_RATE,
                    num_inference_steps=LTX23_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=LTX23_LPIPS_GUIDANCE_SCALE,
                )
            video = out.video.detach().cpu()
            audio = out.audio.detach().cpu() if out.audio is not None else None
            audio_rate = out.audio_sample_rate
        finally:
            del pipeline
            _cleanup_cuda()

    _save_lpips_video_mp4(video, video_out_path, frame_rate=LTX23_LPIPS_FRAME_RATE)
    assert audio is not None, "LTX-2.3 render returned no audio"
    _save_wav(audio, audio_out_path, audio_rate)


@pytest.fixture(scope="session")
def ltx23_av_candidate(tmp_path_factory):
    """Render the candidate clip once per session; return the (video, audio) paths."""
    out_dir = str(tmp_path_factory.mktemp("ltx23_av"))
    video_path = os.path.join(out_dir, "ltx23_candidate_video.mp4")
    audio_path = os.path.join(out_dir, "ltx23_candidate_audio.wav")
    _generate_ltx23_av(video_path, audio_path)
    return video_path, audio_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_video_lpips_against_golden(tmp_path, ltx23_av_candidate):
    _skip_if_missing(LTX23_GOLDEN_VIDEO, "LTX-2.3 LPIPS golden video")
    video_path, _ = ltx23_av_candidate
    score = _run_lpips_eval(
        tmp_path, "ltx23", "video", LTX23_LPIPS_PROMPT, LTX23_GOLDEN_VIDEO, video_path
    )
    _assert_lpips_below_threshold(score, LTX23_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_audio_against_golden(ltx23_av_candidate):
    """The only gate on the vocoder and BWE output; LPIPS cannot see audio."""
    _skip_if_missing(LTX23_GOLDEN_AUDIO, "LTX-2.3 golden audio")
    _, audio_path = ltx23_av_candidate
    distance = _audio_mel_l1_distance(audio_path, LTX23_GOLDEN_AUDIO)
    assert distance < LTX23_AUDIO_MEL_L1_THRESHOLD, (
        f"LTX-2.3 audio log-mel L1 too high: {distance:.6f} "
        f"(expected < {LTX23_AUDIO_MEL_L1_THRESHOLD:.6f})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "config_name", ["ltx23-t2v-bf16-1gpu.yaml", "ltx23-t2v-fp8-1gpu.yaml"], ids=["bf16", "fp8"]
)
def test_ltx23_example(tmp_path, config_name):
    """Each shipped config renders an MP4 through the --model_type ltx23 driver."""
    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True)

    examples_root = os.path.join(REPO_ROOT, "examples", "visual_gen")
    output_path = os.path.join(str(tmp_path), "ltx23_output.mp4")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(examples_root, "models", "ltx2.py"),
            "--model_type",
            "ltx23",
            "--model",
            LTX23_MODEL_PATH,
            "--visual_gen_args",
            os.path.join(examples_root, "configs", config_name),
            "--text_encoder_path",
            LTX23_TEXT_ENCODER_PATH,
            "--output_path",
            output_path,
        ],
        check=False,
    )
    assert result.returncode == 0, "LTX-2.3 example script exited non-zero"
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
