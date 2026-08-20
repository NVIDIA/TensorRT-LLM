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
"""End-to-end golden-media tests for LTX-2.3, the counterpart to the
test_ltx2_lpips_against_golden family in test_visual_gen.py.

LTX-2.3 emits audio and video, so quality is gated on two axes: decoded pixels
against a golden MP4 via the shared LPIPS eval script, and the audio waveform
against a golden WAV via a librosa log-mel L1 distance. Generation is
deterministic (fixed seed, deterministic algorithms, forced eager) so a matching
kernel stack reproduces the goldens within the threshold margins.

Paths resolve through the CI harness and are overridable with LTX23_MODEL_PATH,
LTX23_TEXT_ENCODER_PATH and LTX23_GOLDEN_DIR.
"""

import contextlib
import gc
import json
import os
import subprocess
import sys

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from defs.conftest import llm_models_root

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
VISUAL_GEN_LPIPS_EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "visualgen_eval", "visual_gen_lpips_score_eval.py"
)

_MODELS_ROOT = str(llm_models_root())
LTX23_MODEL_PATH = os.environ.get(
    "LTX23_MODEL_PATH", os.path.join(_MODELS_ROOT, "LTX-2.3")
)
LTX23_TEXT_ENCODER_PATH = os.environ.get(
    "LTX23_TEXT_ENCODER_PATH", os.path.join(_MODELS_ROOT, "gemma", "gemma-3-12b-it")
)

LTX23_GOLDEN_DIR = os.environ.get(
    "LTX23_GOLDEN_DIR",
    os.path.join(os.path.dirname(__file__), "golden", "visual_gen_lpips"),
)
LTX23_GOLDEN_VIDEO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_lpips_golden_video.mp4")
LTX23_GOLDEN_AUDIO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_audio_golden.wav")

# Reduced render, mirroring LTX-2's LPIPS setting: cheap to reproduce while still
# exercising the full A/V decode plus vocoder and BWE path.
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

# The video gate matches every other visual_gen model and absorbs the ~0.04
# cross-host kernel drift. The audio gate is mean absolute log-mel difference in
# dB; deterministic same-host reruns land near zero.
LTX23_LPIPS_THRESHOLD = 0.05
LTX23_AUDIO_MEL_L1_THRESHOLD = float(
    os.environ.get("LTX23_AUDIO_MEL_L1_THRESHOLD", "0.45")
)


def _skip_if_missing(path, label, is_dir=False):
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    if not exists:
        pytest.skip(f"{label} not found: {path}")


@contextlib.contextmanager
def _lpips_deterministic_algorithms():
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        if prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas


def _save_lpips_video_mp4(video, output_path, frame_rate):
    """Encode with H.264. A codec fallback would make LPIPS measure artifacts."""
    from tensorrt_llm.media.encoding import save_video

    try:
        save_video(video, output_path, frame_rate=frame_rate)
    except RuntimeError as err:
        if "MP4 format requires ffmpeg" not in str(err):
            raise
        raise RuntimeError(
            f"ffmpeg is unavailable for LPIPS video encoding: {err}"
        ) from err
    assert os.path.isfile(output_path), f"LTX-2.3 did not produce video {output_path}"


def _run_lpips_eval(tmp_dir, sample_id, prompt, reference_path, generated_path):
    """Score generated against golden video with the shared LPIPS eval script."""
    dataset_path = os.path.join(tmp_dir, f"{sample_id}_dataset.json")
    output_json = os.path.join(tmp_dir, f"{sample_id}_lpips_results.json")
    with open(dataset_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "samples": [
                    {
                        "id": sample_id,
                        "media_type": "video",
                        "prompt": prompt,
                        "reference_video_path": str(reference_path),
                        "generated_video_path": str(generated_path),
                    }
                ]
            },
            fh,
            indent=2,
        )

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else REPO_ROOT
    )
    with _lpips_deterministic_algorithms():
        result = subprocess.run(
            [
                sys.executable,
                VISUAL_GEN_LPIPS_EVAL_SCRIPT,
                "--dataset",
                str(dataset_path),
                "--output-json",
                str(output_json),
                "--json",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"LPIPS eval script failed for {sample_id}:\n{result.stdout}")

    with open(output_json, encoding="utf-8") as fh:
        score = float(json.load(fh)["mean_lpips_score"])
    print(f"\n[E2E {sample_id} video LPIPS] score: {score:.6f}")
    return score


def _save_wav(audio, path, sample_rate):
    tensor = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    mono = librosa.to_mono(tensor.reshape(-1, tensor.shape[-1]).numpy())
    sf.write(str(path), np.clip(mono, -1.0, 1.0), int(sample_rate))
    assert os.path.isfile(path), f"Failed to write WAV {path}"


def _audio_mel_l1_distance(generated_path, reference_path):
    """Mean absolute log-mel dB difference between two WAVs.

    power_to_db floors each clip at top_db below its peak, so PCM noise in
    near-silent bins cannot dominate the distance.
    """
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
    n = min(gen_mel.shape[1], ref_mel.shape[1])
    if n == 0:
        raise ValueError("Empty audio: one of the clips decoded to zero frames")
    distance = float(np.abs(gen_mel[:, :n] - ref_mel[:, :n]).mean())
    print(f"[E2E audio log-mel L1] distance: {distance:.6f} dB (frames compared: {n})")
    return distance


def _generate_ltx23_av(video_out_path, audio_out_path):
    """Render the LPIPS clip deterministically into an MP4 plus a WAV."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(
        LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True
    )

    # Nested @torch.compile decorators are not suppressed by
    # TorchCompileConfig(enable=False) alone, so force eager over the whole render.
    with _lpips_deterministic_algorithms(), torch.compiler.set_stance("force_eager"):
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
            gc.collect()
            torch.cuda.empty_cache()

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
        str(tmp_path), "ltx23", LTX23_LPIPS_PROMPT, LTX23_GOLDEN_VIDEO, video_path
    )
    assert score < LTX23_LPIPS_THRESHOLD, (
        f"LTX-2.3 video LPIPS too high: {score:.6f} "
        f"(expected < {LTX23_LPIPS_THRESHOLD:.6f})"
    )


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
def test_ltx23_example(tmp_path):
    """The LTX-2 example driver in --model_type ltx23 mode produces an MP4."""
    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(
        LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True
    )

    examples_root = os.path.join(REPO_ROOT, "examples", "visual_gen")
    output_path = os.path.join(str(tmp_path), "ltx23_output.mp4")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(examples_root, "models", "ltx2.py"),
            "--model_type", "ltx23",
            "--model", LTX23_MODEL_PATH,
            "--visual_gen_args",
            os.path.join(examples_root, "configs", "ltx23-t2v-bf16-1gpu.yaml"),
            "--text_encoder_path", LTX23_TEXT_ENCODER_PATH,
            "--output_path", output_path,
        ],
        check=False,
    )
    assert result.returncode == 0, "LTX-2.3 example script exited non-zero"
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
