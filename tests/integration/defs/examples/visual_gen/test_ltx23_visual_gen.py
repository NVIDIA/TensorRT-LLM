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
"""End-to-end LPIPS/audio-faithfulness integration tests for LTX-2.3.

This is the LTX-2.3 counterpart to the LTX-2 blocks in the upstream
``tests/integration/defs/examples/visual_gen/test_visual_gen.py`` (the
``test_ltx2_lpips_against_golden`` family). It closes the last untested layer:
the unit tests confirm wiring + numerics, but not that a *full render* looks and
sounds right against a frozen golden.

LTX-2.3 emits **audio + video**, so quality is checked on two axes:

- **Video** — decoded pixels vs a golden MP4 via the shared LPIPS eval script
  (``scripts/visualgen_eval/visual_gen_lpips_score_eval.py``), exactly like every
  other visual_gen model. LPIPS is a *visual-only* metric.
- **Audio** — LPIPS says nothing about the soundtrack, so the audio waveform is
  compared to a golden WAV with a self-contained **log-mel L1** distance. This is
  the standard perceptual proxy for "does it sound like the reference" and needs
  no extra model download (unlike LPIPS's AlexNet weights).

Both goldens are frozen media, mirroring upstream's golden-media zip. Generation
is deterministic (fixed seed + ``torch.use_deterministic_algorithms`` + forced
eager), so a matching kernel stack reproduces the golden to within the
cross-host drift margin baked into the thresholds.

Running outside the CI harness (e.g. inside a plain container on a GPU node):

    # 1. Freeze a golden from the current (known-good) build:
    LTX23_MODEL_PATH=/path/LTX-2.3 LTX23_TEXT_ENCODER_PATH=/path/gemma-3-12b-it \\
        LTX23_GOLDEN_DIR=/raid/ltx23_golden \\
        python test_ltx23_visual_gen.py --make-golden

    # 2. Score a fresh render against that golden (video LPIPS + audio mel-L1):
    LTX23_MODEL_PATH=... LTX23_TEXT_ENCODER_PATH=... LTX23_GOLDEN_DIR=/raid/ltx23_golden \\
        python test_ltx23_visual_gen.py --score

The pytest entry points (``test_ltx23_*``) use the same helpers but resolve
paths + goldens through the CI harness (``llm_models_root`` / the golden zip).
"""

import contextlib
import json
import math
import os
import subprocess
import sys
import wave

import pytest
import torch

# ---------------------------------------------------------------------------
# Path + golden resolution.
#
# In CI this file lives at tests/integration/defs/examples/visual_gen/, so the
# repo root is five levels up and models come from ``llm_models_root()``. Outside
# CI (standalone container run) those are absent, so every path is overridable by
# env var -- matching the convention already used by test_ltx23_pipeline.py.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
VISUAL_GEN_LPIPS_EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "visualgen_eval", "visual_gen_lpips_score_eval.py"
)


def _llm_models_root():
    """Best-effort models root: CI harness first, then env var, else ''."""
    try:  # pragma: no cover - CI-only path
        from defs import conftest

        return str(conftest.llm_models_root())
    except Exception:
        return os.environ.get("LLM_MODELS_ROOT", "")


_MODELS_ROOT = _llm_models_root()
_LTX23_BASE = os.path.join(_MODELS_ROOT, "LTX-2.3") if _MODELS_ROOT else ""
_GEMMA3_DEFAULT = os.path.join(_MODELS_ROOT, "gemma", "gemma-3-12b-it") if _MODELS_ROOT else ""

LTX23_MODEL_PATH = os.environ.get("LTX23_MODEL_PATH", _LTX23_BASE)
LTX23_TEXT_ENCODER_PATH = os.environ.get("LTX23_TEXT_ENCODER_PATH", _GEMMA3_DEFAULT)

# Golden media. In CI these are shipped in the golden dir alongside the other
# visual_gen goldens; LTX23_GOLDEN_DIR overrides for a standalone node run.
LTX23_GOLDEN_DIR = os.environ.get(
    "LTX23_GOLDEN_DIR",
    os.path.join(os.path.dirname(__file__), "golden", "visual_gen_lpips"),
)
LTX23_GOLDEN_VIDEO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_lpips_golden_video.mp4")
LTX23_GOLDEN_AUDIO = os.path.join(LTX23_GOLDEN_DIR, "ltx23_audio_golden.wav")

# ---------------------------------------------------------------------------
# Deterministic LPIPS render config (small on purpose -- mirrors LTX-2's reduced
# 49-frame / 8-step LPIPS setting so the golden is cheap to reproduce while still
# exercising the full A/V decode + vocoder + BWE path).
# ---------------------------------------------------------------------------
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

# Thresholds. The video LPIPS gate matches every other visual_gen model (0.05,
# which absorbs the ~0.04 cross-B200-host kernel drift upstream measured). The
# audio log-mel-L1 gate is calibrated from the first golden run on the target
# stack; deterministic same-host reruns land near 0 and the margin covers
# cross-host drift. Override with LTX23_AUDIO_MEL_L1_THRESHOLD after calibration.
LTX23_LPIPS_THRESHOLD = 0.05
LTX23_AUDIO_MEL_L1_THRESHOLD = float(os.environ.get("LTX23_AUDIO_MEL_L1_THRESHOLD", "0.10"))


# ---------------------------------------------------------------------------
# Small helpers (mirrors of the upstream test_visual_gen.py utilities).
# ---------------------------------------------------------------------------
def _skip_if_missing(path, label, is_dir=False):
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    if not exists:
        pytest.skip(f"{label} not found: {path}")


@contextlib.contextmanager
def _lpips_deterministic_algorithms():
    """Deterministic numerics for reproducible golden comparison.

    Same recipe as upstream: pin the cuBLAS workspace and enable deterministic
    algorithms so a fresh render reproduces the golden on a matching stack.
    """
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
    """Encode with H.264 (never a silent codec fallback).

    LPIPS compares decoded pixels; a cv2/mp4v fallback would measure codec
    artifacts instead of model output, so refuse rather than mask the failure
    (this is the exact footgun that produced a spurious LPIPS on wan22 upstream).
    Raises RuntimeError (not pytest.fail) so the standalone driver reports it
    cleanly too.
    """
    from tensorrt_llm.media.encoding import save_video

    try:
        save_video(video, output_path, frame_rate=frame_rate)
    except RuntimeError as err:
        if "MP4 format requires ffmpeg" not in str(err):
            raise
        raise RuntimeError(
            "ffmpeg is unavailable for LPIPS video encoding; refusing to fall back "
            "to another codec because the golden comparison would measure codec "
            f"artifacts instead of model output: {err}"
        ) from err
    assert os.path.isfile(output_path), f"LTX-2.3 did not produce video {output_path}"


def _lpips_script_available():
    return os.path.isfile(VISUAL_GEN_LPIPS_EVAL_SCRIPT)


def _run_lpips_eval(tmp_dir, sample_id, prompt, reference_path, generated_path):
    """Score generated vs golden video with the shared LPIPS eval script.

    Reuses ``scripts/visualgen_eval/visual_gen_lpips_score_eval.py`` (the same
    script every visual_gen model uses) in "rescore existing media" mode --
    reference/generated paths only, no re-generation. Raises FileNotFoundError if
    the script is absent (caller decides skip vs. warn); RuntimeError on failure.
    """
    if not _lpips_script_available():
        raise FileNotFoundError(f"LPIPS eval script not found: {VISUAL_GEN_LPIPS_EVAL_SCRIPT}")

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
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else REPO_ROOT
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
        scores = json.load(fh)
    score = float(scores["mean_lpips_score"])
    print(f"\n[E2E {sample_id} video LPIPS] score: {score:.6f}")
    return score


# ---------------------------------------------------------------------------
# Audio faithfulness: log-mel L1 distance (self-contained; no external model).
#
# LPIPS is a visual metric, so the LTX-2.3 soundtrack needs its own gate. Log-mel
# L1 is the standard perceptual proxy: mismatched pitch/timbre/timing shows up as
# large per-bin differences, while deterministic reruns land near zero. The mel
# filterbank is built here (HTK formula) to avoid a librosa/torchaudio dependency.
# ---------------------------------------------------------------------------
def _wav_to_mono(audio):
    """Any audio tensor/array -> 1-D float32 mono torch tensor in roughly [-1, 1]."""
    a = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    while a.dim() > 2:  # drop batch dims, e.g. (B, C, T) -> (C, T)
        a = a.squeeze(0)
    if a.dim() == 2:  # (C, T) or (T, C) -> mono
        if a.shape[0] <= 8 and a.shape[0] < a.shape[1]:  # channels-first
            a = a.mean(dim=0)
        else:  # samples-first
            a = a.mean(dim=1)
    return a.contiguous()


def _save_wav(audio, path, sample_rate):
    """Write a mono int16 PCM WAV (stdlib only; deterministic on disk)."""
    mono = _wav_to_mono(audio)
    peak = mono.abs().max().item()
    if peak > 1.0:  # vocoder output is ~[-1, 1]; guard clipping just in case
        mono = mono / peak
    pcm = (mono.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    assert os.path.isfile(path), f"Failed to write WAV {path}"


def _load_wav(path):
    """Read a mono int16 WAV -> (float32 tensor in [-1, 1], sample_rate)."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(data.copy()), sample_rate


def _hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=None):
    """Triangular HTK mel filterbank, shape [n_mels, n_fft // 2 + 1]."""
    fmax = fmax if fmax is not None else sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_pts = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = torch.tensor([_mel_to_hz(m.item()) for m in mel_pts])

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        left_slope = (fft_freqs - left) / max((center - left).item(), 1e-6)
        right_slope = (right - fft_freqs) / max((right - center).item(), 1e-6)
        fb[m - 1] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return fb


def _log_mel(wav, sample_rate, n_fft=1024, hop=256, n_mels=64, top_db=80.0):
    """Log-mel power spectrogram (dynamic-range clamped), shape [n_mels, n_frames].

    Uses a per-clip ``top_db`` floor (librosa ``power_to_db`` convention): bins more
    than ``top_db`` below the peak are clamped. Without this, ``log(quiet_bin)`` swings
    wildly on tiny amplitude changes, so int16/codec noise in near-silent bins would
    dominate the L1 distance and swamp real perceptual differences.
    """
    mono = _wav_to_mono(wav)
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)  # [n_freqs, n_frames]
    fb = _mel_filterbank(sample_rate, n_fft, n_mels)
    mel = fb @ power  # [n_mels, n_frames]
    log_mel = torch.log(torch.clamp(mel, min=1e-10))
    # top_db dB below peak -> natural-log units (dB = 10/ln(10) * ln(power)).
    floor = log_mel.max() - (top_db / 10.0) * math.log(10.0)
    return torch.clamp(log_mel, min=floor)


def _audio_mel_l1_distance(generated, reference, gen_rate, ref_rate):
    """Mean |log-mel| difference between two waveforms (rate-matched, len-aligned)."""
    if gen_rate != ref_rate:
        raise ValueError(f"Sample-rate mismatch: generated {gen_rate} vs golden {ref_rate}")
    gen_mel = _log_mel(generated, gen_rate)
    ref_mel = _log_mel(reference, ref_rate)
    n = min(gen_mel.shape[1], ref_mel.shape[1])
    if n == 0:
        raise ValueError("Empty audio: one of the clips decoded to zero frames")
    diff = (gen_mel[:, :n] - ref_mel[:, :n]).abs().mean().item()
    print(f"[E2E audio log-mel L1] distance: {diff:.6f} (frames compared: {n})")
    return diff


# ---------------------------------------------------------------------------
# Deterministic LTX-2.3 A/V render (shared by golden creation + candidate runs).
# ---------------------------------------------------------------------------
def _generate_ltx23_av(video_out_path, audio_out_path):
    """Render the LPIPS clip deterministically; save MP4 (video) + WAV (audio).

    Returns the audio sample rate. Skips (never fails) if any model asset is
    missing so the suite degrades gracefully when checkpoints aren't mounted.
    """
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True)

    # Force eager + deterministic: nested @torch.compile decorators are not
    # suppressed by TorchCompileConfig(enable=False) alone, so wrap the whole
    # render (matches the upstream LTX-2 golden generator).
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
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _save_lpips_video_mp4(video, video_out_path, frame_rate=LTX23_LPIPS_FRAME_RATE)
    if audio is None:
        raise AssertionError("LTX-2.3 render returned no audio; expected an audio+video pipeline")
    _save_wav(audio, audio_out_path, audio_rate)
    return audio_rate


# ---------------------------------------------------------------------------
# Pytest fixtures + tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ltx23_av_candidate(tmp_path_factory):
    """Render the candidate A/V clip once per session; return (video, audio) paths."""
    out_dir = tmp_path_factory.mktemp("ltx23_av")
    video_path = os.path.join(str(out_dir), "ltx23_candidate_video.mp4")
    audio_path = os.path.join(str(out_dir), "ltx23_candidate_audio.wav")
    _generate_ltx23_av(video_path, audio_path)
    return video_path, audio_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_video_lpips_against_golden(tmp_path, ltx23_av_candidate):
    """Decoded video vs golden MP4 must stay under the shared LPIPS threshold."""
    _skip_if_missing(LTX23_GOLDEN_VIDEO, "LTX-2.3 LPIPS golden video")
    if not _lpips_script_available():
        pytest.skip(f"LPIPS eval script not found: {VISUAL_GEN_LPIPS_EVAL_SCRIPT}")
    video_path, _ = ltx23_av_candidate
    score = _run_lpips_eval(
        str(tmp_path),
        "ltx23",
        LTX23_LPIPS_PROMPT,
        LTX23_GOLDEN_VIDEO,
        video_path,
    )
    assert score < LTX23_LPIPS_THRESHOLD, (
        f"LTX-2.3 video LPIPS too high: {score:.6f} (expected < {LTX23_LPIPS_THRESHOLD:.6f})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_audio_against_golden(ltx23_av_candidate):
    """Decoded audio vs golden WAV must stay under the log-mel L1 threshold.

    Audio is the LTX-2.3-specific axis LPIPS cannot see; this is the only gate on
    the vocoder + BWE (48 kHz) output actually sounding like the reference.
    """
    _skip_if_missing(LTX23_GOLDEN_AUDIO, "LTX-2.3 golden audio")
    _, audio_path = ltx23_av_candidate
    generated, gen_rate = _load_wav(audio_path)
    reference, ref_rate = _load_wav(LTX23_GOLDEN_AUDIO)
    distance = _audio_mel_l1_distance(generated, reference, gen_rate, ref_rate)
    assert distance < LTX23_AUDIO_MEL_L1_THRESHOLD, (
        f"LTX-2.3 audio log-mel L1 too high: {distance:.6f} "
        f"(expected < {LTX23_AUDIO_MEL_L1_THRESHOLD:.6f})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_example(tmp_path):
    """Run examples/visual_gen/models/ltx23.py end-to-end with the BF16 config.

    Mirrors ``test_ltx2_example``: validates the example script + shared YAML work
    together and produce an MP4 (a smoke test on top of the quality gates above).
    """
    _skip_if_missing(LTX23_MODEL_PATH, "LTX-2.3 checkpoint", is_dir=True)
    _skip_if_missing(LTX23_TEXT_ENCODER_PATH, "LTX-2.3 text encoder (gemma-3-12b-it)", is_dir=True)

    examples_root = os.path.join(REPO_ROOT, "examples", "visual_gen")
    script_path = os.path.join(examples_root, "models", "ltx23.py")
    config_path = os.path.join(examples_root, "configs", "ltx23-t2v-bf16-1gpu.yaml")
    for label, path in [("example script", script_path), ("BF16 config", config_path)]:
        if not os.path.isfile(path):
            pytest.skip(f"LTX-2.3 {label} not found: {path}")

    output_path = os.path.join(str(tmp_path), "ltx23_output.mp4")
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--model",
            LTX23_MODEL_PATH,
            "--visual_gen_args",
            config_path,
            "--text_encoder_path",
            LTX23_TEXT_ENCODER_PATH,
            "--output_path",
            output_path,
            "--height",
            str(LTX23_LPIPS_HEIGHT),
            "--width",
            str(LTX23_LPIPS_WIDTH),
            "--num_frames",
            str(LTX23_LPIPS_NUM_FRAMES),
            "--num_inference_steps",
            str(LTX23_LPIPS_NUM_INFERENCE_STEPS),
        ],
        check=False,
    )
    assert result.returncode == 0, "LTX-2.3 example script exited non-zero"
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"


# ---------------------------------------------------------------------------
# Standalone driver (no pytest / CI harness needed) for use on a GPU node.
# ---------------------------------------------------------------------------
def _self_test_audio_metric():
    """Validate the log-mel L1 metric with synthetic signals (no model needed).

    Identical clips -> ~0; a pitch-shifted tone -> clearly larger. A cheap guard
    that the mel math + WAV round-trip behave before spending a GPU render.
    """
    sr = 48000
    t = torch.arange(0, sr, dtype=torch.float32) / sr  # 1 s
    tone_a = 0.5 * torch.sin(2 * math.pi * 440.0 * t)
    tone_a2 = 0.5 * torch.sin(2 * math.pi * 440.0 * t)  # identical
    tone_b = 0.5 * torch.sin(2 * math.pi * 660.0 * t)  # a fifth up

    same = _audio_mel_l1_distance(tone_a, tone_a2, sr, sr)
    diff = _audio_mel_l1_distance(tone_a, tone_b, sr, sr)
    print(f"[self-test] identical={same:.6f}  pitch-shifted={diff:.6f}")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        wav_path = os.path.join(d, "rt.wav")
        _save_wav(tone_a, wav_path, sr)
        loaded, loaded_sr = _load_wav(wav_path)
        rt = _audio_mel_l1_distance(tone_a, loaded, sr, loaded_sr)
    print(f"[self-test] wav round-trip distance={rt:.6f} (sr={loaded_sr})")

    # rt is int16 round-trip noise (I/O sanity, not the quality gate); with top_db
    # clamping this sits well under 0.05, but keep margin so it isn't brittle.
    ok = same < 1e-4 and diff > 0.5 and rt < 0.1
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _main(argv):
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2.3 golden / scoring driver")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the audio log-mel metric on synthetic tones (no model/GPU)",
    )
    parser.add_argument(
        "--make-golden",
        action="store_true",
        help=f"Render deterministically and freeze goldens into {LTX23_GOLDEN_DIR}",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Render a candidate and score video LPIPS + audio mel-L1 vs the goldens",
    )
    parser.add_argument("--work-dir", default="/tmp/ltx23_score", help="Scratch dir for candidates")
    args = parser.parse_args(argv)

    if not (args.self_test or args.make_golden or args.score):
        parser.error("pass --self-test, --make-golden and/or --score")

    if args.self_test:
        rc = _self_test_audio_metric()
        if rc != 0:
            return rc

    if args.make_golden:
        os.makedirs(LTX23_GOLDEN_DIR, exist_ok=True)
        rate = _generate_ltx23_av(LTX23_GOLDEN_VIDEO, LTX23_GOLDEN_AUDIO)
        print(f"[golden] video -> {LTX23_GOLDEN_VIDEO}")
        print(f"[golden] audio -> {LTX23_GOLDEN_AUDIO} ({rate} Hz)")

    if args.score:
        os.makedirs(args.work_dir, exist_ok=True)
        cand_video = os.path.join(args.work_dir, "ltx23_candidate_video.mp4")
        cand_audio = os.path.join(args.work_dir, "ltx23_candidate_audio.wav")
        _generate_ltx23_av(cand_video, cand_audio)

        # Audio gate (self-contained, always runs).
        generated, gen_rate = _load_wav(cand_audio)
        reference, ref_rate = _load_wav(LTX23_GOLDEN_AUDIO)
        audio_dist = _audio_mel_l1_distance(generated, reference, gen_rate, ref_rate)
        audio_ok = audio_dist < LTX23_AUDIO_MEL_L1_THRESHOLD

        # Video gate (needs the shared LPIPS eval script + its deps). Degrade to a
        # warning if unavailable so audio + determinism still report a result.
        video_score = None
        try:
            video_score = _run_lpips_eval(
                args.work_dir, "ltx23", LTX23_LPIPS_PROMPT, LTX23_GOLDEN_VIDEO, cand_video
            )
        except FileNotFoundError as err:
            print(f"[WARN] video LPIPS skipped: {err}")
        except RuntimeError as err:
            print(f"[WARN] video LPIPS could not run (missing deps/weights?):\n{err}")

        video_ok = video_score is None or video_score < LTX23_LPIPS_THRESHOLD
        if video_score is not None:
            print(
                f"\n[RESULT] video LPIPS {video_score:.6f} < {LTX23_LPIPS_THRESHOLD} -> "
                f"{'PASS' if video_ok else 'FAIL'}"
            )
        else:
            print("\n[RESULT] video LPIPS -> SKIPPED (eval script/deps unavailable)")
        print(
            f"[RESULT] audio mel-L1 {audio_dist:.6f} < {LTX23_AUDIO_MEL_L1_THRESHOLD} -> "
            f"{'PASS' if audio_ok else 'FAIL'}"
        )
        return 0 if (video_ok and audio_ok) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
