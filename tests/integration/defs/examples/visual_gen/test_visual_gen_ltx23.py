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
"""LTX-2.3 video and audio quality tests against upstream golden media.

BF16 on one/two GPUs, FP8, and NVFP4 share the same golden comparisons and
strict deterministic execution. Override paths with
LTX23_MODEL_PATH (generation checkpoint), LTX23_RETAKE_CHECKPOINT,
LTX23_TEXT_ENCODER_PATH, and LTX23_GOLDEN_DIR.
"""

import os

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from defs.examples.visual_gen.visual_gen_test_utils import (
    REPO_ROOT,
    _assert_lpips_below_threshold,
    _cleanup_cuda,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _multi_resolution_log_stft_distance,
    _preserve_lpips_candidate_on_failure,
    _require_exists,
    _run_lpips_eval,
    _run_reusable_video_lpips_eval,
    _save_lpips_video_mp4,
    _visual_gen_output_path,
)

LTX23_GOLDEN_DIR = os.environ.get("LTX23_GOLDEN_DIR")

LTX23_LPIPS_PROMPT = (
    "A woman with long brown hair and light skin smiles at the camera while standing in a "
    "sunlit park, her hair gently blowing in the breeze as she tilts her head slightly to the side."
)
LTX23_LPIPS_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
LTX23_LPIPS_HEIGHT = 512
LTX23_LPIPS_WIDTH = 768
LTX23_LPIPS_NUM_FRAMES = 49
LTX23_LPIPS_NUM_INFERENCE_STEPS = 8
LTX23_LPIPS_GUIDANCE_SCALE = 5.0
LTX23_LPIPS_FRAME_RATE = 24.0
LTX23_LPIPS_SEED = 42

LTX23_ACCURACY_CASES = (
    pytest.param("bf16-1gpu", id="bf16"),
    pytest.param("bf16-2gpu", id="bf16-2gpu"),
    pytest.param("fp8-1gpu", id="fp8"),
    pytest.param("fp4-1gpu", id="nvfp4"),
)
LTX23_LPIPS_THRESHOLDS = {
    "bf16-1gpu": 0.05,
    "bf16-2gpu": 0.05,
    "fp8-1gpu": 0.20,
    "fp4-1gpu": 0.40,
}
LTX23_AUDIO_LOG_STFT_THRESHOLD = 0.10

LTX23_RETAKE_EXPECTED_FRAMES = 121
LTX23_RETAKE_EXPECTED_HEIGHT = 544
LTX23_RETAKE_EXPECTED_WIDTH = 960
LTX23_RETAKE_EXPECTED_FPS = 24.0
LTX23_RETAKE_EXPECTED_AUDIO_RATE = 48000
LTX23_RETAKE_AUDIO_SAMPLES = 240480
LTX23_RETAKE_CHECKPOINT_ENV = "LTX23_RETAKE_CHECKPOINT"
LTX23_RETAKE_LPIPS_FRAME_START = 48
LTX23_RETAKE_LPIPS_FRAME_STOP = 120
LTX23_RETAKE_LPIPS_THRESHOLDS = {
    "bf16-1gpu": 0.05,
    "bf16-2gpu": 0.05,
    "fp8-1gpu": 0.15,
    "fp4-1gpu": 0.15,
}
LTX23_RETAKE_PROMPT = "the duck flaps hard and flies right out of the frame"


def _ltx23_required_gpus(config_name: str) -> int:
    required = 2 if config_name.endswith("2gpu") else 1
    if torch.cuda.device_count() < required:
        pytest.skip(f"LTX-2.3 {config_name} requires {required} GPUs")
    return required


def _ltx23_model_paths() -> tuple[str, str]:
    # Resolve model-store paths after the distributed harness selects the installed wheel.
    model_path = os.environ.get(
        "LTX23_MODEL_PATH", _lpips_model_path("LTX-2.3", "ltx-2.3-22b-dev.safetensors")
    )
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "ltx-2.3-22b-dev.safetensors")
    text_encoder_path = os.environ.get(
        "LTX23_TEXT_ENCODER_PATH",
        _lpips_model_path("gemma-3-12b-it-qat-q4_0-unquantized"),
    )
    return model_path, text_encoder_path


def _ltx23_golden_media_path(tmp_path, media_name: str, label: str) -> str:
    if LTX23_GOLDEN_DIR is not None:
        path = os.path.join(LTX23_GOLDEN_DIR, media_name)
        _require_exists(path, label)
        return path
    return str(_golden_media_path(tmp_path, media_name, label))


def _save_wav(audio, path, sample_rate):
    tensor = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    mono = librosa.to_mono(tensor.reshape(-1, tensor.shape[-1]).numpy())
    sf.write(str(path), np.clip(mono, -1.0, 1.0), int(sample_rate))
    assert os.path.isfile(path), f"Failed to write WAV {path}"


def _audio_log_stft_distance(generated_path: str, reference_path: str) -> float:
    """Compare WAV channels with the shared multi-resolution log-STFT metric."""
    generated, gen_rate = sf.read(str(generated_path), dtype="float32", always_2d=True)
    reference, ref_rate = sf.read(str(reference_path), dtype="float32", always_2d=True)
    if gen_rate != ref_rate:
        raise ValueError(f"Sample-rate mismatch: generated {gen_rate} vs golden {ref_rate}")
    if generated.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("Empty audio: a clip decoded to zero samples")
    if not np.isfinite(generated).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite audio samples")
    distance = _multi_resolution_log_stft_distance(
        torch.from_numpy(generated.T), torch.from_numpy(reference.T)
    )
    print(f"[E2E audio log-STFT] distance: {distance:.6f} (samples: {generated.shape[0]})")
    return distance


def _save_ltx23_retake_audio(
    video_path: str | os.PathLike[str],
    reference_path: str | os.PathLike[str],
    audio_path: str | os.PathLike[str],
) -> None:
    """Remove only the fixed Duck case's video/AAC tail padding before WAV comparison."""
    import av

    from tensorrt_llm._torch.visual_gen.models.ltx23.media_io import decode_audio_from_file

    reference = sf.info(str(reference_path))
    assert reference.samplerate == LTX23_RETAKE_EXPECTED_AUDIO_RATE
    assert reference.channels == 1
    assert reference.frames == LTX23_RETAKE_AUDIO_SAMPLES
    with av.open(str(video_path)) as container:
        assert len(container.streams.audio) == 1, "LTX-2.3 retake output must have one audio stream"
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac" and stream.codec_context.profile == "LC", (
            "LTX-2.3 retake tail-padding allowance requires AAC-LC"
        )
        assert stream.start_time == 0, "LTX-2.3 retake audio must start at the video time origin"
    audio = decode_audio_from_file(str(video_path))
    assert audio is not None, "LTX-2.3 retake output has no decoded audio"
    assert audio.sample_rate == reference.samplerate, "LTX-2.3 retake audio sample-rate mismatch"
    assert audio.samples.ndim == 3 and audio.samples.shape[:2] == (1, 2), (
        f"LTX-2.3 retake expected (1, 2, samples) audio, got {tuple(audio.samples.shape)}"
    )
    assert torch.isfinite(audio.samples).all(), "LTX-2.3 retake audio has non-finite samples"
    video_samples = round(
        LTX23_RETAKE_EXPECTED_FRAMES * reference.samplerate / LTX23_RETAKE_EXPECTED_FPS
    )
    # 1520 samples pad this video's duration; AAC-LC can add at most 1023 tail samples.
    # Keep every valid sample from time zero; this is not signal alignment or resampling.
    max_samples = video_samples + 1023
    actual_samples = audio.samples.shape[-1]
    assert reference.frames <= actual_samples <= max_samples, (
        f"LTX-2.3 retake audio length {actual_samples} outside "
        f"[{reference.frames}, {max_samples}] valid samples plus known mux/AAC tail padding"
    )
    print(
        f"[Retake audio] decoded={actual_samples}, valid={reference.frames}, "
        f"removed_tail={actual_samples - reference.frames}; "
        f"mux_padding={video_samples - reference.frames}, AAC-LC_tail_max=1023"
    )
    _save_wav(audio.samples[..., : reference.frames], audio_path, audio.sample_rate)


def _generate_ltx23_av(rank, world_size, *, video_out_path, audio_out_path, config_name):
    """Render a configured golden-test candidate; only rank zero writes media."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    model_path, text_encoder_path = _ltx23_model_paths()
    _require_exists(model_path, "LTX-2.3 dev checkpoint")
    _require_exists(text_encoder_path, "LTX-2.3 text encoder (Gemma QAT-unquantized)", is_dir=True)

    with _lpips_deterministic_algorithms():
        config_path = os.path.join(
            REPO_ROOT, "examples", "visual_gen", "configs", f"ltx23-t2v-{config_name}.yaml"
        )
        args = VisualGenArgs.from_yaml(config_path)
        assert args.parallel_config.n_workers == world_size
        args.model = model_path
        args.pipeline_config = {
            **args.pipeline_config,
            "text_encoder_path": text_encoder_path,
        }
        args.compilation_config.resolutions = [(LTX23_LPIPS_HEIGHT, LTX23_LPIPS_WIDTH)]
        args.compilation_config.num_frames = [LTX23_LPIPS_NUM_FRAMES]
        pipeline = PipelineLoader(args, device=f"cuda:{rank}").load()
        try:
            with torch.no_grad():
                out = pipeline.forward(
                    prompt=LTX23_LPIPS_PROMPT,
                    negative_prompt=LTX23_LPIPS_NEGATIVE_PROMPT,
                    seed=LTX23_LPIPS_SEED,
                    height=LTX23_LPIPS_HEIGHT,
                    width=LTX23_LPIPS_WIDTH,
                    num_frames=LTX23_LPIPS_NUM_FRAMES,
                    frame_rate=LTX23_LPIPS_FRAME_RATE,
                    num_inference_steps=LTX23_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=LTX23_LPIPS_GUIDANCE_SCALE,
                )
            if rank == 0:
                video = out.video.detach().cpu()
                audio = out.audio.detach().cpu() if out.audio is not None else None
                audio_rate = out.audio_sample_rate
        finally:
            del pipeline
            _cleanup_cuda()

    if rank == 0:
        assert video.shape == (
            1,
            LTX23_LPIPS_NUM_FRAMES,
            LTX23_LPIPS_HEIGHT,
            LTX23_LPIPS_WIDTH,
            3,
        )
        _save_lpips_video_mp4(video, video_out_path, frame_rate=LTX23_LPIPS_FRAME_RATE)
        assert audio is not None, "LTX-2.3 render returned no audio"
        assert torch.isfinite(audio).all(), "LTX-2.3 render returned non-finite audio"
        assert audio_rate == 48000
        _save_wav(audio, audio_out_path, audio_rate)


@pytest.fixture(scope="session", params=LTX23_ACCURACY_CASES)
def ltx23_av_candidate(request, llm_venv, _visual_gen_deps):
    """Render each precision/parallel configuration once per session."""
    from defs.examples.visual_gen.test_visual_gen_multi_gpu import run_test_in_distributed

    config_name = request.param
    num_gpus = _ltx23_required_gpus(config_name)
    video_path = _visual_gen_output_path(llm_venv, f"ltx23_av_{config_name}")
    audio_path = os.path.join(os.path.dirname(video_path), "ltx23_candidate_audio.wav")
    with _lpips_deterministic_algorithms():
        run_test_in_distributed(
            num_gpus,
            _generate_ltx23_av,
            video_out_path=video_path,
            audio_out_path=audio_path,
            config_name=config_name,
        )
    return config_name, video_path, audio_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_lpips_against_golden(request, tmp_path, ltx23_av_candidate):
    """Check every generation configuration against the upstream video/audio goldens."""
    video_golden_path = _ltx23_golden_media_path(
        tmp_path, "ltx23_generation_lpips_golden_video.mp4", "LTX-2.3 LPIPS golden video"
    )
    audio_golden_path = _ltx23_golden_media_path(
        tmp_path, "ltx23_generation_lpips_golden_audio.wav", "LTX-2.3 golden audio"
    )
    config_name, video_path, audio_path = ltx23_av_candidate
    threshold = LTX23_LPIPS_THRESHOLDS[config_name]
    score = _run_lpips_eval(
        tmp_path,
        f"ltx23-{config_name}",
        "video",
        LTX23_LPIPS_PROMPT,
        video_golden_path,
        video_path,
    )
    distance = _audio_log_stft_distance(audio_path, audio_golden_path)
    _preserve_lpips_candidate_on_failure(
        request, score, threshold, video_path, f"ltx23_{config_name}_generated.mp4"
    )
    _assert_lpips_below_threshold(score, threshold)
    assert distance < LTX23_AUDIO_LOG_STFT_THRESHOLD, (
        f"LTX-2.3 audio log-STFT too high: {distance:.6f} "
        f"(expected < {LTX23_AUDIO_LOG_STFT_THRESHOLD:.6f})"
    )


def _generate_ltx23_retake_video(
    rank,
    world_size,
    *,
    source_path,
    output_path,
    config_filename="ltx23-retake-bf16-1gpu.yaml",
):
    """Run native retake under the same distributed harness as generation."""
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_video
    from tensorrt_llm.visual_gen.args import VisualGenArgs
    from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams, prepare_reference_slots

    model_path, text_encoder_path = _ltx23_model_paths()
    _require_exists(text_encoder_path, "LTX-2.3 text encoder (Gemma QAT-unquantized)", is_dir=True)
    checkpoint = os.environ.get(LTX23_RETAKE_CHECKPOINT_ENV) or os.path.join(
        os.path.dirname(model_path), "ltx-2.3-22b-distilled-1.1.safetensors"
    )
    _require_exists(checkpoint, "LTX-2.3 retake checkpoint")
    with _lpips_deterministic_algorithms():
        config = os.path.join(REPO_ROOT, "examples", "visual_gen", "configs", config_filename)
        args = VisualGenArgs.from_yaml(config)
        assert args.parallel_config.n_workers == world_size
        args.model = checkpoint
        args.pipeline_config = {**args.pipeline_config, "text_encoder_path": text_encoder_path}
        pipeline = PipelineLoader(args, device=f"cuda:{rank}").load()
        try:
            params = VisualGenParams(
                **pipeline.default_generation_params,
                video_reference=[MediaRef(content=str(source_path), format="path")],
                extra_params={
                    "retake_start_time": 2.0,
                    "retake_end_time": 5.0,
                    "retake_regenerate_video": True,
                    "retake_regenerate_audio": True,
                },
            )
            prepare_reference_slots(params)
            output = pipeline.run_inference(
                DiffusionRequest(request_id=0, prompt=[LTX23_RETAKE_PROMPT], params=params)
            )
            if rank == 0:
                video = output.video.detach().cpu()
                audio = output.audio.detach().cpu()
                frame_rate = output.frame_rate
                audio_rate = output.audio_sample_rate
        finally:
            del pipeline
            _cleanup_cuda()
    if rank == 0:
        save_video(
            video, output_path, audio=audio, frame_rate=frame_rate, audio_sample_rate=audio_rate
        )
        assert os.path.isfile(output_path), f"retake pipeline did not produce {output_path}"
        assert os.path.getsize(output_path) > 0, f"retake pipeline produced an empty {output_path}"


@pytest.fixture(scope="session")
def _ltx23_retake_deps(_visual_gen_deps, llm_venv):
    llm_venv.run_cmd(
        [
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torchaudio==2.11.0+cpu",
        ]
    )


@pytest.fixture(scope="session", params=LTX23_ACCURACY_CASES)
def ltx23_retake_candidate(request, _ltx23_retake_deps, llm_venv, tmp_path_factory):
    from defs.examples.visual_gen.test_visual_gen_multi_gpu import run_test_in_distributed

    config_name = request.param
    num_gpus = _ltx23_required_gpus(config_name)
    work_dir = tmp_path_factory.mktemp(f"ltx23_retake_{config_name}")
    source_path = _ltx23_golden_media_path(
        work_dir,
        "ltx23_retake_input.mp4",
        "LTX-2.3 official Duck retake input",
    )
    output_path = _visual_gen_output_path(llm_venv, f"ltx23_retake_{config_name}")
    with _lpips_deterministic_algorithms():
        run_test_in_distributed(
            num_gpus,
            _generate_ltx23_retake_video,
            source_path=source_path,
            output_path=output_path,
            config_filename=f"ltx23-retake-{config_name}.yaml",
        )
    return config_name, output_path


def _assert_ltx23_retake_output(video_path):
    """Check the retake output container, decoded frame count, and audio stream."""
    import av

    from tensorrt_llm._torch.visual_gen.models.ltx23.media_io import (
        decode_video_by_frame,
        get_videostream_metadata,
    )

    metadata = get_videostream_metadata(str(video_path))
    assert metadata.frames == LTX23_RETAKE_EXPECTED_FRAMES
    assert metadata.height == LTX23_RETAKE_EXPECTED_HEIGHT
    assert metadata.width == LTX23_RETAKE_EXPECTED_WIDTH
    assert metadata.fps == pytest.approx(LTX23_RETAKE_EXPECTED_FPS)
    assert sum(1 for _ in decode_video_by_frame(str(video_path))) == LTX23_RETAKE_EXPECTED_FRAMES

    container = av.open(str(video_path))
    try:
        audio_streams = list(container.streams.audio)
        assert len(audio_streams) == 1
        assert audio_streams[0].rate == LTX23_RETAKE_EXPECTED_AUDIO_RATE
        assert audio_streams[0].codec_context.channels == 2
    finally:
        container.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx23_retake_lpips_against_golden(
    request,
    tmp_path,
    ltx23_retake_candidate,
    _visual_gen_lpips_scorer,
):
    """Compare every retake configuration against the upstream video/audio goldens."""
    config_name, video_path = ltx23_retake_candidate
    threshold = LTX23_RETAKE_LPIPS_THRESHOLDS[config_name]
    _assert_ltx23_retake_output(video_path)
    golden_path = _ltx23_golden_media_path(
        tmp_path,
        "ltx23_retake_lpips_golden_video.mp4",
        "LTX-2.3 retake LPIPS golden video",
    )
    score = _run_reusable_video_lpips_eval(
        f"ltx23_retake-{config_name}",
        golden_path,
        video_path,
        _visual_gen_lpips_scorer,
        frame_start=LTX23_RETAKE_LPIPS_FRAME_START,
        frame_stop=LTX23_RETAKE_LPIPS_FRAME_STOP,
    )
    audio_golden_path = _ltx23_golden_media_path(
        tmp_path, "ltx23_retake_lpips_golden_audio.wav", "LTX-2.3 retake golden audio"
    )
    audio_path = tmp_path / "ltx23_retake_candidate_audio.wav"
    _save_ltx23_retake_audio(video_path, audio_golden_path, audio_path)
    distance = _audio_log_stft_distance(audio_path, audio_golden_path)
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        threshold,
        video_path,
        f"ltx23_retake_{config_name}_generated.mp4",
    )
    _assert_lpips_below_threshold(score, threshold)
    assert distance < LTX23_AUDIO_LOG_STFT_THRESHOLD, (
        f"LTX-2.3 retake audio log-STFT too high: {distance:.6f} "
        f"(expected < {LTX23_AUDIO_LOG_STFT_THRESHOLD:.6f})"
    )
