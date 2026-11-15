#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate an independent golden in a separately provisioned Lightricks environment."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Callable

OFFICIAL_COMMIT = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"
PROMPT = "A cinematic video following the structure and motion of the reference."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    commit = subprocess.check_output(
        ["git", "-C", str(args.ltx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"Reference must use {OFFICIAL_COMMIT}, found {commit}")
    changes = subprocess.check_output(
        ["git", "-C", str(args.ltx_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if changes:
        raise ValueError("Reference source has tracked modifications")
    for package in ("ltx-core", "ltx-pipelines"):
        sys.path.insert(0, str(args.ltx_root / "packages" / package / "src"))

    import av
    import soundfile as sf
    import torch
    import transformers
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.attention import (
        attention_label,
        automatic_attention,
        automatic_masked_attention,
    )
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths

    paths = {
        "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae_path": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "spatial_upsampler_path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "iclora_path": "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    }
    resolved = {key: str(args.model_root / value) for key, value in paths.items()}
    hashes = {value: sha256(args.model_root / value) for value in paths.values()}
    source_hash = sha256(args.source)
    # Never overwrite a previously accepted golden.
    args.output_directory.mkdir(parents=True, exist_ok=False)
    events = []
    stage_outputs = {}
    initializations = []
    original_sample_noise = GaussianNoiser._sample_noise

    def trace_noise(self: GaussianNoiser, latent_state) -> torch.Tensor:
        noise = original_sample_noise(self, latent_state)
        initializations.append(
            {
                "noise": noise.detach().cpu(),
                "initial_latent": latent_state.latent.detach().cpu(),
                "clean_latent": latent_state.clean_latent.detach().cpu(),
                "denoise_mask": latent_state.denoise_mask.detach().cpu(),
            }
        )
        return noise

    GaussianNoiser._sample_noise = trace_noise

    def record_event(event: dict) -> None:
        events.append(event)
        (args.output_directory / "timings.json").write_text(json.dumps(events, indent=2) + "\n")
        logging.info("TIMING %s", event)

    original_build = SingleGPUModelBuilder.build

    @wraps(original_build)
    def trace_build(
        self: SingleGPUModelBuilder, *values: object, **kwargs: object
    ) -> torch.nn.Module:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        model = original_build(self, *values, **kwargs)
        torch.cuda.synchronize()
        record_event(
            {
                "component": "model_build_including_loading_and_fusion",
                "checkpoint": str(self.checkpoint),
                "model_class": type(model).__name__,
                "seconds": time.perf_counter() - begin,
                "parameter_dtypes_numel": dict(
                    Counter(
                        {
                            dtype: sum(
                                p.numel() for p in model.parameters() if str(p.dtype) == dtype
                            )
                            for dtype in {str(p.dtype) for p in model.parameters()}
                        }
                    )
                ),
                "buffer_dtypes_numel": {
                    dtype: sum(b.numel() for b in model.buffers() if str(b.dtype) == dtype)
                    for dtype in {str(b.dtype) for b in model.buffers()}
                },
            }
        )
        return model

    SingleGPUModelBuilder.build = trace_build
    model_paths = ModelPaths.from_split(
        **{
            key: value
            for key, value in resolved.items()
            if key not in ("spatial_upsampler_path", "iclora_path")
        }
    )
    begin = time.perf_counter()
    pipeline = ICLoraPipeline(
        model_paths,
        resolved["spatial_upsampler_path"],
        [LoraPathStrengthAndSDOps(resolved["iclora_path"], 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
    )
    record_event({"component": "pipeline_constructor", "seconds": time.perf_counter() - begin})

    class MeasuredCall:
        def __init__(self, label: str, callback: Callable) -> None:
            self.label = label
            self.callback = callback

        def __call__(self, *values: object, **kwargs: object) -> object:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = self.callback(*values, **kwargs)
            torch.cuda.synchronize()
            record_event({"component": self.label, "seconds": time.perf_counter() - begin})
            if self.label in ("stage_1", "stage_2"):
                stage_outputs[self.label] = {
                    "video": result[0].latent.detach().cpu(),
                    "audio": result[1].latent.detach().cpu(),
                }
            elif self.label == "spatial_upsampler":
                stage_outputs["upsampled_video"] = result.detach().cpu()
            return result

    for attribute, label in (
        ("stage_1", "stage_1"),
        ("stage_2", "stage_2"),
        ("upsampler", "spatial_upsampler"),
        ("audio_decoder", "audio_decode"),
    ):
        setattr(pipeline, attribute, MeasuredCall(label, getattr(pipeline, attribute)))
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_pipelines.utils.blocks import PromptEncoder

    steps, contexts = [], {}
    original_step = EulerDiffusionStep.step
    original_prompt = PromptEncoder.__call__

    def trace_step(self, sample, denoised_sample, sigmas, step_index, **kwargs):
        result = original_step(self, sample, denoised_sample, sigmas, step_index, **kwargs)
        steps.append(
            {
                "latent": sample.detach().cpu(),
                "next": result.detach().cpu(),
                "sigma": float(sigmas[step_index]),
            }
        )
        return result

    def trace_prompt(self, *values, **kwargs):
        result = original_prompt(self, *values, **kwargs)
        contexts.update(video=result[0].video_encoding.cpu(), audio=result[0].audio_encoding.cpu())
        return result

    # This standalone reference process owns these instrumentation hooks.
    EulerDiffusionStep.step = trace_step
    PromptEncoder.__call__ = trace_prompt
    inference_started = time.perf_counter()
    with torch.inference_mode():
        frames, audio, resolved_tiling = pipeline(
            prompt=PROMPT,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=24,
            images=[],
            video_conditioning=[(str(args.source), 1.0)],
            vae_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        begin = time.perf_counter()
        video = torch.cat(list(frames), dim=0)
        torch.cuda.synchronize()
        record_event({"component": "video_decode_iterator", "seconds": time.perf_counter() - begin})
    inference_seconds = time.perf_counter() - inference_started
    if len(steps) != 22:
        raise ValueError(f"Expected 22 video/audio Euler updates, found {len(steps)}")
    if tuple(video.shape) != (args.num_frames, args.height, args.width, 3):
        raise ValueError(f"Unexpected raw video shape: {tuple(video.shape)}")
    torch.save(
        {
            "video": video.cpu(),
            "audio": audio.waveform.cpu(),
            "sample_rate": audio.sampling_rate,
            "steps": steps,
            "contexts": contexts,
            "stage_outputs": stage_outputs,
            "initializations": initializations,
            "recipe": {
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "seed": args.seed,
            },
        },
        args.output_directory / "official-bf16.pt",
    )
    # Video-only golden prevents audio duration rounding from truncating video.
    encode_video(
        video,
        24,
        None,
        str(
            args.output_directory
            / f"official-bf16-{args.height}x{args.width}-{args.num_frames}f.mp4"
        ),
        1,
    )
    waveform = audio.waveform.detach().float().cpu()
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 2 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid decoded audio: {tuple(waveform.shape)}")
    wav_path = args.output_directory / "official-bf16.wav"
    sf.write(wav_path, waveform.T.numpy(), audio.sampling_rate, subtype="FLOAT")
    video_path = (
        args.output_directory / f"official-bf16-{args.height}x{args.width}-{args.num_frames}f.mp4"
    )
    av_path = args.output_directory / "official-bf16-av.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-n",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-filter:a",
            "apad",
            "-t",
            str(args.num_frames / 24),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(av_path),
        ],
        check=True,
    )
    validation = {}
    for output in (video_path, av_path):
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            info = {
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate),
                "audio_streams": len(container.streams.audio),
            }
            decoded = list(container.decode(video=0))
            info["frames"] = len(decoded)
            info["frame_std"] = float(decoded[0].to_ndarray(format="rgb24").std())
            info["minimum_frame_std"] = min(
                float(frame.to_ndarray(format="rgb24").std()) for frame in decoded
            )
        if (info["width"], info["height"], info["frames"], info["fps"]) != (
            args.width,
            args.height,
            args.num_frames,
            24.0,
        ):
            raise ValueError(f"Output video validation failed: {info}")
        if info["minimum_frame_std"] <= 1.0:
            raise ValueError(f"Output contains an empty or nearly uniform frame: {info}")
        validation[output.name] = info
    if validation[av_path.name]["audio_streams"] != 1:
        raise ValueError("Muxed output must contain one audio stream")
    decoded_wav, rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if rate != audio.sampling_rate or not torch.equal(
        torch.from_numpy(decoded_wav.T.copy()), waveform
    ):
        raise ValueError("Float WAV does not preserve the decoded waveform")
    validation["audio"] = {
        "sample_rate": rate,
        "channels": waveform.shape[0],
        "samples": waveform.shape[1],
        "duration_seconds": waveform.shape[1] / rate,
        "rms": float(waveform.square().mean().sqrt()),
        "peak": float(waveform.abs().max()),
    }
    with av.open(str(av_path)) as container:
        audio_stream = container.streams.audio[0]
        decoded_audio = list(container.decode(audio=0))
        if not decoded_audio:
            raise ValueError("Muxed audio contains no decodable samples")
        validation["muxed_audio"] = {
            "sample_rate": audio_stream.rate,
            "channels": audio_stream.channels,
            "decoded_samples_including_codec_padding": sum(
                frame.samples for frame in decoded_audio
            ),
        }
    (args.output_directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "official_commit": commit,
        "checkpoint_sha256": hashes,
        "source_sha256": source_hash,
        "prompt": PROMPT,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": 24,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "iclora_scale": 1.0,
        "reference_strength": 1.0,
        "stages": [8, 3],
        "stage_1_sigmas": DISTILLED_SIGMAS.tolist(),
        "stage_2_sigmas": STAGE_2_DISTILLED_SIGMAS.tolist(),
        "reference_downscale_factor": pipeline.reference_downscale_factor,
        "reference_temporal_scale_factor": pipeline.reference_temporal_scale_factor,
        "offload_mode": "NONE",
        "compile": False,
        "quantization": None,
        "tiling_requested": "official AUTO_TILING default",
        "tiling_resolved": repr(resolved_tiling),
        "attention_policy": attention_label(automatic_attention()),
        "masked_attention_policy": attention_label(automatic_masked_attention()),
        "stage_1": {"iclora_scale": 1.0, "reference_strength": 1.0, "generate_audio": True},
        "stage_2": {"iclora_scale": 0.0, "reference_tokens": False, "refine_audio": True},
        "prompt_enhancement": False,
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "inference_seconds_including_lazy_model_loading_and_tracing": inference_seconds,
        "process_seconds_before_manifest": time.perf_counter() - started,
        "steady_state_latency": None,
        "validation": validation,
        "output_sha256": {
            path.name: sha256(path) for path in args.output_directory.iterdir() if path.is_file()
        },
    }
    (args.output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
