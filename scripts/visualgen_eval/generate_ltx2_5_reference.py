#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate an independent golden in a separately provisioned Lightricks environment."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

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

    import torch
    import transformers
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ic_lora import ICLoraPipeline
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
    model_paths = ModelPaths.from_split(
        **{
            key: value
            for key, value in resolved.items()
            if key not in ("spatial_upsampler_path", "iclora_path")
        }
    )
    pipeline = ICLoraPipeline(
        model_paths,
        resolved["spatial_upsampler_path"],
        [LoraPathStrengthAndSDOps(resolved["iclora_path"], 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
    )
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
    with torch.inference_mode():
        frames, audio, _ = pipeline(
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
        video = torch.cat(list(frames), dim=0)
    torch.save(
        {
            "video": video.cpu(),
            "audio": audio.waveform.cpu(),
            "sample_rate": audio.sampling_rate,
            "steps": steps,
            "contexts": contexts,
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
    }
    (args.output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
