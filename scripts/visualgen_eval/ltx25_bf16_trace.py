#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline tensor/memory qualification; public API coverage lives in integration tests."""

import argparse
import json
import math
import resource
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    import torch

    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import LTX25ICLoraPipeline
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import pipeline_ltx2_5 as impl
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs
    from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams

    parser = argparse.ArgumentParser(
        description="Capture installed-wheel BF16 tensors and memory for independent comparison."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-allocated-gib",
        type=float,
        required=True,
        help="Explicit device-memory acceptance budget for the target GPU.",
    )
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    if not math.isfinite(options.max_allocated_gib) or options.max_allocated_gib <= 0:
        parser.error("--max-allocated-gib must be positive and finite")
    torch.set_num_threads(8)
    assets = options.model_root
    opts = {
        "text_encoder_path": str(
            assets / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        ),
        "video_vae_path": str(assets / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        "audio_vae_path": str(assets / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        "spatial_upsampler_path": str(
            assets / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        ),
        "iclora_path": str(assets / "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
        "iclora_scale": 1.0,
    }
    model = str(assets / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
    args = VisualGenArgs(
        model=model,
        pipeline_config=opts,
        attention_config={"backend": "VANILLA"},
        cuda_graph_config={"enable": False},
        torch_compile_config={"enable": False},
    )
    loader = PipelineLoader(args, device="cuda")
    config = DiffusionPipelineConfig.from_pretrained(model, args=args, pipeline_config=opts)
    config.extra_attrs.update(opts)
    loader._setup_visual_gen_mapping(config)
    with MetaInitMode():
        pipeline = LTX25ICLoraPipeline(config)
    loader._materialize_meta_tensors(pipeline)
    weights = pipeline.load_transformer_weights(model)
    pipeline.load_weights(weights)
    del weights
    pipeline.load_standard_components(
        model, torch.device("cuda"), text_encoder_path=opts["text_encoder_path"]
    )
    pipeline.post_load_weights()
    contexts = {}
    original_context = pipeline._process_connectors

    def traced_context(*args, **kwargs):
        result = original_context(*args, **kwargs)
        contexts.update(video=result[0].cpu(), audio=result[1].cpu())
        return result

    pipeline._process_connectors = traced_context
    steps = []
    original = impl.euler_step

    def traced(state, velocity, sigma, next_sigma, **kwargs):
        result = original(state, velocity, sigma, next_sigma, **kwargs)
        steps.append(
            {
                "latent": state.latent.detach().cpu(),
                "velocity": velocity.detach().cpu(),
                "next": result.latent.detach().cpu(),
                "sigma": float(sigma),
            }
        )
        return result

    impl.euler_step = traced
    source = options.source
    load_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    params = VisualGenParams(
        height=options.height,
        width=options.width,
        num_frames=options.num_frames,
        frame_rate=24,
        seed=options.seed,
        num_inference_steps=8,
        guidance_scale=1,
        max_sequence_length=1024,
        video_reference=MediaRef(content=source.read_bytes(), format="bytes"),
    )
    with torch.inference_mode():
        result = pipeline.infer(
            SimpleNamespace(
                params=params,
                prompt="A cinematic video following the structure and motion of the reference.",
            )
        )
    torch.cuda.synchronize()
    metrics = {
        "load_peak_allocated_gib": load_peak_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "host_maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }
    torch.save(
        {
            "video": result.video,
            "audio": result.audio.cpu(),
            "steps": steps,
            "contexts": contexts,
            "metrics": metrics,
            "recipe": {
                "height": options.height,
                "width": options.width,
                "num_frames": options.num_frames,
                "seed": options.seed,
            },
        },
        options.output,
    )
    print(json.dumps(metrics, indent=2))
    if max(load_peak_gib, metrics["peak_allocated_gib"]) >= options.max_allocated_gib:
        raise AssertionError(
            f"Peak allocation exceeds the {options.max_allocated_gib} GiB qualification budget"
        )


if __name__ == "__main__":
    main()
