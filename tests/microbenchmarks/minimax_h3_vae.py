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

"""Benchmark H3 VAE decode using identical latents and checkpoint weights.

Run with torchrun --standalone --nproc_per_node=4, or python for one GPU.
The baseline modes explicitly disable tiling or retain sequential tiling.
Model loading and CPU copies are excluded; synchronization and tile collectives
are included. Reports the slowest rank's latency and peak allocated memory.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import video_latent_num_frames
from tensorrt_llm._torch.visual_gen.models.minimax_h3.tiled_vae import (
    MINIMAX_H3_VAE_DEFAULTS,
    TiledAutoencoderKLMiniMaxH3,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latents", type=Path, help="Optional unnormalized NCTHW video latents.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sequential", "parallel", "untiled"],
        choices=["sequential", "parallel", "untiled"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if distributed else 0
    group = dist.group.WORLD if distributed else None
    vae = (
        TiledAutoencoderKLMiniMaxH3.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    ratio = vae.spatial_compression_ratio
    if args.latents is not None:
        z = torch.load(args.latents, map_location=device, weights_only=True)
    else:
        if args.height % ratio or args.width % ratio:
            parser.error("height and width must align to the VAE compression ratio")
        z = torch.randn(
            1,
            vae.config.latent_channels,
            video_latent_num_frames(args.frames),
            args.height // ratio,
            args.width // ratio,
            generator=torch.Generator(device=device).manual_seed(42),
            device=device,
        )
        z = z * torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        z = z + torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(device),
        "world_size": dist.get_world_size() if distributed else 1,
        "latent_shape": list(z.shape),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "modes": {},
    }
    reference = None
    for mode in args.modes:
        vae.configure_tiling(
            {
                **MINIMAX_H3_VAE_DEFAULTS,
                "vae_use_tiling": mode != "untiled",
                "vae_tile_parallel": mode == "parallel",
            },
            group=group,
        )
        seconds = []
        peaks = []
        for iteration in range(args.warmup + args.repeats):
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                output = vae.decode(z, return_dict=False)[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()
            stats = torch.tensor([elapsed, peak], device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            if iteration >= args.warmup:
                seconds.append(stats[0].item())
                peaks.append(stats[1].item())
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "iteration": iteration,
                            "seconds": stats[0].item(),
                            "peak_bytes": stats[1].item(),
                        }
                    ),
                    flush=True,
                )
            if iteration == args.warmup + args.repeats - 1:
                candidate = output.cpu() if rank == 0 else None
            del output
        if rank == 0:
            assert torch.isfinite(candidate).all(), f"Non-finite {mode} output"
            result = {
                "seconds": seconds,
                "mean_seconds": statistics.mean(seconds),
                "peak_allocated_gib": max(peaks) / 1024**3,
                "output_shape": list(candidate.shape),
            }
            if mode == "sequential":
                reference = candidate
            elif reference is not None:
                delta = candidate.float() - reference.float()
                result["max_abs_diff_vs_sequential"] = delta.abs().max().item()
                result["rmse_vs_sequential"] = delta.square().mean().sqrt().item()
                if mode == "parallel":
                    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
            results["modes"][mode] = result
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
