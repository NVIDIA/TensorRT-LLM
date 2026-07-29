#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cold-start ring-4 test.

Each run is a fresh process with no prior ring-4 calls.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nccl-cumem", action="store_true")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model-path", default="/workspace/models/Cosmos3-Nano")
    ap.add_argument("--out-dir", default="/workspace/results")
    a = ap.parse_args()

    if a.nccl_cumem:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
    else:
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ.setdefault("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "1")

    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    PROMPTS = [
        "A majestic lion resting on a sun-drenched savanna",
        "Futuristic city skyline reflected in a calm river",
        "A waterfall cascading into a turquoise pool",
        "Northern lights above a snow-covered forest",
        "Ancient temple ruins illuminated by morning mist",
        "Desert dunes at sunset with long shadows",
        "Underwater coral reef teeming with tropical fish",
    ]

    tag = a.tag or ("cumem" if a.nccl_cumem else "base")
    print(f"\n{'=' * 60}")
    print(f"COLD RING-4  NCCL_CUMEM_ENABLE={os.environ['NCCL_CUMEM_ENABLE']}  tag={tag}")
    print(f"warmup={a.warmup}  iters={a.iters}")
    print(f"{'=' * 60}\n")

    args = VisualGenArgs(
        attention_config={"backend": "CUTEDSL"},
        parallel_config={"ring_size": 4},
    )
    print("Loading model...")
    vg = VisualGen(a.model_path, args=args)

    gen_kwargs = dict(
        height=720,
        width=1280,
        num_inference_steps=20,
        guidance_scale=6.0,
        seed=42,
        num_frames=57,
    )

    times = []
    try:
        for i in range(a.warmup):
            print(f"  warmup {i + 1}/{a.warmup}...")
            vg.generate(inputs=PROMPTS[i % len(PROMPTS)], params=VisualGenParams(**gen_kwargs))
            print(f"  warmup {i + 1} done")

        for i in range(a.iters):
            t0 = time.perf_counter()
            vg.generate(inputs=PROMPTS[i % len(PROMPTS)], params=VisualGenParams(**gen_kwargs))
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  iter {i + 1}/{a.iters}: {elapsed:.2f}s")
    finally:
        vg.shutdown()

    result = {
        "tag": tag,
        "nccl_cumem": os.environ["NCCL_CUMEM_ENABLE"],
        "warmup": a.warmup,
        "iters": a.iters,
        "all_s": [round(t, 3) for t in times],
        "min_s": round(min(times), 3),
        "median_s": round(statistics.median(times), 3),
        "mean_s": round(statistics.mean(times), 3),
    }
    print(f"\nResult: {result}")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cold_ring4_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
