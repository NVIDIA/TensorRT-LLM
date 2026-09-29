# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check FA4 tactics, then report standalone CUDA-graph timings as JSON."""

import argparse
import json
import os
import statistics
from importlib.metadata import version
from pathlib import Path

import torch
from flash_attn.cute._trtllm_build_info import BUILD_ID

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TuningConfig, autotune
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import FlashAttn4Attention

# Import after the backend's CUTLASS compatibility shims.
from tensorrt_llm._torch.visual_gen.attention_backend.fa4_autotuner import (  # isort: skip
    Fa4Runner,
    _TACTICS,
)


def _measure(runner: Fa4Runner, inputs: list[torch.Tensor], tactic: int, repeats: int) -> float:
    for _ in range(3):
        runner(inputs, tactic=tactic)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(inputs, tactic=tactic)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096])
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TLLM_VISUAL_GEN_FA4_AUTOTUNE") != "1":
        parser.error("Set TLLM_VISUAL_GEN_FA4_AUTOTUNE=1 to enable the demo backend integration")
    if min(*args.seq_lens, args.heads, args.repeats, args.rounds) <= 0:
        parser.error("shapes, repeats and rounds must be positive")
    torch.manual_seed(42)
    report = {
        "scope": "standalone attention; not an end-to-end performance claim",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fa4": BUILD_ID,
        "cutlass": version("nvidia-cutlass-dsl"),
        "dtype": "bfloat16",
        "heads": args.heads,
        "head_dim": 128,
        "seed": 42,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "results": [],
    }
    for seq in args.seq_lens:
        inputs = [
            torch.randn(1, seq, args.heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        backend = FlashAttn4Attention(num_heads=args.heads, head_dim=128)
        runner = Fa4Runner(inputs, backend.scale)
        tactics = runner.get_valid_tactics(inputs, OptimizationProfile())
        reference, reference_lse = runner(inputs)
        rows = []
        for tactic in tactics:
            output, lse = runner(inputs, tactic=tactic)
            # LSE is part of the distributed-attention contract, not just a diagnostic.
            torch.testing.assert_close(output, reference, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(lse, reference_lse, atol=2e-3, rtol=2e-3)
            rows.append(
                {
                    "tactic": tactic,
                    "cta_exp2": "FA4 default/auto split" if tactic == -1 else _TACTICS[tactic],
                    "correctness": "output_and_lse_pass",
                    "samples_ms": [],
                }
            )
        # Alternate traversal order so the control is not always timed first.
        for round_idx in range(args.rounds):
            for row in rows if round_idx % 2 == 0 else reversed(rows):
                row["samples_ms"].append(_measure(runner, inputs, row["tactic"], args.repeats))
        for row in rows:
            row["median_ms"] = statistics.median(row["samples_ms"])
        with autotune():
            tuned_output = backend.forward(*inputs)
        torch.testing.assert_close(tuned_output, reference, atol=5e-3, rtol=5e-3)
        tuner = AutoTuner.get()
        # The cache itself is the authoritative selected tactic, not this benchmark's ranking.
        selected = tuner.choose_one("visual_gen::fa4_dense", [runner], TuningConfig(), inputs)[1]
        report["results"].append(
            {
                "seq_len": seq,
                "disable_2cta": runner.disable_2cta,
                "clc": runner.clc,
                "selected_tactic": selected,
                "tactics": rows,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
