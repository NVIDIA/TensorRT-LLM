# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a single image from text using the MiniMax-H3-Image checkpoint."""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the approved image checkpoint.")
    parser.add_argument(
        "--visual_gen_args",
        default=str(
            Path(__file__).resolve().parents[1] / "configs/minimax-h3-image-bf16-1gpu.yaml"
        ),
        help="VisualGenArgs YAML (default: the single-GPU eager configuration).",
    )
    parser.add_argument("--prompt", default="A watercolor of a mountain lake at sunrise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="minimax_h3_image.png")
    args = parser.parse_args()

    visual_gen = VisualGen(model=args.model, args=VisualGenArgs.from_yaml(args.visual_gen_args))
    params = visual_gen.default_params  # The checkpoint uses six learned PDD steps and no CFG.
    params.seed = args.seed
    output = visual_gen.generate(inputs=args.prompt, params=params)
    print(f"Saved image to {output.save(args.output_path)}")


if __name__ == "__main__":
    main()
