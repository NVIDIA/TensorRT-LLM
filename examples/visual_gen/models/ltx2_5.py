#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Official LTX-2.5 BF16 IC-LoRA two-stage editing through VisualGen."""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen.params import MediaRef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual_gen_args", required=True, help="BF16 component YAML.")
    parser.add_argument("--source", required=True, help="SDR control video.")
    parser.add_argument(
        "--prompt", default="A cinematic video following the structure and motion of the reference."
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="ltx2_5_iclora.mp4")
    args = parser.parse_args()
    config = VisualGenArgs.from_yaml(args.visual_gen_args)
    with VisualGen(model=args.model, args=config) as generator:
        params = generator.default_params
        params.height, params.width = args.height, args.width
        params.num_frames, params.seed = args.num_frames, args.seed
        params.video_reference = [MediaRef(content=args.source, format="path")]
        output = generator.generate(inputs=args.prompt, params=params)
        output.save(args.output_path)


if __name__ == "__main__":
    main()
