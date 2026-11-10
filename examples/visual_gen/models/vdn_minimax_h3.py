# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate video and stereo audio with the VDN-H3 eight-step checkpoint."""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="OpenVDN/vdn-minimax-h3")
    parser.add_argument(
        "--base_checkpoint", default="", help="Optional existing MiniMax-H3 checkpoint."
    )
    parser.add_argument("--output", default="vdn_h3.mp4")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num_frames", type=int, default=124)
    parser.add_argument(
        "--prompt",
        default="A woman smiles at the camera in a sunlit park. Birds sing in the background.",
    )
    options = parser.parse_args()
    args = VisualGenArgs(
        pipeline_config={"base_checkpoint": options.base_checkpoint},
        attention_config={"backend": "FA4"},
        cuda_graph_config={"enable": False},
        torch_compile_config={"enable": True, "enable_fullgraph": False},
        compilation_config={"skip_warmup": True},
    )
    generator = VisualGen(model=options.model, args=args)
    params = generator.default_params
    params.height, params.width = options.height, options.width
    params.num_frames, params.seed = options.num_frames, 0
    output = generator.generate(inputs=options.prompt, params=params)
    if output.error:
        raise RuntimeError(output.error)
    print(output.save(options.output))


if __name__ == "__main__":
    main()
