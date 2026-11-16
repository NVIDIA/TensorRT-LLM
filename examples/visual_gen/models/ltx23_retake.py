#!/usr/bin/env python3
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
"""Run one-GPU native LTX-2.3 retake with the public VisualGen API.

Defaults to the LTX-2.3 distilled 1.1 checkpoint.
Use ``--model`` to select a checkpoint file, directory, or HuggingFace Hub ID.
Use ``--text_encoder_path`` to override the Gemma text encoder. Source
decoding requires PyAV, and audio conditioning requires a ``torchaudio`` build
compatible with the installed PyTorch.
Both video and audio are regenerated in the selected window by default.
Use ``--no-regenerate_video`` or ``--no-regenerate_audio`` to freeze a modality.
Frozen streams still pass through the native VAE; they are not file copies.
An input without an audio track generates audio from scratch, even when
``--no-regenerate_audio`` is set, matching the upstream retake pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.ltx23.checkpoint import LTX23_MODEL_ID
from tensorrt_llm.visual_gen import MediaRef

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ltx23-retake-bf16-1gpu.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=LTX23_MODEL_ID,
        help="Checkpoint file, directory, or Hub ID. Directories/Hub IDs select distilled 1.1.",
    )
    parser.add_argument(
        "--input", required=True, help="Input media file containing video and optional audio."
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Retake start time in seconds. Defaults to the beginning of the video.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Retake end time in seconds. Defaults to the end of the video.",
    )
    parser.add_argument("--output_path", default="retake_output.mp4", help="Output MP4 path.")
    parser.add_argument("--prompt", default="", help="Retake text prompt.")
    parser.add_argument(
        "--regenerate_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate source audio in the window (default: true).",
    )
    parser.add_argument(
        "--regenerate_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate video in the window (default: true).",
    )
    parser.add_argument(
        "--visual_gen_args",
        default=str(_DEFAULT_CONFIG),
        help="VisualGen YAML configuration.",
    )
    parser.add_argument(
        "--text_encoder_path",
        default=None,
        help="Optional Gemma text encoder path override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visual_gen_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    pipeline_config = dict(visual_gen_args.pipeline_config)
    if args.text_encoder_path is not None:
        pipeline_config["text_encoder_path"] = args.text_encoder_path
    visual_gen_args = visual_gen_args.model_copy(update={"pipeline_config": pipeline_config})

    with VisualGen(model=args.model, args=visual_gen_args) as visual_gen:
        params = visual_gen.default_params
        params.video_reference = [MediaRef(content=args.input, format="path")]
        params.extra_params["retake_regenerate_video"] = args.regenerate_video
        params.extra_params["retake_regenerate_audio"] = args.regenerate_audio
        if args.start is not None:
            params.extra_params["retake_start_time"] = args.start
        if args.end is not None:
            params.extra_params["retake_end_time"] = args.end
        output = visual_gen.generate(inputs=args.prompt, params=params)
        output.save(args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
