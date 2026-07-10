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
"""Generate a synthetic edge-map control video: a ball bouncing off walls.

Draws white outlines on black — a room border plus a ball following simple
elastic-bounce physics — which is exactly what the Cosmos3 transfer ``edge``
hint expects. No media assets required: the control is 30 lines of math, and
transfer turns it into a photorealistic video whose subject follows the
physics frame by frame.

Generate the control, then run transfer with it:

    python generate_bouncing_ball_control.py --out_dir ./ball_control

    python cosmos3.py --model nvidia/Cosmos3-Nano \\
        --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml \\
        --prompt "A photorealistic beach ball with colorful panels bouncing \\
                  between the walls of an enclosed room, studio lighting." \\
        --extra_params '{"edge": {"control_path": "./ball_control/control.mp4"}}' \\
        --output_path cosmos3_bouncing_ball.mp4

Tip: keep synthetic controls edge-style. The ``blur`` hint expects the low
frequencies of natural video; flat synthetic color fields are far from its
training distribution and degrade generation quality.
"""

import argparse
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bouncing-ball edge-control generator")
    parser.add_argument("--out_dir", default="./ball_control")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--radius", type=int, default=90)
    parser.add_argument("--wall_inset", type=int, default=10, help="Room border inset in px")
    parser.add_argument("--line_width", type=int, default=6)
    parser.add_argument("--start", type=float, nargs=2, default=(300.0, 250.0))
    parser.add_argument(
        "--velocity",
        type=float,
        nargs=2,
        default=(26.0, 19.0),
        help="px/frame; the defaults bounce a few times over 49 frames",
    )
    parser.add_argument(
        "--save_frames", action="store_true", help="Also write the individual PNG frames"
    )
    return parser.parse_args()


def draw_frame(args: argparse.Namespace, x: float, y: float) -> PIL.Image.Image:
    image = PIL.Image.new("RGB", (args.width, args.height), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(image)
    inset, r = args.wall_inset, args.radius
    draw.rectangle(
        [inset, inset, args.width - inset, args.height - inset],
        outline=(255, 255, 255),
        width=args.line_width,
    )
    draw.ellipse(
        [x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=args.line_width
    )
    return image


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo_x, hi_x = args.wall_inset + args.radius, args.width - args.wall_inset - args.radius
    lo_y, hi_y = args.wall_inset + args.radius, args.height - args.wall_inset - args.radius
    x, y = args.start
    vx, vy = args.velocity

    frames = []
    for i in range(args.num_frames):
        frame = draw_frame(args, x, y)
        frames.append(frame)
        if args.save_frames:
            frame.save(out_dir / f"frame_{i:03d}.png")
        x, y = x + vx, y + vy
        if x < lo_x or x > hi_x:
            vx = -vx
            x = max(lo_x, min(x, hi_x))
        if y < lo_y or y > hi_y:
            vy = -vy
            y = max(lo_y, min(y, hi_y))

    # mpeg4 is a built-in FFmpeg encoder, available on any PyAV wheel.
    import av

    video_path = out_dir / "control.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=args.fps)
        stream.width = args.width
        stream.height = args.height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 8_000_000  # keep the thin white lines crisp
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")))
        container.mux(stream.encode())

    print(f"Wrote {video_path}" + (f" and {args.num_frames} PNG frames" if args.save_frames else ""))


if __name__ == "__main__":
    main()
