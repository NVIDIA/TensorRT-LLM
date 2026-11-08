#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm import VisualGen, VisualGenParams


def main():
    visual_gen = VisualGen(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    params = VisualGenParams(
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=50,
        seed=42,
    )
    output = visual_gen.generate(
        inputs="A cat sitting on a windowsill",
        params=params,
    )
    output.save("output.avi")


if __name__ == "__main__":
    main()
