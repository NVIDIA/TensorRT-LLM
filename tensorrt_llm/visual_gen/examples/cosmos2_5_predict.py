# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.distributed as dist
from common import BaseArgumentParser
from common import (
    autotuning,
    configure_cpu_offload,
    create_dit_config,
    generate_autotuner_dir,
    generate_output_path,
    save_output,
    setup_distributed,
    validate_parallel_config,
)
from diffusers.utils import load_image

from visual_gen import setup_configs
from visual_gen.pipelines.cosmos_pipeline import ditCosmos2_5_PredictBasePipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Cosmos pipeline."""
    # Create visual_gen configuration
    pipe_cfg = create_dit_config(args)
    setup_configs(**pipe_cfg)

    logger.info(f"Loading 2VideoToWorldPipeline with extra configs {pipe_cfg}...")
    pipe = ditCosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        **pipe_cfg,
    )

    # Setup distributed training and CPU offload
    local_rank, _world_size = setup_distributed()
    configure_cpu_offload(pipe, args, local_rank)

    return pipe


def run_inference(pipe, args, image, enable_autotuner: bool = False):
    """Run warmup and actual inference."""

    def inference_fn(warmup: bool = False):
        if warmup:
            num_inference_steps = args.num_warmup_steps
        else:
            num_inference_steps = args.num_inference_steps
        return pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=args.max_sequence_length,
        ).frames[0]  # List of PIL frames (T x H x W x C)

    autotuner_dir = args.autotuner_result_dir
    if enable_autotuner and not args.skip_autotuning:
        if autotuner_dir is None:
            autotuner_dir = generate_autotuner_dir(args)
        autotuning(inference_fn, autotuner_dir)

    frames = inference_fn(warmup=False)
    return frames, None


def main():
    """Main function for Cosmos image-to-video generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Cosmos Image-to-Video Generation")
    parser.add_image_input_args(
        default_image="https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/bus_terminal.jpg"
    )
    parser.add_video_args(default_num_frames=93, default_fps=16)
    parser.parser.add_argument(
        "--revision",
        type=str,
        default="diffusers/base/post-trained",
        help="HuggingFace model branch",
    )

    # Set defaults for Cosmos I2V
    parser.parser.set_defaults(
        model_path="nvidia/Cosmos-Predict2.5-2B",
        revision="diffusers/base/post-trained",
        prompt=(
            "A nighttime city bus terminal gradually shifts from stillness to subtle movement. "
            "At first, multiple double-decker buses are parked under the glow of overhead lights, "
            "with a central bus labeled '87D' facing forward and stationary. "
            "As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area "
            "and casting reflections onto adjacent vehicles. "
            "The motion creates space in the lineup, signaling activity within the otherwise quiet station. "
            "It then comes to a smooth stop, resuming its position in line. "
            "Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
        ),
        negative_prompt=(
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, "
            "low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, "
            "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
            "Overall, the video is of poor quality."
        ),
        height=704,
        width=1280,
        num_frames=93,
        guidance_scale=7.0,
        num_inference_steps=36,
        max_sequence_length=512,
        teacache_thresh=0.13,
    )
    args = parser.parse_args()

    enable_autotuner = False
    if args.linear_type == "auto" or args.attn_type == "auto":
        enable_autotuner = True
    if enable_autotuner:
        raise RuntimeError("Autotuner not supported by Cosmos 2.5.")

    if not args.disable_parallel_vae:
        logger.warning("Parallel VAE is not supported for Cosmos I2V, disable parallel VAE")
        args.disable_parallel_vae = True

    # Validate configuration
    validate_parallel_config(args)

    # Generate output path
    output_path = generate_output_path(args, save_type="mp4")

    # Load pipeline and prepare inputs
    pipe = load_and_setup_pipeline(args)

    # Load and resize the image
    if len(args.image) > 0:
        image = load_image(args.image)
        image = image.resize((args.width, args.height))
    else:
        image = None

    # Run inference
    frames, _ = run_inference(pipe, args, image, enable_autotuner)

    # Log results and save output
    save_output(frames, output_path, output_type="video")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
