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

import numpy as np
import torch
import torch.distributed as dist
from common import (
    BaseArgumentParser,
    autotuning,
    benchmark_inference,
    configure_cpu_offload,
    create_dit_config,
    generate_autotuner_dir,
    generate_output_path,
    log_args_and_timing,
    save_output,
    setup_distributed,
    validate_parallel_config,
)
from diffusers.utils import load_image
from visual_gen import setup_configs
from visual_gen.models.transformers.cosmos_transformer import ditCosmosTransformer3DModel
from visual_gen.pipelines.cosmos_pipeline import ditCosmos2VideoToWorldPipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Cosmos pipeline."""
    # Create dit configuration
    pipe_cfg = create_dit_config(args)
    setup_configs(**pipe_cfg)

    logger.info("Loading ditCosmos2VideoToWorldPipeline...")
    transformer = ditCosmosTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = ditCosmos2VideoToWorldPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, **pipe_cfg
    )

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Cosmos CPU offload configuration (no specific model_wise/block_wise needed)
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
            fps=args.fps,
            max_sequence_length=args.max_sequence_length,
        ).frames[0]

    autotuner_dir = args.autotuner_result_dir
    if enable_autotuner and not args.skip_autotuning:
        if autotuner_dir is None:
            autotuner_dir = generate_autotuner_dir(args)
        autotuning(inference_fn, autotuner_dir)

    frames, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=True,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return frames, elapsed_time


def main():
    """Main function for Cosmos image-to-video generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Cosmos Image-to-Video Generation")
    parser.add_image_input_args(
        default_image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
    )
    parser.add_video_args(default_num_frames=93, default_fps=16)

    # Set defaults for Cosmos I2V
    parser.parser.set_defaults(
        model_path="nvidia/Cosmos-Predict2-2B-Video2World",
        prompt="A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess.",
        negative_prompt="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
        height=704,
        width=1280,
        guidance_scale=7.0,
        fps=16,
        max_sequence_length=512,
    )
    args = parser.parse_args()

    enable_autotuner = False
    if args.linear_type == "auto" or args.attn_type == "auto":
        enable_autotuner = True
        if not args.disable_torch_compile:
            logger.warning("Disable torch compile when using autotuner")
            args.disable_torch_compile = True
        if args.enable_async_cpu_offload:
            logger.warning("Disable visual_gen cpu offload when using autotuner")
            args.enable_async_cpu_offload = False

    if not args.disable_parallel_vae:
        logger.warning("Parallel VAE is not supported for Cosmos I2V, disable parallel VAE")
        args.disable_parallel_vae = True
        # args.height, args.width = recompute_shape_for_vae(height, width, pipe.vae_scale_factor_spatial, pipe.transformer.config.patch_size[1], args.parallel_vae_split_dim)

    if args.enable_teacache:
        logger.warning("TeaCache is not supported for Cosmos I2V, disable TeaCache")
        args.enable_teacache = False

    # Validate configuration
    validate_parallel_config(args)

    # Generate output path
    output_path = generate_output_path(args, save_type="mp4")

    # Load pipeline and prepare inputs
    pipe = load_and_setup_pipeline(args)

    """Load image and calculate optimal dimensions."""
    image = load_image(args.image)
    max_area = args.height * args.width
    aspect_ratio = image.height / image.width
    round(np.sqrt(max_area * aspect_ratio))
    round(np.sqrt(max_area / aspect_ratio))

    image = image.resize((args.width, args.height))

    # Run inference
    frames, elapsed_time = run_inference(pipe, args, image, enable_autotuner)

    # Log results and save output
    log_args_and_timing(args, elapsed_time)
    save_output(frames, output_path, output_type="video", fps=args.fps)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
