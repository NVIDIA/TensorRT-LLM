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
    recompute_shape_for_vae,
    save_output,
    setup_distributed,
    validate_parallel_config,
)
from diffusers.utils import load_image
from transformers import CLIPVisionModel

from visual_gen import setup_configs
from visual_gen.models.transformers.wan_transformer import ditWanTransformer3DModel
from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL
from visual_gen.pipelines.wan_pipeline import ditWanImageToVideoPipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Wan I2V pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    # Load components
    vae = ditWanAutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = ditWanTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # Load image encoder for Wan2.1+ models
    if "Wan2.1" in args.model_path:
        image_encoder = CLIPVisionModel.from_pretrained(
            args.model_path, subfolder="image_encoder", torch_dtype=torch.float32
        )
        pipe = ditWanImageToVideoPipeline.from_pretrained(
            args.model_path,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
            **dit_configs,
        )
    else:
        pipe = ditWanImageToVideoPipeline.from_pretrained(
            args.model_path,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16,
            **dit_configs,
        )

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Wan I2V CPU offload configuration
    model_wise = ["text_encoder", "image_encoder"]  # I2V models have image encoder
    block_wise = ["transformer"]
    if "Wan2.2" in args.model_path:
        block_wise.append("transformer_2")

    configure_cpu_offload(pipe, args, local_rank, model_wise=model_wise, block_wise=block_wise)

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
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=num_inference_steps,
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
        deterministic=False,  # torch compile can't work with deterministic=True in Wan2.2
    )
    return frames, elapsed_time


def main():
    """Main function for Wan image-to-video generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Wan Image-to-Video Generation")
    parser.add_image_input_args()  # No default image, will use astronaut.jpg as fallback
    parser.add_video_args(default_num_frames=81, default_fps=16)

    # Set defaults for Wan I2V
    parser.set_defaults(
        model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        prompt="An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
        height=480,
        width=832,  # Will be auto-configured based on model
        num_warmup_steps=2,  # wan2.2 has 2 transformers, so we need to warmup 2 steps to ensure all the transformers are warmed up
    )
    args = parser.parse_args()

    if "Wan2.2" in args.model_path:
        args.torch_compile_models = "transformer,transformer_2"

    enable_autotuner = False
    if args.linear_type == "auto" or args.attn_type == "auto":
        enable_autotuner = True
        if not args.disable_torch_compile:
            logger.warning("Disable torch compile when using autotuner")
            args.disable_torch_compile = True
        if args.enable_async_cpu_offload:
            logger.warning("Disable visual_gen cpu offload when using autotuner")
            args.enable_async_cpu_offload = False

    # Validate configuration
    validate_parallel_config(args)

    # Generate output path
    output_path = generate_output_path(args, save_type="mp4")

    # Load pipeline
    pipe = load_and_setup_pipeline(args)

    """Load image and calculate optimal dimensions."""
    image = load_image(args.image)
    max_area = args.height * args.width
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio))
    width = round(np.sqrt(max_area / aspect_ratio))

    args.height, args.width = recompute_shape_for_vae(
        height, width, pipe.vae_scale_factor_spatial, pipe.transformer.config.patch_size[1], args.parallel_vae_split_dim
    )

    image = image.resize((args.width, args.height))

    # Run inference
    frames, elapsed_time = run_inference(pipe, args, image, enable_autotuner)

    # Log results and save output
    log_args_and_timing(args, elapsed_time)
    save_output(frames, output_path, output_type="video", fps=16)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
