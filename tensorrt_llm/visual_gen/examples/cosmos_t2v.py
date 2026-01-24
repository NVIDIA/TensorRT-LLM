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

import json
import os
import random

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
from visual_gen import setup_configs
from visual_gen.models.transformers.cosmos_transformer import ditCosmosTransformer3DModel
from visual_gen.pipelines.cosmos_pipeline import ditCosmosTextToWorldPipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Cosmos pipeline."""
    # Create dit configuration
    pipe_cfg = create_dit_config(args)
    setup_configs(**pipe_cfg)

    logger.info("Loading ditCosmosTextToWorldPipeline...")
    transformer = ditCosmosTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = ditCosmosTextToWorldPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, **pipe_cfg
    )

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Cosmos CPU offload configuration (no specific model_wise/block_wise needed)
    configure_cpu_offload(pipe, args, local_rank)

    return pipe


def run_inference(pipe, args, enable_autotuner: bool = False):
    """Run warmup and actual inference."""

    def inference_fn(warmup: bool = False):
        if warmup:
            num_inference_steps = args.num_warmup_steps
        else:
            num_inference_steps = args.num_inference_steps
        return pipe(
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

    warmup_bool = False if args.multiple_prompts else True
    frames, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=warmup_bool,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return frames, elapsed_time


def run_multiple_prompts(pipe, args, enable_autotuner: bool = False):
    """Run inference for multiple prompts and save inference information for vbench evaluation."""
    # Generate output path
    output_file = args.output_path
    assert os.path.isdir(output_file), "Output path is not a folder or does not exist!"

    # load prompts from json or txt file
    assert args.prompt.endswith((".json", ".txt")), (
        "Invalid prompt format, the prompt file should be a json or txt file."
    )
    with open(args.prompt, "r") as f:
        if args.prompt.endswith(".json"):
            prompts = json.load(f)
        else:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
        assert isinstance(prompts, list), "Invalid prompt format"

    # check if the prompt file is the vbench standard prompts
    vbench_standard_list = [
        "subject_consistency.json",
        "temporal_flickering.json",
        "object_class.json",
        "multiple_objects.json",
        "human_action.json",
        "color.json",
        "spatial_relationship.json",
        "scene.json",
        "temporal_style.json",
        "appearance_style.json",
        "overall_consistency.json",
        "VBench_standard_prompts.json",
    ]

    # generate videos for 2 categories: vbench standard prompts and custom prompts
    output_info = {}
    if args.prompt.endswith(tuple(vbench_standard_list)):
        # if use the vbech standard prompts, generate 5 videos for each prompt and use prompt as video name
        for prompt in prompts:
            seed_list = random.sample(range(0, 100), 4)
            seed_list = [args.random_seed] + seed_list
            args.prompt = prompt

            for idx, seed in enumerate(seed_list):
                args.random_seed = seed
                frames, elapsed_time = run_inference(pipe, args, enable_autotuner)
                log_args_and_timing(args, elapsed_time)
                output_path = os.path.join(output_file, f"{prompt}-{idx}.mp4")
                save_output(frames, output_path, output_type="video", fps=args.fps)
                output_info[f"{prompt}-{idx}.mp4"] = prompt
                print(f"Saved video to {output_path}")

        # save output_info for later evaluation
        with open(os.path.join(output_file, "output_info.json"), "w") as f:
            json.dump(output_info, f)

    else:
        #  if use custom prompts, generate one video for each prompt and save output_info for later evaluation
        for idx, prompt in enumerate(prompts):
            args.prompt = prompt
            frames, elapsed_time = run_inference(pipe, args, enable_autotuner)
            log_args_and_timing(args, elapsed_time)
            output_path = os.path.join(output_file, f"{idx}.mp4")
            save_output(frames, output_path, output_type="video", fps=args.fps)
            output_info[f"{idx}.mp4"] = prompt
            print(f"Saved video {idx} to {output_path}")

        # save output_info for later evaluation
        with open(os.path.join(output_file, "output_info.json"), "w") as f:
            json.dump(output_info, f)


def main():
    """Main function for Cosmos text-to-image generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Cosmos Text-to-Image Generation")
    parser.add_video_args(default_num_frames=121, default_fps=30)

    # Set defaults for Cosmos T2V
    parser.parser.set_defaults(
        model_path="nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
        prompt="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect.",
        negative_prompt=None,
        height=704,
        width=1280,
        num_inference_steps=36,
        guidance_scale=7.0,
        max_sequence_length=512,
        teacache_thresh=0.3,
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
        logger.warning("Parallel VAE is not supported for Cosmos T2V, disable parallel VAE")
        args.disable_parallel_vae = True
        # args.height, args.width = recompute_shape_for_vae(height, width, pipe.vae_scale_factor_spatial, pipe.transformer.config.patch_size[1], args.parallel_vae_split_dim)

    # Validate configuration
    validate_parallel_config(args)

    # Load pipeline and prepare inputs
    pipe = load_and_setup_pipeline(args)

    if not args.multiple_prompts:
        # Generate output path
        output_path = generate_output_path(args, save_type="mp4")

        # Run inference
        frames, elapsed_time = run_inference(pipe, args, enable_autotuner)

        # Log results and save output
        log_args_and_timing(args, elapsed_time)
        save_output(frames, output_path, output_type="video", fps=args.fps)

    else:
        run_multiple_prompts(pipe, args, enable_autotuner)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
