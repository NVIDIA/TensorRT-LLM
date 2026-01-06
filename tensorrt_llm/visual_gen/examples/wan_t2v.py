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
    recompute_shape_for_vae,
    save_output,
    setup_distributed,
    validate_parallel_config,
)

from visual_gen import setup_configs
from visual_gen.models.transformers.wan_transformer import ditWanTransformer3DModel
from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL
from visual_gen.pipelines.wan_pipeline import ditWanPipeline
from visual_gen.utils.load_export_ckpt import export_quantized_checkpoint, load_quantized_checkpoint
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Wan pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    # Load components
    vae = ditWanAutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    if args.load_visual_gen_dit:
        transformer = load_quantized_checkpoint(ditWanTransformer3DModel, args.visual_gen_ckpt_path, torch_dtype=torch.bfloat16)
    else:
        transformer = ditWanTransformer3DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )

    # Create pipeline
    pipe = ditWanPipeline.from_pretrained(
        args.model_path, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16, **dit_configs
    )

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Wan T2V CPU offload configuration
    model_wise = ["text_encoder"]
    block_wise = ["transformer"]
    if "Wan2.2" in args.model_path:
        block_wise.append("transformer_2")

    configure_cpu_offload(pipe, args, local_rank, model_wise=model_wise, block_wise=block_wise)

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

    warmup_bool = False if args.multiple_prompts else True
    frames, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=warmup_bool,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
        deterministic=False,  # torch compile can't work with deterministic=True in Wan2.2
    )
    return frames, elapsed_time


def run_multiple_prompts(pipe, args, enable_autotuner: bool = False):
    """Run inference for multiple prompts and save inference information for vbench evaluation."""
    # Generate output path
    output_file = args.output_path
    assert os.path.isdir(output_file), "Output path is not a folder or does not exist!"

    # load prompts from json or txt file
    assert args.prompt.endswith(
        (".json", ".txt")
    ), "Invalid prompt format, the prompt file should be a json or txt file."
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
                save_output(frames, output_path, output_type="video", fps=16)
                output_info[f"{prompt}-{idx}.mp4"] = prompt

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
            save_output(frames, output_path, output_type="video", fps=16)
            output_info[f"{idx}.mp4"] = prompt

        # save output_info for later evaluation
        with open(os.path.join(output_file, "output_info.json"), "w") as f:
            json.dump(output_info, f)


def main():
    """Main function for Wan text-to-video generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Wan Text-to-Video Generation")
    parser.add_video_args(default_num_frames=33, default_fps=16)  # 1.3B defaults

    # Set defaults for Wan T2V
    parser.set_defaults(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        height=480,
        width=832,
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

    if args.export_visual_gen_dit:
        logger.info("Exporting quantized DIT...")
        # Create dit configuration
        dit_configs = create_dit_config(args)
        setup_configs(**dit_configs)

        args.disable_torch_compile = True  # keep origin param names
        args.disable_qkv_fusion = True
        transformer = ditWanTransformer3DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        if args.linear_type == "auto":
            raise NotImplementedError("Dont support export quantized DIT when linear_type is auto")
        export_quantized_checkpoint(transformer, args.visual_gen_ckpt_path)
        return

    # Validate configuration
    validate_parallel_config(args)

    # Load pipeline
    pipe = load_and_setup_pipeline(args)

    # Recompute shape for vae
    args.height, args.width = recompute_shape_for_vae(
        args.height,
        args.width,
        pipe.vae_scale_factor_spatial,
        pipe.transformer.config.patch_size[1],
        args.parallel_vae_split_dim,
    )

    if not args.multiple_prompts:
        # Generate output path
        output_path = generate_output_path(args, save_type="mp4")

        # Run inference
        frames, elapsed_time = run_inference(pipe, args, enable_autotuner)

        # Log results and save output
        log_args_and_timing(args, elapsed_time)
        save_output(frames, output_path, output_type="video", fps=16)

    else:
        run_multiple_prompts(pipe, args, enable_autotuner)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
