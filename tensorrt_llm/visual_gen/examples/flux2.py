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
import visual_gen
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
from visual_gen.layers import apply_visual_gen_linear
from visual_gen.models.transformers.flux2_transformer import ditFlux2Transformer2DModel
from visual_gen.pipelines.flux2_pipeline import ditFlux2Pipeline
from visual_gen.utils import get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Flux pipeline."""
    if args.enable_cuda_graph:
        assert args.attn_type != "sage-attn", "sage-attn has accuracy issue when enable cudagraph"
        assert not (
            args.enable_async_cpu_offload
            or args.enable_sequential_cpu_offload
            or args.enable_model_cpu_offload
        ), "CudaGraph is not supported when using cpu offload"

    # Create dit configuration
    dit_configs = create_dit_config(args)
    # Apply dit configuration
    visual_gen.setup_configs(**dit_configs)
    # Load pipe
    transformer = ditFlux2Transformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    pipe = ditFlux2Pipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, **dit_configs
    )

    if args.linear_recipe == "dynamic":
        assert args.linear_type != "flashinfer-nvfp4-cutlass", (
            "Linear type must be flashinfer-nvfp4-cutlass if linear_recipe=dynamic"
        )

        exclude_pattern = r".*(embedder|norm_out|proj_out|to_add_out|to_added_qkv|stream).*"
        apply_visual_gen_linear(
            pipe.transformer,
            load_parameters=True,
            quantize_weights=True,
            exclude_pattern=exclude_pattern,
        )

    if args.enable_cuda_graph:
        pipe.enable_cuda_graph()

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Flux CPU offload configuration (no text encoder offloading for Flux)
    model_wise = None  # Flux doesn't offload text encoder
    block_wise = ["transformer"]
    configure_cpu_offload(pipe, args, local_rank, model_wise=model_wise, block_wise=block_wise)

    return pipe


def run_inference(pipe, args, enable_autotuner: bool = False):
    """Run warmup and actual inference."""

    def inference_fn(warmup: bool = False):
        if warmup:
            num_inference_steps = 1  # Single step for warmup
        else:
            num_inference_steps = args.num_inference_steps
        return pipe(
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
        ).images[0]

    autotuner_dir = args.autotuner_result_dir
    if enable_autotuner and not args.skip_autotuning:
        if autotuner_dir is None:
            autotuner_dir = generate_autotuner_dir(args)
        autotuning(inference_fn, autotuner_dir)

    image, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=True,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return image, elapsed_time


def main():
    """Main function for Flux text-to-image generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Flux Text-to-Image Generation")

    # Set defaults for Flux
    parser.set_defaults(
        model_path="black-forest-labs/FLUX.2-dev",
        prompt="dog dancing near the sun",
        height=1024,
        width=1024,
        guidance_scale=4,
        enable_cuda_graph=True,
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

    # Validate configuration
    validate_parallel_config(args)

    # Generate output path
    output_path = generate_output_path(args, save_type="png")

    # Load pipeline
    pipe = load_and_setup_pipeline(args)

    # Run inference
    image, elapsed_time = run_inference(pipe, args, enable_autotuner)

    # Log results and save output
    log_args_and_timing(args, elapsed_time)
    save_output(image, output_path, output_type="image")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
