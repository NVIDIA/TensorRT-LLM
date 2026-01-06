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
from huggingface_hub import hf_hub_download

from visual_gen import setup_configs
from visual_gen.models.transformers.flux_transformer import ditFluxTransformer2DModel
from visual_gen.pipelines.flux_pipeline import ditFluxPipeline
from visual_gen.utils import cudagraph_wrapper, get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Flux pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    # Load transformer
    transformer = ditFluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # Create pipeline
    pipe = ditFluxPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, **dit_configs
    )

    if args.linear_type == "svd-nvfp4" or args.linear_type == "auto":
        if args.svd_fp4_checkpoint_path == "":
            hf_hub_download(
                repo_id="mit-han-lab/nunchaku-flux.1-dev",
                filename="svdq-fp4_r32-flux.1-dev.safetensors",
                local_dir="./svd_fp4_checkpoint",
            )
            svd_fp4_checkpoint_path = "./svd_fp4_checkpoint/svdq-fp4_r32-flux.1-dev.safetensors"
        else:
            svd_fp4_checkpoint_path = args.svd_fp4_checkpoint_path
        svd_weight_name_table = {
            "attn.to_qkv": "qkv_proj",
            "attn.to_added_qkv": "qkv_proj_context",
            "attn.to_out.0": "out_proj",
            "attn.to_add_out": "out_proj_context",
            "ff.net.0.proj": "mlp_fc1",
            "ff.net.2": "mlp_fc2",
            "ff_context.net.0.proj": "mlp_context_fc1",
            "ff_context.net.2": "mlp_context_fc2",
            "proj_out0": "out_proj",
            "proj_mlp": "mlp_fc1",
            "proj_out1": "mlp_fc2",
        }
        pipe.load_fp4_weights(svd_fp4_checkpoint_path, svd_weight_name_table)

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
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=args.max_sequence_length,
        ).images[0]

    autotuner_dir = args.autotuner_result_dir
    if enable_autotuner and not args.skip_autotuning:
        if autotuner_dir is None:
            autotuner_dir = generate_autotuner_dir(args)
        autotuning(inference_fn, autotuner_dir)

    if args.enable_cuda_graph:
        assert args.attn_type != "sage-attn", "sage-attn has accuracy issue when enable cudagraph"
        assert not (
            args.enable_async_cpu_offload or args.enable_sequential_cpu_offload or args.enable_model_cpu_offload
        ), "CudaGraph is not supported when using cpu offload"
        pipe.transformer.forward = cudagraph_wrapper(pipe.transformer.forward)

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
        model_path="black-forest-labs/FLUX.1-dev",
        prompt="A cat holding a sign that says hello world",
        negative_prompt="",  # Flux typically doesn't use negative prompts
        height=1024,
        width=1024,
        guidance_scale=3.5,
        teacache_thresh=0.6,
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
