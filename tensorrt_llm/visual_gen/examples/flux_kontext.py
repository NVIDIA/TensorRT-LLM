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
from PIL import Image
from visual_gen import setup_configs
from visual_gen.models.transformers.flux_transformer import ditFluxTransformer2DModel
from visual_gen.pipelines.flux_kontext_pipeline import ditFluxKontextPipeline
from visual_gen.utils import cudagraph_wrapper, get_logger

logger = get_logger(__name__)


def load_and_setup_pipeline(args):
    """Load and configure the Flux Kontext pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    # Load transformer
    transformer = ditFluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # Create pipeline
    pipe = ditFluxKontextPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, **dit_configs
    )

    # Setup distributed training and CPU offload
    local_rank, world_size = setup_distributed()

    # Flux CPU offload configuration
    model_wise = None
    block_wise = ["transformer"]

    configure_cpu_offload(pipe, args, local_rank, model_wise=model_wise, block_wise=block_wise)

    return pipe


def run_inference(pipe, args, image, enable_autotuner: bool = False):
    """Run warmup and actual inference."""

    def inference_fn(warmup: bool = False):
        if warmup:
            num_inference_steps = 1
        else:
            num_inference_steps = args.num_inference_steps
        return pipe(
            prompt=args.prompt,
            image=image,
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
            args.enable_async_cpu_offload
            or args.enable_sequential_cpu_offload
            or args.enable_model_cpu_offload
        ), "CudaGraph is not supported when using cpu offload"

        # === Text Encoders (prompt encode) ===
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.forward = cudagraph_wrapper(pipe.text_encoder.forward)
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.forward = cudagraph_wrapper(pipe.text_encoder_2.forward)

        # === VAE Decode ===
        if hasattr(pipe, "run_vae_decode"):
            pipe.run_vae_decode = cudagraph_wrapper(pipe.run_vae_decode)

        # === Transformer ===
        if args.enable_teacache:
            pipe.transformer.run_pre_processing = cudagraph_wrapper(
                pipe.transformer.run_pre_processing
            )
            pipe.transformer.run_teacache_check = cudagraph_wrapper(
                pipe.transformer.run_teacache_check
            )
            pipe.transformer.run_transformer_blocks = cudagraph_wrapper(
                pipe.transformer.run_transformer_blocks
            )
            pipe.transformer.run_post_processing = cudagraph_wrapper(
                pipe.transformer.run_post_processing
            )
        else:
            pipe.transformer.forward = cudagraph_wrapper(pipe.transformer.forward)

    result, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=True,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return result, elapsed_time


def main():
    """Main function for Flux Kontext."""
    # Setup argument parser
    parser = BaseArgumentParser("Flux Kontext (FLUX.1-Kontext)")

    # Flux Kontext specific arguments
    parser.parser.add_argument(
        "--image", type=str, required=True, help="Path to the reference image"
    )

    # Set defaults for Flux Kontext
    parser.set_defaults(
        model_path="black-forest-labs/FLUX.1-Kontext-dev",
        prompt="The same scene with different lighting",
        negative_prompt="",
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

    # Load reference image
    logger.info(f"Loading reference image from: {args.image}")
    image = Image.open(args.image).convert("RGB")

    logger.info(f"Image size: {args.width}x{args.height}")
    logger.info(f"Prompt: {args.prompt}")

    # Generate output path
    output_path = generate_output_path(args, save_type="png")

    # Load pipeline
    pipe = load_and_setup_pipeline(args)

    # Run inference
    result, elapsed_time = run_inference(pipe, args, image, enable_autotuner)

    # Log results and save output
    log_args_and_timing(args, elapsed_time)
    save_output(result, output_path, output_type="image")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
