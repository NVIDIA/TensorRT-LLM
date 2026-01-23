# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark NVFP4 MoE kernel with CUDA Graph support.

Usage:
    # Profile and export to SQLite for analysis
    nsys profile -t cuda,nvtx -o report --force-overwrite true --export=sqlite \
        python bench_nvfp4_moe.py --moe_backend CUTEDSL --seq_len 128 --enable_cudagraph

    # Parse the report to extract kernel times within benchmark range
    python parse_nsys_report.py report.sqlite

    # With CUDA graph trace for detailed kernel analysis
    nsys profile -t cuda,nvtx -o report --cuda-graph-trace=node \
        --force-overwrite true --export=sqlite \
        python bench_nvfp4_moe.py --moe_backend CUTEDSL --enable_cudagraph

The benchmark iterations are marked with "benchmark" NVTX range.
Use parse_nsys_report.py to extract kernel times within this range.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parents[3]))
sys.path.insert(0, str(Path(__file__).parents[3] / "tests/unittest"))

from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import DeepSeekV3MoeRoutingMethod, create_moe
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


class CUDAGraphRunner:
    """Wrapper for running model inference with CUDA Graph."""

    def __init__(self, model, x: torch.Tensor, router_logits: torch.Tensor):
        self.model = model
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Static input tensors for CUDA Graph
        self.static_x = x.clone()
        self.static_router_logits = router_logits.clone()
        self.static_output: Optional[torch.Tensor] = None

    def capture(self, warmup_runs: int = 3):
        """Capture CUDA Graph after warmup runs."""
        # Warmup runs (required before capture)
        for _ in range(warmup_runs):
            _ = self.model.forward(self.static_x, self.static_router_logits)
        torch.cuda.synchronize()

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model.forward(self.static_x, self.static_router_logits)

    def replay(self) -> torch.Tensor:
        """Replay the captured CUDA Graph."""
        assert self.graph is not None, "Graph not captured yet"
        self.graph.replay()
        return self.static_output

    def update_inputs(self, x: torch.Tensor, router_logits: torch.Tensor):
        """Update static inputs (must have same shape)."""
        self.static_x.copy_(x)
        self.static_router_logits.copy_(router_logits)


def create_nvfp4_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    x_sf_global: torch.Tensor,
):
    """Create NVFP4 quantized weights for MoE."""
    SCALING_VECTOR_SIZE = 16
    weights = {}

    for expert_id in range(num_experts):
        w1_weight = torch.randn((intermediate_size, hidden_size), dtype=dtype, device="cuda") * 0.05
        w1_sf_global = (448 * 6) / w1_weight.abs().max().float()

        w2_weight = torch.randn((hidden_size, intermediate_size), dtype=dtype, device="cuda") * 0.05
        w2_sf_global = (448 * 6) / w2_weight.abs().max().float()

        w3_weight = torch.randn((intermediate_size, hidden_size), dtype=dtype, device="cuda") * 0.05
        w3_sf_global = (448 * 6) / w3_weight.abs().max().float()

        # w3 global and w1 global must be the same
        w3_w1_global = min(w1_sf_global, w3_sf_global)

        w1_weight_nvfp4, w1_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
            w1_weight, w3_w1_global, SCALING_VECTOR_SIZE, False, False
        )
        w1_sf_block_unswizzled = w1_sf_block_unswizzled.view(intermediate_size, -1)

        w2_weight_nvfp4, w2_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
            w2_weight, w2_sf_global, SCALING_VECTOR_SIZE, False, False
        )
        w2_sf_block_unswizzled = w2_sf_block_unswizzled.view(hidden_size, -1)

        w3_weight_nvfp4, w3_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
            w3_weight, w3_w1_global, SCALING_VECTOR_SIZE, False, False
        )
        w3_sf_block_unswizzled = w3_sf_block_unswizzled.view(intermediate_size, -1)

        w1_input_scale = x_sf_global.cuda()
        w2_input_scale = x_sf_global.cuda()
        w3_input_scale = x_sf_global.cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
        weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
        weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
        weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
        weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
        weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
        weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
        weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
        weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global

    return weights


def run_benchmark(args):
    """Run NVFP4 MoE benchmark."""
    # Validate backend and SM version
    sm_version = get_sm_version()
    if args.moe_backend in ["CUTEDSL", "DENSEGEMM"]:
        if sm_version not in (100, 103):
            print(
                f"Warning: {args.moe_backend} NVFP4 MoE backend supports SM 100 (B200) and SM 103 (B300) only"
            )
            print(f"Current SM version: {sm_version}")
            if not args.force:
                print("Use --force to run anyway")
                return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 60)
    print("NVFP4 MoE Benchmark Configuration")
    print("=" * 60)
    print(f"  Backend:           {args.moe_backend}")
    print(f"  Hidden size:       {args.hidden_size}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Num experts:       {args.num_experts}")
    print(f"  Top-K:             {args.top_k}")
    print(f"  N-Group:           {args.n_group}")
    print(f"  TopK-Group:        {args.topk_group}")
    print(f"  Routed scaling:    {args.routed_scaling_factor}")
    print(f"  Sequence length:   {args.seq_len}")
    print(f"  Dtype:             {args.dtype}")
    print(f"  CUDA Graph:        {'enabled' if args.enable_cudagraph else 'disabled'}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.iterations}")
    print(f"  SM version:        {sm_version}")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with torch.device("cuda"):
        # Create input tensors
        x = torch.randn((args.seq_len, args.hidden_size), dtype=dtype, device="cuda")
        x_sf_global = (448 * 6) / x.abs().max().float()
        # router_logits must be float32 for TRTLLM backend
        router_logits = torch.randn(
            (args.seq_len, args.num_experts), dtype=torch.float32, device="cuda"
        )

        # Create weights
        print("\nCreating NVFP4 quantized weights...")
        weights = create_nvfp4_weights(
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            dtype=dtype,
            x_sf_global=x_sf_global,
        )

        # Create MoE model with DeepSeek R1 routing method
        # DeepSeek R1 parameters: n_group=8, topk_group=4, routed_scaling_factor=2.5
        # routing_bias must be bfloat16 for TRTLLM backend
        e_score_correction_bias = torch.zeros(args.num_experts, dtype=dtype, device="cuda")
        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=args.top_k,
            n_group=args.n_group,
            topk_group=args.topk_group,
            routed_scaling_factor=args.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )
        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = args.num_experts
        pretrained_config.hidden_size = args.hidden_size
        pretrained_config.intermediate_size = args.intermediate_size
        pretrained_config.torch_dtype = dtype

        fused_moe = create_moe(
            routing_method=routing_method,
            reduce_results=True,
            model_config=ModelConfig(
                pretrained_config=pretrained_config,
                quant_config=quant_config,
                moe_backend=args.moe_backend,
                moe_disable_finalize_fusion=not args.finalize_fusion,
            ),
            bias=False,
        )
        fused_moe.load_weights([weights])
        fused_moe.post_load_weights()
        fused_moe.cuda()

        # Run autotune
        print("\nRunning autotune...")
        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune():
            fused_moe.forward(x, router_logits)
        print("Autotune completed.")

        # Setup CUDA Graph if enabled
        cuda_graph_runner = None
        if args.enable_cudagraph:
            print("\nCapturing CUDA Graph...")
            with torch.inference_mode():
                cuda_graph_runner = CUDAGraphRunner(fused_moe, x, router_logits)
                cuda_graph_runner.capture(warmup_runs=3)
            print("CUDA Graph captured.")

        # Define forward function
        def forward_fn():
            if cuda_graph_runner is not None:
                return cuda_graph_runner.replay()
            else:
                return fused_moe.forward(x, router_logits)

        # Warmup
        print(f"\nWarmup ({args.warmup} iterations)...")
        with torch.inference_mode():
            for _ in range(args.warmup):
                forward_fn()
        torch.cuda.synchronize()

        # Benchmark for nsys profiling
        print(f"Running benchmark ({args.iterations} iterations)...")
        print("Look for 'benchmark' NVTX range in nsys result to see benchmark kernels.")

        # Ensure all previous operations are complete
        torch.cuda.synchronize()

        # NVTX range for nsys capture
        torch.cuda.nvtx.range_push("benchmark")

        for i in range(args.iterations):
            forward_fn()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        print("\nBenchmark completed. Analyze results with nsys.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 MoE kernel with CUDA Graph support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--moe_backend",
        type=str,
        default="CUTEDSL",
        choices=["TRTLLM", "CUTLASS", "CUTEDSL", "DENSEGEMM"],
        help="MoE backend to use",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=7168,
        help="Hidden size (DeepSeek R1: 7168)",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=2048,
        help="Intermediate size (FFN dimension, DeepSeek R1: 2048)",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=256,
        help="Number of experts (DeepSeek R1: 256)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Top-K experts to route to (DeepSeek R1: 8)",
    )
    parser.add_argument(
        "--n_group",
        type=int,
        default=8,
        help="Number of expert groups for DeepSeek routing (DeepSeek R1: 8)",
    )
    parser.add_argument(
        "--topk_group",
        type=int,
        default=4,
        help="Top-K groups for DeepSeek routing (DeepSeek R1: 4)",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        type=float,
        default=2.5,
        help="Routed scaling factor for DeepSeek routing (DeepSeek R1: 2.5)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length (number of tokens)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for computation",
    )
    parser.add_argument(
        "--finalize_fusion",
        action="store_true",
        help="Enable finalize fusion",
    )

    # Benchmark configuration
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # CUDA Graph
    parser.add_argument(
        "--enable_cudagraph",
        action="store_true",
        help="Enable CUDA Graph for inference",
    )

    # Misc
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if SM version is not supported",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
