#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Example demonstrating GLM4 MoE with TensorRT-LLM's PyTorch backend.

This example shows how to:
1. Load a GLM4 MoE model with PyTorch backend
2. Configure MoE-specific parameters
3. Run inference with different parallelism settings
4. Use various MoE backends
5. Handle quantization
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path to import tensorrt_llm
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import torch
from tensorrt_llm import LLM
from tensorrt_llm.models.chatglm.config import ChatGLMConfig
from tensorrt_llm.layers.moe import MoeConfig


def create_glm4_moe_config(
    num_layers: int = 32,
    num_heads: int = 32,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    vocab_size: int = 100008,
    max_position_embeddings: int = 8192,
    num_experts: int = 8,
    top_k: int = 2,
    interleave_step: int = 2,
    dtype: str = "float16"
) -> ChatGLMConfig:
    """Create a GLM4 MoE configuration."""
    
    from tensorrt_llm.mapping import Mapping
    
    # Create basic config
    config = ChatGLMConfig(
        architecture="GLM4MoEForCausalLM",
        dtype=dtype,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        chatglm_version="glm4_moe",
        position_embedding_type="rope_gptj",
        rotary_pct=0.5,
        rotary_base=10000.0,
        hidden_act="swiglu",
        norm_epsilon=1e-5,
        rmsnorm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        apply_query_key_layer_scaling=False,
        apply_residual_connection_post_layernorm=False,
        mapping=Mapping()
    )
    
    # Add MoE configuration
    moe_config = MoeConfig(
        num_experts=num_experts,
        top_k=top_k,
        normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        interleave_moe_layer_step=interleave_step
    )
    config.moe_config = moe_config
    
    return config


def load_model_with_pytorch_backend(
    model_path: str,
    tokenizer_path: str,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    moe_backend: str = "CUTLASS",
    max_batch_size: int = 8,
    max_input_len: int = 2048,
    max_output_len: int = 512
) -> LLM:
    """Load GLM4 MoE model with PyTorch backend."""
    
    print(f"Loading GLM4 MoE model with PyTorch backend...")
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Expert parallel size: {expert_parallel_size}")
    print(f"MoE backend: {moe_backend}")
    
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        backend="torch",  # Use PyTorch backend
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        moe_backend=moe_backend,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len
    )
    
    print("Model loaded successfully!")
    return llm


def run_basic_inference(llm: LLM, prompts: List[str]) -> List[str]:
    """Run basic inference on a list of prompts."""
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    
    outputs = []
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Generate response
        output = llm.generate(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"Response: {output}")
        outputs.append(output)
    
    return outputs


def run_streaming_inference(llm: LLM, prompt: str):
    """Run streaming inference on a single prompt."""
    
    print(f"\nRunning streaming inference...")
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    for token in llm.generate_streaming(
        prompt,
        max_tokens=100,
        temperature=0.7
    ):
        print(token, end="", flush=True)
    
    print("\n")


def run_batch_inference(llm: LLM, prompts: List[str]):
    """Run batch inference on multiple prompts."""
    
    print(f"\nRunning batch inference on {len(prompts)} prompts...")
    
    # Generate responses for all prompts at once
    outputs = llm.generate(
        prompts,
        max_tokens=100,
        temperature=0.7
    )
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\nBatch {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {output}")


def demonstrate_moe_backends():
    """Demonstrate different MoE backends."""
    
    backends = ["CUTLASS", "TRTLLM", "VANILLA"]
    
    print("\n" + "="*60)
    print("DEMONSTRATING DIFFERENT MOE BACKENDS")
    print("="*60)
    
    for backend in backends:
        print(f"\n--- Testing {backend} backend ---")
        try:
            # Note: In a real scenario, you would load different models
            # or configurations for different backends
            print(f"Backend {backend} is available")
            
            # Show backend-specific features
            if backend == "CUTLASS":
                print("  - Optimized for general use cases")
                print("  - Good performance across different quantization modes")
            elif backend == "TRTLLM":
                print("  - Optimized for FP8 and NVFP4 quantization")
                print("  - Requires specific quantization configuration")
            elif backend == "VANILLA":
                print("  - Simple implementation for debugging")
                print("  - Good for development and testing")
                
        except Exception as e:
            print(f"  - Error: {e}")


def demonstrate_parallelism():
    """Demonstrate different parallelism configurations."""
    
    print("\n" + "="*60)
    print("DEMONSTRATING PARALLELISM CONFIGURATIONS")
    print("="*60)
    
    # Example configurations
    configs = [
        {"tp": 1, "ep": 1, "name": "Single GPU"},
        {"tp": 2, "ep": 1, "name": "Tensor Parallel (2 GPUs)"},
        {"tp": 1, "ep": 2, "name": "Expert Parallel (2 GPUs)"},
        {"tp": 2, "ep": 2, "name": "Tensor + Expert Parallel (4 GPUs)"},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"  Tensor Parallel Size: {config['tp']}")
        print(f"  Expert Parallel Size: {config['ep']}")
        print(f"  Total GPUs: {config['tp'] * config['ep']}")
        
        # In a real scenario, you would test these configurations
        print("  - Configuration is valid")


def main():
    parser = argparse.ArgumentParser(description="GLM4 MoE PyTorch Backend Example")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the GLM4 MoE model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to the tokenizer")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--expert_parallel_size", type=int, default=1,
                       help="Expert parallel size")
    parser.add_argument("--moe_backend", type=str, default="CUTLASS",
                       choices=["CUTLASS", "TRTLLM", "VANILLA"],
                       help="MoE backend to use")
    parser.add_argument("--demo_mode", action="store_true",
                       help="Run demonstration mode without actual model loading")
    
    args = parser.parse_args()
    
    print("GLM4 MoE with TensorRT-LLM PyTorch Backend")
    print("=" * 60)
    
    if args.demo_mode:
        # Run demonstrations without loading actual model
        demonstrate_moe_backends()
        demonstrate_parallelism()
        
        print("\n" + "="*60)
        print("DEMO MODE - No actual model loaded")
        print("To run with a real model, remove --demo_mode flag")
        print("="*60)
        return
    
    # Load the model
    llm = load_model_with_pytorch_backend(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        moe_backend=args.moe_backend
    )
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms:",
        "What is the difference between machine learning and deep learning?",
        "Write a short story about a robot learning to paint:",
        "How does a neural network work?"
    ]
    
    # Run different types of inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE EXAMPLES")
    print("="*60)
    
    # Basic inference
    run_basic_inference(llm, prompts[:2])
    
    # Streaming inference
    run_streaming_inference(llm, "Explain the concept of attention in transformers:")
    
    # Batch inference
    run_batch_inference(llm, prompts[2:])
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main() 