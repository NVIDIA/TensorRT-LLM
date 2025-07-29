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

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import tensorrt_llm
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tensorrt_llm import LLM


def main():
    parser = argparse.ArgumentParser(description="Run inference with GLM4 MoE model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the TensorRT-LLM model directory")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to the tokenizer")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--backend", type=str, default="torch",
                       help="Backend to use (torch, trt)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--expert_parallel_size", type=int, default=1,
                       help="Expert parallel size")
    parser.add_argument("--moe_backend", type=str, default="CUTLASS",
                       help="MoE backend to use")
    
    args = parser.parse_args()
    
    print(f"Loading GLM4 MoE model from {args.model}")
    print(f"Using backend: {args.backend}")
    print(f"MoE backend: {args.moe_backend}")
    
    # Initialize the model
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        moe_backend=args.moe_backend
    )
    
    print(f"Generating response for prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print("-" * 50)
    
    # Generate response
    output = llm.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("Generated response:")
    print(output)
    print("-" * 50)
    
    # Example with streaming
    print("Streaming generation:")
    for token in llm.generate_streaming(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    ):
        print(token, end="", flush=True)
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main() 