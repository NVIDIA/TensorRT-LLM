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
import json
import os
import sys
from pathlib import Path

import torch
import transformers

# Add the parent directory to the path to import tensorrt_llm
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tensorrt_llm.models import GLM4MoEForCausalLM
from tensorrt_llm.models.chatglm.config import ChatGLMConfig
from tensorrt_llm.models.chatglm.convert import load_weights_from_hf_model


def convert_checkpoint(args):
    """Convert a GLM4 MoE checkpoint from HuggingFace format to TensorRT-LLM format."""
    
    print(f"Loading GLM4 MoE model from {args.model_dir}")
    
    # Load the model configuration
    config = ChatGLMConfig.from_hugging_face(
        args.model_dir,
        dtype=args.dtype,
        mapping=args.mapping,
        quant_config=args.quant_config,
        chatglm_version='glm4_moe',
        trust_remote_code=True
    )
    
    # Set MoE configuration
    from tensorrt_llm.layers.moe import MoeConfig
    moe_config = MoeConfig(
        num_experts=getattr(config, 'num_experts', 8),
        top_k=getattr(config, 'num_experts_per_tok', 2),
        normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        interleave_moe_layer_step=getattr(config, 'interleave_moe_layer_step', 2)
    )
    config.moe_config = moe_config
    
    print(f"Model configuration: {config}")
    print(f"MoE configuration: {moe_config}")
    
    # Load the HuggingFace model
    hf_model = transformers.AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto' if not args.load_model_on_cpu else 'cpu'
    )
    
    # Convert weights
    weights = load_weights_from_hf_model(hf_model, config)
    
    # Create the TensorRT-LLM model
    model = GLM4MoEForCausalLM(config)
    model.load(weights)
    
    # Save the converted model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving converted model to {output_dir}")
    model.save(output_dir)
    
    # Save the configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    print("Conversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert GLM4 MoE checkpoint from HuggingFace to TensorRT-LLM")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to the HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save the converted TensorRT-LLM model")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Data type for the model (auto, float16, bfloat16, float32)")
    parser.add_argument("--load_model_on_cpu", action="store_true",
                       help="Load the model on CPU instead of GPU")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Pipeline parallel size")
    parser.add_argument("--expert_parallel_size", type=int, default=1,
                       help="Expert parallel size")
    
    args = parser.parse_args()
    
    # Set up mapping
    from tensorrt_llm.mapping import Mapping
    args.mapping = Mapping(
        world_size=args.tensor_parallel_size * args.pipeline_parallel_size,
        tp_size=args.tensor_parallel_size,
        pp_size=args.pipeline_parallel_size,
        ep_size=args.expert_parallel_size
    )
    
    # Set up quantization config (if needed)
    args.quant_config = None
    
    convert_checkpoint(args)


if __name__ == "__main__":
    main() 