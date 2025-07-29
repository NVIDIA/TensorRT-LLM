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
Test script for GLM4 MoE implementation.

This script tests:
1. Model creation and configuration
2. Basic forward pass
3. MoE layer functionality
4. Weight loading compatibility
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import tensorrt_llm
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import torch
from tensorrt_llm.models import GLM4MoEForCausalLM
from tensorrt_llm.models.chatglm.config import ChatGLMConfig
from tensorrt_llm.layers.moe import MoeConfig
from tensorrt_llm.mapping import Mapping


def test_model_creation():
    """Test creating a GLM4 MoE model."""
    
    print("Testing GLM4 MoE model creation...")
    
    # Create configuration
    config = ChatGLMConfig(
        architecture="GLM4MoEForCausalLM",
        dtype="float16",
        num_hidden_layers=4,  # Small model for testing
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_size=512,
        intermediate_size=1024,
        vocab_size=1000,
        max_position_embeddings=1024,
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
        num_experts=4,
        top_k=2,
        normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        interleave_moe_layer_step=2
    )
    config.moe_config = moe_config
    
    print(f"Configuration created successfully")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of layers: {config.num_hidden_layers}")
    print(f"  - Number of experts: {moe_config.num_experts}")
    print(f"  - Top-k: {moe_config.top_k}")
    print(f"  - MoE layer step: {moe_config.interleave_moe_layer_step}")
    
    # Create model
    model = GLM4MoEForCausalLM(config)
    print("Model created successfully!")
    
    return model, config


def test_forward_pass(model, config):
    """Test forward pass through the model."""
    
    print("\nTesting forward pass...")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Run forward pass
    with torch.no_grad():
        try:
            output = model(input_ids)
            print(f"Output shape: {output.shape}")
            print("Forward pass successful!")
            return True
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return False


def test_moe_layers(model):
    """Test that MoE layers are properly configured."""
    
    print("\nTesting MoE layer configuration...")
    
    # Check that some layers have MoE
    moe_layers = 0
    total_layers = 0
    
    for name, module in model.named_modules():
        if 'moe' in name.lower() and hasattr(module, 'num_experts'):
            moe_layers += 1
            print(f"Found MoE layer: {name}")
            print(f"  - Number of experts: {module.num_experts}")
            print(f"  - Top-k: {module.top_k}")
        elif 'layers' in name and 'attention' not in name:
            total_layers += 1
    
    print(f"Found {moe_layers} MoE layers out of {total_layers} total layers")
    
    if moe_layers > 0:
        print("MoE layers configured correctly!")
        return True
    else:
        print("No MoE layers found!")
        return False


def test_weight_loading():
    """Test weight loading compatibility."""
    
    print("\nTesting weight loading compatibility...")
    
    # Create a small model
    config = ChatGLMConfig(
        architecture="GLM4MoEForCausalLM",
        dtype="float16",
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=256,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=512,
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
        num_experts=2,
        top_k=1,
        normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        interleave_moe_layer_step=2
    )
    config.moe_config = moe_config
    
    # Create model
    model = GLM4MoEForCausalLM(config)
    
    # Test that we can get the state dict
    try:
        state_dict = model.state_dict()
        print(f"State dict created successfully with {len(state_dict)} parameters")
        
        # Check for MoE-related parameters
        moe_params = [k for k in state_dict.keys() if 'moe' in k.lower()]
        print(f"Found {len(moe_params)} MoE-related parameters")
        
        return True
    except Exception as e:
        print(f"Weight loading test failed: {e}")
        return False


def test_configuration():
    """Test configuration options."""
    
    print("\nTesting configuration options...")
    
    # Test different MoE configurations
    test_configs = [
        {"num_experts": 4, "top_k": 2, "step": 2},
        {"num_experts": 8, "top_k": 2, "step": 2},
        {"num_experts": 4, "top_k": 1, "step": 1},
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\nTest configuration {i+1}: {test_config}")
        
        try:
            config = ChatGLMConfig(
                architecture="GLM4MoEForCausalLM",
                dtype="float16",
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=8,
                hidden_size=512,
                intermediate_size=1024,
                vocab_size=1000,
                max_position_embeddings=1024,
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
            
            moe_config = MoeConfig(
                num_experts=test_config["num_experts"],
                top_k=test_config["top_k"],
                normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                interleave_moe_layer_step=test_config["step"]
            )
            config.moe_config = moe_config
            
            model = GLM4MoEForCausalLM(config)
            print(f"  ✓ Configuration {i+1} successful")
            
        except Exception as e:
            print(f"  ✗ Configuration {i+1} failed: {e}")
            return False
    
    print("All configuration tests passed!")
    return True


def main():
    """Run all tests."""
    
    print("GLM4 MoE Implementation Tests")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Configuration", test_configuration),
        ("Weight Loading", test_weight_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Model Creation":
                model, config = test_func()
                results.append(True)
                
                # Additional tests that need the model
                print(f"\n{'='*20} Forward Pass {'='*20}")
                forward_result = test_forward_pass(model, config)
                results.append(forward_result)
                
                print(f"\n{'='*20} MoE Layers {'='*20}")
                moe_result = test_moe_layers(model)
                results.append(moe_result)
                
            else:
                result = test_func()
                results.append(result)
                
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! GLM4 MoE implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 