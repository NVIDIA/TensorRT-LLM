#!/usr/bin/env python3
"""
Example demonstrating block prediction with TensorRT-LLM.

Block prediction allocates a block of N tokens, all starting as masked tokens,
then runs forward passes with no causal mask, unmasking tokens with softmax 
probabilities greater than a threshold (e.g., 0.8), always unmasking at least one token.
This process repeats until all tokens are unmasked.
"""

import torch
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig


def main():
    # Create a PyTorchConfig with block prediction enabled
    pytorch_config = PyTorchConfig(
        enable_block_prediction=True,
        block_size=8,  # Number of tokens to predict in each block
        keep_threshold=0.8,  # Confidence threshold for keeping tokens
        mask_token_id=151666,  # Token ID to use for masked positions
        max_iterations=10,  # Maximum iterations for block prediction
    )
    
    # Initialize the LLM with block prediction
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use a small model for demo
        backend='pytorch',
        pytorch_backend_config=pytorch_config,
        max_seq_len=2048,
        max_batch_size=4,
        max_num_tokens=4096,
    )
    
    # Sample prompts
    prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, technology will",
        "The most important invention of the 21st century is",
        "Climate change solutions include",
    ]
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.7,
        top_p=0.9,
    )
    
    print("=== Block Prediction Example ===\n")
    print("Configuration:")
    print(f"  Block size: {pytorch_config.block_size}")
    print(f"  Keep threshold: {pytorch_config.keep_threshold}")
    print(f"  Mask token ID: {pytorch_config.mask_token_id}")
    print(f"  Max iterations: {pytorch_config.max_iterations}")
    print()
    
    # Generate text with block prediction
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")
        
        # Generate response
        outputs = llm.generate([prompt], sampling_params)
        
        # Print the generated text
        # Access the first output's text
        output = outputs[0]
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")
        
        # Print block prediction statistics if available
        # Note: Block prediction results are stored internally and may not be directly accessible
        # through the public API in this version
        print(f"  Generated {len(generated_text.split())} words")
        print()
    
    print("=== Block Prediction Process Explanation ===")
    print("""
    Block prediction works as follows:
    
    1. Allocate a block of N tokens (e.g., 8 tokens), all starting as masked tokens
    2. Run a forward pass for all these tokens using existing KV-cache with no causal mask
    3. Compute softmax probabilities for each position
    4. Unmask tokens with probabilities greater than the threshold (e.g., 0.8)
    5. Always unmask at least one token (the one with highest confidence)
    6. Repeat steps 2-5 until all tokens are unmasked
    7. Return the first unmasked token as the next generated token
    
    This approach can potentially generate multiple tokens in parallel,
    improving throughput compared to traditional autoregressive generation.
    
    Key Benefits:
    - Parallel token prediction within blocks
    - Adaptive unmasking based on confidence
    - Maintains quality through threshold-based filtering
    - Always generates at least one token per iteration
    
    Use Cases:
    - High-throughput text generation
    - Real-time applications requiring low latency
    - Batch processing with multiple requests
    - Scenarios where some quality trade-off is acceptable for speed
    """)


def demonstrate_different_configurations():
    """Demonstrate different block prediction configurations."""
    print("\n=== Different Block Prediction Configurations ===\n")
    
    configs = [
        {
            "name": "Conservative (High Quality)",
            "block_size": 4,
            "keep_threshold": 0.9,
            "max_iterations": 5,
        },
        {
            "name": "Balanced (Default)",
            "block_size": 8,
            "keep_threshold": 0.8,
            "max_iterations": 10,
        },
        {
            "name": "Aggressive (High Speed)",
            "block_size": 16,
            "keep_threshold": 0.6,
            "max_iterations": 15,
        },
    ]
    
    for config in configs:
        print(f"Configuration: {config['name']}")
        print(f"  Block size: {config['block_size']}")
        print(f"  Keep threshold: {config['keep_threshold']}")
        print(f"  Max iterations: {config['max_iterations']}")
        print(f"  Expected behavior:")
        
        if config['keep_threshold'] > 0.8:
            print("    - Higher quality, fewer tokens per iteration")
        elif config['keep_threshold'] < 0.7:
            print("    - Lower quality, more tokens per iteration")
        else:
            print("    - Balanced quality and speed")
        
        if config['block_size'] > 8:
            print("    - Larger blocks, more parallel prediction")
        else:
            print("    - Smaller blocks, more conservative prediction")
        print()


if __name__ == "__main__":
    main()
    demonstrate_different_configurations() 