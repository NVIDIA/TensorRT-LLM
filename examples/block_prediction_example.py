#!/usr/bin/env python3
"""
Example demonstrating block prediction with TensorRT-LLM.

Block prediction allocates a block of N tokens, all starting as masked tokens,
then runs forward passes with no causal mask, unmasking tokens with softmax
probabilities greater than a threshold (e.g., 0.8), always unmasking at least one token.
This process repeats until all tokens are unmasked.
"""
from contextlib import contextmanager
import time


@contextmanager
def timer(name: str | None = None):
    """Context manager for timing code execution."""
    if name is None:
        name = "Execution"

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = end - start
        print(f"{name} took {elapsed:.2f} seconds")


with timer("Importing torch..."):
    import torch
with timer("Importing transformers..."):
    from transformers import AutoTokenizer
with timer("Importing tensorrt_llm..."):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig


def main():
    # Create an LLM with block prediction enabled
    llm = LLM(
        model="Dream-org/Dream-v0-Instruct-7B",  # Using a supported model
        tokenizer=AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True),
        backend="pytorch",
        enable_block_prediction=True,
        disable_overlap_scheduler=True,
        block_size=8,  # Number of tokens to predict in each block
        keep_threshold=0.6,  # Confidence threshold for keeping tokens
        mask_token_id=151666,  # Token ID to use as mask
        max_iterations=10,  # Maximum number of iterations
        max_batch_size=1,
        max_num_tokens=4096,
        max_seq_len=5000,
    )
    print("Inited the LLM")

    print("=== Block Prediction Example ===\n")
    print("Configuration:")
    # Access block prediction config through the args
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    if isinstance(llm.args, TorchLlmArgs):
        print(f"  Block prediction enabled: {llm.args.enable_block_prediction}")
        print(f"  Block size: {llm.args.block_size}")
        print(f"  Keep threshold: {llm.args.keep_threshold}")
        print(f"  Mask token ID: {llm.args.mask_token_id}")
        print(f"  Max iterations: {llm.args.max_iterations}")
    else:
        print("  Block prediction configuration not available")
    print()

    # Define sample prompts
    prompts = [
        "The future of artificial intelligence is not the"+"<|mask|>"*8,
        # "In a world where technology advances rapidly,",
        # "The most important thing to remember is",
    ]

    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        end_id=151643,  # Add end_id to prevent ValueError
    )

    # Generate text with block prediction
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")

        # Generate response
        outputs = llm.generate([prompt], sampling_params)

        # Print the generated text
        # outputs is a list of RequestOutput objects
        # Each RequestOutput has an outputs attribute which is a list of CompletionOutput objects
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            if hasattr(output, 'outputs') and isinstance(output.outputs, list) and len(output.outputs) > 0:
                generated_text = output.outputs[0].text
                print(f"Generated: {generated_text}")
                print(f"  Generated {len(generated_text.split())} words")
            else:
                print("No output generated")
        else:
            print("No outputs returned")
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
    - Real-time applications requiring low latency
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
