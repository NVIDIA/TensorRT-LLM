"""Example: Using PyTorch backend with TensorRT-LLM Scaffolding.

This example demonstrates how to use the PyTorchWorker for running inference
with HuggingFace models without TensorRT compilation. This is useful for:
- Rapid prototyping
- Research models (reward models, verifiers)
- Models not yet supported by TensorRT

The example compares PyTorch and TRT-LLM backends side-by-side.
"""

import argparse
import asyncio
import time

from tensorrt_llm.scaffolding import (
    BestOfNController,
    NativeGenerationController,
    NativeRewardController,
    PyTorchWorker,
    ScaffoldingLlm,
    TRTLLMWorker,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run scaffolding with PyTorch backend")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="gpt2",
        help="HuggingFace model name or path (e.g., 'gpt2', 'meta-llama/Llama-3.2-1B')",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on ('cuda', 'cpu', 'cuda:0', etc.)"
    )
    parser.add_argument(
        "--compare_trt",
        action="store_true",
        help="Compare with TRT-LLM backend (requires TRT-compiled model in --trt_model_dir)",
    )
    parser.add_argument(
        "--trt_model_dir", type=str, default=None, help="Path to TRT-compiled model for comparison"
    )
    parser.add_argument("--run_async", action="store_true", help="Run async generation demo")
    parser.add_argument(
        "--run_reward", action="store_true", help="Run Best-of-N with reward model demo"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="HuggingFace reward model name or path for --run_reward demo",
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument(
        "--sample_num", type=int, default=4, help="Number of samples for Best-of-N (default: 4)"
    )
    args = parser.parse_args()
    return args


def test_pytorch_worker(prompts, model_name, device, max_tokens):
    """Test PyTorch worker with generation tasks."""
    print(f"\n{'=' * 60}")
    print(f"PyTorch Backend: {model_name}")
    print(f"{'=' * 60}\n")

    # Create PyTorch worker
    print(f"Loading model '{model_name}' on device '{device or 'auto'}'...")
    start_time = time.time()
    pytorch_worker = PyTorchWorker.from_pretrained(model_name, device=device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s\n")

    # Create controller with sampling params
    controller = NativeGenerationController(
        sampling_params={
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 50,
        }
    )

    # Create scaffolding LLM
    llm = ScaffoldingLlm(
        controller,
        {NativeGenerationController.WorkerTag.GENERATION: pytorch_worker},
    )

    # Run generation
    start_time = time.time()
    results = llm.generate(prompts)
    gen_time = time.time() - start_time

    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"Prompt {i + 1}: {prompt[:60]}...")
        print(f"Output: {result.outputs[0].text}")
        print(f"Tokens: {len(result.outputs[0].token_ids)}")
        print()

    print(f"Generation time: {gen_time:.2f}s")
    print(f"Throughput: {len(prompts) / gen_time:.2f} prompts/s\n")

    # Cleanup
    llm.shutdown()
    pytorch_worker.shutdown()

    return results


def test_trt_worker(prompts, trt_model_dir, max_tokens):
    """Test TRT-LLM worker for comparison."""
    print(f"\n{'=' * 60}")
    print(f"TRT-LLM Backend: {trt_model_dir}")
    print(f"{'=' * 60}\n")

    # Create TRT worker
    print(f"Loading TRT model from '{trt_model_dir}'...")
    start_time = time.time()
    trt_worker = TRTLLMWorker.init_with_new_llm(
        trt_model_dir,
        backend="pytorch",
        max_batch_size=8,
        max_num_tokens=2048,
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s\n")

    # Create controller
    controller = NativeGenerationController(
        sampling_params={
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 50,
        }
    )

    # Create scaffolding LLM
    llm = ScaffoldingLlm(
        controller,
        {NativeGenerationController.WorkerTag.GENERATION: trt_worker},
    )

    # Run generation
    start_time = time.time()
    results = llm.generate(prompts)
    gen_time = time.time() - start_time

    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"Prompt {i + 1}: {prompt[:60]}...")
        print(f"Output: {result.outputs[0].text}")
        print(f"Tokens: {len(result.outputs[0].token_ids)}")
        print()

    print(f"Generation time: {gen_time:.2f}s")
    print(f"Throughput: {len(prompts) / gen_time:.2f} prompts/s\n")

    # Cleanup
    llm.shutdown()
    trt_worker.shutdown()

    return results


def test_async(prompt, model_name, device, max_tokens):
    """Test async generation with PyTorch worker."""

    async def test_async_func(prompt, model_name, device, max_tokens):
        print(f"\n{'=' * 60}")
        print(f"Async PyTorch Backend: {model_name}")
        print(f"{'=' * 60}\n")

        pytorch_worker = PyTorchWorker.from_pretrained(model_name, device=device)

        controller = NativeGenerationController(
            sampling_params={
                "temperature": 0.7,
                "max_tokens": max_tokens,
            },
            streaming=True,
        )

        llm = ScaffoldingLlm(
            controller,
            {NativeGenerationController.WorkerTag.GENERATION: pytorch_worker},
        )

        print(f"Prompt: {prompt}\n")
        print("Streaming output:")
        print("-" * 60)

        step = 0
        async for result in llm.generate_async(prompt):
            step += 1
            print(f"Step {step}: {result.outputs[0].text}")

        print("-" * 60)
        print(f"\nFinal output ({len(result.outputs[0].token_ids)} tokens):")
        print(result.outputs[0].text)

        llm.shutdown()
        pytorch_worker.shutdown()

    asyncio.run(test_async_func(prompt, model_name, device, max_tokens))


def test_best_of_n_with_reward(prompt, gen_model, reward_model, device, max_tokens, sample_num):
    """Demo: Best-of-N selection using generation + reward workers.

    This demonstrates the core use case from issue #3333 -- using a PyTorch
    reward model alongside a generation model in the scaffolding framework.
    """
    print(f"\n{'=' * 60}")
    print("Best-of-N with Reward Model")
    print(f"  Generation: {gen_model}")
    print(f"  Reward:     {reward_model}")
    print(f"  Samples:    {sample_num}")
    print(f"{'=' * 60}\n")

    # Create generation worker
    print("Loading generation model...")
    gen_worker = PyTorchWorker.from_pretrained(gen_model, device=device)

    # Create reward worker
    print("Loading reward model...")
    reward_worker = PyTorchWorker.from_pretrained_reward_model(reward_model, device=device)

    # Build Best-of-N controller: generate N candidates, score each, pick best
    gen_controller = NativeGenerationController(
        sampling_params={
            "temperature": 0.9,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }
    )
    reward_controller = NativeRewardController()
    controller = BestOfNController(gen_controller, reward_controller, default_sample_num=sample_num)

    # Wire workers
    llm = ScaffoldingLlm(
        controller,
        {
            NativeGenerationController.WorkerTag.GENERATION: gen_worker,
            NativeRewardController.WorkerTag.REWARD: reward_worker,
        },
    )

    print(f"\nPrompt: {prompt}")
    print(f"Generating {sample_num} candidates and scoring...\n")

    start_time = time.time()
    result = llm.generate(prompt)
    elapsed = time.time() - start_time

    print(f"Best output: {result.outputs[0].text}")
    print(f"Time: {elapsed:.2f}s\n")

    llm.shutdown()
    gen_worker.shutdown()
    reward_worker.shutdown()


def main():
    args = parse_arguments()

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about coding.",
    ]

    if args.run_reward:
        if args.reward_model is None:
            print("Error: --reward_model required when using --run_reward")
            print("Example: --reward_model OpenAssistant/reward-model-deberta-v3-large-v2")
            return
        test_best_of_n_with_reward(
            prompts[0],
            args.model_dir,
            args.reward_model,
            args.device,
            args.max_tokens,
            args.sample_num,
        )
    elif args.run_async:
        # Run async demo with first prompt
        test_async(prompts[0], args.model_dir, args.device, args.max_tokens)
    else:
        # Run PyTorch worker
        test_pytorch_worker(prompts, args.model_dir, args.device, args.max_tokens)

        # Optionally compare with TRT
        if args.compare_trt:
            if args.trt_model_dir is None:
                print("Error: --trt_model_dir required when using --compare_trt")
                return

            test_trt_worker(prompts, args.trt_model_dir, args.max_tokens)

            print(f"\n{'=' * 60}")
            print("Comparison Summary")
            print(f"{'=' * 60}")
            print("PyTorch backend enables easy integration of research models")
            print("TRT-LLM backend provides optimized performance")
            print("Both can be mixed in the same scaffolding workflow!\n")


if __name__ == "__main__":
    main()
