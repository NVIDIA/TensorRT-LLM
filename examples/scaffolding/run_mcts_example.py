import argparse

from tensorrt_llm.scaffolding import (MCTSController, NativeGenerationController,
                                      NativeRewardController, ScaffoldingLlm,
                                      TRTLLMWorker)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default="HuggingFaceTB/SmolLM-1.7B-Instruct",  # Much smaller default model
        help="Path to the directory containing the generation model")
    parser.add_argument(
        '--reward_model_dir',
        type=str,
        default="HuggingFaceTB/SmolLM-360M-Instruct",  # Smaller reward model
        help="Path to the directory containing the reward model (optional)")
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=20)
    parser.add_argument('--exploration_constant', type=float, default=1.414)
    parser.add_argument('--num_thoughts_per_step', type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}

    # Initialize generation worker
    gen_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=16,
        max_num_tokens=4096,
        kv_cache_free_gpu_memory_fraction=0.4)
    workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

    # Initialize reward worker if provided
    reward_controller = None
    if args.reward_model_dir:
        reward_worker = TRTLLMWorker.init_with_new_llm(
            args.reward_model_dir,
            backend="pytorch",
            max_batch_size=8,
            max_num_tokens=2048,
            kv_cache_free_gpu_memory_fraction=0.2)
        workers[NativeRewardController.WorkerTag.REWARD] = reward_worker
        reward_controller = NativeRewardController()

    # Create generation controller
    generation_controller = NativeGenerationController(sampling_params={
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    # Create MCTS controller
    controller = MCTSController(
        generation_controller=generation_controller,
        reward_controller=reward_controller,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        exploration_constant=args.exploration_constant,
        num_thoughts_per_step=args.num_thoughts_per_step
    )

    llm = ScaffoldingLlm(controller, workers=workers)

    # Test problems - mathematical reasoning tasks
    test_problems = [
        "Solve for x: 3x + 7 = 22. Show your work step by step.",
        "A rectangle has a perimeter of 24 cm and a length that is 2 cm more than its width. Find the dimensions.",
        "If a train travels 180 km in 2.5 hours, what is its average speed in km/h?",
    ]

    print("🔄 Testing MCTS Controller on Mathematical Problems")
    print("=" * 60)

    for i, problem in enumerate(test_problems):
        print(f"\n📝 Problem {i+1}: {problem}")
        print("-" * 40)

        try:
            results = llm.generate([problem])
            
            for result in results:
                print(f"🎯 MCTS Solution:")
                print(result.outputs[0].text)
                print(f"📊 Tokens generated: {len(result.outputs[0].token_ids)}")
                
        except Exception as e:
            print(f"❌ Error solving problem {i+1}: {e}")

    print(f'\n🔄 Shutting down...')
    llm.shutdown(shutdown_workers=True)
    print(f'✅ Shutdown complete')


if __name__ == "__main__":
    main() 