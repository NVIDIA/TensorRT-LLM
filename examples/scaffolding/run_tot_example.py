import argparse

from tensorrt_llm.scaffolding import (TOTController, NativeGenerationController,
                                      NativeRewardController, ScaffoldingLlm,
                                      TRTLLMWorker)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument(
        '--reward_model_dir',
        type=str,
        help="Path to the directory containing the reward model (optional)")
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=15)
    parser.add_argument('--num_thoughts_per_step', type=int, default=3)
    parser.add_argument('--selection_strategy', type=str, default="best", 
                        choices=["best", "vote", "random"])
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
        "max_tokens": 400,
        "temperature": 0.6,
        "top_p": 0.9,
    })

    # Create TOT controller
    controller = TOTController(
        generation_controller=generation_controller,
        reward_controller=reward_controller,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        num_thoughts_per_step=args.num_thoughts_per_step,
        selection_strategy=args.selection_strategy
    )

    llm = ScaffoldingLlm(controller, workers=workers)

    # Test problems - complex reasoning tasks that benefit from tree search
    test_problems = [
        """A farmer has chickens and rabbits in his farm. He counts 35 heads and 94 legs in total. 
        How many chickens and how many rabbits does he have? Show your step-by-step reasoning.""",
        
        """You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water? 
        Explain each step clearly.""",
        
        """Three friends Alice, Bob, and Carol are sitting in a row. Alice is not sitting next to Bob. 
        Carol is sitting to the right of Alice. Who is in the middle? Explain your reasoning."""
    ]

    print("🌲 Testing TOT Controller on Complex Reasoning Problems")
    print("=" * 60)

    for i, problem in enumerate(test_problems):
        print(f"\n📝 Problem {i+1}: {problem}")
        print("-" * 40)

        try:
            results = llm.generate([problem])
            
            for result in results:
                print(f"🎯 TOT Solution:")
                print(result.outputs[0].text)
                print(f"📊 Tokens generated: {len(result.outputs[0].token_ids)}")
                
        except Exception as e:
            print(f"❌ Error solving problem {i+1}: {e}")

    print(f'\n🔄 Shutting down...')
    llm.shutdown(shutdown_workers=True)
    print(f'✅ Shutdown complete')


if __name__ == "__main__":
    main() 