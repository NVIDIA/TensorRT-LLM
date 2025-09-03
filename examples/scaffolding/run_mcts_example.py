import argparse

from tensorrt_llm.scaffolding import (MCTSController,
                                      NativeGenerationController, PRMController)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import TRTLLMWorker


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir',
                        type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument('--reward_model_dir',
                        type=str,
                        default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument('--jsonl_file', type=str, default='./test.jsonl')
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=20)
    parser.add_argument('--exploration_constant', type=float, default=1.414)
    parser.add_argument('--num_thoughts_per_step', type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}
    gen_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=4,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=0.1)
    workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

    # Initialize reward worker if provided
    reward_controller = None
    reward_worker = TRTLLMWorker.init_with_new_llm(
        args.reward_model_dir,
        backend="pytorch",
        max_batch_size=4,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=0.2,
        disable_overlap_scheduler=True)
    workers[PRMController.WorkerTag.REWARD] = reward_worker

    # Create generation controller
    generation_controller = NativeGenerationController(sampling_params={
        "max_tokens": 4096,
        "temperature": 0.8,
    })
    reward_controller = PRMController(tokenizer=reward_worker.tokenizer)

    # Create MCTS controller
    controller = MCTSController(
        generation_controller=generation_controller,
        reward_controller=reward_controller,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        exploration_constant=args.exploration_constant,
        num_thoughts_per_step=args.num_thoughts_per_step)

    llm = ScaffoldingLlm(controller, workers=workers)

    query = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"
    prompts = [query]

    results = llm.generate(prompts)
    print(results[0].outputs[0].text)
    llm.shutdown(shutdown_workers=True)
    print(f'main shut down done')


if __name__ == "__main__":
    main()
