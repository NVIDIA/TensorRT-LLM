#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse

from tensorrt_llm.scaffolding import NativeGenerationController, PRMController
from tensorrt_llm.scaffolding.contrib.TreeInference import TOTController
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
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=15)
    parser.add_argument('--num_thoughts_per_step', type=int, default=3)
    parser.add_argument('--selection_strategy',
                        type=str,
                        default="best",
                        choices=["best", "vote", "random"])
    parser.add_argument('--gen_kv_cache_free_gpu_memory_fraction',
                        type=float,
                        default=0.1)
    parser.add_argument('--reward_kv_cache_free_gpu_memory_fraction',
                        type=float,
                        default=0.2)
    parser.add_argument(
        '--reward_overlap_scheduler',
        action='store_true',
        help='Enable overlap scheduler for reward worker (disabled by default)')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}

    # Initialize generation worker
    gen_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=4,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=args.
        gen_kv_cache_free_gpu_memory_fraction)
    workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

    # Initialize reward worker if provided
    reward_controller = None
    reward_worker = TRTLLMWorker.init_with_new_llm(
        args.reward_model_dir,
        backend="pytorch",
        max_batch_size=4,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=args.
        reward_kv_cache_free_gpu_memory_fraction,
        disable_overlap_scheduler=not args.reward_overlap_scheduler)
    workers[PRMController.WorkerTag.REWARD] = reward_worker

    # Create generation controller
    generation_controller = NativeGenerationController(sampling_params={
        "max_tokens": 8192,
        "temperature": 0.8,
    })
    reward_controller = PRMController(tokenizer=reward_worker.tokenizer)

    # Create TOT controller
    controller = TOTController(generation_controller=generation_controller,
                               reward_controller=reward_controller,
                               max_depth=args.max_depth,
                               max_iterations=args.max_iterations,
                               num_thoughts_per_step=args.num_thoughts_per_step,
                               selection_strategy=args.selection_strategy,
                               branch_factor=2)

    llm = ScaffoldingLlm(controller, workers=workers)

    query = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"
    prompts = [query]

    results = llm.generate(prompts)
    print(results[0].outputs[0].text)
    llm.shutdown(shutdown_workers=True)
    print(f'main shut down done')


if __name__ == "__main__":
    main()
