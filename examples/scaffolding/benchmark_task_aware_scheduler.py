import argparse
import asyncio
import statistics
import time

from tensorrt_llm.scaffolding.controller import (BestOfNController,
                                                 NativeGenerationController,
                                                 PRMController)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import TRTLLMWorker


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generation_model',
        type=str,
        default="DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        '--reward_model',
        type=str,
        default="Qwen2.5-Math-PRM-7B",
    )
    parser.add_argument('--sample_num', type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}
    gen_worker = TRTLLMWorker.init_with_new_llm(
        args.generation_model,
        backend="pytorch",
        max_batch_size=args.sample_num,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=0.1)
    reward_worker = TRTLLMWorker.init_with_new_llm(
        args.reward_model,
        backend="pytorch",
        max_batch_size=args.sample_num,
        max_num_tokens=8192,
        kv_cache_free_gpu_memory_fraction=0.2,
        disable_overlap_scheduler=True)
    workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker
    workers[PRMController.WorkerTag.REWARD] = reward_worker

    gen_controller = NativeGenerationController(sampling_params={
        "max_tokens": 2048,
        "temperature": 0.6,
    },
                                                # streaming=True,
                                                )
    reward_controller = PRMController(tokenizer=reward_worker.tokenizer)
    controller = BestOfNController(
        generation_controller=gen_controller,
        reward_controller=reward_controller,
        default_sample_num=args.sample_num,
    )
    llm = ScaffoldingLlm(controller, workers=workers)

    # query = "Solve for x: 4x + 5 = 6x - 7"
    query_num = 1
    base_query = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"
    prompts = []
    for i in range(query_num):
        modified_query = base_query.replace("Sue", f"{i+1}")
        prompts.append(modified_query)

    durations = asyncio.run(benchmark_concurrent_prompts(prompts, llm))
    print(f"average duration: {statistics.mean(durations)}")
    print(f"median duration: {statistics.median(durations)}")
    print(f"max duration: {max(durations)}")
    print(f"min duration: {min(durations)}")

    llm.shutdown(shutdown_workers=True)
    print(f'main shut down done')


async def benchmark_single_prompt(prompt, llm: ScaffoldingLlm):
    time_start = time.time()
    i = 0
    async for result in llm.generate_async(prompt):
        i += 1
        print(">>>", i, result)
        async for output in result.cur_output:
            print(">>>", i, len(output.outputs[0].token_ids), "\n",
                  output.outputs[0].text)
    time_duration = time.time() - time_start
    return time_duration


# async def benchmark_single_prompt(prompt, llm: ScaffoldingLlm):
#     time_start = time.time()

#     # ✅ 使用 aresult() 而不是异步迭代器
#     result = llm.generate_async(prompt)
#     await result.aresult()

#     # 处理最终结果
#     if result.cur_output:
#         print("Final result:", result.cur_output.outputs[0].text)

#     time_duration = time.time() - time_start
#     return time_duration


async def benchmark_concurrent_prompts(prompts, llm: ScaffoldingLlm):
    print(f"concurrently execute {len(prompts)} prompts")

    tasks = [benchmark_single_prompt(prompt, llm) for prompt in prompts]

    durations = await asyncio.gather(*tasks)

    return durations


if __name__ == '__main__':
    main()
