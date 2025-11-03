import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from utils import equal_func, prepare_prompt

from tensorrt_llm.scaffolding import (NativeGenerationController,
                                      ScaffoldingLlm, TRTLLMWorker,
                                      extract_answer_from_boxed)
from tensorrt_llm.scaffolding.contrib.DeepConf import (
    DeepConfOfflineController, DeepConfOfflineMajorityVoteController,
    DeepConfOnlineController, DeepConfOnlineMajorityVoteController)

_RUN_TYPE_TO_IMPL = {
    "offline": DeepConfOfflineController,
    "online": DeepConfOnlineController,
    "offline_majority_vote": DeepConfOfflineMajorityVoteController,
    "online_majority_vote": DeepConfOnlineMajorityVoteController,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument('--run_type',
                        type=str,
                        required=True,
                        choices=list(_RUN_TYPE_TO_IMPL.keys()),
                        help="Type of the run. Available choices: %(choices)s")
    parser.add_argument('--warmup_sample_num', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=256)
    parser.add_argument('--conf_group_size', type=int, default=2048)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--vote_policy',
                        type=str,
                        default="top10_bottom_window_filtered")
    parser.add_argument('--confidence_percentile', type=int, default=10)
    parser.add_argument('--logprobs_topk', type=int, default=20)
    parser.add_argument('--max_tokens', type=int, default=64000)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--qid', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="brumo_2025.jsonl")
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    args = parser.parse_args()
    return args


@dataclass
class BenchResult:
    right_answer_count: int = 0
    total_answer_count: int = 0
    accuracy: float = 0.0
    generated_tokens: int = 0


def run_scaffolding_llm(prompts,
                        proposer_worker,
                        controller,
                        repeat_times=1,
                        ground_truth=None,
                        **kwargs):
    llm = ScaffoldingLlm(
        controller,
        {
            NativeGenerationController.WorkerTag.GENERATION: proposer_worker,
        },
    )

    is_majority_vote = isinstance(
        controller, DeepConfOnlineMajorityVoteController) or isinstance(
            controller, DeepConfOfflineMajorityVoteController)
    vote_policy_to_bench_result: Dict[str, BenchResult] = {}
    times = []
    for i in range(repeat_times):
        print(f"=========== round {i} ===========")
        start_time = time.time()
        results = llm.generate(prompts)
        times.append(time.time() - start_time)

        for j, result in enumerate(results):
            print(
                f"result {j}: {extract_answer_from_boxed(result.outputs[0].text)}"
            )

            if is_majority_vote and ground_truth is not None:
                vote_policy_to_voted_task = result.cur_output.vote_policy_to_voted_task
                for vote_policy, voted_task in vote_policy_to_voted_task.items(
                ):
                    bench_result = vote_policy_to_bench_result.get(
                        vote_policy, BenchResult())

                    voted_answer = voted_task.customized_result_fields[
                        'extracted_answer']
                    if equal_func(voted_answer, ground_truth[j]):
                        bench_result.right_answer_count += 1
                    bench_result.total_answer_count += 1
                    bench_result.generated_tokens += result.cur_output.output_token_num

                    vote_policy_to_bench_result[vote_policy] = bench_result

    print(f"e2e inference median time cost: {np.median(times):.2f} seconds")

    if is_majority_vote:
        for vote_policy, bench_result in vote_policy_to_bench_result.items():
            bench_result.accuracy = bench_result.right_answer_count / bench_result.total_answer_count
            print(
                f"vote_policy: {vote_policy}, accuracy: {bench_result.accuracy}"
            )

        print(f"generated tokens: {bench_result.generated_tokens}")

    llm.shutdown(shutdown_workers=True)


def test_single_vote_controller(prompts,
                                proposer_worker,
                                conf_group_size,
                                conf_threshold,
                                temperature,
                                max_tokens,
                                logprobs_topk,
                                top_p,
                                run_type="offline",
                                **kwargs):
    generation_controller = NativeGenerationController(
        sampling_params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_logprobs": logprobs_topk,
            "top_p": top_p,
        })
    DeepConfControllerImpl = _RUN_TYPE_TO_IMPL[run_type]
    prototype_controller = DeepConfControllerImpl(
        generation_controller=generation_controller,
        conf_group_size=conf_group_size,
        conf_threshold=conf_threshold,
    )
    run_scaffolding_llm(prompts, proposer_worker, prototype_controller,
                        **kwargs)


def test_majority_vote_controller(prompts,
                                  proposer_worker,
                                  conf_group_size,
                                  conf_threshold,
                                  logprobs_topk,
                                  temperature,
                                  max_tokens,
                                  top_p,
                                  top_k,
                                  sample_num,
                                  warmup_sample_num,
                                  vote_policy,
                                  confidence_percentile,
                                  run_type="offline_majority_vote",
                                  **kwargs):
    generation_controller = NativeGenerationController(
        sampling_params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_logprobs": logprobs_topk,
            "top_p": top_p,
            "top_k": top_k,
        })
    DeepConfControllerKwargs = {
        "generation_controller": generation_controller,
        "conf_group_size": conf_group_size,
        "conf_threshold": conf_threshold,
    }
    warmup_generation_controller = DeepConfOfflineController(
        **DeepConfControllerKwargs)
    final_generation_controller = DeepConfOnlineController(
        **DeepConfControllerKwargs)
    DeepConfMajorityVoteControllerImpl = _RUN_TYPE_TO_IMPL[run_type]
    majority_vote_controller = DeepConfMajorityVoteControllerImpl(
        generation_controller=warmup_generation_controller,
        warmup_generation_controller=warmup_generation_controller,
        final_generation_controller=final_generation_controller,
        sample_num=sample_num,
        vote_policy=vote_policy,
        warmup_sample_num=warmup_sample_num,
        confidence_percentile=confidence_percentile)
    run_scaffolding_llm(prompts, proposer_worker, majority_vote_controller,
                        **kwargs)


def main():
    args = parse_arguments()
    kwargs = {
        "sample_num": args.sample_num,
        "conf_group_size": args.conf_group_size,
        "conf_threshold": args.conf_threshold,
        "vote_policy": args.vote_policy,
        "warmup_sample_num": args.warmup_sample_num,
        "confidence_percentile": args.confidence_percentile,
        "logprobs_topk": args.logprobs_topk,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_times": args.repeat_times,
        "max_tokens": args.max_tokens,
    }

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=2048,
        max_num_tokens=args.max_tokens,
    )
    print(f"init llm worker done")

    dataset_path = Path(__file__).parent / args.dataset
    with open(dataset_path, 'r', encoding='utf-8') as file:
        question_data = [json.loads(line.strip()) for line in file]

    if args.qid != -1:
        question_data = [question_data[args.qid]]
    prompts = [
        prepare_prompt(question_data['question'], llm_worker.tokenizer)
        for question_data in question_data
    ]
    ground_truth = [
        str(question_data.get('answer', '')).strip()
        for question_data in question_data
    ]
    kwargs["ground_truth"] = ground_truth

    print(f"has {len(prompts)} prompts")

    if args.run_type == "offline" or args.run_type == "online":
        test_single_vote_controller(prompts,
                                    llm_worker,
                                    run_type=args.run_type,
                                    **kwargs)
    elif args.run_type == "offline_majority_vote" or args.run_type == "online_majority_vote":
        test_majority_vote_controller(prompts,
                                      llm_worker,
                                      run_type=args.run_type,
                                      **kwargs)

    llm_worker.shutdown()
    print('llm worker shutdown done')


if __name__ == "__main__":
    main()
