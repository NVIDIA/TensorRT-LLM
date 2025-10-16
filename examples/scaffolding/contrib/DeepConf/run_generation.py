import argparse
import time

from tensorrt_llm.scaffolding import (NativeGenerationController,
                                      ScaffoldingLlm, TRTLLMWorker)
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
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--conf_group_size', type=int, default=128)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--vote_policy',
                        type=str,
                        default="top10_bottom_window_filtered")
    parser.add_argument('--warmup_sample_num', type=int, default=5)
    parser.add_argument('--confidence_percentile', type=int, default=90)
    parser.add_argument('--logprobs_topk', type=int, default=20)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()
    return args


def run_scaffolding_llm(prompts, proposer_worker, controller):
    llm = ScaffoldingLlm(
        controller,
        {
            NativeGenerationController.WorkerTag.GENERATION: proposer_worker,
        },
    )
    time_start = time.time()
    results = llm.generate(prompts)
    time_end = time.time()
    print(f"time cost: {time_end - time_start} seconds")
    for i, result in enumerate(results):
        print(f"result {i}:\n{result.outputs[0].text}")
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
    run_scaffolding_llm(prompts, proposer_worker, prototype_controller)


def test_majority_vote_controller(prompts,
                                  proposer_worker,
                                  conf_group_size,
                                  conf_threshold,
                                  logprobs_topk,
                                  temperature,
                                  max_tokens,
                                  top_p,
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
    run_scaffolding_llm(prompts, proposer_worker, majority_vote_controller)


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
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=32,
        max_num_tokens=kwargs.get("max_tokens"),
    )
    print(f"init llm worker done")

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
