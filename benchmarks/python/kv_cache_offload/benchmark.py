import argparse
import datetime
import json
import random
import time

import tensorrt_llm.bindings.executor as trtllm

output_config = trtllm.OutputConfig()
output_config.exclude_input_from_output = False
sampling_config = trtllm.SamplingConfig(1)


def generate_random_tokens(rounds=10, count=64) -> list[list[int]]:
    ret = []
    for i in range(rounds):
        ret.append([random.randint(0, 1000) for _ in range(count)])
    return ret


# Read input tokens from json file
def read_input_json(input_dataset_path: str,
                    num_users) -> tuple[list[list[int]], list[int]]:
    with open(input_dataset_path, "r") as f:
        data = json.load(f)

    input_tokens = []
    output_lens = []
    for n in range(num_users):
        sample = data["samples"][n]
        input_tokens.append(sample["input_ids"])
        output_lens.append(sample["output_len"])

    return input_tokens, output_lens


# Prepare and enqueue the requests
def enqueue_requests(args: argparse.Namespace, executor: trtllm.Executor,
                     input_tokens) -> list[int]:
    request_ids = []
    for tokens in input_tokens:
        req = trtllm.Request(input_token_ids=tokens,
                             max_tokens=args.output_len,
                             streaming=False,
                             sampling_config=sampling_config,
                             output_config=output_config)
        req_id = executor.enqueue_request(req)
        request_ids.append(req_id)

    return request_ids


def get_TTFT(stats_queue):
    iter_latency = []
    cache_hit_rates = []
    for stats in stats_queue:
        iter_latency.append(stats.iter_latency_ms)
        cache_hit_rates.append(stats.kv_cache_stats.cache_hit_rate)

    TTFT_idx = [i for i, x in enumerate(cache_hit_rates) if x > 0.01][1]
    return iter_latency[TTFT_idx]


# Wait for responses and store output tokens
def wait_for_responses(args: argparse.Namespace, request_ids: list[int],
                       executor: trtllm.Executor) -> list[list[int]]:

    output_tokens = {req_id: [] for req_id in request_ids}
    num_finished = 0
    iterations = 0
    while (num_finished < len(request_ids) and iterations < args.timeout_ms):
        responses = executor.await_responses(
            datetime.timedelta(milliseconds=args.timeout_ms))
        for response in responses:
            req_id = response.request_id
            if not response.has_error():
                result = response.result
                num_finished += 1 if result.is_final else 0
                for _, outTokens in enumerate(result.output_token_ids):
                    output_tokens[req_id].extend(outTokens)
            else:
                raise RuntimeError(
                    str(req_id) + " encountered error:" + response.error_msg)

    return list(output_tokens.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Bindings Example")
    parser.add_argument("--n", type=int, required=True, help="Number of users")
    parser.add_argument("--free_gpu_memory_fraction",
                        required=False,
                        type=float,
                        default=0.9,
                        help="free_gpu_memory_fraction")
    parser.add_argument("--kv_host_cache_bytes",
                        required=False,
                        type=int,
                        default=55000000000,
                        help="host_cache_size")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    parser.add_argument("--input_dataset_path",
                        type=str,
                        required=True,
                        help="Text file containing the input tokens")
    parser.add_argument("--beam_width",
                        type=int,
                        required=False,
                        default=1,
                        help="The beam width")
    parser.add_argument("--streaming",
                        default=False,
                        action="store_true",
                        help="Operate in streaming mode")
    parser.add_argument("--output_len",
                        type=int,
                        required=False,
                        default=64,
                        help="The number of tokens to be generated for output.")
    parser.add_argument("--rounds",
                        type=int,
                        required=False,
                        default=10,
                        help="How many runs of user input to run.")
    parser.add_argument(
        "--timeout_ms",
        type=int,
        required=False,
        default=10000,
        help="The maximum time to wait for all responses, in milliseconds")
    parser.add_argument(
        "--log_iteration_data",
        action='store_true',
        help="Print the verbose iteration status data (default: False).")

    args = parser.parse_args()

    kv_cache_config = trtllm.KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=args.free_gpu_memory_fraction,
        host_cache_size=args.kv_host_cache_bytes)

    executor_config = trtllm.ExecutorConfig(args.beam_width,
                                            kv_cache_config=kv_cache_config)

    # Create the executor.
    executor = trtllm.Executor(args.model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    new_inputs = [generate_random_tokens(args.rounds) for _ in range(args.n)]
    stats_queue = []

    if executor.can_enqueue_requests():
        ## Process long context to generate kvcache
        context_tokens, _ = read_input_json(args.input_dataset_path, args.n)

        # Enqueue the requests
        request_ids = enqueue_requests(args, executor, context_tokens)

        # Wait for the responses
        output_tokens = wait_for_responses(args, request_ids, executor)

        stats_queue.extend(executor.get_latest_iteration_stats())

        # Start the multi-turn runs
        ## Start timing
        start_time = time.time()

        for r in range(args.rounds):
            current_input_tokens = [
                output_tokens[i] + new_inputs[i][r] for i in range(args.n)
            ]
            # Enqueue the requests
            request_ids = enqueue_requests(args, executor, current_input_tokens)

            # Wait for the responses
            output_tokens = wait_for_responses(args, request_ids, executor)

            stats_queue.extend(executor.get_latest_iteration_stats())
        ## End timing
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"E2E TIME: {elapsed_time:.2f} (ms)")
        print(f"TTFT: {get_TTFT(stats_queue)} (ms)")

    if args.log_iteration_data:
        for stats in stats_queue:
            print(stats.to_json_str())
