import argparse
import csv
import datetime
from pathlib import Path

import tensorrt_llm

trtllm_package_dir = Path(tensorrt_llm.__file__).parent
executor_worker_path = trtllm_package_dir / 'bin' / 'executorWorker'

import tensorrt_llm.bindings.executor as trtllm


# Read input tokens from csv file
def read_input_tokens(input_tokens_csv_file: str) -> list[int]:

    input_tokens = []
    with open(input_tokens_csv_file, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            input_tokens.append([int(item) for item in lines])
    return input_tokens


# Prepare and enqueue the requests
def enqueue_requests(args: argparse.Namespace,
                     executor: trtllm.Executor) -> None:

    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = args.exclude_input_from_output
    sampling_config = trtllm.SamplingConfig(args.beam_width)
    input_tokens = read_input_tokens(args.input_tokens_csv_file)

    request_ids = []
    for tokens in input_tokens:
        req = trtllm.Request(input_token_ids=tokens,
                             max_tokens=args.max_tokens,
                             streaming=args.streaming,
                             sampling_config=sampling_config,
                             output_config=output_config)
        req_id = executor.enqueue_request(req)
        request_ids.append(req_id)

    return request_ids


# Wait for responses and store output tokens
def wait_for_responses(args: argparse.Namespace, request_ids: list[int],
                       executor: trtllm.Executor) -> dict[dict[list[int]]]:

    output_tokens = {
        req_id: {
            beam: []
            for beam in range(args.beam_width)
        }
        for req_id in request_ids
    }
    num_finished = 0
    iter = 0
    while (num_finished < len(request_ids) and iter < args.timeout_ms):
        responses = executor.await_responses(
            datetime.timedelta(milliseconds=args.timeout_ms))
        for response in responses:
            req_id = response.request_id
            if not response.has_error():
                result = response.result
                num_finished += 1 if result.is_final else 0
                for beam, outTokens in enumerate(result.output_token_ids):
                    output_tokens[req_id][beam].extend(outTokens)
            else:
                raise RuntimeError(
                    str(req_id) + " encountered error:" + response.error_msg)

    return output_tokens


# Write the output tokens to file
def write_output_tokens(output_tokens_csv_file: str, request_ids: list[int],
                        output_tokens: dict[dict[list[int]]],
                        beam_width: int) -> None:

    with open(output_tokens_csv_file, 'w') as csvfile:

        writer = csv.writer(csvfile)
        for req_id in request_ids:
            out_tokens = output_tokens[req_id]
            for beam in range(args.beam_width):
                beam_tokens = out_tokens[beam]
                writer.writerow(beam_tokens)

    print("Output tokens written to:", output_tokens_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Bindings Example")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    parser.add_argument("--input_tokens_csv_file",
                        type=str,
                        required=True,
                        help="CSV file containing the input tokens")
    parser.add_argument("--output_tokens_csv_file",
                        type=str,
                        required=False,
                        default="output_tokens.csv",
                        help="CSV file where to write output tokens")
    parser.add_argument("--beam_width",
                        type=int,
                        required=False,
                        default=1,
                        help="The beam width")
    parser.add_argument("--streaming",
                        default=False,
                        action="store_true",
                        help="Operate in streaming mode")

    parser.add_argument("--use_orchestrator_mode",
                        default=False,
                        action="store_true",
                        help="Operate in orchestrator mode for multi-GPU runs")

    parser.add_argument(
        "--exclude_input_from_output",
        default=False,
        action="store_true",
        help=
        "Exclude input token when writing output tokens. Only has effect for streaming=False since in streaming mode, input tokens are never included in output."
    )
    parser.add_argument("--max_tokens",
                        type=int,
                        required=False,
                        default=10,
                        help="The max number of tokens to be generated")
    parser.add_argument(
        "--timeout_ms",
        type=int,
        required=False,
        default=10000,
        help="The maximum time to wait for all responses, in milliseconds")

    args = parser.parse_args()
    executor_config = trtllm.ExecutorConfig(args.beam_width)

    if args.use_orchestrator_mode:
        orchestrator_config = trtllm.OrchestratorConfig(
            True, str(executor_worker_path))
        executor_config.parallel_config = trtllm.ParallelConfig(
            trtllm.CommunicationType.MPI, trtllm.CommunicationMode.ORCHESTRATOR,
            None, None, orchestrator_config)

    # Create the executor.
    executor = trtllm.Executor(args.model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    if executor.can_enqueue_requests():
        # Enqueue the requests
        request_ids = enqueue_requests(args, executor)

        # Wait for the responses
        output_tokens = wait_for_responses(args, request_ids, executor)

        # Write the output tokens
        write_output_tokens(args.output_tokens_csv_file, request_ids,
                            output_tokens, args.beam_width)
