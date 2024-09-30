import argparse
import pathlib as pl

import numpy as np

import tensorrt_llm.bindings.executor as trtllm

# This example hows to use the python bindings to create an executor, enqueue a
# request, and get the generated tokens.

# First, follow the steps in README.md to generate the engines.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Bindings Example")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    args = parser.parse_args()

    # debug_config = trtllm.DebugConfig(dump_input_tensors=True,
    #                                   dump_output_tensors=True,
    #                                   debug_tensor_names=["test"])

    # Select which tensors should be dumped
    debug_config = trtllm.DebugConfig(debug_tensor_names=["host_request_types"])

    # Create the executor.
    executor = trtllm.Executor(
        args.model_path, trtllm.ModelType.DECODER_ONLY,
        trtllm.ExecutorConfig(1, debug_config=debug_config))

    if executor.can_enqueue_requests():
        # Create the request.
        request = trtllm.Request(input_token_ids=[1, 2, 3, 4], max_tokens=2)

        # Enqueue the request.
        request_id = executor.enqueue_request(request)

        # Wait for the new tokens.
        responses = executor.await_responses(request_id)
        output_tokens = responses[0].result.output_token_ids

        # Print tokens.
        print(output_tokens)

    print("debug tensors:")
    debug_dir = pl.Path("/tmp/tllm_debug/PP_1/TP_1")
    for iter_dir in [x for x in debug_dir.iterdir() if x.is_dir()]:
        print(iter_dir.name)
        for file in [x for x in iter_dir.iterdir() if x.is_file()]:
            print(file.name, np.load(file))
