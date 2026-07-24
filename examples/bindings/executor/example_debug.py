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
    parser.add_argument("--dump_tensors",
                        action="store_true",
                        help="Dump debug tensors to files")
    args = parser.parse_args()

    max_tokens = 2

    # Select which tensors should be kept or dumped
    debug_config = trtllm.DebugConfig(
        debug_tensor_names=["sequence_length"],
        debug_tensors_max_iterations=0 if args.dump_tensors else max_tokens)

    # Create the executor.
    executor = trtllm.Executor(
        args.model_path, trtllm.ModelType.DECODER_ONLY,
        trtllm.ExecutorConfig(1, debug_config=debug_config))

    if executor.can_enqueue_requests():
        # Create the request.
        request = trtllm.Request(input_token_ids=[1, 2, 3, 4],
                                 max_tokens=max_tokens)

        # Enqueue the request.
        request_id = executor.enqueue_request(request)

        # Wait for the new tokens.
        responses = executor.await_responses(request_id)
        output_tokens = responses[0].result.output_token_ids

        # Print tokens.
        print(output_tokens)

    if args.dump_tensors:
        print("debug tensors from files:")
        debug_dir = pl.Path("/tmp/tllm_debug/PP_1/TP_1")
        if debug_dir.is_dir():
            for iter_dir in [x for x in debug_dir.iterdir() if x.is_dir()]:
                print(iter_dir.name)
                for file in [x for x in iter_dir.iterdir() if x.is_file()]:
                    print(file.name, np.load(file))
        else:
            print("debug dir not found")
    else:
        print("debug tensors from queue:")
        debug_tensors = executor.get_latest_debug_tensors()
        for debug_iter in debug_tensors:
            print(f"iteration {debug_iter.iter}")
            for [name, tensor] in debug_iter.debug_tensors.items():
                print(name, tensor)
