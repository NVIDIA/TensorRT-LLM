import argparse

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

    # Create the executor.
    executor = trtllm.Executor(args.model_path, trtllm.ModelType.DECODER_ONLY,
                               trtllm.ExecutorConfig(1))

    if executor.can_enqueue_requests():
        # Create the request.
        request = trtllm.Request(input_token_ids=[1, 2, 3, 4], max_tokens=10)

        # Enqueue the request.
        request_id = executor.enqueue_request(request)

        # Wait for the new tokens.
        responses = executor.await_responses(request_id)
        output_tokens = responses[0].result.output_token_ids

        # Print tokens.
        print(output_tokens)
