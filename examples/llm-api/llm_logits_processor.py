### Control generated text using logits processor
from typing import List, Optional

import torch

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import (BatchedLogitsProcessor,
                                          LogitsProcessor, SamplingParams)


# The recommended way to create a customized logits processor:
#     * Subclass LogitsProcessor and implement the processing logics in the __call__ method.
#     * Create an instance and pass to SamplingParams.
# Alternatively, you can create any callable with the same signature with the __call__ method.
# This simple callback will output a specific token at each step irrespective of prompt.
# Refer to ../bindings/executor/example_logits_processor.py for a more
# sophisticated callback that generates JSON structured output.
class MyLogitsProcessor(LogitsProcessor):

    def __init__(self, allowed_token_id: int):
        self.allowed_token_id = allowed_token_id

    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: Optional[int],
                 client_id: Optional[int]):

        # No stream needed for pure PyTorch backend
        if stream_ptr is None:
            # Create a mask that disables all tokens except allowed_token_id
            mask = torch.full_like(
                logits, fill_value=float("-inf"))  # Shape: [vocab_size]
            mask[self.allowed_token_id] = 0
            logits += mask

            return logits

        mask = torch.full_like(logits, fill_value=float("-inf"), device="cpu")
        mask[:, :, self.allowed_token_id] = 0

        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            mask = mask.to(logits.device, non_blocking=True)
            # logits are modified in-place
            logits += mask


# The recommended way to create a customized batched logits processor:
#     * Subclass BatchedLogitsProcessor and implement the processing logics in the __call__ method.
#     * Create an instance and pass to LLM.
# Alternatively, you can create any callable with the same signature with the __call__ method.
# A batched logits processor's arguments for all requests in a batch are made available as lists.
# This helps user optimize the callback for large batch sizes. For example:
# 1. Process more work on host, e.g. running a JSON state machine, in parallel with model forward pass on device.
# 2. Coalesce H2D memory transfers for all requests into a single cudaMemcpyAsync call.
# 3. Launch a single batched kernel, e.g. for updating logits on device.
class MyBatchedLogitsProcessor(BatchedLogitsProcessor):

    def __init__(self, allowed_token_id: int):
        self.allowed_token_id = allowed_token_id

    def __call__(self, req_ids: List[int], logits: List[torch.Tensor],
                 token_ids: List[List[List[int]]], stream_ptr: Optional[int],
                 client_ids: List[Optional[int]]):
        # Generate masks for all requests on host
        masks = []
        for req_id, req_logits, req_token_ids, client_id in zip(
                req_ids, logits, token_ids, client_ids):
            mask = torch.full_like(req_logits,
                                   fill_value=float("-inf"),
                                   device="cpu")
            mask[:, :, self.allowed_token_id] = 0
            masks.append(mask)

        # Move masks to device and add to logits using non-blocking operations
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for req_logits, mask in zip(logits, masks):
                req_logits += mask.to(req_logits.device, non_blocking=True)


def main():

    # Toggle this to switch between TensorRT and PyTorch backends
    use_pytorch_backend = True

    if use_pytorch_backend:
        from tensorrt_llm._torch import LLM as LLM_Torch
        llm = LLM_Torch(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        # Batched logits processor is only supported in TensorRT backend
        llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  batched_logits_processor=MyBatchedLogitsProcessor(
                      allowed_token_id=42))

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    # [Example Usage 1] - Specify logit processor per generation call
    for prompt_id, prompt in enumerate(prompts):
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            logits_processor=MyLogitsProcessor(allowed_token_id=42))

        for output in llm.generate([prompt], sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )

        # Got output like
        # Prompt: 'Hello, my name is', Generated text: "''''''''''''''''''''''''''''''''"
        # Prompt: 'The president of the United States is', Generated text: "''''''''''''''''''''''''''''''''"

    if not use_pytorch_backend:
        # [Example Usage 2] - Use batched processor with batch size = 2 (TensorRT-backend only)
        sampling_params = SamplingParams(apply_batched_logits_processor=True)
        for output in llm.generate(prompts, sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )

        # Got output like
        # Prompt: 'Hello, my name is', Generated text: "''''''''''''''''''''''''''''''''"
        # Prompt: 'The president of the United States is', Generated text: "''''''''''''''''''''''''''''''''"


if __name__ == '__main__':
    main()
