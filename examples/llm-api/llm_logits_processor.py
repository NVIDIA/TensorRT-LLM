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
                 token_ids: List[List[int]], stream_ptr: int,
                 client_id: Optional[int]):
        mask = torch.full_like(logits, fill_value=float("-inf"), device="cpu")
        mask[:, :, self.allowed_token_id] = 0

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(
            stream_ptr)
        with torch.cuda.stream(stream):
            mask = mask.to(logits.device, non_blocking=True)
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
                 token_ids: List[List[List[int]]], stream_ptr: int,
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

    # Batched logits processor (only supported in TensorRT backend)
    # should be specified when initializing LLM.
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        batched_logits_processor=MyBatchedLogitsProcessor(allowed_token_id=42))

    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    # Generate text
    for prompt_id, prompt in enumerate(prompts):
        # Use non-batched logits processor callback only for odd-numbered prompts
        if prompt_id % 2 == 0:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        else:
            # Each prompt can be specified with a logits processor at runtime
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                logits_processor=MyLogitsProcessor(allowed_token_id=42))

        for output in llm.generate([prompt], sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: "''''''''''''''''''''''''''''''''"

    # Use batched processor with batch size = 2
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
