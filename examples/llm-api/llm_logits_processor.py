### Control generated text using logits post processor
import typing as tp

import torch

from tensorrt_llm import LLM, SamplingParams


def get_allowed_tokens(ids):
    return 42


# Define the logits post-processor callback. This simple callback will output
# a specific token at each step irrespective of prompt.
# Refer to ../bindings/executor/example_logits_processor.py for a more
# sophisticated callback that generates JSON structured output.
def logits_post_processor(req_id: int, logits: torch.Tensor,
                          token_ids: tp.List[tp.List[int]], stream_ptr: int,
                          client_id: tp.Optional[int]):
    mask = torch.full_like(logits, fill_value=float("-inf"), device="cpu")
    allowed = get_allowed_tokens(token_ids)
    mask[:, :, allowed] = 0

    with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
        mask = mask.to(logits.device, non_blocking=True)
        logits += mask


# Define batched processor in which arguments for all requests in a batch are made available as lists.
# This helps user optimize the callback for large batch sizes. For example:
# 1. Process more work on host, e.g. running a JSON state machine, in parallel with model forward pass on device.
# 2. Coalesce H2D memory transfers for all requests into a single cudaMemcpyAsync call.
# 3. Launch a single batched kernel, e.g. for updating logits on device.
def logits_post_processor_batched(
        req_ids_batch: tp.List[int], logits_batch: tp.List[torch.Tensor],
        token_ids_batch: tp.List[tp.List[tp.List[int]]], stream_ptr,
        client_ids_batch: tp.List[tp.Optional[int]]):
    # Generate masks for all requests on host
    masks = []
    for req_id, logits, token_ids, client_id in zip(req_ids_batch, logits_batch,
                                                    token_ids_batch,
                                                    client_ids_batch):
        mask = torch.full_like(logits, fill_value=float("-inf"), device="cpu")
        allowed = get_allowed_tokens(token_ids)
        mask[:, :, allowed] = 0
        masks.append(mask)

    # Move masks to device and add to logits using non-blocking operations
    with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
        for logits, mask in zip(logits_batch, masks):
            logits += mask.to(logits.device, non_blocking=True)


def main():

    # Several callbacks can be specified when initializing LLM. In addition to multiple non-batched callbacks,
    # a single batched callback can be specified using the key SamplingParams.BATCHED_POST_PROCESSOR_NAME
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              logits_post_processor_map={
                  "my_logits_pp":
                  logits_post_processor,
                  SamplingParams.BATCHED_POST_PROCESSOR_NAME:
                  logits_post_processor_batched
              })

    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    # Generate text
    for prompt_id, prompt in enumerate(prompts):
        # We will use non-batched logits post processor callback only for odd-numbered prompts
        if prompt_id % 2 == 0:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        else:
            # Each prompt can use one callback from the choices that were provided to LLM
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                logits_post_processor_name='my_logits_pp')

        for output in llm.generate([prompt], sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: "''''''''''''''''''''''''''''''''"

    # Use batched processor with batch size = 2
    sampling_params = SamplingParams(
        logits_post_processor_name=SamplingParams.BATCHED_POST_PROCESSOR_NAME)
    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: "''''''''''''''''''''''''''''''''"
    # Prompt: 'The president of the United States is', Generated text: "''''''''''''''''''''''''''''''''"


if __name__ == '__main__':
    main()
