### Control generated text using logits post processor
import typing as tp

import torch

from tensorrt_llm import LLM, SamplingParams


# Define the logits post-processor callback. This simple callback will output
# a specific token at each step irrespective of prompt.
# Refer to ../bindings/executor/example_logits_processor.py for a more
# sophisticated callback that generates JSON structured output.
def logits_post_processor(req_id: int, logits: torch.Tensor,
                          ids: tp.List[tp.List[int]], stream_ptr: int,
                          client_id: tp.Optional[int]):
    target_token_id = 42
    with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
        logits[:] = float("-inf")
        logits[..., target_token_id] = 0


def main():

    # Several callbacks can be specified when initializing LLM
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              logits_post_processor_map={"my_logits_pp": logits_post_processor})

    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    # Generate text
    for prompt_id, prompt in enumerate(prompts):
        # We will use logits post processor callback only for odd-numbered prompts
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


if __name__ == '__main__':
    main()
