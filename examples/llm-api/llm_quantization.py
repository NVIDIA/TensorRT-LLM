### Generation with Quantization
import logging

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.hlapi import QuantAlgo, QuantConfig

major, minor = torch.cuda.get_device_capability()
post_ada = major > 8 or (major == 8 and minor >= 9)

quant_configs = [
    QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ),
]

if post_ada:
    quant_configs.append(
        QuantConfig(quant_algo=QuantAlgo.FP8,
                    kv_cache_quant_algo=QuantAlgo.FP8))
else:
    logging.error(
        "FP8 quantization only works on post-ada GPUs, skipped in the example.")

for quant_config in quant_configs:

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # define the quantization config to trigger built-in end-to-end quantization.
        quant_config=quant_config)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

# Got output like
# Prompt: 'Hello, my name is', Generated text: 'Jane Smith. I am a resident of the city. Can you tell me more about the public services provided in the area?'
# Prompt: 'The president of the United States is', Generated text: 'considered the head of state, and the vice president of the United States is considered the head of state. President and Vice President of the United States (US)'
# Prompt: 'The capital of France is', Generated text: 'located in Paris, France. The population of Paris, France, is estimated to be 2 million. France is home to many famous artists, including Picasso'
# Prompt: 'The future of AI is', Generated text: 'an open and collaborative project. The project is an ongoing effort, and we invite participation from members of the community.\n\nOur community is'
