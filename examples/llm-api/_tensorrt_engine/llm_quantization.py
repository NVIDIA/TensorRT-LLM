### Generation with Quantization
import logging

import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import CalibConfig, QuantAlgo, QuantConfig

major, minor = torch.cuda.get_device_capability()
enable_fp8 = major > 8 or (major == 8 and minor >= 9)
enable_nvfp4 = major >= 10

quant_and_calib_configs = []

if not enable_nvfp4:
    # Example 1: Specify int4 AWQ quantization to QuantConfig.
    # We can skip specifying CalibConfig or leave a None as the default value.
    quant_and_calib_configs.append(
        (QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ), None))

if enable_fp8:
    # Example 2: Specify FP8 quantization to QuantConfig.
    # We can create a CalibConfig to specify the calibration dataset and other details.
    # Note that the calibration dataset could be either HF dataset name or a path to local HF dataset.
    quant_and_calib_configs.append(
        (QuantConfig(quant_algo=QuantAlgo.FP8,
                     kv_cache_quant_algo=QuantAlgo.FP8),
         CalibConfig(calib_dataset='cnn_dailymail',
                     calib_batches=256,
                     calib_max_seq_length=256)))
else:
    logging.error(
        "FP8 quantization only works on post-ada GPUs. Skipped in the example.")

if enable_nvfp4:
    # Example 3: Specify NVFP4 quantization to QuantConfig.
    quant_and_calib_configs.append(
        (QuantConfig(quant_algo=QuantAlgo.NVFP4,
                     kv_cache_quant_algo=QuantAlgo.FP8),
         CalibConfig(calib_dataset='cnn_dailymail',
                     calib_batches=256,
                     calib_max_seq_length=256)))
else:
    logging.error(
        "NVFP4 quantization only works on Blackwell. Skipped in the example.")


def main():

    for quant_config, calib_config in quant_and_calib_configs:
        # The built-in end-to-end quantization is triggered according to the passed quant_config.
        llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  quant_config=quant_config,
                  calib_config=calib_config)

        # Sample prompts.
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ]

        # Create a sampling params.
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        for output in llm.generate(prompts, sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )
        llm.shutdown()

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: 'Jane Smith. I am a resident of the city. Can you tell me more about the public services provided in the area?'
    # Prompt: 'The capital of France is', Generated text: 'located in Paris, France. The population of Paris, France, is estimated to be 2 million. France is home to many famous artists, including Picasso'
    # Prompt: 'The future of AI is', Generated text: 'an open and collaborative project. The project is an ongoing effort, and we invite participation from members of the community.\n\nOur community is'


if __name__ == '__main__':
    main()
