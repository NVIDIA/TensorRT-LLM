### Generate Text Using Medusa Decoding
import argparse
from pathlib import Path

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import (BuildConfig, KvCacheConfig,
                                 MedusaDecodingConfig, SamplingParams)


def run_medusa_decoding(use_modelopt_ckpt=False, model_dir=None):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    # The end user can customize the sampling configuration with the SamplingParams class
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # The end user can customize the build configuration with the BuildConfig class
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
    )

    # The end user can customize the kv cache configuration with the KVCache class
    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    llm_kwargs = {}

    if use_modelopt_ckpt:
        # This is a Llama-3.1-8B combined with Medusa heads provided by TensorRT Model Optimizer.
        # Both the base model (except lm_head) and Medusa heads have been quantized in FP8.
        model = model_dir or "nvidia/Llama-3.1-8B-Medusa-FP8"

        # ModelOpt ckpt uses 3 Medusa heads
        speculative_config = MedusaDecodingConfig(
                            max_draft_len=63,
                            num_medusa_heads=3,
                            medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], \
                                [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], \
                                    [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], \
                                        [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], \
                                            [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]]
        )
    else:
        # In this path, base model and Medusa heads are stored and loaded separately.
        model = "lmsys/vicuna-7b-v1.3"

        # The end user can customize the medusa decoding configuration by specifying the
        # speculative_model_dir, max_draft_len, medusa heads num and medusa choices
        # with the MedusaDecodingConfig class
        speculative_config = MedusaDecodingConfig(
                                        speculative_model_dir="FasterDecoding/medusa-vicuna-7b-v1.3",
                                        max_draft_len=63,
                                        num_medusa_heads=4,
                                        medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                                [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                                [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                                [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                                [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                                [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
        )

    # Add 'tensor_parallel_size=2' if using ckpt for
    # a larger model like nvidia/Llama-3.1-70B-Medusa.
    llm = LLM(model=model,
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              **llm_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate text using Medusa decoding.")
    parser.add_argument(
        '--use_modelopt_ckpt',
        action='store_true',
        help="Use FP8-quantized checkpoint from TensorRT Model Optimizer.")
    # TODO: remove this arg after ModelOpt ckpt is public on HF
    parser.add_argument('--model_dir', type=Path, default=None)
    args = parser.parse_args()

    run_medusa_decoding(args.use_modelopt_ckpt, args.model_dir)
