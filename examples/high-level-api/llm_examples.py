#!/usr/bin/env python3
import asyncio
import inspect
import sys
from argparse import ArgumentParser
from typing import List, Optional

import torch

from tensorrt_llm import LLM, ModelConfig
from tensorrt_llm.hlapi.llm import SamplingConfig
from tensorrt_llm.hlapi.utils import get_device_count

# NOTE, Currently, the following examples are only available for LLaMA models.


def run_llm_from_huggingface_model(prompts: List[str],
                                   llama_model_dir: str,
                                   dump_engine_dir: Optional[str] = None,
                                   tp_size: int = 1):
    ''' Loading a HuggingFace model. '''
    if get_device_count() < tp_size:
        print(
            "Skip the example for TP!!! Since the number of GPUs is less than required"
        )
        return
    if tp_size > 1:
        print(f'Running LLM with Tensor Parallel on {tp_size} GPUs.')

    config = ModelConfig(llama_model_dir)
    config.parallel_config.tp_size = tp_size

    llm = LLM(config)
    if dump_engine_dir:
        llm.save(dump_engine_dir)

    for output in llm.generate(prompts):
        print(output)


def run_llm_from_tllm_engine(prompts: List[str],
                             llama_engine_dir: str,
                             tp_size: int = 1):
    ''' Loading a built TensorRT-LLM engine. '''

    config = ModelConfig(llama_engine_dir)
    config.parallel_config.tp_size = tp_size
    llm = LLM(config)

    for output in llm.generate(prompts):
        print(output)


def run_llm_without_tokenizer_from_tllm_engine(llama_engine_dir: str):
    ''' Loading a TensorRT-LLM engine built by trtllm-build, and the tokenizer is missing too. '''

    config = ModelConfig(llama_engine_dir)
    llm = LLM(config)

    # since tokenizer is missing, so we cannot get a default sampling config, create one manually
    sampling_config = SamplingConfig(end_id=2,
                                     pad_id=2,
                                     output_sequence_lengths=True,
                                     return_dict=True)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)


def run_llm_generate_async_example(prompts: List[str],
                                   llama_model_dir: str,
                                   streaming: bool = False,
                                   tp_size: int = 1):
    ''' Running LLM generation asynchronously. '''

    if get_device_count() < tp_size:
        print(
            "Skip the example for TP!!! Since the number of GPUs is less than required"
        )
        return
    if tp_size > 1:
        print(f'Running LLM with Tensor Parallel on {tp_size} GPUs.')

    config = ModelConfig(llama_model_dir)
    config.parallel_config.tp_size = tp_size

    llm = LLM(config, async_mode=True, kvcahe_free_gpu_memory_fraction=0.4)

    async def task(prompt: str):
        outputs = []
        async for output in llm.generate_async(prompt, streaming=streaming):
            outputs.append(output.text)
        print(' '.join(outputs))

    async def main():
        tasks = [task(prompt) for prompt in prompts]
        await asyncio.gather(*tasks)

    asyncio.run(main())


def run_llm_with_quantization(prompts: List[str],
                              llama_model_dir: str,
                              quant_type: str = 'int4_awq'):
    ''' Running LLM with quantization.
    quant_type could be 'int4_awq' or 'fp8'.
    '''

    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Quantization currently only supported on post Ampere")
        return

    if 'fp8' in quant_type:
        if not (major > 8):
            print("Hopper GPUs are required for fp8 quantization")
            return

    config = ModelConfig(llama_model_dir)
    if quant_type == 'int4_awq':
        config.quant_config.init_from_description(quantize_weights=True,
                                                  use_int4_weights=True,
                                                  per_group=True)
        config.quant_config.quantize_lm_head = True

    else:
        config.quant_config.set_fp8_qdq()
        config.quant_config.set_fp8_kv_cache()

    llm = LLM(config)

    for output in llm.generate(prompts):
        print(output)


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, choices=_get_functions())
    parser.add_argument('--hf_model_dir',
                        type=str,
                        help='The directory of the model.')
    parser.add_argument('--dump_engine_dir',
                        type=str,
                        help='The directory to dump the engine.',
                        default=None)
    parser.add_argument('--quant_type', type=str, choices=['int4_awq', 'fp8'])
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--streaming', action='store_true')
    return parser.parse_args()


def _get_functions():
    cur_module = sys.modules[__name__]
    function_names = [
        name for name, _ in inspect.getmembers(cur_module, inspect.isfunction)
        if not name.startswith('_')
    ]
    return function_names


if __name__ == '__main__':
    args = _parse_arguments()

    tasks = dict(
        run_llm_from_huggingface_model=lambda: run_llm_from_huggingface_model(
            [args.prompt],
            args.hf_model_dir,
            args.dump_engine_dir,
            tp_size=args.tp_size),
        run_llm_from_tllm_engine=lambda: run_llm_from_tllm_engine(
            [args.prompt],
            args.dump_engine_dir,
            tp_size=args.tp_size,
        ),
        run_llm_generate_async_example=lambda: run_llm_generate_async_example(
            [args.prompt],
            args.hf_model_dir,
            tp_size=args.tp_size,
            streaming=args.streaming),
        run_llm_with_quantization=lambda: run_llm_with_quantization(
            [args.prompt], args.hf_model_dir, args.quant_type),
        run_llm_without_tokenizer_from_tllm_engine=lambda:
        run_llm_without_tokenizer_from_tllm_engine(args.dump_engine_dir),
    )

    print(f'Running {args.task} ...')

    tasks[args.task]()
