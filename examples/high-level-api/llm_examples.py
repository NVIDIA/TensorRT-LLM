#!/usr/bin/env python3
import asyncio
import inspect
import sys
from argparse import ArgumentParser
from typing import List, Optional

from tensorrt_llm import LLM, ModelConfig

# NOTE, Currently, the following examples are only available for LLaMA models.


def run_llm_from_huggingface_model(prompts: List[str],
                                   llama_model_dir: str,
                                   dump_engine_dir: Optional[str] = None):
    ''' Loading a HuggingFace model. '''
    # Load the model from a local HuggingFace model
    config = ModelConfig(llama_model_dir)
    llm = LLM(config)
    if dump_engine_dir:
        llm.save(dump_engine_dir)

    for output in llm.generate(prompts):
        print(output)


def run_llm_from_tllm_engine(prompts: List[str], llama_engine_dir: str):
    ''' Loading a built TensorRT-LLM engine.  '''
    config = ModelConfig(llama_engine_dir)
    llm = LLM(config)

    for output in llm.generate(prompts):
        print(output)


def run_llm_on_tensor_parallel(prompts: List[str], llama_model_dir: str):
    ''' Running LLM with Tensor Parallel on multiple GPUs. '''
    config = ModelConfig(llama_model_dir)
    config.parallel_config.tp_size = 2  # 2 GPUs

    llm = LLM(config)

    for output in llm.generate(prompts):
        print(output)


def run_llm_generate_async_example(prompts: List[str],
                                   llama_model_dir: str,
                                   streaming: bool = False):
    ''' Running LLM generation asynchronously. '''
    config = ModelConfig(llama_model_dir)

    llm = LLM(config, async_mode=True)

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
                              engine_dump_dir: Optional[str] = None,
                              quant_type: str = 'int4_awq'):
    ''' Running LLM with quantization.
    quant_type could be 'int4_awq' or 'fp8'.
    '''

    config = ModelConfig(llama_model_dir)
    if quant_type == 'int4_awq':
        config.quant_config.init_from_description(quantize_weights=True,
                                                  use_int4_weights=True,
                                                  per_group=True)
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
    parser.add_argument('--prompt', type=str, required=True)
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
            [args.prompt], args.hf_model_dir, args.dump_engine_dir),
        run_llm_from_tllm_engine=lambda: run_llm_from_tllm_engine(
            [args.prompt], args.dump_engine_dir),
        run_llm_on_tensor_parallel=lambda: run_llm_on_tensor_parallel(
            [args.prompt], args.hf_model_dir),
        run_llm_generate_async_example=lambda: run_llm_generate_async_example(
            [args.prompt], args.hf_model_dir),
        run_llm_with_quantization=lambda: run_llm_with_quantization([
            args.prompt
        ], args.hf_model_dir, args.dump_engine_dir, args.quant_type),
    )

    print(f'Running {args.task} ...')

    tasks[args.task]()
