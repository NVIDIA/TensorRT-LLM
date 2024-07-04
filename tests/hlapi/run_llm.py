#!/usr/bin/env python3
import os

import click

from tensorrt_llm.hlapi import LLM, SamplingParams


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, required=True)
@click.option("--engine_dir", type=str, default=None)
@click.option("--prompt", type=str, default=None)
def main(model_dir: str, tp_size: int, engine_dir: str, prompt: str):
    llm = LLM(model_dir, tensor_parallel_size=tp_size)

    if engine_dir is not None and os.path.abspath(
            engine_dir) != os.path.abspath(model_dir):
        llm.save(engine_dir)

    sampling_params = SamplingParams(max_new_tokens=10, end_id=-1)

    # For intentional failure test, need a simple prompt here to start LLM
    prompt_token_ids = [45, 12, 13]
    for output in llm.generate([prompt_token_ids],
                               sampling_params=sampling_params):
        print(output)

    if prompt is not None:
        for output in llm.generate([prompt], sampling_params=sampling_params):
            print(output)


if __name__ == '__main__':
    main()
