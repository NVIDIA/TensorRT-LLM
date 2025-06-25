#!/usr/bin/env python3
import os

import click

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig, SamplingParams


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, default=1)
@click.option("--engine_dir", type=str, default=None)
def main(model_dir: str, tp_size: int, engine_dir: str):
    build_config = BuildConfig()
    build_config.max_batch_size = 8
    build_config.max_input_len = 256
    build_config.max_seq_len = 512

    llm = LLM(model_dir,
              tensor_parallel_size=tp_size,
              build_config=build_config)

    if engine_dir is not None and os.path.abspath(
            engine_dir) != os.path.abspath(model_dir):
        llm.save(engine_dir)

    sampling_params = SamplingParams(max_tokens=10, end_id=-1)

    prompt_token_ids = [45, 12, 13]
    for output in llm.generate([prompt_token_ids],
                               sampling_params=sampling_params):
        print(output)


if __name__ == '__main__':
    main()
