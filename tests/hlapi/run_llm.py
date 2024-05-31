#!/usr/bin/env python3
import os

import click

from tensorrt_llm.hlapi import LLM, ModelConfig, SamplingConfig


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, required=True)
@click.option("--engine_dir", type=str, default=None)
def main(model_dir: str, tp_size: int, engine_dir: str):
    config = ModelConfig(model_dir)
    config.parallel_config.tp_size = tp_size

    llm = LLM(config)

    if engine_dir is not None and os.path.abspath(
            engine_dir) != os.path.abspath(model_dir):
        llm.save(engine_dir)

    prompt = [45, 12, 13]
    sampling_config = SamplingConfig(max_new_tokens=10, end_id=-1)
    for output in llm.generate([prompt], sampling_config=sampling_config):
        print(output)


if __name__ == '__main__':
    main()
