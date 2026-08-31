#!/usr/bin/env python3
from typing import Optional

import click

from tensorrt_llm.llmapi import LLM, SamplingParams


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, default=1)
@click.option("--backend", type=str, default=None)
def main(model_dir: str, tp_size: int, backend: Optional[str]):
    backend = backend or "pytorch"
    assert backend == "pytorch"

    llm = LLM(model_dir, tensor_parallel_size=tp_size)

    sampling_params = SamplingParams(max_tokens=10, end_id=-1)

    prompt_token_ids = [45, 12, 13]
    for output in llm.generate([prompt_token_ids],
                               sampling_params=sampling_params):
        print(output)


if __name__ == '__main__':
    main()
