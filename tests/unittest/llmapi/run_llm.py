#!/usr/bin/env python3
from typing import Optional

import click

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, required=True)
@click.option("--n", type=int, default=1)
@click.option("--best_of", type=int, default=None)
@click.option("--top_k", type=int, default=1)
@click.option("--use_beam_search", is_flag=True)
def main(model_dir: str, tp_size: int, n: int, best_of: Optional[int],
         top_k: int, use_beam_search: bool):
    llm = LLM(model_dir,
              tensor_parallel_size=tp_size,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=10,
                                     end_id=-1,
                                     n=n,
                                     best_of=best_of,
                                     use_beam_search=use_beam_search,
                                     top_k=top_k)
    print(sampling_params)
    prompt_token_ids = [45, 12, 13]
    for output in llm.generate([prompt_token_ids],
                               sampling_params=sampling_params):
        print(output)

    print("run_llm.py Done")


if __name__ == '__main__':
    main()
