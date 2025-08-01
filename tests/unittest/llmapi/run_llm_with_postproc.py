#!/usr/bin/env python3
import asyncio
import json
import os
from typing import Optional

import click

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.utils import print_colored
from tensorrt_llm.serve.postprocess_handlers import (ChatPostprocArgs,
                                                     chat_stream_post_processor)


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, required=True)
@click.option("--engine_dir", type=str, default=None)
@click.option("--n", type=int, default=1)
@click.option("--best_of", type=int, default=None)
@click.option("--top_k", type=int, default=1)
def main(model_dir: str, tp_size: int, engine_dir: Optional[str], n: int,
         best_of: Optional[int], top_k: int):

    # Simplified postprocessing configuration
    postproc_config = {
        "num_postprocess_workers": tp_size,
        "postprocess_tokenizer_dir": model_dir,
    }

    print_colored("Enabled OAI postprocessing\n", "yellow")

    llm = LLM(model_dir,
              tensor_parallel_size=tp_size,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
              **postproc_config)

    if engine_dir is not None and os.path.abspath(
            engine_dir) != os.path.abspath(model_dir):
        llm.save(engine_dir)

    sampling_params = SamplingParams(max_tokens=10,
                                     end_id=-1,
                                     n=n,
                                     best_of=best_of,
                                     top_k=top_k)
    postproc_args = ChatPostprocArgs(role="assistant",
                                     model="TinyLlama-1.1B-Chat-v1.0")
    postproc_params = PostprocParams(
        post_processor=chat_stream_post_processor,
        postproc_args=postproc_args,
    )

    prompt = "A B C D E F"

    outputs = []

    async def generate_async():
        async for output in llm.generate_async(prompt,
                                               sampling_params=sampling_params,
                                               _postproc_params=postproc_params,
                                               streaming=True):
            print(output)
            outputs.append(output.outputs[0]._postprocess_result)

    asyncio.run(generate_async())

    expected = "G H I J K L M N O P"
    actual = get_concatenated_content(outputs)
    assert actual == expected, f"Expected '{expected}', got '{actual}'"


def get_concatenated_content(outputs):
    content = []
    for chunk in outputs:
        for line in chunk:
            line = line.strip()
            if not line.startswith('data: '):
                continue

            json_str = line.split('data: ', 1)[1]
            if json_str == '[DONE]':
                continue

            data = json.loads(json_str)
            for choice in data.get('choices', []):
                if 'delta' in choice and 'content' in choice['delta']:
                    content_value = choice['delta']['content']
                    if content_value is not None:
                        content.append(content_value)
    return ''.join(content)


if __name__ == '__main__':
    main()
