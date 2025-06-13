#!/usr/bin/env python3
import asyncio
import json
import os
from typing import Optional

import click

from tensorrt_llm.executor import GenerationResultBase
from tensorrt_llm.executor.postproc_worker import PostprocArgs, PostprocParams
from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.utils import print_colored
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    DeltaMessage)


def perform_faked_oai_postprocess(rsp: GenerationResultBase,
                                  args: PostprocArgs):
    first_iteration = len(rsp.outputs[0].token_ids) == 1
    num_choices = 1
    finish_reason_sent = [False] * num_choices
    role = "assistant"
    model = "LLaMA"

    def yield_first_chat(idx: int, role: str = None, content: str = None):
        choice_data = ChatCompletionResponseStreamChoice(index=idx,
                                                         delta=DeltaMessage(
                                                             role=role,
                                                             content=content),
                                                         finish_reason=None)
        chunk = ChatCompletionStreamResponse(choices=[choice_data], model=model)

        data = chunk.model_dump_json(exclude_unset=True)
        return data

    res = []
    if first_iteration:
        for i in range(num_choices):
            res.append(f"data: {yield_first_chat(i, role=role)} \n\n")
    first_iteration = False

    for output in rsp.outputs:
        i = output.index

        if finish_reason_sent[i]:
            continue

        delta_text, args.last_text_len = output.text_diff_safe(args.last_text_len)
        delta_message = DeltaMessage(content=delta_text)

        choice = ChatCompletionResponseStreamChoice(index=i,
                                                    delta=delta_message,
                                                    finish_reason=None)
        if output.finish_reason is not None:
            choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(choices=[choice], model=model)
        data = chunk.model_dump_json(exclude_unset=True)
        res.append(f"data: {data}\n\n")

    if rsp._done:
        res.append(f"data: [DONE]\n\n")

    return res


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
        "_num_postprocess_workers": tp_size,
        "_postprocess_tokenizer_dir": model_dir,
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
    postproc_params = PostprocParams(
        post_processor=perform_faked_oai_postprocess,
        postproc_args=PostprocArgs(),
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
