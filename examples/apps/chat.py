#! /usr/bin/env python3
import code

import click
import colorama
from transformers import AutoTokenizer, PreTrainedTokenizer

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SamplingParams


class LlmConsole(code.InteractiveConsole):

    def __init__(self,
                 llm: LLM,
                 tokenizer: PreTrainedTokenizer,
                 sampling_params: SamplingParams,
                 locals=None):
        super().__init__(locals=locals)
        self.llm = llm
        self.tokenizer = tokenizer

        self.sampling_params = sampling_params

        self.history = []

    def runsource(self,
                  source: str,
                  filename: str = "<input>",
                  symbol: str = "single") -> bool:
        prompt = source.strip()
        if prompt == "quit":
            self.llm.__exit__(None, None, None)
            return True  # exit the console

        message = {"role": "user", "content": prompt}
        self.history.append(message)

        input = self.tokenizer.apply_chat_template(self.history,
                                                   add_generation_prompt=True)

        output = self.llm.generate([input],
                                   sampling_params=self.sampling_params)[0]
        generation = self.tokenizer.decode(output.outputs[0].token_ids,
                                           skip_special_tokens=True)
        print(colorama.Fore.CYAN + "AI: " + colorama.Style.RESET_ALL +
              generation.strip())
        print()

        self.history.append({
            "role": "assistant",
            "content": generation.strip()
        })
        return False


@click.command()
@click.option(
    "--model",
    required=True,
    help=
    "The model to use, either a path to a model or a model name from Hugging Face's model hub."
)
@click.option("--tokenizer", default=None, help="The tokenizer to use")
@click.option("--tp_size",
              default=1,
              help="The number of devices for tensor parallelism to use")
def main(model: str, tokenizer: str, tp_size: int):
    kv_cache_config = KvCacheConfig(
        # you can also set max_tokens instead
        free_gpu_memory_fraction=0.8)
    kv_cache_config.enable_block_reuse = True

    build_config = BuildConfig(max_batch_size=1,
                               max_input_len=6000,
                               max_num_tokens=10240)

    sampling_params = SamplingParams(max_tokens=100,
                                     temperature=0.5,
                                     top_p=0.95,
                                     n=1)

    llm = LLM(model,
              tokenizer,
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              tensor_parallel_size=tp_size)

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer or model)

    console = LlmConsole(llm,
                         tokenizer=hf_tokenizer,
                         sampling_params=sampling_params)
    console.interact(banner="Welcome to LLM Console!", exitmsg="Goodbye!")


if __name__ == '__main__':
    main()
