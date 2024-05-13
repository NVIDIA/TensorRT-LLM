#!/usr/bin/env python3
import click

from tensorrt_llm.hlapi.llm import LLM, ModelConfig, ModelLoader


@click.command()
@click.option('--model_dir',
              type=str,
              required=True,
              help='Path to the model directory')
@click.option('--tokenizer_dir',
              type=str,
              default=None,
              help='Path to the tokenizer directory')
@click.option('--prompt',
              type=str,
              default="Tell a story",
              help='Prompt to generate text from')
def main(model_dir: str, tokenizer_dir: str, prompt: str):
    config = ModelConfig(model_dir)

    if tokenizer_dir is None:
        tokenizer_dir = model_dir

    tokenizer = ModelLoader.load_hf_tokenizer(tokenizer_dir)

    llm = LLM(config, tokenizer=tokenizer)

    for output in llm.generate([prompt]):
        print("OUTPUT:", output.text)


if __name__ == '__main__':
    main()
