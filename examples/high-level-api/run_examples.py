#!/usr/bin/env python
import subprocess
import sys

from llm_examples import *

from tensorrt_llm.hlapi.utils import print_colored


@click.group()
def cli():
    pass


@click.command('run_single_gpu')
@click.option('--prompt', type=str, default="What is LLM?")
@click.option('--model_dir', type=str, help='The directory of the model.')
@click.option('--examples_root',
              type=str,
              help='The root directory of the examples.')
@click.option('--llm_examples',
              type=str,
              help='The path to the llm_examples.py.',
              default='llm_examples.py')
@click.option('--engine_dir',
              type=str,
              help='The directory of the engine.',
              default="/tmp/hlapi.engine.example")
def run_single_gpu(
    prompt: str,
    model_dir: str,
    examples_root: str,
    llm_examples: str,
    engine_dir: str,
):
    run_example(
        "Running LLM from HuggingFace model",
        f"{sys.executable} {llm_examples} run_llm_generate --prompt=\"{prompt}\" --model_dir={model_dir} --engine_dir={engine_dir}"
    )

    run_example(
        "Running LLM from built engine with streaming enabled",
        f"{sys.executable} {llm_examples} run_llm_generate_async_example --prompt=\"{prompt}\" --model_dir={engine_dir} --streaming"
    )

    run_example(
        "Running LLM with async future",
        f"{sys.executable} {llm_examples} run_llm_with_async_future --prompt=\"{prompt}\" --model_dir={engine_dir}"
    )


@click.command("run_multi_gpu")
@click.option('--prompt', type=str, default="What is LLM?")
@click.option('--model_dir', type=str, help='The directory of the model.')
@click.option('--examples_root',
              type=str,
              help='The root directory of the examples.')
@click.option('--llm_examples',
              type=str,
              help='The path to the llm_examples.py.',
              default='llm_examples.py')
@click.option('--engine_dir',
              type=str,
              help='The directory of the engine.',
              default="/tmp/hlapi.engine.example")
@click.option('--run_autopp',
              type=bool,
              help='Whether to run with auto parallel.',
              default=True)
def run_multi_gpu(
    prompt: str,
    model_dir: str,
    examples_root: str,
    llm_examples: str,
    engine_dir: str,
    run_autopp: bool = True,
):
    run_example(
        "Running LLM from HuggingFace model with TP enabled",
        f"{sys.executable} {llm_examples} run_llm_generate --prompt=\"{prompt}\" --model_dir={model_dir} --tp_size=2 --engine_dir={engine_dir}.tp2"
    )

    run_example(
        "Running LLM from built engine with streaming enabled and TP=2",
        f"{sys.executable} {llm_examples} run_llm_generate_async_example --prompt=\"{prompt}\" --model_dir={engine_dir}.tp2 --streaming"
    )  # Loading the engine with TP=2.

    if run_autopp:
        run_example(
            "Running LLM with auto parallel",
            f"{sys.executable} {llm_examples} run_llm_with_auto_parallel --prompt=\"{prompt}\" --model_dir={model_dir} --world_size=2"
        )


@click.command("run_quant")
@click.option('--prompt', type=str, default="What is LLM?")
@click.option('--model_dir', type=str, help='The directory of the model.')
@click.option('--examples_root',
              type=str,
              help='The root directory of the examples.')
@click.option('--llm_examples',
              type=str,
              help='The path to the llm_examples.py.',
              default='llm_examples.py')
def run_quant(
    prompt: str,
    model_dir: str,
    examples_root: str,
    llm_examples: str,
):
    run_example(
        "Running LLM with quantization",
        f"{sys.executable} {llm_examples} run_llm_with_quantization --quant_type=int4_awq --prompt=\"{prompt}\" --model_dir={model_dir}"
    )


def run_example(hint: str, command: str):
    print_colored(hint + "\n", "bold_green")
    print(command)
    subprocess.run(command, shell=True, check=True)


if __name__ == '__main__':
    cli.add_command(run_single_gpu)
    cli.add_command(run_multi_gpu)
    cli.add_command(run_quant)
    cli()
