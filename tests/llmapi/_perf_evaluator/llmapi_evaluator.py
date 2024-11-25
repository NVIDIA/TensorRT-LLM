#!/usr/bin/env python3
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click

from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi._perf_evaluator import LLMPerfEvaluator
from tensorrt_llm.llmapi.llm import ModelLoader
from tensorrt_llm.llmapi.llm_utils import _ModelFormatKind
from tensorrt_llm.llmapi.utils import print_colored


@click.group()
def cli():
    pass


@click.command("benchmark")
@click.option("--model-path", type=str, required=True)
@click.option("--samples-path", type=str, required=True)
@click.option("--report-path-prefix", type=str, required=True)
@click.option("--num-samples", type=int, default=None, show_default=True)
@click.option("--tp-size", type=int, default=1, show_default=True)
@click.option("--streaming/--no-streaming",
              type=bool,
              default=False,
              show_default=True)
@click.option("--warmup", type=int, default=2, show_default=True)
@click.option("--concurrency", type=int, default=None, show_default=True)
@click.option("--max-num-tokens", type=int, default=2048, show_default=True)
@click.option("--max-input-length", type=int, required=True, default=200)
@click.option("--max-seq-length", type=int, required=True, default=400)
@click.option("--max-batch-size", type=int, default=128)
@click.option("--engine-output-dir", type=str, default="")
@click.option(
    "--cpp-executable",
    type=str,
    default=None,
    help="Path to the cpp executable, set it if you want to run the cpp benchmark"
)
@click.option("--return-context-logits", type=bool, default=False)
@click.option("--return-generation-logits", type=bool, default=False)
@click.option("--kv-cache-free-gpu-mem-fraction", type=float, default=None)
def benchmark_main(model_path: str,
                   samples_path: str,
                   report_path_prefix: str,
                   num_samples: Optional[int] = None,
                   tp_size: int = 1,
                   streaming: bool = False,
                   warmup: int = 2,
                   concurrency: Optional[int] = None,
                   max_num_tokens: int = 2048,
                   max_input_length: int = 200,
                   max_seq_length: int = 400,
                   max_batch_size: int = 128,
                   engine_output_dir: str = "",
                   cpp_executable: Optional[str] = None,
                   return_context_logits=False,
                   return_generation_logits=False,
                   kv_cache_free_gpu_mem_fraction: Optional[float] = None):
    ''' Run the benchmark on LLM API.
    If `cpp_executable_path` is provided, it will run the cpp benchmark as well.
    '''
    model_path = Path(model_path)
    samples_path = Path(samples_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} not found")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples path {samples_path} not found")

    engine_output_dir = engine_output_dir or None
    temp_dir = None
    if engine_output_dir:
        engine_output_dir = Path(engine_output_dir)
    elif cpp_executable:
        if ModelLoader.get_model_format(
                model_path) is _ModelFormatKind.TLLM_ENGINE:
            engine_output_dir = model_path
        else:
            temp_dir = tempfile.TemporaryDirectory()
            engine_output_dir = Path(temp_dir.name)

    def run_llmapi():
        print_colored(f"Running LLM API benchmark ...\n", "bold_green")

        build_config = BuildConfig(max_num_tokens=max_num_tokens,
                                   max_input_len=max_input_length,
                                   max_seq_len=max_seq_length,
                                   max_batch_size=max_batch_size)

        evaluator = LLMPerfEvaluator.create(
            model=model_path,
            samples_path=samples_path,
            num_samples=num_samples,
            streaming=streaming,
            warmup=warmup,
            concurrency=concurrency,
            engine_cache_path=engine_output_dir,
            # The options should be identical to the cpp benchmark
            tensor_parallel_size=tp_size,
            build_config=build_config,
            return_context_logits=return_context_logits,
            return_generation_logits=return_generation_logits,
            kv_cache_free_gpu_mem_fraction=kv_cache_free_gpu_mem_fraction)
        assert evaluator
        report = evaluator.run()
        report.display()

        report_path = Path(f"{report_path_prefix}.json")
        i = 0
        while report_path.exists():
            report_path = Path(f"{report_path_prefix}{i}.json")
            i += 1
        report.save_json(report_path)

    def run_gpt_manager_benchmark():
        print_colored(f"Running gptManagerBenchmark ...\n", "bold_green")
        if os.path.isfile(cpp_executable):
            cpp_executable_path = cpp_executable
        else:
            cpp_executable_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cpp/build/benchmarks/gptManagerBenchmark")

        command = f"{cpp_executable_path} --engine_dir {engine_output_dir} --type IFB --dataset {samples_path} --warm_up {warmup} --output_csv {report_path_prefix}.cpp.csv --api executor --enable_chunked_context"
        if streaming:
            command = f"{command} --streaming"
        if concurrency:
            command = f"{command} --concurrency {concurrency}"
        if return_context_logits:
            command = f"{command} --return_context_logits"
        if return_generation_logits:
            command = f"{command} --return_generation_logits"
        if kv_cache_free_gpu_mem_fraction is not None:
            command = f"{command} --kv_cache_free_gpu_mem_fraction {kv_cache_free_gpu_mem_fraction}"
        if tp_size > 1:
            command = f"mpirun -n {tp_size} {command}"
        print_colored(f'cpp benchmark command: {command}\n', "grey")
        output = subprocess.run(command,
                                check=True,
                                universal_newlines=True,
                                shell=True,
                                capture_output=True,
                                env=os.environ)  # nosec B603
        print_colored(f'cpp benchmark output: {output.stdout}',
                      "grey",
                      writer=sys.stdout)
        if output.stderr:
            print_colored(f'cpp benchmark error: {output.stderr}', "red")

    run_llmapi()
    if cpp_executable:
        run_gpt_manager_benchmark()


if __name__ == '__main__':
    cli.add_command(benchmark_main)
    cli()
