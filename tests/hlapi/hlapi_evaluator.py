#!/usr/bin/env python3
import os
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import click

from tensorrt_llm.hlapi import BuildConfig
from tensorrt_llm.hlapi._perf_evaluator import LLMPerfEvaluator
from tensorrt_llm.hlapi.llm import ModelLoader
from tensorrt_llm.hlapi.llm_utils import _ModelFormatKind
from tensorrt_llm.hlapi.utils import print_colored

try:
    from .grid_searcher import GridSearcher
except:
    from grid_searcher import GridSearcher


@click.group()
def cli():
    pass


@click.command("benchmark")
@click.option("--model-path", type=str, required=True)
@click.option("--samples-path", type=str, required=True)
@click.option("--report-path-prefix", type=str, required=True)
@click.option("--num-samples", type=int, default=-1)
@click.option("--tp-size", type=int, default=1, show_default=True)
@click.option("--warmup", type=int, default=100, show_default=True)
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
def benchmark_main(model_path: str,
                   samples_path: str,
                   report_path_prefix: str,
                   num_samples: int = -1,
                   tp_size: int = 1,
                   warmup: int = 100,
                   max_num_tokens=2048,
                   max_input_length: int = 200,
                   max_seq_length: int = 400,
                   max_batch_size: int = 128,
                   engine_output_dir: str = "",
                   cpp_executable: str = None):
    ''' Run the benchmark on HLAPI.
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

    def run_hlapi():
        print_colored(f"Running HLAPI benchmark ...\n", "bold_green")

        build_config = BuildConfig(max_num_tokens=max_num_tokens,
                                   max_input_len=max_input_length,
                                   max_seq_len=max_seq_length,
                                   max_batch_size=max_batch_size)

        evaluator = LLMPerfEvaluator.create(
            model=model_path,
            num_samples=num_samples,
            samples_path=samples_path,
            warmup=warmup,
            engine_cache_path=engine_output_dir,
            # The options should be identical to the cpp benchmark
            tensor_parallel_size=tp_size,
            build_config=build_config)
        assert evaluator
        report = evaluator.run()
        report.display()

        report_path = Path(report_path_prefix + ".json")
        if report_path.exists():
            for i in range(10000):
                if (Path(f"report_path_prefix{i}.json").exists()):
                    continue
                else:
                    report_path = Path(f"report_path_prefix{i}")
                    break

        report.save_json(report_path)

    def run_gpt_manager_benchmark():
        print_colored(f"Running gptManagerBenchmark ...\n", "bold_green")
        if os.path.isfile(cpp_executable):
            cpp_executable_path = cpp_executable
        else:
            cpp_executable_path = os.path.join(
                os.path.dirname(__file__),
                "../../cpp/build/benchmarks/gptManagerBenchmark")

        command = f"{cpp_executable_path} --engine_dir {engine_output_dir} --type IFB --dataset {samples_path} --warm_up {warmup} --output_csv {report_path_prefix}.cpp.csv --api executor"
        if tp_size > 1:
            command = f"mpirun -n {tp_size} {command}"
        print_colored(f'cpp benchmark command: {command}\n', "grey")
        output = subprocess.run(command,
                                check=True,
                                universal_newlines=True,
                                shell=True,
                                capture_output=True,
                                env=os.environ)  # nosec B603
        print_colored(f'cpp benchmark output: {output.stdout}', "grey")
        if output.stderr:
            print_colored(f'cpp benchmark error: {output.stderr}', "red")

    run_hlapi()
    if cpp_executable:
        run_gpt_manager_benchmark()


@click.command("gridsearch")
@click.option("--model-path", type=str, required=True)
@click.option("--samples-path", type=str, required=True)
@click.option("--reports-root", type=str, required=True)
@click.option("--prune-space-for-debug",
              type=int,
              default=1e8,
              help="Specify the first N cases to test")
@click.option("--max-input-len", type=int, default=1024)
@click.option("--max-seq-len", type=int, default=2048)
@click.option("--max-num-tokens", type=int, default=4096)
@click.option("--tp-size", type=int, default=1)
@click.option("--num-samples", type=int, default=200)
def grid_searcher_main(model_path,
                       samples_path,
                       reports_root,
                       prune_space_for_debug: int,
                       max_input_len: int,
                       max_seq_len: int,
                       max_num_tokens: int,
                       tp_size: int = 1,
                       num_samples: int = 200):
    reports_root = Path(reports_root)

    grid_searcher = GridSearcher(prune_space_for_debug=prune_space_for_debug)

    build_config = BuildConfig(max_seq_len=max_seq_len,
                               max_input_len=max_input_len,
                               max_num_tokens=max_num_tokens)

    grid_searcher.evaluate(model=model_path,
                           samples_path=samples_path,
                           report_dir=reports_root,
                           memory_monitor_interval=1,
                           num_samples=num_samples,
                           tensor_parallel_size=tp_size,
                           build_config=build_config)


if __name__ == '__main__':
    cli.add_command(benchmark_main)
    cli.add_command(grid_searcher_main)
    cli()
