#!/usr/bin/env python3
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
import torch.cuda

from tensorrt_llm import logger
from tensorrt_llm.bindings.executor import CapacitySchedulerPolicy
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
    help=
    "Path to the cpp executable, set it if you want to run the cpp benchmark")
@click.option("--backend", type=str, default=None)
@click.option("--torch-compile/--no-torch-compile", type=bool, default=False)
@click.option("--torch-compile-fullgraph/--no-torch-compile-fullgraph",
              type=bool,
              default=False)
@click.option("--torch-compile-inductor/--no-torch-compile-inductor",
              type=bool,
              default=False)
@click.option("--cuda-graph/--no-cuda-graph", type=bool, default=False)
@click.option("--cuda-graph-padding/--no-cuda-graph-padding",
              type=bool,
              default=False)
@click.option("--cuda-graph-batch-sizes", type=str, default=None)
@click.option("--return-context-logits", type=bool, default=False)
@click.option("--return-generation-logits", type=bool, default=False)
@click.option("--kv-cache-free-gpu-mem-fraction", type=float, default=None)
@click.option("--log-level", type=str, default="warning", show_default=True)
@click.option("--chunked-context/--no-chunked-context", type=bool, default=True)
@click.option("--kv-cache-reuse/--no-kv-cache-reuse", type=bool, default=True)
@click.option("--kv-cache-max-tokens", type=int, default=None)
@click.option("--overlap-scheduler/--no-overlap-scheduler",
              type=bool,
              default=False)
@click.option("--capacity_scheduler_policy",
              type=str,
              default="guaranteed_no_evict")
@click.option("--fp8-kv-cache/--no-fp8-kv-cache", type=bool, default=False)
@click.option("--num-postprocess-workers", type=int, default=0)
@click.option("--postprocess-tokenizer-dir", type=str, default=None)
@click.option("--enable-oai-postprocess", type=bool, default=False)
def benchmark_main(
    model_path: str,
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
    kv_cache_free_gpu_mem_fraction: Optional[float] = None,
    backend: str = None,
    torch_compile: bool = False,
    torch_compile_fullgraph: bool = False,
    torch_compile_inductor: bool = False,
    cuda_graph: bool = False,
    cuda_graph_padding: bool = False,
    cuda_graph_batch_sizes: Optional[str] = None,
    log_level: str = "warning",
    chunked_context: bool = True,
    kv_cache_reuse: bool = True,
    kv_cache_max_tokens: int = None,
    overlap_scheduler: bool = False,
    capacity_scheduler_policy: str = "guaranteed_no_evict",
    fp8_kv_cache: bool = False,
    num_postprocess_workers: int = 0,
    postprocess_tokenizer_dir: Optional[str] = None,
    enable_oai_postprocess: bool = False,
):
    ''' Run the benchmark on LLM API.
    If `cpp_executable_path` is provided, it will run the cpp benchmark as well.
    '''
    logger.set_level(log_level)
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

        nonlocal capacity_scheduler_policy
        if capacity_scheduler_policy == "static_batch":
            capacity_scheduler_policy = CapacitySchedulerPolicy.STATIC_BATCH
        elif capacity_scheduler_policy == "max_utilization":
            capacity_scheduler_policy = CapacitySchedulerPolicy.MAX_UTILIZATION
        else:
            capacity_scheduler_policy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT

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
            kv_cache_free_gpu_mem_fraction=kv_cache_free_gpu_mem_fraction,
            chunked_context=chunked_context,
            enable_kv_cache_reuse=kv_cache_reuse,
            kv_cache_max_tokens=kv_cache_max_tokens,
            capacity_scheduler_policy=capacity_scheduler_policy,

            # postprocess parallel related
            num_postprocess_workers=num_postprocess_workers,
            postprocess_tokenizer_dir=postprocess_tokenizer_dir,
            enable_oai_postprocess=enable_oai_postprocess,
        )
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

        command = f"{cpp_executable_path} --engine_dir {engine_output_dir} --type IFB --dataset {samples_path} --warm_up {warmup} --output_csv {report_path_prefix}.cpp.csv --api executor"
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
        if chunked_context:
            command = f"{command} --enable_chunked_context"
        if not kv_cache_reuse:
            command = f"{command} --enable_kv_cache_reuse=false"
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
    torch.cuda.empty_cache()

    if cpp_executable:
        run_gpt_manager_benchmark()


if __name__ == '__main__':
    cli.add_command(benchmark_main)
    cli()
