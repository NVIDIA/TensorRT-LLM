from pathlib import Path
from typing import get_args

import click
from ifb import executor_benchmark
from static import static_benchmark
from utils import VALID_CACHE_DTYPES, VALID_COMPUTE_DTYPES, VALID_QUANT_ALGOS
from utils.dataclasses import BenchmarkConfig


@click.group(context_settings={'show_default': True})
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    help="The Huggingface name of the model to benchmark.",
)
@click.option(
    "--max-batch-size",
    hidden=True,
    default=0,
    type=int,
    help="Maximum batch size to build the benchmark engine with.",
)
@click.option(
    "--kv-dtype",
    type=click.Choice(tuple(get_args(VALID_CACHE_DTYPES))),
    default="float16",
    help="The dtype to store the KV Cache in.",
)
@click.option(
    "--dtype",
    type=click.Choice(tuple(get_args(VALID_COMPUTE_DTYPES))),
    default="float16",
    help="Activation and plugin data type.",
)
@click.option(
    "--quantization",
    "-q",
    type=click.Choice(tuple(get_args(VALID_QUANT_ALGOS))),
    default="None",
    help=
    ("The quantization algorithm to be used when benchmarking. See the "
     "documentations for more information.\n"
     "  - https://nvidia.github.io/TensorRT-LLM/precision.html"
     "  - https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md"
     ),
)
@click.option(
    "--workspace",
    "-w",
    required=False,
    type=click.Path(writable=True, readable=True),
    default="/tmp",
    help="The directory to store benchmarking intermediate files.",
)
@click.option(
    "--tensor-parallel-size",
    "-tp",
    type=int,
    default=1,
    required=False,
    help="Number of tensor parallel shards to run the benchmark with.",
)
@click.option(
    "--pipeline-parallel-size",
    "-pp",
    type=int,
    default=1,
    required=False,
    help="Number of pipeline parallel shards to run the benchmark with.",
)
@click.option(
    "--kv-cache-free-gpu-mem-fraction",
    "-kv-mem",
    type=float,
    default=0.98,
    help="The percentage of free memory that the KV Cache is allowed to occupy.",
)
@click.option(
    "--build-opts",
    type=str,
    default="",
    required=False,
    hidden=True,
    help="Passthrough options for trtllm-build to fine-tuning build commands.")
@click.pass_context
def benchmark(
    ctx,
    model: str,
    max_batch_size: int,
    workspace: Path,
    dtype: str,
    kv_dtype: str,
    quantization: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    kv_cache_free_gpu_mem_fraction: float,
    build_opts: str,
):
    """Utility for using TRT-LLM for benchmarking networks from Huggingface."""
    ctx.obj = BenchmarkConfig(
        model=model,
        max_batch_size=max_batch_size,
        workspace=Path(workspace),
        dtype=dtype,
        cache_dtype=kv_dtype,
        quantization=quantization,
        tensor_parallel=tensor_parallel_size,
        pipeline_parallel=pipeline_parallel_size,
        kv_cache_mem_percentage=kv_cache_free_gpu_mem_fraction,
        build_overrides=build_opts.split(),
    )

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


# Add nested subcommands to main benchmark CLI.
benchmark.add_command(static_benchmark)
benchmark.add_command(executor_benchmark)

if __name__ == "__main__":
    benchmark()
