from pathlib import Path
from typing import get_args

import click
from static import static_benchmark
from utils import (VALID_CACHE_DTYPES, VALID_COMPUTE_DTYPES, VALID_MODELS,
                   VALID_QUANT_ALGOS)
from utils.dataclasses import BenchmarkConfig


@click.group(context_settings={'show_default': True})
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(tuple(get_args(VALID_MODELS))),
    help="The Huggingface name of the model to benchmark.",
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
     ))
@click.option(
    "--workspace",
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
@click.pass_context
def benchmark(
    ctx,
    model: str,
    workspace: Path,
    dtype: str,
    kv_dtype: str,
    quantization: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
):
    """Utility for using TRT-LLM for benchmarking networks from Huggingface."""
    ctx.obj = BenchmarkConfig(
        model=model,
        workspace=Path(workspace),
        dtype=dtype,
        cache_dtype=kv_dtype,
        quantization=quantization,
        tensor_parallel=tensor_parallel_size,
        pipeline_parallel=pipeline_parallel_size,
    )

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


# Add nested subcommands to main benchmark CLI.
benchmark.add_command(static_benchmark)

if __name__ == "__main__":
    benchmark()
