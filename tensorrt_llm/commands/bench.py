from pathlib import Path

import click

from tensorrt_llm.bench.benchmark.low_latency import latency_command
from tensorrt_llm.bench.benchmark.throughput import throughput_command
from tensorrt_llm.bench.build.build import build_command
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.logger import logger, severity_map


@click.group(name="trtllm-bench", context_settings={'show_default': True})
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    help="The Huggingface name of the model to benchmark.",
)
@click.option(
    "--model_path",
    required=False,
    default=None,
    type=click.Path(writable=False, readable=True, path_type=Path),
    help=
    "Path to a Huggingface checkpoint directory for loading model components.",
)
@click.option(
    "--workspace",
    "-w",
    required=False,
    type=click.Path(writable=True, readable=True, path_type=Path),
    default="/tmp",  # nosec B108
    help="The directory to store benchmarking intermediate files.",
)
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.pass_context
def main(
    ctx,
    model: str,
    model_path: Path,
    workspace: Path,
    log_level: str,
) -> None:
    logger.set_level(log_level)
    ctx.obj = BenchmarkEnvironment(model=model,
                                   checkpoint_path=model_path,
                                   workspace=workspace)

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


main.add_command(build_command)
main.add_command(throughput_command)
main.add_command(latency_command)

if __name__ == "__main__":
    main()
