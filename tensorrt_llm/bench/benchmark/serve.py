from __future__ import annotations

from pathlib import Path

import click
from click_option_group import optgroup
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (
    TuningConstraints, get_settings_from_engine, get_settings)
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.serve import launch_server
from tensorrt_llm.llmapi import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger
# isort: on


@click.command(name="serve")
@optgroup.group("Engine run configuration.",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option("--backend",
                 type=click.Choice(["pytorch", "tensorrt", "_autodeploy"]),
                 default="pytorch",
                 help="The backend to use when running benchmarking.")
@optgroup.option(
    "--extra_llm_api_options",
    type=str,
    default=None,
    help=
    "Path to a YAML file that overwrites the parameters specified by trtllm-bench."
)
@optgroup.option(
    "--max_batch_size",
    type=int,
    help="Maximum runtime batch size to run the engine with.",
)
@optgroup.option(
    "--max_num_tokens",
    type=int,
    help="Maximum runtime tokens that an engine can accept.",
)
@optgroup.option(
    "--kv_cache_free_gpu_mem_fraction",
    type=float,
    default=.90,
    help="The percentage of memory to use for KV Cache after model load.",
)
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum number of beams for beam search decoding.")
@optgroup.group(
    "Engine Input Configuration",
    help="Input configuration for driving the engine.",
)
@optgroup.option(
    "--target_input_len",
    required=True,
    type=click.IntRange(min=1),
    help="Target (average) input length for tuning heuristics.",
)
@optgroup.option(
    "--target_output_len",
    required=True,
    type=click.IntRange(min=1),
    help="Target (average) sequence length for tuning heuristics.",
)
@optgroup.option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum sequence length.",
)
@optgroup.option(
    "--max_isl",
    type=int,
    required=True,
    help="Maximum input length.",
)
@optgroup.option(
    "--max_osl",
    type=int,
    required=True,
    help="Maximum output length.",
)
@optgroup.group(
    "World Configuration",
    help="Options for configuring the backend multi-GPU world.",
)
@optgroup.option(
    "--tp",
    type=int,
    default=1,
    help="tensor parallelism size",
)
@optgroup.option(
    "--pp",
    type=int,
    default=1,
    help="pipeline parallelism size",
)
@optgroup.option(
    "--ep",
    type=int,
    default=None,
    help="expert parallelism size",
)
@optgroup.option(
    "--cluster_size",
    type=int,
    default=None,
    help="expert cluster parallelism size",
)
@optgroup.option(
    "--enable_chunked_context",
    is_flag=True,
    default=False,
    help="Enable chunking in prefill stage for enhanced throughput benchmark.",
)
@optgroup.option(
    "--scheduler_policy",
    type=click.Choice(["guaranteed_no_evict", "max_utilization"]),
    default="guaranteed_no_evict",
    help=
    "KV cache scheduler policy: guaranteed_no_evict prevents request eviction, max_utilization optimizes for throughput.",
)
@optgroup.group(
    "Server Configuration",
    help="Options for configuring the server.",
)
@optgroup.option("--host",
                 type=str,
                 default="localhost",
                 help="Hostname of the server.")
@optgroup.option("--port", type=int, default=8000, help="Port of the server.")
@click.pass_obj
def serve_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Stand up a TRT-LLM OpenAI server with tuned settings."""
    logger.info("Preparing to run serve...")

    # Server configuration
    host: str = params.pop("host")
    port: int = params.pop("port")

    # Constraints
    max_seq_len: int = params.pop("max_seq_len")
    target_input_len: int = params.pop("target_input_len")
    target_output_len: int = params.pop("target_output_len")
    max_isl: int = params.pop("max_isl")
    max_osl: int = params.pop("max_osl")

    model: str = bench_env.model
    checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    backend: str = params.get("backend")
    tuning_constraints = TuningConstraints(
        average_isl=target_input_len,
        average_osl=target_output_len,
        maximum_isl=max_isl,
        maximum_osl=max_osl,
        max_sequence_length=max_seq_len,
    )

    # Runtime kwargs and option tracking.
    kwargs = {}
    # Engine configuration parsing
    if backend and backend.lower() in ["pytorch", "_autodeploy"]:
        # If we're dealing with a model name, perform a snapshot download to
        # make sure we have a local copy of the model.
        if bench_env.checkpoint_path is None:
            snapshot_download(model)

        exec_settings = get_settings(params, tuning_constraints,
                                     bench_env.model, bench_env.checkpoint_path)
        kwargs_max_sql = max_seq_len
        logger.info(f"Setting PyTorch max sequence length to {kwargs_max_sql}")
        kwargs["max_seq_len"] = kwargs_max_sql
    elif backend.lower() == "tensorrt":
        assert max_seq_len is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(engine_dir)
        engine_max_seq_len = build_cfg["max_seq_len"]
        kwargs["max_seq_len"] = engine_max_seq_len
    else:
        raise RuntimeError(
            f"Invalid backend: {backend}, please use one of the following: "
            "pytorch, tensorrt, _autodeploy.")

    exec_settings["model"] = model
    engine_bs = exec_settings["settings_config"]["max_batch_size"]
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]

    # Runtime Options
    runtime_max_bs = params.pop("max_batch_size")
    runtime_max_tokens = params.pop("max_num_tokens")
    runtime_max_bs = runtime_max_bs or engine_bs
    runtime_max_tokens = runtime_max_tokens or engine_tokens
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    beam_width = params.pop("max_beam_width")
    enable_chunked_context: bool = params.pop("enable_chunked_context")
    scheduler_policy: str = params.pop("scheduler_policy")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"][
        "scheduler_policy"] = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT if scheduler_policy == "guaranteed_no_evict" else CapacitySchedulerPolicy.MAX_UTILIZATION
    exec_settings["settings_config"]["chunking"] = enable_chunked_context

    # Dynamic runtime features.
    exec_settings["settings_config"]["dynamic_max_batch_size"] = True

    # LlmArgs
    exec_settings["extra_llm_api_options"] = params.pop("extra_llm_api_options")

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)

    logger.info("Setting up throughput benchmark.")
    kwargs = kwargs | runtime_config.get_llm_args()
    kwargs['backend'] = backend
    kwargs['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint_path)
    kwargs['postprocess_tokenizer_dir'] = None
    kwargs['trust_remote_code'] = True
    kwargs['skip_tokenizer_init'] = False
    kwargs["num_postprocess_workers"] = 10

    if runtime_config.backend == 'pytorch':
        if kwargs.pop("extended_runtime_perf_knob_config", None):
            logger.warning(
                "Ignore extended_runtime_perf_knob_config for pytorch backend.")
    elif runtime_config.backend == "_autodeploy":
        if kwargs.pop("extended_runtime_perf_knob_config", None):
            logger.warning(
                "Ignore extended_runtime_perf_knob_config for _autodeploy backend."
            )

    launch_server(host, port, kwargs)
