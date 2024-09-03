from __future__ import annotations

from pathlib import Path
from select import select
from sys import stdin
from typing import Dict, get_args
import click
from click_option_group import AllOptionGroup, optgroup, RequiredMutuallyExclusiveOptionGroup
from transformers import PretrainedConfig as HFPretrainedConfig
import yaml

from tensorrt_llm.bench.dataclasses import BenchmarkEnvironment
from tensorrt_llm.bench.utils.data import create_dataset_from_stream, initialize_tokenizer
from tensorrt_llm.bench.utils import (VALID_QUANT_ALGOS, VALID_COMPUTE_DTYPES)
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.hlapi import LLM
from tensorrt_llm.hlapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from .utils import DEFAULT_HF_MODEL_DIRS


def derive_model_name(model_name):
    model_dir = Path(model_name)
    if model_dir.exists() and model_dir.is_dir():
        hf_config = HFPretrainedConfig.from_pretrained(model_dir)
        for arch in hf_config.architectures:
            if arch in DEFAULT_HF_MODEL_DIRS.keys():
                model_name = DEFAULT_HF_MODEL_DIRS[arch]
    return model_name


def get_benchmark_engine_settings(
    model_name: str,
    tp_size: int,
    pp_size: int,
    max_seq_len: int,
) -> Dict[str, int]:
    """Retrieve benchmark settings for a specific model + configuration.

    Args:
        model_name (str): Huggingface model name.
        tp_size (int): Number of tensor parallel shards.
        pp_size (int): Number of pipeline parallel stages.
        max_seq_len (int): The maximum sequence length to compile the engine.

    Raises:
        ValueError: When the model_name is not supported.
        RuntimeError: When the tp_size/pp_size configuration is not found.

    Returns:
        Dict[str, int]: Dictionary containing engine configuration information
        for engine build (max_num_tokens, max_batch_size).
    """
    # Load up reference configurations so that we can set the appropriate
    # settings.
    settings_yml = Path(__file__).parent / "benchmark_config.yml"
    with open(settings_yml, "r") as config:
        configs = yaml.safe_load(config)

    model_name = derive_model_name(model_name)
    # Check that the model is a supported benchmark model.
    if model_name not in configs:
        raise ValueError(
            f"'{model_name}' is not a model that is configured for benchmarking."
        )
    # Try and load the configuration TP x PP. If not valid, inform the user.
    try:
        model_configs = configs[model_name][f"tp{tp_size}_pp{pp_size}"]
        config = model_configs.get(max_seq_len, None)
        config = config if config is not None else model_configs.get("general")
    except KeyError:
        raise RuntimeError(
            f"TP-{tp_size} x PP-{pp_size} is not a supported configuration."
            "Please specify a valid benchmark configuration.")

    return config


@click.command(name="build")
@optgroup.group("Engine Configuration",
                help="Configuration of the TensorRT-LLM engine.")
@optgroup.option(
    "--tp_size",
    "-tp",
    type=int,
    default=1,
    required=False,
    help="Number of tensor parallel shards to run the benchmark with.",
)
@optgroup.option(
    "--pp_size",
    "-pp",
    type=int,
    default=1,
    required=False,
    help="Number of pipeline parallel shards to run the benchmark with.",
)
@optgroup.option(
    "--dtype",
    type=click.Choice(tuple(get_args(VALID_COMPUTE_DTYPES))),
    default="auto",
    required=False,
    help="Activation and plugin data type.",
)
@optgroup.option(
    "--quantization",
    "-q",
    type=click.Choice(tuple(get_args(VALID_QUANT_ALGOS))),
    default=None,
    help=
    ("The quantization algorithm to be used when benchmarking. See the "
     "documentations for more information.\n"
     "  - https://nvidia.github.io/TensorRT-LLM/precision.html"
     "  - https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md"
     ),
)
@optgroup.group(
    "Engine IFB Engine Limits",
    cls=AllOptionGroup,
    help="Runtime inflight batching scheduler limits.",
)
@optgroup.option(
    "--max_batch_size",
    default=None,
    type=int,
    help="Maximum batch size to build the benchmark engine with.",
)
@optgroup.option(
    "--max_num_tokens",
    type=int,
    default=None,
    help="Maximumn number of tokens the engine can accept.",
)
@optgroup.group(
    "Engine Input Configuration",
    cls=RequiredMutuallyExclusiveOptionGroup,
    help="Input settings for configuring engine limits.",
)
@optgroup.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Pass in a dataset file for parsing instead of stdin.",
)
@optgroup.option("--max_seq_length",
                 type=click.IntRange(min=1),
                 default=None,
                 help="Fixed maximum sequence length for engine build.")
@click.pass_obj
def build_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Build engines for benchmarking."""
    logger.set_level("info")

    # Collect configuration parameters from CLI parameters.
    tp_size = params.get("tp_size")
    pp_size = params.get("pp_size")
    dtype = params.get("dtype")
    quantization = params.pop("quantization")
    max_num_tokens = params.pop("max_num_tokens")
    max_batch_size = params.pop("max_batch_size")

    # Dataset options
    dataset_path: Path = params.pop("dataset")
    max_seq_len: int = params.pop("max_seq_length")
    data_on_stdin: bool = bool(len(select([
        stdin,
    ], [], [], 0.0)[0]))

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(bench_env.model)

    # If we are receiving data from a path or stdin, parse and gather metadata.
    if dataset_path or data_on_stdin:
        logger.info("Found dataset.")
        # Cannot set the data file path and pipe in from stdin. Choose one.
        if dataset_path is not None and data_on_stdin:
            raise ValueError(
                "Cannot provide a dataset on both stdin and by --dataset "
                "option. Please pick one.")
        stream = stdin if data_on_stdin else open(dataset_path, "r")
        # Parse the dataset from stdin and return it plus its metadata.
        metadata, _ = \
            create_dataset_from_stream(tokenizer, stream=stream)
        # The max sequence length option for build is the sum of max osl + isl.
        max_seq_len = metadata.max_sequence_length
        logger.info(metadata.get_summary_for_print())

    # We have a specified ISL:OSL combination.
    elif max_seq_len is None:
        raise RuntimeError("Unknown input configuration. Exiting.")

    # Get the config for the engine
    config = get_benchmark_engine_settings(bench_env.model, tp_size, pp_size,
                                           max_seq_len)

    # If specified on the command line, override max batch size or max num
    # tokens from baseline config.
    max_batch_size = max_batch_size if max_batch_size is not None else config[
        "max_batch_size"]
    max_num_tokens = max_num_tokens if max_num_tokens is not None else config[
        "max_num_tokens"]

    # Construct a TRT-LLM build config.
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_seq_len=max_seq_len,
                               max_num_tokens=max_num_tokens)

    # Set the compute quantization.
    quant_algo = QuantAlgo(quantization) if quantization is not None else None
    quant_config = QuantConfig()
    quant_config.quant_algo = quant_algo
    # If the quantization is FP8, force the KV cache dtype to FP8.
    quant_config.kv_cache_quant_algo = quant_algo.value \
         if quant_algo == QuantAlgo.FP8 else None

    # Enable multiple profiles and paged context FMHA.
    build_config.plugin_config.multiple_profiles = True
    # build_config.plugin_config._reduce_fusion = True

    # Enable FHMA, and FP8 FMHA if FP8 quantization is enabled.
    # TODO: Revisit, there is an issue with enabling FHMA. If only
    # paged FMHA is enabled with FP8 quantization, the Builder
    # will not enable the FP8 FMHA.
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.use_fp8_context_fmha = True \
         if quant_algo == QuantAlgo.FP8 else False

    # Construct the engine path and report the engine metadata.
    model_name = derive_model_name(bench_env.model)
    engine_dir = Path(bench_env.workspace, model_name,
                      f"tp_{tp_size}_pp_{pp_size}")

    logger.info(
        "\n===========================================================\n"
        "= ENGINE BUILD INFO\n"
        "===========================================================\n"
        f"Model Name:\t\t{bench_env.model}\n"
        f"Workspace Directory:\t{bench_env.workspace}\n"
        f"Engine Directory:\t{engine_dir}\n\n"
        "===========================================================\n"
        "= ENGINE CONFIGURATION DETAILS\n"
        "===========================================================\n"
        f"Max Sequence Length:\t\t{max_seq_len}\n"
        f"Max Batch Size:\t\t\t{max_batch_size}\n"
        f"Max Num Tokens:\t\t\t{max_num_tokens}\n"
        f"Quantization:\t\t\t{quantization}\n"
        "===========================================================\n")

    # Build the LLM engine with the HLAPI.
    logger.set_level("error")
    llm = LLM(bench_env.model,
              tokenizer,
              dtype=dtype,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              build_config=build_config,
              quant_config=quant_config)
    # Save the engine.
    llm.save(engine_dir)
    llm._shutdown()
    logger.set_level("info")
    logger.info(
        "\n\n===========================================================\n"
        f"ENGINE SAVED: {engine_dir}\n"
        "===========================================================\n")
