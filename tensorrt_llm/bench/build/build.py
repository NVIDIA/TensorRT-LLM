from __future__ import annotations
from transformers import AutoConfig

from pathlib import Path
from typing import Tuple, get_args
import click
from click_option_group import AllOptionGroup, optgroup

from tensorrt_llm._torch.pyexecutor.config_utils import is_nemotron_hybrid
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.utils.data import create_dataset_from_stream, initialize_tokenizer
from tensorrt_llm.bench.utils import VALID_QUANT_ALGOS
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.bench.build.dataclasses import ModelConfig, NemotronHybridConfig
from tensorrt_llm.bench.build.tuning import calc_engine_setting

TUNED_QUANTS = {
    QuantAlgo.NVFP4, QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.NO_QUANT, None
}
DEFAULT_MAX_BATCH_SIZE = BuildConfig.max_batch_size
DEFAULT_MAX_NUM_TOKENS = BuildConfig.max_num_tokens


def get_benchmark_engine_settings(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    tp_size: int,
    pp_size: int,
    target_input_len: int,
    target_output_len: int,
    kv_cache_gpu_mem_fraction: float = 0.95,
) -> Tuple[int, int]:
    """ Retrieve benchmark settings for a specific model + configuration.

    Args:
        model_config (ModelConfig): Model specific configurations.
        quant_config (QuantConfig): Quantization specifications.
        tp_size (int): Number of tensor parallel shards.
        pp_size (int): Number of pipeline parallel stages.
        target_input_len (int): Target input length to compile the engine.
        target_output_len (int): Target output length to compile the engine.

    Raises:
        ValueError: When the model_name is not supported.
        RuntimeError: When the tp_size/pp_size configuration is not found.

    Returns:
        Tuple[int, int]: Tuple containing engine configuration information
        for engine build (max_batch_size, max_num_tokens).
    """
    if quant_config.quant_algo in TUNED_QUANTS:
        max_batch_size, max_num_tokens = calc_engine_setting(
            model_config,
            quant_config,
            tp_size,
            pp_size,
            target_input_len,
            target_output_len,
            kv_cache_gpu_mem_fraction,
        )
    else:
        max_batch_size = DEFAULT_MAX_BATCH_SIZE
        max_num_tokens = DEFAULT_MAX_NUM_TOKENS
        logger.warning(
            f"Using default settings because quant_algo not supported. "
            f"max_batch_size: {max_batch_size}, max_num_tokens: {max_num_tokens}."
        )

    if max_batch_size <= 0 or max_num_tokens <= 0:
        raise RuntimeError(f"Unable to obtain correct settings for benchmark.")

    return max_batch_size, max_num_tokens


def get_model_config(model_name: str, model_path: Path = None) -> ModelConfig:
    """ Obtain the model-related parameters from Hugging Face.
    Args:
        model_name (str): Huggingface model name.
        model_path (Path): Path to a local Huggingface checkpoint.

    Raises:
        ValueError: When model is not supported.
    """
    if is_nemotron_hybrid(
            AutoConfig.from_pretrained(model_path or model_name,
                                       trust_remote_code=True)):
        return NemotronHybridConfig.from_hf(model_name, model_path)
    return ModelConfig.from_hf(model_name, model_path)


def apply_build_mode_settings(params):
    """ Validate engine build options and update the necessary values for engine
        build settings.
    """
    dataset_path = params.get("dataset")
    max_batch_size = params.get("max_batch_size")
    target_input_len = params.get("target_input_len")
    max_seq_len = params.get("max_seq_len")
    tp_size = params.get("tp_size")
    pp_size = params.get("pp_size")

    # Check of engine build method. User must choose one engine build option.
    build_options = [dataset_path, max_batch_size, target_input_len]
    # If no engine build option is provided, fall back to build engine with
    # TRT-LLM's default max_batch_size and max_num_tokens.
    if sum([bool(opt) for opt in build_options]) == 0:
        logger.warning(
            "No engine build option is selected, use TRT-LLM default "
            "max_batch_size and max_num_tokens to build the engine.")
        params['max_batch_size'] = DEFAULT_MAX_BATCH_SIZE
        params['max_num_tokens'] = DEFAULT_MAX_NUM_TOKENS
    elif sum([bool(opt) for opt in build_options]) > 1:
        raise ValueError("Multiple engine build options detected, please "
                         "choose only one engine build option. Exiting.")

    # Check for supported parallelism mappings: only world size <= 8 for now.
    if tp_size * pp_size > 8:
        raise ValueError(
            f"Parallelism mapping of TP{tp_size}-PP{pp_size} is "
            "currently unsupported. Please try with a mapping with <=8 GPUs.")

    # If dataset is not specified, max_seq_len must be provided.
    if not dataset_path and not max_seq_len:
        raise ValueError("Unspecified max_seq_len for engine build. Exiting.")


@click.command(name="build")
@optgroup.group("Engine Configuration",
                help="Configuration of the TensorRT LLM engine.")
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
@optgroup.option(
    "--max_seq_len",
    default=None,
    type=click.IntRange(min=1),
    help="Maximum total length of one request, including prompt and outputs.",
)
@optgroup.option(
    "--no_weights_loading",
    type=bool,
    default=False,
    help=
    "Do not load the weights from the checkpoint. Use dummy weights instead.")
@optgroup.option(
    "--trust_remote_code",
    type=bool,
    default=False,
    help=
    "Trust remote code for the HF models that are not natively implemented in the transformers library. "
    "This is needed when using LLM API when loading the HF config to build the engine."
)
@optgroup.group(
    "Build Engine with Dataset Information",
    cls=AllOptionGroup,
    help="Optimize engine build parameters with user-specified dataset "
    "statistics, e.g., average input/output length, max sequence length.",
)
@optgroup.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Dataset file to extract the sequence statistics for engine build.",
)
@optgroup.group(
    "Build Engine with IFB Scheduler Limits",
    cls=AllOptionGroup,
    help="Optimize engine build parameters with user-specified inflight "
    "batching scheduler settings.",
)
@optgroup.option(
    "--max_batch_size",
    default=None,
    type=click.IntRange(min=1),
    help="Maximum number of requests that the engine can schedule.",
)
@optgroup.option(
    "--max_num_tokens",
    default=None,
    type=click.IntRange(min=1),
    help="Maximum number of batched tokens the engine can schedule.",
)
@optgroup.group(
    "[Experimental Feature] Build Engine with Tuning Heuristics Hints",
    cls=AllOptionGroup,
    help="Optimize engine build parameters with user-specified target "
    "sequence length information.",
)
@optgroup.option(
    "--target_input_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) input length for tuning heuristics.",
)
@optgroup.option(
    "--target_output_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) sequence length for tuning heuristics.",
)
@click.pass_obj
def build_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Build engines for benchmarking."""

    apply_build_mode_settings(params)
    # Collect configuration parameters from CLI parameters.
    tp_size = params.get("tp_size")
    pp_size = params.get("pp_size")
    quantization = params.get("quantization")
    max_seq_len: int = params.get("max_seq_len")
    # Dataset options
    dataset_path: Path = params.get("dataset")
    # IFB scheduler options
    max_batch_size = params.get("max_batch_size")
    max_num_tokens = params.get("max_num_tokens")
    # Tuning heuristics options
    target_input_len: int = params.get("target_input_len")
    target_output_len: int = params.get("target_output_len")

    load_format = "dummy" if params.get("no_weights_loading") else "auto"
    trust_remote_code: bool = params.get("trust_remote_code")
    model_name = bench_env.model
    checkpoint_path = bench_env.checkpoint_path or model_name
    model_config = get_model_config(model_name, bench_env.checkpoint_path)
    engine_dir = Path(bench_env.workspace, model_name,
                      f"tp_{tp_size}_pp_{pp_size}")

    # Set the compute quantization.
    quant_algo = QuantAlgo(quantization) if quantization is not None else None
    quant_config = QuantConfig(quant_algo=quant_algo)
    # If the quantization is NVFP4 or FP8, force the KV cache dtype to FP8.
    if quant_algo in [QuantAlgo.NVFP4, QuantAlgo.FP8]:
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(checkpoint_path)
    # If we receive dataset from a path or stdin, parse and gather metadata.
    if dataset_path:
        logger.info("Found dataset.")
        # Dataset Loading and Preparation
        with open(dataset_path, "r") as dataset:
            metadata, _ = create_dataset_from_stream(
                tokenizer,
                dataset,
            )
        max_seq_len = metadata.max_sequence_length
        target_input_len = metadata.avg_isl
        target_output_len = metadata.avg_osl
        logger.info(metadata.get_summary_for_print())

    # Use user-specified engine settings if provided.
    if max_batch_size and max_num_tokens:
        logger.info("Use user-provided max batch size and max num tokens for "
                    "engine build and benchmark.")
    # If not provided, use the engine setting provided by trtllm-bench.
    else:
        logger.info(
            "Max batch size and max num tokens are not provided, "
            "use tuning heuristics or pre-defined setting from trtllm-bench.")
        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            model_config,
            quant_config,
            tp_size,
            pp_size,
            target_input_len,
            target_output_len,
        )

    # Construct a TRT-LLM build config.
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_seq_len=max_seq_len,
                               max_num_tokens=max_num_tokens)

    build_config.plugin_config.dtype = model_config.dtype
    # Enable multiple profiles and paged context FMHA.
    build_config.plugin_config.multiple_profiles = True
    # build_config.plugin_config._reduce_fusion = True

    # Enable FHMA, and FP8 FMHA if NVFP4 or FP8 quantization is enabled.
    # TODO: Revisit, there is an issue with enabling FHMA. If only
    # paged FMHA is enabled with NVFP4 or FP8 quantization, the Builder
    # will not enable the FP8 FMHA.
    build_config.plugin_config.use_paged_context_fmha = True
    if quant_algo in [QuantAlgo.NVFP4, QuantAlgo.FP8]:
        build_config.plugin_config.use_fp8_context_fmha = True
    # Enable nvfp4 gemm_plugin explicitly for Blackwell
    if quant_algo == QuantAlgo.NVFP4:
        build_config.plugin_config.gemm_plugin = "nvfp4"

    # Build the LLM engine with the LLMAPI.
    llm = LLM(checkpoint_path,
              tokenizer,
              dtype=model_config.dtype,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              build_config=build_config,
              quant_config=quant_config,
              workspace=str(bench_env.workspace),
              load_format=load_format,
              trust_remote_code=trust_remote_code)
    # Save the engine.
    llm.save(engine_dir)
    llm.shutdown()

    logger.info(
        "\n===========================================================\n"
        "= ENGINE BUILD INFO\n"
        "===========================================================\n"
        f"Model Name:\t\t{bench_env.model}\n"
        f"Model Path:\t\t{bench_env.checkpoint_path}\n"
        f"Workspace Directory:\t{bench_env.workspace}\n"
        f"Engine Directory:\t{engine_dir}\n\n"
        "===========================================================\n"
        "= ENGINE CONFIGURATION DETAILS\n"
        "===========================================================\n"
        f"Max Sequence Length:\t\t{max_seq_len}\n"
        f"Max Batch Size:\t\t\t{max_batch_size}\n"
        f"Max Num Tokens:\t\t\t{max_num_tokens}\n"
        f"Quantization:\t\t\t{quant_config.quant_algo}\n"
        f"KV Cache Dtype:\t\t\t{quant_config.kv_cache_quant_algo}\n"
        "===========================================================\n")

    logger.info(
        "\n\n===========================================================\n"
        f"ENGINE SAVED: {engine_dir}\n"
        "===========================================================\n")
