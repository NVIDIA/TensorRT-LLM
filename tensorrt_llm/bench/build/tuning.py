from typing import Tuple

from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.bench.build.dataclasses import ModelConfig
from .utils import get_device_memory
import math


def calc_engine_setting(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    tp_size: int,
    pp_size: int,
    target_input_len: int,
    target_output_len: int,
    kv_cache_gpu_mem_fraction: float = 0.95,
) -> Tuple[int, int]:
    """ Calculate the engine build settings (max batch size and max num tokens)
        for a specific model + parallelism mapping + dataset configuration.
        trtllm-bench sets a slightly optimistic upper bound for max batch size
        and max num tokens to avoid over-allocation of memory in activation,
        runtime, and decoder buffers. In runtime, TRT-LLM relies on its runtime
        tuning features to adjust the runtime max batch size according to
        incoming traffic.

    Args:
        model_config (ModelConfig): Model specific configurations.
        quant_config (QuantConfig): Quantization specifications.
        tp_size (int): Number of tensor parallel shards.
        pp_size (int): Number of pipeline parallel stages.
        target_input_len (int): Target input length to compile the engine.
        target_output_len (int): Target output length to compile the engine.
        kv_cache_gpu_mem_fraction (float): Fraction of free memory to allocate
            for KV cache.

    Raises:
        RuntimeError: When the number of GPUs or amount of KV cache is unable to
            support the model.

    Returns:
        Tuple[int, int]: Tuple containing engine configuration information for
        engine build (max_num_tokens, max_batch_size).
    """
    byte_per_elem = 1 if quant_config.quant_algo == QuantAlgo.FP8 else 2
    byte_per_kv_elem = 1 if quant_config.kv_cache_quant_algo == QuantAlgo.FP8 else 2

    # Model specific calculation
    param_count = model_config.param_count / (1000**3)
    # Each GPU in TP group has at least 1 kv head
    adjusted_num_kv_heads = max(tp_size, model_config.num_key_value_heads)
    byte_per_token = 2 * model_config.num_hidden_layers * adjusted_num_kv_heads \
        * model_config.head_size * byte_per_kv_elem / (1024 ** 3)

    # Number of GPU used for this run.
    n_gpus = tp_size * pp_size
    # Total engine size.
    engine_size = param_count * byte_per_elem
    total_gpu_memory = get_device_memory() * n_gpus
    # Available memory to allocate KV cache.
    available_memory = total_gpu_memory - engine_size
    logger.info("Estimated total available memory for KV cache: "
                f"{available_memory:.2f} GB")

    # Calculate max requests in KV cache based on target ISL and OSL.
    kv_cache_memory = available_memory * kv_cache_gpu_mem_fraction
    kv_cache_max_tokens = kv_cache_memory / byte_per_token
    kv_cache_max_requests = kv_cache_max_tokens / (target_input_len +
                                                   target_output_len)
    logger.info(f"Estimated total KV cache memory: {kv_cache_memory:.2f} GB")
    logger.info("Estimated max number of requests in KV cache memory: "
                f"{kv_cache_max_requests:.2f}")

    # Fine-tune the max batch size and num token setting for performance.
    max_batch_size, max_num_tokens = finetune_setting(kv_cache_max_requests,
                                                      target_input_len,
                                                      target_output_len,
                                                      pp_size)

    # Functional and performance
    if total_gpu_memory < engine_size:
        raise RuntimeError(
            f"The model requires at least: {engine_size:.2f} GB, "
            f"the total GPU memory of {total_gpu_memory:.2f} is insufficient.")
    if kv_cache_max_requests < 1:
        raise RuntimeError("The amount of KV cache memory is insufficient to "
                           "run this model. Please try with more GPUs.")
    if kv_cache_memory / n_gpus < 10.0:
        logger.warning(
            f"The KV cache memory per GPU is less than 10 GB. "
            "Performance may be undesirable. Please consider using a different "
            "mapping or more GPUs.")
    if kv_cache_max_requests < 32:
        logger.warning(
            f"The maximum number of requests in the KV cache is too "
            "small. Performance may be undesirable. Please consider using more "
            "GPUs or a different mapping to process more concurrent requests.")

    return max_batch_size, max_num_tokens


def finetune_setting(
    kv_cache_max_requests: float,
    input_len: int,
    output_len: int,
    pp_size: int,
) -> Tuple[int, int]:
    """ Calculate and fine-tune the engine build settings (max batch size and
        max num tokens). Both max batch size and max num tokens are fine-tuned
        to be slightly optimistic.

    Args:
        kv_cache_max_requests (float): Max number of requests that can fits in
            the available KV cache memory.
        input_len (int): Input sequence length to compile the engine.
        output_len (int): Output sequence length to compile the engine.
        pp_size (int): Number of pipeline parallel stages.

    Returns:
        Tuple[int, int]: Tuple containing fine-tuned values for engine
        configuration information.
    """
    # Cap total batch size to avoid decoder buffer size becoming too large.
    raw_bs = min(kv_cache_max_requests, 4096) / pp_size
    # Cap num tokens to avoid TRT activation buffer becoming too large.
    raw_token = min(raw_bs * (1 + input_len / output_len), 32768)

    # Fine-tune the max batch size.
    # Set min BS to be 64.
    if raw_bs < 256:
        max_bs = max(64, 32 * math.ceil(raw_bs / 32))
    elif raw_bs < 1024:
        max_bs = 128 * math.ceil(raw_bs / 128)
    else:
        max_bs = 256 * math.ceil(raw_bs / 256)

    # Fine-tune the max num tokens.
    # Set min to 2048 to ensure Ctx/Gen overlap efficiency
    if raw_token < 4096:
        max_token = max(2048, 256 * math.ceil(raw_token / 256))
    elif raw_token < 8192:
        max_token = 512 * math.ceil(raw_token / 512)
    else:
        max_token = 1024 * math.ceil(raw_token / 1024)

    logger.debug(f"Estimated max batch size (before fine-tune): "
                 f"{kv_cache_max_requests / pp_size:.2f}")
    logger.debug(
        f"Estimated max num tokens (before fine-tune): "
        f"{kv_cache_max_requests / pp_size * (1 + input_len / output_len) :.2f}"
    )
    logger.debug(f"Estimated max batch size (after fine-tune): {max_bs}")
    logger.debug(f"Estimated max num tokens (after fine-tune): {max_token}")

    return max_bs, max_token
