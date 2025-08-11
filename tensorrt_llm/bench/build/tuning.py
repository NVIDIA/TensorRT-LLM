from typing import Tuple

import torch

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.bench.build.dataclasses import ModelConfig, NemotronHybridConfig
from .utils import get_device_memory
import math

BYTES_PER_ELEM = {
    QuantAlgo.NO_QUANT: 2.0,
    QuantAlgo.FP8: 1.0,
    QuantAlgo.FP8_BLOCK_SCALES: 1.0,
    QuantAlgo.NVFP4: .5,
}


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
    byte_per_elem = BYTES_PER_ELEM.get(quant_config.quant_algo, 2)
    byte_per_kv_elem = BYTES_PER_ELEM.get(quant_config.kv_cache_quant_algo, 2)

    # Each GPU in TP group has at least 1 kv head
    adjusted_num_kv_heads = max(tp_size, model_config.num_key_value_heads)

    logger.info(
        f"Number of attention layers: {model_config.num_attention_layers}")

    gb_per_token = 2 * model_config.num_attention_layers * adjusted_num_kv_heads \
        * model_config.head_size * byte_per_kv_elem / (1024 ** 3)

    # Number of GPU used for this run.
    n_gpus = tp_size * pp_size
    # Total engine size.
    engine_size = model_config.param_count * byte_per_elem / (1024**3)
    total_gpu_memory = get_device_memory() * n_gpus
    # Available memory to allocate KV cache.
    available_memory = total_gpu_memory - engine_size
    logger.info(f"Estimated engine size: {engine_size:.2f} GB")
    logger.info("Estimated total available memory for KV cache: "
                f"{available_memory:.2f} GB")

    # Calculate max requests in KV cache based on target ISL and OSL.
    target_seq_len = target_input_len + target_output_len
    cache_memory = available_memory * model_config.cache_memory_fraction(
        kv_cache_gpu_mem_fraction)

    bytes_per_elem = BYTES_PER_ELEM.get(QuantAlgo.NO_QUANT)
    if isinstance(model_config, NemotronHybridConfig):
        mamba_ssm_cache_dtype = model_config.mamba_ssm_cache_dtype
        if mamba_ssm_cache_dtype != "auto":
            if str_dtype_to_torch(mamba_ssm_cache_dtype) == torch.float32:
                bytes_per_elem = 4.0

    gb_per_extra_cache = model_config.extra_model_cache_in_gb(
        bytes_per_elem, target_seq_len)
    kv_cache_max_requests = cache_memory / (gb_per_token * target_seq_len +
                                            gb_per_extra_cache)
    extra_cache_memory = gb_per_extra_cache * kv_cache_max_requests
    kv_cache_memory = cache_memory - extra_cache_memory
    kv_cache_max_tokens = kv_cache_memory / gb_per_token

    logger.info(
        f"Estimated total cache memory: {cache_memory:.2f} GB. KV cache: {kv_cache_memory:.2f} GB, Extra cache: {extra_cache_memory:.2f} GB"
    )
    logger.info(f"Estimated kv cache max tokens: {kv_cache_max_tokens:.2f}")
    logger.info("Estimated max number of requests in KV cache memory: "
                f"{kv_cache_max_requests:.2f}")

    # Fine-tune the max batch size and num token setting for performance.
    # For mamba-attn hybrid models, we disable optimistic tuning because the mamba cache leaves less memory for the KV cache
    max_batch_size, max_num_tokens = finetune_setting(
        kv_cache_max_requests,
        target_input_len,
        target_output_len,
        pp_size,
        disable_optimistic_tuning=isinstance(model_config,
                                             NemotronHybridConfig))

    # Functional and performance
    if total_gpu_memory < engine_size:
        raise RuntimeError(
            f"The model requires at least: {engine_size:.2f} GB, "
            f"the total GPU memory of {total_gpu_memory:.2f} is insufficient.\n"
            "----------------------------------------------------------\n"
            f"Estimation based on the following:\n"
            "----------------------------------------------------------\n"
            f"Bytes per Element: {byte_per_elem}\n"
            f"Bytes per KV Element: {byte_per_kv_elem}\n"
            f"Number of GPUs: {n_gpus}\n"
            f"Model Number of KV Heads: {model_config.num_key_value_heads}\n"
            f"Adjusted Number of KV Heads: {adjusted_num_kv_heads}\n"
            f"Head Size: {model_config.head_size}\n"
            f"Number of Hidden Layers: {model_config.num_hidden_layers}\n"
            f"Number of Pipeline Stages: {pp_size}\n"
            f"Number of Tensor Parallel Shards: {tp_size}\n"
            f"Number of Pipeline Parallel Stages: {pp_size}\n"
            f"KV Cache GPU Memory Fraction: {kv_cache_gpu_mem_fraction}\n"
            "----------------------------------------------------------\n")
    if kv_cache_max_requests < 1:
        raise RuntimeError("The amount of KV cache memory is insufficient to "
                           "run this model. Please try with more GPUs.")
    if cache_memory / n_gpus < 10.0:
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
    disable_optimistic_tuning: bool = False,
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
        disable_optimistic_tuning (bool): Whether to disable optimistic tuning.

    Returns:
        Tuple[int, int]: Tuple containing fine-tuned values for engine
        configuration information.
    """
    # Cap total batch size to avoid decoder buffer size becoming too large.
    raw_bs = min(kv_cache_max_requests, 4096) / pp_size
    # Cap num tokens to avoid TRT activation buffer becoming too large.
    raw_token = min(raw_bs * (1 + input_len / output_len), 32768)

    # Fine-tune the max batch size.
    if disable_optimistic_tuning:
        max_bs = 2 * math.floor(raw_bs / 2)
    else:
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
    logger.info(f"Estimated max batch size (after fine-tune): {max_bs}")
    logger.info(f"Estimated max num tokens (after fine-tune): {max_token}")

    return max_bs, max_token
