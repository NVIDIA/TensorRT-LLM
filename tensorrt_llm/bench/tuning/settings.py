from __future__ import annotations

from pathlib import Path
from typing import Tuple

from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_nemotron_hybrid,
    is_qwen3_hybrid,
    load_pretrained_config,
)
from tensorrt_llm.bench.tuning.dataclasses import (
    ModelConfig,
    NemotronHybridConfig,
    Qwen3HybridConfig,
)
from tensorrt_llm.bench.tuning.heuristics import calc_engine_setting
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

TUNED_QUANTS = {
    QuantAlgo.NVFP4,
    QuantAlgo.FP8,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.NO_QUANT,
    None,
}
# Sourced from TorchLlmArgs so the bench defaults track the args-class field
# defaults and can't drift.
DEFAULT_MAX_BATCH_SIZE = TorchLlmArgs.model_fields["max_batch_size"].default
DEFAULT_MAX_NUM_TOKENS = TorchLlmArgs.model_fields["max_num_tokens"].default


def get_benchmark_engine_settings(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    tp_size: int,
    pp_size: int,
    target_input_len: int,
    target_output_len: int,
    kv_cache_gpu_mem_fraction: float = 0.95,
    enable_attention_dp: bool = False,
) -> Tuple[int, int]:
    """Retrieve benchmark settings for a specific model + configuration.

    Args:
        model_config (ModelConfig): Model specific configurations.
        quant_config (QuantConfig): Quantization specifications.
        tp_size (int): Number of tensor parallel shards.
        pp_size (int): Number of pipeline parallel stages.
        target_input_len (int): Target input length to compile the engine.
        target_output_len (int): Target output length to compile the engine.
        kv_cache_gpu_mem_fraction (float): Fraction of free memory to allocate
            for KV cache.
        enable_attention_dp (bool): Whether attention data parallelism is
            enabled.

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
            enable_attention_dp=enable_attention_dp,
        )
    else:
        max_batch_size = DEFAULT_MAX_BATCH_SIZE
        max_num_tokens = DEFAULT_MAX_NUM_TOKENS
        logger.warning(
            f"Using default settings because quant_algo not supported. "
            f"max_batch_size: {max_batch_size}, max_num_tokens: {max_num_tokens}."
        )

    if max_batch_size <= 0 or max_num_tokens <= 0:
        raise RuntimeError("Unable to obtain correct settings for benchmark.")

    return max_batch_size, max_num_tokens


def get_model_config(model_name: str, model_path: Path = None) -> ModelConfig:
    """Obtain the model-related parameters from Hugging Face.

    Args:
        model_name (str): Huggingface model name.
        model_path (Path): Path to a local Huggingface checkpoint.

    Raises:
        ValueError: When model is not supported.
    """
    pretrained_config = load_pretrained_config(model_path or model_name, trust_remote_code=True)
    if is_nemotron_hybrid(pretrained_config):
        return NemotronHybridConfig.from_hf(model_name, model_path)
    if is_qwen3_hybrid(pretrained_config):
        return Qwen3HybridConfig.from_hf(model_name, model_path)
    return ModelConfig.from_hf(model_name, model_path)
