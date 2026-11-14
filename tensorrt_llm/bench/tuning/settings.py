from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import torch
from pydantic import BaseModel

from tensorrt_llm._torch._experimental.modeling_v2 import modeling_v2_resolve
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_nemotron_hybrid,
    is_qwen3_hybrid,
    load_pretrained_config,
)
from tensorrt_llm._torch.pyexecutor.model_loader import (
    ModelLoader,
    construct_checkpoint_loader,
    validate_and_set_kv_cache_quant,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bench.tuning.dataclasses import (
    ModelConfig,
    NemotronHybridConfig,
    Qwen3HybridConfig,
)
from tensorrt_llm.bench.tuning.heuristics import calc_engine_setting
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, update_llm_args_with_extra_dict
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig as TorchModelConfig

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


@contextmanager
def _routing_environment(overrides: dict[str, str] | None):
    """Make the LLM's environment overrides visible to the bench router."""
    overrides = overrides or {}
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update({name: str(value) for name, value in overrides.items()})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def resolve_benchmark_kv_cache_quant(
    config: TorchModelConfig,
    model: str | Path,
    kv_cache_dtype: str,
    llm_options: dict,
    *,
    extra_options: dict | None = None,
    explicit_cli_keys: set[str] | None = None,
) -> TorchModelConfig:
    """Resolve target-dependent cache quantization without loading model weights."""
    requests_nvfp4 = kv_cache_dtype == "nvfp4" or (
        kv_cache_dtype == "auto" and config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
    )
    # Only SM107's NVFP4 fallback depends on the target implementation.
    if requests_nvfp4 and torch.cuda.is_available() and get_sm_version() == 107:
        if extra_options:
            llm_options = update_llm_args_with_extra_dict(
                llm_options, extra_options, explicit_cli_keys=explicit_cli_keys
            )
        if llm_options.get("modeling_v2") != "off":
            # Model defaults mutate nested configs; keep the caller's options intact.
            llm_args = TorchLlmArgs(
                **{
                    name: value.model_copy(deep=True) if isinstance(value, BaseModel) else value
                    for name, value in llm_options.items()
                }
            )
            checkpoint_loader = construct_checkpoint_loader(
                llm_args.checkpoint_loader,
                llm_args.checkpoint_format,
                mx_config=llm_args.mx_config,
                checkpoint_io_policy=llm_args.checkpoint_io_policy,
                load_format=llm_args.load_format,
                partial_model_loading=llm_args.is_partial_model_loading,
            )
            checkpoint_dir = str(model)
            try:
                llm_args = ModelLoader.load_config_and_apply_defaults(
                    checkpoint_dir, llm_args, checkpoint_loader
                )
                routing_config = checkpoint_loader.load_config(
                    checkpoint_dir,
                    trust_remote_code=llm_args.trust_remote_code,
                    mapping=llm_args.parallel_config.to_mapping(),
                    spec_config=llm_args.speculative_config,
                    mm_encoder_only=llm_args.mm_encoder_only,
                    attn_backend=llm_args.attn_backend,
                    sparse_attention_config=llm_args.sparse_attention_config,
                    moe_backend=llm_args.moe_config.backend,
                    moe_load_balancer=llm_args.moe_config.load_balancer,
                    model_kwargs=llm_args.model_kwargs,
                )
            finally:
                checkpoint_loader.cleanup()
            # Bench runs on the driver; the engine checks agreement across workers.
            with _routing_environment(llm_args.env_overrides):
                target = modeling_v2_resolve(routing_config, llm_args)
            if target is not None:
                config = replace(routing_config, modeling_v2_target=target)
                kv_cache_dtype = llm_args.kv_cache_config.dtype

    validate_and_set_kv_cache_quant(config, kv_cache_dtype)
    return config


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
