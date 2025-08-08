from typing import Tuple

import pytest

from tensorrt_llm.bench.build.dataclasses import ModelConfig
from tensorrt_llm.bench.build.tuning import (calc_engine_setting,
                                             finetune_setting)
from tensorrt_llm.bench.build.utils import get_device_memory
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def create_vanilla_model_config() -> ModelConfig:
    return ModelConfig(
        name='vanilla-model',
        model_type='mock',
        param_count=1_000_000_000,
        num_hidden_layers=12,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=1024,
        max_position_embeddings=10000,
        dtype='float16',
    )


def create_swa_model_config() -> ModelConfig:
    return ModelConfig(
        name='swa-model',
        model_type='mock',
        param_count=1_000_000_000,
        num_hidden_layers=12,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=1024,
        max_position_embeddings=10000,
        sliding_window=256,
        dtype='float16',
    )


def create_vswa_model_config() -> ModelConfig:
    return ModelConfig(
        name='vswa-model',
        model_type='mock',
        param_count=1_000_000_000,
        num_hidden_layers=12,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=1024,
        max_position_embeddings=10000,
        sliding_window=256,
        sliding_window_pattern=4,
        dtype='float16',
    )


def legacy_calc_engine_setting(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    tp_size: int,
    pp_size: int,
    target_input_len: int,
    target_output_len: int,
    kv_cache_gpu_mem_fraction: float = 0.95,
) -> Tuple[int, int]:
    """ This is an old calculation method for testing reference. """

    from tensorrt_llm.bench.build.dataclasses import NemotronHybridConfig
    from tensorrt_llm.bench.build.tuning import BYTES_PER_ELEM
    from tensorrt_llm.logger import logger

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
    gb_per_extra_cache = model_config.extra_model_cache_in_gb(
        BYTES_PER_ELEM.get(QuantAlgo.NO_QUANT), target_seq_len)
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


@pytest.mark.parametrize("seq_len", [5, 200, 2000, 8000, 130000])
def test_tuning_max_batch_size_calculation(seq_len):
    model_configs = {
        'vanilla': create_vanilla_model_config(),
        'swa': create_swa_model_config(),
        'vswa': create_vswa_model_config(),
    }
    quant_config = QuantConfig()

    cache_mem_fraction = 0.1
    input_len = seq_len // 2
    output_len = seq_len - input_len

    legacy_vanilla_mbs, legacy_vanilla_mnt = legacy_calc_engine_setting(
        model_config=model_configs['vanilla'],
        quant_config=quant_config,
        tp_size=1,
        pp_size=1,
        target_input_len=input_len,
        target_output_len=output_len,
        kv_cache_gpu_mem_fraction=cache_mem_fraction,
    )

    vanilla_mbs, vanilla_mnt = calc_engine_setting(
        model_config=model_configs['vanilla'],
        quant_config=quant_config,
        tp_size=1,
        pp_size=1,
        target_input_len=input_len,
        target_output_len=output_len,
        kv_cache_gpu_mem_fraction=cache_mem_fraction,
    )
    print(f'vanilla: mbs {vanilla_mbs} mnt {vanilla_mnt}')

    # Computation for a vanilla model should be identical to the legcay.
    assert legacy_vanilla_mbs == vanilla_mbs
    assert legacy_vanilla_mnt == vanilla_mnt

    swa_mbs, swa_mnt = calc_engine_setting(
        model_config=model_configs['swa'],
        quant_config=quant_config,
        tp_size=1,
        pp_size=1,
        target_input_len=input_len,
        target_output_len=output_len,
        kv_cache_gpu_mem_fraction=cache_mem_fraction,
    )

    if seq_len <= model_configs['swa'].sliding_window:
        assert vanilla_mbs == swa_mbs
        assert vanilla_mnt == swa_mnt
    else:
        # SWA is more memory efficient than vanilla
        assert swa_mbs > vanilla_mbs
        assert swa_mnt >= vanilla_mnt

    vswa_mbs, vswa_mnt = calc_engine_setting(
        model_config=model_configs['vswa'],
        quant_config=quant_config,
        tp_size=1,
        pp_size=1,
        target_input_len=input_len,
        target_output_len=output_len,
        kv_cache_gpu_mem_fraction=cache_mem_fraction,
    )

    if seq_len <= model_configs['vswa'].sliding_window:
        assert vanilla_mbs == vswa_mbs
    else:
        # VSWA is more memory efficient than vanilla
        assert vswa_mbs > vanilla_mbs
        assert vswa_mnt >= vanilla_mnt
        # VSWA is less memory efficient than SWA
        assert vswa_mbs < swa_mbs
        assert vswa_mnt <= swa_mnt


if __name__ == "__main__":
    test_tuning_max_batch_size_calculation(8000)
