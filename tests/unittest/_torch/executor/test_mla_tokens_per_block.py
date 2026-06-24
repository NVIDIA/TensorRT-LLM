# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    FLASH_MLA_TOKENS_PER_BLOCK,
    FP4_MLA_TOKENS_PER_BLOCK,
    _select_mla_tokens_per_block,
)
from tensorrt_llm.quantization import QuantAlgo


def _mla_config():
    return SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)


def _non_mla_config():
    return SimpleNamespace()


def _model_config(kv_cache_quant_algo=None, enable_flash_mla=False, attn_backend=None):
    quant_config = SimpleNamespace(kv_cache_quant_algo=kv_cache_quant_algo)
    return SimpleNamespace(
        quant_config=quant_config, enable_flash_mla=enable_flash_mla, attn_backend=attn_backend
    )


def _kv_cache_config(dtype="auto", tokens_per_block=32):
    return SimpleNamespace(dtype=dtype, tokens_per_block=tokens_per_block)


def test_non_mla_keeps_configured_tokens_per_block():
    kv_cache_config = _kv_cache_config(tokens_per_block=32)

    tokens_per_block = _select_mla_tokens_per_block(
        _non_mla_config(),
        _model_config(kv_cache_quant_algo=QuantAlgo.NVFP4, enable_flash_mla=True),
        kv_cache_config,
        kv_cache_config.tokens_per_block,
    )

    assert tokens_per_block == 32
    assert kv_cache_config.tokens_per_block == 32


def test_flash_mla_non_fp4_uses_flash_mla_tokens_per_block():
    kv_cache_config = _kv_cache_config(tokens_per_block=32)

    tokens_per_block = _select_mla_tokens_per_block(
        _mla_config(),
        _model_config(enable_flash_mla=True),
        kv_cache_config,
        kv_cache_config.tokens_per_block,
    )

    assert tokens_per_block == FLASH_MLA_TOKENS_PER_BLOCK
    assert kv_cache_config.tokens_per_block == FLASH_MLA_TOKENS_PER_BLOCK


def test_fp4_mla_dequant_flow_uses_flash_mla_tokens_per_block():
    kv_cache_config = _kv_cache_config(tokens_per_block=32)

    tokens_per_block = _select_mla_tokens_per_block(
        _mla_config(),
        _model_config(kv_cache_quant_algo=QuantAlgo.NVFP4, enable_flash_mla=True),
        kv_cache_config,
        kv_cache_config.tokens_per_block,
    )

    assert tokens_per_block == FLASH_MLA_TOKENS_PER_BLOCK
    assert kv_cache_config.tokens_per_block == FLASH_MLA_TOKENS_PER_BLOCK


def test_fp4_mla_attention_uses_128_tokens_per_block_from_quant_config():
    kv_cache_config = _kv_cache_config(tokens_per_block=32)

    tokens_per_block = _select_mla_tokens_per_block(
        _mla_config(),
        _model_config(kv_cache_quant_algo=QuantAlgo.NVFP4, attn_backend="TRTLLM"),
        kv_cache_config,
        kv_cache_config.tokens_per_block,
    )

    assert tokens_per_block == FP4_MLA_TOKENS_PER_BLOCK
    assert kv_cache_config.tokens_per_block == FP4_MLA_TOKENS_PER_BLOCK


def test_fp4_mla_attention_uses_128_tokens_per_block_from_kv_cache_dtype():
    kv_cache_config = _kv_cache_config(dtype="nvfp4", tokens_per_block=32)

    tokens_per_block = _select_mla_tokens_per_block(
        _mla_config(),
        _model_config(enable_flash_mla=True, attn_backend="TRTLLM"),
        kv_cache_config,
        kv_cache_config.tokens_per_block,
    )

    assert tokens_per_block == FP4_MLA_TOKENS_PER_BLOCK
    assert kv_cache_config.tokens_per_block == FP4_MLA_TOKENS_PER_BLOCK
