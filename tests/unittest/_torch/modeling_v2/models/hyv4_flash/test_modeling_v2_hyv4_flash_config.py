# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash config parsing and ``ModelConfig`` resolution, on CPU."""

import json

import pytest
from _hyv4_flash_config import SPARSE_LAYERS, config_dict  # noqa: E402

from tensorrt_llm._torch.configs import HYV4FlashConfig, HYV4FlashTextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_hyv4_flash_layer_masks,
    load_pretrained_config,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


def _write_checkpoint(directory) -> str:
    (directory / "config.json").write_text(json.dumps(config_dict()))
    return str(directory)


def _model_config(directory, spec_config=None) -> ModelConfig:
    return ModelConfig.from_pretrained(
        _write_checkpoint(directory),
        mapping=Mapping(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
        spec_config=spec_config,
    )


def test_composite_config_parses_the_checkpoint(tmp_path) -> None:
    config = load_pretrained_config(_write_checkpoint(tmp_path))

    assert isinstance(config, HYV4FlashConfig)
    assert isinstance(config.text_config, HYV4FlashTextConfig)
    assert config.architectures == ["HYV4FlashForConditionalGeneration"]
    text = config.text_config
    sparse_mask, _ = get_hyv4_flash_layer_masks(config)
    assert [i for i, sparse in enumerate(sparse_mask) if sparse] == list(SPARSE_LAYERS)
    assert text.index_kpool == 4
    # One RoPE base per layer type, under the checkpoint's own layer-type names.
    assert text.rope_parameters["deepseek_sparse_attention"]["rope_theta"] == 10_000_000
    assert text.rope_parameters["sliding_attention"]["rope_theta"] == 10_000


def test_model_config_resolves_quant_and_dsa(tmp_path) -> None:
    model_config = _model_config(tmp_path)

    quant_config = model_config.quant_config
    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.group_size == 128
    assert quant_config.is_module_excluded_from_quantization(
        "language_model.model.model.layers.0.mlp.down_proj"
    )
    assert not quant_config.is_module_excluded_from_quantization("model.layers.5.mlp.experts")

    sparse = model_config.sparse_attention_config
    assert isinstance(sparse, DeepSeekSparseAttentionConfig)
    assert (sparse.index_topk, sparse.index_n_heads, sparse.index_head_dim) == (2048, 16, 128)
    assert sparse.indexer_rope_interleave
    pretrained = model_config.pretrained_config
    assert sparse.to_sparse_params(pretrained_config=pretrained, layer_idx=5).is_full_indexer_layer
    assert not sparse.to_sparse_params(
        pretrained_config=pretrained, layer_idx=2
    ).is_full_indexer_layer


def test_model_config_fills_the_dfly2_geometry(tmp_path) -> None:
    spec_config = DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5)
    _model_config(tmp_path, spec_config)

    assert spec_config.spec_dec_variant == 2
    assert spec_config.target_layer_ids == [25, 29, 33, 37, 41]
    assert spec_config.mask_token_id == 120009
