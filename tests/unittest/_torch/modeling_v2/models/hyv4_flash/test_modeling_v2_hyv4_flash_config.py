# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash config resolution: the checkpoint's own shape, read on CPU.

The synthetic ``config.json`` comes from ``_hyv4_flash_config``, so the module
runs wherever pytest does and still pins every value the engine reads. The same
expectations are checked against the published checkpoint when it is under
``LLM_MODELS_ROOT``.
"""

import json

import pytest
import torch
from _hyv4_flash_config import SPARSE_LAYERS, config_dict  # noqa: E402

from tensorrt_llm._torch.configs import HYV4FlashConfig, HYV4FlashTextConfig, HYV4FlashVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_hyv4_flash_layer_masks,
    get_hyv4_flash_num_attention_layers,
    is_hyv4_flash,
    load_pretrained_config,
    unwrap_hyv4_flash_text_config,
)
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_CHECKPOINT = "Hy4-flash-FP8"


def _write_checkpoint(directory, config: dict) -> str:
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def _mapping() -> Mapping:
    return Mapping(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1)


def test_config_types_are_registered_with_transformers() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    import tensorrt_llm._torch.configs  # noqa: F401

    assert CONFIG_MAPPING["hy_v4_flash"] is HYV4FlashConfig
    assert CONFIG_MAPPING["hy_v4_flash_text"] is HYV4FlashTextConfig
    assert CONFIG_MAPPING["hy_v4_flash_vision"] is HYV4FlashVisionConfig


def test_composite_config_parses_the_checkpoint(tmp_path) -> None:
    """Every value the engine reads comes back as the checkpoint published it."""
    config = load_pretrained_config(_write_checkpoint(tmp_path, config_dict()))

    assert isinstance(config, HYV4FlashConfig)
    assert isinstance(config.text_config, HYV4FlashTextConfig)
    assert isinstance(config.vision_config, HYV4FlashVisionConfig)
    assert config.model_type == "hy_v4_flash"
    assert config.text_config.model_type == "hy_v4_flash_text"
    assert config.vision_config.model_type == "hy_v4_flash_vision"
    assert config.architectures == ["HYV4FlashForConditionalGeneration"]
    assert config.text_config.architectures == ["HYV4FlashForCausalLM"]
    assert config.torch_dtype is torch.bfloat16
    assert config.text_config.torch_dtype is torch.bfloat16
    assert is_hyv4_flash(config) and is_hyv4_flash(config.text_config)
    assert unwrap_hyv4_flash_text_config(config) is config.text_config

    text = config.text_config
    assert (text.num_hidden_layers, text.hidden_size, text.vocab_size) == (42, 3072, 120832)
    assert (text.num_attention_heads, text.num_key_value_heads) == (32, 32)
    assert (text.q_lora_rank, text.kv_lora_rank) == (1024, 512)
    assert (text.qk_nope_head_dim, text.qk_rope_head_dim, text.v_head_dim) == (192, 64, 256)
    assert (text.qk_head_dim, text.head_dim) == (256, 64)
    assert (text.index_topk, text.index_n_heads, text.index_head_dim) == (2048, 16, 128)
    assert (text.sliding_window, text.swa_num_attention_heads, text.swa_v_head_dim) == (
        512,
        64,
        256,
    )

    # The two attention kinds, from the checkpoint's literal layer_types.
    sparse_mask, sliding_mask = get_hyv4_flash_layer_masks(config)
    assert [i for i, sparse in enumerate(sparse_mask) if sparse] == list(SPARSE_LAYERS)
    assert sum(sliding_mask) == 30 and len(sliding_mask) == 42
    assert get_hyv4_flash_num_attention_layers(config) == 42
    assert get_hyv4_flash_num_attention_layers(config.text_config) == 42
    assert text.index_topk_pattern == ["F" if i in SPARSE_LAYERS else "S" for i in range(42)]
    # The windowed layers run their own head geometry; the sparse ones do not
    # repeat it.
    assert set(text.per_layer_config) == {i for i in range(42) if i not in SPARSE_LAYERS}
    assert text.per_layer_config[2] == {
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "head_dim": 64,
        "v_head_dim": 256,
    }

    # Interleaved 3-D M-RoPE, one base wavelength per layer type.
    assert text.rope_parameters["deepseek_sparse_attention"] == {
        "mrope_interleaved": True,
        "mrope_section": [12, 10, 10],
        "rope_theta": 10_000_000,
        "rope_type": "default",
    }
    assert text.rope_parameters["sliding_attention"] == {
        "mrope_interleaved": True,
        "mrope_section": [12, 10, 10],
        "rope_theta": 10_000,
        "rope_type": "default",
    }

    # The latent MoE, the iHC residual streams and the Engram tables.
    assert (text.n_routed_experts, text.num_experts_per_tok, text.n_shared_experts) == (
        512,
        16,
        1,
    )
    assert text.moe_intermediate_size == text.routed_expert_hidden_size == 1536
    assert (text.routed_scaling_factor, text.swiglu_limit) == (3.998, 10.0)
    assert text.moe_router_use_sigmoid and text.moe_router_enable_expert_bias
    assert text.first_k_dense_replace == 1
    assert (text.hc_mult, text.hc_magnitude, text.hc_eps) == (4, 2.0, 1e-06)
    assert text.enable_ihc and text.ihc_kernel_backend == "naive"
    assert text.engram_layer_ids == [2, 12, 22, 32]
    assert text.engram_layer_indices == [1, 11, 21, 31]
    assert (
        text.engram_max_ngram_size,
        text.engram_n_head_per_ngram,
        text.engram_n_embed_per_ngram,
    ) == (3, 8, 768)
    assert text.engram_pad_id == 120002

    # The checkpoint-native DFly2 drafting geometry and the vision tower.
    assert (config.dfly_num_hidden_layers, config.dfly_mask_token_id) == (5, 120009)
    assert config.dfly_target_kv_layers == config.dfly_target_index_layer == [25, 29, 33, 37, 41]
    assert config.dfly_target_index_topk == 2048
    assert (config.image_token_id, config.video_token_id) == (120120, 120124)
    vision = config.vision_config
    assert (vision.hidden_size, vision.intermediate_size, vision.num_hidden_layers) == (
        1152,
        4304,
        27,
    )
    assert (vision.num_attention_heads, vision.head_dim, vision.out_hidden_size) == (16, 72, 3072)
    assert vision.spatial_merge_size == config.spatial_merge_size == 2
    assert vision.projector_type == "rmsnorm_mlp_adapter"


def test_text_only_config_flattens_to_the_decoder(tmp_path) -> None:
    """``language_model_only`` drops the tower and keeps the decoder's shape."""
    config_dict_without_tower = config_dict(language_model_only=True)
    config = load_pretrained_config(_write_checkpoint(tmp_path, config_dict_without_tower))

    assert isinstance(config, HYV4FlashTextConfig)
    assert not isinstance(config, HYV4FlashConfig)
    assert config.architectures == ["HYV4FlashForCausalLM"]
    assert config.model_type == "hy_v4_flash_text"
    assert config.torch_dtype is torch.bfloat16
    assert is_hyv4_flash(config)
    assert unwrap_hyv4_flash_text_config(config) is config
    assert get_hyv4_flash_num_attention_layers(config) == 42


def test_model_config_reads_the_fp8_block_quant_policy(tmp_path) -> None:
    """The text-side quantization block becomes the engine's FP8 policy."""
    model_config = ModelConfig.from_pretrained(
        _write_checkpoint(tmp_path, config_dict()), mapping=_mapping()
    )

    quant_config = model_config.quant_config
    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.kv_cache_quant_algo is None
    assert quant_config.group_size == 128
    # The checkpoint's own exclusions, plus the generic MLA ones the loader adds.
    assert quant_config.is_module_excluded_from_quantization(
        "language_model.model.model.layers.0.mlp.down_proj"
    )
    assert quant_config.is_module_excluded_from_quantization(
        "language_model.model.model.embed_tokens"
    )
    for name in (
        "model.layers.5.self_attn.kv_b_proj",
        "model.layers.5.self_attn.k_b_proj",
        "model.layers.5.eh_proj",
    ):
        assert quant_config.is_module_excluded_from_quantization(name)
    assert not quant_config.is_module_excluded_from_quantization("model.layers.5.mlp.experts")


def test_model_config_gets_the_dsa_sparse_attention_config(tmp_path) -> None:
    """The sparse layers run DSA with the checkpoint's indexer geometry."""
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    model_config = ModelConfig.from_pretrained(
        _write_checkpoint(tmp_path, config_dict()), mapping=_mapping()
    )
    sparse = model_config.sparse_attention_config

    assert isinstance(sparse, DeepSeekSparseAttentionConfig)
    assert sparse.algorithm == "dsa"
    assert (sparse.index_topk, sparse.index_n_heads, sparse.index_head_dim) == (2048, 16, 128)
    assert sparse.indexer_rope_interleave
    assert sparse.needs_separate_short_long_cuda_graphs()
    assert sparse.seq_len_threshold == 2048

    # Every sparse layer owns an indexer pool; a windowed layer owns none.
    params = sparse.to_sparse_params(pretrained_config=model_config.pretrained_config, layer_idx=5)
    assert params.index_topk == 2048
    assert params.is_full_indexer_layer
    sliding = sparse.to_sparse_params(pretrained_config=model_config.pretrained_config, layer_idx=2)
    assert not sliding.is_full_indexer_layer


def test_model_config_fills_the_dfly2_geometry(tmp_path) -> None:
    """A DFlash deployment with no draft checkpoint gets DFly2 from this config."""
    spec_config = DFlashDecodingConfig(
        decoding_type="DFlash", max_draft_len=5, mask_token_id=120009
    )
    model_config = ModelConfig.from_pretrained(
        _write_checkpoint(tmp_path, config_dict()),
        mapping=_mapping(),
        spec_config=spec_config,
    )

    assert model_config.spec_config is spec_config
    assert spec_config.spec_dec_variant == 2
    assert spec_config.spec_dec_mode == SpeculativeDecodingMode.DFLASH
    assert spec_config.target_layer_ids == [25, 29, 33, 37, 41]
    assert spec_config.mask_token_id == 120009
    assert spec_config.max_total_draft_tokens == 5
    assert spec_config.tokens_per_gen_step == 6


def test_dfly2_geometry_keeps_what_the_deployment_set(tmp_path) -> None:
    spec_config = DFlashDecodingConfig(
        decoding_type="DFlash",
        max_draft_len=5,
        mask_token_id=7,
        target_layer_ids=[25, 29],
    )
    ModelConfig.from_pretrained(
        _write_checkpoint(tmp_path, config_dict()),
        mapping=_mapping(),
        spec_config=spec_config,
    )

    assert spec_config.mask_token_id == 7
    assert spec_config.target_layer_ids == [25, 29]
    assert spec_config.spec_dec_variant == 2


def test_dfly2_geometry_rejects_split_kv_and_index_layers(tmp_path) -> None:
    """A draft layer reuses one target layer's KV and index picks; they must agree."""
    split_layers = config_dict(dfly_target_index_layer=[25, 29, 33, 37, 40])
    with pytest.raises(ValueError, match="dfly_target_index_layer"):
        ModelConfig.from_pretrained(
            _write_checkpoint(tmp_path, split_layers),
            mapping=_mapping(),
            spec_config=DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5),
        )


def test_non_hyv4_config_is_not_hyv4() -> None:
    from transformers import PretrainedConfig

    config = PretrainedConfig(model_type="glm_moe_dsa")
    assert not is_hyv4_flash(config)
    assert unwrap_hyv4_flash_text_config(config) is config


@pytest.fixture(scope="module")
def published_checkpoint():
    from utils.llm_data import llm_models_root

    root = llm_models_root()
    path = None if root is None else root / _CHECKPOINT
    if path is None or not (path / "config.json").is_file():
        pytest.skip(f"{_CHECKPOINT} is not under LLM_MODELS_ROOT")
    return str(path)


def test_published_checkpoint_parses(published_checkpoint) -> None:
    """The same shape, read from the checkpoint itself."""
    config = load_pretrained_config(published_checkpoint)
    assert isinstance(config, HYV4FlashConfig)
    assert config.architectures == ["HYV4FlashForConditionalGeneration"]
    assert config.text_config.num_hidden_layers == 42
    sparse_mask, _ = get_hyv4_flash_layer_masks(config)
    assert [i for i, sparse in enumerate(sparse_mask) if sparse] == list(SPARSE_LAYERS)
    assert config.dfly_target_kv_layers == [25, 29, 33, 37, 41]

    model_config = ModelConfig.from_pretrained(published_checkpoint, mapping=_mapping())
    assert model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert model_config.sparse_attention_config.index_topk == 2048
