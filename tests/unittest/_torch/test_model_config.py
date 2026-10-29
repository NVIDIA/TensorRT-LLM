# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.model_config import _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT, ModelConfig
from tensorrt_llm._torch.pyexecutor.model_loader import (
    validate_and_set_kv_cache_quant,
    validate_encoder_decoder_kv_cache_config,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
    is_encoder_decoder: bool = False,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
        is_encoder_decoder=is_encoder_decoder,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size

    def ceil_div(a, b):
        return (a + b - 1) // b

    assert bindings_cfg.num_heads == ceil_div(cfg.num_attention_heads, attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == ceil_div(cfg.hidden_size, attn_tp_size)
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            ceil_div(kv, attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == ceil_div(
            cfg.num_key_value_heads, attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == ceil_div(cfg.intermediate_size, mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block


def _make_model_config_with_kv_quant(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo))


def _make_kv_cache_config(
    *, use_kv_cache_manager_v2: bool, cross_kv_cache_fraction: float | None = None
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        cross_kv_cache_fraction=cross_kv_cache_fraction,
        enable_block_reuse=True,
    )


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")


def _mock_sm107(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.model_loader.get_sm_version", lambda: 107)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.model_loader.torch.cuda.is_available",
        lambda: True,
    )


def test_validate_and_set_kv_cache_quant_downgrades_nvfp4_on_sm107(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sm107(monkeypatch)
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def _make_mixed_precision_model_config():
    """MIXED_PRECISION checkpoint shape: global config plus per-layer entries
    whose kv_cache_quant_algo comes only from hf_quant_config.json (None here)."""
    return ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=None),
        quant_config_dict={
            "model.layers.0.attention": QuantConfig(quant_algo=QuantAlgo.FP8),
            "model.layers.0.mixer.experts": QuantConfig(quant_algo=QuantAlgo.NVFP4),
        },
    )


def test_validate_and_set_kv_cache_quant_propagates_to_quant_config_dict():
    """Explicit kv_cache_config.dtype must override the per-layer QuantConfigs
    too, otherwise the KV pool (sized from the global config) and attention
    modules (built from per-layer configs) disagree on KV element size."""
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "fp8")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_auto_keeps_quant_config_dict():
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "auto")
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo is None


def test_validate_and_set_kv_cache_quant_sm107_downgrade_propagates_to_quant_config_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SM107 NVFP4->FP8 downgrade must update per-layer QuantConfigs too,
    otherwise a mixed-precision checkpoint keeps NVFP4 in quant_config_dict
    while the KV pool (sized from the global config) uses FP8."""
    _mock_sm107(monkeypatch)
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def _write_safetensors_header(checkpoint_dir, tensor_dtype, tensor_shape):
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: {
            "dtype": tensor_dtype,
            "shape": tensor_shape,
            "data_offsets": [0, 0],
        }
    }
    encoded_header = json.dumps(header).encode("utf-8")

    with open(checkpoint_dir / shard_name, "wb") as f:
        f.write(struct.pack("<Q", len(encoded_header)))
        f.write(encoded_header)

    with open(checkpoint_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: shard_name}}, f)


@pytest.mark.parametrize(
    "tensor_dtype,tensor_shape,expected_layout,expected_is_base",
    [
        pytest.param("I8", [2048, 2048], "mxfp4", False, id="mxfp4"),
        pytest.param("U8", [2048, 2048], "nvfp4", False, id="nvfp4"),
        pytest.param("F8_E4M3", [2048, 4096], None, True, id="base-fp8"),
    ],
)
def test_deepseek_v4_base_checkpoint_detection(
    tmp_path, tensor_dtype, tensor_shape, expected_layout, expected_is_base
):
    _write_safetensors_header(tmp_path, tensor_dtype, tensor_shape)

    assert ModelConfig._detect_deepseek_v4_routed_moe_layout(str(tmp_path)) == expected_layout
    assert ModelConfig._is_deepseek_v4_base_checkpoint(str(tmp_path)) is expected_is_base


@pytest.mark.parametrize("sidecar", [True, False], ids=["modelopt-file", "modelopt-inline"])
def test_deepseek_v41_nvfp4_checkpoint_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar: bool
) -> None:
    """Resolve dense FP8 separately from the checkpoint's NVFP4 expert policy."""
    from tensorrt_llm._torch import model_config as model_config_module
    from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41Config

    quantization = {
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "kv_cache_quant_algo": "FP8",
        "exclude_modules": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
        "quantized_layers": {
            "layers.0.ffn.experts": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.0.self_attn.o_proj": {
                "quant_algo": "FP8_BLOCK_SCALES",
                "group_size": 128,
            },
        },
    }
    inline = {
        **deepcopy(quantization),
        "producer": {"name": "modelopt"},
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "modules_to_not_convert": ["*engram*"],
    }
    pretrained = DeepseekV41Config(
        architectures=["DeepseekV41ForCausalLM"], quantization_config=deepcopy(inline)
    )
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained
    )
    _write_safetensors_header(tmp_path, "U8", [2304, 2560])
    if sidecar:
        # The file's explicit null must win over the inline KV policy.
        quantization["kv_cache_quant_algo"] = None
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps({"producer": inline["producer"], "quantization": quantization})
        )

    model_config = ModelConfig.from_pretrained(str(tmp_path), moe_backend="CUTLASS")
    dense = model_config.quant_config
    assert dense.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert dense.group_size == 128
    assert dense.kv_cache_quant_algo == (None if sidecar else QuantAlgo.FP8)
    assert dense.is_module_excluded_from_quantization("model.layers.1.engram.kv_proj")
    assert not dense.is_module_excluded_from_quantization("model.layers.0.mlp.shared_experts")
    for layer in range(pretrained.num_hidden_layers):
        expert = model_config.quant_config_dict[f"model.layers.{layer}.mlp.experts"]
        assert expert.quant_algo == QuantAlgo.NVFP4
        assert expert.group_size == 16
    explicit = model_config.quant_config_dict["model.layers.0.self_attn.o_proj"]
    assert explicit.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert explicit.group_size == 128
    assert explicit.kv_cache_quant_algo == dense.kv_cache_quant_algo
    assert pretrained.quantization_config == inline


def test_deepseek_v4_missing_compress_ratios_raises(tmp_path, monkeypatch):
    """DeepSeek-V4 load must fail fast with a clear error when neither the
    checkpoint config nor a user ``sparse_attention_config`` provides
    ``compress_ratios``.

    Regression test: previously ``compress_ratios`` could stay ``None`` and the
    internal normalization comprehension raised an opaque
    ``TypeError: 'NoneType' object is not iterable`` mid-load.
    """
    from tensorrt_llm._torch import model_config as model_config_module
    from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config

    pretrained_config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        compress_ratios=None,
        num_hidden_layers=4,
    )

    # Avoid touching the filesystem for the HF config load; the empty tmp_path
    # makes the real ``_is_deepseek_v4_base_checkpoint`` probe return False.
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained_config
    )

    with pytest.raises(ValueError, match="compress_ratios"):
        ModelConfig.from_pretrained(str(tmp_path))


def test_model_config_sets_is_encoder_decoder_from_pretrained_config():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    assert model_config.is_encoder_decoder is True


def test_validate_encoder_decoder_kv_cache_config_accepts_v1_enc_dec():
    """Both V1 and V2 are supported as long as ``cross_kv_cache_fraction`` is set."""
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=False, cross_kv_cache_fraction=0.5),
    )


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_validate_encoder_decoder_kv_cache_config_requires_cross_fraction(
    use_kv_cache_manager_v2: bool,
) -> None:
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    with pytest.raises(ValueError, match="cross_kv_cache_fraction to be set"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(use_kv_cache_manager_v2=use_kv_cache_manager_v2),
        )


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_validate_encoder_decoder_kv_cache_config_rejects_cross_fraction_for_decoder_only(
    use_kv_cache_manager_v2: bool,
) -> None:
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=False,
        )
    )

    with pytest.raises(ValueError, match="should only be set for encoder-decoder models"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(
                use_kv_cache_manager_v2=use_kv_cache_manager_v2, cross_kv_cache_fraction=0.5
            ),
        )


def test_validate_encoder_decoder_kv_cache_config_accepts_v2_enc_dec():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=True, cross_kv_cache_fraction=0.5),
    )


_MISTRAL_LARGE4_NUM_LAYERS = 72


def _mistral_large4_expected_config(**overrides) -> dict:
    """The language-model config adapted from the Mistral-Large-4-1T-A52B NVFP4 params.json."""
    num_layers = _MISTRAL_LARGE4_NUM_LAYERS
    nope_layer_flags = [layer % 4 == 3 for layer in range(num_layers)]
    config = {
        "_name_or_path": "",
        "architectures": ["MistralLarge4ForCausalLM"],
        "chunk_size_feed_forward": 0,
        "dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "fp8_kv_cache": True,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "index_head_dim": 128,
        "index_n_heads": 32,
        # Layers 0-16 run their own indexer; from layer 17 on, two layers reuse
        # the previous full layer's top-k, then two run their own.
        "index_sharing": [
            "full" if layer < 17 or (layer - 17) % 4 >= 2 else "shared"
            for layer in range(num_layers)
        ],
        "index_topk": 2048,
        "indexer_nope_layer_flags": list(nope_layer_flags),
        "indexer_norm_eps": 1e-06,
        "indexer_ssmax_scaling": True,
        "intermediate_size": 24576,
        "is_encoder_decoder": False,
        "kv_lora_rank": 512,
        # Every NoPE layer after layer 35 reads layer 35's latent KV.
        "kv_source_layers": [
            35 if layer > 35 and layer % 4 == 3 else layer for layer in range(num_layers)
        ],
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "lerope": True,
        "max_position_embeddings": 1048576,
        "max_seq_len": 1048576,
        "model_type": "mistral_large_4",
        "moe": {"expert_model_parallel": 1, "expert_parallel": 1},
        "moe_intermediate_size": 6144,
        "moe_layer_freq": 1,
        "n_group": 1,
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "nope_layer_flags": list(nope_layer_flags),
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 4,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": 64,
        "output_attentions": False,
        "output_hidden_states": False,
        "problem_type": None,
        "q_lora_rank": None,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "quantization_config": _mistral_large4_modelopt_quantization_config(),
        "return_dict": True,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {
            "attention_factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 2048.0,
            "mscale_all_dim": 1,
            "original_max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "rope_type": "yarn",
        },
        "rope_theta": 10000.0,
        "routed_scaling_factor": 1.0,
        "scoring_func": "softmax",
        "softmax_tanh_gating": 30.0,
        "ssmax_scaling": True,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": None,
        "v_head_dim": 128,
        "vocab_size": 204800,
        "yarn": {},
    }
    config.update(overrides)
    return config


def _mistral_large4_model_config(checkpoint_dir: Path, fp8_kv_cache: bool = True) -> ModelConfig:
    params = _mistral_large4_native_params()
    params["dsa"]["fp8_kv_cache"] = fp8_kv_cache
    (checkpoint_dir / "params.json").write_text(json.dumps(params))
    # TEP4 with the serving limits passed the way the model loader passes them.
    # moe_backend is explicit so that no GPU is needed to resolve AUTO.
    return ModelConfig.from_pretrained(
        str(checkpoint_dir),
        checkpoint_format="mistral_large_4",
        mapping=Mapping(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1),
        max_seq_len=9216,
        max_num_tokens=9216,
        moe_backend="CUTLASS",
    )


def test_mistral_large4_native_pretrained_config(tmp_path):
    from tensorrt_llm._torch.configs import MistralLarge4Config
    from tensorrt_llm._torch.pyexecutor.config_utils import is_mla

    expected = _mistral_large4_expected_config()
    model_config = _mistral_large4_model_config(tmp_path)
    pretrained_config = model_config.pretrained_config

    assert isinstance(pretrained_config, MistralLarge4Config)
    # Every key holds its expected value. transformers turns the dtype string
    # into a torch.dtype and the id2label keys into ints, and the loader
    # records the checkpoint directory as _name_or_path.
    normalized = {"_name_or_path", "dtype", "id2label"}
    for key, value in expected.items():
        if key not in normalized:
            assert getattr(pretrained_config, key) == value, key
    assert pretrained_config.dtype == torch.bfloat16
    assert pretrained_config.id2label == {0: "LABEL_0", 1: "LABEL_1"}
    assert pretrained_config._name_or_path == str(tmp_path)
    # Nothing else appears beside them (vision_config is covered by
    # test_mistral_large4_native_checkpoint): no DeepSeek-V3 default (an MTP
    # layer count, special-token ids), and the serving limits stay on
    # ModelConfig.
    public = {key for key in vars(pretrained_config) if not key.startswith("_")}
    assert public - expected.keys() <= {"transformers_version", "vision_config"}
    assert not hasattr(pretrained_config, "num_nextn_predict_layers")
    assert pretrained_config.head_dim == 128
    assert pretrained_config.max_seq_len == 1048576
    assert model_config.max_seq_len == 9216
    assert is_mla(pretrained_config)


def test_mistral_large4_derives_dsa_with_shared_indexers(tmp_path):
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
        derive_indexer_k_cache_layer_mask,
    )
    from tensorrt_llm._torch.attention.backends.sparse.mistral_large4.cache_manager import (
        MistralLarge4CacheManager,
    )
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig

    model_config = _mistral_large4_model_config(tmp_path)
    pretrained_config = model_config.pretrained_config
    sparse_config = model_config.sparse_attention_config

    assert type(sparse_config) is DeepSeekSparseAttentionConfig
    assert sparse_config.index_n_heads == 32
    assert sparse_config.index_head_dim == 128
    assert sparse_config.index_topk == 2048
    assert sparse_config.indexer_k_dtype == "fp8"
    assert sparse_config.indexer_rope_interleave is True

    # Exactly the 'full' layers own an indexer-K cache; the 'shared' ones reuse
    # the previous full layer's top-k.
    index_sharing = pretrained_config.index_sharing
    indexer_layers = derive_indexer_k_cache_layer_mask(
        sparse_config, pretrained_config, _MISTRAL_LARGE4_NUM_LAYERS
    )
    assert indexer_layers == [role == "full" for role in index_sharing]
    assert sum(indexer_layers) == 44

    assert get_kv_cache_manager_cls(model_config, KvCacheConfig()) is MistralLarge4CacheManager
    owners = sum(i == source for i, source in enumerate(pretrained_config.kv_source_layers))
    bytes_per_token = MistralLarge4CacheManager.get_cache_size_per_token(
        model_config, model_config.mapping
    )
    assert bytes_per_token == owners * 656 + 44 * 132


def test_mistral_large4_quantization(tmp_path):
    quant_config = _mistral_large4_model_config(tmp_path).quant_config

    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert quant_config.group_size == 16
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8


@pytest.mark.parametrize(
    ("fp8_kv_cache", "kv_cache_dtype", "expected"),
    [
        pytest.param(True, "auto", QuantAlgo.FP8, id="checkpoint_fp8_auto"),
        pytest.param(True, "nvfp4", QuantAlgo.NVFP4, id="checkpoint_fp8_explicit_nvfp4"),
        pytest.param(False, "auto", None, id="checkpoint_bf16_auto"),
        pytest.param(False, "fp8", QuantAlgo.FP8, id="checkpoint_bf16_explicit_fp8"),
    ],
)
def test_mistral_large4_fp8_kv_cache_resolves_kv_cache_quant(
    tmp_path, fp8_kv_cache, kv_cache_dtype, expected
):
    # `fp8_kv_cache` is the checkpoint's KV policy, as an hf_quant_config
    # kv_cache_quant_algo would be: 'auto' keeps it, an explicit dtype wins.
    model_config = _mistral_large4_model_config(tmp_path, fp8_kv_cache=fp8_kv_cache)
    validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)
    assert model_config.quant_config.kv_cache_quant_algo == expected


@pytest.mark.parametrize(
    ("kv_cache_dtype", "expected"),
    [
        pytest.param("auto", QuantAlgo.FP8, id="auto"),
        pytest.param("nvfp4", QuantAlgo.NVFP4, id="explicit_nvfp4"),
    ],
)
def test_mistral_large4_fp8_kv_cache_reaches_llm_args(tmp_path, kv_cache_dtype, expected):
    # llm.args.quant_config reports the same KV policy the engine runs with.
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    (tmp_path / "params.json").write_text(json.dumps(_mistral_large4_native_params()))
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        checkpoint_format="mistral_large_4",
        kv_cache_config=KvCacheConfig(dtype=kv_cache_dtype),
    )

    assert ModelLoader(llm_args)._update_from_hf_quant_config() is True
    assert llm_args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert llm_args.quant_config.kv_cache_quant_algo == expected


def _mistral_large4_modelopt_quantization_config() -> dict:
    """The inline ModelOpt config of the Mistral-Large-4-1T-A52B NVFP4 export, keyed by Mistral
    module names: NVFP4 routed experts, FP8_PB_WO (block-128 FP8) linears, nothing else."""
    fp8 = {"quant_algo": "FP8_PB_WO"}
    layers = {
        f"layers.{layer}.experts": {"quant_algo": "NVFP4", "group_size": 16}
        for layer in range(1, _MISTRAL_LARGE4_NUM_LAYERS)
    }
    for layer in range(_MISTRAL_LARGE4_NUM_LAYERS):
        layers[f"layers.{layer}.attention.wq"] = fp8
        layers[f"layers.{layer}.attention.wo"] = fp8
        if layer < 17 or layer % 4 in (0, 3):
            layers[f"layers.{layer}.attention.indexer.wq_b"] = fp8
        mlp = "feed_forward" if layer == 0 else "shared_experts"
        for w in ("w1", "w2", "w3"):
            layers[f"layers.{layer}.{mlp}.{w}"] = fp8
    return {
        "group_size": 16,
        "ignore": [],
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "quantized_layers": layers,
    }


def test_mistral_large4_modelopt_mixed_precision_keeps_quantized_layers(tmp_path):
    # The engine and the LLM API each normalise their own copy of the ModelOpt config; the
    # config the modeling_v2 router reads keeps the map exactly as params.json states it.
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    quantization_config = _mistral_large4_modelopt_quantization_config()
    (tmp_path / "params.json").write_text(json.dumps(_mistral_large4_native_params()))
    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        checkpoint_format="mistral_large_4",
        mapping=Mapping(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1),
        max_seq_len=32768,
        max_num_tokens=32768,
        moe_backend="CUTLASS",
    )
    validate_and_set_kv_cache_quant(model_config, "fp8")

    quant_config = model_config.quant_config
    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert quant_config.group_size == 16
    layers = model_config.quant_config_dict
    assert set(layers) == set(quantization_config["quantized_layers"])
    algos = [(q.quant_algo, q.group_size) for q in layers.values()]
    assert algos.count((QuantAlgo.NVFP4, 16)) == 71
    assert algos.count((QuantAlgo.FP8_BLOCK_SCALES, None)) == 404
    assert model_config.pretrained_config.quantization_config == quantization_config

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        checkpoint_format="mistral_large_4",
        kv_cache_config=KvCacheConfig(dtype="fp8"),
    )
    assert ModelLoader(llm_args)._update_from_hf_quant_config() is True
    assert llm_args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert llm_args.quant_config.group_size == 16


def _mistral_large4_native_params() -> dict:
    layers = _MISTRAL_LARGE4_NUM_LAYERS
    params = {
        "dim": 6144,
        "n_layers": layers,
        "n_heads": 64,
        "n_kv_heads": 64,
        "head_dim": 128,
        "hidden_dim": 24576,
        "norm_eps": 1e-5,
        "vocab_size": 204800,
        "tied_embeddings": False,
        "max_position_embeddings": 1048576,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "nope": "rope,rope,rope,nope",
        "ssmax_scaling": True,
        "lerope": True,
        "softmax_tanh_gating": 30.0,
        "rope_theta": 10000.0,
        "yarn": {
            "original_max_position_embeddings": 512,
            "factor": 2048,
            "apply_scale": False,
            "beta": 32,
            "alpha": 1,
        },
        "moe": {
            "expert_parallel": 1,
            "expert_model_parallel": 1,
            "route_every_n": 1,
            "first_k_dense_replace": 1,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
            "routed_scale": 1.0,
            "expert_hidden_dim": 6144,
            "num_shared_experts": 1,
        },
        "dsa": {
            "topk": 2048,
            "n_heads": 32,
            "head_dim": 128,
            "match_nope": True,
            "ssmax_scaling": True,
            "norm_eps": 1e-6,
            "sharing": ",".join(
                "full" if i < 17 or i % 4 in (0, 3) else "shared" for i in range(layers)
            ),
            "fp8_kv_cache": True,
        },
        "kv_sharing": {
            "source_layers_str": ",".join(
                str(35 if i > 35 and i % 4 == 3 else i) for i in range(layers)
            )
        },
        "vision_encoder": {"hidden_size": 2048},
        "quantization_config": _mistral_large4_modelopt_quantization_config(),
    }
    return params


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_mistral_large4_native_checkpoint(tmp_path, kv_cache_dtype):
    from tensorrt_llm._torch.configs import MistralLarge4Config
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (
        MistralConfigLoader,
        adapt_config_dict,
    )
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    params = _mistral_large4_native_params()
    (tmp_path / "params.json").write_text(json.dumps(params))
    original = deepcopy(params)
    config = adapt_config_dict(params)
    assert params == original
    expected = _mistral_large4_expected_config()
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "moe_layer_freq",
        "scoring_func",
        "nope_layer_flags",
        "kv_source_layers",
        "index_sharing",
        "indexer_nope_layer_flags",
        "indexer_norm_eps",
        "indexer_ssmax_scaling",
        "fp8_kv_cache",
        "quantization_config",
    ):
        assert getattr(config, field) == expected[field], field
    assert config.rope_parameters == expected["rope_parameters"]
    assert config.index_topk_pattern.count("F") == 44
    assert not hasattr(config, "vision_encoder")
    assert config.vision_config.hidden_size == 2048
    assert type(config).from_dict(config.to_dict()).vision_config.hidden_size == 2048
    assert not hasattr(config, "num_nextn_predict_layers")
    # Native parsing is selected explicitly in both the client and engine paths.
    assert isinstance(
        load_pretrained_config(str(tmp_path), checkpoint_format="mistral_large_4"),
        MistralLarge4Config,
    )
    assert isinstance(
        ModelLoader.load_hf_model_config(str(tmp_path), checkpoint_format="mistral_large_4"),
        MistralLarge4Config,
    )
    from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader

    checkpoint_loader = _construct_checkpoint_loader(None, "mistral_large_4")
    assert checkpoint_loader.checkpoint_format == "mistral_large_4"
    assert isinstance(checkpoint_loader.config_loader, MistralConfigLoader)
    model_config = checkpoint_loader.load_config(
        str(tmp_path),
        mapping=Mapping(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1),
        max_seq_len=32768,
        max_num_tokens=32768,
        moe_backend="CUTLASS",
    )
    assert model_config.pretrained_config.max_seq_len == 1048576
    assert model_config.max_seq_len == 32768
    assert model_config.sparse_attention_config.index_n_heads == 32
    assert model_config.sparse_attention_config.index_topk == 2048
    assert model_config.sparse_attention_config.indexer_rope_interleave is True
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert model_config.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert len(model_config.quant_config_dict) == len(
        expected["quantization_config"]["quantized_layers"]
    )
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        checkpoint_format="mistral_large_4",
        kv_cache_config=KvCacheConfig(dtype=kv_cache_dtype),
    )
    assert ModelLoader(llm_args)._update_from_hf_quant_config() is True
    assert llm_args.quant_config.quant_algo == model_config.quant_config.quant_algo
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert json.loads((tmp_path / "params.json").read_text()) == original
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("stale_config_json", [False, True], ids=["params_only", "stale_config"])
def test_mistral_large4_requires_native_format(tmp_path, stale_config_json):
    # A Mistral-Large-4 params.json loads only with checkpoint_format='mistral_large_4';
    # any other format raises an error naming it, even beside a config.json that states
    # model_type mistral_large_4.
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    (tmp_path / "params.json").write_text(json.dumps(_mistral_large4_native_params()))
    if stale_config_json:
        (tmp_path / "config.json").write_text(
            json.dumps(_mistral_large4_expected_config(hidden_size=128))
        )
    native_format = "checkpoint_format='mistral_large_4'"
    for checkpoint_format in (None, "HF"):
        with pytest.raises(ValueError, match=native_format):
            load_pretrained_config(str(tmp_path), checkpoint_format=checkpoint_format)
        with pytest.raises(ValueError, match=native_format):
            ModelConfig.from_pretrained(str(tmp_path), checkpoint_format=checkpoint_format)
    assert (
        load_pretrained_config(str(tmp_path), checkpoint_format="mistral_large_4").hidden_size
        == 6144
    )


@pytest.mark.parametrize("params", ["mistral", "mla_without_dsa", "invalid_json"])
def test_hf_config_loads_beside_an_unrelated_params_json(tmp_path, params):
    # Only a Mistral-Large-4 params.json requires the native format; the default (HF)
    # format still loads config.json beside any other params.json.
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    if params == "mistral":
        native_params = {
            "dim": 4096,
            "n_layers": 32,
            "head_dim": 128,
            "hidden_dim": 14336,
            "n_heads": 32,
            "n_kv_heads": 8,
            "norm_eps": 1e-5,
            "vocab_size": 32768,
            "rope_theta": 1000000.0,
        }
    elif params == "mla_without_dsa":
        native_params = _mistral_large4_native_params()
        del native_params["dsa"]
    else:
        native_params = None
    (tmp_path / "params.json").write_text(
        "invalid JSON" if native_params is None else json.dumps(native_params)
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_hidden_layers": 2,
                "vocab_size": 1000,
            }
        )
    )
    for checkpoint_format in (None, "HF"):
        config = load_pretrained_config(str(tmp_path), checkpoint_format=checkpoint_format)
        assert config.model_type == "mistral"
        assert config.hidden_size == 128


@pytest.mark.parametrize("field", ["nope", "kv_sharing", "dsa", "yarn"])
def test_mistral_large4_invalid_native_patterns(field):
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import adapt_config_dict

    params = _mistral_large4_native_params()
    if field == "nope":
        params["nope"] = "rope,unknown"
    elif field == "kv_sharing":
        params["kv_sharing"]["source_layers_str"] = "0,1"
    elif field == "dsa":
        params["dsa"]["sharing"] = "full,shared"
    else:
        params["yarn"]["apply_scale"] = True
    with pytest.raises(ValueError):
        adapt_config_dict(params)


def test_deepseek_v32_dsa_derivation_ignores_mistral_large4_keys(tmp_path):
    # The DeepSeek-V3.2 derivation is unchanged: every layer runs its own
    # indexer, the indexer rope is NeoX, and only a quant config states a KV
    # policy.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "num_hidden_layers": 4,
                "index_n_heads": 64,
                "index_head_dim": 128,
                "index_topk": 2048,
                "fp8_kv_cache": True,
                "index_sharing": ["full", "shared", "shared", "shared"],
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    model_config = ModelConfig.from_pretrained(str(tmp_path), moe_backend="CUTLASS")
    sparse_config = model_config.sparse_attention_config

    assert sparse_config.algorithm == "dsa"
    assert sparse_config.indexer_rope_interleave is False
    assert all(
        sparse_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config, layer_idx=layer
        ).is_full_indexer_layer
        for layer in range(4)
    )
    assert model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert model_config.quant_config.kv_cache_quant_algo is None
