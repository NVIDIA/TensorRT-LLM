import json
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_exclude_modules
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.config_utils import _Qwen35ConfigCompat
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.quantization.mode import QuantMode


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
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


def test_quant_mode_distinguishes_w4a16_nvfp4_from_w4a4_nvfp4():
    w4a4_mode = QuantMode.from_quant_algo(QuantAlgo.NVFP4)
    assert w4a4_mode.has_nvfp4()
    assert not w4a4_mode.has_w4a16_nvfp4()

    w4a16_mode = QuantMode.from_quant_algo(QuantAlgo.W4A16_NVFP4)
    assert w4a16_mode.has_w4a16_nvfp4()
    assert not w4a16_mode.has_nvfp4()
    assert w4a16_mode.has_any_quant(exclude_kv_cache=True)


def test_load_hf_quant_config_detects_nvfp4_w4a16_compressed_tensors():
    hf_quant_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["mtp.layers"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear", "lm_head"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 16,
                },
                "input_activations": None,
            },
        },
    }

    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        hf_quant_config, moe_backend="CUTLASS"
    )

    assert quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert quant_config.group_size == 16
    assert quant_config.exclude_modules == ["mtp.layers"]
    assert layer_quant_config is None


def test_load_modelopt_quant_config_respects_hf_w4a16_metadata(tmp_path):
    modelopt_quant_config = {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": ["mtp*"],
        }
    }
    hf_quant_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["mtp.layers.0.mixer.q_proj"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear", "lm_head"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": None,
            },
        },
    }
    quant_config_file = tmp_path / "hf_quant_config.json"
    quant_config_file.write_text(json.dumps(modelopt_quant_config), encoding="utf-8")

    quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
        str(quant_config_file),
        str(tmp_path),
        moe_backend="CUTLASS",
        hf_quant_config=hf_quant_config,
    )

    assert quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert quant_config.group_size == 16
    assert quant_config.exclude_modules == ["mtp.layers.0.mixer.q_proj"]
    assert layer_quant_config is None


def test_load_modelopt_mixed_precision_w4a16_uses_layer_quant_config(tmp_path):
    modelopt_quant_config = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["mtp*"],
            "quantized_layers": {
                "model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_algo": "FP8",
                },
                "model.layers.0.mlp.experts": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        }
    }
    quant_config_file = tmp_path / "hf_quant_config.json"
    quant_config_file.write_text(json.dumps(modelopt_quant_config), encoding="utf-8")

    quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
        str(quant_config_file),
        str(tmp_path),
        moe_backend="CUTLASS",
    )

    assert quant_config.quant_algo is None
    assert quant_config.group_size == 16
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert quant_config.exclude_modules == ["mtp*"]
    assert layer_quant_config is not None
    assert layer_quant_config["model.layers.0.linear_attn.in_proj_qkv"].quant_algo == QuantAlgo.FP8
    assert layer_quant_config["model.layers.0.mlp.experts"].quant_algo == QuantAlgo.W4A16_NVFP4


def test_qwen35_compat_adds_qkvz_excludes_to_modelopt_ignore():
    config_dict = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 4,
            "num_experts": 256,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
            ],
        },
        "quantization_config": {
            "producer": {"name": "modelopt"},
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "ignore": [
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "mtp.layers.0*",
            ],
            "quantized_layers": {},
        },
    }

    normalized = _Qwen35ConfigCompat.normalize(config_dict)

    ignore = normalized["quantization_config"]["ignore"]
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in ignore
    assert "mtp.layers.0*" in ignore
    assert "model.layers.0.linear_attn.in_proj_qkvz" in ignore
    assert "model.layers.2.linear_attn.in_proj_qkvz" in ignore
    assert "model.layers.3.linear_attn.in_proj_qkvz" in ignore
    assert "model.layers.1.linear_attn.in_proj_qkvz" not in ignore


def test_qwen35_file_modelopt_config_adds_qkvz_excludes(tmp_path):
    config_dict = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 1024,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 4096,
            "torch_dtype": "bfloat16",
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
            ],
        },
    }
    hf_quant_config = {
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["mtp.layers.0*", "mtp*"],
            "quantized_layers": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_algo": "FP8",
                },
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant_config), encoding="utf-8")

    model_config = ModelConfig.from_pretrained(
        str(tmp_path),
        moe_backend="CUTLASS",
    )

    exclude_modules = model_config.quant_config.exclude_modules
    assert "mtp.layers.0*" in exclude_modules
    assert "mtp*" in exclude_modules
    assert "model.layers.0.linear_attn.in_proj_qkvz" in exclude_modules
    assert "model.layers.2.linear_attn.in_proj_qkvz" in exclude_modules
    assert "model.layers.3.linear_attn.in_proj_qkvz" in exclude_modules
    assert "model.layers.1.linear_attn.in_proj_qkvz" not in exclude_modules


def test_qwen35_normalizes_layer_quant_config_keys():
    model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(num_hidden_layers=40),
        quant_config=QuantConfig(exclude_modules=["mtp.layers.0*"]),
        quant_config_dict={
            "model.language_model.layers.0.linear_attn.in_proj_qkv": QuantConfig(
                quant_algo=QuantAlgo.FP8
            ),
            "model.language_model.layers.0.linear_attn.in_proj_z": QuantConfig(
                quant_algo=QuantAlgo.FP8
            ),
            "model.language_model.layers.0.linear_attn.in_proj_b": QuantConfig(
                quant_algo=QuantAlgo.FP8
            ),
            "model.language_model.layers.0.mlp.experts": QuantConfig(
                quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
            ),
            "model.language_model.layers.0.mlp.shared_expert.gate_proj": QuantConfig(
                quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
            ),
            "model.language_model.layers.0.mlp.shared_expert.up_proj": QuantConfig(
                quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
            ),
            "model.language_model.layers.0.mlp.shared_expert.down_proj": QuantConfig(
                quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
            ),
            "model.visual.patch_embed": QuantConfig(quant_algo=QuantAlgo.FP8),
            "mtp.layers.0.mlp.experts": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
        },
    )

    _normalize_qwen35_exclude_modules(model_config)

    assert model_config.quant_config.exclude_modules == ["model.layers.40*"]
    assert set(model_config.quant_config_dict) == {
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert.down_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
    }
    assert (
        model_config.quant_config_dict["model.layers.0.linear_attn.in_proj_qkvz"].quant_algo
        == QuantAlgo.FP8
    )
    assert (
        model_config.quant_config_dict["model.layers.0.mlp.experts"].quant_algo
        == QuantAlgo.W4A16_NVFP4
    )
    assert (
        model_config.quant_config_dict["model.layers.0.mlp.shared_expert.down_proj"].quant_algo
        == QuantAlgo.W4A16_NVFP4
    )
    assert (
        model_config.quant_config_dict["model.layers.0.mlp.shared_expert.gate_up_proj"].quant_algo
        == QuantAlgo.W4A16_NVFP4
    )


def test_qwen35_normalizes_frozen_layer_quant_config():
    model_config = ModelConfig(
        pretrained_config=types.SimpleNamespace(num_hidden_layers=40),
        quant_config=QuantConfig(exclude_modules=[]),
        quant_config_dict={
            "model.language_model.layers.0.mlp.shared_expert.down_proj": QuantConfig(
                quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
            ),
        },
    )
    model_config._frozen = True

    _normalize_qwen35_exclude_modules(model_config)

    assert model_config._frozen
    assert set(model_config.quant_config_dict) == {"model.layers.0.mlp.shared_expert.down_proj"}
    assert (
        model_config.quant_config_dict["model.layers.0.mlp.shared_expert.down_proj"].quant_algo
        == QuantAlgo.W4A16_NVFP4
    )


class _LayerwiseQuantRoutingMoE(MoE):
    @classmethod
    def can_implement(cls, quant_algo, dtype_activation=torch.bfloat16, swiglu_gptoss_style=False):
        return True, None

    def __init__(self):
        nn.Module.__init__(self)
        self.quant_config = QuantConfig()

    def create_weights(self):
        pass

    def load_weights(self, weights, allow_partial_loading=False):
        pass

    def quantize_input(self, x, **kwargs):
        return x, None

    def run_moe(self, x, token_selected_experts, token_final_scales, x_sf=None, **kwargs):
        return x

    def forward_impl(self, x, router_logits, **kwargs):
        return x


class _LayerwiseQuantRoutingModel(nn.Module):
    apply_layerwise_quant_config = DecoderModelForCausalLM.apply_layerwise_quant_config

    def __init__(self, quant_config_dict):
        super().__init__()
        self.model_config = types.SimpleNamespace(quant_config_dict=quant_config_dict)
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _LayerwiseQuantRoutingMoE()
        layer.mlp.experts.backend = _LayerwiseQuantRoutingMoE()
        self.model.layers = nn.ModuleList([layer])


class _LmHeadRoutingDecoder(nn.Module):
    def __pp_init__(self):
        pass


class _TiedLmHeadRoutingDecoder(_LmHeadRoutingDecoder):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Module()
        self.embed_tokens.tp_size = 1
        self.embed_tokens.tp_mode = TensorParallelMode.COLUMN
        self.embed_tokens.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))


def test_apply_layerwise_quant_config_updates_moe_inner_backend():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)
    model = _LayerwiseQuantRoutingModel({"model.layers.0.mlp.experts": quant_config})
    outer = model.model.layers[0].mlp.experts
    inner = outer.backend

    model.apply_layerwise_quant_config()

    assert outer.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert inner.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4


def test_decoder_model_routes_layer_quant_config_to_lm_head_before_weight_creation():
    hidden_size = 2048
    vocab_size = 2048
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)
    model_config = ModelConfig(
        pretrained_config=types.SimpleNamespace(
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        ),
        quant_config=QuantConfig(),
        quant_config_dict={"lm_head": quant_config},
        skip_create_weights_in_init=True,
    )

    model = DecoderModelForCausalLM(
        model=_LmHeadRoutingDecoder(),
        config=model_config,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )

    assert model.lm_head.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert tuple(model.lm_head.weight.shape) == (vocab_size, hidden_size // 2)


def test_decoder_model_preserves_tied_lm_head_with_delayed_weight_creation():
    hidden_size = 128
    vocab_size = 256
    decoder = _TiedLmHeadRoutingDecoder(hidden_size, vocab_size)
    model_config = ModelConfig(
        pretrained_config=types.SimpleNamespace(
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=True,
        ),
        skip_create_weights_in_init=True,
    )

    model = DecoderModelForCausalLM(
        model=decoder,
        config=model_config,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )

    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize(
    "architecture",
    ["NemotronHForCausalLM", "Qwen3MoeForCausalLM", "SomeOtherForCausalLM"],
)
def test_auto_moe_backend_selects_cutedsl_for_w4a16_sm12x(architecture):
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 121)
        moe_backend = ModelConfig.resolve_moe_backend_after_quant_config(
            "AUTO", architecture, quant_config
        )

    assert moe_backend == "CUTEDSL"


@pytest.mark.parametrize(
    "architecture",
    ["NemotronHForCausalLM", "Qwen3MoeForCausalLM", "SomeOtherForCausalLM"],
)
def test_auto_moe_backend_selects_cutedsl_for_layer_w4a16_sm12x(architecture):
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    layer_quant_config = {
        "model.layers.0.mlp.experts": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16),
    }

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 121)
        moe_backend = ModelConfig.resolve_moe_backend_after_quant_config(
            "AUTO", architecture, quant_config, layer_quant_config
        )

    assert moe_backend == "CUTEDSL"


@pytest.mark.parametrize(
    "architecture",
    ["NemotronHForCausalLM", "Qwen3MoeForCausalLM", "SomeOtherForCausalLM"],
)
def test_auto_moe_backend_keeps_cutlass_for_w4a16_on_other_sm(architecture):
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("tensorrt_llm._torch.model_config.get_sm_version", lambda: 100)
        moe_backend = ModelConfig.resolve_moe_backend_after_quant_config(
            "AUTO", architecture, quant_config
        )

    assert moe_backend == "CUTLASS"
