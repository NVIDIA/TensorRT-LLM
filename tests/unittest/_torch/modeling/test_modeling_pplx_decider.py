# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_pplx_decider
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import (
    PplxDeciderHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_pplx_decider import (
    PPLX_DECIDER_ARCHITECTURE,
    PPLX_DECIDER_NUM_LABELS,
    PplxDeciderForDecision,
    PplxDeciderFullAttentionDecoderLayer,
    PplxDeciderGatedDeltaNet,
    PplxDeciderLinearDecoderLayer,
    PplxDeciderModel,
)
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextFullAttentionDecoderLayer,
    Qwen3NextLinearDecoderLayer,
)
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _write_composite_config(tmp_path: Path) -> Path:
    config = {
        "architectures": ["Qwen3_5Model"],
        "dtype": "bfloat16",
        "model_type": "qwen3_5",
        "text_config": {
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "vocab_size": 248320,
        },
        "vision_config": {"model_type": "qwen3_5_vision"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_config_override_flattens_composite_checkpoint(tmp_path: Path) -> None:
    config = load_pretrained_config(
        str(_write_composite_config(tmp_path)),
        model_kwargs={"architectures": [PPLX_DECIDER_ARCHITECTURE]},
    )

    assert isinstance(config, transformers.Qwen3NextConfig)
    assert config.architectures == [PPLX_DECIDER_ARCHITECTURE]
    assert config.num_labels == PPLX_DECIDER_NUM_LABELS
    assert config.hidden_size == 5120
    assert not hasattr(config, "vision_config")


def test_model_registration_and_routing() -> None:
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class

    assert get_registered_model_class(PPLX_DECIDER_ARCHITECTURE) is PplxDeciderForDecision
    assert ModelConfig.is_generation_model([PPLX_DECIDER_ARCHITECTURE]) is False
    assert PplxDeciderForDecision.get_model_defaults(None) == {"attn_backend": "FLASHINFER"}


def test_decider_layer_overrides_are_isolated() -> None:
    assert PplxDeciderModel.decoder_layer_types == {
        "full_attention": PplxDeciderFullAttentionDecoderLayer,
        "linear_attention": PplxDeciderLinearDecoderLayer,
    }
    assert PplxDeciderLinearDecoderLayer.linear_attn_cls is PplxDeciderGatedDeltaNet
    assert Qwen3NextLinearDecoderLayer.linear_attn_cls is not PplxDeciderGatedDeltaNet
    assert PplxDeciderFullAttentionDecoderLayer.attention_mask.name == "FULL"
    assert Qwen3NextFullAttentionDecoderLayer.attention_mask.name == "CAUSAL"


def test_native_checkpoint_weight_names_are_mapped() -> None:
    mapper = AutoCheckpointMapper.get("HF", PPLX_DECIDER_ARCHITECTURE)
    backbone = torch.empty(1)
    readout = torch.empty(1)

    assert isinstance(mapper, PplxDeciderHfWeightMapper)
    normalized = mapper._normalize_weight_names(
        {
            "language_model.layers.0.input_layernorm.weight": backbone,
            "weight": readout,
            "visual.blocks.0.attn.qkv.weight": torch.empty(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.empty(1),
        }
    )
    assert set(normalized) == {
        "model.layers.0.input_layernorm.weight",
        "readout.weight",
    }
    assert normalized["model.layers.0.input_layernorm.weight"] is backbone
    assert normalized["readout.weight"] is readout


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_readout_is_replicated(tp_size: int, rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    backbone = torch.nn.Module()
    backbone.preload_weight_modules = []
    monkeypatch.setattr(modeling_pplx_decider, "PplxDeciderModel", lambda _: backbone)
    config = ModelConfig(
        pretrained_config=transformers.Qwen3NextConfig(
            hidden_size=8, num_labels=PPLX_DECIDER_NUM_LABELS, torch_dtype=torch.bfloat16
        ),
        mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=rank),
    )

    model = PplxDeciderForDecision.__new__(PplxDeciderForDecision)
    PplxDeciderForDecision.__init__(model, config)

    assert model.readout.weight.shape == (PPLX_DECIDER_NUM_LABELS, 8)
    assert model.readout.tp_mode is None
    assert model.readout.gather_output is False
    assert model.readout.all_reduce is None


def test_forward_returns_last_token_readout_per_sequence() -> None:
    model = PplxDeciderForDecision.__new__(PplxDeciderForDecision)
    torch.nn.Module.__init__(model)
    hidden = torch.arange(40, dtype=torch.bfloat16).reshape(5, 8)
    model.model = Mock(return_value=hidden)
    model.readout = torch.nn.Linear(8, PPLX_DECIDER_NUM_LABELS, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.readout.weight.fill_(0.125)
    metadata = SimpleNamespace(seq_lens_cuda=torch.tensor([2, 3], dtype=torch.int32))

    output = model(attn_metadata=metadata)

    expected = model.readout(hidden[[1, 4]]).float()
    assert output.shape == (2, PPLX_DECIDER_NUM_LABELS)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_backbone_passes_packed_sequence_boundaries() -> None:
    model = PplxDeciderModel.__new__(PplxDeciderModel)
    torch.nn.Module.__init__(model)
    hidden = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    layer = Mock(return_value=(hidden, None))
    model.layers = [layer]
    model.num_hidden_layers = 1
    metadata = SimpleNamespace(
        seq_lens_cuda=torch.tensor([2, 3], dtype=torch.int32), num_contexts=2
    )

    output = model(attn_metadata=metadata, inputs_embeds=hidden)

    assert output is hidden
    kwargs = layer.call_args.kwargs
    torch.testing.assert_close(
        kwargs["decider_cu_seqlens_i32"], torch.tensor([0, 2, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        kwargs["decider_cu_seqlens_i64"], torch.tensor([0, 2, 5], dtype=torch.int64)
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_mismatched_readout_label_count_is_rejected(lazy: bool) -> None:
    model = PplxDeciderForDecision.__new__(PplxDeciderForDecision)
    torch.nn.Module.__init__(model)
    model.num_labels = PPLX_DECIDER_NUM_LABELS
    shape = (PPLX_DECIDER_NUM_LABELS + 1, 8)
    readout = SimpleNamespace(get_shape=lambda: shape) if lazy else torch.empty(shape)
    weights = {"readout.weight": readout}
    mapper = Mock()
    mapper.preprocess_weights.return_value = weights

    with pytest.raises(ValueError, match="requires 255 decision labels.*256 rows"):
        model.load_weights(weights, weight_mapper=mapper)

    assert weights["readout.weight"] is readout


@pytest.mark.parametrize("has_readout", [True, False])
def test_supported_readout_and_partial_loads_are_forwarded(
    has_readout: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = PplxDeciderForDecision.__new__(PplxDeciderForDecision)
    torch.nn.Module.__init__(model)
    model.num_labels = PPLX_DECIDER_NUM_LABELS
    mapped = {"readout.weight": torch.empty(PPLX_DECIDER_NUM_LABELS, 8)} if has_readout else {}
    weights = {"weight": torch.empty(1)}
    mapper = Mock()
    mapper.preprocess_weights.return_value = mapped
    load_weights = Mock()
    monkeypatch.setattr(modeling_pplx_decider.DecoderModelForCausalLM, "load_weights", load_weights)

    model.load_weights(weights, weight_mapper=mapper, allow_partial_loading=True)

    mapper.preprocess_weights.assert_called_once_with(weights, allow_partial_loading=True)
    load_weights.assert_called_once_with(
        mapped, weight_mapper=mapper, params_map=None, allow_partial_loading=True
    )
    assert weights == {}
