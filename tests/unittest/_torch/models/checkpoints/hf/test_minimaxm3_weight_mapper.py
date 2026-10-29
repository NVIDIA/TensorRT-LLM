# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import re
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.minimaxm3_weight_mapper import (
    MINIMAX_M3_PARAMS_MAP,
    MiniMaxM3HfWeightMapper,
    nvfp4_weight_scale_2,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3ForCausalLM,
    MiniMaxM3VLForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_NUM_KV_HEADS = 4
_ROWS_PER_HEAD = 2


def _make_mapper(tp_size: int = 8, num_kv_heads: int = _NUM_KV_HEADS) -> MiniMaxM3HfWeightMapper:
    config = SimpleNamespace(num_key_value_heads=num_kv_heads, num_attention_heads=8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=tp_size, rank=0, tp_size=tp_size),
    )
    model = SimpleNamespace(model_config=model_config, config=config)
    mapper = MiniMaxM3HfWeightMapper()
    mapper.init_model_and_config(model, model_config)
    return mapper


def _duplicate_heads(tensor: torch.Tensor, repetitions: int) -> torch.Tensor:
    return (
        tensor.reshape(_NUM_KV_HEADS, _ROWS_PER_HEAD, -1)
        .repeat_interleave(repetitions, dim=0)
        .reshape(_NUM_KV_HEADS * repetitions * _ROWS_PER_HEAD, -1)
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ],
)
def test_mapper_registration_and_mx_fallback(architecture: str) -> None:
    assert isinstance(AutoCheckpointMapper.get("HF", architecture), MiniMaxM3HfWeightMapper)
    assert isinstance(AutoCheckpointMapper.get("MX", architecture), MiniMaxM3HfWeightMapper)


@pytest.mark.parametrize(
    "model_class",
    [MiniMaxM3ForCausalLM, MiniMaxM3VLForConditionalGeneration],
)
def test_load_weights_exposes_weight_mapper(model_class: type) -> None:
    assert "weight_mapper" in inspect.signature(model_class.load_weights).parameters


@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
def test_tp8_mxfp8_duplicates_kv_weight_and_scale_via_callbacks(scale_name: str) -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    prefix = "model.layers.0.self_attn"

    weights = {}
    sources = {}
    for offset, projection in enumerate(("k_proj", "v_proj")):
        weight = torch.arange(24, dtype=torch.float32).reshape(8, 3) + offset * 100
        scale = torch.arange(16, dtype=torch.uint8).reshape(8, 2) + offset * 32
        weights[f"{prefix}.{projection}.weight"] = weight
        weights[f"{prefix}.{projection}.{scale_name}"] = scale
        sources[projection] = {"weight": weight, scale_name: scale}

    mapped = mapper.apply_callbacks(module, "qkv_proj", prefix.split("."), weights)

    assert mapped[0] == {}
    for projection, projected in zip(("k_proj", "v_proj"), mapped[1:]):
        for name, source in sources[projection].items():
            torch.testing.assert_close(projected[name], _duplicate_heads(source, repetitions=2))


def test_nvfp4_scale_behavior_is_preserved() -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    weight_scale = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "k_proj",
        {
            "weight": weight,
            "weight_scale": weight_scale,
            "weight_scale_inv": scale_inv,
        },
    )

    torch.testing.assert_close(mapped["weight"], _duplicate_heads(weight, repetitions=2))
    torch.testing.assert_close(
        mapped["weight_scale"], _duplicate_heads(weight_scale, repetitions=2)
    )
    assert mapped["weight_scale_inv"] is scale_inv


def test_quant_config_none_is_guarded() -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=None)
    weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "k_proj",
        {
            "weight": weight,
            "weight_scale_inv": scale_inv,
        },
    )

    torch.testing.assert_close(mapped["weight"], _duplicate_heads(weight, repetitions=2))
    assert mapped["weight_scale_inv"] is scale_inv


def test_kv_scale_is_not_expanded_when_kv_heads_cover_tp() -> None:
    mapper = _make_mapper(tp_size=2)
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "v_proj",
        {"weight_scale_inv": scale_inv},
    )

    torch.testing.assert_close(mapped["weight_scale_inv"], scale_inv)


def test_gate_bias_params_map() -> None:
    mapper = MiniMaxM3HfWeightMapper()
    source_name = "model.layers.3.block_sparse_moe.e_score_correction_bias"
    target_name = "model.layers.3.block_sparse_moe.gate.e_score_correction_bias"
    bias = torch.arange(4, dtype=torch.float32)
    gate_weight = torch.ones(4, 4)

    renamed = mapper.rename_by_params_map(
        mapper.params_map,
        {
            source_name: bias,
            "model.layers.3.block_sparse_moe.gate.weight": gate_weight,
        },
    )

    assert source_name not in renamed
    assert renamed[target_name] is bias
    assert renamed["model.layers.3.block_sparse_moe.gate.weight"] is gate_weight


def _make_layerless_model(model_class: type = MiniMaxM3ForCausalLM) -> MiniMaxM3ForCausalLM:
    config = SimpleNamespace(num_key_value_heads=_NUM_KV_HEADS, num_attention_heads=8)
    model_config = ModelConfig(pretrained_config=config, mapping=Mapping())
    model = object.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.model_config = model_config
    # This mapper-only fixture has no decoder layers to refresh after loading.
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    # Set by SpecDecOneEngineForCausalLM.__init__, which this test bypasses.
    model.spec_config = None
    # Set by MiniMaxM3VLForConditionalGeneration.__init__ for text-only configs.
    model.vision_tower = None
    return model


def test_load_weights_accepts_base_mapper_without_params_map() -> None:
    model = _make_layerless_model()
    mapper = HfWeightMapper()
    source_name = "model.layers.3.block_sparse_moe.e_score_correction_bias"
    target_name = "model.layers.3.block_sparse_moe.gate.e_score_correction_bias"
    bias = torch.arange(4, dtype=torch.float32)

    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        model.load_weights({source_name: bias}, weight_mapper=mapper)

    call_kwargs = base_load_weights.call_args.kwargs
    assert call_kwargs["weight_mapper"] is mapper
    renamed = mapper.rename_by_params_map(call_kwargs["params_map"], {source_name: bias})
    assert renamed[target_name] is bias


# ---------------------------------------------------------------------------
# Compressed-tensors NVFP4 MoE packaging of the MiniMax M3 architecture
# ---------------------------------------------------------------------------

# Checkpoint naming of one routed expert in the multimodal checkpoint layout.
_EXPERT = "language_model.model.layers.3.block_sparse_moe.experts.7"

# weight_scale_2 = fp32(fp32(bf16(2688 / gs) * 2688) * fp32(1 / 2688**2)) as
# float32 bit patterns. Computed outside PyTorch: Python float arithmetic
# rounded to float32 after every operation (exact for these operations), BF16
# by round-to-nearest-even on the float32 bits.
# FP32 reciprocals of the stored weight_global_scale values.
_WEIGHT_SCALE_2_GOLDENS = {
    2688.0: 0x39C30C31,
    3000.0: 0x39AEC33E,
    2913.75: 0x39B3EF92,
    16999.5: 0x3876BB24,
    10000.0: 0x38D1B717,
}
_RECIPROCAL_GOLDENS = {1.0: 0x3F800000, 0.5: 0x40000000, 3.0: 0x3EAAAAAB}


def _f32_bits(value) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _packed_expert(
    gate_up_global_scale: float = 3000.0,
    down_global_scale: float = 2913.75,
    input_global_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Synthetic tensors of one packed NVFP4 expert, shaped like the checkpoint's."""
    generator = torch.Generator().manual_seed(0)
    weights = {}
    # (rows, packed bytes per row) for hidden 64, intermediate 32: w1/w3 are
    # [inter, hidden], w2 is [hidden, inter]; two E2M1 values per byte.
    for proj, (rows, packed_cols) in {"w1": (32, 32), "w2": (64, 16), "w3": (32, 32)}.items():
        prefix = f"{_EXPERT}.{proj}"
        weights[f"{prefix}.weight_packed"] = torch.randint(
            0, 256, (rows, packed_cols), dtype=torch.uint8, generator=generator
        )
        weights[f"{prefix}.weight_scale"] = (
            torch.rand(rows, packed_cols * 2 // 16, generator=generator) * 4
        ).to(torch.float8_e4m3fn)
        weights[f"{prefix}.weight_global_scale"] = torch.tensor(
            down_global_scale if proj == "w2" else gate_up_global_scale
        )
        weights[f"{prefix}.input_global_scale"] = torch.tensor(input_global_scale)
    return weights


@pytest.mark.parametrize("global_scale", sorted(_WEIGHT_SCALE_2_GOLDENS))
def test_nvfp4_weight_scale_2_matches_independent_goldens(global_scale: float) -> None:
    scale_2 = nvfp4_weight_scale_2(torch.tensor(global_scale))
    assert scale_2.dtype == torch.float32
    assert _f32_bits(scale_2) == _WEIGHT_SCALE_2_GOLDENS[global_scale]


def test_packed_nvfp4_expert_translates_to_nvfp4_moe_contract() -> None:
    source = _packed_expert()
    unrelated = {
        "language_model.model.layers.3.self_attn.q_proj.weight": torch.ones(4, 4),
        "language_model.model.layers.3.block_sparse_moe.gate.weight": torch.ones(4, 4),
    }
    out = MiniMaxM3HfWeightMapper().preprocess_weights({**source, **unrelated})

    for name, value in unrelated.items():
        assert out[name] is value
    for proj in ("w1", "w2", "w3"):
        prefix = f"{_EXPERT}.{proj}"
        for stale in ("weight_packed", "weight_global_scale", "input_global_scale"):
            assert f"{prefix}.{stale}" not in out
        # Packed FP4 bytes and E4M3 group scales are passed through, not copied or cast.
        assert out[f"{prefix}.weight"] is source[f"{prefix}.weight_packed"]
        assert out[f"{prefix}.weight"].dtype == torch.uint8
        assert out[f"{prefix}.weight_scale"] is source[f"{prefix}.weight_scale"]
        expected = _WEIGHT_SCALE_2_GOLDENS[2913.75 if proj == "w2" else 3000.0]
        assert _f32_bits(out[f"{prefix}.weight_scale_2"]) == expected
        assert out[f"{prefix}.weight_scale_2"].shape == ()
        assert _f32_bits(out[f"{prefix}.input_scale"]) == _RECIPROCAL_GOLDENS[1.0]
    assert len(out) == len(source) + len(unrelated)


@pytest.mark.parametrize("input_global_scale", sorted(_RECIPROCAL_GOLDENS))
def test_packed_nvfp4_input_scale_is_the_reciprocal(input_global_scale: float) -> None:
    out = MiniMaxM3HfWeightMapper().preprocess_weights(
        _packed_expert(input_global_scale=input_global_scale)
    )
    for proj in ("w1", "w2", "w3"):
        assert (
            _f32_bits(out[f"{_EXPERT}.{proj}.input_scale"])
            == _RECIPROCAL_GOLDENS[input_global_scale]
        )


@pytest.mark.parametrize("kind", ["weight_global_scale", "input_global_scale"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_packed_nvfp4_rejects_invalid_global_scale(kind: str, bad: float) -> None:
    weights = _packed_expert()
    name = f"{_EXPERT}.w2.{kind}"
    weights[name] = torch.tensor(bad)
    with pytest.raises(ValueError, match=re.escape(name)):
        MiniMaxM3HfWeightMapper().preprocess_weights(weights)


def test_packed_nvfp4_rejects_different_gate_up_weight_scales() -> None:
    weights = _packed_expert()
    weights[f"{_EXPERT}.w3.weight_global_scale"] = torch.tensor(3100.0)
    with pytest.raises(ValueError, match=re.escape(f"{_EXPERT}.w3.weight_global_scale")):
        MiniMaxM3HfWeightMapper().preprocess_weights(weights)


def test_canonical_nvfp4_expert_tensors_are_not_inverted_again() -> None:
    canonical = {}
    for proj in ("w1", "w2", "w3"):
        prefix = f"model.layers.3.block_sparse_moe.experts.0.{proj}"
        canonical[f"{prefix}.weight"] = torch.zeros(8, 32, dtype=torch.uint8)
        canonical[f"{prefix}.weight_scale"] = torch.ones(8, 4).to(torch.float8_e4m3fn)
        canonical[f"{prefix}.weight_scale_2"] = torch.tensor(1.0 / 3000.0)
        canonical[f"{prefix}.input_scale"] = torch.tensor(0.25)
    out = MiniMaxM3HfWeightMapper().preprocess_weights(canonical)
    assert out is canonical


def test_packed_nvfp4_translation_is_idempotent() -> None:
    mapper = MiniMaxM3HfWeightMapper()
    once = mapper.preprocess_weights(_packed_expert())
    assert mapper.preprocess_weights(once) is once


def test_packed_nvfp4_translation_takes_over_consumable_weights() -> None:
    source = _packed_expert()
    packed = source[f"{_EXPERT}.w1.weight_packed"]
    consumable = ConsumableWeightsDict(dict(source))
    out = MiniMaxM3HfWeightMapper().preprocess_weights(consumable)
    assert isinstance(out, ConsumableWeightsDict)
    # The translated dict now owns the tensors, so consuming them frees memory.
    assert len(consumable) == 0
    assert out[f"{_EXPERT}.w1.weight"] is packed
    assert out.mark_consumed(_EXPERT) == len(source)


def _bare_model(model_class: type) -> torch.nn.Module:
    config = SimpleNamespace(num_key_value_heads=_NUM_KV_HEADS, num_attention_heads=8)
    model = object.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.model_config = ModelConfig(pretrained_config=config, mapping=Mapping())
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    model.spec_config = None
    if model_class is MiniMaxM3VLForConditionalGeneration:
        model.vision_tower = None
        model.last_loaded_vision_keys = []
        model.last_missing_vision_keys = []
    return model


def test_vl_load_weights_translates_packed_experts_before_loading() -> None:
    model = _bare_model(MiniMaxM3VLForConditionalGeneration)
    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        model.load_weights(_packed_expert(), weight_mapper=MiniMaxM3HfWeightMapper())

    loaded = base_load_weights.call_args.kwargs["weights"]
    expert = _EXPERT[len("language_model.") :]
    assert sorted(loaded.keys()) == sorted(
        f"{expert}.{proj}.{kind}"
        for proj in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    )
    assert _f32_bits(loaded[f"{expert}.w1.weight_scale_2"]) == _WEIGHT_SCALE_2_GOLDENS[3000.0]
    assert _f32_bits(loaded[f"{expert}.w2.weight_scale_2"]) == _WEIGHT_SCALE_2_GOLDENS[2913.75]


def test_vl_load_weights_names_the_checkpoint_tensor_of_an_invalid_scale() -> None:
    model = _bare_model(MiniMaxM3VLForConditionalGeneration)
    weights = _packed_expert()
    name = f"{_EXPERT}.w2.input_global_scale"
    weights[name] = torch.tensor(0.0)
    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        with pytest.raises(ValueError, match=re.escape(name)):
            model.load_weights(weights, weight_mapper=HfWeightMapper())
    base_load_weights.assert_not_called()


def test_text_load_weights_translates_packed_experts_with_any_mapper() -> None:
    # The translation does not depend on the mapper driving the load, and the
    # M3 params map still reaches the base loader.
    model = _bare_model(MiniMaxM3ForCausalLM)
    expert = _EXPERT[len("language_model.") :]
    weights = {k[len("language_model.") :]: v for k, v in _packed_expert().items()}
    mapper = HfWeightMapper()
    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        model.load_weights(weights, weight_mapper=mapper)

    call_kwargs = base_load_weights.call_args.kwargs
    assert call_kwargs["weight_mapper"] is mapper
    assert call_kwargs["params_map"] == MINIMAX_M3_PARAMS_MAP
    loaded = call_kwargs["weights"]
    assert sorted(loaded.keys()) == sorted(
        f"{expert}.{proj}.{kind}"
        for proj in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    )
    assert loaded[f"{expert}.w3.weight"] is weights[f"{expert}.w3.weight_packed"]
    assert _f32_bits(loaded[f"{expert}.w3.weight_scale_2"]) == _WEIGHT_SCALE_2_GOLDENS[3000.0]


@pytest.mark.parametrize(
    ("model_class", "checkpoint_prefix"),
    [
        (MiniMaxM3ForCausalLM, ""),
        (MiniMaxM3VLForConditionalGeneration, "language_model."),
    ],
    ids=["text", "vl"],
)
def test_load_weights_converts_compressed_tensors_nvfp4_experts(
    model_class: type, checkpoint_prefix: str
) -> None:
    model = _make_layerless_model(model_class)
    expert = "model.layers.3.block_sparse_moe.experts.0.w1"
    packed = torch.tensor([[1, 2]], dtype=torch.uint8)
    block_scale = torch.ones(1, 1).to(torch.float8_e4m3fn)
    weights = {
        f"{checkpoint_prefix}{expert}.weight_packed": packed,
        f"{checkpoint_prefix}{expert}.weight_scale": block_scale,
        f"{checkpoint_prefix}{expert}.weight_global_scale": torch.tensor([4.0]),
        f"{checkpoint_prefix}{expert}.input_global_scale": torch.tensor([0.0]),
    }

    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        model.load_weights(weights, weight_mapper=HfWeightMapper())

    loaded = base_load_weights.call_args.kwargs["weights"]
    assert set(loaded.keys()) == {
        f"{expert}.weight",
        f"{expert}.weight_scale",
        f"{expert}.weight_scale_2",
        f"{expert}.input_scale",
    }
    assert loaded[f"{expert}.weight"] is packed
    assert loaded[f"{expert}.weight_scale"] is block_scale
    torch.testing.assert_close(loaded[f"{expert}.weight_scale_2"], torch.tensor([0.25]))
    torch.testing.assert_close(loaded[f"{expert}.input_scale"], torch.tensor([0.0]))
