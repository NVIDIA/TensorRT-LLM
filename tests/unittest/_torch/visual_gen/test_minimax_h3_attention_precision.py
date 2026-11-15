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

"""The ``attention_precision`` entry of a MiniMax-H3 checkpoint: parsing, precedence against the
run-time option, validation against the model, and the configuration each attention is built with
(CPU only)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import attention_precision as ap
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

pytestmark = pytest.mark.cpu_only

_CONFIG: dict[str, object] = {
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 3,
    "num_refiner_layers": 2,
    "ffn_dim": 64,
    "in_channels": 16,
    "audio_in_channels": 16,
    "patch_size": (1, 1, 1),
    "text_dim": 64,
    "freq_dim": 8,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 64,
    "rope_freq_dim": 4,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}
_ALL = [
    "token_refiner.refiner_blocks.0.attn",
    "token_refiner.refiner_blocks.1.attn",
    "transformer_blocks.0.attn",
    "transformer_blocks.1.attn",
    "transformer_blocks.2.attn",
]
# The shape of the recipe's entry on this small model: one transformer block and the refiner.
_LAYERS = {
    "transformer_blocks.1.attn": "bf16",
    "token_refiner.refiner_blocks.0.attn": "bf16",
    "token_refiner.refiner_blocks.1.attn": "bf16",
}


def _entry(layers=None, **overrides) -> dict:
    entry = {"schema_version": 1, "layers": dict(_LAYERS if layers is None else layers)}
    entry.update(overrides)
    return entry


def _model_config(*, checkpoint=None, runtime=None) -> DiffusionModelConfig:
    pretrained = dict(_CONFIG)
    if checkpoint is not None:
        pretrained["quantization_config"] = {
            "quant_algo": "NVFP4",
            "attention_precision": checkpoint,
        }
    extra = {} if runtime is None else {ap.RUNTIME_KEY: runtime}
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**pretrained),
        quant_config=QuantConfig(),
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        extra_attrs=extra,
    )


@pytest.fixture
def built_with(monkeypatch):
    """Build the transformer and return, per attention module, whether it was handed the model's
    own configuration object (True) or a copy without quantized or sparse attention (False)."""

    def build(model_config: DiffusionModelConfig):
        received: dict[str, DiffusionModelConfig] = {}
        original = h3.MiniMaxH3Attention.__init__

        def recording_init(self, *, model_config, module_name, **kwargs):
            received[module_name] = model_config
            original(self, model_config=model_config, module_name=module_name, **kwargs)

        monkeypatch.setattr(h3.MiniMaxH3Attention, "__init__", recording_init)
        model = h3.MiniMaxH3Transformer3DModel(model_config)
        assert sorted(received) == sorted(_ALL)
        for name, config in received.items():
            if config is not model_config:
                assert config.attention.quant_attention_config is None, name
                assert getattr(config.attention, "sparse_attention_config", None) is None, name
                assert config.attention.backend == model_config.attention.backend, name
        return model, {name: config is model_config for name, config in received.items()}

    return build


class TestParsing:
    def test_valid_entry_returns_its_layers(self):
        assert ap.parse_attention_precision(_entry(), "test") == _LAYERS

    def test_empty_layers_is_an_empty_mapping(self):
        assert ap.parse_attention_precision(_entry(layers={}), "test") == {}

    @pytest.mark.parametrize(
        "entry, reason",
        [
            (["transformer_blocks.1.attn"], "must be a mapping"),
            (_entry(schema_version=2), "schema_version 2 is not supported"),
            (_entry(schema_version="1"), "schema_version '1' is not supported"),
            (_entry(schema_version=True), "schema_version True is not supported"),
            ({"layers": dict(_LAYERS)}, "schema_version None is not supported"),
            (_entry(default="fp8"), "unknown keys"),
            ({"schema_version": 1}, "layers must be a mapping"),
            (
                {"schema_version": 1, "layers": ["transformer_blocks.1.attn"]},
                "layers must be a mapping",
            ),
            (_entry(layers={"transformer_blocks.1.attn": "fp8"}), "only 'bf16' is supported"),
            (_entry(layers={"transformer_blocks.1.attn": "BF16"}), "only 'bf16' is supported"),
            (_entry(layers={"transformer_blocks.1.attn": None}), "only 'bf16' is supported"),
            (_entry(layers={"transformer_blocks.*.attn": "bf16"}), "is a pattern"),
            (_entry(layers={"token_refiner*": "bf16"}), "is a pattern"),
            (_entry(layers={"": "bf16"}), "invalid module name"),
            (_entry(layers={45: "bf16"}), "invalid module name"),
            (_entry(schema_version=1.0), "schema_version 1.0 is not supported"),
            (_entry(schema_version=False), "schema_version False is not supported"),
        ],
    )
    def test_malformed_entry_is_refused(self, entry, reason):
        with pytest.raises(ValueError, match=reason):
            ap.parse_attention_precision(entry, "test")


class TestResolution:
    def test_absent_everywhere_is_none(self):
        assert ap.resolve_attention_precision(SimpleNamespace(**_CONFIG), {}) is None
        no_entry = SimpleNamespace(quantization_config={"quant_algo": "NVFP4"})
        assert ap.resolve_attention_precision(no_entry, {ap.RUNTIME_KEY: None}) is None
        assert ap.resolve_attention_precision(None, None) is None

    def test_checkpoint_entry_alone(self):
        config = _model_config(checkpoint=_entry())
        resolved = ap.resolve_attention_precision(config.pretrained_config, config.extra_attrs)
        assert resolved == _LAYERS

    def test_runtime_option_alone(self):
        config = _model_config(runtime=_entry())
        resolved = ap.resolve_attention_precision(config.pretrained_config, config.extra_attrs)
        assert resolved == _LAYERS

    def test_equal_checkpoint_and_runtime_are_accepted(self):
        config = _model_config(checkpoint=_entry(), runtime=_entry())
        resolved = ap.resolve_attention_precision(config.pretrained_config, config.extra_attrs)
        assert resolved == _LAYERS

    @pytest.mark.parametrize(
        "runtime_layers",
        [
            {"transformer_blocks.1.attn": "bf16"},
            {**_LAYERS, "transformer_blocks.0.attn": "bf16"},
            {},
        ],
    )
    def test_differing_checkpoint_and_runtime_fail(self, runtime_layers):
        config = _model_config(checkpoint=_entry(), runtime=_entry(layers=runtime_layers))
        with pytest.raises(ValueError, match="differ in"):
            ap.resolve_attention_precision(config.pretrained_config, config.extra_attrs)

    def test_malformed_runtime_option_names_its_source(self):
        config = _model_config(runtime=_entry(layers={"transformer_blocks.1.attn": "fp8"}))
        with pytest.raises(ValueError, match="pipeline_config"):
            ap.resolve_attention_precision(config.pretrained_config, config.extra_attrs)


class TestValidation:
    _KWARGS = dict(num_layers=3, num_refiner_layers=2, model_dtype=torch.bfloat16, ulysses_size=1)

    def test_module_names_follow_the_model(self):
        assert ap.attention_module_names(3, 2) == _ALL

    def test_recipe_shaped_entry_is_valid(self):
        ap.validate_attention_precision(_LAYERS, **self._KWARGS)

    @pytest.mark.parametrize(
        "name",
        [
            "transformer_blocks.3.attn",  # one past the last block
            "token_refiner.refiner_blocks.2.attn",
            "transformer_blocks.1",
            "transformer_blocks.1.attn.to_q",
            "transformer_blocks.1.ff",
            "proj_in",
        ],
    )
    def test_unknown_module_is_refused(self, name):
        with pytest.raises(ValueError, match="not attention modules of this model"):
            ap.validate_attention_precision({name: "bf16"}, **self._KWARGS)

    def test_other_model_dtype_is_refused(self):
        with pytest.raises(ValueError, match="model's dtype"):
            ap.validate_attention_precision(
                _LAYERS, **{**self._KWARGS, "model_dtype": torch.float16}
            )

    def test_ulysses_4_is_valid(self):
        ap.validate_attention_precision(_LAYERS, **{**self._KWARGS, "ulysses_size": 4})

    @pytest.mark.parametrize("ulysses", [2, 3, 8])
    def test_other_ulysses_sizes_are_refused(self, ulysses):
        with pytest.raises(NotImplementedError, match="Ulysses 1 and 4"):
            ap.validate_attention_precision(_LAYERS, **{**self._KWARGS, "ulysses_size": ulysses})

    def test_async_ulysses_is_refused(self):
        with pytest.raises(NotImplementedError, match="async_ulysses"):
            ap.validate_attention_precision(
                _LAYERS, **{**self._KWARGS, "ulysses_size": 4, "async_ulysses": True}
            )

    def test_async_flag_at_ulysses_1_is_ignored(self):
        ap.validate_attention_precision(_LAYERS, **{**self._KWARGS, "async_ulysses": True})

    def test_refusal_names_the_supported_sizes(self):
        """The message is derived from ``SUPPORTED_ULYSSES_SIZES``, not spelled beside it."""
        text = " and ".join(str(size) for size in ap.SUPPORTED_ULYSSES_SIZES)
        assert ap.SUPPORTED_ULYSSES_SIZES_TEXT == text
        with pytest.raises(NotImplementedError, match=f"Ulysses {text}, found Ulysses 8"):
            ap.validate_attention_precision(_LAYERS, **{**self._KWARGS, "ulysses_size": 8})

    def test_empty_mapping_skips_every_check(self):
        ap.validate_attention_precision({}, **{**self._KWARGS, "ulysses_size": 4})


class TestUnquantizedAttentionConfig:
    def test_copy_drops_quantized_and_sparse_attention_only(self):
        base = _model_config()
        quantized = object()
        sparse = SimpleNamespace(algorithm="vsa")
        attention = base.attention.model_copy(
            update={"quant_attention_config": quantized, "sparse_attention_config": sparse}
        )
        config = base.model_copy(update={"attention": attention})
        copy = h3._unquantized_attention_model_config(config)
        assert copy is not config and copy.attention is not config.attention
        assert copy.attention.quant_attention_config is None
        assert copy.attention.sparse_attention_config is None
        assert copy.attention.backend == "VANILLA"
        assert copy.quant_config is config.quant_config
        # The model's own configuration is left as it was.
        assert config.attention.quant_attention_config is quantized
        assert config.attention.sparse_attention_config is sparse


class TestModel:
    def test_absent_entry_builds_every_attention_from_the_model_config(self, built_with):
        model, own_config = built_with(_model_config())
        assert model.attention_precision == {}
        assert own_config == dict.fromkeys(_ALL, True)

    @pytest.mark.parametrize("source", ["checkpoint", "runtime", "both"])
    def test_entry_selects_the_listed_attentions(self, built_with, source):
        kwargs = {}
        if source in ("checkpoint", "both"):
            kwargs["checkpoint"] = _entry()
        if source in ("runtime", "both"):
            kwargs["runtime"] = _entry()
        model, own_config = built_with(_model_config(**kwargs))
        assert model.attention_precision == _LAYERS
        assert own_config == {name: name not in _LAYERS for name in _ALL}

    def test_empty_entry_changes_nothing(self, built_with):
        model, own_config = built_with(_model_config(checkpoint=_entry(layers={})))
        assert model.attention_precision == {}
        assert own_config == dict.fromkeys(_ALL, True)

    @pytest.mark.parametrize(
        "layers, error, reason",
        [
            ({"transformer_blocks.3.attn": "bf16"}, ValueError, "not attention modules"),
            ({"transformer_blocks.9.ff.net.2": "bf16"}, ValueError, "not attention modules"),
            ({"transformer_blocks.1.attn": "fp8"}, ValueError, "only 'bf16'"),
            ({"transformer_blocks.*.attn": "bf16"}, ValueError, "is a pattern"),
        ],
    )
    def test_bad_entry_fails_before_the_model_is_built(self, monkeypatch, layers, error, reason):
        def no_attention(*args, **kwargs):
            raise AssertionError("an attention was built before the entry was validated")

        monkeypatch.setattr(h3.MiniMaxH3Attention, "__init__", no_attention)
        with pytest.raises(error, match=reason):
            h3.MiniMaxH3Transformer3DModel(_model_config(checkpoint=_entry(layers=layers)))

    def test_conflicting_sources_fail_before_the_model_is_built(self, monkeypatch):
        def no_attention(*args, **kwargs):
            raise AssertionError("an attention was built before the entry was validated")

        monkeypatch.setattr(h3.MiniMaxH3Attention, "__init__", no_attention)
        config = _model_config(
            checkpoint=_entry(), runtime=_entry(layers={"transformer_blocks.0.attn": "bf16"})
        )
        with pytest.raises(ValueError, match="differ in"):
            h3.MiniMaxH3Transformer3DModel(config)
