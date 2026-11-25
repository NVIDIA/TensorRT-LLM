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

"""ModelOpt ``MIXED_PRECISION`` exports (per-layer NVFP4 / FP8) on the MiniMax-H3 transformer.

Model construction and the per-Linear quant methods need no GPU; the checks run on CPU. The
producer decisions belong to the scenario ``minimax_h3_nvfp4__sm_107__u1_u4``, whose modules build
SM 107 kernels at import, so that one test needs an SM 107 device.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    FP8QDQLinearMethod,
    Linear,
    NVFP4LinearMethod,
    UnquantizedLinearMethod,
)
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    DiffusionPipelineConfig,
    resolve_layer_quant_config,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig

_SCENARIO_PACKAGE = (
    "tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4"
)

# Static NVFP4 / FP8 Linears need in_features % 16 == 0 and GEMM-friendly widths, so every
# dimension that feeds a Linear is widened from the tiny architecture tests use.
_STATIC_CONFIG: dict[str, object] = {
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 0,
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

_BLOCK_LINEARS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

# The BF16 set of a MiniMax-H3 ModelOpt export, in the export's own spelling (fnmatch globs).
_IGNORE = [
    "audio_proj_in",
    "audio_proj_out",
    "context_embedder",
    "norm_out*",
    "proj_in",
    "proj_out",
    "time_embedder*",
    "token_refiner*",
    "transformer_blocks.0.adaln_proj*",
    "transformer_blocks.1.adaln_proj*",
]

_NVFP4_GROUP = {
    "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
}
_FP8_GROUP = {
    "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
}


def _mixed_precision_export(fp8_layers: set[str], *, quantized_layers: bool = True) -> dict:
    """A two-group ModelOpt export: ``fp8_layers`` in the FP8 group, every other block Linear NVFP4."""
    names = [f"transformer_blocks.{i}.{leaf}" for i in range(2) for leaf in _BLOCK_LINEARS]
    nvfp4 = [n for n in names if n not in fp8_layers]
    fp8 = [n for n in names if n in fp8_layers]
    config = {
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "config_groups": {
            "group_0": {**_NVFP4_GROUP, "targets": nvfp4},
            "group_1": {**_FP8_GROUP, "targets": fp8},
        },
        "ignore": list(_IGNORE),
    }
    if quantized_layers:
        config["quantized_layers"] = {
            **{n: {"quant_algo": "NVFP4", "group_size": 16} for n in nvfp4},
            **{n: {"quant_algo": "FP8"} for n in fp8},
        }
    return config


# Block 1's QKV and block 0's down projection recovered to FP8, as an AutoQuantize export does.
_FP8_LAYERS = {
    "transformer_blocks.1.attn.to_q",
    "transformer_blocks.1.attn.to_k",
    "transformer_blocks.1.attn.to_v",
    "transformer_blocks.0.ff.net.2",
}


def _make_model_config(
    quant_config: QuantConfig, quant_config_dict: dict | None = None
) -> DiffusionModelConfig:
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**_STATIC_CONFIG),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
    )


def _build(export: dict) -> h3.MiniMaxH3Transformer3DModel:
    quant_config, layer_quant_config, dynamic_w, dynamic_a = (
        DiffusionPipelineConfig.load_diffusion_quant_config(export)
    )
    assert (dynamic_w, dynamic_a) == (False, False)
    return h3.MiniMaxH3Transformer3DModel(_make_model_config(quant_config, layer_quant_config))


class TestMixedPrecisionParsing:
    def test_two_group_export_parses(self):
        quant_config, layers, dynamic_w, dynamic_a = (
            DiffusionPipelineConfig.load_diffusion_quant_config(
                _mixed_precision_export(_FP8_LAYERS)
            )
        )
        # The dominant group (8 of 12 Linears) is the global recipe; ``ignore`` is its exclusion list.
        assert quant_config.quant_algo == QuantAlgo.NVFP4
        assert quant_config.group_size == 16
        assert quant_config.exclude_modules == _IGNORE
        assert (dynamic_w, dynamic_a) == (False, False)
        assert len(layers) == 12
        for name in _FP8_LAYERS:
            assert layers[name].quant_algo == QuantAlgo.FP8
        assert layers["transformer_blocks.0.attn.to_q"].quant_algo == QuantAlgo.NVFP4
        assert layers["transformer_blocks.0.attn.to_q"].group_size == 16

    def test_targets_alone_are_enough(self):
        """Older exports carry ``config_groups`` targets without ``quantized_layers``."""
        _, layers, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
            _mixed_precision_export(_FP8_LAYERS, quantized_layers=False)
        )
        assert {n for n, c in layers.items() if c.quant_algo == QuantAlgo.FP8} == _FP8_LAYERS

    def test_fp8_majority_becomes_global(self):
        names = {f"transformer_blocks.{i}.{leaf}" for i in range(2) for leaf in _BLOCK_LINEARS}
        quant_config, _, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
            _mixed_precision_export(names - {"transformer_blocks.0.ff.net.2"})
        )
        assert quant_config.quant_algo == QuantAlgo.FP8

    def test_conflicting_assignment_raises(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["quantized_layers"]["transformer_blocks.0.attn.to_q"] = {"quant_algo": "FP8"}
        with pytest.raises(ValueError, match="both NVFP4 and FP8"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @pytest.mark.parametrize(
        "group",
        [
            {**_NVFP4_GROUP, "weights": {**_NVFP4_GROUP["weights"], "dynamic": True}},
            {**_FP8_GROUP, "weights": {"dynamic": False, "num_bits": 8, "type": "int"}},
            {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            },
            {**_NVFP4_GROUP, "weights": {**_NVFP4_GROUP["weights"], "group_size": 32}},
        ],
        ids=["dynamic", "int8", "w4a8", "block32"],
    )
    def test_unsupported_group_raises(self, group):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["config_groups"]["group_1"] = {**group, "targets": sorted(_FP8_LAYERS)}
        with pytest.raises(ValueError, match="MIXED_PRECISION group 'group_1'"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    def test_group_without_targets_raises(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["config_groups"]["group_1"]["targets"] = []
        with pytest.raises(ValueError, match="lists no targets"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @pytest.mark.parametrize(
        "target", ["transformer_blocks.*.ff.net.2", "transformer_blocks.1.attn.to_?", "blocks[0]"]
    )
    def test_pattern_target_is_refused(self, target):
        """One pattern would count as one Linear and could contradict an exact name unseen."""
        export = _mixed_precision_export(_FP8_LAYERS, quantized_layers=False)
        export["config_groups"]["group_1"]["targets"].append(target)
        with pytest.raises(ValueError, match="names the pattern"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    def test_pattern_in_quantized_layers_is_refused(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["quantized_layers"]["transformer_blocks.*.attn.to_q"] = {"quant_algo": "FP8"}
        with pytest.raises(ValueError, match="quantized_layers names the pattern"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @pytest.mark.parametrize("target", ["", None, 7, ["transformer_blocks.0.attn.to_q"]])
    def test_invalid_target_name_is_refused(self, target):
        export = _mixed_precision_export(_FP8_LAYERS, quantized_layers=False)
        export["config_groups"]["group_1"]["targets"].append(target)
        with pytest.raises(ValueError, match="invalid module name"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @staticmethod
    def _regrouped(groups: dict) -> dict:
        return {
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {
                name: {**(_NVFP4_GROUP if algo == "nvfp4" else _FP8_GROUP), "targets": targets}
                for name, (algo, targets) in groups.items()
            },
        }

    def test_dominant_algorithm_counts_names_not_groups(self):
        """Three NVFP4 names split over three groups outnumber two FP8 names in one group."""
        names = [f"transformer_blocks.0.{leaf}" for leaf in _BLOCK_LINEARS]
        groups = {
            "a": ("nvfp4", names[0:1]),
            "b": ("fp8", names[1:3]),
            "c": ("nvfp4", names[3:4]),
            "d": ("nvfp4", names[4:5]),
        }
        for order in (list(groups), list(reversed(groups))):
            quant_config, layers, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
                self._regrouped({name: groups[name] for name in order})
            )
            assert quant_config.quant_algo == QuantAlgo.NVFP4, order
            assert quant_config.group_size == 16
            assert len(layers) == 5

    def test_repeated_names_do_not_inflate_the_count(self):
        names = [f"transformer_blocks.0.{leaf}" for leaf in _BLOCK_LINEARS]
        # NVFP4: one name listed three times and once more in a second group. FP8: two names.
        groups = {
            "a": ("nvfp4", [names[0]] * 3),
            "b": ("fp8", names[1:3]),
            "c": ("nvfp4", names[0:1]),
        }
        quant_config, layers, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
            self._regrouped(groups)
        )
        assert quant_config.quant_algo == QuantAlgo.FP8
        assert len(layers) == 3

    def test_tie_goes_to_nvfp4_in_either_group_order(self):
        names = [f"transformer_blocks.0.{leaf}" for leaf in _BLOCK_LINEARS]
        groups = {"x": ("fp8", names[0:3]), "y": ("nvfp4", names[3:6])}
        for order in (["x", "y"], ["y", "x"]):
            quant_config, _, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
                self._regrouped({name: groups[name] for name in order})
            )
            assert quant_config.quant_algo == QuantAlgo.NVFP4, order
            assert quant_config.group_size == 16

    @pytest.mark.parametrize(
        "quantized_layers", [[], ["transformer_blocks.0.attn.to_q"], "NVFP4", 0, False]
    )
    def test_quantized_layers_must_be_a_mapping(self, quantized_layers):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["quantized_layers"] = quantized_layers
        with pytest.raises(ValueError, match="quantized_layers must be a mapping"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @pytest.mark.parametrize(
        "layer, reason",
        [
            ("NVFP4", "must be a mapping"),
            (None, "must be a mapping"),
            (["NVFP4"], "must be a mapping"),
            ({}, "unsupported quant_algo None"),
            ({"quant_algo": "FP8_BLOCK_SCALES"}, "unsupported quant_algo 'FP8_BLOCK_SCALES'"),
            ({"quant_algo": "W4A16_AWQ"}, "unsupported quant_algo 'W4A16_AWQ'"),
            ({"quant_algo": ["NVFP4"]}, "unsupported quant_algo"),
            ({"quant_algo": "NVFP4", "group_size": 32}, "group_size 32"),
            ({"quant_algo": "NVFP4", "group_size": 16.0}, "group_size 16.0"),
            ({"quant_algo": "NVFP4", "group_size": None}, "group_size None"),
            ({"quant_algo": "NVFP4", "group_size": True}, "group_size True"),
        ],
    )
    def test_malformed_quantized_layer_is_refused(self, layer, reason):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["quantized_layers"]["transformer_blocks.0.attn.to_q"] = layer
        with pytest.raises(ValueError, match=reason):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    def test_fp8_layer_with_a_group_size_is_refused(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["quantized_layers"]["transformer_blocks.0.ff.net.2"] = {
            "quant_algo": "FP8",
            "group_size": 16,
        }
        with pytest.raises(ValueError, match="group_size 16"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    def test_agreeing_repeated_declarations_pass(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        export["config_groups"]["group_2"] = {
            **_FP8_GROUP,
            "targets": ["transformer_blocks.0.ff.net.2"],
        }
        _, layers, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(export)
        assert len(layers) == 12
        assert layers["transformer_blocks.0.ff.net.2"].quant_algo == QuantAlgo.FP8

    def test_conflicting_groups_raise(self):
        export = _mixed_precision_export(_FP8_LAYERS, quantized_layers=False)
        export["config_groups"]["group_2"] = {
            **_NVFP4_GROUP,
            "targets": ["transformer_blocks.0.ff.net.2"],
        }
        with pytest.raises(ValueError, match="both FP8 and NVFP4"):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    @pytest.mark.parametrize(
        "config_groups, reason",
        [
            ([], "has no config_groups"),
            ("group_0", "has no config_groups"),
            ({"group_0": ["transformer_blocks.0.attn.to_q"]}, "group 'group_0' must be a mapping"),
            ({"group_0": {"targets": ["a"]}}, "needs a 'weights' mapping"),
            (
                {"group_0": {"weights": _NVFP4_GROUP["weights"], "targets": ["a"]}},
                "needs a 'input_activations' mapping",
            ),
            ({"group_0": {**_NVFP4_GROUP, "targets": "a"}}, "lists no targets"),
        ],
    )
    def test_malformed_config_groups_are_refused(self, config_groups, reason):
        export = {"quant_algo": "MIXED_PRECISION", "config_groups": config_groups}
        with pytest.raises(ValueError, match=reason):
            DiffusionPipelineConfig.load_diffusion_quant_config(export)

    def test_pattern_targets_resolve(self):
        layers = {
            "transformer_blocks.*.ff.net.2": QuantConfig(quant_algo=QuantAlgo.FP8),
            "transformer_blocks.3.ff.net.2": QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        }
        # An exact entry wins over a pattern that also matches.
        assert (
            resolve_layer_quant_config(layers, "transformer_blocks.3.ff.net.2").quant_algo
            == QuantAlgo.NVFP4
        )
        assert (
            resolve_layer_quant_config(layers, "transformer_blocks.7.ff.net.2").quant_algo
            == QuantAlgo.FP8
        )
        assert resolve_layer_quant_config(layers, "transformer_blocks.7.attn.to_q") is None
        model_config = _make_model_config(QuantConfig(quant_algo=QuantAlgo.NVFP4), layers)
        assert model_config.get_layer_quant_config("transformer_blocks.7.attn.to_q") is None
        assert (
            model_config.get_quant_config("transformer_blocks.7.attn.to_q").quant_algo
            == QuantAlgo.NVFP4
        )
        assert (
            model_config.get_quant_config("transformer_blocks.7.ff.net.2").quant_algo
            == QuantAlgo.FP8
        )

    def test_single_precision_export_has_no_layer_dict(self):
        """The NVFP4 twin (ignore list instead of a second group) keeps the single-recipe path."""
        export = {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "config_groups": {"group_0": {**_NVFP4_GROUP, "targets": ["Linear"]}},
            "ignore": _IGNORE + ["transformer_blocks.1.attn.to_out*"],
        }
        quant_config, layers, dynamic_w, dynamic_a = (
            DiffusionPipelineConfig.load_diffusion_quant_config(export)
        )
        assert layers is None
        assert quant_config.quant_algo == QuantAlgo.NVFP4
        assert quant_config.exclude_modules[-1] == "transformer_blocks.1.attn.to_out*"
        assert (dynamic_w, dynamic_a) == (False, False)


def _linear(model, name):
    return model.get_submodule(name)


class TestMixedPrecisionModel:
    def test_each_linear_gets_its_group_method(self):
        model = _build(_mixed_precision_export(_FP8_LAYERS))

        fp8 = ("transformer_blocks.1.attn.qkv_proj", "transformer_blocks.0.ff.down_proj")
        nvfp4 = (
            "transformer_blocks.0.attn.qkv_proj",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.gate_up_proj",
            "transformer_blocks.1.attn.to_out.0",
            "transformer_blocks.1.ff.gate_up_proj",
            "transformer_blocks.1.ff.down_proj",
        )
        bf16 = (
            "proj_in",
            "audio_proj_in",
            "context_embedder",
            "norm_out.linear",
            "proj_out",
            "audio_proj_out",
            "transformer_blocks.0.adaln_proj.linear",
            "transformer_blocks.1.adaln_proj.linear",
        )
        for name in fp8:
            module = _linear(model, name)
            assert type(module.quant_method) is FP8QDQLinearMethod, name
            assert module.weight.dtype == torch.float8_e4m3fn, name
            assert module.has_fp8_qdq and not module.has_nvfp4
        for name in nvfp4:
            module = _linear(model, name)
            assert type(module.quant_method) is NVFP4LinearMethod, name
            assert module.weight.dtype == torch.uint8, name
            assert module.weight_scale_2 is not None
        for name in bf16:
            module = _linear(model, name)
            assert type(module.quant_method) is UnquantizedLinearMethod, name
            assert not module.has_any_quant, name
        # The census counts TRT-LLM Linears (a fused QKV is one), logged once at build.
        assert model.apply_layer_quant_configs() == {"NVFP4": 6, "FP8": 2, "BF16": 8}

    @pytest.mark.parametrize(
        "ulysses, async_ulysses, reason",
        [(2, False, "Ulysses 1 and 4"), (8, False, "Ulysses 1 and 4"), (4, True, "async_ulysses")],
    )
    def test_other_ulysses_configurations_are_refused(self, ulysses, async_ulysses, reason):
        model = _build(_mixed_precision_export(set(_FP8_LAYERS)))
        model.model_config.visual_gen_mapping = SimpleNamespace(ulysses_size=ulysses)
        model.model_config.parallel.async_ulysses = async_ulysses
        with pytest.raises(NotImplementedError, match=reason):
            model.apply_layer_quant_configs()

    def test_ulysses_4_keeps_the_census(self):
        model = _build(_mixed_precision_export(set(_FP8_LAYERS)))
        model.model_config.visual_gen_mapping = SimpleNamespace(ulysses_size=4)
        assert model.apply_layer_quant_configs() == {"NVFP4": 6, "FP8": 2, "BF16": 8}

    @pytest.mark.skipif(
        not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (10, 7),
        reason="the scenario's kernels are SM 107 only",
    )
    def test_scenario_producers_skip_fp8_linears(self):
        """Only a static-NVFP4 consumer takes a fused NVFP4 producer in the scenario block."""
        # The scenario's modules build SM 107 kernels at import.
        static_nvfp4_input = importlib.import_module(
            f"{_SCENARIO_PACKAGE}.kernels.fused_out_quant"
        ).static_nvfp4_input
        ScenarioBlock = importlib.import_module(f"{_SCENARIO_PACKAGE}.modeling").ScenarioBlock

        model = _build(_mixed_precision_export(_FP8_LAYERS))
        block0, block1 = model.transformer_blocks
        # The family attention built here stands in for a scenario attention on the FP8 path.
        for block in (block0, block1):
            block.attn.on_fp8_path = lambda: True
        # block 0: NVFP4 qkv and gate/up keep their producers, the FP8 down projection does not.
        assert ScenarioBlock._nvfp4_producers(block0) == (True, True, False)
        # block 1: FP8 QKV gets the BF16 norm output, the NVFP4 MLP keeps both producers.
        assert ScenarioBlock._nvfp4_producers(block1) == (False, True, True)
        # An attention on BF16 operands takes the BF16 norm output even over an NVFP4 QKV.
        block0.attn.on_fp8_path = lambda: False
        assert ScenarioBlock._nvfp4_producers(block0) == (False, True, False)
        # The attention-output quantize reads the to_out Linear itself.

        assert static_nvfp4_input(block0.attn.to_out[0])
        assert static_nvfp4_input(block1.attn.to_out[0])

    def test_qkv_must_share_a_precision(self):
        with pytest.raises(ValueError, match="several precisions"):
            _build(_mixed_precision_export({"transformer_blocks.1.attn.to_q"}))

    def test_layer_in_no_group_raises(self):
        export = _mixed_precision_export(_FP8_LAYERS)
        for group in export["config_groups"].values():
            group["targets"] = [
                t for t in group["targets"] if t != "transformer_blocks.1.attn.to_out.0"
            ]
        del export["quantized_layers"]["transformer_blocks.1.attn.to_out.0"]
        with pytest.raises(ValueError, match="assigns no precision to Linear"):
            _build(export)

    def test_nvfp4_twin_with_ignore_list_excludes_fused_linears(self):
        """The NVFP4 export that recovers layers to BF16 through ``ignore`` (checkpoint names)."""
        quant_config = QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            group_size=16,
            exclude_modules=_IGNORE
            + [
                "transformer_blocks.0.attn.to_out*",
                "transformer_blocks.1.attn.to_k",
                "transformer_blocks.1.attn.to_q",
                "transformer_blocks.1.attn.to_v",
                "transformer_blocks.0.ff.net.0*",
                "transformer_blocks.1.ff.net.2",
            ],
        )
        model = h3.MiniMaxH3Transformer3DModel(_make_model_config(quant_config, None))
        for name in (
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.1.attn.qkv_proj",
            "transformer_blocks.0.ff.gate_up_proj",
            "transformer_blocks.1.ff.down_proj",
            "proj_in",
        ):
            module = _linear(model, name)
            assert not module.has_any_quant, name
            assert module.weight.dtype == torch.bfloat16 or name == "proj_in", name
        for name in ("transformer_blocks.0.attn.qkv_proj", "transformer_blocks.1.ff.gate_up_proj"):
            assert type(_linear(model, name).quant_method) is NVFP4LinearMethod, name
        assert model.apply_layer_quant_configs() == {"NVFP4": 4, "BF16": 12}

    def test_runtime_names_and_patterns_still_exclude(self):
        """The fused module's own name, and a pattern over either spelling, exclude it too."""
        quant_config = QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            group_size=16,
            exclude_modules=_IGNORE
            + ["transformer_blocks.0.attn.qkv_proj", "transformer_blocks.1.ff*"],
        )
        model = h3.MiniMaxH3Transformer3DModel(_make_model_config(quant_config, None))
        for name in (
            "transformer_blocks.0.attn.qkv_proj",
            "transformer_blocks.1.ff.gate_up_proj",
            "transformer_blocks.1.ff.down_proj",
        ):
            assert not _linear(model, name).has_any_quant, name
        assert model.apply_layer_quant_configs() == {"NVFP4": 5, "BF16": 11}

    def test_partial_exclusion_of_fused_linear_raises(self):
        quant_config = QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            group_size=16,
            exclude_modules=_IGNORE + ["transformer_blocks.1.attn.to_k"],
        )
        with pytest.raises(ValueError, match="covers only part of the fused Linear"):
            h3.MiniMaxH3Transformer3DModel(_make_model_config(quant_config, None))


# Every other VisualGen transformer: none of them reads a per-layer table.
_OTHER_TRANSFORMERS = [
    ("flux.transformer_flux", "FluxTransformer2DModel"),
    ("flux.transformer_flux2", "Flux2Transformer2DModel"),
    ("qwen_image.transformer_qwen_image", "QwenImageTransformer2DModel"),
    ("wan.transformer_wan", "WanTransformer3DModel"),
    ("ltx2.transformer_ltx2", "LTXModel"),
    ("hunyuan_video1_5.transformer_hunyuan_video1_5", "HunyuanVideo15Transformer3DModel"),
    ("glm_image.transformer_glm_image", "GlmImageTransformer2DModel"),
    ("cosmos3.transformer_cosmos3", "Cosmos3VFMTransformer"),
]


class TestPerLayerQuantSupport:
    @staticmethod
    def _refuse_linear(monkeypatch):
        def no_linear(*args, **kwargs):
            raise AssertionError("a Linear was built before the per-layer table was refused")

        monkeypatch.setattr(Linear, "__init__", no_linear)

    @pytest.mark.parametrize("module, name", _OTHER_TRANSFORMERS, ids=lambda v: v.split(".")[-1])
    def test_other_models_refuse_a_per_layer_table_before_allocation(
        self, monkeypatch, module, name
    ):
        cls = getattr(
            importlib.import_module(f"tensorrt_llm._torch.visual_gen.models.{module}"), name
        )
        assert issubclass(cls, BaseDiffusionModel)
        assert not cls._supports_per_layer_quant
        _, layers, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
            _mixed_precision_export(_FP8_LAYERS)
        )
        self._refuse_linear(monkeypatch)
        config = DiffusionModelConfig(
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
            quant_config_dict=layers,
        )
        with pytest.raises(ValueError, match=f"not implemented for {name}"):
            cls(model_config=config)

    def test_base_model_refuses_before_its_subclass_runs(self):
        class _Model(BaseDiffusionModel):
            def __init__(self, model_config):
                super().__init__(model_config)
                raise AssertionError("the subclass constructor ran")

        table = {"blocks.0.attn.to_q": QuantConfig(quant_algo=QuantAlgo.FP8)}
        with pytest.raises(ValueError, match="not implemented for _Model"):
            _Model(DiffusionModelConfig(quant_config_dict=table))

    @pytest.mark.parametrize("table", [None, {}])
    def test_base_model_accepts_no_table(self, table):
        class _Model(BaseDiffusionModel):
            pass

        _Model(DiffusionModelConfig(quant_config_dict=table))

    def test_minimax_h3_declares_support(self):
        assert h3.MiniMaxH3Transformer3DModel._supports_per_layer_quant
