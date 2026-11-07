# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU component, cache geometry and checkpoint contracts; no engine registration."""

import ast
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models import beam
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    layout_bf16 as layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    loader_bf16 as loader,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (
    BeamFullModelNvfp4Sm103Tp4 as Component,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only


def config():
    return dict(
        layout.SUPPORTED_CONFIG,
        layer_types=["sliding_attention"] * 51 + ["full_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 49,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
    )


@pytest.mark.parametrize("length", [128, 512, 8192])
@pytest.mark.parametrize("retention", ["sliding", "full", "normalized"])
def test_cache_allows_estimator_and_short_sequence_geometry(length, retention):
    component = NS(
        head_dim=128, local_kv_heads=2, window=512, schedule=layout.check_config(config())
    )
    window = {"sliding": 512, "full": length, "normalized": None}[retention]
    manager = NS(
        layer_offsets=dict(enumerate(range(52))),
        dtype=DataType.BF16,
        head_dim=128,
        num_kv_heads_per_layer=[2] * 52,
        max_attention_window_vec=[window] * 51 + [None],
        max_seq_len=length,
    )
    Component.check_cache_manager(component, manager)
    # Cache retention does not change the mathematical attention window.
    assert Component.attention_windows(component, length) == [512] * 51 + [length]


@pytest.mark.parametrize("layer,window", [(0, 256), (0, 1024), (51, 512)])
def test_cache_rejects_incompatible_retention(layer, window):
    component = NS(
        head_dim=128, local_kv_heads=2, window=512, schedule=layout.check_config(config())
    )
    manager = NS(
        layer_offsets=dict(enumerate(range(52))),
        dtype=DataType.BF16,
        head_dim=128,
        num_kv_heads_per_layer=[2] * 52,
        max_attention_window_vec=[512] * 51 + [None],
        max_seq_len=8192,
    )
    manager.max_attention_window_vec[layer] = window
    with pytest.raises(ValueError, match="cache keeps a window"):
        Component.check_cache_manager(component, manager)


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError, TypeError])
def test_prefetch_failure_is_raised_in_consumer(tmp_path, monkeypatch, error_type):
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(range(8)))
    entry = loader.Entry("weight", str(path), layout.F32, (2, 1), 0, 8)
    reader = loader.TileReader(tile_bytes=4, pin=False)
    original = reader._read
    failure = error_type("injected background read failure")

    def read(buf, fd, offset, length):
        if offset == 4:
            raise failure
        original(buf, fd, offset, length)

    monkeypatch.setattr(reader, "_read", read)
    tiles = reader.tiles(entry, 0, 2)
    try:
        assert next(tiles)[:2] == (0, 1)
        with pytest.raises(error_type) as caught:
            next(tiles)
        assert caught.value is failure
    finally:
        tiles.close()
        reader.close()


@pytest.mark.parametrize(
    "field,value", [("partial_rotary_factor", 0.5), ("use_sliding_window", False)]
)
def test_math_changing_config_is_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        layout.check_config(cfg)


def test_nested_partial_rope_is_rejected():
    cfg = config()
    cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 0.5
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


def test_benign_config_metadata_and_supported_optional_values():
    cfg = dict(
        config(),
        partial_rotary_factor=1.0,
        use_sliding_window=True,
        transformers_version="test",
        _name_or_path="checkpoint",
    )
    assert layout.check_config(cfg).num_layers == 52


@pytest.fixture
def recipe_specs():
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    expert = "model.layers.3.mlp.experts.0.gate_proj"
    return {
        expert: nvfp4_layout.QuantSpec(expert, (2560, 5120), 3, 0, "gate_proj", 0),
        "model.layers.3.self_attn.q_proj.weight": NS(shape=(6144, 5120)),
        "lm_head.weight": NS(shape=(151936, 5120)),
    }


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    "attention_pattern",
    ["model.layers.*.self_attn.*", "model.layers.3.self_attn", "re:.*self_attn"],
)
def test_shared_quantization_shapes_and_exclusions(recipe_specs, flat, attention_pattern):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    values = dict(quant_algo="NVFP4", group_size=16, exclude_modules=[attention_pattern, "lm_head"])
    recipe = {"quantization": values}
    if flat:
        recipe = dict(values, quant_method="modelopt", ignore=values["exclude_modules"])
        del recipe["exclude_modules"]
    summary = nvfp4_layout.check_recipe(recipe, recipe_specs)
    assert summary["quantized_modules"] == 1
    assert summary["excluded_linear_modules"] == 2


def test_parent_exclusion_cannot_hide_quantized_experts(recipe_specs):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    recipe = {
        "quantization": dict(
            quant_algo="NVFP4", group_size=16, exclude_modules=["model.layers.3", "lm_head"]
        )
    }
    with pytest.raises(ValueError, match="excludes quantized experts"):
        nvfp4_layout.check_recipe(recipe, recipe_specs)


def test_quantization_file_precedes_inline_recipe(tmp_path):
    import json

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    inline = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    config = {"quantization_config": inline}
    assert nvfp4_layout.read_quant_config(tmp_path, config) == inline
    file_recipe = {"quantization": {"quant_algo": "FP8"}}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(file_recipe))
    assert nvfp4_layout.read_quant_config(tmp_path, config) == file_recipe


@pytest.mark.parametrize("scheme", ["FP8", "NVFP4", "unknown", {"type": "float", "num_bits": 8}])
def test_flat_recipe_never_silently_disables_requested_kv_quantization(recipe_specs, scheme):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    with pytest.raises(ValueError, match="BF16 KV cache"):
        nvfp4_layout.check_recipe(
            dict(
                quant_method="modelopt", quant_algo="NVFP4", group_size=16, kv_cache_scheme=scheme
            ),
            recipe_specs,
        )


@pytest.mark.parametrize("in_file", [False, True])
def test_checkpoint_validation_uses_shared_recipe_reader(
    tmp_path, monkeypatch, recipe_specs, in_file
):
    import json

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )

    recipe = dict(
        quant_method="modelopt",
        quant_algo="NVFP4",
        group_size=16,
        ignore=["model.layers.3.self_attn", "lm_head"],
    )
    cfg = config()
    if in_file:
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(recipe))
    else:
        cfg["quantization_config"] = recipe
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(nvfp4_layout, "tensor_specs", lambda config: recipe_specs)
    assert nvfp4_layout.check_checkpoint(tmp_path) == cfg


def test_components_do_not_import_full_model_internals():
    family = Path(next(iter(beam.__path__)))
    target = family / "full_model_nvfp4__sm_103__tp4"
    target_module = f"{beam.__name__}.{target.name}"
    for source in family.rglob("*.py"):
        if source.is_relative_to(target):
            continue
        package = ".".join((beam.__name__, *source.parent.relative_to(family).parts))
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = resolve_name("." * node.level + (node.module or ""), package)
                modules = [f"{module}.{alias.name}" for alias in node.names]
            else:
                continue
            for module in modules:
                assert module != target_module and not module.startswith(target_module + "."), (
                    f"{source.relative_to(family)} imports target-private code: {module}"
                )
