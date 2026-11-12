# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU component, cache geometry and checkpoint contracts; no engine registration."""

import ast
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models import beam
from tensorrt_llm._torch._experimental.modeling_v2.models.beam._tp4 import layout_bf16 as layout
from tensorrt_llm._torch._experimental.modeling_v2.models.beam._tp4 import loader_bf16 as loader
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
@pytest.mark.parametrize("retention", ["sliding", "full", "normalized", "extended"])
def test_cache_allows_estimator_and_short_sequence_geometry(length, retention):
    component = NS(
        head_dim=128, local_kv_heads=2, window=512, schedule=layout.check_config(config())
    )
    window = {"sliding": 512, "full": length, "normalized": None, "extended": 1024}[retention]
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


@pytest.mark.parametrize(
    "length,layer,window", [(8192, 0, 256), (8192, 0, 511), (128, 0, 127), (8192, 51, 512)]
)
def test_cache_rejects_incompatible_retention(length, layer, window):
    component = NS(
        head_dim=128, local_kv_heads=2, window=512, schedule=layout.check_config(config())
    )
    manager = NS(
        layer_offsets=dict(enumerate(range(52))),
        dtype=DataType.BF16,
        head_dim=128,
        num_kv_heads_per_layer=[2] * 52,
        max_attention_window_vec=[512] * 51 + [None],
        max_seq_len=length,
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


@pytest.mark.parametrize(
    "field", ["factor", "attention_factor", "original_max_position_embeddings", "unknown"]
)
def test_unknown_sliding_rope_options_are_rejected(field: str) -> None:
    cfg = config()
    cfg["rope_parameters"]["sliding_attention"][field] = 1.0
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


@pytest.mark.parametrize("field", ["factor", "global_attention", "unknown"])
def test_unknown_rope_groups_are_rejected(field: str) -> None:
    cfg = config()
    cfg["rope_parameters"][field] = None
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


def test_explicit_full_head_rope_is_supported() -> None:
    cfg = config()
    cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 1.0
    assert layout.check_config(cfg).num_layers == 52


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
        # The family router selects and imports targets; component helpers must not.
        if source.is_relative_to(target) or source == family / "routing.py":
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


@pytest.mark.parametrize("local_rank", range(4))
def test_load_failures_identify_remote_rank_on_every_rank(monkeypatch, local_rank):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam._load import load_failure_details
    from tensorrt_llm._torch.distributed.communicator import Distributed

    outcomes = [
        (True, ""),
        (False, "ValueError('bad block scale')"),
        (True, ""),
        (False, "short read"),
    ]
    monkeypatch.setattr(Distributed, "get", lambda mapping: NS(allgather=lambda outcome: outcomes))
    error = ValueError("bad block scale") if local_rank == 1 else None
    detail = load_failure_details(
        NS(world_size=4, rank=local_rank), local_rank not in (1, 3), error
    )
    assert detail == "rank 1: ValueError('bad block scale'); rank 3: short read"


def test_local_load_success_and_incomplete_coverage():
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam._load import load_failure_details

    mapping = NS(world_size=1)
    assert load_failure_details(mapping, True, None) is None
    assert load_failure_details(mapping, False, None) == "rank 0: incomplete tensor coverage"


@pytest.mark.parametrize("field", ["layer_types", "mlp_layer_types"])
@pytest.mark.parametrize("value", [1, "dense", None, [[]]])
def test_malformed_layer_schedule_is_value_error(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        layout.check_config(cfg)


def test_direct_call_device_and_metadata_admission(monkeypatch):
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam._validation import (
        check_positions,
        explicit_device,
        metadata_values,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert explicit_device("cuda") == torch.device("cuda:2")
    assert explicit_device(None) == torch.device("cuda:2")
    assert explicit_device("cuda:1") == torch.device("cuda:1")
    for dtype in (torch.int32, torch.int64):
        check_positions(torch.arange(3, dtype=dtype), (3,), torch.device("cpu"))
    for dtype in (torch.uint8, torch.int16, torch.bool, torch.float32):
        with pytest.raises(ValueError, match="int32/int64"):
            check_positions(torch.ones(3, dtype=dtype), (3,), torch.device("cpu"))
    with pytest.raises(ValueError, match="shape"):
        check_positions(torch.ones(1, dtype=torch.int64), (3,), torch.device("cpu"))
    for values in ([2, 3], torch.tensor([2, 3])):
        assert metadata_values(values, 2, "seq_lens") == [2, 3]
        with pytest.raises(ValueError, match="seq_lens needs 3"):
            metadata_values(values, 3, "seq_lens")


@pytest.mark.parametrize("offset", [0.0, False, "0"])
def test_safetensors_offsets_must_be_integers(tmp_path, monkeypatch, offset):
    import json

    (tmp_path / loader.INDEX_FILE).write_text(json.dumps({"weight_map": {"w": "shard"}}))
    (tmp_path / "shard").write_bytes(bytes(8))
    monkeypatch.setattr(
        loader,
        "_read_header",
        lambda path: (
            {"w": {"dtype": "F32", "shape": [1], "data_offsets": [offset, 4]}},
            0,
        ),
    )
    with pytest.raises(loader.CheckpointFormatError, match="offsets must be integers"):
        loader.SafetensorsCatalog(str(tmp_path))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("failure", [None, "metadata", "copy", "close", "remote"])
def test_checkpoint_load_transaction(
    tmp_path, monkeypatch, packed: bool, failure: str | None
) -> None:
    """Real CPU tensor reads preserve load atomicity and propagate every failure phase."""
    import json

    import torch
    from safetensors.torch import save_file

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam._tp4 import (
        component_bf16 as bf16,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        component as nvfp4,
    )

    cls = (
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4 if packed else bf16.BeamCheckpointLayoutBf16Sm103Tp4
    )
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    # CPU copies are synchronous; omit only the CUDA stream synchronization.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: NS(synchronize=lambda: None))
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    spec = layout.TensorSpec("weight", (4, 2), layout.F32, "dense", None, None, None)
    obj.config = {"test": True}
    obj.specs = {spec.name: spec}
    obj.mapping = NS(world_size=1, rank=0)
    obj.rank, obj.layers, obj.device = 0, (), torch.device("cpu")
    obj.schedule = NS(is_sparse=lambda layer: False)
    obj.expert_names, obj._shared_fc1 = [], {3: [True]}
    obj.usable, obj.load_record = True, {"previous": True}
    obj.register_parameter(
        bf16._attr(spec.name), torch.nn.Parameter(torch.zeros_like(source), requires_grad=False)
    )
    obj._check_diagnostics_agree = lambda: None
    monkeypatch.setattr(cls, "_tensor_specs", staticmethod(lambda *args: obj.specs))
    if packed:
        monkeypatch.setattr(nvfp4, "tensor_specs", lambda *args: obj.specs)
        monkeypatch.setattr(nvfp4, "read_quant_config", lambda *args: {})
        monkeypatch.setattr(nvfp4, "check_recipe", lambda *args: {"verified": True})
    (tmp_path / "config.json").write_text(json.dumps(obj.config))
    save_file({"weight": source}, str(tmp_path / "weights.safetensors"))
    (tmp_path / loader.INDEX_FILE).write_text(
        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    )
    if failure == "metadata":
        (tmp_path / "config.json").write_text("{}")
    reader = loader.TileReader(tile_bytes=16, pin=False)
    monkeypatch.setattr(bf16, "TileReader", lambda: reader)
    closed = []
    real_close = reader.close

    def close() -> None:
        real_close()
        closed.append(True)
        if failure == "close":
            raise OSError("close failed")

    monkeypatch.setattr(reader, "close", close)
    if failure == "copy":

        def copy_failed(*args, **kwargs) -> None:
            raise OSError("copy failed")

        monkeypatch.setattr(bf16, "read_slice_into", copy_failed)
    phases = []

    def agree(mapping, ok, error):
        phases.append(ok)
        if failure == "remote" and len(phases) == 2:
            return "rank 2: remote copy failed"
        return None if ok else str(error)

    monkeypatch.setattr(bf16, "load_failure_details", agree)
    try:
        if failure is not None:
            with pytest.raises(
                bf16.LoadError,
                match={
                    "metadata": "checkpoint rejected before loading",
                    "copy": "copy failed",
                    "close": "close failed",
                    "remote": "rank 2",
                }[failure],
            ):
                obj.load(str(tmp_path), record_digests=False)
            if failure == "metadata":
                assert obj.usable and obj.load_record == {"previous": True}
                assert torch.count_nonzero(obj.tensor("weight")) == 0
                assert phases == [False] and not closed
            else:
                assert not obj.usable and obj.load_record is None
                assert len(phases) == 2 and closed == [True]
                if packed:
                    assert obj._shared_fc1 == {}
        else:
            record = obj.load(str(tmp_path), record_digests=False)
            assert torch.equal(obj.tensor("weight"), source)
            assert obj.usable and record is obj.load_record
            assert record["tensors"] == 1 and record["bytes_read"] == 32
            assert record["coverage"]["weight"]["source_sha256"] is None
            assert phases == [True, True] and closed == [True]
            if packed:
                assert record["recipe"] == {"verified": True} and record["scalars"] == {}
    finally:
        real_close()


@pytest.mark.parametrize("dtype,itemsize", [("BF16", 2), ("F32", 4), ("F8_E4M3", 1), ("U8", 1)])
def test_stored_entry_validation_is_shared(dtype: str, itemsize: int) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.loader import (
        validate,
    )

    catalog = loader.SafetensorsCatalog.__new__(loader.SafetensorsCatalog)
    catalog.entries = {"weight": loader.Entry("weight", "unused", dtype, (2, 4), 0, 8 * itemsize)}
    validate(catalog, {"weight": (dtype, (2, 4))})
    catalog.entries["weight"].end -= 1
    with pytest.raises(loader.CheckpointFormatError, match="byte span"):
        validate(catalog, {"weight": (dtype, (2, 4))})


@pytest.mark.parametrize("packed", [False, True])
def test_component_constructor_uses_format_storage(monkeypatch, packed: bool) -> None:
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam._tp4 import (
        component_bf16 as bf16,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        component as nvfp4,
    )

    cls = (
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4 if packed else bf16.BeamCheckpointLayoutBf16Sm103Tp4
    )
    spec = layout.TensorSpec(layout.EMBED, (8, 2), layout.BF16, "embedding", None, None, 0)
    monkeypatch.setattr(cls, "_tensor_specs", staticmethod(lambda *args: {spec.name: spec}))
    monkeypatch.setattr(bf16, "AllReduce", lambda **kwargs: NS())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    real_empty = torch.empty

    def empty_on_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_on_cpu)
    mapping = NS(
        rank=0,
        world_size=4,
        tp_size=4,
        pp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        cp_size=1,
        enable_attention_dp=False,
        tp_group=list(range(4)),
    )
    obj = cls(config(), mapping, layers=[], device=torch.device("cuda:0"), max_tokens=17)
    assert obj.tensor(layout.EMBED).shape == (2, 2)
    assert obj.vocab_interval == (0, 2) and obj.max_tokens == 17
    assert not obj.usable and obj.load_record is None
    assert obj.local_heads == 12 and obj.local_kv_heads == 2
    if packed:
        assert obj.expert_names == [] and obj._shared_fc1 == {}


def test_bringup_helpers_and_tensor_captures_are_not_runtime_apis():
    family = Path(next(iter(beam.__path__)))
    assert not list(family.glob("*_bf16__sm_103__tp1"))
    for source in family.rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
                assert not names.intersection({"capture", "layer_capture"}), source
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("_components"), source
