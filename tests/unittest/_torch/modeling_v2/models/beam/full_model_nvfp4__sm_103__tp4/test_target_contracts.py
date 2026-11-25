# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the routed Beam NVFP4 target and its checkpoint loader."""

__extra_import_path__ = [".."]

import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _beam_component_support import checkpoint_config

from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import component_bf16 as bf16
from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import layout_bf16 as layout
from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import loader_bf16 as loader
from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    component as nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    layout as nvfp4_layout,
)
from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (
    BeamFullModelNvfp4Sm103Tp4 as Component,
)
from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (
    ModelingV2BeamFullModelNvfp4Sm103Tp4 as Target,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "ids,end_id,lora,valid",
    [
        ([0, 9], None, None, True),
        ([9], -1, None, True),
        ([0], 9, None, True),
        ([-1], None, None, False),
        ([10], None, None, False),
        ([0], -2, None, False),
        ([0], 10, None, False),
        ([0], None, 0, False),
    ],
)
def test_request_rejected_before_batch_execution(ids, end_id, lora, valid):
    target = SimpleNamespace(model=SimpleNamespace(vocab_size=10))
    request = SimpleNamespace(
        py_end_id=end_id,
        lora_task_id=lora,
        check_token_id_range=lambda n: all(0 <= token < n for token in ids),
    )
    if valid:
        Target.validate_request(target, request)
    else:
        with pytest.raises(ValueError):
            Target.validate_request(target, request)


def test_default_weight_loader_is_refused_explicitly():
    with pytest.raises(ValueError, match="BeamNvfp4CheckpointLoader"):
        Target.load_weights(SimpleNamespace(), {})


@pytest.mark.parametrize("diagnostics", [False, True])
def test_checkpoint_digests_follow_diagnostic_mode(diagnostics, tmp_path) -> None:
    checkpoint = str(tmp_path)
    record = {"loaded": True}
    load = Mock(return_value=record)
    target = SimpleNamespace(
        checkpoint_dir=checkpoint,
        model=SimpleNamespace(diagnostics=diagnostics, load=load),
        load_record=None,
        kv_cache_quant_algo=None,
    )
    target._check_kv_cache_scales = Target._check_kv_cache_scales.__get__(target)
    target._apply_kv_cache_scales = Target._apply_kv_cache_scales.__get__(target)
    assert Target.load_checkpoint(target, checkpoint) is record
    load.assert_called_once_with(checkpoint, record_digests=diagnostics)
    assert target.load_record is record


@pytest.mark.parametrize("length", [128, 512, 8192])
@pytest.mark.parametrize("retention", ["sliding", "full", "normalized", "extended"])
def test_cache_allows_estimator_and_short_sequence_geometry(length, retention):
    component = SimpleNamespace(
        kv_cache_dtype=DataType.BF16,
        head_dim=128,
        local_kv_heads=2,
        window=512,
        schedule=layout.check_config(checkpoint_config()),
    )
    window = {"sliding": 512, "full": length, "normalized": None, "extended": 1024}[retention]
    manager = SimpleNamespace(
        layer_offsets=dict(enumerate(range(52))),
        dtype=DataType.BF16,
        head_dim=128,
        num_kv_heads_per_layer=[2] * 52,
        max_attention_window_vec=[window] * 51 + [None],
        max_seq_len=length,
    )
    Component.check_cache_manager(component, manager)


@pytest.mark.parametrize(
    "length,layer,window", [(8192, 0, 256), (8192, 0, 511), (128, 0, 127), (8192, 51, 512)]
)
def test_cache_rejects_incompatible_retention(length, layer, window):
    component = SimpleNamespace(
        kv_cache_dtype=DataType.BF16,
        head_dim=128,
        local_kv_heads=2,
        window=512,
        schedule=layout.check_config(checkpoint_config()),
    )
    manager = SimpleNamespace(
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


@pytest.fixture
def recipe_specs():
    expert = "model.layers.3.mlp.experts.0.gate_proj"
    return {
        expert: nvfp4_layout.QuantSpec(expert, (2560, 5120), 3, 0, "gate_proj", 0),
        "model.layers.3.self_attn.q_proj.weight": SimpleNamespace(shape=(6144, 5120)),
        "lm_head.weight": SimpleNamespace(shape=(151936, 5120)),
    }


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    "attention_pattern",
    ["model.layers.*.self_attn.*", "model.layers.3.self_attn", "re:.*self_attn"],
)
def test_shared_quantization_shapes_and_exclusions(recipe_specs, flat, attention_pattern):
    values = dict(quant_algo="NVFP4", group_size=16, exclude_modules=[attention_pattern, "lm_head"])
    recipe = {"quantization": values}
    if flat:
        recipe = dict(values, quant_method="modelopt", ignore=values["exclude_modules"])
        del recipe["exclude_modules"]
    summary = nvfp4_layout.check_recipe(recipe, recipe_specs)
    assert summary["quantized_modules"] == 1
    assert summary["excluded_linear_modules"] == 2


def test_parent_exclusion_cannot_hide_quantized_experts(recipe_specs):
    recipe = {
        "quantization": dict(
            quant_algo="NVFP4", group_size=16, exclude_modules=["model.layers.3", "lm_head"]
        )
    }
    with pytest.raises(ValueError, match="excludes quantized experts"):
        nvfp4_layout.check_recipe(recipe, recipe_specs)


def test_quantization_file_precedes_inline_recipe(tmp_path):
    inline = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    config = {"quantization_config": inline}
    assert nvfp4_layout.read_quant_config(tmp_path, config) == inline
    file_recipe = {"quantization": {"quant_algo": "FP8"}}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(file_recipe))
    assert nvfp4_layout.read_quant_config(tmp_path, config) == file_recipe


@pytest.mark.parametrize("scheme", ["FP8", "NVFP4", "unknown", {"type": "float", "num_bits": 8}])
def test_flat_recipe_never_silently_disables_requested_kv_quantization(recipe_specs, scheme):
    with pytest.raises(ValueError, match="kv_cache_scheme"):
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
    recipe = dict(
        quant_method="modelopt",
        quant_algo="NVFP4",
        group_size=16,
        ignore=["model.layers.3.self_attn", "lm_head"],
    )
    cfg = checkpoint_config()
    if in_file:
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(recipe))
    else:
        cfg["quantization_config"] = recipe
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(nvfp4_layout, "tensor_specs", lambda config: recipe_specs)
    assert nvfp4_layout.check_checkpoint(tmp_path) == cfg


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("failure", [None, "metadata", "copy", "close", "remote"])
def test_checkpoint_load_transaction(
    tmp_path, monkeypatch, packed: bool, failure: str | None
) -> None:
    """Real CPU tensor reads preserve load atomicity and propagate every failure phase."""

    from safetensors.torch import save_file

    cls = (
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4 if packed else bf16.BeamCheckpointLayoutBf16Sm103Tp4
    )
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    # CPU copies are synchronous; omit only the CUDA stream synchronization.
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None)
    )
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    spec = layout.TensorSpec("weight", (4, 2), layout.F32, "dense", None, None, None)
    obj.config = {"test": True}
    obj.specs = {spec.name: spec}
    obj.mapping = SimpleNamespace(world_size=1, rank=0)
    obj.rank, obj.layers, obj.device = 0, (), torch.device("cpu")
    obj.schedule = SimpleNamespace(is_sparse=lambda layer: False)
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
    from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.loader import (
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
    cls = (
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4 if packed else bf16.BeamCheckpointLayoutBf16Sm103Tp4
    )
    spec = layout.TensorSpec(layout.EMBED, (8, 2), layout.BF16, "embedding", None, None, 0)
    monkeypatch.setattr(cls, "_tensor_specs", staticmethod(lambda *args: {spec.name: spec}))
    monkeypatch.setattr(
        bf16, "allreduce_group_args", lambda mapping: dict(group=[0, 1, 2, 3], process_group=None)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    real_empty = torch.empty

    def empty_on_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_on_cpu)
    mapping = SimpleNamespace(
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
    obj = cls(checkpoint_config(), mapping, layers=[], device=torch.device("cuda:0"), max_tokens=17)
    assert obj.tensor(layout.EMBED).shape == (2, 2)
    assert obj.vocab_interval == (0, 2) and obj.max_tokens == 17
    assert not obj.usable and obj.load_record is None
    assert obj.local_heads == 12 and obj.local_kv_heads == 2
    if packed:
        assert obj.expert_names == [] and obj._shared_fc1 == {}


@pytest.fixture(params=[bf16, nvfp4], ids=["bf16", "nvfp4"])
def layout_loader(request, monkeypatch, tmp_path):
    module = request.param
    cls = (
        bf16.BeamCheckpointLayoutBf16Sm103Tp4
        if module is bf16
        else nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "hf_quant_config.json").write_text("{}")
    spec = SimpleNamespace(
        name="weight", bank=None, expert=None, interval=lambda rank: (0, 2), local_shape=(2,)
    )
    dst = torch.zeros(2, dtype=torch.bfloat16)
    component = cls.__new__(cls)
    torch.nn.Module.__init__(component)
    state = dict(
        config={},
        rank=0,
        layers=(),
        expert_names=(),
        specs=OrderedDict(weight=spec),
        usable=True,
        load_record={"old": True},
        tensor=lambda name: dst,
    )
    for key, value in state.items():
        setattr(component, key, value)
    component.mapping = SimpleNamespace(world_size=1, rank=0)
    component.schedule = SimpleNamespace(is_sparse=lambda layer: False)
    agreements = []

    def agree(ok):
        agreements.append(ok)
        return ok

    component._agree = agree
    monkeypatch.setattr(
        bf16,
        "load_failure_details",
        lambda mapping, ok, error: None if component._agree(ok) else str(error),
    )
    component._check_diagnostics_agree = lambda: None
    catalog = SimpleNamespace(
        entries={"weight": SimpleNamespace(path="part", begin=0)},
        validate=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(bf16, "SafetensorsCatalog", lambda path: catalog)
    monkeypatch.setattr(cls, "_tensor_specs", staticmethod(lambda config: {}))
    monkeypatch.setattr(module, "tensor_specs", lambda config: {})
    if module is nvfp4:
        monkeypatch.setattr(module, "check_recipe", lambda *args: {})
        monkeypatch.setattr(module, "stored_entries", lambda *args: {})
        monkeypatch.setattr(module, "validate", lambda *args, **kwargs: None)
    reader = SimpleNamespace(bytes_read=4, reads=1, closed=False)

    def close():
        reader.closed = True

    reader.close = close
    monkeypatch.setattr(bf16, "TileReader", lambda: reader)
    monkeypatch.setattr(bf16, "read_slice_into", lambda *args, **kwargs: dst.fill_(1))
    return bf16, cls.load, component, agreements, catalog, reader, tmp_path


@pytest.mark.parametrize("stored", [None, ["bad"]])
def test_malformed_config_reaches_validation_agreement(layout_loader, stored):
    _, load, component, agreements, _, _, path = layout_loader
    (path / "config.json").write_text(json.dumps(stored))
    with pytest.raises(bf16.LoadError) as exc:
        load(component, str(path))
    assert isinstance(exc.value.__cause__, (TypeError, AttributeError))
    assert agreements == [False]
    assert component.usable  # Validation rejected the reload before touching storage.
    assert component.load_record == {"old": True}


@pytest.mark.parametrize("stage", ["catalog", "reader", "ordering", "read", "close"])
def test_rank_local_exception_reaches_agreement(layout_loader, monkeypatch, stage):
    module, load, component, agreements, catalog, reader, path = layout_loader
    failure = TypeError("injected rank-local failure")

    def fail(*args, **kwargs):
        raise failure

    if stage == "catalog":
        monkeypatch.setattr(module, "SafetensorsCatalog", fail)
    elif stage == "reader":
        monkeypatch.setattr(module, "TileReader", fail)
    elif stage == "ordering":
        catalog.entries.clear()
    elif stage == "read":
        monkeypatch.setattr(module, "read_slice_into", fail)
    else:
        reader.close = fail
    with pytest.raises(bf16.LoadError) as exc:
        load(component, str(path))
    assert agreements == ([False] if stage == "catalog" else [True, False])
    if stage == "ordering":
        assert isinstance(exc.value.__cause__, KeyError)
    else:
        assert exc.value.__cause__ is failure
    if stage != "catalog":
        assert not component.usable and component.load_record is None
    if stage in ("ordering", "read"):
        assert reader.closed


def test_successful_load_and_peer_failure(layout_loader):
    _, load, component, agreements, _, reader, path = layout_loader
    record = load(component, str(path))
    assert agreements == [True, True]
    assert component.usable and record["tensors"] == 1
    assert reader.closed and torch.equal(component.tensor("weight"), torch.ones(2))
    agreements.clear()

    def peer_fails_completion(ok):
        agreements.append(ok)
        return len(agreements) == 1

    component._agree = peer_fails_completion
    with pytest.raises(bf16.LoadError, match="every rank"):
        load(component, str(path))
    assert agreements == [True, True]
    assert not component.usable and component.load_record is None
