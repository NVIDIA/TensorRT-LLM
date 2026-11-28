# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU component, cache geometry and checkpoint contracts; no engine registration."""

__extra_import_path__ = ["."]

import ast
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from _beam_component_support import checkpoint_config

from tensorrt_llm._torch.modeling_v2.models import beam
from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import layout_bf16 as layout
from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import loader_bf16 as loader

pytestmark = pytest.mark.cpu_only


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
    cfg = checkpoint_config()
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        layout.check_config(cfg)


def test_nested_partial_rope_is_rejected():
    cfg = checkpoint_config()
    cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 0.5
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


@pytest.mark.parametrize(
    "field", ["factor", "attention_factor", "original_max_position_embeddings", "unknown"]
)
def test_unknown_sliding_rope_options_are_rejected(field: str) -> None:
    cfg = checkpoint_config()
    cfg["rope_parameters"]["sliding_attention"][field] = 1.0
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


@pytest.mark.parametrize("field", ["factor", "global_attention", "unknown"])
def test_unknown_rope_groups_are_rejected(field: str) -> None:
    cfg = checkpoint_config()
    cfg["rope_parameters"][field] = None
    with pytest.raises(ValueError, match="rope_parameters"):
        layout.check_config(cfg)


def test_explicit_full_head_rope_is_supported() -> None:
    cfg = checkpoint_config()
    cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 1.0
    assert layout.check_config(cfg).num_layers == 52


def test_benign_config_metadata_and_supported_optional_values():
    cfg = dict(
        checkpoint_config(),
        partial_rotary_factor=1.0,
        use_sliding_window=True,
        transformers_version="test",
        _name_or_path="checkpoint",
    )
    assert layout.check_config(cfg).num_layers == 52


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
    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.modeling_v2.models.beam._load import load_failure_details

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
    from tensorrt_llm._torch.modeling_v2.models.beam._load import load_failure_details

    mapping = NS(world_size=1)
    assert load_failure_details(mapping, True, None) is None
    assert load_failure_details(mapping, False, None) == "rank 0: incomplete tensor coverage"


@pytest.mark.parametrize("field", ["layer_types", "mlp_layer_types"])
@pytest.mark.parametrize("value", [1, "dense", None, [[]]])
def test_malformed_layer_schedule_is_value_error(field, value):
    cfg = checkpoint_config()
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        layout.check_config(cfg)


def test_direct_call_device_and_metadata_admission(monkeypatch):
    import torch

    from tensorrt_llm._torch.modeling_v2.models.beam._validation import (
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


@pytest.mark.parametrize(
    "ids,error",
    [
        ([[1, 0], [0, 2]], None),
        ([[-1, 0]], "nonnegative"),
        ([[1, 1]], "distinct within each token"),
    ],
)
def test_dispatch_routing_preconditions(ids, error):
    import torch

    from tensorrt_llm._torch.modeling_v2.models.beam.ops.moe.beam_dispatch_combine import (
        beam_dispatch_combine as op,
    )

    ids = torch.tensor(ids, dtype=torch.int64)
    hidden = torch.ones(ids.shape[0], 4, dtype=torch.bfloat16)
    weights = torch.full(ids.shape, 0.5, dtype=torch.float32)

    def provider(expert, rows):
        return rows * (expert + 1)

    args = (hidden, ids, weights, provider, 2.5)
    if error is not None:
        with pytest.raises(AssertionError, match=error):
            op.is_valid(*args)
    else:
        op.is_valid(*args)
        assert torch.equal(op(*args), op.reference(*args))


@pytest.mark.parametrize(
    "attention_window,max_seq_len,retained,accepted",
    [
        (512, 8192, 511, False),
        (512, 8192, 512, True),
        (512, 8192, 1024, True),
        (512, 128, 128, True),
        (512, 128, 127, False),
        (512, 8192, None, True),
        (None, 8192, 512, False),
        (None, 8192, 8192, True),
        (None, 8192, None, True),
    ],
)
def test_standalone_block_cache_retention(
    monkeypatch, attention_window, max_seq_len, retained, accepted
):
    import torch

    from tensorrt_llm._torch.modeling_v2.models.beam import _attention

    # Metadata geometry is checked separately; use a non-identity logical-to-local mapping.
    monkeypatch.setattr(_attention, "check_metadata", lambda *args, **kwargs: 0)
    md = NS(
        kv_cache_manager=NS(
            layer_offsets={5: 1}, max_attention_window_vec=[0, retained], max_seq_len=max_seq_len
        )
    )
    args = (torch.ones(1, 2, 4, dtype=torch.bfloat16), torch.arange(2).reshape(1, 2), md)
    kwargs = dict(
        layer_idx=5,
        hidden_size=4,
        num_kv_heads=2,
        head_dim=128,
        device=torch.device("cpu"),
        max_tokens=17,
        attention_window=attention_window,
    )
    if accepted:
        _attention.check_step(*args, **kwargs)
    else:
        with pytest.raises(ValueError, match="requires cache retention"):
            _attention.check_step(*args, **kwargs)
