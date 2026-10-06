# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-to-payload coverage for lists of configuration models."""

import json
from typing import Any, Literal

import pytest
from pydantic import PositiveInt

from tensorrt_llm.llmapi.llm_args import Field
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.usage import llmapi_config as capture

pytestmark = pytest.mark.cpu_only


class _InputSpec(StrictBaseModel):
    name: str = "private_customer_input"
    dtype: str = "float32"
    shape: tuple[Literal["num_tokens", "batch_size"] | PositiveInt, ...]
    private_size: int = Field(default=12345, telemetry=False)


class _GraphConfig(StrictBaseModel):
    extra_model_inputs: list[_InputSpec] = Field(default_factory=list)


class _Args(StrictBaseModel):
    cuda_graph_config: _GraphConfig | None = None
    encoder_cuda_graph_config: _GraphConfig | None = None


def _collect(args: StrictBaseModel) -> tuple[dict, dict]:
    config, metadata = capture.collect_llm_api_config_payloads(args)
    return json.loads(config), json.loads(metadata)


@pytest.mark.parametrize("field", ["cuda_graph_config", "encoder_cuda_graph_config"])
def test_model_list_manifest_and_payload_agree(field: str) -> None:
    """The leaf policy includes the outer list, and raw/private fields stay absent."""
    path = f"{field}.extra_model_inputs.shape"
    specs = [_InputSpec(shape=("num_tokens",)), _InputSpec(shape=("batch_size", 20))]
    args = _Args(**{field: _GraphConfig(extra_model_inputs=specs)})

    rows = {row["path"]: row for row in capture.manifest_rows(_Args)}
    assert rows[path] == {
        "path": path,
        "kind": "categorical",
        "capture_policy": "list[tuple[int|literal]]",
        "allowed_values": ["num_tokens", "batch_size"],
    }
    config, metadata = _collect(args)
    assert config == {path: [["num_tokens"], ["batch_size", 20]]}
    assert metadata["capture_succeeded"] is True
    assert metadata["unsafe_excluded"] is False
    assert metadata["capturable_field_count"] == 2
    assert metadata["captured_field_count"] == 1


def test_empty_model_list_is_distinct_from_unset_parent() -> None:
    assert _collect(_Args())[0] == {}
    config, metadata = _collect(_Args(cuda_graph_config=_GraphConfig()))
    assert config == {"cuda_graph_config.extra_model_inputs.shape": []}
    assert metadata["unsafe_excluded"] is False


class _Leaf(StrictBaseModel):
    value: int | None = None


class _Group(StrictBaseModel):
    members: list[_Leaf]


def test_duplicate_model_route_fails_manifest_build(monkeypatch: pytest.MonkeyPatch) -> None:
    nested_model_routes = capture._nested_model_routes
    monkeypatch.setattr(capture, "_nested_model_routes", lambda ann: nested_model_routes(ann) * 2)

    with pytest.raises(ValueError, match="key 'members.value' has a duplicate model route"):
        capture.build_capture_manifest(_Group)


def test_nested_model_lists_and_nullable_leaves_preserve_positions() -> None:
    class _Groups(StrictBaseModel):
        groups: list[_Group]

    args = _Groups(groups=[_Group(members=[_Leaf(value=7), _Leaf()]), _Group(members=[])])
    assert _collect(args)[0] == {"groups.members.value": [[7, None], []]}
    assert capture.manifest_rows(_Groups)[0]["capture_policy"] == "list[list[int|none]]"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_optional_model_inside_list_does_not_fabricate_null_leaf(
    nested: bool, invalid: bool
) -> None:
    annotation = list[list[_Leaf | None]] if nested else list[_Leaf | None]

    class _OptionalItems(StrictBaseModel):
        items: annotation

    items = [None, _Leaf(value=1)]
    args = _OptionalItems(items=[items] if nested else items)
    if invalid:
        # A valid missing model must not hide a later unsafe value.
        items = args.items[0] if nested else args.items
        items[-1].value = "private"
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["capture_succeeded"] is True
    assert metadata["unsafe_excluded"] is invalid
    assert metadata["excluded_field_count"] == int(invalid)


def test_missing_optional_model_does_not_mark_projection_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OptionalItems(StrictBaseModel):
        items: list[_InputSpec | None]

    monkeypatch.setattr(capture, "MAX_SEQ_ITEMS", 1)
    args = _OptionalItems(items=[_InputSpec(shape=("num_tokens", 20)), None])
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["sequence_truncated"] is False
    assert metadata["unsafe_excluded"] is False
    assert metadata["excluded_field_count"] == 0


@pytest.mark.parametrize("container", ["tuple", "mixed_tuple", "set", "list_of_tuples"])
def test_unsupported_model_containers_do_not_enroll_or_capture(container: str) -> None:
    class _FrozenLeaf(StrictBaseModel):
        model_config = {"frozen": True}
        value: int

    first, second = _FrozenLeaf(value=1), _FrozenLeaf(value=987654)
    annotation, items = {
        "tuple": (tuple[_FrozenLeaf, ...], (first, second)),
        "mixed_tuple": (tuple[_FrozenLeaf, Any], (first, second)),
        "set": (set[_FrozenLeaf], {first, second}),
        "list_of_tuples": (list[tuple[_FrozenLeaf, ...]], [(first, second)]),
    }[container]

    class _Sequence(StrictBaseModel):
        items: annotation
        enabled: bool = True

    config, metadata = _collect(_Sequence(items=items))
    assert config == {"enabled": True}
    assert metadata["capture_succeeded"] is True
    assert [row["path"] for row in capture.manifest_rows(_Sequence)] == ["enabled"]


class _ModeA(StrictBaseModel):
    mode: Literal["a"] = "a"


class _ModeB(StrictBaseModel):
    mode: Literal["b"] = "b"


def test_model_union_uses_each_elements_policy_not_merged_domain() -> None:
    class _Modes(StrictBaseModel):
        items: list[_ModeA | _ModeB]

    args = _Modes(items=[_ModeA(), _ModeB()])
    assert _collect(args)[0] == {"items.mode": ["a", "b"]}
    assert capture.manifest_rows(_Modes)[0]["allowed_values"] == ["a", "b"]

    # Even another arm's allowed literal must not pass this arm's policy.
    args.items[0].mode = "b"
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["excluded_field_count"] == 1
    assert metadata["unsafe_excluded"] is True


class _Visible(StrictBaseModel):
    leaf: _Leaf


class _Hidden(StrictBaseModel):
    leaf: _Leaf = Field(telemetry=False)


@pytest.mark.parametrize("sequence", [False, True])
def test_excluded_parent_cannot_borrow_another_arms_leaf_policy(sequence: bool) -> None:
    annotation = list[_Visible | _Hidden] if sequence else _Visible | _Hidden

    class _Parents(StrictBaseModel):
        parent: annotation

    hidden = _Hidden(leaf=_Leaf(value=987654))
    parent = [_Visible(leaf=_Leaf(value=1)), hidden] if sequence else hidden
    config, metadata = _collect(_Parents(parent=parent))
    assert config == {}
    assert metadata["unsafe_excluded"] is sequence


class _PrivateLeaf(_Leaf):
    pass


@pytest.mark.parametrize("bad_element", [None, {}, "private", _ModeA(), _PrivateLeaf(value=987654)])
def test_unresolvable_element_omits_whole_projection(bad_element: object) -> None:
    args = _Group.model_construct(members=[_Leaf(value=1), bad_element, _Leaf(value=2)])
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["capture_succeeded"] is True
    assert metadata["excluded_field_count"] == 1
    assert metadata["unsafe_excluded"] is True


def test_scalar_and_sequence_model_union_preserves_each_wire_shape() -> None:
    class _Either(StrictBaseModel):
        item: _Leaf | list[_Leaf]

    assert capture.manifest_rows(_Either)[0]["capture_policy"] == "int|list[int|none]|none"
    assert _collect(_Either(item=_Leaf(value=3)))[0] == {"item.value": 3}
    assert _collect(_Either(item=[_Leaf(value=3)]))[0] == {"item.value": [3]}


@pytest.mark.parametrize("field", ["cuda_graph_config", "encoder_cuda_graph_config"])
def test_model_sequence_caps_outer_and_leaf_sequences(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(capture, "MAX_SEQ_ITEMS", 2)
    specs = [_InputSpec(shape=("num_tokens", 10, 20)) for _ in range(3)]
    args = _Args(**{field: _GraphConfig(extra_model_inputs=specs)})
    config, metadata = _collect(args)
    assert config == {f"{field}.extra_model_inputs.shape": [["num_tokens", 10]] * 2}
    assert metadata["sequence_truncated"] is True
    assert metadata["unsafe_excluded"] is False

    # Unsafe values beyond the retained prefix must still exclude the field.
    getattr(args, field).extra_model_inputs[-1].shape = ("private",)
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["unsafe_excluded"] is True
    assert metadata["sequence_truncated"] is False

    # A rejected field must not clear truncation recorded by another field.
    other = "encoder_cuda_graph_config" if field == "cuda_graph_config" else "cuda_graph_config"
    setattr(args, other, _GraphConfig(extra_model_inputs=specs[:1]))
    config, metadata = _collect(args)
    assert config == {f"{other}.extra_model_inputs.shape": [["num_tokens", 10]]}
    assert metadata["sequence_truncated"] is True
    assert metadata["unsafe_excluded"] is True


def test_model_sequence_respects_total_payload_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "MAX_CONFIG_BYTES", 64)
    args = _Group(members=[_Leaf(value=i) for i in range(100)])
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["payload_truncated"] is True


def test_excluded_sequence_and_mapping_do_not_enroll_children() -> None:
    class _Excluded(StrictBaseModel):
        hidden: list[_Leaf] = Field(telemetry=False)
        mapping: dict[str, _Leaf]

    args = _Excluded(hidden=[_Leaf(value=3)], mapping={"private": _Leaf(value=4)})
    assert capture.manifest_rows(_Excluded) == []
    assert _collect(args)[0] == {}
