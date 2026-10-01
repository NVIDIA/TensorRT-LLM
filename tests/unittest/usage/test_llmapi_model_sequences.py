# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-to-payload coverage for sequences of configuration models."""

import json
from typing import Literal

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
    wire = json.dumps(config)
    assert "private_customer_input" not in wire
    assert "float32" not in wire
    assert "12345" not in wire


def test_empty_model_list_is_distinct_from_unset_parent() -> None:
    assert _collect(_Args())[0] == {}
    config, metadata = _collect(_Args(cuda_graph_config=_GraphConfig()))
    assert config == {"cuda_graph_config.extra_model_inputs.shape": []}
    assert metadata["unsafe_excluded"] is False


class _Leaf(StrictBaseModel):
    value: int | None = None


class _Group(StrictBaseModel):
    members: list[_Leaf]


def test_nested_model_lists_and_nullable_leaves_preserve_positions() -> None:
    class _Groups(StrictBaseModel):
        groups: list[_Group]

    args = _Groups(groups=[_Group(members=[_Leaf(value=7), _Leaf()]), _Group(members=[])])
    assert _collect(args)[0] == {"groups.members.value": [[7, None], []]}
    assert capture.manifest_rows(_Groups)[0]["capture_policy"] == "list[list[int|none]]"


def test_nested_containers_before_model_are_retained() -> None:
    class _Matrix(StrictBaseModel):
        items: list[tuple[_Leaf, ...]]

    args = _Matrix(items=[(_Leaf(value=1), _Leaf(value=2)), ()])
    assert _collect(args)[0] == {"items.value": [[1, 2], []]}
    assert capture.manifest_rows(_Matrix)[0]["capture_policy"] == "list[tuple[int|none]]"


def test_optional_model_inside_list_does_not_fabricate_null_leaf() -> None:
    class _OptionalItems(StrictBaseModel):
        items: list[_Leaf | None]

    config, metadata = _collect(_OptionalItems(items=[_Leaf(value=1), None]))
    assert config == {}
    assert metadata["unsafe_excluded"] is True


@pytest.mark.parametrize("container", [list, tuple, set])
def test_model_sequence_container_types(container: type) -> None:
    class _FrozenLeaf(StrictBaseModel):
        model_config = {"frozen": True}
        value: int

    annotation = tuple[_FrozenLeaf, ...] if container is tuple else container[_FrozenLeaf]

    class _Sequence(StrictBaseModel):
        items: annotation

    args = _Sequence(items=container([_FrozenLeaf(value=2), _FrozenLeaf(value=1)]))
    expected = [1, 2] if container is set else [2, 1]
    assert _collect(args)[0] == {"items.value": expected}
    assert capture.manifest_rows(_Sequence)[0]["capture_policy"] == f"{container.__name__}[int]"


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


@pytest.mark.parametrize("bad_element", [None, {}, "private", _ModeA()])
def test_unresolvable_element_omits_whole_projection(bad_element: object) -> None:
    args = _Group.model_construct(members=[_Leaf(value=1), bad_element, _Leaf(value=2)])
    config, metadata = _collect(args)
    assert config == {}
    assert metadata["capture_succeeded"] is True
    assert metadata["excluded_field_count"] == 1


def test_unrecognized_subclass_is_not_captured() -> None:
    class _PrivateLeaf(_Leaf):
        pass

    config, metadata = _collect(_Group(members=[_PrivateLeaf(value=987654)]))
    assert config == {}
    assert metadata["unsafe_excluded"] is True


def test_scalar_and_sequence_model_union_preserves_each_wire_shape() -> None:
    class _Either(StrictBaseModel):
        item: _Leaf | list[_Leaf]

    assert capture.manifest_rows(_Either)[0]["capture_policy"] == "int|list[int|none]|none"
    assert _collect(_Either(item=_Leaf(value=3)))[0] == {"item.value": 3}
    assert _collect(_Either(item=[_Leaf(value=3)]))[0] == {"item.value": [3]}


def test_model_sequence_caps_outer_and_leaf_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "MAX_SEQ_ITEMS", 2)
    specs = [_InputSpec(shape=("num_tokens", 10, 20)) for _ in range(3)]
    args = _Args(cuda_graph_config=_GraphConfig(extra_model_inputs=specs))
    config, metadata = _collect(args)
    assert config == {"cuda_graph_config.extra_model_inputs.shape": [["num_tokens", 10]] * 2}
    assert metadata["sequence_truncated"] is True
    assert metadata["unsafe_excluded"] is False

    # Unsafe values beyond the retained prefix must still exclude the field.
    args.cuda_graph_config.extra_model_inputs[-1].shape = ("private",)
    config, metadata = _collect(args)
    assert config == {}
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
