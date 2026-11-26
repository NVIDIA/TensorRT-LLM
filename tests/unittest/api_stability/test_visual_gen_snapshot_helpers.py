# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutation tests for the API guard, independent of product imports and GPUs."""

import dataclasses
import sys
from enum import Enum
from types import ModuleType
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, computed_field, create_model
from visual_gen_api_snapshot import (
    check_reference,
    render_snapshot,
    snapshot_callable,
    snapshot_class,
    snapshot_classes,
    snapshot_exports,
    verify_source_module,
)

pytestmark = pytest.mark.cpu_only


def _factory_a() -> list:
    raise AssertionError("The snapshot must not call factories")


def _factory_b() -> list:
    raise AssertionError("The snapshot must not call factories")


def _changed(before: dict, after: dict, tmp_path) -> None:
    reference = tmp_path / "reference.yaml"
    reference.write_text(render_snapshot(before))
    check_reference(before, reference)
    with pytest.raises(AssertionError, match="VisualGen Python API changed"):
        check_reference(after, reference)


@pytest.mark.parametrize("change", ["required", "type", "constraint", "alias", "factory"])
def test_pydantic_field_drift(change, tmp_path) -> None:
    before = {"value": (int | None, Field(default=None))}
    after = {
        "required": {"value": (int | None, Field())},
        "type": {"value": (str | None, Field(default=None))},
        "constraint": {"value": (int | None, Field(default=None, ge=1))},
        "alias": {"value": (int | None, Field(default=None, alias="renamed"))},
        "factory": {"value": (int | None, Field(default_factory=_factory_a))},
    }[change]
    _changed(
        snapshot_class(create_model("Before", **before)),
        snapshot_class(create_model("After", **after)),
        tmp_path,
    )


def test_factories_are_not_called_and_identity_is_checked(tmp_path) -> None:
    before = create_model("Before", value=(list, Field(default_factory=_factory_a)))
    after = create_model("After", value=(list, Field(default_factory=_factory_b)))
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)
    assert render_snapshot(snapshot_class(before)) == render_snapshot(snapshot_class(before))


def test_unnamed_factories_fail() -> None:
    model = create_model("Model", value=(list, Field(default_factory=lambda: [])))
    with pytest.raises(TypeError, match="stable module-level name"):
        snapshot_class(model)


@pytest.mark.parametrize(
    "setting",
    [
        "extra",
        "frozen",
        "strict",
        "validate_assignment",
        "arbitrary_types_allowed",
        "populate_by_name",
    ],
)
def test_model_config_drift(setting, tmp_path) -> None:
    before = create_model("Before", value=(int, 1))
    config = ConfigDict(**{setting: "forbid" if setting == "extra" else True})
    after = create_model("After", __config__=config, value=(int, 1))
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)


def test_computed_fields_are_checked(tmp_path) -> None:
    class Before(BaseModel):
        value: int = 1

    class After(Before):
        @computed_field
        @property
        def doubled(self) -> int:
            raise AssertionError("Do not evaluate properties")

    assert "doubled" in snapshot_class(After)["computed_fields"]
    _changed(snapshot_class(Before), snapshot_class(After), tmp_path)


@pytest.mark.parametrize(
    "change",
    ["keyword_only", "required", "kwargs", "return", "async", "async_generator", "argument_order"],
)
def test_callable_drift(change, tmp_path) -> None:
    def original(value: int = 1, other: int = 2) -> int:
        pass

    def keyword_only(*, value: int = 1, other: int = 2) -> int:
        pass

    def required(value: int, other: int = 2) -> int:
        pass

    def kwargs(value: int = 1, other: int = 2, **overrides) -> int:
        pass

    def returns(value: int = 1, other: int = 2) -> str:
        pass

    async def coroutine(value: int = 1, other: int = 2) -> int:
        pass

    async def generator(value: int = 1, other: int = 2) -> int:
        yield 1

    def reordered(other: int = 2, value: int = 1) -> int:
        pass

    changed = {
        "keyword_only": keyword_only,
        "required": required,
        "kwargs": kwargs,
        "return": returns,
        "async": coroutine,
        "async_generator": generator,
        "argument_order": reordered,
    }[change]
    _changed(snapshot_callable(original), snapshot_callable(changed), tmp_path)


def test_default_none_and_missing_annotation_remain_distinct(tmp_path) -> None:
    def before(value):
        pass

    def after(value=None) -> None:
        pass

    _changed(snapshot_callable(before), snapshot_callable(after), tmp_path)


def test_equal_numeric_defaults_keep_their_types(tmp_path) -> None:
    _changed({"default": 1}, {"default": True}, tmp_path)
    _changed({"default": 1}, {"default": 1.0}, tmp_path)


@pytest.mark.parametrize("binding", ["class", "static", "property"])
def test_descriptor_kinds(binding, tmp_path) -> None:
    class Api:
        def method(self) -> int:
            raise AssertionError("Do not call methods")

    before = snapshot_class(Api)
    function = Api.method
    setattr(
        Api,
        "method",
        {"class": classmethod, "static": staticmethod, "property": property}[binding](function),
    )
    _changed(before, snapshot_class(Api), tmp_path)


def test_snapshot_never_creates_instance_or_evaluates_property() -> None:
    class Api:
        def __new__(cls):
            raise AssertionError("Do not construct instances")

        @property
        def value(self) -> int:
            raise AssertionError("Do not read properties")

    assert "value" in snapshot_class(Api)["members"]


def test_dataclass_fields_and_factories(tmp_path) -> None:
    before = dataclasses.make_dataclass(
        "Before", [("value", list, dataclasses.field(default_factory=_factory_a))]
    )
    after = dataclasses.make_dataclass(
        "After", [("value", list, dataclasses.field(default_factory=_factory_a, kw_only=True))]
    )
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)


@pytest.mark.parametrize("change", ["discriminator", "order"])
def test_alias_discriminator_and_union_order(change, tmp_path) -> None:
    class A(BaseModel):
        kind: Literal["a"] = "a"
        mode: Literal["a"] = "a"

    class B(BaseModel):
        kind: Literal["b"] = "b"
        mode: Literal["b"] = "b"

    before = create_model("Before", value=(Annotated[A | B, Field(discriminator="kind")], ...))
    after_type = (
        Annotated[B | A, Field(discriminator="kind")]
        if change == "order"
        else Annotated[A | B, Field(discriminator="mode")]
    )
    after = create_model("After", value=(after_type, ...))
    assert snapshot_class(before)["fields"]["value"]["discriminator"] == "kind"
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)


def test_new_nested_models_are_discovered(tmp_path, monkeypatch) -> None:
    class Nested(BaseModel):
        value: int = 1

    class Root(BaseModel):
        nested: Nested | None = None

    for cls in (Nested, Root):
        monkeypatch.setattr(cls, "__module__", "tensorrt_llm.visual_gen.synthetic")
    result = snapshot_classes([Root])
    assert len(result) == 2
    without_nested = {name: value for name, value in result.items() if name.endswith("Root")}
    _changed(without_nested, result, tmp_path)


def test_export_drift_and_provider_identity(tmp_path, monkeypatch) -> None:
    package = ModuleType("api_package")
    provider = ModuleType("api_provider")
    provider.Api = type("Api", (), {})
    package.Api = provider.Api
    package.__all__ = ["Api"]
    package._LAZY_ATTRS = {"Api": "api_provider"}
    monkeypatch.setitem(sys.modules, "api_provider", provider)
    before = snapshot_exports(package)
    package.__all__ = []
    _changed(before, snapshot_exports(package), tmp_path)
    package.__all__ = ["Api"]
    package.Api = object()
    with pytest.raises(AssertionError, match="differs from"):
        snapshot_exports(package)


def test_unsupported_defaults_fail() -> None:
    def function(value=object()):
        pass

    with pytest.raises(TypeError, match="Unsupported API default"):
        snapshot_callable(function)


def test_enum_wire_value_is_checked(tmp_path) -> None:
    before = Enum("Choice", {"VALUE": "before"})
    after = Enum("Choice", {"VALUE": "after"})

    def original(value=before.VALUE):
        pass

    def changed(value=after.VALUE):
        pass

    _changed(snapshot_callable(original), snapshot_callable(changed), tmp_path)


def test_matching_wheel_sources_are_accepted_and_stale_ones_fail(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "api").mkdir(parents=True)
    (source / "api/module.py").write_text("class Api: pass\n")
    installed = tmp_path / "wheel.py"
    installed.write_text("class Api: pass\n")
    module = ModuleType("api.module")
    module.__file__ = str(installed)
    verify_source_module(module, source)
    installed.write_text("class Different: pass\n")
    with pytest.raises(AssertionError, match="does not match checkout source"):
        verify_source_module(module, source)


def test_literal_enum_values_are_discovered(tmp_path, monkeypatch) -> None:
    before = Enum("Choice", {"VALUE": "before"})
    after = Enum("Choice", {"VALUE": "after"})
    for cls in (before, after):
        monkeypatch.setattr(cls, "__module__", "tensorrt_llm.visual_gen.synthetic")
    _changed(
        snapshot_classes([Literal[before.VALUE]]),
        snapshot_classes([Literal[after.VALUE]]),
        tmp_path,
    )


def test_shadowed_base_members_are_not_inherited() -> None:
    class Parent:
        @property
        def value(self) -> int:
            raise AssertionError("Do not read properties")

    class Child(Parent):
        value = None

    assert "value" not in snapshot_class(Child)["members"]


def test_unknown_constraint_metadata_fails() -> None:
    model = create_model("Model", value=(int, 1))
    model.model_fields["value"].metadata.append(object())
    with pytest.raises(TypeError, match="Unsupported field metadata"):
        snapshot_class(model)


@pytest.mark.parametrize(
    "option", ["init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots", "match_args"]
)
def test_dataclass_options_are_checked(option, tmp_path) -> None:
    before = dataclasses.make_dataclass("Output", [("value", int, 1)])
    enabled_by_default = {"init", "repr", "eq", "match_args"}
    after = dataclasses.make_dataclass(
        "Output", [("value", int, 1)], **{option: option not in enabled_by_default}
    )
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)


def test_required_dataclass_initvar_is_checked(tmp_path) -> None:
    before = dataclasses.make_dataclass("Output", [("value", int, 1)])
    after = dataclasses.make_dataclass(
        "Output", [("required", dataclasses.InitVar[int]), ("value", int, 1)]
    )
    _changed(snapshot_class(before), snapshot_class(after), tmp_path)


def test_owned_mixin_after_framework_base_is_checked(monkeypatch) -> None:
    class Mixin:
        def public(self) -> None:
            raise AssertionError("Do not call methods")

    monkeypatch.setattr(Mixin, "__module__", "tensorrt_llm.visual_gen.synthetic")

    class Model(BaseModel, Mixin):
        value: int = 1

    members = snapshot_class(Model)["members"]
    assert "public" in members
    assert "model_dump" not in members


def test_property_deleter_is_checked(tmp_path) -> None:
    class Api:
        @property
        def value(self) -> int:
            raise AssertionError("Do not read properties")

    before = snapshot_class(Api)

    def deleter(self) -> None:
        raise AssertionError("Do not delete properties")

    Api.value = Api.value.deleter(deleter)
    _changed(before, snapshot_class(Api), tmp_path)
