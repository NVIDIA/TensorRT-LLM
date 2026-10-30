# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot the VisualGen Python contract without constructing runtime objects.

Run this file with --write to regenerate the reference in a matching TRT-LLM
environment. YAML configuration validation remains covered by the editor schemas.
"""

import argparse
import dataclasses
import difflib
import enum
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import get_args

import yaml
from pydantic import BaseModel

REFERENCE_PATH = Path(__file__).parent / "references/visual_gen_api.yaml"
_OWNED_PREFIXES = ("tensorrt_llm.visual_gen", "tensorrt_llm._torch.visual_gen")
_API_DUNDERS = {
    "__init__",
    "__await__",
    "__call__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__iter__",
    "__aiter__",
}


def _qualified(value: object) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _factory_name(factory: object) -> str:
    name = _qualified(factory)
    if "<lambda>" in name or "<locals>" in name:
        raise TypeError("API default factories must have a stable module-level name")
    return name


def _annotation(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return "<unspecified>"
    if isinstance(annotation, str):
        return annotation
    if inspect.isclass(annotation):
        return _qualified(annotation)
    return inspect.formatannotation(annotation)


def _value(value: object) -> object:
    """Encode defaults without executing factories or falling back to object repr."""
    if isinstance(value, enum.Enum):
        return {"enum": f"{_qualified(type(value))}.{value.name}", "value": _value(value.value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, list):
        return [_value(item) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [_value(item) for item in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _value(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not inspect.isclass(value):
        return {
            "type": _qualified(type(value)),
            "fields": {
                field.name: _value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.isbuiltin(value):
        return {"callable": _qualified(value)}
    raise TypeError(f"Unsupported API default: {type(value)}; add an explicit stable encoding")


def _default(default: object, factory: object, *, required: bool) -> dict:
    if required:
        return {"required": True}
    if factory is not None:
        return {"required": False, "factory": _factory_name(factory)}
    return {"required": False, "default": _value(default)}


def _metadata(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _value(value)
    # Pydantic uses this metadata object for constraints such as pattern.
    if type(value).__name__ == "_PydanticGeneralMetadata":
        return {"type": _qualified(type(value)), "fields": _value(vars(value))}
    raise TypeError(f"Unsupported field metadata: {type(value)}")


def snapshot_callable(
    function: object, binding: str = "instance", *, factories: dict[str, object] | None = None
) -> dict:
    """Read an unbound callable, retaining argument order and invocation semantics."""
    signature = inspect.signature(function, eval_str=False)
    return {
        "binding": binding,
        "async": inspect.iscoroutinefunction(function),
        "async_generator": inspect.isasyncgenfunction(function),
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "annotation": _annotation(parameter.annotation),
                **_default(
                    parameter.default,
                    (factories or {}).get(parameter.name),
                    required=parameter.default is inspect.Parameter.empty,
                ),
            }
            for parameter in signature.parameters.values()
        ],
        "returns": _annotation(signature.return_annotation),
    }


def _owned(cls: type) -> bool:
    return cls.__module__.startswith(_OWNED_PREFIXES)


def _members(cls: type) -> dict:
    result = {}
    seen = set()
    for base in cls.__mro__:
        if (
            base is object
            or base is BaseModel
            or (base is not cls and not _owned(base) and base.__module__ != cls.__module__)
        ):
            # Excluded framework methods still shadow later mixin members.
            seen.update(vars(base))
            continue
        for name, member in vars(base).items():
            if name in seen or (name.startswith("_") and name not in _API_DUNDERS):
                continue
            seen.add(name)
            if isinstance(member, property):
                result[name] = {
                    "getter": snapshot_callable(member.fget, "property"),
                    "setter": snapshot_callable(member.fset, "property") if member.fset else None,
                    "deleter": snapshot_callable(member.fdel, "property") if member.fdel else None,
                }
            elif isinstance(member, (classmethod, staticmethod)):
                binding = "class" if isinstance(member, classmethod) else "static"
                result[name] = snapshot_callable(member.__func__, binding)
            elif inspect.isfunction(member):
                factories = None
                # Generated constructors use a sentinel for default factories.
                # Keep their signature as well: InitVar parameters aren't fields.
                if (
                    name == "__init__"
                    and dataclasses.is_dataclass(cls)
                    and member.__code__.co_filename == "<string>"
                ):
                    factories = {
                        field.name: field.default_factory
                        for field in dataclasses.fields(cls)
                        if field.default_factory is not dataclasses.MISSING
                    }
                result[name] = snapshot_callable(member, factories=factories)
    return result


def snapshot_class(cls: type) -> dict:
    """Read class dictionaries and field metadata, never an instance or property value."""
    result = {"bases": [_qualified(base) for base in cls.__bases__], "members": _members(cls)}
    if issubclass(cls, enum.Enum):
        result["kind"] = "enum"
        result["values"] = {name: _value(member.value) for name, member in cls.__members__.items()}
    elif issubclass(cls, BaseModel):
        result["kind"] = "pydantic"
        result["config"] = _value(dict(cls.model_config))
        result["fields"] = {
            name: {
                "annotation": _annotation(field.annotation),
                **_default(field.default, field.default_factory, required=field.is_required()),
                "alias": _value(field.alias),
                "validation_alias": _value(field.validation_alias),
                "serialization_alias": _value(field.serialization_alias),
                "discriminator": _value(field.discriminator),
                "frozen": field.frozen,
                "validate_default": field.validate_default,
                "metadata": [_metadata(item) for item in field.metadata],
                "status": (field.json_schema_extra or {}).get("status")
                if isinstance(field.json_schema_extra, dict)
                else None,
            }
            for name, field in cls.model_fields.items()
            if not name.startswith("_")
        }
        result["computed_fields"] = {
            name: {
                "annotation": _annotation(field.return_type),
                "alias": field.alias,
            }
            for name, field in cls.model_computed_fields.items()
        }
    elif dataclasses.is_dataclass(cls):
        result["kind"] = "dataclass"
        result["options"] = {
            name: getattr(cls.__dataclass_params__, name)
            for name in ("init", "repr", "eq", "order", "unsafe_hash", "frozen")
        }
        result["slots"] = _value(vars(cls).get("__slots__"))
        result["match_args"] = _value(vars(cls).get("__match_args__"))
        result["fields"] = [
            {
                "name": field.name,
                "annotation": _annotation(field.type),
                **_default(
                    field.default,
                    None if field.default_factory is dataclasses.MISSING else field.default_factory,
                    required=field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING,
                ),
                "init": field.init,
                "kw_only": field.kw_only,
            }
            for field in dataclasses.fields(cls)
            if not field.name.startswith("_")
        ]
    else:
        result["kind"] = "class"
    return result


def snapshot_classes(roots: list[object]) -> dict:
    """Follow exported aliases, nested fields and VisualGen-owned base classes."""
    result = {}
    pending = list(roots)
    while pending:
        annotation = pending.pop()
        pending.extend(get_args(annotation))
        if isinstance(annotation, enum.Enum):
            pending.append(type(annotation))
            continue
        if not inspect.isclass(annotation) or not _owned(annotation):
            continue
        name = _qualified(annotation)
        if name in result:
            continue
        result[name] = snapshot_class(annotation)
        pending.extend(annotation.__bases__)
        if issubclass(annotation, BaseModel):
            pending.extend(field.annotation for field in annotation.model_fields.values())
        elif dataclasses.is_dataclass(annotation):
            pending.extend(field.type for field in dataclasses.fields(annotation))
    return result


def snapshot_exports(package: ModuleType) -> dict:
    """Verify every lazy public export resolves to its declared provider."""
    names = package.__all__
    if len(names) != len(set(names)):
        raise AssertionError(f"Duplicate public exports in {package.__name__}")
    exports = {}
    for name in names:
        provider = package._LAZY_ATTRS[name]
        value = getattr(package, name)
        if value is not getattr(importlib.import_module(provider), name):
            raise AssertionError(f"{package.__name__}.{name} differs from {provider}.{name}")
        exports[name] = {"provider": provider, "annotation": _annotation(value)}
    return {"exports": exports, "lazy_exports": dict(package._LAZY_ATTRS)}


def verify_source_module(module: ModuleType, source_root: Path) -> None:
    """Accept checkout imports or identical Python sources in a built wheel."""
    origin = Path(module.__file__).resolve()
    expected = source_root.joinpath(*module.__name__.split("."))
    expected = (
        expected / "__init__.py" if origin.name == "__init__.py" else expected.with_suffix(".py")
    )
    if origin.read_bytes() != expected.read_bytes():
        raise AssertionError(
            f"API module {module.__name__} does not match checkout source: {origin}"
        )


def collect_snapshot() -> dict:
    """Collect live public definitions; requires a matching TensorRT-LLM build."""
    import tensorrt_llm
    from tensorrt_llm import visual_gen
    from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema, RefSlotSpec
    from tensorrt_llm.llmapi.utils import get_api_status

    result = snapshot_exports(visual_gen)
    roots = [getattr(visual_gen, name) for name in visual_gen.__all__]
    roots.extend([ExtraParamSchema, RefSlotSpec])
    result["classes"] = snapshot_classes(roots)
    source_root = Path(__file__).resolve().parents[3]
    modules = {
        tensorrt_llm.__name__,
        visual_gen.__name__,
        "tensorrt_llm.llmapi.llm_args",
        "tensorrt_llm.llmapi.utils",
        "tensorrt_llm.models.modeling_utils",
    }
    modules.update(name.rpartition(".")[0] for name in result["classes"])
    for module in modules:
        verify_source_module(importlib.import_module(module), source_root)
    # Status tags on classes and methods are metadata, not semantic guarantees.
    for name, snapshot in result["classes"].items():
        module, _, symbol = name.rpartition(".")
        cls = getattr(importlib.import_module(module), symbol)
        snapshot["status"] = get_api_status(cls)
        for member_name, member in snapshot["members"].items():
            raw = inspect.getattr_static(cls, member_name)
            if isinstance(raw, (classmethod, staticmethod)):
                raw = raw.__func__
            if isinstance(raw, property):
                raw = raw.fget
            member["status"] = get_api_status(raw)
    root_exports = {}
    for name, (module, attribute) in tensorrt_llm._LAZY_ATTRS.items():
        if module.startswith("tensorrt_llm.visual_gen"):
            if name not in tensorrt_llm.__all__:
                raise AssertionError(f"Missing top-level __all__ entry: {name}")
            if getattr(tensorrt_llm, name) is not getattr(visual_gen, attribute):
                raise AssertionError(f"Top-level VisualGen re-export differs: {name}")
            root_exports[name] = {"module": module, "attribute": attribute}
    result["root_exports"] = root_exports
    return result


def render_snapshot(snapshot: dict) -> str:
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Generated by tests/unittest/api_stability/visual_gen_api_snapshot.py --write\n"
    )
    return header + yaml.safe_dump(snapshot, sort_keys=True, allow_unicode=True)


def check_reference(snapshot: dict, path: Path = REFERENCE_PATH) -> None:
    """Compare plain data, with a readable diff; never evaluate reference strings."""
    expected = yaml.safe_load(path.read_text())
    if render_snapshot(snapshot) != render_snapshot(expected):
        diff = "".join(
            difflib.unified_diff(
                render_snapshot(expected).splitlines(keepends=True),
                render_snapshot(snapshot).splitlines(keepends=True),
                fromfile="reference",
                tofile="live VisualGen API",
            )
        )
        raise AssertionError(
            f"VisualGen Python API changed:\n{diff}\n"
            "Review compatibility, then regenerate with "
            "python3 tests/unittest/api_stability/visual_gen_api_snapshot.py --write"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Regenerate the reviewed API reference"
    )
    args = parser.parse_args()
    snapshot = collect_snapshot()
    if args.write:
        REFERENCE_PATH.write_text(render_snapshot(snapshot))
    else:
        check_reference(snapshot)
