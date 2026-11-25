#!/usr/bin/env python3
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
"""Generate the native FmhaParams structs and their Python-object readers.

Parse the Python sources without importing tensorrt_llm, whose bindings do not
exist yet. Two kinds of source feed the schema:

* Schema dataclasses (``FmhaParams``, ``AttentionForwardArgs``, the sparse
  structs): every annotated field of a supported type crosses, ``cpp_metadata``
  renames, skips, or attaches a tensor dtype for a generated accessor.
* Mirrored runtime classes (``TrtllmAttention``, ``TrtllmAttentionMetadata``,
  ``RopeParams``): every typed member is collected from class-body annotations,
  ``self.x: T`` assignments in any method, ``self.x = param`` in ``__init__`` with
  an annotated parameter, and ``@property`` return annotations, walking base
  classes. Only members the C++
  consumer references (``attn.x`` / ``meta.x`` in the sources passed with
  ``--usage-source``) are emitted, so the C++ code is the declaration of need.

Per struct the generator writes ``<struct>_fields.inc`` and, for schema
dataclasses, ``<struct>_accessors.inc``. For the whole schema it writes
``fmha_params_forwarding.inc`` (nested accessor forwarding) and ``fmha_params_update.inc``
(nanobind readers that fill the native structs from the Python objects).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_CLASS_NAME = "FmhaParams"
METADATA_FACTORY = "cpp_metadata"
FIELD_FACTORY = "cpp_field"

# Schema dataclasses declared in attentionOp.h, independent of their fields' dtype metadata.
NATIVE_STRUCT_NAMES = (
    "AttentionForwardArgs",
    "SparseBackendForwardArgs",
    "SparseRuntimeParams",
)

# Runtime classes mirrored into native structs: Python class -> C++ struct.
MIRRORED_CLASSES = {
    "TrtllmAttention": "AttentionLayerParams",
    "TrtllmAttentionMetadata": "AttentionMetadataParams",
    "RopeParams": "AttentionRopeParams",
}

# Root fields whose C++ usage decides which mirrored members are emitted.
USAGE_PREFIXES = ("attn", "meta")

DEFAULT_MODULE_PATHS = (
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "fmha" / "phased.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "interface.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "trtllm.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "sparse" / "params.py",
)

DEFAULT_USAGE_PATHS = (
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "attentionOp.h",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "attentionFeature.h",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "attentionVariant.h",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "attentionOp.cpp",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "phasedFmha.h",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "thop" / "phasedFmha.cpp",
    REPO_ROOT / "cpp" / "tensorrt_llm" / "nanobind" / "thop" / "bindings.cpp",
)

_TORCH_DTYPE_CPP = {
    "torch.bool": "bool",
    "torch.uint8": "std::uint8_t",
    "torch.uint32": "std::uint32_t",
    "torch.int32": "std::int32_t",
    "torch.int64": "std::int64_t",
    "torch.float32": "float",
    "torch.float64": "double",
}

# Python spells one integer and one floating type, so every scalar widens to the
# larger native type rather than carrying a redundant dtype.
_SCALAR_CPP = {
    "bool": "bool",
    "int": "std::int64_t",
    "float": "double",
}

# Enumerations cross as their integer value and are cast on the native side.
_ENUM_CPP = {
    "AttentionMaskType": "tensorrt_llm::kernels::AttentionMaskType",
    "PositionEmbeddingType": "tensorrt_llm::kernels::PositionEmbeddingType",
    "RotaryScalingType": "tensorrt_llm::kernels::RotaryScalingType",
    "DataType": "tensorrt_llm::DataType",
    # A Python IntEnum with no native counterpart crosses as its integer value.
    "AttentionInputType": "std::int64_t",
}

# QuantMode is a class wrapping an integer on both sides.
_QUANT_MODE = "QuantMode"
_QUANT_MODE_CPP = "tensorrt_llm::common::QuantMode"

_TENSOR = "torch.Tensor"

_INDENT = "    "


@dataclass(frozen=True)
class Field:
    name: str  # native member name
    source: str  # Python attribute read to fill it
    annotation: str
    dtype: str | None = None  # a torch dtype, or None (no generated accessor)


@dataclass(frozen=True)
class Struct:
    name: str  # native struct name
    fields: tuple[Field, ...]
    mirrored: bool = False  # collected from a runtime class rather than a schema dataclass


@dataclass(frozen=True)
class Schema:
    source: str
    structs: dict[str, Struct]

    def nested(self, field_: Field) -> Struct | None:
        """Return the struct a field's annotation names, if it is one."""
        return self.structs.get(_struct_name(_base_type(field_.annotation)))


@dataclass
class Usage:
    """Attribute paths the C++ consumer reads under each root prefix, e.g. attn.rope_params.theta."""

    paths: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)

    def under(self, prefix: str) -> set[tuple[str, ...]] | None:
        return self.paths.get(prefix)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _strip_quotes(annotation: str) -> str:
    """Return X for a forward reference 'X' or "X"."""
    if len(annotation) >= 2 and annotation[0] == annotation[-1] and annotation[0] in "'\"":
        return annotation[1:-1]
    return annotation


def _optional_inner(annotation: str) -> str | None:
    """Return X for Optional[X] or X | None, else None."""
    if annotation.startswith("Optional[") and annotation.endswith("]"):
        return _strip_quotes(annotation[len("Optional[") : -1])
    parts = [part.strip() for part in annotation.split("|")]
    if len(parts) == 2 and "None" in parts:
        return _strip_quotes(next(part for part in parts if part != "None"))
    return None


def _set_element(annotation: str) -> str | None:
    """Return X for Set[X] or set[X], else None."""
    for prefix in ("Set[", "set["):
        if annotation.startswith(prefix) and annotation.endswith("]"):
            return annotation[len(prefix) : -1]
    return None


def _struct_name(annotation: str) -> str:
    """Map a Python class name to its native struct name."""
    return MIRRORED_CLASSES.get(annotation, annotation)


def _base_type(annotation: str) -> str:
    return _strip_quotes(_optional_inner(annotation) or annotation)


def _is_supported(annotation: str, struct_names: set[str]) -> bool:
    base = _base_type(annotation)
    return (
        base in _SCALAR_CPP
        or base in _ENUM_CPP
        or base == _QUANT_MODE
        or base == _TENSOR
        or _struct_name(base) in struct_names
        or (_set_element(base) is not None and _set_element(base) in _SCALAR_CPP)
    )


def _is_tensor(field_: Field) -> bool:
    return _base_type(field_.annotation) == _TENSOR


# ---------------------------------------------------------------------------
# Reading schema dataclasses
# ---------------------------------------------------------------------------


def _is_property(function: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "property" for d in function.decorator_list)


def _metadata_call(statement: ast.AnnAssign) -> ast.Call | None:
    """Return the native-options call attached to a dataclass field, if any.

    Accepts ``x: T = cpp_field(default=..., <options>)``, whose keywords other than
    ``default`` are the options, and the long form
    ``x: T = field(default=..., metadata=cpp_metadata(...))``.
    """
    value = statement.value
    if not isinstance(value, ast.Call):
        return None
    if getattr(value.func, "id", None) == METADATA_FACTORY:
        raise ValueError(
            f"{statement.target.id}: write {FIELD_FACTORY}(...); "
            f"a bare {METADATA_FACTORY}(...) would become the field's default"
        )
    if getattr(value.func, "id", None) == FIELD_FACTORY:
        options = ast.Call(
            func=value.func, args=[], keywords=[k for k in value.keywords if k.arg != "default"]
        )
        return ast.copy_location(options, value)
    if getattr(value.func, "id", None) != "field":
        return None
    for keyword in value.keywords:
        if keyword.arg == "metadata" and isinstance(keyword.value, ast.Call):
            if getattr(keyword.value.func, "id", None) == METADATA_FACTORY:
                return keyword.value
    return None


def _read_struct(class_def: ast.ClassDef, struct_names: set[str]) -> Struct | None:
    """Collect native fields of a schema dataclass.

    Annotated fields cross by their annotation; a ``@property`` with a supported return
    annotation crosses as a read-only scalar, so a dataclass can derive a native field
    (``AttentionForwardArgs.mask_type``) instead of storing it.
    """
    fields = []
    for statement in class_def.body:
        if (
            isinstance(statement, ast.FunctionDef)
            and _is_property(statement)
            and statement.returns is not None
        ):
            annotation = ast.unparse(statement.returns)
            if _is_supported(annotation, struct_names) and _base_type(annotation) != _TENSOR:
                fields.append(
                    Field(name=statement.name, source=statement.name, annotation=annotation)
                )
            continue
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        annotation = ast.unparse(statement.annotation)
        metadata = _metadata_call(statement)
        if metadata is None and not _is_supported(annotation, struct_names):
            continue
        options = {keyword.arg: keyword.value for keyword in metadata.keywords} if metadata else {}
        unknown = options.keys() - {"dtype"}
        if unknown:
            raise ValueError(f"{statement.target.id}: unsupported cpp_metadata arguments {unknown}")
        if metadata is not None and not _is_supported(annotation, struct_names):
            raise ValueError(f"{statement.target.id}: unsupported annotation {annotation}")
        fields.append(
            Field(
                name=statement.target.id,
                source=statement.target.id,
                annotation=annotation,
                dtype=ast.unparse(options["dtype"]) if "dtype" in options else None,
            )
        )
    names = [field_.name for field_ in fields]
    if len(set(names)) != len(names):
        raise ValueError(f"{class_def.name}: duplicate native field names")
    return Struct(class_def.name, tuple(fields)) if fields else None


# ---------------------------------------------------------------------------
# Collecting mirrored runtime classes
# ---------------------------------------------------------------------------


def _self_target(node: ast.expr) -> str | None:
    """Return x for a `self.x` target."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _collect_members(class_def: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> dict[str, str]:
    """Return name -> annotation for every typed member, base classes first."""
    members: dict[str, str] = {}
    for base in class_def.bases:
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if isinstance(base, ast.Subscript):  # AttentionBackend[TrtllmAttentionMetadata]
            base_name = getattr(base.value, "id", None)
        if base_name in classes:
            members.update(_collect_members(classes[base_name], classes))

    for statement in class_def.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            members[statement.target.id] = ast.unparse(statement.annotation)
        elif isinstance(statement, ast.FunctionDef):
            if _is_property(statement) and statement.returns is not None:
                members[statement.name] = ast.unparse(statement.returns)
                continue
            # `self.x = param` is typed by the parameter only in __init__, matching
            # how a type checker infers constructor attributes. Elsewhere a member
            # needs `self.x: T = ...` to be collected.
            params = (
                {
                    arg.arg: ast.unparse(arg.annotation)
                    for arg in statement.args.args + statement.args.kwonlyargs
                    if arg.annotation is not None
                }
                if statement.name == "__init__"
                else {}
            )
            for node in ast.walk(statement):
                if isinstance(node, ast.AnnAssign):
                    name = _self_target(node.target)
                    if name is not None:
                        members[name] = ast.unparse(node.annotation)
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = _self_target(node.targets[0])
                    if (
                        name is not None
                        and name not in members
                        and isinstance(node.value, ast.Name)
                        and node.value.id in params
                    ):
                        members[name] = params[node.value.id]
    return {name: annotation for name, annotation in members.items() if not name.startswith("_")}


def _read_mirror(
    class_name: str,
    classes: dict[str, ast.ClassDef],
    struct_names: set[str],
    used: set[tuple[str, ...]] | None,
    mirrors: dict[str, Struct],
) -> Struct:
    """Build the native struct mirroring a runtime class, restricted to used members."""
    struct_name = _struct_name(class_name)
    if struct_name in mirrors:
        return mirrors[struct_name]
    if class_name not in classes:
        raise ValueError(f"mirrored class {class_name} not found in the schema sources")
    members = _collect_members(classes[class_name], classes)
    if used is not None:
        unknown = sorted({path[0] for path in used} - members.keys())
        if unknown:
            raise ValueError(
                f"{class_name}: C++ reads {unknown} but the class declares no typed member of that name"
            )
    fields = []
    for name, annotation in members.items():
        if not _is_supported(annotation, struct_names):
            continue
        if used is not None and not any(path[0] == name for path in used):
            continue
        base = _base_type(annotation)
        if base in MIRRORED_CLASSES:
            sub_used = (
                None
                if used is None
                else {path[1:] for path in used if path[0] == name and len(path) > 1}
            )
            _read_mirror(base, classes, struct_names, sub_used, mirrors)
        fields.append(Field(name=name, source=name, annotation=annotation))
    struct = Struct(struct_name, tuple(fields), mirrored=True)
    mirrors[struct_name] = struct
    return struct


# ---------------------------------------------------------------------------
# Usage scan of the C++ consumer
# ---------------------------------------------------------------------------

# A trailing `(` is a method call, never a member read.
_USAGE = re.compile(r"\b(" + "|".join(USAGE_PREFIXES) + r")((?:\.[A-Za-z_]\w*)+)\b(?!\s*\()")


def scan_usage(paths: Iterable[Path]) -> Usage:
    """Collect every `attn.x[.y]` / `meta.x` path the C++ sources read."""
    usage = Usage({prefix: set() for prefix in USAGE_PREFIXES})
    for path in paths:
        for prefix, dotted in _USAGE.findall(path.read_text()):
            usage.paths[prefix].add(tuple(dotted[1:].split(".")))
    return usage


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_schemas(
    source_paths: Sequence[Path],
    root_class: str,
    struct_names: Sequence[str] = NATIVE_STRUCT_NAMES,
    usage: Usage | None = None,
) -> Schema:
    """Read every schema class and mirrored class out of the given sources without importing them."""
    classes: dict[str, ast.ClassDef] = {}
    for path in source_paths:
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, node)

    all_struct_names = {root_class, *struct_names, *MIRRORED_CLASSES.values()}
    structs: dict[str, Struct] = {}
    for name in (root_class, *struct_names):
        if name not in classes:
            raise ValueError(f"schema class {name} not found in the schema sources")
        struct = _read_struct(classes[name], all_struct_names)
        if struct is not None:
            structs[name] = struct
    if root_class not in structs:
        raise ValueError(f"no class named {root_class} declares native fields")

    mirrors: dict[str, Struct] = {}
    for field_ in structs[root_class].fields:
        base = _base_type(field_.annotation)
        if base in MIRRORED_CLASSES:
            used = usage.under(field_.name) if usage is not None else None
            _read_mirror(base, classes, all_struct_names, used, mirrors)
    structs.update(mirrors)
    return Schema(source=", ".join(str(p) for p in source_paths), structs=structs)


# ---------------------------------------------------------------------------
# Rendering: types
# ---------------------------------------------------------------------------


def _element_cpp(field_: Field) -> str:
    if field_.dtype is None:
        raise ValueError(f"{field_.name}: this type needs an explicit dtype")
    try:
        return _TORCH_DTYPE_CPP[field_.dtype]
    except KeyError:
        raise ValueError(f"{field_.name}: unsupported dtype {field_.dtype}") from None


def render_field_type(schema: Schema, field_: Field, annotation: str | None = None) -> str:
    """Render the C++ type of one field from its annotation."""
    annotation = field_.annotation if annotation is None else annotation
    inner = _optional_inner(annotation)
    if inner is not None:
        # A nested struct is held by value; C++ mirrors the Python layout, and an
        # absent one is the value-initialized struct rather than an empty optional.
        if _struct_name(inner) in schema.structs:
            return render_field_type(schema, field_, inner)
        return f"std::optional<{render_field_type(schema, field_, inner)}>"
    base = _strip_quotes(annotation)
    if _struct_name(base) in schema.structs:
        return _struct_name(base)
    if base == _TENSOR:
        return "torch::Tensor"
    element = _set_element(base)
    if element is not None:
        return f"std::set<{_SCALAR_CPP[element]}>"
    if base in _SCALAR_CPP:
        return _SCALAR_CPP[base]
    if base in _ENUM_CPP:
        return _ENUM_CPP[base]
    if base == _QUANT_MODE:
        return _QUANT_MODE_CPP
    raise ValueError(f"{field_.name}: unsupported annotation {annotation}")


def _header_lines(schema: Schema, struct: Struct | None = None) -> list[str]:
    what = f"{struct.name} in {schema.source}" if struct is not None else schema.source
    return [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "// SPDX-License-Identifier: Apache-2.0",
        "// Generated file; do not edit.",
        "// Generated by scripts/generate_fmha_params.py.",
        f"// Source: {what}.",
        "",
    ]


# ---------------------------------------------------------------------------
# Rendering: fields and accessors (per struct)
# ---------------------------------------------------------------------------


def render_fields(schema: Schema, struct: Struct) -> str:
    """Render one struct's member declarations.

    Python owns semantic initialization; the members are only value-initialized
    here so none of them start out indeterminate.
    """
    lines = _header_lines(schema, struct) + [
        "#if !defined(TRTLLM_FMHA_PARAM_FIELD)",
        '# error "Define TRTLLM_FMHA_PARAM_FIELD before including this file"',
        "#endif",
        "",
    ]
    for field_ in struct.fields:
        lines.append(f"TRTLLM_FMHA_PARAM_FIELD({field_.name}, {render_field_type(schema, field_)})")
    return "\n".join(lines) + "\n"


def _accessor_name(field_name: str) -> str:
    return "get" + "".join(part.capitalize() for part in field_name.split("_"))


def _render_accessor(field_: Field) -> list[str]:
    """Render one tensor getter: the pointer the kernels expect, or nullptr if absent."""
    pointer = _element_cpp(field_)
    read = "static_cast<" + pointer + "*>({0}.data_ptr())"
    name = field_.name
    if _optional_inner(field_.annotation) is not None:
        body = f"return {name}.has_value() ? {read.format(name + '.value()')} : nullptr;"
    else:
        body = f"return {read.format(name)};"
    return [f"{pointer}* {_accessor_name(name)}() const", "{", f"    {body}", "}"]


def render_accessors(schema: Schema, struct: Struct) -> str:
    """Render the getters that are a plain typed view of a tensor field.

    Unmarked tensors use handwritten getters for runtime-dispatched pointer
    types and semantic views that need offsets or a different buffer layout.
    """
    lines = _header_lines(schema, struct)
    for field_ in struct.fields:
        if not _is_tensor(field_) or field_.dtype is None:
            continue
        lines.extend(_render_accessor(field_))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_forwarding(schema: Schema, struct: Struct) -> str:
    """Render getters that reach through a nested member.

    Keeps call sites reading `p.getX()` even after X moved into a nested struct. A
    name the outer struct defines itself is left alone: `output` and
    `attention_mask` mean different things at the two levels.
    """
    lines = _header_lines(schema, struct)
    own = {_accessor_name(f.name) for f in struct.fields if _is_tensor(f) and f.dtype is not None}
    seen = set(own)
    for field_ in struct.fields:
        nested = schema.nested(field_)
        if nested is None:
            continue
        for inner, prefix in _forwardable(schema, nested, field_.name):
            name = _accessor_name(inner.name)
            if name in seen:
                continue
            seen.add(name)
            pointer = _element_cpp(inner)
            lines += [f"{pointer}* {name}() const", "{", f"    return {prefix}.{name}();", "}", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _forwardable(schema: Schema, struct: Struct, prefix: str):
    """Yield (field, access path) for every getter reachable through a nested member."""
    for field_ in struct.fields:
        nested = schema.nested(field_)
        if nested is not None:
            yield from _forwardable(schema, nested, f"{prefix}.{field_.name}")
        elif _is_tensor(field_) and field_.dtype is not None:
            yield field_, prefix


# ---------------------------------------------------------------------------
# Rendering: update (whole schema, nested structs first)
# ---------------------------------------------------------------------------


def _dependency_order(schema: Schema, root: str) -> list[Struct]:
    """Structs reachable from the root, each after the structs it contains."""
    order: list[Struct] = []
    seen: set[str] = set()

    def visit(struct: Struct) -> None:
        if struct.name in seen:
            return
        seen.add(struct.name)
        for field_ in struct.fields:
            nested = schema.nested(field_)
            if nested is not None:
                visit(nested)
        order.append(struct)

    visit(schema.structs[root])
    return order


def _read_expression(schema: Schema, field_: Field) -> str:
    """The expression that reads one non-struct field from the Python object.

    Spelled the way the hand-written binding reads fields: `get<T>` for a required value,
    `opt<T>` for an Optional, enums cast from their integer, QuantMode from its bits. A
    required tensor that is None arrives undefined so the op's own checks name it.
    """
    optional = _optional_inner(field_.annotation) is not None
    base = _base_type(field_.annotation)
    source = f'source, "{field_.source}"'
    if base == _TENSOR:
        if optional:
            return f"opt<torch::Tensor>({source})"
        return f"opt<torch::Tensor>({source}).value_or(torch::Tensor{{}})"
    if base in _SCALAR_CPP:
        cpp = _SCALAR_CPP[base]
        return f"opt<{cpp}>({source})" if optional else f"get<{cpp}>({source})"
    if base in _ENUM_CPP:
        cpp = _ENUM_CPP[base]
        if cpp == "std::int64_t":
            return f"opt<std::int64_t>({source})" if optional else f"get<std::int64_t>({source})"
        if optional:
            raise ValueError(f"{field_.name}: Optional enums are not supported")
        return f"static_cast<{cpp}>(get<std::int64_t>({source}))"
    if base == _QUANT_MODE:
        if optional:
            raise ValueError(f"{field_.name}: Optional[QuantMode] is not supported")
        return f"{_QUANT_MODE_CPP}(get<std::uint32_t>({source}))"
    element = _set_element(base)
    if element is not None:
        cpp = f"std::set<{_SCALAR_CPP[element]}>"
        return f"opt<{cpp}>({source})" if optional else f"get<{cpp}>({source})"
    raise ValueError(f"{field_.name}: unsupported annotation {field_.annotation}")


def _render_update(schema: Schema, struct: Struct) -> list[str]:
    lines = [f"inline void update(torch_ext::{struct.name}& target, nb::handle source)", "{"]
    for field_ in struct.fields:
        nested = schema.nested(field_)
        if nested is None:
            lines.append(f"{_INDENT}target.{field_.name} = {_read_expression(schema, field_)};")
        elif _optional_inner(field_.annotation) is not None:
            lines += [
                f"{_INDENT}{{",
                f'{_INDENT}{_INDENT}nb::object const value = attr(source, "{field_.source}");',
                f"{_INDENT}{_INDENT}if (value.is_none())",
                f"{_INDENT}{_INDENT}{{",
                f"{_INDENT}{_INDENT}{_INDENT}target.{field_.name} = {{}};",
                f"{_INDENT}{_INDENT}}}",
                f"{_INDENT}{_INDENT}else",
                f"{_INDENT}{_INDENT}{{",
                f"{_INDENT}{_INDENT}{_INDENT}update(target.{field_.name}, value);",
                f"{_INDENT}{_INDENT}}}",
                f"{_INDENT}}}",
            ]
        else:
            lines.append(f'{_INDENT}update(target.{field_.name}, attr(source, "{field_.source}"));')
    lines.append("}")
    return lines


def render_update(schema: Schema, root: str) -> str:
    """Render the nanobind readers for every struct, nested structs first.

    Every field is read on every call. Requires `attr`, `get` and `opt` to be visible at
    the point of inclusion.
    """
    lines = _header_lines(schema)
    for struct in _dependency_order(schema, root):
        lines.extend(_render_update(schema, struct))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _filename(struct_name: str, kind: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", struct_name).lower()
    return f"{snake}_{kind}.inc"


def render_all(schema: Schema, root_class: str) -> dict[str, str]:
    """Every generated file, keyed by filename."""
    outputs: dict[str, str] = {}
    for struct in schema.structs.values():
        outputs[_filename(struct.name, "fields")] = render_fields(schema, struct)
        if not struct.mirrored:
            outputs[_filename(struct.name, "accessors")] = render_accessors(schema, struct)
    root = schema.structs[root_class]
    outputs[_filename(root_class, "forwarding")] = render_forwarding(schema, root)
    outputs[_filename(root_class, "update")] = render_update(schema, root_class)
    return outputs


def _write_if_changed(path: Path, content: str) -> bool:
    old_content = path.read_text() if path.exists() else None
    if old_content == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def _check_file(path: Path, content: str) -> bool:
    try:
        old_content = path.read_text()
    except FileNotFoundError:
        print(f"{path}: missing generated file", file=sys.stderr)
        return False
    if old_content == content:
        return True
    sys.stderr.writelines(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(path),
            tofile=f"{path} (expected)",
        )
    )
    return False


def generate(schema: Schema, out_dir: Path, check: bool, root_class: str) -> int:
    status = 0
    for filename, content in render_all(schema, root_class).items():
        path = out_dir / filename
        if check:
            status |= 0 if _check_file(path, content) else 1
            continue
        action = "updated" if _write_if_changed(path, content) else "unchanged"
        print(f"{action}: {path}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module-path",
        type=Path,
        action="append",
        dest="module_paths",
        help="Python source to parse; repeatable (default: the schema and runtime modules).",
    )
    parser.add_argument(
        "--usage-source",
        type=Path,
        action="append",
        dest="usage_paths",
        help="C++ source whose attn.x / meta.x reads select the mirrored members; repeatable "
        "(default: the attention op and its binding). Pass --no-usage-filter to emit every typed member.",
    )
    parser.add_argument(
        "--no-usage-filter",
        action="store_true",
        help="Emit every supported member of the mirrored classes instead of only the ones C++ reads.",
    )
    parser.add_argument(
        "--class-name",
        default=ROOT_CLASS_NAME,
        help="Outermost schema class (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory that receives the generated includes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated output against --out-dir without writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = args.module_paths or list(DEFAULT_MODULE_PATHS)
        usage = (
            None if args.no_usage_filter else scan_usage(args.usage_paths or DEFAULT_USAGE_PATHS)
        )
        schema = load_schemas(paths, args.class_name, usage=usage)
        return generate(schema, args.out_dir, args.check, args.class_name)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"FmhaParams generation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
