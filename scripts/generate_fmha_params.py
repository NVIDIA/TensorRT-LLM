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
"""Generate native FmhaParams declarations from the Python schema.

The schema is parsed, never imported: it lives inside the tensorrt_llm package,
whose import pulls in the very bindings this generator runs before. Parsing also
means a field's annotation is read exactly as written, which is where optionality
and the non-dtype types come from.

Each field's C++ type comes from its annotation, and ``ctype`` supplies the
element type the annotation cannot carry -- a tensor's dtype, or a set's.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_CLASS_NAME = "FmhaParams"
METADATA_FACTORY = "cpp_metadata"

# The schema classes are spread over the modules that own them, so every source
# is parsed and a nested member is resolved by the class its annotation names.
DEFAULT_MODULE_PATHS = (
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "fmha" / "interface.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "interface.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "sparse" / "params.py",
)

# Template parameter for a tensor whose element type is the op's runtime dtype.
RUNTIME_DTYPE = "QkvType"

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
# larger native type rather than carrying a redundant ctype.
_ANNOTATION_CPP = {
    "bool": "bool",
    "int": "std::int64_t",
    "float": "double",
    "AttentionMaskType": "tensorrt_llm::kernels::AttentionMaskType",
    "PositionEmbeddingType": "tensorrt_llm::kernels::PositionEmbeddingType",
    "RotaryScalingType": "tensorrt_llm::kernels::RotaryScalingType",
    "QuantMode": "tensorrt_llm::common::QuantMode",
    "DataType": "tensorrt_llm::DataType",
    "BlockSparseParams": "tensorrt_llm::kernels::BlockSparseParams",
    "MlaMetaParams": "tensorrt_llm::kernels::MlaMetaParams",
    # A Python IntEnum with no native counterpart crosses as its integer value.
    "AttentionInputType": "std::int64_t",
}

# Torch spells these itself, so the getter uses the typed data_ptr<T>() overload.
_TORCH_TYPED_PTR = frozenset(_TORCH_DTYPE_CPP) - {"torch.uint8"}

_ABSENT = object()


@dataclass(frozen=True)
class Field:
    name: str
    annotation: str
    ctype: str | None | object  # a torch dtype, None, or _ABSENT


@dataclass(frozen=True)
class Struct:
    name: str
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class Schema:
    source: str
    structs: dict[str, Struct]

    def nested(self, field: Field) -> Struct | None:
        """Return the struct a field's annotation names, if it is one."""
        return self.structs.get(_optional_inner(field.annotation) or field.annotation)


def _optional_inner(annotation: str) -> str | None:
    """Return X for Optional[X], else None."""
    if annotation.startswith("Optional[") and annotation.endswith("]"):
        return annotation[len("Optional[") : -1]
    return None


def _set_element(annotation: str) -> str | None:
    """Return X for Set[X], else None."""
    if annotation.startswith("Set[") and annotation.endswith("]"):
        return annotation[len("Set[") : -1]
    return None


def _read_struct(class_def: ast.ClassDef) -> Struct | None:
    """Collect one class's cpp_metadata fields, or None if it declares none."""
    fields = []
    for statement in class_def.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if getattr(call.func, "id", None) != METADATA_FACTORY:
            continue
        ctype: object = _ABSENT
        for keyword in call.keywords:
            if keyword.arg == "ctype":
                ctype = (
                    None
                    if isinstance(keyword.value, ast.Constant) and keyword.value.value is None
                    else ast.unparse(keyword.value)
                )
        fields.append(
            Field(
                name=statement.target.id,
                annotation=ast.unparse(statement.annotation),
                ctype=ctype,
            )
        )
    return Struct(class_def.name, tuple(fields)) if fields else None


def load_schemas(source_paths: Sequence[Path], root_class: str) -> Schema:
    """Read every schema class out of the given sources without importing them."""
    structs: dict[str, Struct] = {}
    for path in source_paths:
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            struct = _read_struct(node)
            if struct is not None:
                structs[struct.name] = struct
    if root_class not in structs:
        raise ValueError(f"no class named {root_class} declares {METADATA_FACTORY} fields")
    return Schema(source=", ".join(str(p) for p in source_paths), structs=structs)


def _element_cpp(field: Field) -> str:
    if field.ctype is _ABSENT or field.ctype is None:
        raise ValueError(f"{field.name}: this type needs an explicit ctype")
    try:
        return _TORCH_DTYPE_CPP[field.ctype]
    except KeyError:
        raise ValueError(f"{field.name}: unsupported ctype {field.ctype}") from None


def render_field_type(schema: Schema, field: Field, annotation: str | None = None) -> str:
    """Render the C++ type of one field from its annotation."""
    annotation = field.annotation if annotation is None else annotation
    inner = _optional_inner(annotation)
    if inner is not None:
        # A nested struct is held by value; C++ mirrors the Python layout, and an
        # absent one is the value-initialized struct rather than an empty optional.
        if annotation in schema.structs or inner in schema.structs:
            return render_field_type(schema, field, inner)
        return f"std::optional<{render_field_type(schema, field, inner)}>"
    if annotation in schema.structs:
        return annotation
    if annotation == "torch.Tensor":
        return "torch::Tensor"
    if _set_element(annotation) is not None:
        return f"std::set<{_element_cpp(field)}>"
    try:
        return _ANNOTATION_CPP[annotation]
    except KeyError:
        raise ValueError(f"{field.name}: unsupported annotation {annotation}") from None


def _header_lines(schema: Schema, struct: Struct) -> list[str]:
    return [
        "// Generated file; do not edit.",
        "// Generated by scripts/generate_fmha_params.py.",
        f"// Source: {struct.name} in {schema.source}.",
        "",
    ]


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
    for field in struct.fields:
        lines.append(f"TRTLLM_FMHA_PARAM_FIELD({field.name}, {render_field_type(schema, field)})")
    return "\n".join(lines) + "\n"


def _accessor_name(field_name: str) -> str:
    return "get" + "".join(part.capitalize() for part in field_name.split("_"))


def _is_tensor(field: Field) -> bool:
    return "torch.Tensor" in (field.annotation, _optional_inner(field.annotation))


def _render_accessor(field: Field) -> list[str]:
    """Render one tensor getter: the pointer the kernels expect, or nullptr if absent."""
    templated = field.ctype is _ABSENT
    pointer = RUNTIME_DTYPE if templated else _element_cpp(field)
    if templated or field.ctype not in _TORCH_TYPED_PTR:
        read = "static_cast<" + pointer + "*>({0}.data_ptr())"
    else:
        read = "{0}.data_ptr<" + pointer + ">()"

    name = field.name
    if _optional_inner(field.annotation) is not None:
        body = f"return {name}.has_value() ? {read.format(name + '.value()')} : nullptr;"
    else:
        body = f"return {read.format(name)};"

    lines = [f"template <typename {pointer}>"] if templated else []
    lines += [f"{pointer}* {_accessor_name(name)}() const", "{", f"    {body}", "}"]
    return lines


def render_accessors(schema: Schema, struct: Struct) -> str:
    """Render the getters that are a plain typed view of a tensor field.

    A field with ``ctype=None`` is skipped: its native view is not its dtype, so
    the getter is handwritten next to the ones that take an index or a size.
    """
    lines = _header_lines(schema, struct)
    for field in struct.fields:
        if not _is_tensor(field) or field.ctype is None:
            continue
        lines.extend(_render_accessor(field))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _accessor_names(schema: Schema, struct: Struct) -> set[str]:
    """Names a struct answers itself, generated or forwarded."""
    names = {_accessor_name(f.name) for f in struct.fields if _is_tensor(f) and f.ctype is not None}
    for field in struct.fields:
        nested = schema.nested(field)
        if nested is not None:
            names |= _accessor_names(schema, nested)
    return names


def render_forwarding(schema: Schema, struct: Struct) -> str:
    """Render getters that reach through a nested member.

    Keeps call sites reading `p.getX()` even after X moved into a nested struct. A
    name the outer struct defines itself is left alone: `output` and
    `attention_mask` mean different things at the two levels.
    """
    lines = _header_lines(schema, struct)
    own = {_accessor_name(f.name) for f in struct.fields if _is_tensor(f) and f.ctype is not None}
    seen = set(own)
    for field in struct.fields:
        nested = schema.nested(field)
        if nested is None:
            continue
        for inner, prefix in _forwardable(schema, nested, field.name):
            name = _accessor_name(inner.name)
            if name in seen:
                continue
            seen.add(name)
            templated = inner.ctype is _ABSENT
            pointer = RUNTIME_DTYPE if templated else _element_cpp(inner)
            call = f"{name}<{pointer}>()" if templated else f"{name}()"
            if templated:
                lines.append(f"template <typename {pointer}>")
            lines += [
                f"{pointer}* {name}() const",
                "{",
                f"    return {prefix}.{call};",
                "}",
                "",
            ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _forwardable(schema: Schema, struct: Struct, prefix: str):
    """Yield (field, access path) for every getter reachable through a nested member."""
    for field in struct.fields:
        nested = schema.nested(field)
        if nested is not None:
            yield from _forwardable(schema, nested, f"{prefix}.{field.name}")
        elif _is_tensor(field) and field.ctype is not None:
            yield field, prefix


def _filename(struct_name: str, kind: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", struct_name).lower()
    return f"{snake}_{kind}.inc"


# One entry per generated file: the suffix to write and the renderer that fills it.
BACKENDS = (("fields", render_fields), ("accessors", render_accessors))


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
    outputs = [
        (_filename(struct.name, kind), render(schema, struct))
        for struct in schema.structs.values()
        for kind, render in BACKENDS
    ]
    outputs.append(
        (_filename(root_class, "forwarding"), render_forwarding(schema, schema.structs[root_class]))
    )
    for filename, content in outputs:
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
        help="Python schema source to parse; repeatable (default: the schema modules).",
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
        schema = load_schemas(paths, args.class_name)
        return generate(schema, args.out_dir, args.check, args.class_name)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"FmhaParams generation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
