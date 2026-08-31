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

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_fmha_params.py"
SCHEMA_PATHS = [
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "fmha" / "interface.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "interface.py",
    REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "sparse" / "params.py",
]


@pytest.fixture(scope="module")
def generator():
    """Load the generator script directly; it depends on nothing but the stdlib."""
    spec = importlib.util.spec_from_file_location("test_fmha_params_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _schema_from_source(generator, tmp_path: Path, fields: str, name: str = "Example"):
    """Run the generator over a throwaway schema written as source.

    The generator parses rather than imports, so a test schema only has to be
    syntactically valid; none of the names in it need to resolve.
    """
    path = tmp_path / "schema.py"
    path.write_text(f"class {name}:\n" + textwrap.indent(textwrap.dedent(fields).strip(), "    "))
    schema = generator.load_schemas([path], name)
    return schema, schema.structs[name]


def test_scalars_widen_to_one_integer_and_one_float_type(generator, tmp_path: Path) -> None:
    schema, struct = _schema_from_source(
        generator,
        tmp_path,
        """
        flag: bool = cpp_metadata(default=False)
        count: int = cpp_metadata(default=0)
        ratio: float = cpp_metadata(default=0.0)
        limit: Optional[int] = cpp_metadata(default=None)
        """,
    )
    rendered = generator.render_fields(schema, struct)
    assert "TRTLLM_FMHA_PARAM_FIELD(flag, bool)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(count, std::int64_t)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(ratio, double)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(limit, std::optional<std::int64_t>)" in rendered


def test_named_types_come_from_the_annotation(generator, tmp_path: Path) -> None:
    schema, struct = _schema_from_source(
        generator,
        tmp_path,
        """
        mask_type: AttentionMaskType = cpp_metadata(default=1)
        quant_mode: QuantMode = cpp_metadata(default=0)
        cp_group: Set[int] = cpp_metadata(ctype=torch.int32, default=None)
        """,
    )
    rendered = generator.render_fields(schema, struct)
    assert "(mask_type, tensorrt_llm::kernels::AttentionMaskType)" in rendered
    assert "(quant_mode, tensorrt_llm::common::QuantMode)" in rendered
    assert "(cp_group, std::set<std::int32_t>)" in rendered


def test_optionality_is_read_off_the_annotation(generator, tmp_path: Path) -> None:
    schema, struct = _schema_from_source(
        generator,
        tmp_path,
        """
        required: torch.Tensor = cpp_metadata(ctype=torch.int32, default=None)
        optional: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)
        """,
    )
    rendered = generator.render_fields(schema, struct)
    assert "TRTLLM_FMHA_PARAM_FIELD(required, torch::Tensor)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(optional, std::optional<torch::Tensor>)" in rendered

    accessors = generator.render_accessors(schema, struct)
    assert "return required.data_ptr<std::int32_t>();" in accessors
    assert "return optional.has_value() ? optional.value().data_ptr<std::int32_t>() : nullptr;" in (
        accessors
    )


def test_ctype_selects_how_the_getter_is_produced(generator, tmp_path: Path) -> None:
    schema, struct = _schema_from_source(
        generator,
        tmp_path,
        """
        typed: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
        runtime_dtype: Optional[torch.Tensor] = cpp_metadata(default=None)
        handwritten: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
        """,
    )
    accessors = generator.render_accessors(schema, struct)
    assert "float* getTyped() const" in accessors
    # An omitted ctype means the element type is the op's runtime dtype.
    assert f"template <typename {generator.RUNTIME_DTYPE}>" in accessors
    assert f"{generator.RUNTIME_DTYPE}* getRuntimeDtype() const" in accessors
    # ctype=None hands the getter to attentionOp.h.
    assert "getHandwritten" not in accessors


def test_unsupported_annotation_names_the_field(generator, tmp_path: Path) -> None:
    schema, struct = _schema_from_source(
        generator, tmp_path, "value: SomeUnknownType = cpp_metadata(default=None)"
    )
    with pytest.raises(ValueError, match=r"value: unsupported annotation SomeUnknownType"):
        generator.render_fields(schema, struct)


def test_rendering_is_deterministic_and_write_is_content_stable(tmp_path: Path, generator) -> None:
    schema = generator.load_schemas(SCHEMA_PATHS, "FmhaParams")
    root = schema.structs["FmhaParams"]
    assert generator.render_fields(schema, root) == generator.render_fields(schema, root)
    assert generator.generate(schema, tmp_path, check=False, root_class="FmhaParams") == 0
    generated = tmp_path / generator._filename("FmhaParams", "fields")
    initial_mtime = generated.stat().st_mtime_ns
    assert generator.generate(schema, tmp_path, check=False, root_class="FmhaParams") == 0
    assert generated.stat().st_mtime_ns == initial_mtime
    assert generator.generate(schema, tmp_path, check=True, root_class="FmhaParams") == 0


def test_legacy_arguments_build_python_params() -> None:
    from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaParams

    params = FmhaParams._from_arguments(
        {
            "num_heads": 8,
            "max_attention_window_size": 2048,
            "not_an_fmha_field": "ignored",
        },
        layer_idx=3,
    )

    assert params.num_heads == 8
    assert params.max_attention_window_size == 2048
    assert params.layer_idx == 3


def _stub_native_holder(monkeypatch):
    """Stand in for the native holder, mirroring its nested layout.

    Members start out value-initialized the way the C++ struct does, so a Python
    None leaving a field untouched is observable.
    """

    class NativeSparseRuntimeParams:
        def __init__(self):
            self.threshold_scale_factor_prefill = 0.0
            self.threshold_scale_factor_decode = 0.0

    class NativeForwardArgs:
        def __init__(self):
            self.attention_window_size = 0
            self.sparse_runtime_params = NativeSparseRuntimeParams()
            self.sparse_backend_args = SimpleNamespace()

    class NativeParams:
        def __init__(self):
            self.fwd = NativeForwardArgs()

    internal = ModuleType("tensorrt_llm.bindings.internal")
    internal.thop = SimpleNamespace(FmhaParams=NativeParams)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.bindings.internal", internal)
    return NativeParams


def test_python_params_convert_to_native_holder(monkeypatch) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaParams

    native_type = _stub_native_holder(monkeypatch)

    params = FmhaParams(num_heads=8)
    params.max_attention_window_size = 2048
    params.remove_padding = True
    native = params.to_thop_params()

    assert isinstance(native, native_type)
    assert native.num_heads == 8
    assert native.max_attention_window_size == 2048
    assert native.remove_padding is True


def _real_schema_classes():
    from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaParams
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        PredefinedAttentionMask,
    )
    from tensorrt_llm._torch.attention.backends.sparse.params import SparseRuntimeParams

    return FmhaParams, AttentionForwardArgs, SparseRuntimeParams, PredefinedAttentionMask


def test_nested_python_args_are_lowered_once(monkeypatch) -> None:
    from tensorrt_llm.functional import AttentionMaskType

    FmhaParams, ForwardArgs, SparseParams, Mask = _real_schema_classes()
    _stub_native_holder(monkeypatch)

    native = FmhaParams(
        fwd=ForwardArgs(
            output="full-output",
            output_sf="output-scale",
            attention_mask=Mask.CAUSAL,
            attention_window_size=2048,
            sparse_runtime_params=SparseParams(
                sparse_kv_indices="sparse-indices",
                threshold_scale_factor_prefill=0.25,
                threshold_scale_factor_decode=0.5,
            ),
        ),
        output="phase-output",
        beam_width=4,
        is_cross=True,
    ).to_thop_params()

    # The native side is handed the phase-local slice only. `fwd.output` is the
    # caller-facing buffer and is Python-only, so it must not reach the native struct;
    # letting it through would put two `output` fields in scope.
    assert native.output == "phase-output"
    assert not hasattr(native.fwd, "output")
    assert native.fwd.output_sf == "output-scale"
    assert native.fwd.attention_window_size == 2048
    assert native.fwd.sparse_runtime_params.sparse_kv_indices == "sparse-indices"
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_prefill == 0.25
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_decode == 0.5
    # Derived rather than declared, so it has no field on either side.
    assert native.mask_type == AttentionMaskType.causal
    assert native.beam_width == 1


def test_nested_none_does_not_replace_native_value(monkeypatch) -> None:
    FmhaParams, ForwardArgs, _, Mask = _real_schema_classes()
    _stub_native_holder(monkeypatch)

    native = FmhaParams(fwd=ForwardArgs(attention_mask=Mask.CAUSAL)).to_thop_params()

    # attention_window_size defaults to None on the Python side, which must leave
    # the value-initialized native field alone.
    assert native.fwd.attention_window_size == 0


def test_generated_declarations_compile(tmp_path: Path, generator) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("No C++ compiler is available")

    schema, struct = _schema_from_source(
        generator,
        tmp_path,
        """
        enabled: bool = cpp_metadata(default=False)
        count: int = cpp_metadata(default=3)
        offset: Optional[int] = cpp_metadata(default=None)
        """,
    )
    assert generator.generate(schema, tmp_path, check=False, root_class=struct.name) == 0
    source = tmp_path / "smoke.cpp"
    source.write_text(
        """#include <cstdint>
#include <optional>
struct Params
{
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type) cpp_type name{};
#include "example_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD
};
int main()
{
    Params params{};
    return params.enabled || params.count != 0 || params.offset.has_value();
}
"""
    )
    subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", str(source), "-I", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
