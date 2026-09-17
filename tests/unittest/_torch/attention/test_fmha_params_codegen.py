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

import dataclasses
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_fmha_params.py"
CPP_SCHEMA_PATH = REPO_ROOT / "tensorrt_llm" / "_torch" / "attention" / "backends" / "cpp_schema.py"


@pytest.fixture(scope="module")
def generator():
    """Load the generator script directly; it depends on nothing but the stdlib."""
    spec = importlib.util.spec_from_file_location("test_fmha_params_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cpp_schema():
    """Load field metadata without importing TensorRT-LLM or PyTorch."""
    spec = importlib.util.spec_from_file_location("test_cpp_schema", CPP_SCHEMA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("postponed", [False, True])
def test_schema_types_and_defaults(
    generator, cpp_schema, tmp_path: Path, monkeypatch, postponed: bool
) -> None:
    fields = """
        flag: bool = False
        count: int = 1099511627776
        ratio: float = 1e-6
        optional_flag: Optional[bool] = None
        optional_count: Optional[int] = None
        optional_ratio: Optional[float] = None
        attention_input_type: AttentionInputType = AttentionInputType.mixed
        typed: Optional[torch.Tensor] = cpp_metadata(dtype=torch.float32)
        indices: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
        handwritten: Optional[torch.Tensor] = None
        required: torch.Tensor = None
        nested: Nested = None
        optional_nested: Optional[Nested] = None
        private: PythonOnly = None
        python_only: Any = None
    """
    source = (
        ("from __future__ import annotations\n" if postponed else "")
        + "from dataclasses import dataclass\nfrom typing import Any, Optional\n"
        + "from enum import IntEnum\nclass AttentionInputType(IntEnum):\n    mixed = 0\n"
        + "@dataclass\nclass Nested:\n"
        + "    count: int = 0\n"
        + "@dataclass\nclass PythonOnly:\n    count: int = 0\n"
        + "@dataclass\nclass Example:\n"
        + textwrap.indent(textwrap.dedent(fields).strip(), "    ")
    )
    path = tmp_path / "schema.py"
    path.write_text(source)
    schema = generator.load_schemas([path], "Example", struct_names=("Nested",))
    struct = schema.structs["Example"]
    module = ModuleType("test_scalar_schema")
    module.cpp_metadata = cpp_schema.cpp_metadata
    module.torch = SimpleNamespace(
        Tensor=type("Tensor", (), {"__module__": "torch"}), float32=object(), int32=object()
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "schema.py", "exec", dont_inherit=True), module.__dict__)
    assert [f.name for f in struct.fields] == [
        "flag",
        "count",
        "ratio",
        "optional_flag",
        "optional_count",
        "optional_ratio",
        "attention_input_type",
        "typed",
        "indices",
        "handwritten",
        "required",
        "nested",
        "optional_nested",
    ]
    assert set(schema.structs) == {"Example", "Nested"}
    params = module.Example()
    assert params.flag is False
    assert params.count == 1099511627776
    assert params.ratio == 1e-6
    assert params.optional_count is None
    assert params.typed is None
    assert params.nested is None
    assert params.optional_nested is None
    assert params.attention_input_type is module.AttentionInputType.mixed
    rendered = generator.render_fields(schema, struct)
    assert "TRTLLM_FMHA_PARAM_FIELD(flag, bool)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(count, std::int64_t)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(ratio, double)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(optional_count, std::optional<std::int64_t>)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(attention_input_type, std::int64_t)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(nested, Nested)" in rendered
    assert "TRTLLM_FMHA_PARAM_FIELD(optional_nested, Nested)" in rendered
    accessors = generator.render_accessors(schema, struct)
    # Scalar widening must not widen a tensor's explicitly specified dtype.
    assert "float* getTyped() const" in accessors
    assert "std::int32_t* getIndices() const" in accessors
    assert "getHandwritten" not in accessors
    assert "getRequired" not in accessors


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

    @dataclasses.dataclass(slots=True)
    class NativeSparseRuntimeParams:
        sparse_kv_indices: object = None
        threshold_scale_factor_prefill: float = 0.0
        threshold_scale_factor_decode: float = 0.0

    class NativeForwardArgs:
        __slots__ = (
            "output",
            "output_sf",
            "kv_norm_eps",
            "update_kv_cache",
            "is_fused_qkv",
            "attention_window_size",
            "attention_input_type",
            "sparse_runtime_params",
            "sparse_backend_args",
        )

        def __init__(self):
            self.attention_window_size = 0
            self.sparse_runtime_params = NativeSparseRuntimeParams()
            self.sparse_backend_args = SimpleNamespace()

    class NativeParams:
        __slots__ = ("fwd", "output", "kv_pool", "beam_width", "is_cross", "mask_type")

        def __init__(self):
            self.fwd = NativeForwardArgs()

    internal = ModuleType("tensorrt_llm.bindings.internal")
    internal.thop = SimpleNamespace(FmhaParams=NativeParams)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.bindings.internal", internal)
    return NativeParams


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
        kv_pool="full-kv-pool",
        beam_width=4,
        is_cross=True,
    ).to_thop_params()

    # Keep the phase-local output distinct from the caller-facing full buffer;
    # native kernels consume the former and leave the latter and kv_pool unused.
    assert native.output == "phase-output"
    assert native.kv_pool == "full-kv-pool"
    assert native.fwd.output == "full-output"
    assert native.fwd.output_sf == "output-scale"
    assert native.fwd.kv_norm_eps == 1e-6
    assert native.fwd.update_kv_cache is True
    assert native.fwd.is_fused_qkv is False
    assert native.fwd.attention_window_size == 2048
    assert native.fwd.sparse_runtime_params.sparse_kv_indices == "sparse-indices"
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_prefill == 0.25
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_decode == 0.5
    # The mask type is derived from the public attention-mask representation.
    assert native.mask_type == AttentionMaskType.causal
    assert native.beam_width == 4


def test_static_config_plain_scalars_reach_native_holder(monkeypatch) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.interface import StaticAttentionConfig

    _stub_native_holder(monkeypatch)

    @dataclasses.dataclass(slots=True)
    class NativeStaticConfig:
        num_heads: int = 0
        q_scaling: float = 0.0
        remove_padding: bool = False
        use_kv_cache: bool = False

    sys.modules["tensorrt_llm.bindings.internal"].thop.StaticAttentionConfig = NativeStaticConfig
    native = StaticAttentionConfig(num_heads=8, q_scaling=0.125).to_thop_config()

    assert native.num_heads == 8
    assert native.q_scaling == 0.125
    assert native.remove_padding is True
    assert native.use_kv_cache is False


def test_nested_none_does_not_replace_native_value(monkeypatch) -> None:
    FmhaParams, ForwardArgs, SparseParams, Mask = _real_schema_classes()
    _stub_native_holder(monkeypatch)

    forward_args = ForwardArgs()
    assert forward_args.attention_mask is Mask.CAUSAL
    assert forward_args.sparse_runtime_params is None
    native = FmhaParams(fwd=forward_args).to_thop_params()

    # attention_window_size defaults to None on the Python side, which must leave
    # the value-initialized native field alone.
    assert native.fwd.attention_window_size == 0
    assert native.fwd.attention_input_type == 0
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_prefill == 0.0
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_decode == 0.0

    # The cached lowering plan contains field names, never values or None decisions.
    forward_args.attention_window_size = 128
    forward_args.sparse_runtime_params = SparseParams(threshold_scale_factor_prefill=0.5)
    updated = FmhaParams(fwd=forward_args).to_thop_params()
    assert updated.fwd.attention_window_size == 128
    assert updated.fwd.sparse_runtime_params.threshold_scale_factor_prefill == 0.5
    assert native.fwd.attention_window_size == 0
    assert native.fwd.sparse_runtime_params.threshold_scale_factor_prefill == 0.0
