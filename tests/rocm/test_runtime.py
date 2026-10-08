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

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

import tensorrt_llm
from tensorrt_llm._backend import select_backend
from tensorrt_llm.rocm.runtime import architecture_name, diagnostics, resolve_device, resolve_dtype
from tensorrt_llm.rocm.sampling import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "choice,hip,expected",
    [
        ("auto", None, "cuda"),
        ("auto", "7.1", "rocm"),
        ("rocm", None, "rocm"),
        ("cuda", None, "cuda"),
    ],
)
def test_backend_selection(monkeypatch, choice, hip, expected) -> None:
    monkeypatch.setenv("TRTLLM_BACKEND", choice)
    assert select_backend(hip) == expected


def test_invalid_backend(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_BACKEND", "hipify-everything")
    with pytest.raises(ValueError, match="must be"):
        select_backend(None)
    monkeypatch.setenv("TRTLLM_BACKEND", "cuda")
    with pytest.raises(ValueError, match="HIP"):
        select_backend("7.1")


def test_rocm_import_has_no_native_nvidia_dependencies() -> None:
    script = """
import sys
import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
assert tensorrt_llm._BACKEND == 'rocm'
assert LLM.__module__ == 'tensorrt_llm.rocm.llm'
assert not any(name.startswith(('tensorrt_llm.bindings', 'tensorrt', 'mpi4py', 'flashinfer'))
               for name in sys.modules if name != 'tensorrt_llm' and not name.startswith('tensorrt_llm.'))
assert 'tensorrt_llm.bindings' not in sys.modules
from tensorrt_llm.llmapi import LLM as api_llm
from tensorrt_llm._torch import LLM as torch_llm
assert api_llm is LLM and torch_llm is LLM
assert 'tensorrt_llm.bindings' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "TRTLLM_BACKEND": "rocm"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_is_explicit(monkeypatch) -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="no HIP"):
        resolve_device("cuda:0")
    with pytest.raises(ValueError, match="supports"):
        resolve_device("mps")


def test_rdna4_identity_and_architecture_spoofing(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    properties = SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-", name="RDNA4 fixture")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    assert architecture_name(properties.gcnArchName) == "gfx1201"
    properties.gcnArchName = "gfx1100"
    with pytest.raises(RuntimeError, match="Expected RDNA4"):
        resolve_device("cuda:0")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
    with pytest.raises(RuntimeError, match="Remove HSA_OVERRIDE"):
        resolve_device("cuda:0")


@pytest.mark.parametrize("value", ["float8", "nvfp4", torch.float64])
def test_invalid_dtype(value) -> None:
    with pytest.raises(ValueError):
        resolve_dtype(value, torch.device("cpu"))


def test_diagnostics_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = diagnostics()
    assert not report["ready"]
    assert report["devices"] == []
    assert report["supported_architectures"] == ["gfx1200", "gfx1201"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_tokens": 0},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": -1},
        {"seed": -2},
        {"max_tokens": 2, "min_tokens": 3},
        {"stop": ""},
        {"n": 2, "temperature": 0},
        {"beam_width": 2, "n": 3},
        {"cuda_graph_config": {}},
    ],
)
def test_sampling_rejects_invalid_and_unsupported_fields(arguments) -> None:
    with pytest.raises(ValidationError):
        SamplingParams(**arguments)


def test_sampling_allows_beams() -> None:
    assert SamplingParams(temperature=0, n=2, beam_width=2).n == 2


def test_static_exports_preserve_the_lazy_public_namespace() -> None:
    tree = ast.parse(Path(tensorrt_llm.__file__).read_text())
    type_checking = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )
    exports = {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in type_checking.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert set(tensorrt_llm._LAZY_ATTRS) <= exports


def test_strict_config_compatibility_export() -> None:
    from tensorrt_llm._config import StrictBaseModel
    from tensorrt_llm.llmapi.utils import StrictBaseModel as compatibility_export

    assert compatibility_export is StrictBaseModel
    with pytest.raises(ValidationError):
        compatibility_export(unknown_option=True)
