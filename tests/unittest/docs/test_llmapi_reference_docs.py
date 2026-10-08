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

"""Tests for generated LLM API reference pages."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


def _load_docs_helper(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "pygit2", ModuleType("pygit2"))
    helper_path = Path(__file__).resolve().parents[3] / "docs/source/helper.py"
    spec = importlib.util.spec_from_file_location("llmapi_docs_helper", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_generate_llmapi_excludes_pydantic_members_only(tmp_path, monkeypatch) -> None:
    helper = _load_docs_helper(monkeypatch)
    llmapi_init = Path(__file__).resolve().parents[3] / "tensorrt_llm/llmapi/__init__.py"
    public_symbols = helper.extract_all_and_eval(llmapi_init)["__all__"]
    assert "TorchLlmArgs" in public_symbols
    assert "LLM" in public_symbols
    assert helper.is_pydantic_model("TorchLlmArgs")
    assert not helper.is_pydantic_model("LLM")

    source_dir = tmp_path / "docs/source"
    source_dir.mkdir(parents=True)
    monkeypatch.setattr(helper, "__file__", str(source_dir / "helper.py"))
    monkeypatch.setattr(
        helper,
        "extract_all_and_eval",
        lambda _path: {"__all__": ["TorchLlmArgs", "LLM"]},
    )

    helper.generate_llmapi()

    reference_dir = source_dir / "llm-api/reference"
    pydantic_rst = (reference_dir / "TorchLlmArgs.rst").read_text()
    non_pydantic_rst = (reference_dir / "LLM.rst").read_text()
    assert ":exclude-members:" in pydantic_rst
    assert ":exclude-members:" not in non_pydantic_rst
