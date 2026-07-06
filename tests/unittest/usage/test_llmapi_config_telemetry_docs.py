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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_generator() -> ModuleType:
    module_path = _repo_root() / "docs/source/_ext/llmapi_config_telemetry.py"
    spec = importlib.util.spec_from_file_location("llmapi_config_telemetry", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # exec_module needs the module registered (frozen dataclasses resolve their
    # owning module via sys.modules), but restore the prior state afterwards so
    # the loader does not leak its temporary module on success or failure.
    sentinel = object()
    previous = sys.modules.get(spec.name, sentinel)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is sentinel:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def _load_manifest_generator() -> ModuleType:
    module_path = _repo_root() / "scripts/generate_llm_args_golden_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_llm_args_golden_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_manifest() -> dict[str, list[dict[str, object]]]:
    # Keep both mapping levels unsorted so the test exercises recursive key sorting.
    return {
        "TrtLlmArgs": [],
        "TorchLlmArgs": [
            {
                "path": "flag",
                "kind": "value",
                "converter": "",
                "annotation": "<class 'bool'>",
                "allowed_values": [],
            }
        ],
    }


def test_manifest_generator_write_is_canonical_and_idempotent(tmp_path, monkeypatch):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"
    canonical = json.dumps(_sample_manifest(), indent=2, sort_keys=True) + "\n"

    assert generator._write_manifest(manifest_path)
    assert manifest_path.read_text() == canonical
    assert not generator._write_manifest(manifest_path)
    assert generator._check_manifest(manifest_path)

    manifest_path.write_bytes(canonical.replace("\n", "\r\n").encode())
    assert not generator._check_manifest(manifest_path)
    assert generator._write_manifest(manifest_path)
    assert manifest_path.read_bytes() == canonical.encode()


def test_manifest_generator_check_reports_unified_diff(tmp_path, monkeypatch, capfd):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"stale": true}\n')

    assert not generator._check_manifest(manifest_path)
    stderr = capfd.readouterr().err
    assert f"--- {manifest_path} (committed)" in stderr
    assert f"+++ {manifest_path} (generated)" in stderr
    assert '-{"stale": true}' in stderr
    assert '+  "TorchLlmArgs": [' in stderr
    assert manifest_path.read_text() == '{"stale": true}\n'


def test_manifest_generator_preserves_target_when_generation_fails(tmp_path, monkeypatch):
    import pytest

    generator = _load_manifest_generator()
    manifest_path = tmp_path / "manifest.json"
    old_content = '{"old": true}\n'
    manifest_path.write_text(old_content)

    def _fail_generation():
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(generator, "golden_manifest", _fail_generation)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        generator._write_manifest(manifest_path)

    assert manifest_path.read_text() == old_content
    assert list(tmp_path.iterdir()) == [manifest_path]


def test_manifest_generator_preserves_target_when_replace_fails(tmp_path, monkeypatch):
    import pytest

    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"
    old_content = '{"old": true}\n'
    manifest_path.write_text(old_content)

    def _fail_replace(*_args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(generator.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        generator._write_manifest(manifest_path)

    assert manifest_path.read_text() == old_content
    assert list(tmp_path.iterdir()) == [manifest_path]


def test_manifest_generator_main_reports_file_io_failure(monkeypatch, capfd):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "_render_manifest", lambda: "{}\n")

    def _fail_write(*_args, **_kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(generator, "_write_manifest", _fail_write)
    assert generator.main([]) == 2
    assert "synthetic write failure" in capfd.readouterr().err


def test_manifest_generator_main_propagates_generation_failure(monkeypatch):
    import pytest

    generator = _load_manifest_generator()

    def _fail_generation():
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(generator, "_render_manifest", _fail_generation)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        generator.main([])


def test_build_capture_manifest_matches_committed_golden():
    """The CI privacy gate (closes TRTLLM-12872).

    Regenerate in-memory and diff against the committed golden; any drift
    ('field X now phones home') must be a deliberate, privacy-reviewed golden
    update committed in the same change.
    """
    generator = _load_manifest_generator()
    assert generator.main(["--check"]) == 0


def test_load_generator_does_not_leak_sys_modules():
    """_load_generator must not leak its temporary module into sys.modules.

    The loader needs the module registered while exec_module runs (frozen
    dataclasses resolve their module via sys.modules), but it must restore the
    prior state on both success and failure.
    """
    name = "llmapi_config_telemetry"
    sys.modules.pop(name, None)

    _load_generator()
    assert name not in sys.modules

    import importlib.util as _util

    real_spec_from_file_location = _util.spec_from_file_location

    def _boom(*args, **kwargs):
        spec = real_spec_from_file_location(*args, **kwargs)

        class _BoomLoader:
            name = spec.name

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise RuntimeError("synthetic load failure")

        spec.loader = _BoomLoader()
        return spec

    _util.spec_from_file_location = _boom
    try:
        try:
            _load_generator()
        except RuntimeError:
            pass
        assert name not in sys.modules
    finally:
        _util.spec_from_file_location = real_spec_from_file_location


def test_domain_values_cover_literal_and_enum():
    from enum import Enum
    from typing import Literal, Optional

    from tensorrt_llm.usage import llmapi_config as rc

    class _Color(str, Enum):
        RED = "red"
        BLUE = "blue"

    assert rc._domain_values(Optional[Literal["a", "b"]], {}) == ["a", "b"]
    assert rc._domain_values(Optional[_Color], {}) == ["red", "blue"]
    assert rc._domain_values(int, {"converter": "allowlist", "allowed_values": ["x", "y"]}) == [
        "x",
        "y",
    ]


def _small_models():
    from enum import Enum

    from pydantic import BaseModel

    from tensorrt_llm.llmapi.llm_args import Field

    class Mode(str, Enum):
        A = "a"
        B = "b"

    class Nested(BaseModel):
        n_int: int = 0
        n_str: str = "secret"  # bare str -> OUT
        n_secret: int = Field(default=0, telemetry=False)  # honored exclude

    class Root(BaseModel):
        flag: bool = True
        mode: Mode = Mode.A
        sizes: list[int] = Field(default_factory=list)
        path_like: str = "x"  # bare str -> OUT
        allow: str = Field(
            default="a",
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["a", "b"],
            },
        )
        nested: Nested | None = None
        loose: object | str | None = None  # no BaseModel arm -> not recursed

    return Root


def test_build_capture_manifest_selection_and_recursion():
    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    Root = _small_models()
    paths = {e.path for e in build_capture_manifest(Root)}
    assert paths == {"flag", "mode", "sizes", "allow", "nested.n_int"}
    assert "path_like" not in paths  # bare str OUT
    assert "nested.n_str" not in paths  # bare str OUT
    assert "nested.n_secret" not in paths  # honored telemetry=False
    assert not any(p.startswith("loose") for p in paths)  # loose has no model arm


def test_build_capture_manifest_kinds_and_domains():
    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    Root = _small_models()
    by_path = {e.path: e for e in build_capture_manifest(Root)}
    assert by_path["flag"].kind == "value"
    assert by_path["mode"].kind == "categorical"
    assert list(by_path["mode"].allowed_values) == ["a", "b"]  # Enum domain
    assert by_path["allow"].kind == "categorical"
    assert by_path["allow"].converter == "allowlist"
    assert list(by_path["allow"].allowed_values) == ["a", "b"]


def test_renderer_emits_table_from_committed_golden(tmp_path):
    generator = _load_generator()
    out = tmp_path / "telemetry.md"
    generator.generate_telemetry_reference(_repo_root(), out)
    text = out.read_text()
    assert "## LLM API Configuration Fields" in text
    assert "explicitly marked" not in text  # opt-in prose must be gone
    assert "`backend`" in text  # a known captured key renders


def test_build_capture_manifest_fails_on_divergent_kind_across_union_arms():
    from typing import Literal, Union

    import pytest
    from pydantic import BaseModel

    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    class ArmA(BaseModel):
        tag: Literal["a"] = "a"
        shared: int = 0  # kind=value

    class ArmB(BaseModel):
        tag: Literal["b"] = "b"
        shared: Literal["x", "y"] = "x"  # kind=categorical -> conflict on "arm.shared"

    class Root(BaseModel):
        arm: Union[ArmA, ArmB] = ArmA()

    with pytest.raises(ValueError, match="conflicting kinds"):
        build_capture_manifest(Root)


def test_build_capture_manifest_cycle_guard_terminates_on_self_reference():
    from typing import Optional

    from pydantic import BaseModel

    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    class Node(BaseModel):
        value: int = 0
        child: Optional["Node"] = None

    Node.model_rebuild()
    entries = build_capture_manifest(Node)  # must TERMINATE (cycle guard), not infinite-recurse
    paths = {e.path for e in entries}
    assert "value" in paths
    assert "child.value" not in paths
