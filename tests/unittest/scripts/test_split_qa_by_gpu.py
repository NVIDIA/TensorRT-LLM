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
"""CPU tests for static GPU bucketing, covered by the l0_cpu unittest/scripts list."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def splitter() -> ModuleType:
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    modules = {}
    for name in ("check_test_list", "split_qa_by_gpu"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with pytest.MonkeyPatch.context() as patch:
            for dependency, value in modules.items():
                patch.setitem(sys.modules, dependency, value)
            spec.loader.exec_module(module)
        modules[name] = module
    return modules["split_qa_by_gpu"]


def _split(splitter: ModuleType, tmp_path: Path, source: str, entries: str) -> dict[str, list[str]]:
    (tmp_path / "test_cases.py").write_text(dedent(source), encoding="utf-8")
    test_list = tmp_path / "qa.txt"
    test_list.write_text(dedent(entries), encoding="utf-8")
    counts = splitter.split_test_list(test_list, tmp_path / "out", tmp_path)
    result = {}
    for path, count in counts.items():
        text = path.read_text()
        assert text.endswith("\n") and not text.endswith("\n\n")
        lines = [line for line in text.splitlines() if line and not line.startswith("#")]
        assert len(lines) == count
        result[path.name] = lines
    return result


def test_independent_device_and_mpi_bounds(splitter: ModuleType, tmp_path: Path) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        raise RuntimeError('test sources must never be imported')
        def test_one(): pass
        @pytest.mark.skip_less_device(4)
        def test_four(): pass
        @pytest.mark.skip_less_device(6)
        def test_six(): pass
        @pytest.mark.skip_less_device(4)
        @pytest.mark.skip_less_mpi_world_size(16)
        def test_mpi(): pass
        @pytest.mark.skip_less_device(8)
        @pytest.mark.skip_less_mpi_world_size(2)
        def test_local(): pass
    """,
        """
        # Input comments are not cases.
        test_cases.py::test_one
        test_cases.py::test_four TIMEOUT (180)
        test_cases.py::test_six XFAIL (known issue)
        test_cases.py::test_mpi
        test_cases.py::test_local
    """,
    )
    assert result == {
        "qa_1to4gpu.txt": ["test_cases.py::test_one", "test_cases.py::test_four TIMEOUT (180)"],
        "qa_5to8gpu.txt": [
            "test_cases.py::test_six XFAIL (known issue)",
            "test_cases.py::test_local",
        ],
        "qa_9to16gpu.txt": ["test_cases.py::test_mpi"],
    }


def test_closest_markers_and_parameter_ids(splitter: ModuleType, tmp_path: Path) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        import pytest
        @pytest.mark.skip_less_mpi_world_size(8)
        class TestCases:
            @pytest.mark.parametrize('x', [
                pytest.param(4, marks=pytest.mark.skip_less_mpi_world_size(4)),
                pytest.param(16, marks=pytest.mark.skip_less_mpi_world_size(16)),
            ], ids=['tp4-with-hyphens', 'tp16'])
            @pytest.mark.parametrize('mode', [False, True])
            def test_param(self, x, mode): pass
            @pytest.mark.skip_less_mpi_world_size(4)
            @pytest.mark.parametrize('x', [pytest.param(0, marks=pytest.mark.skip_less_mpi_world_size(16))])
            def test_function_wins(self, x): pass
            @pytest.mark.skip_less_mpi_world_size(16)
            @pytest.mark.skip_less_mpi_world_size(4)
            def test_nearest(self): pass
    """,
        """
        test_cases.py::TestCases::test_param[False-tp4-with-hyphens]
        test_cases.py::TestCases::test_param[True-tp16]
        test_cases.py::TestCases::test_function_wins[0]
        test_cases.py::TestCases::test_nearest
    """,
    )
    assert len(result["qa_1to4gpu.txt"]) == 3
    assert result["qa_5to8gpu.txt"] == []
    assert result["qa_9to16gpu.txt"] == ["test_cases.py::TestCases::test_param[True-tp16]"]


def test_module_marks_and_local_inheritance(splitter: ModuleType, tmp_path: Path) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        import pytest
        GPU_COUNT = 8
        MARK = pytest.mark.skip_less_device(GPU_COUNT)
        pytestmark = [MARK]
        class Base:
            def test_inherited(self): pass
        @pytest.mark.skip_less_device(4)
        class TestChild(Base): pass
        def test_module(): pass
    """,
        """
        test_cases.py::TestChild::test_inherited
        test_cases.py::test_module
    """,
    )
    assert result["qa_1to4gpu.txt"] == ["test_cases.py::TestChild::test_inherited"]
    assert result["qa_5to8gpu.txt"] == ["test_cases.py::test_module"]


def test_class_and_function_selectors_are_split_when_needed(
    splitter: ModuleType, tmp_path: Path
) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        import pytest
        class TestCases:
            def test_one(self): pass
            @pytest.mark.parametrize('x', [
                pytest.param(4, marks=pytest.mark.skip_less_device(4), id='four'),
                pytest.param(8, marks=pytest.mark.skip_less_device(8), id='eight'),
            ])
            def test_param(self, x): pass
    """,
        "test_cases.py::TestCases TIMEOUT (60)\n",
    )
    assert result["qa_1to4gpu.txt"] == [
        "test_cases.py::TestCases::test_one TIMEOUT (60)",
        "test_cases.py::TestCases::test_param[four] TIMEOUT (60)",
    ]
    assert result["qa_5to8gpu.txt"] == ["test_cases.py::TestCases::test_param[eight] TIMEOUT (60)"]


def test_callable_ids_with_uniform_resource_marks(splitter: ModuleType, tmp_path: Path) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        import pytest
        @pytest.mark.parametrize('x', [
            pytest.param(object(), marks=pytest.mark.skip_less_device(8)),
            pytest.param(object(), marks=pytest.mark.skip_less_device(8)),
        ], ids=lambda x: x.description)
        def test_dynamic(x): pass
    """,
        "test_cases.py::test_dynamic[generated-id]\n",
    )
    assert result["qa_5to8gpu.txt"] == ["test_cases.py::test_dynamic[generated-id]"]


@pytest.mark.parametrize(
    "source,selector,message",
    [
        (
            "@pytest.mark.skip_less_device(gpu_count())\ndef test_bad(): pass",
            "test_bad",
            "malformed",
        ),
        (
            "@pytest.mark.skip_less_mpi_world_size(32)\ndef test_bad(): pass",
            "test_bad",
            "maximum of 16",
        ),
        ("def test_good(): pass", "test_missing", "not found"),
        (
            """
        @pytest.mark.parametrize('x', [
            pytest.param(0, marks=pytest.mark.skip_less_device(4)),
            pytest.param(1, marks=pytest.mark.skip_less_device(8)),
        ], ids=lambda x: str(x))
        def test_bad(x): pass
    """,
            "test_bad[0]",
            "cannot resolve parameter IDs",
        ),
    ],
)
def test_bad_markers_and_selectors_are_skipped(
    splitter: ModuleType,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    source: str,
    selector: str,
    message: str,
) -> None:
    result = _split(
        splitter,
        tmp_path,
        dedent(source) + "\ndef test_good(): pass\n",
        f"test_cases.py::test_good\ntest_cases.py::{selector}\ntest_cases.py::test_good\n",
    )
    assert result["qa_1to4gpu.txt"] == ["test_cases.py::test_good"] * 2
    assert result["qa_5to8gpu.txt"] == []
    assert result["qa_9to16gpu.txt"] == []
    skipped = [
        json.loads(line) for line in (tmp_path / "out/qa_skipped.jsonl").read_text().splitlines()
    ]
    assert len(skipped) == 1
    assert skipped[0]["line"] == 2
    assert skipped[0]["entry"] == f"test_cases.py::{selector}"
    assert message in skipped[0]["reason"]
    assert f"skipped test_cases.py::{selector}" in capfd.readouterr().err


def test_unreadable_test_source_aborts_split(
    splitter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "test_cases.py"
    source.write_text("def test_good(): pass\n", encoding="utf-8")
    test_list = tmp_path / "qa.txt"
    test_list.write_text("test_cases.py::test_good\n", encoding="utf-8")
    read_text = Path.read_text

    def fail_source_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == source:
            raise PermissionError("test source is unreadable")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_source_read)
    with pytest.raises(PermissionError, match="test source is unreadable"):
        splitter.split_test_list(test_list, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_inherited_class_mark_order_and_class_pytestmark(
    splitter: ModuleType, tmp_path: Path
) -> None:
    result = _split(
        splitter,
        tmp_path,
        """
        import pytest
        @pytest.mark.skip_less_device(8)
        class Base:
            def test_inherited(self): pass
        @pytest.mark.skip_less_device(4)
        class TestChild(Base):
            @pytest.mark.skip_less_device(4)
            def test_override(self): pass
        class TestOwnMarks:
            pytestmark = pytest.mark.skip_less_mpi_world_size(16)
            def test_sixteen(self): pass
    """,
        """
        test_cases.py::TestChild::test_inherited
        test_cases.py::TestChild::test_override
        test_cases.py::TestOwnMarks::test_sixteen
    """,
    )
    assert result["qa_1to4gpu.txt"] == ["test_cases.py::TestChild::test_override"]
    assert result["qa_5to8gpu.txt"] == ["test_cases.py::TestChild::test_inherited"]
    assert result["qa_9to16gpu.txt"] == ["test_cases.py::TestOwnMarks::test_sixteen"]
