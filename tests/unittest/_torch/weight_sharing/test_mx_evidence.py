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
"""Unit tests for the shared ModelExpress transfer-evidence rules (`mx_evidence.py`).

The module lives next to the integration harness and is standard-library only,
so it is loaded here by file path rather than through the `defs` package.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "model_express"
    / "mx_evidence.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mx_evidence_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, _MODULE_PATH
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their module through
    # `sys.modules`; register before executing or `@dataclass` raises.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evidence() -> ModuleType:
    return _load_module()


def _completion(rank: int, tensors: int = 291, size_gb: float = 2.2) -> str:
    return (
        f"[Worker {rank}] INFO modelexpress.engines.trtllm: "
        f"RDMA transfer complete: {tensors} tensors, {size_gb} GB\n"
    )


def _good_log(tp_size: int) -> str:
    return "INFO some TRT-LLM chatter\n" + "".join(_completion(rank) for rank in range(tp_size))


def _import_contract_violations(source: str) -> tuple[list[str], list[int]]:
    """Return `(third_party_modules, relative_import_lines)` found in `source`.

    Every absolute import must resolve to a standard-library top-level module,
    and relative imports are rejected because the worker imports the module as
    a bare top-level `mx_evidence`, where they cannot resolve.
    """
    imported: set[str] = set()
    relative_lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_lines.append(node.lineno)
            elif node.module:
                imported.add(node.module.partition(".")[0])
    return sorted(imported - sys.stdlib_module_names), relative_lines


def test_module_is_standard_library_only() -> None:
    third_party, relative_lines = _import_contract_violations(
        _MODULE_PATH.read_text(encoding="utf-8")
    )
    assert not third_party, f"mx_evidence.py must stay stdlib-only, found {third_party}"
    assert not relative_lines, f"mx_evidence.py must not use relative imports: {relative_lines}"


def test_import_contract_check_catches_third_party_and_relative_imports() -> None:
    source = (
        "from __future__ import annotations\n"
        "import re\n"
        "import numpy as np\n"
        "from arbitrary_third_party.sub import helper\n"
        "from . import sibling\n"
        "def f():\n"
        "    import torch.nn\n"
    )
    third_party, relative_lines = _import_contract_violations(source)
    assert third_party == ["arbitrary_third_party", "numpy", "torch"]
    assert relative_lines == [5]


def test_complete_evidence_has_no_problems(evidence: ModuleType) -> None:
    assert evidence.check_receiver_log(_good_log(2), tp_size=2) == []


def test_missing_rank_is_reported(evidence: ModuleType) -> None:
    problems = evidence.check_receiver_log(_good_log(1), tp_size=2)
    assert problems == ["Expected RDMA transfer completion for ranks [0, 1], got [0]"]


def test_no_completion_at_all_is_reported(evidence: ModuleType) -> None:
    problems = evidence.check_receiver_log("MX loading unavailable: missing model\n", tp_size=1)
    assert problems == ["Expected RDMA transfer completion for ranks [0], got []"]


@pytest.mark.parametrize(
    ("line", "marker"),
    [
        ("... Falling back to DISK ...", "falling back to disk"),
        (
            "MX loading unavailable: missing model_config; "
            "falling back to native Hugging Face checkpoint loading.",
            "falling back to native hugging face checkpoint loading",
        ),
        ("MX P2P unavailable (no source)", "mx p2p unavailable"),
    ],
)
def test_failure_marker_is_reported(evidence: ModuleType, line: str, marker: str) -> None:
    problems = evidence.check_receiver_log(_good_log(1) + line + "\n", tp_size=1)
    assert problems == [f"MX receiver log contains failure marker {marker!r}"]


def test_duplicate_completion_is_reported(evidence: ModuleType) -> None:
    problems = evidence.check_receiver_log(_good_log(1) + _completion(0), tp_size=1)
    assert len(problems) == 1
    assert problems[0].startswith("Expected one RDMA transfer completion for rank 0, got [")


@pytest.mark.parametrize(
    ("tensors", "size_gb"), [(0, 2.2), (291, 0.0)], ids=["no-tensors", "no-bytes"]
)
def test_empty_transfer_is_reported(evidence: ModuleType, tensors: int, size_gb: float) -> None:
    problems = evidence.check_receiver_log(_completion(0, tensors, size_gb), tp_size=1)
    assert problems == [
        f"MX receiver rank 0 reported an empty transfer: {tensors} tensors, {size_gb} GB"
    ]


def test_completions_are_parsed_case_insensitively_and_json_friendly(
    evidence: ModuleType,
) -> None:
    text = _completion(1, tensors=3, size_gb=0.75).upper() + _completion(0)
    transfers = evidence.rdma_transfers_by_rank(text)
    assert sorted(transfers) == [0, 1]
    assert [record.to_dict() for record in transfers[1]] == [
        {"rank": 1, "tensor_count": 3, "size_gb": 0.75}
    ]
    assert evidence.find_failure_markers(text) == ()
