# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the Blackwell MLA tile helpers are shared by ULL and stock decode."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[8]
_FAMILY = REPO / "tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl"
_SHARED = REPO / "tensorrt_llm/_torch/kernels/blackwell/mla_helpers.py"
_LEGACY = REPO / "tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py"
_SHARED_IMPORT = "tensorrt_llm._torch.kernels.blackwell.mla_helpers"


def test_stock_and_ull_import_shared_helpers():
    for path in (_LEGACY, _FAMILY / "kernels/k3r_helix/k3_helix_tail_ull.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == _SHARED_IMPORT
            and any(alias.name == "LOG2_E" for alias in node.names)
            for node in tree.body
        ), path


def test_shared_helpers_import_only_cutlass():
    tree = ast.parse(_SHARED.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"cutlass"}


def test_shared_helpers_are_recorded():
    entry = json.loads((_FAMILY / "kernels/SOURCE.json").read_text())["shared_files"][
        _SHARED.relative_to(REPO).as_posix()
    ]
    assert entry["source"].startswith(
        "tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py@"
    )
    assert entry["sha256"] == hashlib.sha256(_SHARED.read_bytes()).hexdigest()
