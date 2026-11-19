# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's MLA decode tile helpers are the Blackwell MLA decode kernels' own, byte for byte.

``kernels/k3r_mla_decode/mla_helpers.py`` is a copy of
``tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py`` (provenance in the family
``kernels/SOURCE.json``); the family's FP8 MLA decode kernel and the ULL helix tail import it.
Parsing only: no GPU, no imports of either module.
"""

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[8]
_FAMILY = REPO / "tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl"
_COPY = _FAMILY / "kernels/k3r_mla_decode/mla_helpers.py"
_V1 = REPO / "tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py"


def test_copy_is_the_v1_module():
    assert _COPY.read_bytes() == _V1.read_bytes()


def test_copy_imports_only_cutlass():
    tree = ast.parse(_COPY.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"cutlass"}


def test_copy_is_recorded_in_source_json():
    entry = json.loads((_FAMILY / "kernels/SOURCE.json").read_text())["files"][
        "kernels/k3r_mla_decode/mla_helpers.py"
    ]
    assert entry["source"].startswith(
        "tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py@"
    )
    assert entry["import_edits"] == [] and entry["local_edits"] == []
