# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The import scope of the ``UNCERTIFIED_GENERIC_CALLS`` check (``_kimi_k3_generic_calls``): which modeling_v2
imports count as entries a declaring module may call. Each target's test directory runs the check on that target's
modules."""

from __future__ import annotations

import ast

import _kimi_k3_generic_calls as gc
import pytest

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "module, accepted",
    [
        ("catalog.attention.entry", True),
        ("models.family.ops.attention.entry", True),
        ("models.family.target.ops.attention.entry", True),
        ("models.family.sibling.ops.attention.entry", False),
        ("models.other.ops.attention.entry", False),
        ("models.family.ops_extra.attention.entry", False),
        ("models.family.helper", False),
        ("models.family.kernels.kernel", False),
    ],
)
def test_generic_call_import_scope(module, accepted):
    tree = ast.parse(f"from {gc.PACKAGE}.{module} import op\n")
    assert (gc.stock_imports(tree, "family", "target") == set()) == accepted


def test_family_level_scope_excludes_target_ops():
    tree = ast.parse(f"from {gc.PACKAGE}.models.family.target.ops.attention.entry import op\n")
    assert gc.stock_imports(tree, "family", None) != set()
