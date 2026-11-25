# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's multi-rank launchers (`_v41_launch`): the ranks get the checkout on PYTHONPATH only when it is the
tensorrt_llm this process imports (CI runs an unbuilt checkout beside an installed wheel; the checkout first on the
path broke every rank's import), and every launcher's deadline plus the grace stays below the CI timeouts around it."""

import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

__extra_import_path__ = ["."]
import _v41_launch  # noqa: E402

pytestmark = pytest.mark.cpu_only

_HERE = Path(__file__).resolve().parent
_INNER_PYTEST_TIMEOUT_S = 2400  # tests/integration/defs/test_unittests.py


def _origin(path):
    return lambda name: NS(origin=str(path))


def test_an_installed_wheel_keeps_the_checkout_off_the_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setattr(
        _v41_launch.importlib.util,
        "find_spec",
        _origin(tmp_path / "site" / "tensorrt_llm" / "__init__.py"),
    )
    assert _v41_launch.rank_env("0,1")["PYTHONPATH"] == "/somewhere"


def test_an_in_tree_build_puts_the_checkout_first(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setattr(
        _v41_launch.importlib.util,
        "find_spec",
        _origin(_v41_launch.REPO_ROOT / "tensorrt_llm" / "__init__.py"),
    )
    env = _v41_launch.rank_env("0,1")
    assert env["PYTHONPATH"].split(":")[0] == str(_v41_launch.REPO_ROOT)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize(
    "launcher, entry",
    [
        (
            "ops/moe/_cute_dsl_megamoe_nvfp4_op_matrix.py",
            "ops/moe/test_modeling_v2_cute_dsl_megamoe_nvfp4.py",
        ),
        ("ops/moe/_moe_a2a_op_matrix.py", "ops/moe/test_modeling_v2_moe_a2a.py"),
        ("ops/moe/_moe_a2a_vs_stock.py", "ops/moe/test_modeling_v2_moe_a2a_vs_stock.py"),
        (
            "_megamoe_target_layer_matrix.py",
            "flash_nvfp4__sm_103__dep/test_modeling_v2_dep_megamoe_target_layer.py",
        ),
        ("_megamoe_placement_invariance.py", None),
    ],
)
def test_deadlines_end_before_the_ci_timeouts(launcher, entry):
    deadline = _v41_launch.launcher_deadline(_HERE / launcher) + _v41_launch.GRACE_S
    assert deadline < _INNER_PYTEST_TIMEOUT_S
    if entry is None:
        return
    # the test-db entries that run it (TIMEOUT in minutes)
    lists = (_v41_launch.REPO_ROOT / "tests/integration/test_lists/test-db").glob("*.yml")
    pattern = rf"deepseek_v41/{re.escape(entry)} TIMEOUT \((\d+)\)"
    minutes = [int(m) for p in lists for m in re.findall(pattern, p.read_text())]
    assert minutes and all(deadline < m * 60 for m in minutes), (launcher, minutes)
