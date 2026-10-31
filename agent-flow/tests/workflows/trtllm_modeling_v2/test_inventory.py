# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The surface survey: what the build supplies, measured on the host."""

from __future__ import annotations

import subprocess
import sys

from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import (
    PROBE_SOURCE,
    RECEIPT_PASSED,
    format_surface,
    parse_inventory,
)

# ------------------------------------------------- the probe, actually run


def _entry_source(impl, archs):
    arch_set = ", ".join(f"Arch.{a}" for a in archs)
    cell_list = "".join(f"Cell(why='c{i}', spec={{}}), " for i in range(2))
    return (
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper\n\n\n"
        "class Entry(OpWrapper):\n"
        '    """Mentions torch.ops.trtllm.neighbour in prose, which is not the op."""\n\n'
        f"    ARCHS = frozenset({{{arch_set}}})\n"
        f"    CELLS = ({cell_list})\n\n"
        "    def __call__(self, x):\n"
        f"        return {impl}(x)\n\n"
        "    def reference(self, x):\n"
        "        return x\n"
    )


def _fake_repo(tmp_path, entries, model_ops=()):
    """A repo tree with catalog entries, and nothing else.

    ``entries`` are ``(category/name.py, impl, archs)`` written under
    ``catalog/``; ``model_ops`` are ``(family/target, category/name.py, impl,
    archs)`` kept in a target's ``ops/``.
    """
    catalog = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
    catalog.mkdir(parents=True)
    (catalog / "_op.py").write_text("class OpWrapper:\n    pass\n", encoding="utf-8")
    for path, impl, archs in entries:
        py = catalog / path
        py.parent.mkdir(parents=True, exist_ok=True)
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
    models = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
    for scope, path, impl, archs in model_ops:
        py = models / scope / "ops" / path
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _run_probe(repo, arch="sm_103", prelude=""):
    proc = subprocess.run(
        [sys.executable, "-c", prelude + PROBE_SOURCE, str(repo), arch],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


_INLINE_PASS = ("SM_103",)


def test_another_targets_ops_are_promotable_not_spendable(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)],
        model_ops=[
            ("fam/ckpt__sm_103__tp1", "moe/router.py", "torch.ops.trtllm.router", _INLINE_PASS)
        ],
    )
    inventory = _run_probe(repo)
    (op,) = inventory.promotable
    assert (op.scope, op.path, op.impl) == (
        "fam/ckpt__sm_103__tp1",
        "moe/router.py",
        "torch.ops.trtllm.router",
    )
    assert op.receipt == RECEIPT_PASSED
    assert op not in inventory.spendable and op not in inventory.uncertified
    assert "router" not in inventory.unwrapped_ops
    assert "`fam/ckpt__sm_103__tp1/ops/moe/router`" in format_surface(inventory)
