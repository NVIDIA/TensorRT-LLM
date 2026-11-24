# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The private MNNVL helpers of ``tensorrt_llm._torch.distributed.ops`` that the Kimi K3.1 family calls, pinned.

The family's ops and kernels build their multicast workspaces (the MoE front and tail, the DSA key shard, the
sequence-parallel rows, the attention-DP exchanges and all-to-all) with the engine's private helpers, which stay
inside those op and kernel modules. A leading underscore promises no stability, and a rename or a signature change
upstream would surface only when a multi-GPU workspace is first built. This test parses every family module
(nothing of the family is imported), checks that each private name it takes from the module is pinned here, and that
each pinned helper still exists with the parameters the family passes.
"""

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[6]
_FAM = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31"
_MODULE = "tensorrt_llm._torch.distributed.ops"

#: Private function -> its leading parameters, in the order the family passes them.
PINNED_FUNCTIONS = {
    "_get_mnnvl_workspace_comm": ("mapping",),
    "_mnnvl_device_index": ("mapping",),
    "_make_mnnvl_mcast_buffer": ("comm", "workspace_size_bytes", "mapping", "use_fabric_handle"),
    "_mnnvl_workspace_all_succeeded": ("comm", "local_success"),
    "_mnnvl_one_shot_threshold_bytes": (),
}
#: Private module constants the family reads.
PINNED_CONSTANTS = {"_MNNVL_ONE_SHOT_THRESHOLD_BYTES": int}


def _private_uses() -> dict:
    """Private name of ``_MODULE`` -> the family files that use it (``from ... import _x`` or ``<alias>._x``)."""
    uses = {}
    for path in sorted(_FAM.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = set()
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == _MODULE:
                    names += [a.name for a in node.names if a.name.startswith("_")]
                elif node.module == _MODULE.rsplit(".", 1)[0]:
                    aliases |= {a.asname or a.name for a in node.names if a.name == "ops"}
            elif isinstance(node, ast.Import):
                aliases |= {a.asname for a in node.names if a.name == _MODULE and a.asname}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr.startswith("_")
            ):
                names.append(node.attr)
        for name in names:
            uses.setdefault(name, set()).add(str(path.relative_to(_FAM)))
    return uses


def test_family_private_uses_are_pinned():
    uses = _private_uses()
    assert uses, "the family no longer uses any private MNNVL helper: drop this test"
    unpinned = {
        name: sorted(files)
        for name, files in uses.items()
        if name not in PINNED_FUNCTIONS and name not in PINNED_CONSTANTS
    }
    assert not unpinned, (
        f"private {_MODULE} names used by the family but not pinned here: {unpinned}"
    )


def test_pinned_helpers_exist_with_their_parameters():
    from tensorrt_llm._torch.distributed import ops

    for name, params in PINNED_FUNCTIONS.items():
        fn = getattr(ops, name, None)
        assert callable(fn), f"{_MODULE}.{name} is gone"
        got = tuple(inspect.signature(fn).parameters)
        assert got[: len(params)] == params, f"{_MODULE}.{name}{got}: the family passes {params}"
    for name, kind in PINNED_CONSTANTS.items():
        assert isinstance(getattr(ops, name, None), kind), (
            f"{_MODULE}.{name} is not a {kind.__name__}"
        )
