# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static source checks for the specialized SM107 targets and their family operations."""

import ast
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[6]
FAMILY = REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31"
TARGETS = sorted(FAMILY.glob("kimi_k31_pro_nvfp4__sm_107__*"))
# What a target directory holds beyond its three files: dep16's MegaMoE token-path kernel package.
EXTRA_FILES = {
    "kimi_k31_pro_nvfp4__sm_107__dep16": (
        "kernels/__init__.py",
        "kernels/moe/__init__.py",
        "kernels/moe/megamoe_token_path.py",
    ),
}


def test_the_specialized_targets():
    """The four specialized targets, the ULL generation target (Helix CP16, MoE TP16 / EP1) and
    its experimental wide-batch fork (``TLLM_K31_HELIX16_WIDE``)."""
    assert [t.name for t in TARGETS] == [
        f"kimi_k31_pro_nvfp4__sm_107__{layout}"
        for layout in ("cep", "dep16", "helix16_moetp16ep1", "helix16_wide", "helix8", "tep8")
    ]


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.name)
def test_target_directory_holds_three_files(target):
    files = sorted(
        p.relative_to(target).as_posix()
        for p in target.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    want = ["__init__.py", "modeling.py", "weights.py", *EXTRA_FILES.get(target.name, ())]
    assert files == sorted(want)


def test_no_two_test_modules_share_a_basename():
    """The family's test directories have no ``__init__.py``, so pytest imports a test module under
    its basename, and a second module of that name fails to collect. The CPU stage collects the
    whole family directory in one session."""
    names = Counter(p.name for p in Path(__file__).resolve().parent.rglob("test_*.py"))
    assert not [name for name, count in names.items() if count > 1], names.most_common(3)


def _import_time_nodes(tree: ast.Module):
    """Nodes that run at import: everything but function bodies (decorators and defaults run)."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(getattr(node, "decorator_list", []))
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _env_reads(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), str(path))
    lines = []
    for node in _import_time_nodes(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("environ", "getenv")
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ) or (isinstance(node, ast.Name) and node.id in ("environ", "getenv")):
            lines.append(node.lineno)
    return lines


def test_no_environment_read_at_import():
    """The family's routing reads the environment inside functions only."""
    lines = _env_reads(FAMILY / "routing.py")
    assert not lines, f"routing.py reads the environment at import, lines {lines}"


def _launch_constants(target: Path, tree: ast.Module | None = None) -> dict:
    """``_CONSTANTS`` of a target (role -> key -> value), read off its source (or ``tree``)."""

    def value(node):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MappingProxyType":
            return value(node.args[0])
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): value(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)

    tree = tree or ast.parse((target / "modeling.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            "_CONSTANTS"
        ]:
            return value(node.value)
    raise AssertionError(f"{target.name}: no _CONSTANTS")


def _keys_read(path: Path, tree: ast.Module | None = None) -> set:
    """The string keys the source (or ``tree``) reads: ``x["key"]``, ``x.get("key", ...)`` and
    ``_moe_require(role, "key", value)``."""
    keys = set()
    for node in ast.walk(tree or ast.parse(path.read_text())):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            if getattr(node.func, "attr", None) == "get":
                keys.add(node.args[0].value)
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_moe_require"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
        ):
            keys.add(node.args[1].value)
    return keys


# tep8's generation switches of the low-latency MoE and the o_proj all-reduce (the constants line
# prints them), and the targets that read each. helix8 reads ``oproj_ar_early_poll``,
# ``moe_cp_tail`` and ``moe_tail_early_poll`` and sets them as well. A key renamed on one side only
# would leave its feature off without an error.
_LOW_LATENCY_SWITCHES = {
    "moe_cp_tail": ("helix8", "tep8"),
    "moe_sharded_front": ("tep8",),
    "moe_tail_early_poll": ("helix8", "tep8"),
    "oproj_ar_early_poll": ("helix8", "tep8"),
}


def test_the_low_latency_switches_are_read_under_the_keys_tep8_sets():
    by_layout = {t.name.rsplit("__", 1)[1]: t for t in TARGETS}
    generation = _launch_constants(by_layout["tep8"])["generation"]
    for key, readers in _LOW_LATENCY_SWITCHES.items():
        assert generation.get(key) is True, key
        for layout, target in by_layout.items():
            reads = key in _keys_read(target / "modeling.py")
            assert reads == (layout in readers), (layout, key)
