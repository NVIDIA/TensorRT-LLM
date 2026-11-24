# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash window, admission, and history contracts of the specialized SM107 targets."""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pytest

pytestmark = pytest.mark.cpu_only

# The window-attention entry a specialized target without the FP32 window decode binds.
_WINDOW_ENTRIES = frozenset({"rubin_dflash2_window_triton"})
# The generic target's window attention: its exact FP32 decode, the TRTLLM-Gen operation it runs
# on SM 10.3, the window index rows both read, and its helix draft attention.
_GENERIC_WINDOW_PATHS = frozenset(
    {
        "sparse_mla_rows_eager",
        "dflash2_window_attention",
        "dflash2_window_attention_buffers",
        "dflash2_window_rows",
        "helix_draft_attention",
        "HelixDraftAttention",
    }
)
# Targets that keep the exact FP32 window decode: they bind exactly one of its entries (the torch
# decode, or the same decode with its operands staged by two Triton launches) and build the
# window index rows the generic target builds; the generic target's own FP32 decode is not
# theirs to call.
_FP32_WINDOW_TARGETS = frozenset(
    {
        "kimi_k31_pro_nvfp4__sm_107__cep",
        "kimi_k31_pro_nvfp4__sm_107__dep16",
        "kimi_k31_pro_nvfp4__sm_107__tep8",
    }
)
_FP32_WINDOW_ENTRIES = frozenset({"rubin_dflash2_window_exact", "rubin_dflash2_window_fp32"})
_FP32_WINDOW_PATHS = frozenset({"dflash2_window_rows"})
# The default target: one modeling.py over the three layouts, a layout bound at construction. Each
# layout's drafter must keep the window path of the specialized target of that layout.
# The fields of a ``_BLOCKS`` row that bind the drafter.
_DRAFTER_FIELDS = ("draft_config_for", "DraftModel", "DFlashWorker")
_SWITCH_NAME = re.compile(r"^_*(tp_fusions?|(\w+_)?window_mode)$", re.IGNORECASE)
_ADMISSION_FIELDS = frozenset({"cuda_graph_config", "disable_overlap_scheduler"})
# The engine worker's host-side draft-history preparation, inherited unchanged.
_HOST_PREPARE = frozenset(
    {"prepare_managed_draft_cache", "_prepare_managed_history", "snapshot_managed_draft_history"}
)


def _package_root() -> Path:
    """The ``tensorrt_llm`` package under test, located without importing it."""
    module = sys.modules.get("tensorrt_llm")
    paths = list(getattr(module, "__path__", None) or ())
    if not paths:
        try:
            spec = importlib.util.find_spec("tensorrt_llm")
        except ValueError:
            spec = None
        paths = list((spec and spec.submodule_search_locations) or ())
    if paths:
        return Path(paths[0]).resolve()
    # A source checkout: <root>/tests/unittest/_torch/modeling_v2/models/kimi_k31/<this file>.
    return Path(__file__).resolve().parents[6] / "tensorrt_llm"


_FAMILY = _package_root() / "_torch/_experimental/modeling_v2/models/kimi_k31"


def _sm107_targets() -> list[Path]:
    """Every specialized SM107 target in the tree, whatever it is called."""
    return sorted(p for p in _FAMILY.glob("kimi_k31_*__sm_107__*") if (p / "modeling.py").is_file())


def _target_params() -> list:
    targets = _sm107_targets()
    if not targets:
        return [pytest.param(None, id="none")]
    return [pytest.param(p, id=p.name.rsplit("__", 1)[-1]) for p in targets]


# --- reading a module ---


@dataclass(frozen=True)
class Violation:
    rule: str
    line: int
    what: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.what}"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of the string constants that are docstrings (they describe, they do not run)."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    ids.add(id(value))
    return ids


def _strings(tree: ast.AST) -> Iterator[ast.Constant]:
    """The module's string constants that are not docstrings."""
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            yield node


def _names(tree: ast.AST) -> Iterator[tuple[str, int]]:
    """Every identifier the module binds, reads or imports, with its line."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno
        elif isinstance(node, ast.alias):
            yield node.name.rsplit(".", 1)[-1], node.lineno
            if node.asname:
                yield node.asname, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.rsplit(".", 1)[-1], node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg, node.value.lineno


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name / Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# --- the rules: each takes a parsed module and returns what breaks it ---


def _entry_aliases(tree: ast.AST) -> set[str]:
    """The window entries' names plus the names a module or class binds them to (``W = entry``)."""
    aliases = set(_WINDOW_ENTRIES)
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and node.name in _WINDOW_ENTRIES and node.asname:
            aliases.add(node.asname)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            source = node.value
            if (isinstance(source, ast.Name) and source.id in aliases) or (
                isinstance(source, ast.Attribute) and source.attr in aliases
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        aliases.add(target.attr)
    return aliases


def _window_bindings(tree: ast.AST, fp32_window: bool = False) -> list[Violation]:
    """One window entry and no generic window path; with ``fp32_window``, exactly one of the FP32
    window decode's entries instead, and of the generic paths only the window index rows."""
    names = {}
    for name, line in _names(tree):
        names.setdefault(name, line)
    bound = sorted(n for n in names if n in _WINDOW_ENTRIES)
    allowed = _FP32_WINDOW_PATHS if fp32_window else frozenset()
    generic = sorted(n for n in names if n in _GENERIC_WINDOW_PATHS - allowed)
    out = [
        Violation("window", names[n], f"reaches the generic target's window path {n}")
        for n in generic
    ]
    if fp32_window and bound:
        out.append(
            Violation(
                "window",
                min(names[n] for n in bound),
                f"keeps the FP32 window decode but binds the window entries {bound}",
            )
        )
    if fp32_window:
        exact = sorted(n for n in names if n in _FP32_WINDOW_ENTRIES)
        if len(exact) != 1:
            out.append(
                Violation(
                    "window",
                    min((names[n] for n in exact), default=1),
                    f"keeps the FP32 window decode with {len(exact)} of its entries {exact}, "
                    f"not exactly one of {sorted(_FP32_WINDOW_ENTRIES)}",
                )
            )
    if not fp32_window and len(bound) != 1:
        out.append(
            Violation(
                "window",
                min((names[n] for n in bound), default=1),
                f"binds {len(bound)} window entries {bound}, not exactly one of "
                f"{sorted(_WINDOW_ENTRIES)}",
            )
        )
    return out


def _switches(tree: ast.AST) -> list[Violation]:
    out = []
    seen = set()
    for name, line in _names(tree):
        if _SWITCH_NAME.match(name) and (name, line) not in seen:
            seen.add((name, line))
            out.append(Violation("switch", line, f"a fusion or mode switch {name}"))
    for const in _strings(tree):
        if _SWITCH_NAME.match(const.value):
            out.append(Violation("switch", const.lineno, f"a mode literal {const.value!r}"))
    aliases = _entry_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = _dotted(node.func) or ""
            if callee.rsplit(".", 1)[-1] in aliases and any(k.arg == "mode" for k in node.keywords):
                out.append(Violation("switch", node.lineno, f"passes mode= to {callee}"))
    return out


def _admission_reads(tree: ast.AST) -> list[Violation]:
    out = [
        Violation("admission", line, f"reads the admission field {name}")
        for name, line in _names(tree)
        if name in _ADMISSION_FIELDS
    ]
    out += [
        Violation("admission", c.lineno, f"names {c.value!r}")
        for c in _strings(tree)
        if c.value in _ADMISSION_FIELDS
    ]
    return out


def _store_overrides(tree: ast.AST) -> list[Violation]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_store_context_kv_paged":
                out.append(Violation("store", node.lineno, "defines _store_context_kv_paged"))
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                elif isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "_store_context_kv_paged":
                        out.append(Violation("store", stmt.lineno, "binds _store_context_kv_paged"))
        elif isinstance(node, ast.Call) and (_dotted(node.func) or "").endswith("setattr"):
            if any(
                isinstance(a, ast.Constant) and a.value == "_store_context_kv_paged"
                for a in node.args
            ):
                out.append(Violation("store", node.lineno, "sets _store_context_kv_paged"))
    return out


def _host_prepare_overrides(tree: ast.AST) -> list[Violation]:
    return [
        Violation("host-prepare", node.lineno, f"defines {node.name}")
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _HOST_PREPARE
    ]


def _helix_post_process_calls(tree: ast.AST) -> list[Violation]:
    out = [
        Violation("helix-post-process", line, "reaches _helix_post_process directly")
        for name, line in _names(tree)
        if name == "_helix_post_process"
    ]
    out += [
        Violation("helix-post-process", c.lineno, "names _helix_post_process")
        for c in _strings(tree)
        if c.value == "_helix_post_process"
    ]
    return out


_TARGET_RULES: dict[str, Callable[[ast.AST], list[Violation]]] = {
    "window": _window_bindings,
    "window-fp32": functools.partial(_window_bindings, fp32_window=True),
    "switch": _switches,
    "admission": _admission_reads,
    "store": _store_overrides,
    "host-prepare": _host_prepare_overrides,
    "helix-post-process": _helix_post_process_calls,
}


def _window_entries_in_tree() -> list[str]:
    spec = _FAMILY / "ops" / "spec"
    return sorted(e for e in _WINDOW_ENTRIES if (spec / f"{e}.py").is_file())


def _parse_target(target: Path | None) -> ast.AST:
    if target is None:
        pytest.fail("the family has no kimi_k31_*__sm_107__* target")
    path = target / "modeling.py"
    return ast.parse(path.read_text(), filename=str(path))


def _report(violations: list[Violation]) -> str:
    return "\n".join(str(v) for v in sorted(set(violations), key=lambda v: (v.line, v.what)))


# --- the specialized targets ---


@pytest.mark.parametrize("target", _target_params())
@pytest.mark.parametrize(
    "rule", ["switch", "admission", "store", "host-prepare", "helix-post-process"]
)
def test_target_keeps_the_rule(target, rule):
    tree = _parse_target(target)
    violations = _TARGET_RULES[rule](tree)
    assert not violations, (
        f"{target.name}/modeling.py breaks the {rule} rule:\n{_report(violations)}"
    )


@pytest.mark.parametrize("target", [p for p in _target_params() if p.id != "default"])
def test_target_binds_its_window(target):
    """helix8 binds the window entry; CEP, DEP16 and TEP8 one entry of the exact FP32 decode (the
    default target: per layout, below)."""
    tree = _parse_target(target)
    assert _window_entries_in_tree(), f"the family has no window entry {sorted(_WINDOW_ENTRIES)}"
    violations = _window_bindings(tree, fp32_window=target.name in _FP32_WINDOW_TARGETS)
    assert not violations, f"{target.name}/modeling.py:\n{_report(violations)}"


# --- each rule fires on a source that breaks it, and stays quiet on one that keeps it ---

_ENTRY = "from ..ops.spec.rubin_dflash2_window_triton import rubin_dflash2_window_triton\n"

_CASES = [
    # window
    ("window", _ENTRY + "W = rubin_dflash2_window_triton\n", False),
    ("window", _ENTRY + "from ..ops.attention.dflash2_window import sparse_mla_rows_eager\n", True),
    ("window", _ENTRY + "from ..ops.attention.helix import helix_draft_attention\n", True),
    ("window", "X = 1\n", True),
    # switch
    ("switch", "_TP_FUSIONS = False\n", True),
    ("switch", "class DraftModel:\n    _tp_fusions = False\n", True),
    ("switch", "_WINDOW_MODE = 'triton'\n", True),
    ("switch", _ENTRY + "def f(q):\n    return rubin_dflash2_window_triton(q, mode=1)\n", True),
    (
        "switch",
        _ENTRY + "W = rubin_dflash2_window_triton\ndef f(q):\n    return W(q, mode=1)\n",
        True,
    ),
    ("switch", _ENTRY + "def f(q):\n    return rubin_dflash2_window_triton(q, splits=2)\n", False),
    ("switch", "BACKEND = 'triton'\n", False),
    # admission
    (
        "admission",
        "def check(a):\n    if a.cuda_graph_config is not None or not a.disable_overlap_scheduler:\n"
        "        raise ValueError('no')\n",
        True,
    ),
    ("admission", "DEFAULTS = {'disable_overlap_scheduler': True}\n", True),
    ("admission", '"""Graphs and overlap are admitted by the routing."""\n', False),
    # store
    (
        "store",
        "class W:\n    def _store_context_kv_paged(self, k, v, rows, positions):\n        pass\n",
        True,
    ),
    ("store", "class W:\n    _store_context_kv_paged = staticmethod(print)\n", True),
    (
        "store",
        "class W:\n    def f(self, *a):\n        return super()._store_context_kv_paged(*a)\n",
        False,
    ),
    # host-prepare
    (
        "host-prepare",
        "class W:\n    def _prepare_managed_history(self, request_ids):\n        return []\n",
        True,
    ),
    (
        "host-prepare",
        "class W:\n    def prepare_managed_draft_cache(self, *a):\n        pass\n",
        True,
    ),
    (
        "host-prepare",
        "class W:\n    def f(self, ids):\n        return super()._prepare_managed_history(ids)\n",
        False,
    ),
    # helix-post-process
    (
        "helix-post-process",
        "from tensorrt_llm._torch.modules.attention import _helix_post_process\n",
        True,
    ),
    ("helix-post-process", "import m\nY = m._helix_post_process\n", True),
    (
        "helix-post-process",
        "from ..ops.spec.rubin_dflash2_helix_combine import rubin_dflash2_helix_combine\n",
        False,
    ),
    # window, FP32 decode (DEP16, TEP8)
    (
        "window-fp32",
        "from ..ops.spec.rubin_dflash2_window_exact import rubin_dflash2_window_exact\n"
        "from ..ops.spec.dflash2_window_rows import dflash2_window_rows\n",
        False,
    ),
    (
        "window-fp32",
        "from ..ops.attention.dflash2_window import sparse_mla_rows_eager\n"
        "from ..ops.spec.dflash2_window_rows import dflash2_window_rows\n",
        True,
    ),
    ("window-fp32", _ENTRY + "W = rubin_dflash2_window_triton\n", True),
    ("window-fp32", "from ..ops.attention.helix import helix_draft_attention\n", True),
    (
        "window-fp32",
        "from ..ops.spec.rubin_dflash2_window_fp32 import rubin_dflash2_window_fp32\n"
        "from ..ops.spec.dflash2_window_rows import dflash2_window_rows\n",
        False,
    ),
    (
        "window-fp32",
        "from ..ops.spec.rubin_dflash2_window_exact import rubin_dflash2_window_exact\n"
        "from ..ops.spec.rubin_dflash2_window_fp32 import rubin_dflash2_window_fp32\n"
        "from ..ops.spec.dflash2_window_rows import dflash2_window_rows\n",
        True,
    ),
]


@pytest.mark.parametrize(
    "rule, source, breaks",
    _CASES,
    ids=[
        f"{rule}-{i}-{'breaks' if breaks else 'keeps'}"
        for i, (rule, _, breaks) in enumerate(_CASES)
    ],
)
def test_rule_fires_exactly_on_a_source_that_breaks_it(rule, source, breaks):
    violations = _TARGET_RULES[rule](ast.parse(source))
    assert bool(violations) is breaks, _report(violations) or "no violation"
