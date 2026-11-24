# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA state and convolution-window views of the specialized SM107 targets."""

import ast
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[6]
FAMILY = REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31"
LAYOUTS = ("dep16", "helix8", "tep8")
_PLAIN_VIEW = REPO / "tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py"
_REPLAY_VIEW = REPO / "tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py"
_FLAG = "has_kda_replay_caches"
_HELPER = "_kda_commit_window"
# Where the blocks commit the live conv window: the prefill core after the recurrence and on a
# step whose context rows are all on other ranks, and the one-token decode core. The verify
# kernel keeps the replay windows itself. helix8 serves generation steps only: no prefill core.
_COMMITS = {"_kda_prefill_core": 2, "_kda_decode_core": 1}
_LAYOUT_COMMITS = {"helix8": {"_kda_decode_core": 1}}


def _tree(layout: str) -> ast.Module:
    path = FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{layout}" / "modeling.py"
    return ast.parse(path.read_text(), str(path))


def _owners(tree: ast.Module) -> dict:
    """The top-level function each node sits in (None at module level)."""
    owner = {}
    for top in tree.body:
        name = top.name if isinstance(top, ast.FunctionDef) else None
        for node in ast.walk(top):
            owner[node] = name
    return owner


def _class_fields(path: Path, name: str) -> tuple[set, set]:
    """The annotated fields and the other names a class of ``path`` defines."""
    (cls,) = [
        node
        for node in ast.parse(path.read_text(), str(path)).body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    others = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    return fields, others


class _PlainView:
    """A disaggregated context server's view: the engine's ``MambaLayerCache`` fields only."""

    def __init__(self):
        self.conv = torch.zeros(4, 6, 3, dtype=torch.bfloat16)
        self.temporal = torch.zeros(4, 1, 2, 2)


class _ReplayView(_PlainView):
    """A generation or aggregated server's view: the replay flag and the window commit."""

    def __init__(self, has_replay: bool):
        super().__init__()
        self.has_kda_replay_caches = has_replay
        self.commits = []

    def commit_conv_window(self, slot_indices, conv_pool):
        self.commits.append((slot_indices, conv_pool))


def test_the_stand_in_views_have_the_engines_fields():
    plain_fields, plain_names = _class_fields(_PLAIN_VIEW, "MambaLayerCache")
    assert plain_fields == set(vars(_PlainView())), plain_fields
    assert {_FLAG, "commit_conv_window"}.isdisjoint(plain_fields | plain_names)
    _, replay_names = _class_fields(_REPLAY_VIEW, "KDAReplayLayerCache")
    assert {_FLAG, "commit_conv_window"} <= replay_names


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_block_reads_the_replay_flag_as_an_attribute(layout):
    """A plain view has no replay flag, so only a getattr with a False default may read it."""
    _check_flag_reads(_tree(layout), layout)


def _check_flag_reads(tree: ast.Module, layout: str) -> None:
    owner = _owners(tree)
    direct = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == _FLAG]
    assert not direct, f"{layout}: modeling.py lines {direct} read {_FLAG} as an attribute"
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("getattr", "hasattr")
        and any(isinstance(a, ast.Constant) and a.value == _FLAG for a in node.args)
    ]
    assert [owner[n] for n in reads] == [_HELPER], layout
    (read,) = reads
    assert read.func.id == "getattr" and len(read.args) == 3, ast.unparse(read)
    assert isinstance(read.args[2], ast.Constant) and read.args[2].value is False, ast.unparse(read)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_blocks_commit_the_window_through_the_helper_only(layout):
    _check_commits(_tree(layout), layout)


def _check_commits(tree: ast.Module, layout: str) -> None:
    owner = _owners(tree)
    commits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit_conv_window"
    ]
    assert {owner[n] for n in commits} == {_HELPER}, layout
    callers = Counter(
        owner[node]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _HELPER
    )
    assert callers == _LAYOUT_COMMITS.get(layout, _COMMITS), (layout, callers)


def _helper(layout: str):
    """The target's ``_kda_commit_window``, compiled on its own."""
    tree = _tree(layout)
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == _HELPER]
    namespace = {"Any": Any, "torch": torch}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), f"{layout}/modeling.py", "exec"),
        namespace,
    )
    return namespace[_HELPER]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_helper_on_each_servers_view(layout):
    """A context server's plain view: nothing to commit and no error. A replay view commits the
    live window exactly when it has replay caches."""
    commit = _helper(layout)
    slots = torch.tensor([1, 3], dtype=torch.int32)
    plain = _PlainView()
    before = plain.conv.clone()
    commit(plain, slots, plain.conv)
    assert torch.equal(plain.conv, before) and set(vars(plain)) == {"conv", "temporal"}
    for has_replay, calls in ((True, 1), (False, 0)):
        view = _ReplayView(has_replay)
        commit(view, slots, view.conv)
        assert len(view.commits) == calls, (layout, has_replay)
        if calls:
            assert view.commits[0][0] is slots and view.commits[0][1] is view.conv
