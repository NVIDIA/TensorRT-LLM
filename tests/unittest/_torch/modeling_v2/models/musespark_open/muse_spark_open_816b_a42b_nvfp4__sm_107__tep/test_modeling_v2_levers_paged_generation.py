# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 TEP target's wide generation levers over a paged drafter context, on the host only
(no torch, no device).

``levers.py`` and ``dispatch.py`` are loaded from their files with the three leaf modules
``dispatch`` imports stubbed (``_router_index``, the communicator and the logger), and the
compositions are ``modeling.py``'s composition statements run over that ``dispatch``. The tests
pin, at the generation shape sm107 TEP4, ``max_batch_size`` 96 and ``max_num_tokens`` 768:

* a wide generation server (``levers._wide_generation_refusal``) has a windowed or a paged
  drafter context arena: over pages its decode steps keep the raised row cap of the router GEMM
  and the two-shot finalize and the capture layers' multi-request post-FFN join;
* the context-K/V fork of more than 64 requests (``context_kv_fork_wide``) stays on the
  windowed arena, whose parallel windowed commit it builds on;
* the paged drafter's fast paths (the paged context commit, the sampled draft head and the
  rejection fold) resolve on in the paged tables, greedy or with rejection sampling;
* a dense arena, a context server and a generation server of at most 64 requests are not wide.
"""

from __future__ import annotations

import ast
import enum
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
_PACKAGE = "_musespark_open_tep_paged_generation_levers_under_test"


class _Mode(str, enum.Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"


class _Log:
    """The logger stub: every call as (level, message)."""

    def __init__(self) -> None:
        self.records: list = []

    def _record(self, level: str, message) -> None:
        self.records.append((level, str(message)))

    def info(self, message, *args, **kwargs) -> None:
        self._record("info", message)

    def warning(self, message, *args, **kwargs) -> None:
        self._record("warning", message)

    def debug(self, message, *args, **kwargs) -> None:
        self._record("debug", message)

    def error(self, message, *args, **kwargs) -> None:
        self._record("error", message)


class _Group:
    """A TP group of one rank's view: every rank lists this rank's items."""

    def __init__(self) -> None:
        self.world = 1

    def tp_allgather(self, items, small_payload: bool = False) -> list:
        return [list(items) for _ in range(self.world)]


class _Distributed:
    group = _Group()

    @classmethod
    def get(cls, mapping) -> _Group:
        cls.group.world = int(mapping.tp_size)
        return cls.group


_LOG = _Log()


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _load():
    stubs = {
        "tensorrt_llm._torch._experimental.modeling_v2._router_index": _module(
            "_router_index", ModelingV2Mode=_Mode
        ),
        "tensorrt_llm._torch.distributed.communicator": _module(
            "communicator", Distributed=_Distributed
        ),
        "tensorrt_llm.logger": _module("logger", logger=_LOG),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_TARGET)]
        sys.modules[_PACKAGE] = package
        levers = importlib.import_module(f"{_PACKAGE}.levers")
        dispatch = importlib.import_module(f"{_PACKAGE}.dispatch")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return levers, dispatch


def _assigned(node) -> list:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _compositions(dispatch):
    """``modeling.py``'s composition tables, run over ``dispatch``."""
    path = _TARGET / "modeling.py"
    wanted = re.compile(r"^(_(TEP|SM10|PREFETCH)\w*|TARGET_PATHS)$")
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if _assigned(node) and all(wanted.match(name) for name in _assigned(node))
    ]
    namespace = {"_dispatch": dispatch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TARGET_PATHS"]


LEVERS, DISPATCH = _load()
TARGET_PATHS = _compositions(DISPATCH)

#: The generation shape: sm107 TEP4, max_batch_size 96, max_num_tokens 768.
KEY = (107, 4)
MAX_BATCH_SIZE = 96
MAX_NUM_TOKENS = 768

#: The wide generation server's levers that do not read the drafter context arena.
WIDE_GENERATION = ("multi_request_ffn_join",)
#: The fork of more than 64 requests and the parallel windowed commit it builds on.
WIDE_FORK = ("context_kv_fork_wide", "windowed_commit_parallel")
#: The paged drafter's fast paths.
PAGED_DRAFTER = ("paged_context_commit", "draft_head_sampled", "rejection_fold")


def table(arena, rejection_sampling=False, max_batch_size=MAX_BATCH_SIZE, role="gen"):
    return DISPATCH.table_for(
        TARGET_PATHS,
        *KEY,
        max_batch_size,
        role,
        arena,
        {LEVERS.ROLE_KNOB: role},
        max_num_tokens=MAX_NUM_TOKENS,
        draft_logit_transform=True,
        rejection_sampling=rejection_sampling,
    )


@pytest.mark.parametrize("rejection_sampling", [False, True], ids=["greedy", "rs"])
def test_the_paged_generation_server_is_wide(rejection_sampling):
    t = table("paged", rejection_sampling)
    assert LEVERS._wide_generation_refusal(t.deployment) is None
    assert LEVERS.row_cap(t.deployment) == LEVERS.WIDE_ROW_CAP
    for name in WIDE_GENERATION:
        assert t.on(name), (name, t.decisions[name])


@pytest.mark.parametrize("rejection_sampling", [False, True], ids=["greedy", "rs"])
def test_the_wide_fork_keeps_its_windowed_arena(rejection_sampling):
    paged = table("paged", rejection_sampling)
    windowed = table("windowed", rejection_sampling)
    for name in WIDE_FORK:
        assert not paged.on(name), name
        assert windowed.on(name), name
    refusal = LEVERS._wide_fork_refusal(paged.deployment)
    assert refusal is not None and "windowed" in refusal and "paged" in refusal
    assert LEVERS._wide_fork_refusal(windowed.deployment) is None


@pytest.mark.parametrize("rejection_sampling", [False, True], ids=["greedy", "rs"])
def test_the_paged_drafter_paths_stay_on_over_pages(rejection_sampling):
    t = table("paged", rejection_sampling)
    for name in PAGED_DRAFTER:
        assert t.on(name), (name, t.decisions.get(name))


def test_the_paged_table_differs_from_the_windowed_one_only_off_the_wide_levers():
    paged = table("paged")
    windowed = table("windowed")
    for name in (*WIDE_GENERATION, "router_gemm", "finalize_two_shot"):
        assert paged.on(name) == windowed.on(name), name
    assert LEVERS.row_cap(paged.deployment) == LEVERS.row_cap(windowed.deployment)


@pytest.mark.parametrize(
    "arena,max_batch_size,role,needle",
    [
        ("dense", MAX_BATCH_SIZE, "gen", "dense"),
        ("paged", 64, "gen", "never exceeds"),
        ("paged", MAX_BATCH_SIZE, "ctx", "ctx"),
    ],
    ids=["dense", "narrow", "ctx"],
)
def test_the_wide_generation_refusals_name_what_they_miss(arena, max_batch_size, role, needle):
    t = table(arena, max_batch_size=max_batch_size, role=role)
    refusal = LEVERS._wide_generation_refusal(t.deployment)
    assert refusal is not None and needle in refusal
    assert LEVERS.row_cap(t.deployment) == LEVERS.ROW_CAP
    assert not t.on("multi_request_ffn_join")


def test_the_baseline_keeps_the_row_cap_over_pages():
    t = table("paged")
    assert LEVERS.row_cap(t.deployment, LEVERS.BASELINE) == LEVERS.ROW_CAP
