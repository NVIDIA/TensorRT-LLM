# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The start-up line of the ULL generation cache manager's KV cache pools (the Kimi K3.1 helix16
generation targets).

Each rank logs its GPU pool groups once, on one line in the format of the Rubin manager's
``Kimi K3.1 KV cache pools`` line: the slots (pages of the paged pools), slot bytes and bytes of
every pool group, each segment ending with what the group holds (the target's latent KV and index
keys, the drafter's history or the KDA state). CPU tests: the managers carry only the attributes
the log reads; the constructor and the targets are read from their sources.
"""

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _kda_ops_util import bare

pytest.importorskip("fla")  # the shared kimi_kda package imports fla at load time

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_ull_kda_dsa_cache_manager as ull_cache_manager,
)

# CPU test (l0_cpu); a GPU host other than sm_107 skips it.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability(0) != (10, 7),
        reason="Kimi K3.1 Rubin: sm_107 or CPU only",
    ),
]

_MANAGER = ull_cache_manager.KimiK3DSAHybridCacheManager
_MODULE_PATH = Path(inspect.getsourcefile(ull_cache_manager))
_PACKAGE = _MODULE_PATH.parents[2]  # ops/ssm -> kimi_k31
_GENERATIONS = sorted(p.name for p in _PACKAGE.glob("kimi_k31_pro_nvfp4__sm_107__helix16*"))

#: Local layers of a generation rank: KDA layers, MLA layers and the unified drafter's 5 layers.
_KDA_LAYERS = (0, 2)
_MLA_LAYERS = (1, 3)
_DRAFT_LAYERS = (4, 5, 6, 7, 8)
#: GPU pool groups of a helix16 generation rank with a whole drafter history, as the KV cache
#: manager V2 reports them (illustrative counts): the drafter, the DSA layers (latent KV and index
#: keys) and the KDA state.
_POOL_STATS = (
    SimpleNamespace(total=158375, slot_sizes=[184320]),
    SimpleNamespace(total=24000, slot_sizes=[1376256, 393216]),
    SimpleNamespace(total=182, slot_sizes=[54263808, 1907712, 2119680, 16]),
)
_LINE = (
    "Kimi K3.1 KV cache pools (ULL generation, rank {rank}): "
    "pool_group_id=0, num_slots=158375, slot_size=[184320], bytes=29191680000, pool=drafter; "
    "pool_group_id=1, num_slots=24000, slot_size=[1376256, 393216], bytes=42467328000, pool=main; "
    "pool_group_id=2, num_slots=182, slot_size=[54263808, 1907712, 2119680, 16], "
    "bytes=10609001312, pool=state"
)
#: Each pool group's segment of the Rubin manager's line, and of this line with its pool field.
_RUBIN_SEGMENT = r"pool_group_id=(\d+), num_slots=(\d+), slot_size=\[([\d, ]*)\], bytes=(\d+)"
_SEGMENT = _RUBIN_SEGMENT + r", pool=(\w+)(?:;|$)"


def _pool_group(index: int, layers, roles) -> SimpleNamespace:
    """A pool group descriptor holding ``roles`` of ``layers`` (one variant, one coalesced buffer)."""
    buffer_ids = [SimpleNamespace(layer_id=layer, role=role) for layer in layers for role in roles]
    return SimpleNamespace(
        pool_group_index=index,
        slot_desc=SimpleNamespace(
            variants=[SimpleNamespace(coalesced_buffers=[SimpleNamespace(buffer_ids=buffer_ids)])]
        ),
    )


def _manager(rank: int = 0, estimating: bool = False, statistics=None):
    if statistics is None:

        def statistics(level):
            assert level == ull_cache_manager.GPU_LEVEL
            return list(_POOL_STATS)

    # Pool-group descriptors in another order than their indices.
    descs = [
        _pool_group(2, _KDA_LAYERS, ("ssm_state", "conv_state", "kda_ckpt")),
        _pool_group(0, _DRAFT_LAYERS, ("key",)),
        _pool_group(1, _MLA_LAYERS, ("key", "index_key")),
    ]
    return bare(
        _MANAGER,
        is_estimating_kv_cache=estimating,
        mapping=SimpleNamespace(rank=rank),
        impl=SimpleNamespace(get_storage_statistics=statistics, pool_group_descs=descs),
        cp_as_tp_layer_ids=frozenset(_DRAFT_LAYERS),
        _attention_local_layers=list(_MLA_LAYERS),
    )


def _capture(monkeypatch):
    infos, warnings = [], []
    monkeypatch.setattr(ull_cache_manager.logger, "info", infos.append)
    monkeypatch.setattr(ull_cache_manager.logger, "warning", warnings.append)
    return infos, warnings


@pytest.mark.parametrize("rank", [0, 15])
def test_the_serving_cache_logs_its_pools_on_one_line(monkeypatch, rank) -> None:
    """The cache that serves logs one line naming its rank: a segment per pool group in pool-group
    order with its slots, slot bytes and bytes (slots x slot bytes), in the Rubin manager's segment
    format, and whether it holds the drafter's history, the target's latent KV and index keys or
    the KDA state."""
    infos, warnings = _capture(monkeypatch)
    _manager(rank=rank)._log_kv_cache_pools()
    assert warnings == []
    assert infos == [_LINE.format(rank=rank)]
    segments = re.findall(_SEGMENT, infos[0])
    assert [int(group) for group, *_ in segments] == [0, 1, 2]
    assert [kind for *_, kind in segments] == ["drafter", "main", "state"]
    for _, slots, sizes, nbytes, _ in segments:
        assert int(nbytes) == int(slots) * sum(int(size) for size in sizes.split(", "))
    # The Rubin manager's segment reading finds the same groups.
    assert [int(group) for group, *_ in re.findall(_RUBIN_SEGMENT, infos[0])] == [0, 1, 2]


def test_the_estimate_cache_logs_nothing(monkeypatch) -> None:
    """The cache the engine builds for the cache size estimate logs no pool line."""
    infos, warnings = _capture(monkeypatch)
    _manager(estimating=True)._log_kv_cache_pools()
    assert (infos, warnings) == ([], [])


def test_pools_a_cache_cannot_report_are_a_warning(monkeypatch) -> None:
    """The pool line is a log line only: a cache whose pools cannot be read still starts."""

    def unavailable(level):
        raise RuntimeError("no storage statistics")

    infos, warnings = _capture(monkeypatch)
    _manager(statistics=unavailable)._log_kv_cache_pools()
    assert infos == []
    assert warnings == ["Kimi K3.1 KV cache pools unavailable: no storage statistics"]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, name
    return found[0]


def _calls_to(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attribute
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]


def test_the_constructor_logs_the_pools_once() -> None:
    """The manager's constructor logs the pool line once, after the pools are built, and nothing
    else in the module logs it."""
    tree = ast.parse(_MODULE_PATH.read_text())
    manager = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == _MANAGER.__name__
    )
    init = _function(manager, "__init__")
    calls = _calls_to(init, "_log_kv_cache_pools")
    assert len(calls) == 1
    assert ast.unparse(init.body[-1]) == "self._log_kv_cache_pools()"
    assert len(_calls_to(tree, "_log_kv_cache_pools")) == 1
    super_call = [
        n
        for n in init.body
        if isinstance(n, ast.Expr) and ast.unparse(n.value).startswith("super().__init__(")
    ]
    assert len(super_call) == 1 and init.body.index(super_call[0]) < len(init.body) - 1


@pytest.mark.parametrize("generation", _GENERATIONS)
def test_each_helix16_generation_target_builds_this_manager(generation) -> None:
    """Each helix16 generation target (``helix16_moetp16ep1`` and its wide-batch fork, when
    present) selects this manager, so each of its ranks logs the pool line."""
    tree = ast.parse((_PACKAGE / generation / "modeling.py").read_text())
    select = _function(tree, "select_kv_cache_manager_cls")
    imported = {
        alias.name
        for n in ast.walk(select)
        if isinstance(n, ast.ImportFrom)
        and (n.module or "").endswith("rubin_ull_kda_dsa_cache_manager")
        for alias in n.names
    }
    assert imported == {_MANAGER.__name__}, generation
    assert ast.unparse(select.body[-1]) == f"return {_MANAGER.__name__}", generation


def test_the_helix16_generation_targets_are_found() -> None:
    assert "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1" in _GENERATIONS, _GENERATIONS
