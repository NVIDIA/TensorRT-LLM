# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layers' attention window on the KV cache manager V2 of every server (CPU, from source).

The KV cache manager V2 fixes one attention window per local layer in its constructor
(``_resolve_v2_max_attention_window_vec``) and holds a full-context layer's window as None; only
a sliding window is a number. The engine's attention backend reads the entry at the layer's local
index and takes the metadata's ``max_seq_len`` for None (``TrtllmAttention.forward``). No server
of the deployments sets ``max_attention_window``, so every MLA layer's entry is None on the
context and the generation servers alike. The engine's and the targets' functions run from their
source.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_PYEXECUTOR = _PKG_ROOT / "_torch/pyexecutor"
_V2 = _PYEXECUTOR / "kv_cache/kv_cache_manager_v2.py"
_TRTLLM = _PKG_ROOT / "_torch/attention/backends/trtllm.py"
_TARGETS = ("tep8", "dep16", "helix8")
_NUM_LAYERS = 93
_MLA = (*range(3, 88, 4), 91, 92)
# Each server's max_seq_len in the serving configurations (a context server 996672, a generation
# server 997696); none sets max_attention_window. pp is 1, so each rank holds every layer.
_SERVERS = {
    "tep8-context": ("tep8", 996672, True),
    "tep8-generation": ("tep8", 997696, False),
    "dep16-context": ("dep16", 996672, True),
    "dep16-generation": ("dep16", 997696, False),
    "helix8-generation": ("helix8", 997696, False),
}


def _product(target: str) -> Path:
    return _FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{target}/modeling.py"


def _node(path: Path, name: str, cls: str | None = None) -> ast.stmt:
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(
        n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    return node


def _run(nodes: list[ast.stmt], filename: str) -> dict:
    ns = {name: getattr(typing, name) for name in ("Any", "List", "Optional", "Sequence")}
    ns.update(_Step=typing.Any, _Rt=typing.Any)
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), filename, "exec"), ns)
    return ns


def _engine() -> dict:
    """The manager's window functions, from their source."""
    rm = _PYEXECUTOR / "resource_manager.py"
    nodes = [
        _node(rm, "_clamp_max_attention_window_vec"),
        _node(rm, "_project_max_attention_window_vec"),
        _node(_V2, "_resolve_v2_max_attention_window_vec"),
        _node(_V2, "_extend_swa_windows_for_reuse"),
    ]
    return _run(nodes, str(_V2))


def _window_vec(max_seq_len: int, reuse: bool) -> list:
    """What the constructor assigns on a server: the resolved vector, widened for a reuse
    backoff on a server with block reuse (any backoff: it widens numeric windows only)."""
    eng = _engine()
    vec = eng["_resolve_v2_max_attention_window_vec"](None, max_seq_len, list(range(_NUM_LAYERS)))
    if reuse:
        vec = eng["_extend_swa_windows_for_reuse"](vec, 8, max_seq_len)
    return vec


def _mla_window(target: str):
    return _run([_node(_product(target), "_mla_window")], str(_product(target)))["_mla_window"]


def test_the_constructor_and_the_backend_read_the_window_this_way():
    """The constructor assigns the resolved vector (then widens it for a reuse backoff); the
    backend reads it at ``layer_offsets[layer]`` and takes ``metadata.max_seq_len`` for None;
    the targets' DSA pools take the same local index."""
    init = ast.unparse(_node(_V2, "__init__", "KVCacheManagerV2"))
    steps = (
        "self.layer_offsets = {idx: offset for offset, idx in enumerate(self.pp_layers)}",
        "self.max_attention_window_vec = _resolve_v2_max_attention_window_vec("
        "kv_cache_config.max_attention_window, max_seq_len, self.pp_layers)",
        "self.max_attention_window_vec = _extend_swa_windows_for_reuse("
        "self.max_attention_window_vec, self.reuse_match_backoff, max_seq_len)",
    )
    at = [init.find(s) for s in steps]
    assert all(i >= 0 for i in at) and at == sorted(at), at
    forward = ast.unparse(_node(_TRTLLM, "forward", "TrtllmAttention"))
    rule = (
        "window = metadata.kv_cache_manager.max_attention_window_vec[self.local_layer_idx]",
        "if window is not None:",
        "forward_args.attention_window_size = window",
        "forward_args.attention_window_size = metadata.max_seq_len",
    )
    at = [forward.find(s) for s in rule]
    assert all(i >= 0 for i in at) and at == sorted(at), at
    local = ast.unparse(_node(_TRTLLM, "get_local_layer_idx", "TrtllmAttention"))
    assert "metadata.kv_cache_manager.layer_offsets[self.layer_idx]" in local
    for target in _TARGETS:
        bind = ast.unparse(_node(_product(target), "bind", "_DsaPools"))
        assert "self.local = {i: int(cache.layer_offsets[i]) for i in self.mla_layers}" in bind


@pytest.mark.parametrize("max_seq_len, reuse", sorted({s[1:] for s in _SERVERS.values()}))
def test_every_mla_layer_has_no_window_number(max_seq_len, reuse):
    """The engine's window functions leave None for every MLA layer, also after a reuse backoff;
    a server's ``max_seq_len`` and block reuse are all they read that differs between servers."""
    vec = _window_vec(max_seq_len, reuse)
    assert len(vec) == _NUM_LAYERS
    assert {vec[i] for i in _MLA} == {None}


@pytest.mark.parametrize("server", sorted(_SERVERS))
def test_mla_window_gives_the_backends_window(server):
    """Each target's ``_mla_window`` gives the backend's value: ``max_seq_len`` for None. The
    context call on a context server and both calls on a generation server read it."""
    target, max_seq_len, reuse = _SERVERS[server]
    st = SimpleNamespace(
        cache=SimpleNamespace(max_attention_window_vec=_window_vec(max_seq_len, reuse)),
        md=SimpleNamespace(max_seq_len=max_seq_len),
    )
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(local={i: i for i in _MLA}))
    window = _mla_window(target)
    assert [window(i, st, rt) for i in _MLA] == [max_seq_len] * len(_MLA)


@pytest.mark.parametrize("target", _TARGETS)
def test_mla_window_keeps_a_window_number(target):
    """``_mla_window`` hands a sliding window's number through."""
    st = SimpleNamespace(
        cache=SimpleNamespace(max_attention_window_vec=[4096] * _NUM_LAYERS),
        md=SimpleNamespace(max_seq_len=997696),
    )
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(local={i: i for i in _MLA}))
    assert _mla_window(target)(_MLA[0], st, rt) == 4096
