# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's MegaMoE generation levers (CPU, from source).

Four constants of a generation server (``_CONSTANTS["generation"]``; a context server runs no
MegaMoE): ``megamoe_pusher_ctas`` (12 pusher CTAs instead of 27), ``megamoe_topk_sentinel`` (the
top-k planes announce themselves by their words), ``megamoe_sort_local_prefix`` (the sort CTAs' own
segment bases) and ``megamoe_refine_sf_first`` (refine polls the scale-factor counter first). The
three token-path levers are the target's own kernel code (``kernels/moe/megamoe_token_path.py``,
which ``kernels/moe/test_megamoe_token_path.py`` beside this target's tests checks). With a stand-in
for the kernel class (the real one needs the GPU stack):

1. the generation constants hold the four, and routing records no variable for them;
2. the runtime build binds the three token-path switches (``bind_token_path_levers``) before the
   MegaMoE workspace, and hands the workspace the pusher count, which it passes to the launcher;
3. the launcher binds the pusher count on the kernel class before its first workspace query,
   leaves the default when unset, refuses pusher counts outside 1 to 27 and a process that already
   runs another count; the family's launcher and entry know no token-path switch;
4. the engine kernel ships 27 pushers and no token-path constant.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_ROUTING = _FAMILY / "routing.py"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_OPS = _FAMILY / "ops/moe/rubin_megamoe.py"
_LAUNCHER = _FAMILY / "kernels/moe/rubin_megamoe.py"
_MEGA = _PKG_ROOT / "_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega"
_KERNEL = _MEGA / "block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py"
# The levers' arguments (rubin_megamoe_workspace's pusher count, bind_token_path_levers' switches)
# and the generation constants that set them
_KEYS = {
    "pusher_ctas": "megamoe_pusher_ctas",
    "topk_sentinel": "megamoe_topk_sentinel",
    "sort_local_prefix": "megamoe_sort_local_prefix",
    "refine_sf_first": "megamoe_refine_sf_first",
}
_SWITCHES = ("topk_sentinel", "sort_local_prefix", "refine_sf_first")
_CONSTANTS = {s: f"token_comm_{s}" for s in _SWITCHES}
_BINDING = "kernels.moe.megamoe_token_path"
_KERNEL_MODULE = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _definitions(path: Path, names: set, ns: dict | None = None) -> dict:
    """The module-level functions and assignments of ``path`` named in ``names``, run from its
    source."""
    wanted = set(names)
    nodes = []
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef):
            found = {node.name}
        elif isinstance(node, ast.Assign):
            found = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if found & wanted:
            nodes.append(node)
            wanted -= found
    assert not wanted, f"{path.name} no longer defines {sorted(wanted)}"
    ns = {} if ns is None else ns
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _function(path: Path, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in ast.walk(_tree(path)) if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _calls(node: ast.AST, name: str) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def test_a_generation_server_runs_the_four_levers_and_reads_no_variable():
    consts = _definitions(_DEP16, {"_CONSTANTS"}, {"MappingProxyType": MappingProxyType})[
        "_CONSTANTS"
    ]
    gen = consts["generation"]
    assert {arg: gen[key] for arg, key in _KEYS.items()} == {
        "pusher_ctas": 12,
        **{s: True for s in _SWITCHES},
    }
    assert gen["moe_backend"] == "megamoe_cutedsl"
    # a context server runs the CuteDSL all-to-all, no MegaMoE
    assert consts["context"]["moe_backend"] == "cutedsl"
    assert not set(_KEYS.values()) & set(consts["context"])
    lever_env = _definitions(_ROUTING, {"LEVER_ENV"})["LEVER_ENV"]
    assert not [v for v in lever_env if "MEGAMOE" in v], lever_env


def test_the_runtime_binds_the_switches_before_the_workspace_and_hands_it_the_pushers():
    build = _function(_DEP16, "_build_runtime")
    (bind,) = _calls(build, "bind_token_path_levers")
    assert {k.arg: ast.unparse(k.value) for k in bind.keywords} == {
        s: f"c['{_KEYS[s]}']" for s in _SWITCHES
    }
    (imported,) = [
        n
        for n in ast.walk(build)
        if isinstance(n, ast.ImportFrom) and [a.name for a in n.names] == ["bind_token_path_levers"]
    ]
    assert (imported.level, imported.module) == (1, _BINDING)
    (call,) = _calls(build, "rubin_megamoe_workspace")
    assert imported.lineno < bind.lineno < call.lineno
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {
        "pusher_ctas": "c['megamoe_pusher_ctas']"
    }
    ops = _function(_OPS, "rubin_megamoe_workspace")
    assert [a.arg for a in ops.args.kwonlyargs] == ["rungs", "pusher_ctas"]
    assert [ast.unparse(d) for d in ops.args.kw_defaults[1:]] == ["None"]
    (inner,) = _calls(ops, "build_workspace")
    assert {k.arg: ast.unparse(k.value) for k in inner.keywords}["pusher_ctas"] == "pusher_ctas"


@pytest.fixture
def launcher(monkeypatch):
    """The launcher's binding, run from source over a stand-in kernel class."""
    kernel = type("RubinInferenceGenphaseMegaMoE", (), {"pusher_cta_count": 27})
    module = ModuleType(_KERNEL_MODULE)
    module.RubinInferenceGenphaseMegaMoE = kernel
    monkeypatch.setitem(sys.modules, _KERNEL_MODULE, module)
    ns = _definitions(_LAUNCHER, {"_DEFAULT_PUSHER_CTAS", "_MAX_PUSHER_CTAS", "_bind_pusher_ctas"})
    return SimpleNamespace(pushers=ns["_bind_pusher_ctas"], kernel=kernel)


def test_off_leaves_the_kernel_as_shipped(launcher):
    launcher.pushers(None)
    assert launcher.kernel.pusher_cta_count == 27


def test_the_pusher_count_binds_once_for_the_process(launcher):
    launcher.pushers(12)
    launcher.pushers(12)
    assert launcher.kernel.pusher_cta_count == 12
    for other in (27, 20):
        with pytest.raises(ValueError, match="already runs the generation kernel with 12"):
            launcher.pushers(other)


@pytest.mark.parametrize("bad", [0, 28])
def test_pusher_counts_outside_the_kernel_range_are_refused(launcher, bad):
    with pytest.raises(ValueError, match="takes 1 to 27"):
        launcher.pushers(bad)
    assert launcher.kernel.pusher_cta_count == 27


def test_the_launcher_binds_before_its_first_workspace_query():
    build = _function(_LAUNCHER, "build_workspace")
    assert [a.arg for a in build.args.kwonlyargs][-1:] == ["pusher_ctas"]
    calls = sorted(
        (n for n in ast.walk(build) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    order = [ast.unparse(n.func) for n in calls]
    query = order.index("query_megamoe_kernel_properties")
    assert order.count("_bind_pusher_ctas") == 1 and order.index("_bind_pusher_ctas") < query


def test_the_family_knows_no_token_path_switch():
    """The switches are dep16's: the family's launcher and entry neither define nor pass them."""
    for path in (_LAUNCHER, _OPS):
        names = {
            n.id if isinstance(n, ast.Name) else n.arg
            for n in ast.walk(_tree(path))
            if isinstance(n, (ast.Name, ast.arg, ast.keyword))
        }
        assert not names & {*_SWITCHES, "_TOKEN_LEVERS", "_bind_token_levers"}, path.name
        assert not [c for c in _CONSTANTS.values() if c in path.read_text()], path.name


def test_the_engine_kernel_ships_27_pushers_and_no_token_path_constant():
    shipped = {
        ast.unparse(node.target): ast.literal_eval(node.value)
        for node in ast.walk(_tree(_KERNEL))
        if isinstance(node, ast.AnnAssign)
        and node.value is not None
        and ast.unparse(node.target) in {"pusher_cta_count", *_CONSTANTS.values()}
    }
    assert shipped == {"pusher_cta_count": 27}
    assert _definitions(_LAUNCHER, {"_DEFAULT_PUSHER_CTAS"})["_DEFAULT_PUSHER_CTAS"] == 27
