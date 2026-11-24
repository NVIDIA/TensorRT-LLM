# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's shared-experts lever (CPU, from source).

``shared_gate_up_leave_sms``, a generation server's constant: on a CUDA graph step the shared
experts' gate_up GEMM keeps MegaMoE's token exchange CTAs and ``_SHARED_GATE_UP_SPARE_SMS`` more
out of its persistent grid, so the exchange starts on them instead of after the GEMM. With
stand-ins for what needs the GPU stack:

1. the generation constants turn it on, a context server's do not, and routing records no variable
   for it;
2. the target's reserve: the exchange's CTAs and the spares under the constant, 0 without it; the
   runtime builds it from ``megamoe_exchange_ctas`` after the MegaMoE workspace bound its levers
   and holds it (0 by default);
3. only a generation server's graph step hands it to the shared experts, and only their gate_up
   GEMM takes it;
4. the MXFP8 linear and GEMM entries pass it to the runner only when it is set, and their
   ``reference`` and ``is_valid`` mirror the new parameter;
5. the runner takes the static scheduler and fewer active clusters for a reserve (the rule of
   ``TRTLLM_CUTEDSL_SM_RESERVE``, for one launch), before it sizes and keys the launch;
6. the exchange's grid is its CTAs (one per rank, the helper, the pushers) padded to whole launch
   clusters, as the exchange component computes it.
"""

from __future__ import annotations
import __future__

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_ROUTING = _FAMILY / "routing.py"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_OPS_MEGAMOE = _FAMILY / "ops/moe/rubin_megamoe.py"
_LAUNCHER = _FAMILY / "kernels/moe/rubin_megamoe.py"
_LINEAR = _FAMILY / "ops/gemm/rubin_mxfp8_linear.py"
_GEMM_OP = _FAMILY / "ops/gemm/rubin_mxfp8_gemm.py"
_GEMM_KERNEL = _FAMILY / "kernels/gemm/rubin_dense_blockscaled_gemm.py"
_CUTE_DSL_UTILS = _PKG_ROOT / "_torch/cute_dsl_utils.py"
_EXCHANGE = (
    _PKG_ROOT
    / "_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega"
    / "token_comm_gen_specialized.py"
)
_KERNEL_MODULE = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
_MEGA_PACKAGE = f"{_KERNEL_MODULE}.kernel_src.rubin.inference.mega"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _definitions(path: Path, names: set, ns: dict | None = None) -> dict:
    """The module-level functions and assignments of ``path`` named in ``names``, run from its
    source (annotations unevaluated, as the module has them)."""
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
    flags = __future__.annotations.compiler_flag
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _function(path: Path, name: str, owner: str | None = None) -> ast.FunctionDef:
    scope = _tree(path)
    if owner is not None:
        (scope,) = [n for n in scope.body if isinstance(n, ast.ClassDef) and n.name == owner]
    (fn,) = [n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _calls(node: ast.AST, name: str) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _reserve_fn(constants=None):
    """``_shared_gate_up_reserve`` over modeling.py's constants, or over ``constants``."""
    ns = {"MappingProxyType": MappingProxyType}
    names = {"_SHARED_GATE_UP_SPARE_SMS", "_shared_gate_up_reserve"}
    if constants is None:
        names.add("_CONSTANTS")
    else:
        ns["_CONSTANTS"] = constants
    ns = _definitions(_DEP16, names, ns)
    assert ns["_SHARED_GATE_UP_SPARE_SMS"] == 8
    return ns["_shared_gate_up_reserve"]


def test_a_generation_server_leaves_the_exchange_its_sms_and_reads_no_variable():
    consts = _definitions(_DEP16, {"_CONSTANTS"}, {"MappingProxyType": MappingProxyType})[
        "_CONSTANTS"
    ]
    assert consts["generation"]["shared_gate_up_leave_sms"] is True
    assert "shared_gate_up_leave_sms" not in consts["context"]
    lever_env = _definitions(_ROUTING, {"LEVER_ENV"})["LEVER_ENV"]
    assert not [v for v in lever_env if "SHARED_GATE_UP" in v], lever_env


@pytest.mark.parametrize("exchange", [30, 44])
@pytest.mark.parametrize("role, reserved", [("generation", True), ("context", False)])
def test_the_reserve_is_the_exchange_and_the_spares_on_a_generation_server(
    exchange, role, reserved
):
    assert _reserve_fn()(role, exchange) == (exchange + 8 if reserved else 0)


def test_without_the_constant_the_gemm_keeps_its_full_grid():
    off = {"generation": {"shared_gate_up_leave_sms": False}, "context": {}}
    assert _reserve_fn(off)("generation", 30) == 0


def test_the_runtime_builds_the_reserve_after_the_workspace_and_holds_it():
    build = _function(_DEP16, "_build_runtime")
    assigns = [
        n
        for n in ast.walk(build)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "shared_gate_up_reserve"
    ]
    assert sorted(ast.unparse(a.value) for a in assigns) == sorted(
        [
            "0",
            "_shared_gate_up_reserve(core.role, megamoe_exchange_ctas(megamoe))",
        ]
    )
    (workspace,) = _calls(build, "rubin_megamoe_workspace")
    (reserve,) = _calls(build, "_shared_gate_up_reserve")
    assert workspace.lineno < reserve.lineno
    (rt,) = _calls(build, "_Rt")
    held = {k.arg: ast.unparse(k.value) for k in rt.keywords}
    assert held["shared_gate_up_reserve"] == "shared_gate_up_reserve"
    (cls,) = [n for n in _tree(_DEP16).body if isinstance(n, ast.ClassDef) and n.name == "_Rt"]
    (field,) = [
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "shared_gate_up_reserve"
    ]
    assert ast.unparse(field.annotation) == "int" and ast.unparse(field.value) == "0"


def test_only_a_graph_step_hands_the_reserve_to_the_shared_experts():
    def shared_calls(name: str) -> list:
        return [ast.unparse(c) for c in _calls(_function(_DEP16, name), "_moe_shared")]

    assert shared_calls("_moe_dep_gen_streams") == ["_moe_shared(L, x, rt.shared_gate_up_reserve)"]
    assert shared_calls("_moe_dep_gen") == ["_moe_shared(L, x)"]
    assert shared_calls("_moe_shared_ctx") == ["_moe_shared(L, x)"]


def test_only_the_gate_up_gemm_takes_the_reserve():
    """gate_up takes the reserve, and the dense GEMM with or without it (``dense=True``: the same
    bytes on a graph step and the in-order step); SiTU writes down's MXFP8 operand, and the
    prequantized down GEMM has no reserve to take."""
    calls = []

    def rubin_mxfp8_linear(x, w, s, reserve_sms=0, *, dense=False):
        calls.append((w, s, reserve_sms, dense))
        return f"{w}({x})"

    def rubin_situ_and_mul_mxfp8(h, beta, linear_beta, *, zero_padding=True):
        calls.append(("situ_mx", beta, linear_beta, zero_padding))
        return f"codes(situ({h}))", "scales"

    def rubin_mxfp8_linear_prequantized(codes, scales, w, s):
        calls.append((w, s, scales))
        return f"{w}({codes})"

    ns = _definitions(
        _DEP16,
        {"_moe_shared"},
        {
            "rubin_mxfp8_linear": rubin_mxfp8_linear,
            "rubin_situ_and_mul_mxfp8": rubin_situ_and_mul_mxfp8,
            "rubin_mxfp8_linear_prequantized": rubin_mxfp8_linear_prequantized,
        },
    )
    ns["_MOE_SITU"] = (4.0, 25.0)
    layer = SimpleNamespace(
        shared_gu_w="gu_w", shared_gu_s="gu_s", shared_dn_w="dn_w", shared_dn_s="dn_s"
    )
    down = [("situ_mx", 4.0, 25.0, False), ("dn_w", "dn_s", "scales")]
    assert ns["_moe_shared"](layer, "x", 38) == "dn_w(codes(situ(gu_w(x))))"
    assert calls == [("gu_w", "gu_s", 38, True), *down]
    calls.clear()
    ns["_moe_shared"](layer, "x")
    assert calls == [("gu_w", "gu_s", 0, True), *down]


def _args(fn: ast.FunctionDef) -> list:
    names = [a.arg for a in fn.args.args]
    defaults = [ast.unparse(d) for d in fn.args.defaults]
    return list(zip(names, [None] * (len(names) - len(defaults)) + defaults))


@pytest.mark.parametrize(
    "path, owner, base",
    [
        (_LINEAR, "RubinMxfp8Linear", ["input", "weight", "weight_scale"]),
        (_GEMM_OP, "RubinMxfp8Gemm", ["input", "weight", "input_scale", "weight_scale"]),
    ],
)
def test_the_entries_take_the_reserve_and_mirror_it(path, owner, base):
    # the MXFP8 levers' pinned flag comes first, then the reserve (both default off)
    want = [
        ("self", None),
        *((name, None) for name in base),
        ("pinned", "False"),
        ("reserve_sms", "0"),
    ]
    for method in ("__call__", "reference", "is_valid"):
        assert _args(_function(path, method, owner)) == want, method


def test_the_linear_hands_the_reserve_to_the_gemm():
    call = ast.unparse(_function(_LINEAR, "__call__", "RubinMxfp8Linear"))
    assert (
        "return rubin_mxfp8_gemm(act, weight, act_scale, weight_scale, pinned=pinned, "
        "reserve_sms=reserve_sms)" in call
    )


def test_the_gemm_hands_the_reserve_to_the_runner_only_when_set():
    fn = _function(_GEMM_OP, "__call__", "RubinMxfp8Gemm")
    (choose,) = _calls(fn, "tuner.get().choose_one")
    assert [ast.unparse(a) for a in choose.args] == [
        "TUNING_KEY",
        "[runner]",
        "runner.tuning_config",
        "inputs",
    ]
    assert not choose.keywords  # the tactic is the autotuner's, as without a reserve
    runs = sorted(_calls(fn, "runner"), key=lambda n: n.lineno)
    assert [ast.unparse(r) for r in runs] == [
        "runner(inputs, tactic=tactic, reserve_sms=reserve_sms)",
        "runner(inputs, tactic=tactic)",
    ]
    # the pinned tactic's two Ifs stand beside it; the reserve's guard is its own
    (guard,) = [
        n for n in ast.walk(fn) if isinstance(n, ast.If) and "reserve_sms" in ast.unparse(n.test)
    ]
    assert ast.unparse(guard.test) == "reserve_sms > 0"


def _runner_rules() -> dict:
    ns = _definitions(_CUTE_DSL_UTILS, {"reserved_max_active_clusters"}, {"math": math})
    return _definitions(_GEMM_KERNEL, {"_reserved_scheduler", "_reserved_clusters"}, ns)


@pytest.mark.parametrize(
    "mode, reserve, want",
    [("clc_dynamic", 38, "static"), ("static", 38, "static"), ("clc_dynamic", 0, "clc_dynamic")],
)
def test_a_reserve_takes_the_static_scheduler(mode, reserve, want):
    assert _runner_rules()["_reserved_scheduler"](mode, reserve) == want


@pytest.mark.parametrize(
    "full, size, reserve, want",
    [
        (106, 2, 0, 106),  # no reserve: the device's clusters
        (106, 2, 30, 91),  # 182 SMs: the 30 exchange CTAs of 12 pushers at 16 ranks left out
        (106, 2, 38, 87),  # 174 SMs: and the eight spares
        (212, 1, 38, 174),
        (52, 4, 38, 42),
        (1, 2, 38, 1),  # never below one cluster
        (0, 2, 38, 0),  # a shape the device cannot host stays unsupported
    ],
)
def test_a_reserve_takes_clusters_off_the_grid(full, size, reserve, want):
    assert _runner_rules()["_reserved_clusters"](full, size, reserve) == want


def test_the_runner_applies_the_reserve_before_it_sizes_and_keys_the_launch():
    forward = _function(_GEMM_KERNEL, "forward", "RubinBlockScaledGemmRunner")
    statements = {ast.unparse(n): n.lineno for n in ast.walk(forward) if isinstance(n, ast.Assign)}
    read = statements["reserve_sms = int(kwargs.get('reserve_sms') or 0)"]
    take = statements["scheduler_mode = _reserved_scheduler(scheduler_mode, reserve_sms)"]
    (raster,) = _calls(forward, "_launchable_raster_order")
    (key,) = [
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "cache_key"
    ]
    assert read < take < raster.lineno < key.lineno
    sized = _calls(forward, "get_max_activate_clusters")
    wrapped = [c.args[0] for c in _calls(forward, "_reserved_clusters")]
    assert len(sized) == 2 and all(any(s is w for w in wrapped) for s in sized)
    for call in _calls(forward, "_reserved_clusters"):
        assert ast.unparse(call.args[2]) == "reserve_sms" and call.lineno < key.lineno


@pytest.fixture
def exchange_grid(monkeypatch):
    """The launcher's exchange grid, run from source over stand-in kernel and exchange classes."""
    kernel = type("RubinInferenceGenphaseMegaMoE", (), {"pusher_cta_count": 27})
    comm = type("GenphaseTokenComm", (), {"launch_cluster_width": 2})
    package = ModuleType(_KERNEL_MODULE)
    package.RubinInferenceGenphaseMegaMoE = kernel
    mega = ModuleType(_MEGA_PACKAGE)
    exchange = ModuleType(f"{_MEGA_PACKAGE}.token_comm_gen_specialized")
    exchange.GenphaseTokenComm = comm
    mega.token_comm_gen_specialized = exchange
    monkeypatch.setitem(sys.modules, _KERNEL_MODULE, package)
    monkeypatch.setitem(sys.modules, _MEGA_PACKAGE, mega)
    monkeypatch.setitem(sys.modules, exchange.__name__, exchange)
    fn = _definitions(_LAUNCHER, {"exchange_grid_ctas"})["exchange_grid_ctas"]
    return SimpleNamespace(fn=fn, kernel=kernel, comm=comm)


@pytest.mark.parametrize(
    "ranks, pushers, width, want",
    [(16, 27, 2, 44), (16, 12, 2, 30), (4, 12, 2, 18), (16, 12, 1, 29), (4, 27, 2, 32)],
)
def test_the_exchange_grid_is_its_ctas_in_whole_clusters(
    exchange_grid, ranks, pushers, width, want
):
    exchange_grid.kernel.pusher_cta_count = pushers
    exchange_grid.comm.launch_cluster_width = width
    assert exchange_grid.fn(ranks) == want


def test_the_exchange_grid_follows_the_exchange_component():
    tree = _tree(_EXCHANGE)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GenphaseTokenComm"]
    (width,) = [
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "launch_cluster_width"
    ]
    assert ast.literal_eval(width.value) == 2

    def returned(name: str) -> str:
        (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
        (ret,) = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        return ast.unparse(ret.value)

    assert returned("grid_cta_count") == "self.world_size + 1 + self.pusher_cta_count"
    assert returned("launch_cluster_count") == (
        "ceil_div(self.grid_cta_count, self.launch_cluster_width)"
    )
    helper = ast.unparse(_function(_OPS_MEGAMOE, "megamoe_exchange_ctas"))
    assert "return exchange_grid_ctas(ws.ep_size)" in helper
