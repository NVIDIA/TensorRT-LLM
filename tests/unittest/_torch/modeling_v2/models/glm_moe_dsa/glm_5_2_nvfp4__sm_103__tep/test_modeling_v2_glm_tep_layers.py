# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tep target's adoption at each tensor-parallel size, the family's layer
composition under its plans, its decode loop, and how its ``modeling.py`` differs from the tp8
target's.

Host-only, with the stand-in built-in modules and kernel factories of the tp8 target's layer
test (``test_modeling_v2_glm_layers.py``, loaded by path): the fast paths the stand-ins see are
the ones the tep plan serves at that size.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling as tp8_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    attention,
    dispatch,
    layers,
    spec_worker,
)

pytestmark = pytest.mark.cpu_only


def _load_tp8_layer_test():
    path = (
        Path(__file__).resolve().parents[1]
        / "glm_5_2_nvfp4__sm_103__tp8"
        / "test_modeling_v2_glm_layers.py"
    )
    spec = importlib.util.spec_from_file_location("_glm_tp8_layer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _load_tp8_layer_test()


def _served(tp_size):
    return modeling.TARGET_PATHS[tp_size].served


@pytest.fixture
def fast_paths(monkeypatch):
    def install(enabled, post_attn_hop=True):
        paths = T._FastPaths(frozenset(enabled))
        if not post_attn_hop:
            # As at four ranks: the call site's native all-reduce is not multi-node NVLink, so
            # the hop factory returns None.
            paths.entries[("post_attention_hop", "make_post_attn_hop")] = lambda *a, **k: None
        monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
        monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
        monkeypatch.setattr(dispatch, "log_capability", paths.log_capability)
        return paths

    return install


def _state(model):
    return dict(modeling.adopted_state(types.SimpleNamespace(model=model.model)))


def test_at_eight_ranks_adoption_builds_the_tp8_objects_but_the_routed_moe_ones(fast_paths):
    paths = fast_paths(_served(8))
    model = T._built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)

    tail = [("make", "attn_tail"), ("make", "staged_attn_tail")]
    pre_moe = tail + [("make", "post_attn_hop"), ("make", "into_hop")]
    post_moe = [("make", "router_gemm"), ("make", "post_moe_hop")]
    expected = (
        pre_moe
        + pre_moe
        + post_moe
        + [("make", "deferred_norm_evict_hints_hop"), ("deferred hop serves", 2, "model config")]
        + pre_moe
        + post_moe
        + [
            ("make", "post_attn_hop"),
            ("make", "router_gemm"),
            ("log_capability", "draft_front", "draft_front", True, ""),
            ("make", "draft_live_row_hop"),
        ]
    )
    assert [e for e in paths.events if e[0] != "set_successor_weights"] == expected
    dense, moe, last, draft = model.model.layers
    assert [type(m) for m in (dense, moe, last)] == [layers.GlmDecoderLayer] * 3
    assert type(draft) is layers.GlmMtpLayer
    assert [type(m.mlp) for m in (moe, last, draft)] == [layers.GlmMoE] * 3
    assert type(moe.self_attn) is attention.GlmMLA
    assert type(model.spec_worker) is spec_worker.GlmMtpWorker
    state = _state(model)
    assert [entry[0] for entry in state["model.layers.1"]] == [
        "post_attn_hop",
        "_into_hop",
        "post_moe_hop",
        "deferred_norm_hop",
        "self_attn._attn_tail",
        "mlp.router_gemm",
    ]
    assert [entry[0] for entry in state["model.layers.3"]] == [
        "post_attn_hop",
        "draft_front",
        "draft_live_row_hop",
        "mlp.router_gemm",
    ]


def test_at_four_ranks_adoption_builds_the_router_gemm_and_the_draft_front(fast_paths):
    paths = fast_paths(_served(4), post_attn_hop=False)
    model = T._built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)

    assert [e for e in paths.events if e[0] == "make"] == [("make", "router_gemm")] * 3
    state = _state(model)
    assert [entry[0] for entry in state["model.layers.0"]] == []
    assert [entry[0] for entry in state["model.layers.1"]] == ["mlp.router_gemm"]
    assert [entry[0] for entry in state["model.layers.3"]] == ["draft_front", "mlp.router_gemm"]
    for layer in model.model.layers:
        assert layer.post_attn_hop is None and layer._into_hop is None
        assert layer.post_moe_hop is None and layer.deferred_norm_hop is None
        assert layer.self_attn._attn_tail is None


# The fast paths only the tp8 target serves: their kernels take tensor-parallel experts. The
# objects only they attach, as (owner submodule, attribute, path).
_ROUTED_MOE_PATHS = frozenset(
    {
        "routed_moe",
        "routed_moe_into_hop",
        "hop_routing",
        "shared_expert_prefetch",
        "draft_routed_moe",
        "cross_layer_expert_prefetch",
    }
)
_ROUTED_MOE_STATE = (
    ("", "_hop_routing", "hop_routing"),
    ("mlp", "_routed_into_hop", "routed_moe_into_hop"),
)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_adoption_attaches_no_routed_moe_object(fast_paths, tp_size):
    paths = fast_paths(_served(tp_size), post_attn_hop=tp_size == 8)
    model = T._built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)
    for layer in model.model.layers:
        for submodule, name, _ in _ROUTED_MOE_STATE:
            owner = getattr(layer, submodule) if submodule else layer
            assert getattr(owner, name, None) is None, (submodule, name)
        assert layer._expert_prefetch_link is None


def _enclosing_tests(func, target):
    """The tests of the ``if`` statements whose body contains ``target``."""
    found = []

    def visit(node, stack):
        if node is target:
            found.extend(stack)
            return True
        if isinstance(node, ast.If):
            return any(visit(c, stack + [ast.unparse(node.test)]) for c in node.body) or any(
                visit(c, stack) for c in node.orelse
            )
        return any(visit(c, stack) for c in ast.iter_child_nodes(node))

    visit(func, [])
    return found


def _paths_named(tree):
    """Every fast path a ``dispatch.is_enabled`` / ``dispatch.get_impl`` call names."""
    return {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) in ("dispatch.is_enabled", "dispatch.get_impl")
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }


def test_every_path_the_composition_names_is_served_or_routed_moe():
    """The family's layers and this target's ``modeling.py`` name, besides the paths the target
    serves at some size, only routed-MoE paths, which it serves at no size: their blocks stay
    off under both plans."""
    served = _served(4) | _served(8)
    assert not served & _ROUTED_MOE_PATHS
    for module in (layers, modeling):
        named = _paths_named(ast.parse(Path(module.__file__).read_text()))
        assert named <= set(dispatch.FAST_PATH_NAMES), module.__name__
        unserved = named - served
        assert unserved <= _ROUTED_MOE_PATHS, (module.__name__, sorted(unserved))
    named = _paths_named(ast.parse(Path(layers.__file__).read_text()))
    assert named & _ROUTED_MOE_PATHS == _ROUTED_MOE_PATHS - {"cross_layer_expert_prefetch"}


def test_the_routed_moe_objects_are_attached_only_under_their_paths():
    """``_hop_routing`` and the MoE's ``_routed_into_hop`` are set only inside a block gated on
    their path, so under a plan without it they keep their class default None."""
    tree = ast.parse(Path(layers.__file__).read_text())
    seen = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                for submodule, name, path in _ROUTED_MOE_STATE:
                    owner = f"self.{submodule}" if submodule else "self"
                    if ast.unparse(target) == f"{owner}.{name}":
                        seen.add(name)
                        assert any(
                            f"dispatch.is_enabled({path!r})" in test
                            for test in _enclosing_tests(func, node)
                        ), (func.name, name)
    assert seen == {name for _, name, _ in _ROUTED_MOE_STATE}


@pytest.mark.parametrize("tp_size", [4, 8])
def test_the_moe_runs_the_builtin_experts_on_the_produced_logits(fast_paths, monkeypatch, tp_size):
    fast_paths(_served(tp_size))
    calls = []
    moe = T._moe(calls)
    moe.__class__ = layers.GlmMoE
    moe.router_gemm = lambda hidden: "router gemm logits"
    monkeypatch.setattr(dispatch, "phase", lambda: dispatch.Phase.DECODE)

    hidden = torch.zeros(2, T.HIDDEN)
    assert moe.compute_routed_output(hidden, None, [2], True) == "routed"
    assert calls[-1][0][1] == "router gemm logits"
    assert moe.compute_routed_output(hidden, None, [2], False, "hop logits") == "routed"
    assert calls[-1][0][1] == "hop logits" and calls[-1][1]["do_finalize"] is False
    moe.router_gemm = None
    moe.compute_routed_output(hidden, None, [2], True)
    assert calls[-1][0][1] == "gate logits"


def test_the_decode_loop_has_no_expert_prefetch_join():
    source = inspect.getsource(modeling.DecodeTarget.forward)
    assert "cross_layer_expert_prefetch" not in source
    assert "setup_aliases" not in vars(modeling.ModelingV2Glm52Nvfp4Sm103Tep)


def _function_bodies(module):
    """Every function of ``module`` as (qualified name, AST dump without docstrings)."""
    tree = ast.parse(Path(module.__file__).read_text())
    out = {}

    def strip(node):
        for child in ast.walk(node):
            body = getattr(child, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                child.body = body[1:] or [ast.Pass()]
        return node

    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef):
                    out[f"{cls.name}.{fn.name}"] = ast.dump(strip(fn))
        elif isinstance(cls, ast.FunctionDef):
            out[cls.name] = ast.dump(strip(cls))
    return out


# The tp8 target's class each tep class corresponds to, and the functions the tep target
# changes: its plan per tensor-parallel size, and no cross-layer expert prefetch (a routed-MoE
# path) in its decode loop and shell.
_RENAMES = {
    "GlmMoeDsaTepModelingV2": "GlmMoeDsaTp8ModelingV2",
    "ModelingV2Glm52Nvfp4Sm103Tep": "ModelingV2Glm52Nvfp4Sm103Tp8",
}
_CHANGED = {"DecodeTarget.forward", "ModelingV2Glm52Nvfp4Sm103Tp8.__init__"}


def test_the_targets_modeling_differs_only_in_the_declared_functions():
    """Every function of the tep ``modeling.py`` equals the tp8 one's (with the tep class
    names), except exactly the declared ones; the tp8 module has only the declared extra
    function. Both adopt their layers with the family's ``adopt_model``."""
    ours = {}
    for name, body in _function_bodies(modeling).items():
        for new, old in _RENAMES.items():
            name = name.replace(new, old)
            body = body.replace(f"'{new}'", f"'{old}'")
        ours[name] = body
    theirs = _function_bodies(tp8_modeling)
    differ = {name for name in ours if name in theirs and ours[name] != theirs[name]}
    assert differ == _CHANGED, (sorted(differ - _CHANGED), sorted(_CHANGED - differ))
    only_tp8 = set(theirs) - set(ours)
    assert only_tp8 <= {"ModelingV2Glm52Nvfp4Sm103Tp8.setup_aliases"}, sorted(only_tp8)
    assert set(ours) - set(theirs) == set()
    assert modeling.adopt_model is tp8_modeling.adopt_model is layers.adopt_model
    assert modeling.adopted_state is tp8_modeling.adopted_state is layers.adopted_state


def test_the_classes_add_no_parameter_buffer_or_module(fast_paths):
    paths = fast_paths(_served(8))
    model = T._built_in_model(paths.events)
    before = [T._keys(layer) for layer in model.model.layers]
    modeling.adopt_model(model, plan=None)
    assert [T._keys(layer) for layer in model.model.layers] == before
    assert all(isinstance(layer, nn.Module) for layer in model.model.layers)
