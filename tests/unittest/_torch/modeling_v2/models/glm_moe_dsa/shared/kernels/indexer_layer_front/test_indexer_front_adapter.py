# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""indexer_layer_front selector and call sites.

* the plan: indexer_layer_front runs only with its closure (absorbed_query, indexer_topk,
  draft_absorbed_query, rope_kv_append, cross_rank_topk) and has no numerical variant;
* the registered entry points and their call sites in the target (the absorbed-query packet,
  the cross-rank top-k region and the owned ``forward_absorption_generation``);
* the ``skip_reused_iq`` selector (0 / 1, anything else fails before any GPU
  work) and the reuse condition it acts on, checked against the indexer's
  own source;
* deferred_input_prep's source map lists indexer_layer_front's directories and side selector.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402

CLOSURE = (
    "absorbed_query",
    "indexer_topk",
    "draft_absorbed_query",
    "rope_kv_append",
    "cross_rank_topk",
)


@pytest.fixture
def dispatch(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
        dispatch as d,
    )

    return d


def _select(monkeypatch, d, features):
    monkeypatch.setattr(d, "_log_info", lambda message: None)
    return plan_testlib.select(d, features, monkeypatch)


def test_selector_tables(dispatch):
    assert "indexer_layer_front" in dispatch.SELECTOR_PATHS
    assert dispatch.REQUIRED_FEATURES["indexer_layer_front"] == frozenset(CLOSURE)
    assert "indexer_layer_front" not in dispatch.NUMERICAL_VARIANTS


def test_closure_enforced(dispatch, monkeypatch):
    """Turning off any path of the closure turns indexer_layer_front off with it."""
    monkeypatch.setattr(dispatch, "_require", [False])
    for base in CLOSURE:
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: base})
        assert not plan.enabled("indexer_layer_front"), base
    assert "indexer_layer_front" in _select(
        monkeypatch, dispatch, CLOSURE + ("indexer_layer_front",)
    )
    assert "indexer_layer_front" in _select(
        monkeypatch, dispatch, CLOSURE + ("cross_rank_topk_cluster_merge", "indexer_layer_front")
    )


def test_entry_points_registered(dispatch):
    import tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front  # noqa: E501, F401

    for name in ("dsa_proj", "rope_available", "mla_rope_generation", "cp_indexer_decode"):
        assert callable(dispatch.get_impl("indexer_layer_front", name)), name


def test_call_sites_patched():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        attention_forward,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        region,
    )

    assert 'get_impl("indexer_layer_front", "dsa_proj")' in inspect.getsource(
        attention_forward._absorbed_query_packet
    )
    src = inspect.getsource(region.cross_rank_topk_decode)
    assert src.index('"indexer_layer_front", "cp_indexer_decode"') < src.index(
        '"cross_rank_topk_cluster_merge", "cp_indexer_decode"'
    )
    assert '"indexer_layer_front", "rope_available"' in inspect.getsource(
        attention_forward.forward_absorption_generation
    )


def test_skip_reused_iq_is_off_in_this_target():
    """Every runner starts with the option off and the entry never turns it on."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        query_proj,
    )

    init = inspect.getsource(query_proj.IndexerFrontQueryProjection.__init__)
    assert "self.skip_reused_iq = False" in init
    assert "skip_reused_iq" not in inspect.getsource(query_proj._runner)


def test_reuses_shared_topk_condition():
    """The option fires only under the indexer's own ``reuse_topk`` condition."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        query_proj,
    )

    def mla(share):
        return SimpleNamespace(mqa=SimpleNamespace(indexer=SimpleNamespace(mtp_index_share=share)))

    def md(loop, skip, rows):
        return SimpleNamespace(
            in_mtp_draft_loop=loop, indexer_skip_topk=skip, shared_topk_indices=rows
        )

    assert query_proj.reuses_shared_topk(mla(True), md(True, True, object()))
    assert not query_proj.reuses_shared_topk(mla(False), md(True, True, object()))
    assert not query_proj.reuses_shared_topk(mla(True), md(False, True, object()))
    assert not query_proj.reuses_shared_topk(mla(True), md(True, False, object()))
    assert not query_proj.reuses_shared_topk(mla(True), md(True, True, None))


def _conjuncts(node):
    import ast

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [c for v in node.values for c in _conjuncts(v)]
    return [node]


def _normalized(node):
    """``getattr(obj, "x", default)`` -> ``obj.x``; the indexer and metadata names unified."""
    import ast

    roles = {"self": "indexer", "metadata": "metadata", "attn_metadata": "metadata"}

    class Norm(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            if isinstance(call.func, ast.Name) and call.func.id == "getattr":
                return ast.Attribute(value=call.args[0], attr=call.args[1].value, ctx=ast.Load())
            return call

        def visit_Name(self, name):
            return ast.Name(id=roles.get(name.id, name.id), ctx=ast.Load())

    return ast.unparse(Norm().visit(node))


def test_reuses_shared_topk_matches_indexer_source():
    """``reuses_shared_topk`` is term for term the indexer's ``reuse_topk``.

    The branch the indexer takes under it reads none of ``q_fp8`` / ``q_scale`` / ``weights``
    (the outputs the ``skip_reused_iq`` option leaves unwritten).
    """
    import ast
    import textwrap

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        query_proj,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as indexer_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(indexer_mod.Indexer.sparse_attn_indexer)))
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "reuse_topk" for t in n.targets)
    ]
    assert len(assigns) == 1, "the indexer assigns reuse_topk once"
    indexer_terms = sorted(_normalized(c) for c in _conjuncts(assigns[0].value))

    fn = ast.parse(textwrap.dedent(inspect.getsource(query_proj.reuses_shared_topk)))
    aliases = {
        ast.unparse(s.targets[0]): ast.unparse(s.value)
        for s in fn.body[0].body
        if isinstance(s, ast.Assign)
    }
    assert aliases == {"indexer": "mla.mqa.indexer"}, aliases
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    value = ret.value.args[0] if isinstance(ret.value, ast.Call) else ret.value  # bool(...)
    indexer_front_terms = sorted(_normalized(c) for c in _conjuncts(value))
    assert indexer_front_terms == indexer_terms, (indexer_front_terms, indexer_terms)

    branches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and ast.unparse(n.test.values[-1]) == "reuse_topk"
    ]
    assert len(branches) == 1
    names = {n.id for s in branches[0].body for n in ast.walk(s) if isinstance(n, ast.Name)}
    assert not names & {"q_fp8", "q_scale", "weights"}, names


def test_entry_declines_without_feature(dispatch, monkeypatch):
    """With indexer_layer_front off the entry returns None before touching the module."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        query_proj,
    )

    _select(monkeypatch, dispatch, [])

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"module touched: {name}")

    assert not dispatch.is_enabled("indexer_layer_front")
    assert query_proj.dsa_proj(Boom(), None, None, None) is None


@pytest.mark.skipif(
    not os.environ.get("GLM_MOE_DSA_INDEXER_LAYER_FRONT_JIT_LIB"),
    reason="needs the development build of the indexer_layer_front op",
)
def test_rope_op_available():
    import torch

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_layer_front import (  # noqa: E501
        rope,
    )

    if not rope.rope_available():
        torch.ops.load_library(os.environ["GLM_MOE_DSA_INDEXER_LAYER_FRONT_JIT_LIB"])
    assert rope.rope_available()
