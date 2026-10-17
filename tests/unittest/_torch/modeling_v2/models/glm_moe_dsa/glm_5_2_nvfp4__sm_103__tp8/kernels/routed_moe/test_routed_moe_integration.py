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
"""routed_moe on the served module types (sm_103, real GLM-5.2 NVFP4 weights).

The routed experts are built the way ``Deepseekv3MoE`` builds them:
``create_moe`` with the served TP8/EP1 ``ModelConfig`` (rank ``GLM_MOE_DSA_ROUTED_MOE_RANK``),
which returns a ``ConfigurableMoE`` around the resolved TRT-LLM-Gen leaf, and
loaded from unsliced checkpoint tensors so the native loader does the TP
slicing. routed_moe is called through ``dispatch.get_impl("routed_moe", "routed_output")``
with that wrapper and compared bit for bit with ``wrapper.forward`` (the
served native path) and with the hand-mirrored ``native_forward`` oracle of
the other tests.

``test_adopted_deepseek_moe_*`` run the served MoE: the built-in
``Deepseekv3MoE`` adopted by the target's ``GlmMoE``, called inside a decode
step.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "shared", "kernels"
    ),
)
import plan_testlib  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

LAYER = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10"))
RANK = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0"))


@pytest.fixture(scope="module")
def full():
    return fx.load_layer_full(LAYER)


@pytest.fixture(scope="module")
def served(full):
    wrapper, bias = fx.build_served_experts(LAYER, RANK, full)
    return wrapper, bias


@pytest.fixture(scope="module")
def leaf_layer():
    return fx.build_native_layer(layer=LAYER, tp_rank=RANK)


@pytest.fixture(scope="module")
def leaf_runner(leaf_layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    return RoutedMoe.from_native(leaf_layer.module, leaf_layer.bias)


@pytest.fixture(scope="module")
def decoder(leaf_layer):
    return fx.ExpertDecoder(leaf_layer, leaf_layer.hidden_proxy(fx.prompt_token_ids()[:1])[0])


def _impl():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    return dispatch.get_impl("routed_moe", "routed_output")


def _inputs(leaf_layer, rows, kind, seed):
    ids = fx.prompt_token_ids()
    if kind == "proxy":
        start = (seed * 7) % (len(ids) - rows)
        x = leaf_layer.hidden_proxy(ids[start : start + rows])
        return x, leaf_layer.router_logits(x)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    if kind == "random":
        return x, leaf_layer.router_logits(x)
    d = int(kind[len("distinct") :])
    sets = fx.expert_sets_with_distinct(rows, d, seed=seed)
    return x, fx.logits_for_expert_sets(sets, leaf_layer.bias, seed=seed)


def test_served_wrapper_type_and_resolution(served):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        eligibility,
        resolve_expert_owner,
    )
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    wrapper, _ = served
    leaf = wrapper.backend
    print(
        f"[routed_moe] served experts: {type(wrapper).__name__} -> {type(leaf).__name__} "
        f"provider={getattr(type(leaf), 'provider', None)} comm={wrapper.comm} "
        f"tp={leaf.tp_size}/{leaf.tp_rank} ep={leaf.ep_size} "
        f"w3_w1={tuple(leaf.w3_w1_weight.shape)}"
    )
    assert isinstance(wrapper, ConfigurableMoE)
    # the wrapper only delegates: the tensors live on the leaf
    assert "w3_w1_weight" not in wrapper._parameters
    owner, reason = resolve_expert_owner(wrapper)
    assert owner is leaf, reason
    x = torch.zeros(6, fx.HIDDEN, device="cuda", dtype=torch.bfloat16)
    lg = torch.zeros(6, fx.NUM_EXPERTS, device="cuda", dtype=torch.float32)
    ok, reason = eligibility(wrapper, x, lg)
    assert ok, reason


def test_served_weights_equal_fixture_slices(served, leaf_layer):
    """The native TP8 loader's rank shard equals the fixture's hand-sliced
    shard byte for byte, so the leaf-level tests run on served buffers.
    """
    leaf = served[0].backend
    ref = leaf_layer.module
    for name in (
        "w3_w1_weight",
        "w3_w1_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "fc31_alpha",
        "fc31_scale_c",
        "fc2_alpha",
        "fc31_input_scale",
    ):
        a, b = getattr(leaf, name), getattr(ref, name)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), name
    assert torch.equal(served[1], leaf_layer.bias)


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("kind", ["proxy", "random", "distinct40"])
def test_served_wrapper_matches_served_forward(
    served, leaf_layer, leaf_runner, decoder, rows, kind
):
    wrapper, bias = served
    if kind == "distinct40" and rows * fx.TOP_K < 40:
        kind = "distinct8"
    x, logits = _inputs(leaf_layer, rows, kind, seed=50 + rows)
    out = _impl()(wrapper, bias, x, logits, True)
    assert out is not None, "routed_moe did not engage on the served wrapper"
    served_native = wrapper.forward(x, logits, do_finalize=True, output_dtype=torch.bfloat16)
    mirrored = leaf_layer.native_forward(x, logits)
    torch.cuda.synchronize()
    # the hand-mirrored oracle is the served path, bit for bit
    assert torch.equal(mirrored, served_native)
    raw = _impl()(wrapper, bias, x, logits, False)
    raw_native = wrapper.forward(x, logits, do_finalize=False, output_dtype=torch.bfloat16)
    mirrored_raw = leaf_layer.native_forward(x, logits, do_finalize=False)
    assert torch.equal(fx.raw_rows_by_topk(mirrored_raw), fx.raw_rows_by_topk(raw_native))
    cls, rep = fx.classify_vs_native(
        out,
        raw,
        served_native,
        raw_native,
        lambda: fx.emulate_moe(leaf_layer, x, logits),
        f"rows={rows} {kind}",
        lambda: decoder.compare(leaf_runner, logits),
    )
    print(f"[routed_moe] served wrapper rows={rows} {kind}: {cls} {rep}")


def test_served_wrapper_fallbacks(served, leaf_layer):
    """Wrapper-level behaviour routed_moe does not implement selects the native path
    before any device work.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        _WORKSPACES,
    )

    wrapper, bias = served
    x, logits = _inputs(leaf_layer, 6, "proxy", seed=3)
    snapshot = {k: v.ws.clone() for k, v in _WORKSPACES.items()}
    for attr, value in (
        ("unfinalized_combine_fn", lambda *a, **k: None),
        ("reduce_results", True),
        ("apply_router_weight_on_input", True),
        ("_enable_perfect_router", True),
    ):
        saved = getattr(wrapper, attr, None)
        setattr(wrapper, attr, value)
        try:
            assert _impl()(wrapper, bias, x, logits, True) is None, attr
        finally:
            setattr(wrapper, attr, saved)
    # FP32 activations and more than eight rows stay native
    assert _impl()(wrapper, bias, x.float(), logits, True) is None
    x9 = torch.cat([x, x[:3]])
    assert _impl()(wrapper, bias, x9, torch.cat([logits, logits[:3]]), True) is None
    torch.cuda.synchronize()
    for k, v in _WORKSPACES.items():
        assert torch.equal(v.ws, snapshot[k])
    assert _impl()(wrapper, bias, x, logits, True) is not None


# ---------------------------------------------------------------------------
# The served MoE: Deepseekv3MoE adopted by the target's GlmMoE.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def deepseek_moe(full):
    return fx.build_deepseek_moe(LAYER, RANK, full)


class _Features:
    """Run exactly the fast paths ``value`` in a block run as a decode step, and count
    routed_moe engagements; the selector state before the block is restored after it.
    """

    def __init__(self, value: str):
        self.value = value
        self.engaged = 0
        self.declined = 0

    def __enter__(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

        self.dispatch = dispatch
        self.step = dispatch.step(dispatch.Phase.DECODE)
        self.step.__enter__()
        self.saved_state = (dispatch._INSTALLED[0], dispatch._VALIDATE[0], dispatch._require[0])
        plan_testlib.select(dispatch, self.value)
        dispatch.get_impl("routed_moe", "routed_output")
        self.saved_fn = dispatch._REGISTRY["routed_moe"]["routed_output"]

        def counting(*args, **kwargs):
            out = self.saved_fn(*args, **kwargs)
            if out is None:
                self.declined += 1
            else:
                self.engaged += 1
            return out

        dispatch._REGISTRY["routed_moe"]["routed_output"] = counting
        return self

    def __exit__(self, *exc):
        self.dispatch._REGISTRY["routed_moe"]["routed_output"] = self.saved_fn
        (
            self.dispatch._INSTALLED[0],
            self.dispatch._VALIDATE[0],
            self.dispatch._require[0],
        ) = self.saved_state
        self.step.__exit__(*exc)
        return False


def _hidden(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)


def _assert_forward_equivalent(on, off, emul_fn, label, id_check=None):
    """forward(do_finalize=False) results [shared, fc2, scales, idx]: the
    shared expert output is identical; the routed raw tuples agree on every
    expert and scale, and their stock finalization is exact or in budget.
    """
    assert torch.equal(on[0], off[0]), f"{label}: shared expert output differs"
    raw_on, raw_off = on[1:], off[1:]
    cls, rep = fx.classify_vs_native(
        fx.finalize_raw(raw_on),
        raw_on,
        fx.finalize_raw(raw_off),
        raw_off,
        emul_fn,
        label,
        id_check,
    )
    return cls


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_adopted_deepseek_moe_forward(deepseek_moe, leaf_layer, leaf_runner, decoder, rows):
    """The served MoE forward (shared expert on the aux stream in
    parallel, gate, routed experts) with ``routed_moe`` on matches it with ``routed_moe``
    off (same experts and scales, shared output identical, routed output exact
    or in budget), and routed_moe actually engaged.
    """
    moe = deepseek_moe
    classes = []
    for it in range(4):
        h = _hidden(rows, 1000 * rows + it)
        with _Features("routed_moe") as f:
            on = moe.forward(h, do_finalize=False)
            fin_on = moe.compute_routed_output(h, None, None, True)
        with _Features("") as f_off:
            off = moe.forward(h, do_finalize=False)
            fin_off = moe.compute_routed_output(h, None, None, True)
        torch.cuda.synchronize()
        assert f.engaged == 2 and f.declined == 0, (f.engaged, f.declined)
        assert f_off.engaged == 0 and f_off.declined == 0
        logits = moe.gate(h)

        def emul(h=h, logits=logits):
            return fx.emulate_moe(leaf_layer, h, logits)

        def ids(logits=logits):
            return decoder.compare(leaf_runner, logits)

        label = f"rows={rows} it={it}"
        classes.append(_assert_forward_equivalent(on, off, emul, label, ids))
        cls, _ = fx.classify_vs_native(fin_on, on[1:], fin_off, off[1:], emul, label, ids)
        classes.append(cls)
    print(f"[routed_moe] adopted Deepseekv3MoE rows={rows}: classes {classes}")


def test_adopted_deepseek_moe_graph(deepseek_moe, leaf_layer, leaf_runner, decoder):
    """CUDA graph of the served forward with ``routed_moe`` on (eager warmup first,
    as the model's graph runner does), replayed on new inputs and compared
    with eager native forwards; overlapping shared-expert GEMMs on the aux
    stream run in every replay.
    """
    moe = deepseek_moe
    rows = 6
    h_static = _hidden(rows, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with _Features("routed_moe") as f:
        with torch.cuda.stream(stream):
            for _ in range(2):
                moe.forward(h_static, do_finalize=False)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_static = moe.forward(h_static, do_finalize=False)
        torch.cuda.synchronize()
    assert f.engaged == 3 and f.declined == 0, (f.engaged, f.declined)
    for it in range(16):
        h = _hidden(rows, 500 + it)
        h_static.copy_(h)
        graph.replay()
        torch.cuda.synchronize()
        with _Features("") as f_off:
            ref = moe.forward(h, do_finalize=False)
        torch.cuda.synchronize()
        assert f_off.engaged == 0
        logits = moe.gate(h)
        _assert_forward_equivalent(
            [t.clone() for t in out_static],
            ref,
            lambda h=h, logits=logits: fx.emulate_moe(leaf_layer, h, logits),
            f"graph replay {it}",
            lambda logits=logits: decoder.compare(leaf_runner, logits),
        )


def test_adopted_deepseek_moe_capture_without_warmup_falls_back(full):
    """A routed_moe call whose first use is inside a CUDA graph capture takes the
    native path (no compilation or workspace initialization is recorded).
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe,
    )

    moe = fx.build_deepseek_moe(LAYER, RANK, full)
    h = _hidden(6, 7)
    # warm the native path, which the capture below records
    with _Features(""):
        moe.compute_routed_output(h, None, None, True)
    torch.cuda.synchronize()
    saved_ws = dict(routed_moe._WORKSPACES)
    saved_fn = dict(routed_moe._COMPILED)
    routed_moe._WORKSPACES.clear()
    routed_moe._COMPILED.clear()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with _Features("routed_moe") as f:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                moe.compute_routed_output(h, None, None, True)
        assert f.declined == 1 and f.engaged == 0
        assert not routed_moe._WORKSPACES and not routed_moe._COMPILED
    finally:
        routed_moe._WORKSPACES.update(saved_ws)
        routed_moe._COMPILED.update(saved_fn)
