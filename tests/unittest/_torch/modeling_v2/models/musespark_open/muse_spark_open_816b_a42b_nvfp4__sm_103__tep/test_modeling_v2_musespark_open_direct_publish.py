# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``attention_output_into_hop`` (o_proj publishes into the post-attention hop) is wired.

Host-only. The consumer hop itself is checked bit for bit against the post-attention hop by its
multi-GPU test (``allreduce/test_modeling_v2_post_attn_consumer_hop.py``); here: what
it builds on, what ``_build_direct_publish`` checks before it builds
anything, which steps arm the publication (only the 8-row ones) and that the
arming never outlives the forward, what the o_proj dispatcher writes where while
armed (cuBLAS into the slot, or the usual path and a copy), and what the armed
join hands the consumer hop. Also: the build drops the dense GEMM prefetch's o_proj -> pool prefetch
at 8 rows; a step that raises while armed resets the transport (and a failed
reset refuses later arming); the step's residual is checked before anything is
published; the producer check after the weights load; the guard watchdog.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce import (  # noqa: E501
    post_attn_hop as hop,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual

pytestmark = pytest.mark.cpu_only

_JOIN = target_layers.MuseSparkOpenTepPostAttnJoin


# --- what it builds on ---------------------------------------------------------------------


def test_direct_publish_builds_on_the_hop_and_the_dense_gemms():
    assert dispatch.closure({"attention_output_into_hop"}) == {
        "attention_output_into_hop",
        "post_attention_hop",
        "dense_gemms",
    }
    for missing in ("post_attention_hop", "dense_gemms"):
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(dispatch.closure({"attention_output_into_hop"}) - {missing})


# --- stand-ins ------------------------------------------------------------------------------


class _Plan:
    def __init__(self, kernel=dense_gemms.CUBLAS):
        self.kernels = {"o_proj": (kernel,) * 8}
        self.prefetch = False
        self.enable_pdl = False
        self.forks = 0
        # No named fork registered (the next-QKV prefetch would be one): no role join.
        self.named_forks = {}

    def join(self):
        pass

    def fork_prefetch(self, weight):
        self.forks += 1


class _Linear(dense_gemms.MuseSparkOpenTepDenseLinear):
    """A dense GEMM o_proj as far as ``_build_direct_publish`` and the dispatcher go."""

    def __init__(self, plan, index):
        nn.Module.__init__(self)
        site = dense_gemms._Site(plan, "o_proj", index, (16, 32), None)
        self.__dict__["_low_m_dispatcher"] = dense_gemms.DenseGemmDispatcher(site)


class _Layer(nn.Module):
    def __init__(self, index, plan, o_proj=None):
        super().__init__()
        self.layer_idx = index
        self.mapping = "mapping"
        self.self_attn = nn.Module()
        self.self_attn.o_proj = o_proj if o_proj is not None else _Linear(plan, index)
        self.add_attn = MuseSparkOpenGatedResidual(hop.HIDDEN, torch.bfloat16)
        object.__setattr__(self, "post_attn_norm", types_norm(1e-8))
        object.__setattr__(self, "pre_ffn_norm", types_norm(1e-5))


def types_norm(eps):
    return types.SimpleNamespace(variance_epsilon=eps)


class _Workspace:
    """A direct workspace without a GPU: slots are distinct CPU tensors."""

    def __init__(self):
        self.slots = {
            p: torch.zeros(hop.MAX_ROWS, hop.HIDDEN, dtype=torch.bfloat16) for p in (0, 1)
        }

    def slot(self, parity, rows):
        return self.slots[parity][:rows]


def _model(layers):
    model = nn.Module()
    model.layers = nn.ModuleList(layers)
    return model


def _call(layers, multicast=True):
    workspace = types_ws(multicast)
    joins = tuple(_JOIN.adopt(layer.add_attn, None) for layer in layers)
    return target_layers.PostAttnHopCall(workspace, joins)


def types_ws(multicast):
    return types.SimpleNamespace(multicast=multicast)


@pytest.fixture
def no_fabric(monkeypatch):
    built = []
    workspace = _Workspace()

    def shared_direct_workspace(mapping, *, variants=(), early_trigger=False):
        built.append((mapping, list(variants), early_trigger))
        return workspace

    monkeypatch.setattr(hop, "shared_direct_workspace", shared_direct_workspace)
    return built, workspace


# --- _build_direct_publish -----------------------------------------------------------------


def test_the_build_takes_every_layers_o_proj_dispatcher_and_parity(no_fabric):
    built, workspace = no_fabric
    plan = _Plan()
    layers = [_Layer(i, plan) for i in range(4)]
    call = _call(layers)
    direct = target_layers._build_direct_publish(_model(layers), call, early_trigger=True)
    assert built == [("mapping", [(1e-8, 1e-5, False)], True)]
    assert direct.dispatchers == tuple(
        layer.self_attn.o_proj._low_m_gemm_dispatcher for layer in layers
    )
    assert direct.joins == call.joins
    assert direct.op is hop.post_attn_consumer_hop and direct.workspace is workspace
    for index, slot in enumerate(direct.slots):
        assert slot.data_ptr() == workspace.slots[index % 2].data_ptr()
        assert tuple(slot.shape) == (target_layers.DirectPublish.ROWS, hop.HIDDEN)
    for dispatcher in direct.dispatchers:
        policy = dispatcher.publish_policy
        assert (policy.owner, policy.direct, policy.copy) == (
            "direct_publish",
            frozenset({dense_gemms.CUBLAS}),
            hop.publish_copy,
        )


def test_the_build_refuses_an_o_proj_another_feature_publishes_from(no_fabric):
    built, _ = no_fabric
    plan = _Plan()
    layers = [_Layer(i, plan) for i in range(4)]
    other = dense_gemms.PublishPolicy("zz", frozenset({dense_gemms.CUBLAS}), hop.publish_copy)
    dense_gemms.claim_publish(layers[2].self_attn.o_proj._low_m_gemm_dispatcher, other)
    with pytest.raises(ValueError, match="already publishes for zz"):
        target_layers._build_direct_publish(_model(layers), _call(layers))
    assert built == []


@pytest.mark.parametrize(
    "case, message",
    [
        ("odd", "even count"),
        ("unicast", "without the multicast transport"),
        ("not-dense-gemm", "not on the dense GEMM dispatcher"),
        ("splitk", "not cuBLAS"),
    ],
)
def test_the_build_refuses_before_building_anything(no_fabric, case, message):
    built, _ = no_fabric
    plan = _Plan("splitk:64,8,1,12" if case == "splitk" else dense_gemms.CUBLAS)
    layers = [_Layer(i, plan) for i in range(3 if case == "odd" else 4)]
    if case == "not-dense-gemm":
        layers[2].self_attn.o_proj = nn.Linear(4, 4)
    call = _call(layers, multicast=case != "unicast")
    with pytest.raises(ValueError, match=message):
        target_layers._build_direct_publish(_model(layers), call)
    assert built == []


class _TpGroup:
    """Stands in for ``Distributed`` over a TP group of 8: this rank, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] + self.peers


@pytest.mark.parametrize("local_refuses", [False, True], ids=["peer", "this-rank"])
def test_the_refusal_is_agreed_over_the_tp_group_before_the_workspace(
    no_fabric, monkeypatch, local_refuses
):
    # The workspace's construction is collective: a refusal only one rank sees must make
    # every rank raise before it, not leave the others waiting there.
    built, _ = no_fabric
    peer = (
        "[musespark_open] attention_output_into_hop cannot publish o_proj directly: "
        "layer 2: o_proj runs x at 8 rows"
    )
    group = _TpGroup([None] * 6 + [None if local_refuses else peer])
    monkeypatch.setattr(dispatch, "Distributed", group)
    layers = [_Layer(i, _Plan()) for i in range(3 if local_refuses else 4)]
    for layer in layers:
        layer.mapping = types.SimpleNamespace(tp_size=8)
    match = (
        r"even count \(on TP ranks \[0\] of 8\)"
        if local_refuses
        else r"at 8 rows \(on TP ranks \[7\] of 8\)"
    )
    with pytest.raises(ValueError, match=match):
        target_layers._build_direct_publish(_model(layers), _call(layers))
    assert built == []
    assert len(group.gathered) == 1
    assert (group.gathered[0] is None) is not local_refuses


# --- arming ----------------------------------------------------------------------------------


def _direct(no_fabric, count=4):
    plan = _Plan()
    layers = [_Layer(i, plan) for i in range(count)]
    call = _call(layers)
    call.direct = target_layers._build_direct_publish(_model(layers), call)
    return call, layers


@pytest.mark.parametrize("rows", [1, 4, 7, 8])
def test_only_a_step_of_8_rows_arms_the_publication(no_fabric, rows):
    call, layers = _direct(no_fabric)
    with call.armed(rows):
        for index, layer in enumerate(layers):
            dispatcher = layer.self_attn.o_proj._low_m_gemm_dispatcher
            assert layer.add_attn._armed is call
            if rows == 8:
                assert dispatcher.publish is call.direct.slots[index]
                assert layer.add_attn._direct == (call.direct.slots[index], index % 2)
            else:
                assert dispatcher.publish is None and layer.add_attn._direct is None
    for layer in layers:
        assert layer.self_attn.o_proj._low_m_gemm_dispatcher.publish is None
        assert layer.add_attn._direct is None and layer.add_attn._armed is None


def test_the_arming_is_dropped_when_the_forward_raises(no_fabric):
    call, layers = _direct(no_fabric)
    with pytest.raises(RuntimeError, match="boom"):
        with call.armed(8):
            raise RuntimeError("boom")
    for layer in layers:
        assert layer.self_attn.o_proj._low_m_gemm_dispatcher.publish is None
        assert layer.add_attn._direct is None and layer.add_attn._armed is None


def test_without_direct_publish_the_8_row_step_takes_the_hop(no_fabric):
    plan = _Plan()
    layers = [_Layer(i, plan) for i in range(2)]
    call = _call(layers)
    with call.armed(8):
        assert all(layer.add_attn._direct is None for layer in layers)
        assert all(layer.add_attn._armed is call for layer in layers)


# --- the o_proj dispatcher while armed -------------------------------------------------------


def _dispatcher(kernel=dense_gemms.CUBLAS, prefetch=True):
    plan = _Plan(kernel)
    plan.prefetch = prefetch
    site = dense_gemms._Site(plan, "o_proj", 3, (16, 32), lambda: "next-weight")
    dispatcher = dense_gemms.DenseGemmDispatcher(site)
    dense_gemms.claim_publish(
        dispatcher,
        dense_gemms.PublishPolicy(
            "direct_publish", frozenset({dense_gemms.CUBLAS}), hop.publish_copy
        ),
    )
    return dispatcher, plan


def test_an_armed_cublas_call_writes_the_product_into_the_slot(monkeypatch):
    dispatcher, plan = _dispatcher()
    monkeypatch.setattr(dispatcher, "fits", lambda *a: True)
    x = torch.randn(8, 16).to(torch.bfloat16)
    w = torch.randn(32, 16).to(torch.bfloat16)
    slot = torch.zeros(8, 32, dtype=torch.bfloat16)
    dispatcher.publish = slot
    out = dispatcher.apply(nn.Module(), x, w, None)
    assert out.data_ptr() == slot.data_ptr() and tuple(out.shape) == (8, 32)
    assert torch.equal(slot, torch.mm(x, w.t()))
    assert plan.forks == 1


def test_a_call_cublas_cannot_take_runs_the_usual_path_and_is_copied(monkeypatch):
    dispatcher, plan = _dispatcher("splitk:64,8,1,12")
    product = torch.randn(8, 32).to(torch.bfloat16)
    product[0, :4] = -0.0
    usual = []

    def apply_usual(module, input_tensor, weight, bias, force_active=False):
        usual.append(dispatcher.publish)
        return product

    monkeypatch.setattr(dispatcher, "apply", apply_usual)
    slot = torch.zeros(8, 32, dtype=torch.bfloat16)
    dispatcher.publish = slot
    x = torch.randn(8, 16).to(torch.bfloat16)
    w = torch.randn(32, 16).to(torch.bfloat16)
    out = dense_gemms.DenseGemmDispatcher._apply_publish(dispatcher, nn.Module(), x, w, None, False)
    assert usual == [None]  # the usual path ran unarmed
    assert dispatcher.publish is slot  # and the arming is back
    assert out.data_ptr() == slot.data_ptr()
    assert torch.equal(slot.view(torch.int16), product.view(torch.int16))  # -0 kept


def test_an_armed_call_of_other_rows_is_an_error():
    dispatcher, _ = _dispatcher()
    dispatcher.publish = torch.zeros(8, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="armed to publish"):
        dispatcher.apply(nn.Module(), torch.zeros(4, 16, dtype=torch.bfloat16),
                         torch.zeros(32, 16, dtype=torch.bfloat16), None)  # fmt: skip


# --- the armed join ----------------------------------------------------------------------------


class _Direct:
    def __init__(self):
        self.workspace = object()
        self.calls = []

        def op(*args):
            self.calls.append(args)
            return "consumed"

        self.op = op


class _Norm:
    def __init__(self):
        self.variance_epsilon = 1e-5
        self.use_gemma = True
        self.weight = torch.zeros(hop.HIDDEN, dtype=torch.bfloat16)


def _armed_direct_join(monkeypatch, slot, parity=1):
    join = MuseSparkOpenGatedResidual(hop.HIDDEN, torch.bfloat16)
    join._fused_enabled = True
    _JOIN.adopt(join, None)
    call = types_ws(True)
    call.direct = _Direct()
    join._armed = call
    join._direct = (slot, parity)
    alpha, beta = torch.full((hop.HIDDEN,), 0.25), torch.full((hop.HIDDEN,), 0.75)
    monkeypatch.setattr(join, "coefficients", lambda: (alpha, beta))
    monkeypatch.setattr(hop, "hop_takes", lambda *args: True)
    return join, call.direct, alpha, beta


def test_the_armed_join_runs_the_consumer_on_its_slot(monkeypatch):
    slot = torch.zeros(8, hop.HIDDEN, dtype=torch.bfloat16)
    join, direct, alpha, beta = _armed_direct_join(monkeypatch, slot)
    norm = _Norm()
    residual = torch.zeros(8, hop.HIDDEN)
    assert join._add_after_rmsnorm_and_norm(residual, slot, 1e-8, norm) == "consumed"
    ((args),) = direct.calls
    assert args[0] is slot and args[1] is residual and args[2] is alpha and args[3] is beta
    assert args[4] is norm.weight
    assert args[5:] == (1e-8, norm.variance_epsilon, join._preqkv_norm_fp32, 1, direct.workspace)


def test_a_partial_that_is_not_the_slot_is_copied_into_it(monkeypatch):
    slot = torch.zeros(8, hop.HIDDEN, dtype=torch.bfloat16)
    join, direct, _, _ = _armed_direct_join(monkeypatch, slot)
    partial = torch.randn(8, hop.HIDDEN).to(torch.bfloat16)
    partial[0, :8] = -0.0
    assert join._add_after_rmsnorm_and_norm(torch.zeros(8, hop.HIDDEN), partial, 1e-8, _Norm())
    assert torch.equal(slot.view(torch.int16), partial.view(torch.int16))
    assert direct.calls[0][0] is slot


def test_a_join_the_consumer_does_not_take_raises_instead_of_falling_back(monkeypatch):
    slot = torch.zeros(8, hop.HIDDEN, dtype=torch.bfloat16)
    join, direct, _, _ = _armed_direct_join(monkeypatch, slot)
    with pytest.raises(RuntimeError, match="cannot fall back"):
        join._add_after_rmsnorm_and_norm(
            torch.zeros(8, hop.HIDDEN), slot, 1e-8, _Norm(), gain=torch.ones(1)
        )
    assert direct.calls == []


# --- the o_proj prefetch at the published row count ------------------------------------------


def test_the_build_drops_the_o_proj_prefetch_at_8_rows_on_every_layer(no_fabric, monkeypatch):
    logged = []
    monkeypatch.setattr(hop, "log_once_direct", lambda *args: logged.append(args))
    plan = _Plan()
    plan.prefetch = True
    layers = [_Layer(i, plan) for i in range(4)]
    for layer in layers:
        layer.self_attn.o_proj._low_m_gemm_dispatcher.site.next_weight = lambda: "pool-weight"
    sites = [layer.self_attn.o_proj._low_m_gemm_dispatcher.site for layer in layers]
    sites[1].skip_prefetch_rows = frozenset({3})  # another feature's rows stay
    target_layers._build_direct_publish(_model(layers), _call(layers))
    assert [site.skip_prefetch_rows for site in sites] == [
        frozenset({8}),
        frozenset({3, 8}),
        frozenset({8}),
        frozenset({8}),
    ]
    assert logged == [(4, 8, False, 4)]


@pytest.mark.parametrize("skip, forks", [(frozenset(), 1), (frozenset({8}), 0)])
def test_an_armed_publish_forks_the_prefetch_as_apply_would(monkeypatch, skip, forks):
    dispatcher, plan = _dispatcher()
    dispatcher.site.skip_prefetch_rows = skip
    monkeypatch.setattr(dispatcher, "fits", lambda *a: True)
    dispatcher.publish = torch.zeros(8, 32, dtype=torch.bfloat16)
    dispatcher.apply(nn.Module(), torch.randn(8, 16).to(torch.bfloat16),
                     torch.randn(32, 16).to(torch.bfloat16), None)  # fmt: skip
    assert plan.forks == forks


# --- a step that raises while armed ----------------------------------------------------------


class _ResettableWorkspace(_Workspace):
    def __init__(self, fail=None):
        super().__init__()
        self.device = torch.device("cpu")
        self.resets = 0
        self.fail = fail

    def reset(self):
        self.resets += 1
        if self.fail is not None:
            raise self.fail


def _recoverable(monkeypatch, fail=None, capturing=False):
    workspace = _ResettableWorkspace(fail)
    monkeypatch.setattr(hop, "shared_direct_workspace", lambda *a, **k: workspace)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    plan = _Plan()
    layers = [_Layer(i, plan) for i in range(4)]
    call = _call(layers)
    call.direct = target_layers._build_direct_publish(_model(layers), call)
    return call, layers, workspace


def _residual(rows=8, dtype=torch.float32):
    return torch.zeros(rows, hop.HIDDEN, dtype=dtype)


def _disarmed(layers):
    return all(
        layer.self_attn.o_proj._low_m_gemm_dispatcher.publish is None
        and layer.add_attn._direct is None
        and layer.add_attn._armed is None
        for layer in layers
    )


def test_an_exception_after_arming_resets_the_transport(monkeypatch):
    call, layers, workspace = _recoverable(monkeypatch)
    with pytest.raises(RuntimeError, match="between o_proj and its join"):
        with call.armed(8, _residual()):
            raise RuntimeError("between o_proj and its join")
    assert workspace.resets == 1 and call.direct.poisoned is None and _disarmed(layers)
    with call.armed(8, _residual()):  # the next step arms again
        assert layers[0].self_attn.o_proj._low_m_gemm_dispatcher.publish is not None


_BAD_RESIDUALS = {
    "bf16": lambda: _residual(dtype=torch.bfloat16),
    "rows": lambda: _residual(rows=4),
    "strided": lambda: torch.zeros(hop.HIDDEN, 8).t(),
}


@pytest.mark.parametrize("case", sorted(_BAD_RESIDUALS))
def test_a_residual_the_consumer_cannot_take_is_refused_before_anything_is_published(
    monkeypatch, case
):
    residual = _BAD_RESIDUALS[case]()
    call, layers, workspace = _recoverable(monkeypatch)
    with pytest.raises(ValueError, match="consumer hop takes"):
        with call.armed(8, residual):
            pytest.fail("the step must not run")
    assert workspace.resets == 0 and _disarmed(layers)


def test_other_row_counts_and_captures_never_reset(monkeypatch):
    call, layers, workspace = _recoverable(monkeypatch)
    with pytest.raises(RuntimeError):
        with call.armed(4, _residual(rows=4)):
            raise RuntimeError("the hop's step")
    assert workspace.resets == 0
    call, layers, workspace = _recoverable(monkeypatch, capturing=True)
    with pytest.raises(RuntimeError):
        with call.armed(8, _residual()):
            raise RuntimeError("inside a capture: nothing ran")
    assert workspace.resets == 0 and _disarmed(layers)


def test_a_failed_reset_poisons_the_publication(monkeypatch):
    call, layers, workspace = _recoverable(monkeypatch, fail=RuntimeError("peer gone"))
    with pytest.raises(RuntimeError, match="the step's own error"):
        with call.armed(8, _residual()):
            raise RuntimeError("the step's own error")
    assert workspace.resets == 1 and "peer gone" in call.direct.poisoned and _disarmed(layers)
    with pytest.raises(RuntimeError, match="was not reset"):
        with call.armed(8, _residual()):
            pytest.fail("a poisoned transport must not arm")
    assert workspace.resets == 1 and _disarmed(layers)
    with call.armed(4, _residual(rows=4)):  # the hop's steps are unaffected
        assert all(layer.add_attn._armed is call for layer in layers)


# --- the producer check after the weights load ------------------------------------------------


def test_verify_direct_publish_captures_layer_0s_armed_dispatcher(monkeypatch):
    call, layers, workspace = _recoverable(monkeypatch)
    model = _model(layers)
    model._post_attn_hop = call
    o_proj = layers[0].self_attn.o_proj
    object.__setattr__(o_proj, "weight", torch.zeros(32, 16, dtype=torch.bfloat16))
    dispatcher = o_proj._low_m_gemm_dispatcher
    seen = []
    monkeypatch.setattr(
        dispatcher,
        "apply",
        lambda module, x, w, bias: seen.append((module, tuple(x.shape), w, dispatcher.publish)),
    )

    def check_producer(produce, warm):
        produce()
        return "nvjet"

    workspace.check_producer = check_producer
    assert target_layers.verify_direct_publish(model) == "nvjet"
    ((module, shape, weight, publish),) = seen
    assert module is o_proj and shape == (8, 16) and weight is o_proj.weight
    assert publish is call.direct.slots[0] and dispatcher.publish is None
    model._post_attn_hop.direct = None
    with pytest.raises(ValueError, match="no direct publication"):
        target_layers.verify_direct_publish(model)


@pytest.mark.parametrize(
    "nodes, error, accepted",
    [
        (["nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"], None, True),
        (["<CU_GRAPH_NODE_TYPE_MEMSET>", "nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"], None, False),
        (["nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT", "splitKreduce_kernel"], None, False),
        (["nvjet_sm103_tst_128x8_64x16_2x1_v_bz_splitK_TNT"], None, False),
        ([], RuntimeError("capture failed"), False),
    ],
    ids=["validated", "memset", "reduce-pass", "other-kernel", "capture-error"],
)
def test_check_producer_accepts_exactly_one_validated_kernel(monkeypatch, nodes, error, accepted):
    from tensorrt_llm._torch.distributed import ops as dist_ops

    def captured(produce, device, warm):
        if error is not None:
            raise error
        return list(nodes)

    agreed = []

    def agree(comm, ok):
        agreed.append(ok)
        return ok

    monkeypatch.setattr(hop, "captured_nodes", captured)
    monkeypatch.setattr(dist_ops, "_mnnvl_workspace_all_succeeded", agree)
    workspace = types.SimpleNamespace(_comm=None, device=torch.device("cpu"), rank=3)
    if accepted:
        name = hop.PostAttnDirectWorkspace.check_producer(workspace, lambda: None)
        assert name == nodes[0] and agreed == [True]
    else:
        with pytest.raises(ValueError, match="rank 3 captured"):
            hop.PostAttnDirectWorkspace.check_producer(workspace, lambda: None)
        assert agreed == [False]


# --- the guard watchdog -----------------------------------------------------------------------


def test_the_watchdog_logs_once_when_the_guard_word_is_set(monkeypatch):
    errors = []
    monkeypatch.setattr(hop.logger, "error", errors.append)
    monkeypatch.setattr(hop.time, "sleep", lambda seconds: None)
    reads = iter([False, False, True])
    workspace = types.SimpleNamespace(rank=5, guard_ns=1 << 33, guard_set=lambda: next(reads))
    hop._watch_guard(lambda: workspace, 10.0)
    (message,) = errors
    assert "rank 5" in message and "8.6 s" in message and "parities differ" in message


def test_the_watchdog_stops_with_its_workspace(monkeypatch):
    monkeypatch.setattr(hop.time, "sleep", lambda seconds: None)
    hop._watch_guard(lambda: None, 10.0)  # returns at once
