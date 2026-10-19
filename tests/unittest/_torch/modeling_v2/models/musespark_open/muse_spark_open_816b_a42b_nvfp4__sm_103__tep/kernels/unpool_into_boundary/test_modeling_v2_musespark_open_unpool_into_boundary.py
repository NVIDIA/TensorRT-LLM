# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host test of ``unpool_into_boundary`` (``boundary.py`` and its seams): what it
builds on and its refusal, when
the transport is armed, what the armed sites carry and that they disarm on any exit, the
buffer sequence (no two neighbouring calls on one buffer, the wrap included), the
re-classed join's checks, the dispatcher's publish paths under a site's producer policy
(its own, and a cuBLAS-storing one as the attention output projection's), the one-policy-per-site
claim, the installer's checks, the guard watch, and that the consumer kernel's arithmetic is
``unpool_gather_boundary``'s
statement for statement.

No GPU and no collective: the boundary op, the GEMM and the copy are replaced by
recorders. The kernel's bytes are the 8-rank op test's
(``comm/test_modeling_v2_unpool_direct_boundary.py``); the in-model launch counts are the
composition tests' (``MODELING_V2_COMPOSE_TIER=unpool_direct``).
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
    gathers as projection_gathers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.unpool_into_boundary import (  # noqa: E501
    boundary as unpool_direct,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
    unpool_direct_boundary as direct_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
    unpool_gather_boundary as f1_op,
)

HIDDEN, LOCAL = 8192, 1024


def test_it_builds_on_the_gathers_and_the_dense_gemms():
    assert dispatch.closure({"unpool_into_boundary"}) == {
        "unpool_into_boundary",
        "projection_gathers",
        "dense_gemms",
    }
    for missing in ("projection_gathers", "dense_gemms"):
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(dispatch.closure({"unpool_into_boundary"}) - {missing})


def test_refused_when_gathers_kept_the_builtin_unpool_site():
    plan = dispatch.plan_of("unpool_into_boundary")
    with pytest.raises(ValueError, match="unpool_direct"):
        target_layers._refuse_unfused_sites({"fused_pool_gather": True}, plan)
    target_layers._refuse_unfused_sites(
        {"fused_pool_gather": True, "fused_unpool_gather": True}, plan
    )


@pytest.mark.parametrize("capturing", [True, False])
def test_only_an_eight_row_forward_inside_a_capture_takes_the_transport(monkeypatch, capturing):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    for rows in range(1, 17):
        assert unpool_direct.DirectUnpool.takes(rows) == (capturing and rows == 8)


class _Workspace:
    """A stand-in for ``UnpoolDirectWorkspace``: distinct slot tensors per (buffer, rows)."""

    def __init__(self):
        self.world = 8
        self.slots = torch.zeros(direct_op.BUFFERS, 8, LOCAL, dtype=torch.bfloat16)
        self.built = []
        self.watches = 0

    def slot(self, buffer, rows):
        if buffer not in range(direct_op.BUFFERS) or not 1 <= rows <= 8:
            raise ValueError("no slot")
        return self.slots[buffer, :rows]

    def build_variants(self, variants):
        self.built.append(list(variants))

    def start_guard_watch(self, interval_s=None):
        self.watches += 1
        return True


def _sites(count=5):
    return [
        (
            types.SimpleNamespace(publish=None),
            types.SimpleNamespace(_unpool_direct_armed=None),
            buffer,
        )
        for buffer in direct_op.buffer_sequence(count)
    ]


@pytest.mark.parametrize("calls", [2, 3, 4, 5, 6, 61, 62])
def test_the_buffer_sequence_never_repeats_a_buffer_between_neighbours(calls):
    buffers = direct_op.buffer_sequence(calls)
    assert len(buffers) == calls and set(buffers) <= set(range(direct_op.BUFFERS))
    assert all(buffers[i] != buffers[(i + 1) % calls] for i in range(calls))
    assert buffers[:-1] == tuple(i % 2 for i in range(calls - 1))
    assert buffers[-1] == (2 if calls % 2 else 1)


@pytest.mark.parametrize(
    "buffers", [(0,), (0, 0), (0, 1, 0), (0, 1, 1, 0), (0, 1, 3), (1, 0, 1, 0, 1)]
)
def test_a_cycle_with_neighbours_on_one_buffer_is_refused(buffers):
    with pytest.raises(ValueError):
        direct_op.check_buffer_sequence(buffers)
    sites = [
        (types.SimpleNamespace(publish=None), types.SimpleNamespace(_unpool_direct_armed=None), b)
        for b in buffers
    ]
    with pytest.raises(ValueError):
        unpool_direct.DirectUnpool(_Workspace(), sites)


def test_armed_points_every_site_at_its_slot_and_disarms_on_any_exit():
    workspace = _Workspace()
    sites = _sites()
    direct = unpool_direct.DirectUnpool(workspace, sites)
    with direct.armed(8):
        for dispatcher, join, buffer in sites:
            assert dispatcher.publish.data_ptr() == workspace.slot(buffer, 8).data_ptr()
            assert dispatcher.publish.shape == (8, LOCAL)
            assert join._unpool_direct_armed == (workspace, buffer)
    assert all(d.publish is None and j._unpool_direct_armed is None for d, j, _ in sites)
    with pytest.raises(RuntimeError, match="boom"):
        with direct.armed(8):
            raise RuntimeError("boom")
    assert all(d.publish is None and j._unpool_direct_armed is None for d, j, _ in sites)


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _join(monkeypatch, fits=True):
    join = object.__new__(unpool_direct.MuseSparkOpenTepDirectGatedResidual)
    torch.nn.Module.__init__(join)
    join._preqkv_norm_fp32 = False
    join._boundary_trigger = "entry"
    alpha, beta = torch.ones(HIDDEN), torch.zeros(HIDDEN)
    monkeypatch.setattr(
        unpool_direct.MuseSparkOpenTepDirectGatedResidual,
        "coefficients",
        lambda self: (alpha, beta),
        raising=False,
    )
    monkeypatch.setattr(
        unpool_direct.MuseSparkOpenTepDirectGatedResidual,
        "_fits_fused",
        lambda self, *args: fits,
    )
    monkeypatch.setattr(f1_op, "capture_fits", lambda capture, rows, device: True)
    return join, alpha, beta


def _norm():
    return types.SimpleNamespace(
        weight=torch.zeros(HIDDEN, dtype=torch.bfloat16), variance_epsilon=1e-5
    )


def test_a_disarmed_join_is_the_gather_join(monkeypatch):
    join, _, _ = _join(monkeypatch)
    gathers_on = _Recorder(result=("gathers",))
    boundary = _Recorder()
    monkeypatch.setattr(
        projection_gathers.MuseSparkOpenTepGatedResidual, "_add_after_rmsnorm_and_norm", gathers_on
    )
    monkeypatch.setattr(direct_op, "unpool_direct_boundary", boundary)
    residual, branch, norm = torch.zeros(8, HIDDEN), torch.zeros(8, LOCAL), _norm()
    assert join._add_after_rmsnorm_and_norm(residual, branch, 1e-6, norm, None) == ("gathers",)
    assert len(gathers_on.calls) == 1 and boundary.calls == []


@pytest.mark.parametrize("capture", [False, True])
def test_an_armed_join_consumes_its_slot(monkeypatch, capture):
    join, alpha, beta = _join(monkeypatch)
    workspace = _Workspace()
    boundary = _Recorder(result=("direct",))
    monkeypatch.setattr(direct_op, "unpool_direct_boundary", boundary)
    join._unpool_direct_armed = (workspace, 1)
    view = torch.zeros(8, HIDDEN, dtype=torch.bfloat16) if capture else None
    join._capture_fold = view
    join._capture_stored = False
    residual, gain, norm = torch.zeros(8, HIDDEN), torch.ones(HIDDEN, dtype=torch.bfloat16), _norm()
    slot = workspace.slot(1, 8)
    assert join._add_after_rmsnorm_and_norm(residual, slot, 1e-6, norm, gain) == ("direct",)
    ((args, kwargs),) = boundary.calls
    assert kwargs == {}
    assert args[0] is slot and args[1] is workspace and args[2] == 1 and args[3] is residual
    assert args[4] is gain and args[5] is alpha and args[6] is beta and args[7] is norm.weight
    assert args[8:12] == (1e-6, 1e-5, False, "entry")
    assert args[12] is view
    assert join._capture_stored is capture


@pytest.mark.parametrize("case", ["not_fused", "wide_branch", "other_parity", "fresh_tensor"])
def test_an_armed_join_refuses_a_call_that_is_not_its_slot(monkeypatch, case):
    join, _, _ = _join(monkeypatch, fits=case != "not_fused")
    workspace = _Workspace()
    boundary = _Recorder()
    gathers_on = _Recorder()
    monkeypatch.setattr(direct_op, "unpool_direct_boundary", boundary)
    monkeypatch.setattr(
        projection_gathers.MuseSparkOpenTepGatedResidual, "_add_after_rmsnorm_and_norm", gathers_on
    )
    join._unpool_direct_armed = (workspace, 0)
    branch = {
        "not_fused": workspace.slot(0, 8),
        "wide_branch": torch.zeros(8, HIDDEN, dtype=torch.bfloat16),
        "other_parity": workspace.slot(1, 8),
        "fresh_tensor": torch.zeros(8, LOCAL, dtype=torch.bfloat16),
    }[case]
    with pytest.raises(RuntimeError, match="unpool_direct"):
        join._add_after_rmsnorm_and_norm(torch.zeros(8, HIDDEN), branch, 1e-6, _norm(), None)
    assert boundary.calls == [] and gathers_on.calls == []


def _dispatcher(monkeypatch, kernel="splitk:64,8,4,12", prefetch=True, skip=frozenset()):
    kernels = {site: (kernel,) * 8 for site in dense_gemms.SITES}
    plan = dense_gemms.DenseGemmPlan(kernels, prefetch=prefetch, enable_pdl=True)
    forks = []
    monkeypatch.setattr(plan, "fork_prefetch", lambda weight: forks.append(weight))
    next_weight = torch.zeros(4, 4)
    site = dense_gemms._Site(
        plan=plan,
        role="unpool",
        layer_idx=3,
        shape=(4096, LOCAL),
        next_weight=lambda: next_weight,
        skip_prefetch_rows=frozenset(skip),
    )
    dispatcher = dense_gemms.DenseGemmDispatcher(site)
    dense_gemms.claim_publish(dispatcher, unpool_direct.PUBLISH_POLICY)
    return dispatcher, forks, next_weight


@pytest.mark.parametrize("skip", [False, True])
def test_a_fitting_flashinfer_call_writes_the_slot_itself(monkeypatch, skip):
    dispatcher, forks, next_weight = _dispatcher(monkeypatch, skip={8} if skip else ())
    gemm, copy = _Recorder(), _Recorder()
    monkeypatch.setattr(dense_gemms, "dense_gemm", gemm)
    monkeypatch.setattr(direct_op, "publish_copy", copy)
    monkeypatch.setattr(dense_gemms.DenseGemmDispatcher, "fits", lambda self, *a: True)
    slot = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    dispatcher.publish = slot
    x, weight = torch.zeros(8, 4096, dtype=torch.bfloat16), torch.zeros(LOCAL, 4096)
    out = dispatcher.apply(None, x, weight, None)
    assert out.data_ptr() == slot.data_ptr() and out.shape == (8, LOCAL)
    ((args, kwargs),) = gemm.calls
    assert args[2] == "splitk:64,8,4,12" and args[3] is True and kwargs["out"] is slot
    assert copy.calls == []
    assert forks == ([] if skip else [next_weight])


def test_a_cublas_call_is_copied_into_the_slot(monkeypatch):
    dispatcher, forks, _ = _dispatcher(monkeypatch, kernel="cublas", prefetch=False)
    product = torch.ones(8, LOCAL, dtype=torch.bfloat16)
    gemm, copy = _Recorder(result=product), _Recorder()
    monkeypatch.setattr(dense_gemms, "dense_gemm", gemm)
    monkeypatch.setattr(direct_op, "publish_copy", copy)
    monkeypatch.setattr(dense_gemms.DenseGemmDispatcher, "fits", lambda self, *a: True)
    slot = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    dispatcher.publish = slot
    out = dispatcher.apply(None, torch.zeros(8, 4096), torch.zeros(LOCAL, 4096), None)
    assert out.data_ptr() == slot.data_ptr()
    ((args, kwargs),) = gemm.calls
    assert args[2] == "cublas" and "out" not in kwargs
    ((copied, _),) = copy.calls
    assert copied[0] is slot and copied[1].data_ptr() == product.data_ptr()
    assert forks == []
    assert dispatcher.publish is slot


def test_a_call_that_does_not_fit_is_computed_as_usual_and_copied(monkeypatch):
    dispatcher, _, _ = _dispatcher(monkeypatch)
    product = torch.ones(8, LOCAL, dtype=torch.bfloat16)
    seen = []

    def builtin_apply(module, x, weight, bias, force_active=False):
        seen.append(dispatcher.publish)
        return product

    dispatcher.builtin = types.SimpleNamespace(apply=builtin_apply)
    copy = _Recorder()
    monkeypatch.setattr(direct_op, "publish_copy", copy)
    monkeypatch.setattr(dense_gemms.DenseGemmDispatcher, "fits", lambda self, *a: False)
    slot = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    dispatcher.publish = slot
    out = dispatcher.apply(None, torch.zeros(8, 4096), torch.zeros(LOCAL, 4096), None)
    assert seen == [None], "the inner call must run disarmed"
    assert dispatcher.publish is slot, "and the dispatcher re-armed after it"
    assert copy.calls == [((slot, product), {})]
    assert out.data_ptr() == slot.data_ptr()


def test_a_call_of_another_shape_is_refused(monkeypatch):
    dispatcher, _, _ = _dispatcher(monkeypatch)
    monkeypatch.setattr(dense_gemms, "dense_gemm", _Recorder())
    dispatcher.publish = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="unpool_direct"):
        dispatcher.apply(None, torch.zeros(7, 4096), torch.zeros(LOCAL, 4096), None)


def test_an_armed_site_no_feature_claimed_is_refused(monkeypatch):
    dispatcher, _, _ = _dispatcher(monkeypatch)
    dispatcher.publish_policy = None
    gemm = _Recorder()
    monkeypatch.setattr(dense_gemms, "dense_gemm", gemm)
    dispatcher.publish = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="claim"):
        dispatcher.apply(None, torch.zeros(8, 4096), torch.zeros(LOCAL, 4096), None)
    assert gemm.calls == []


def test_one_site_has_one_producer_policy():
    dispatcher = types.SimpleNamespace(
        site=types.SimpleNamespace(layer_idx=5, role="unpool"), publish_policy=None
    )
    dense_gemms.claim_publish(dispatcher, unpool_direct.PUBLISH_POLICY)
    dense_gemms.claim_publish(dispatcher, unpool_direct.PUBLISH_POLICY)  # same owner: idempotent
    other = dense_gemms.PublishPolicy(
        "direct_publish", frozenset({"cublas"}), lambda slot, product: None
    )
    with pytest.raises(ValueError, match="layer 5 unpool GEMM already publishes for unpool_direct"):
        dense_gemms.claim_publish(dispatcher, other)
    assert dispatcher.publish_policy is unpool_direct.PUBLISH_POLICY
    with pytest.raises(ValueError, match="unknown kernel families"):
        dense_gemms.claim_publish(
            types.SimpleNamespace(publish_policy=None),
            dense_gemms.PublishPolicy("zz", frozenset({"tensorrt"}), lambda slot, product: None),
        )
    assert unpool_direct.PUBLISH_POLICY.direct == frozenset({"splitk", "direct"})


@pytest.mark.parametrize("kernel", ["cublas", "splitk:64,8,4,12"])
def test_a_cublas_storing_policy_stores_cublas_and_copies_flashinfer(monkeypatch, kernel):
    """The other contract a site can claim (``direct_publish``'s o_proj: cuBLAS stores into the
    slot): a fitting cuBLAS call writes the slot itself, any other kernel is copied."""
    dispatcher, forks, next_weight = _dispatcher(monkeypatch, kernel=kernel)
    copies = []
    dispatcher.publish_policy = dense_gemms.PublishPolicy(
        "direct_publish",
        frozenset({"cublas"}),
        lambda slot, product: copies.append((slot, product)),
    )
    product = torch.ones(8, LOCAL, dtype=torch.bfloat16)
    gemm = _Recorder(result=product)
    monkeypatch.setattr(dense_gemms, "dense_gemm", gemm)
    monkeypatch.setattr(dense_gemms.DenseGemmDispatcher, "fits", lambda self, *a: True)
    slot = torch.zeros(8, LOCAL, dtype=torch.bfloat16)
    dispatcher.publish = slot
    out = dispatcher.apply(None, torch.zeros(8, 4096), torch.zeros(LOCAL, 4096), None)
    assert out.data_ptr() == slot.data_ptr()
    ((args, kwargs),) = gemm.calls
    assert args[2] == kernel
    if kernel == "cublas":
        assert kwargs["out"] is slot and copies == []
    else:
        assert "out" not in kwargs
        ((copied_slot, copied),) = copies
        assert copied_slot is slot and copied.data_ptr() == product.data_ptr()
    assert forks == [next_weight]


def _model(layers=4, join_class=projection_gathers.MuseSparkOpenTepGatedResidual, linear=True):
    built = []
    for index in range(layers):
        join = object.__new__(join_class)
        torch.nn.Module.__init__(join)
        join._gather_inbox = object()
        join._preqkv_norm_fp32 = index == 2
        join._boundary_trigger = "entry"
        dispatcher = types.SimpleNamespace(publish=None, index=index, publish_policy=None)
        if linear:
            unpooler = object.__new__(dense_gemms.MuseSparkOpenTepDenseLinear)
            torch.nn.Module.__init__(unpooler)
            unpooler.__dict__["_low_m_dispatcher"] = dispatcher
        else:
            unpooler = torch.nn.Identity()
        built.append(
            types.SimpleNamespace(
                add_ffn=join,
                feed_forward=types.SimpleNamespace(unpooler=unpooler),
                post_ffn_norm=types.SimpleNamespace(variance_epsilon=1e-6, gain=lambda: None),
                pre_qkv_proj_norm=types.SimpleNamespace(variance_epsilon=1e-5 + index),
                mapping="mapping",
            )
        )
    return types.SimpleNamespace(layers=built)


def test_install_checks_every_layer_then_builds_and_reclasses(monkeypatch):
    workspace = _Workspace()
    monkeypatch.setattr(direct_op, "shared_workspace", lambda mapping: workspace)
    monkeypatch.setattr(unpool_direct, "describe", lambda kernels: "unpool=splitk@1-8")
    plan = types.SimpleNamespace(kernels={})
    gainless = _model()
    gainless.layers[1].post_ffn_norm = types.SimpleNamespace(variance_epsilon=1e-6)
    claimed = _model()
    claimed.layers[2].feed_forward.unpooler.__dict__[
        "_low_m_dispatcher"
    ].publish_policy = dense_gemms.PublishPolicy(
        "direct_publish", frozenset({"cublas"}), lambda slot, product: None
    )
    for model, kwargs, args in (
        (_model(), {"fused_unpool_gather": True}, (None,)),
        (_model(), {}, (plan,)),
        (_model(join_class=unpool_direct.MuseSparkOpenTepDirectGatedResidual), None, (plan,)),
        (_model(linear=False), None, (plan,)),
        (_model(layers=1), None, (plan,)),
        (_model(layers=2), None, (plan,)),
        (gainless, None, (plan,)),
        (claimed, None, (plan,)),
    ):
        before = [type(layer.add_ffn) for layer in model.layers]
        with pytest.raises(ValueError):
            unpool_direct.install(
                model,
                *args,
                {"fused_unpool_gather": True} if kwargs is None else kwargs,
                frozenset(),
            )
        assert workspace.built == [] and workspace.watches == 0
        assert [type(layer.add_ffn) for layer in model.layers] == before

    model = _model()
    direct = unpool_direct.install(model, plan, {"fused_unpool_gather": True}, frozenset({1, 3}))
    # Layers 0..2 carry a fused boundary; the last keeps the built-in one. Three calls a
    # step: the last takes the third buffer, so the next step's first is not on its buffer.
    assert [type(layer.add_ffn) for layer in model.layers] == [
        unpool_direct.MuseSparkOpenTepDirectGatedResidual
    ] * 3 + [projection_gathers.MuseSparkOpenTepGatedResidual]
    assert [(d.index, buffer) for d, _, buffer in direct.sites] == [(0, 0), (1, 1), (2, 2)]
    assert all(d.publish_policy is unpool_direct.PUBLISH_POLICY for d, _, _ in direct.sites)
    assert workspace.watches == 1
    assert all(j is model.layers[i].add_ffn for i, (_, j, _) in enumerate(direct.sites))
    (built,) = workspace.built
    assert sorted(built) == sorted(
        {
            direct_op.variant(8, True, 1e-6, 1e-5 + 1, False, "entry", False),
            direct_op.variant(8, True, 1e-6, 1e-5 + 2, False, "entry", False),
            direct_op.variant(8, True, 1e-6, 1e-5 + 2, False, "entry", True),
            direct_op.variant(8, True, 1e-6, 1e-5 + 3, True, "entry", False),
        }
    )
    assert direct.workspace is workspace


class _TpGroup:
    """Stands in for ``Distributed`` over a TP group of 8: this rank, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        assert small_payload
        self.gathered.append(obj)
        return [obj] + self.peers


_TP8 = types.SimpleNamespace(tp_size=8)


@pytest.mark.parametrize("local_refuses", [False, True], ids=["peer", "this-rank"])
def test_the_refusal_is_agreed_over_the_tp_group_before_the_workspace(monkeypatch, local_refuses):
    """The workspace's construction is collective: a refusal only one rank sees makes every
    rank raise before it, with nothing built or re-classed."""
    workspace = _Workspace()
    taken = []
    monkeypatch.setattr(
        direct_op, "shared_workspace", lambda mapping: taken.append(mapping) or workspace
    )
    monkeypatch.setattr(unpool_direct, "describe", lambda kernels: "unpool=splitk@1-8")
    peer = (
        "[musespark_open] unpool_direct: layer 2's unpool GEMM already publishes for direct_publish"
    )
    group = _TpGroup([None] * 6 + [None if local_refuses else peer])
    monkeypatch.setattr(unpool_direct._tp_agreement, "Distributed", group)
    model = _model(layers=2 if local_refuses else 4)
    before = [type(layer.add_ffn) for layer in model.layers]
    match = (
        r"two fused post-FFN boundaries \(on TP ranks \[0\] of 8\)"
        if local_refuses
        else r"for direct_publish \(on TP ranks \[7\] of 8\)"
    )
    plan = types.SimpleNamespace(kernels={})
    with pytest.raises(ValueError, match=match):
        unpool_direct.install(model, plan, {"fused_unpool_gather": True}, frozenset(), mapping=_TP8)
    assert taken == [] and workspace.built == [] and workspace.watches == 0
    assert [type(layer.add_ffn) for layer in model.layers] == before
    assert len(group.gathered) == 1 and (group.gathered[0] is None) is not local_refuses


def test_a_failed_variant_compile_is_agreed_before_anything_is_reclassed(monkeypatch):
    """The variants compile after the collective workspace; a failure on one rank is agreed
    (the installs after this one run collectives) and nothing is re-classed."""
    workspace = _Workspace()

    def build_variants(variants):
        raise RuntimeError("cute compile failed")

    workspace.build_variants = build_variants
    monkeypatch.setattr(direct_op, "shared_workspace", lambda mapping: workspace)
    monkeypatch.setattr(unpool_direct, "describe", lambda kernels: "unpool=splitk@1-8")
    group = _TpGroup([None] * 7)
    monkeypatch.setattr(unpool_direct._tp_agreement, "Distributed", group)
    model = _model()
    before = [type(layer.add_ffn) for layer in model.layers]
    plan = types.SimpleNamespace(kernels={})
    with pytest.raises(ValueError, match=r"could not build its boundary variants: RuntimeError"):
        unpool_direct.install(model, plan, {"fused_unpool_gather": True}, frozenset(), mapping=_TP8)
    assert [type(layer.add_ffn) for layer in model.layers] == before
    assert workspace.watches == 0
    assert group.gathered[0] is None and "cute compile failed" in group.gathered[1]


def test_the_guard_watch_logs_an_expired_guard_from_host_memory(monkeypatch):
    """The watch thread reads the guard word as it stands in host memory and logs once."""
    import threading
    import time

    workspace = object.__new__(direct_op.UnpoolDirectWorkspace)
    workspace.control = torch.zeros(direct_op.CONTROL_WORDS, dtype=torch.int32)
    workspace.guard_ns = direct_op.DIRECT_GUARD_NS
    workspace.rank = 3
    workspace._watch = None
    errors, aborts = [], []
    monkeypatch.setattr(direct_op.logger, "error", lambda message: errors.append(message))
    monkeypatch.setattr(direct_op.os, "abort", lambda: aborts.append(True))
    assert workspace.start_guard_watch(0, abort=True) is False
    assert workspace.start_guard_watch(0.005, abort=True) is True
    assert workspace.start_guard_watch(0.005, abort=True) is True  # idempotent
    time.sleep(0.05)
    assert errors == [] and not workspace.guard_fired(sync=False)
    workspace.control[direct_op.GUARD_WORD] = 1
    deadline = time.monotonic() + 5
    while not errors and time.monotonic() < deadline:
        time.sleep(0.005)
    workspace._watch.join(timeout=5)
    assert len(errors) == 1 and "rank 3" in errors[0] and "aborting" in errors[0]
    assert aborts == [True]
    assert not any(
        t.name == "musespark_open.unpool_direct_watchdog" and t.is_alive()
        for t in threading.enumerate()
    )
