# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode robustness without a GPU kernel: the multicast all-reduce epilogue options keep the cluster-barrier bit,
an engine build that fails on one TP rank drops that engine on every rank (two ranks simulated with threads), and
the Eagle3 fc fork's output is recorded on the stream that reads it."""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4 import (  # noqa: E501
    dispatch,
    layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.kernels import (  # noqa: E501
    m3_mc_allreduce as mc,
)


@pytest.fixture(autouse=True)
def _restore_modes():
    require = dispatch.require_mode()
    yield
    dispatch.set_require(require)
    dispatch.reset_logged()


@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("bucket", [8, 16, 32, 48])
def test_ar_opts_keep_cluster_barrier_bit(monkeypatch, bucket, finalize):
    # Tables without bit 2 and with buckets missing: every lookup, fallback included, keeps the barrier.
    monkeypatch.setattr(mc, "AR_OPTS_BY_M", (0, 1, 4, 5, 8, 9, 12, 13))
    monkeypatch.setattr(mc, "AR_OPTS_BY_BUCKET", {})
    monkeypatch.setattr(mc, "AR_FIN_OPTS_BY_BUCKET", {})
    for m in range(1, min(bucket, 8) + 1):
        opts = mc._ar_opts(m, bucket, finalize=finalize)
        assert opts & mc._OPT_CLUSTER_BARRIER and 0 <= opts <= 15, (m, bucket, finalize, opts)


def test_ar_opts_shipped_tables_unchanged():
    for m in range(1, 9):
        assert mc._ar_opts(m, 8) == mc.AR_OPTS_BY_M[m - 1]
    for b, o in mc.AR_OPTS_BY_BUCKET.items():
        assert mc._ar_opts(b, b) == o
    for b, o in mc.AR_FIN_OPTS_BY_BUCKET.items():
        assert mc._ar_opts(b, b, finalize=True) == o


class _FakeComm:
    """An MPI-like TP communicator for ``world`` threads: allreduce sums the ranks' ints."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.lock = threading.Lock()
        self.vals = []
        self.calls = 0

    def allreduce(self, v):
        with self.lock:
            self.vals.append(v)
            self.calls += 1
        self.barrier.wait()
        total = sum(self.vals)
        self.barrier.wait()
        with self.lock:
            self.vals.clear()
        self.barrier.wait()
        return total

    def Get_size(self):  # noqa: N802 (MPI name)
        return self.world


def _mapping(tp_size=2, adp=False):
    return SimpleNamespace(tp_size=tp_size, enable_attention_dp=adp, tp_group=list(range(tp_size)))


def _run_ranks(fns):
    out, errs = [None] * len(fns), []

    def run(i):
        try:
            out[i] = fns[i]()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(fns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errs, errs
    return out


@pytest.fixture
def two_rank_comm(monkeypatch):
    from tensorrt_llm._torch.distributed import ops

    comm = _FakeComm(2)
    monkeypatch.setattr(mc, "_tp_comm", lambda mapping: comm)
    monkeypatch.setattr(ops, "mpi_disabled", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return comm


def test_tp_agree_single_rank_and_adp_take_no_collective(monkeypatch):
    monkeypatch.setattr(mc, "_tp_comm", mock.Mock(side_effect=AssertionError("no collective")))
    assert mc.tp_agree(_mapping(1), True) is True
    assert mc.tp_agree(_mapping(1), False) is False
    assert mc.tp_agree(_mapping(4, adp=True), False) is False


def test_tp_agree_refuses_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError):
        mc.tp_agree(_mapping(2), True)


@pytest.mark.parametrize("oks", [(True, True), (True, False), (False, True), (False, False)])
def test_tp_agree_built_two_ranks(two_rank_comm, oks):
    built = [object() if ok else None for ok in oks]
    out = _run_ranks([lambda i=i: mc.tp_agree_built(_mapping(2), built[i], "t") for i in range(2)])
    assert out == (built if all(oks) else [None, None])


def _moe(layer_idx: int, fail: bool) -> layers.M3MoE:
    """An M3MoE on two simulated TP ranks whose engine builds are stubbed (and fail on one)."""

    def stock_init(self, model_config, aux_stream_dict, layer_idx=None):
        nn.Module.__init__(self)
        self.mapping = _mapping(2)
        self.shared_experts = None

    with mock.patch.object(layers.MiniMaxM3MoE, "__init__", stock_init):
        moe = layers.M3MoE(None, {}, layer_idx)
    moe.shared_experts = object()

    def build(num_tokens, shared=True):
        if fail:
            raise RuntimeError("compile failed on this rank")
        return ("engine", num_tokens)

    moe._build_engine = build
    moe._build_moe16_engine = lambda num_tokens: build(num_tokens)
    return moe


@pytest.mark.parametrize("fail", [(False, False), (False, True)])
def test_engine_build_agreed_across_ranks(two_rank_comm, fail):
    moes = [_moe(3, f) for f in fail]
    _run_ranks([m.prebuild_engines for m in moes])
    keys = sorted(moes[0]._engines) + sorted(moes[0]._moe16_engines)
    assert keys and two_rank_comm.calls == 2 * len(keys)
    for engines in ("_engines", "_moe16_engines"):
        a, b = getattr(moes[0], engines), getattr(moes[1], engines)
        assert a.keys() == b.keys()
        for key in a:
            assert (a[key] is None) == (b[key] is None) == any(fail), (engines, key)
    # Both ranks then take the same path in the forward: the engine, or (after a miss) the built-in MoE.
    dispatch.set_require(False)
    for n in sorted(dispatch.MOE_ENGINE.keys):
        got = [m.engine(n, shared=True) for m in moes]
        assert (got[0] is None) == (got[1] is None)


@pytest.mark.parametrize("capturing", [False, True])
def test_fc_join_records_output_on_main(monkeypatch, capturing):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.spec_worker import (  # noqa: E501
        M3Eagle3Worker,
    )

    main, fork = mock.Mock(name="main"), mock.Mock(name="fork")
    hidden = mock.Mock(name="hidden", is_cuda=True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: main)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    worker = SimpleNamespace(_m3_fc_hidden=hidden)
    M3Eagle3Worker._m3_fc_join(worker, fork)
    main.wait_stream.assert_called_once_with(fork)
    if capturing:
        hidden.record_stream.assert_not_called()
    else:
        hidden.record_stream.assert_called_once_with(main)
    worker._m3_fc_hidden = None
    M3Eagle3Worker._m3_fc_join(worker, fork)
