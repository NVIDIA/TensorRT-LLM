# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode robustness without a GPU kernel: the multicast all-reduce epilogue options keep the cluster-barrier bit,
an engine build that fails on one TP rank drops that engine on every rank (two ranks, and the tp8 target's eight,
simulated with threads), and the Eagle3 fc fork's output is recorded on the stream that reads it."""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops import (  # noqa: E501
    m3_mc_allreduce as mc,
)


@pytest.fixture(autouse=True)
def _restore_modes():
    require = modeling.require_mode()
    yield
    modeling.set_require(require)
    modeling.reset_logged()


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


def _patch_comm(monkeypatch, world: int) -> _FakeComm:
    from tensorrt_llm._torch.distributed import ops

    comm = _FakeComm(world)
    monkeypatch.setattr(mc, "_tp_comm", lambda mapping: comm)
    monkeypatch.setattr(ops, "mpi_disabled", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return comm


@pytest.fixture
def two_rank_comm(monkeypatch):
    return _patch_comm(monkeypatch, 2)


def test_tp_agree_single_rank_and_adp_take_no_collective(monkeypatch):
    monkeypatch.setattr(mc, "_tp_comm", mock.Mock(side_effect=AssertionError("no collective")))
    assert mc.tp_agree(_mapping(1), True) is True
    assert mc.tp_agree(_mapping(1), False) is False
    assert mc.tp_agree(_mapping(8, adp=True), False) is False


def test_tp_agree_refuses_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError):
        mc.tp_agree(_mapping(2), True)


@pytest.mark.parametrize("oks", [(True, True), (True, False), (False, True), (False, False)])
def test_tp_agree_built_two_ranks(two_rank_comm, oks):
    built = [object() if ok else None for ok in oks]
    out = _run_ranks([lambda i=i: mc.tp_agree_built(_mapping(2), built[i], "t") for i in range(2)])
    assert out == (built if all(oks) else [None, None])


def _core(fail: bool, world: int = 2) -> SimpleNamespace:
    """The engine-building half of the core (``MiniMaxM3Tp8ModelingV2``) on one of ``world`` simulated TP ranks,
    its engine builds stubbed (and failing where ``fail``)."""
    core = SimpleNamespace(mapping=_mapping(world), moe_engines={})
    cls = modeling.MiniMaxM3Tp8ModelingV2

    def build(i, num_tokens, shared=True):
        if fail:
            raise RuntimeError("compile failed on this rank")
        return ("engine", num_tokens)

    core._build_engine = build
    for name in ("_prebuild_engines", "_try_build", "engine", "_lookup"):
        setattr(core, name, getattr(cls, name).__get__(core))
    return core


@pytest.mark.parametrize(
    "world,failing",
    [(2, ()), (2, (1,)), (8, ()), (8, (5,)), (8, (0, 7))],
    ids=["w2_ok", "w2_rank1_fails", "w8_ok", "w8_rank5_fails", "w8_ranks0_7_fail"],
)
def test_engine_build_agreed_across_ranks(monkeypatch, world, failing):
    comm = _patch_comm(monkeypatch, world)
    cores = [_core(rank in failing, world) for rank in range(world)]
    _run_ranks([lambda c=c: c._prebuild_engines(3) for c in cores])
    keys = sorted(cores[0].moe_engines[3])
    assert keys and comm.calls == world * len(keys)
    assert not any(hasattr(c, "moe16_engines") for c in cores)
    for c in cores[1:]:
        assert c.moe_engines[3].keys() == cores[0].moe_engines[3].keys()
    for key in keys:
        for c in cores:
            assert (c.moe_engines[3][key] is None) == bool(failing), key
    # Every rank then takes the same path in the forward: the engine, or (after a miss) the generic MoE.
    modeling.set_require(False)
    for n in sorted(modeling.MOE_ENGINE.keys):
        got = [c.engine(3, n, shared=True) is None for c in cores]
        assert len(set(got)) == 1, (n, got)


@pytest.mark.parametrize("capturing", [False, True])
def test_fc_join_records_output_on_main(monkeypatch, capturing):
    M3Eagle3Worker = modeling.M3Eagle3Worker
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
