# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 o_proj Lamport (TRTLLM_K3_OPROJ_LAMPORT) host logic, CPU only: batch-size limits,
the switch, the GEMM tactic routing, the mailbox schedule after a failed forward and the slots
kernel options. The bitwise multi-GPU test is
tests/unittest/_torch/multi_gpu/test_k3_oproj_lamport.py."""

import types

import pytest
import torch

from tensorrt_llm.mapping import Mapping

_BASE = ("base", (256, 64, 128), (256, 64, 64), (2, 1), True, False, "static", "m")
_SPLIT_K = ("splitk", (128, 64, 128), (128, 64, 64), (1, 1), True, True, "static", "m", 4)
_MIXED = ("mixed_clusters", (256, 64, 128), (256, 64, 64), (4, 2), (2, 1), False, False, "m")


def test_max_m_defaults_and_validation(monkeypatch):
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    from tensorrt_llm._torch.modules import kimi_k3_oproj_lamport as lam

    tp16 = min(16, _MNNVL_ONE_SHOT_THRESHOLD_BYTES // (7168 * 16 * 2))
    monkeypatch.delenv(lam.OPROJ_LAMPORT_MAX_M_ENV, raising=False)
    assert [lam._max_m(tp) for tp in (2, 4, 8, 16)] == [64, 64, 32, tp16]
    monkeypatch.setenv(lam.OPROJ_LAMPORT_MAX_M_ENV, "24")
    assert lam._max_m(8) == 24
    # Above TP8 the two-shot all-reduce sums in another order: the one-shot range caps the value.
    assert lam._max_m(16) == min(24, _MNNVL_ONE_SHOT_THRESHOLD_BYTES // (7168 * 16 * 2))
    for bad in ("0", "65"):
        monkeypatch.setenv(lam.OPROJ_LAMPORT_MAX_M_ENV, bad)
        with pytest.raises(ValueError):
            lam._max_m(8)


def test_switch_off_builds_nothing(monkeypatch):
    from tensorrt_llm._torch.modules import kimi_k3_oproj_lamport as lam

    monkeypatch.delenv(lam.OPROJ_LAMPORT_ENV, raising=False)
    mapping = Mapping(world_size=8, rank=0, gpus_per_node=8, tp_size=8)
    assert lam.K3OprojLamport.create(mapping) is None


def test_lamport_epilogue_tactics():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import _lamport_epilogue_tactic

    assert _lamport_epilogue_tactic(_BASE)
    assert _lamport_epilogue_tactic(_BASE[1:6])  # the legacy 5-tuple base form
    assert _lamport_epilogue_tactic(-1) and _lamport_epilogue_tactic(None)  # the default
    assert not _lamport_epilogue_tactic(_SPLIT_K)
    assert not _lamport_epilogue_tactic(_MIXED)


def test_gemm_routing_publishes_every_tactic(monkeypatch):
    """Base tactics publish from the epilogue; the others run as is and are copied."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo

    if not hasattr(cdo, "mxfp8_gemm_rubin_lamport"):
        pytest.skip("needs the CuTe DSL")
    from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_lamport_publish

    calls = []

    def runner(inputs, tactic, **kwargs):
        calls.append(("gemm", tactic, kwargs.get("lamport_out")))
        return None if "lamport_out" in kwargs else "stock output"

    def publish(x, slot_address, leading_dim):
        calls.append(("publish", x, slot_address, leading_dim))

    monkeypatch.setattr(k3_lamport_publish, "run_k3_lamport_publish", publish)
    slot, ld = 0x10000, 4 * 7168
    for tactic in (_BASE, -1):
        calls.clear()
        cdo.mxfp8_gemm_rubin_lamport(runner, [], tactic, slot, ld)
        assert calls == [("gemm", tactic, (slot, ld))]
    for tactic in (_SPLIT_K, _MIXED):
        calls.clear()
        cdo.mxfp8_gemm_rubin_lamport(runner, [], tactic, slot, ld)
        assert calls == [("gemm", tactic, None), ("publish", "stock output", slot, ld)]


def _bookkeeping_arena():
    """A K3LamportArena without its allocation: what begin_forward / cancel / poison do."""
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportArena

    arena = K3LamportArena.__new__(K3LamportArena)
    arena._open, arena._poisoned, arena.tp_rank = None, None, 0
    arena.uc_bases, arena.mc_base, arena.region_bytes = [1 << 30], 1 << 31, 1 << 20
    return arena


def test_arena_schedule_after_a_failed_forward():
    arena = _bookkeeping_arena()
    schedule = arena.begin_forward(3)
    schedule.take()
    with pytest.raises(RuntimeError, match="not finished"):
        arena.begin_forward(2)
    schedule.cancel()  # e.g. a failed CUDA-graph capture: nothing ran
    arena.begin_forward(2).cancel()
    arena.poison("test")
    with pytest.raises(RuntimeError, match="unusable"):
        arena.begin_forward(2)


@pytest.mark.parametrize("capturing", [False, True])
def test_failed_forward_abort(monkeypatch, capturing):
    """Outside capture a failed forward poisons the arena and its error is immediately fatal for
    the executor; under capture (nothing ran) the schedule is cancelled and the arena stays usable."""
    from tensorrt_llm._torch.modules import kimi_k3_oproj_lamport as lam
    from tensorrt_llm._torch.pyexecutor.error_classification import classify_error

    assert classify_error(lam.FORWARD_FAILED) == "immediate_fatal"
    arena = _bookkeeping_arena()
    forward = lam.K3OprojLamportForward(types.SimpleNamespace(arena=arena, world=4), 6, [1, 2, 3])
    forward.take()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    assert forward.abort() is capturing
    if capturing:
        arena.begin_forward(2).cancel()
    else:
        with pytest.raises(RuntimeError, match="unusable"):
            arena.begin_forward(2)


def test_slots_options_are_validated():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res as ar

    with pytest.raises(ValueError):
        ar.K3AttnResKernel(3, ar.ADD_PREFIX, True, slots=1)
    with pytest.raises(ValueError):
        ar.K3AttnResKernel(3, ar.ADD_PREFIX, True, slots=32)
    with pytest.raises(ValueError):
        ar.K3AttnResKernel(3, ar.ADD_MOE, True, slots=8)
    with pytest.raises(ValueError):
        ar.K3AttnResKernel(3, ar.ADD_PREFIX, False, slots=8, poll_early=True)
    with pytest.raises(ValueError):
        ar.K3AttnResKernel(3, ar.ADD_PREFIX, True, slots=0, poll_early=True)
    with pytest.raises(ValueError):  # 896 fragments do not fit 2 x 256 threads
        ar.K3AttnResKernel(3, ar.ADD_PREFIX, True, slots=8, poll_ctas=2)
    kernel = ar.K3AttnResKernel(3, ar.ADD_PREFIX, True, slots=8)
    assert (kernel.launch_c, kernel.poll_frags) == (ar.SLOTS_POLL_CTAS, 896 // ar.SLOTS_POLL_CTAS)
    assert ar.K3AttnResKernel(6, ar.ADD_NONE, True, slots=16).launch_c == 8
