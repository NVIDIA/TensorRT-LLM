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
"""deferred_input_prep's executor-loop domain, its engagement rule, its debug level and its window (CPU only).

The step-input stager engages when the target declares it (the plan runs
``deferred_input_prep``) and the step is in its support domain; the store writer engages
once a non-attention-DP overlap loop declared its domain. Nothing depends on which other
packages are selected or on their sources. An engaged preparation opens the deferred-write window, which
``PrepStager.flush`` can close early for the writes staged so far, also handing ``input_ids`` /
``position_ids`` back to the graph replay's own copy.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import input_prep_testlib  # noqa: E402

HOST = torch.arange(10, 14, dtype=torch.int32)


@pytest.fixture
def deferred_input_prep(monkeypatch):
    """deferred_input_prep's host modules with logging captured in ``deferred_input_prep.lines``."""
    modules = input_prep_testlib.load()
    lines = []
    monkeypatch.setattr(modules.dispatch, "_log_info", lines.append)
    monkeypatch.setattr(modules.dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(modules.dispatch, "_INSTALLED", [None])
    monkeypatch.setattr(
        modules.compat, "_LOOP_STATE", {"declared": False, "reason": "", "logged": False}
    )
    modules.lines = lines
    return modules


@pytest.fixture
def compat(deferred_input_prep):
    return deferred_input_prep.compat


def test_loop_domain_state(compat):
    """Store writer: only after a non-ADP overlap loop declared."""
    state = compat._LOOP_STATE
    assert compat.loop_domain() is False
    compat.declare_overlap_loop(False)
    assert compat.loop_domain() is True
    compat.declare_overlap_loop(True)
    assert compat.loop_domain() is False and "attention DP" in state["reason"]
    compat.declare_overlap_loop(False)
    assert compat.loop_domain() is False
    state.update(declared=False, reason="")
    compat.declare_overlap_loop(None)
    assert compat.loop_domain() is False and "did not pass" in state["reason"]


def test_every_part_engages_without_any_record(deferred_input_prep, monkeypatch):
    """A shared switch set to a value no oracle ran: deferred_input_prep engages.

    Engagement reads no package source and no table of package versions, so
    editing another package's sources cannot turn deferred_input_prep off.
    """
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    monkeypatch.setattr(deferred_input_prep.stager, "_host_memory_pinned", lambda: True)
    st = deferred_input_prep.stager.PrepStager()
    st.packet = input_prep_testlib.FakePacket(deferred_input_prep.packet)
    case = input_prep_testlib.eligibility_case()
    attached = (
        case["runner"],
        case["attn_metadata"],
        case["spec_metadata"],
        case["attn_metadata"].kv_cache_manager,
    )
    assert st.begin_if_eligible(**case) is st
    # The attribute every engine call site reads (RuntimeHooks.step_input_stager).
    assert all(obj.step_input_stager is st for obj in attached)
    st.end()
    assert all(obj.step_input_stager is None for obj in attached)
    assert st.decisions == {"engaged": 1} and st.num_commits == 1
    assert deferred_input_prep.lines[0].startswith(
        "[glm_moe_dsa] deferred_input_prep/prep: engaged"
    )
    deferred_input_prep.compat.declare_overlap_loop(False)
    assert deferred_input_prep.compat.loop_domain()


def test_prep_declines_only_for_pageable_host_memory(deferred_input_prep, monkeypatch):
    """Whether a stager exists is the plan's decision; the stager declines only pageable memory."""
    monkeypatch.setattr(deferred_input_prep.stager, "_host_memory_pinned", lambda: True)
    st = deferred_input_prep.stager.PrepStager()
    st.packet = input_prep_testlib.FakePacket(deferred_input_prep.packet)
    assert st.begin_if_eligible(**input_prep_testlib.eligibility_case()) is st
    st.end()
    monkeypatch.setattr(deferred_input_prep.stager, "_host_memory_pinned", lambda: False)
    st = deferred_input_prep.stager.PrepStager()
    assert st.begin_if_eligible(**input_prep_testlib.eligibility_case()) is None
    assert st.last_reason == "host page tables are not pinned (prefer_pinned)"
    assert not st.active


def test_debug_level_is_the_constructor_argument(deferred_input_prep, monkeypatch):
    """The stager reads no environment: its debug level is the argument (0, 1 or 2)."""
    monkeypatch.setattr(deferred_input_prep.stager, "_host_memory_pinned", lambda: True)
    monkeypatch.setenv("TRTLLM_MODELING_V2_VALIDATE", "1")
    PrepStager = deferred_input_prep.stager.PrepStager
    assert PrepStager().debug is None
    assert PrepStager(debug_level=1).debug.level == 1
    assert PrepStager(debug_level=2).debug.level == 2
    with pytest.raises(ValueError, match="0, 1 or 2"):
        PrepStager(debug_level=3)


def _engaged(deferred_input_prep, monkeypatch, packet=None, debug_level=0):
    """An engaged stager on a CPU case whose runner owns a 16-element ``input_ids_cuda``."""
    monkeypatch.setattr(deferred_input_prep.stager, "_host_memory_pinned", lambda: True)
    case = input_prep_testlib.eligibility_case()
    case["runner"].input_ids_cuda = torch.zeros(16, dtype=torch.int32)
    st = deferred_input_prep.stager.PrepStager(debug_level=debug_level)
    st.packet = (
        packet if packet is not None else input_prep_testlib.FakePacket(deferred_input_prep.packet)
    )
    assert st.begin_if_eligible(**case) is st
    return st, case


def test_engaged_line_states_the_window_contract(deferred_input_prep, monkeypatch):
    st, _ = _engaged(deferred_input_prep, monkeypatch)
    st.end()
    assert deferred_input_prep.lines == [
        "[glm_moe_dsa] deferred_input_prep/prep: engaged (deferred-write window from "
        "DecoderRunner._forward_decoder's begin_if_eligible to stager.end(): the buffers written "
        "through the deferred_input_prep stager are written later by the commit; do not read or "
        "write them in between with other device ops; route writes through the stager API or "
        "call stager.flush() first; for input_ids_cuda and position_ids_cuda (also staged into "
        "the graph's static inputs) call stager.flush() and then write natively; "
        "disable with TRTLLM_STAIRCASE_GLM_LL_DISABLE=deferred_input_prep)"
    ]


def test_flush_then_another_write_keeps_native_order(deferred_input_prep, monkeypatch):
    """After flush, a write to a staged buffer lands after the staged write, as natively."""
    st, case = _engaged(deferred_input_prep, monkeypatch)
    buf = case["runner"].input_ids_cuda
    st.copy(buf[:4], HOST)
    st.flush()
    assert buf[:4].tolist() == HOST.tolist() and st.num_flushes == 1
    buf[:4].add_(100)
    st.zero(buf[4:])
    st.end()
    assert st.packet.num_commits == 2 and st.num_commits == 1
    assert buf.tolist() == (HOST + 100).tolist() + [0] * 12


@pytest.mark.parametrize("max_desc", [0, 2, 1 << 20])
def test_flush_issues_several_writes_in_staging_order(deferred_input_prep, monkeypatch, max_desc):
    """Overlapping writes staged before one flush land in staging order.

    ``max_desc`` 0: every write is native; 2: the packet fills up; otherwise
    only the overlaps split the packet.
    """
    packet = input_prep_testlib.FakePacket(deferred_input_prep.packet, max_desc=max_desc)
    st, case = _engaged(deferred_input_prep, monkeypatch, packet=packet)
    buf = case["runner"].input_ids_cuda
    buf.fill_(5)
    st.copy(buf[:4], torch.tensor([-3, 1, 7, -9], dtype=torch.int32))
    st.zero(buf[4:])
    st.clamp_min(buf[:4], 0)
    st.values(buf[:2], [-1, -2])
    st.flush()
    expected = [-1, -2, 7, 0] + [0] * 12
    assert buf.tolist() == expected
    st.end()
    assert buf.tolist() == expected


def test_flush_and_commit_fall_back_to_native_writes(deferred_input_prep, monkeypatch):
    """With a packet that holds nothing, flush and the commit issue each write natively."""
    full = input_prep_testlib.FakePacket(deferred_input_prep.packet, max_desc=0)
    st, case = _engaged(deferred_input_prep, monkeypatch, packet=full)
    buf = case["runner"].input_ids_cuda
    st.copy(buf[:4], HOST)
    st.flush()
    assert buf[:4].tolist() == HOST.tolist() and st.num_native_copies == 1
    st.values(buf[4:8], [1, 2, 3, 4])
    st.zero(buf[8:])
    st.end()
    assert st.num_native_copies == 3 and full.num_writes == 0
    assert buf.tolist() == HOST.tolist() + [1, 2, 3, 4] + [0] * 8
    st.flush()  # outside a window: nothing to do
    assert st.num_native_copies == 3


def test_flush_ends_the_guarded_span_of_the_flushed_writes(deferred_input_prep, monkeypatch):
    """Debug guard: a flushed buffer may be touched; a buffer staged later stays guarded."""
    guard = sys.modules[deferred_input_prep.stager.DebugWindow.__module__]

    def cpu_span(t):
        lo = t.data_ptr()
        return (lo, lo + t.numel() * t.element_size()) if t.numel() else None

    monkeypatch.setattr(guard, "tensor_span", cpu_span)
    st, case = _engaged(deferred_input_prep, monkeypatch, debug_level=1)
    buf = case["runner"].input_ids_cuda
    st.copy(buf[:4], HOST)
    st.flush()
    buf[:4].add_(100)
    st.zero(buf[4:])
    try:
        with pytest.raises(guard.InputPrepHazardError, match="touches a destination staged"):
            buf[4:].add_(1)
    finally:
        st.discard()
    assert buf.tolist() == (HOST + 100).tolist() + [0] * 12


STEP_TOKENS = 4
STAGED_INPUT_IDS = [11, 21, 12, 22]
STAGED_POSITIONS = [7, 8, 9, 10]


def _stage_input_ids(st, runner):
    """``overlap_inputs`` of a steady step: two rows with previous tensors, 2 tokens per row."""
    new_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32)
    next_draft_tokens = torch.tensor([[31], [32], [33], [34]], dtype=torch.int32)
    new_tokens_lens = torch.tensor([2, 1, 2, 2], dtype=torch.int32)
    st.overlap_inputs(
        runner, [0, 1], [0, 1, 2, 3], 0, 0, 2, 2, new_tokens, next_draft_tokens, new_tokens_lens
    )


def _stage_position_ids(st, runner):
    st.position_ids(
        runner,
        runner.position_ids_cuda[:STEP_TOKENS],
        torch.tensor(STAGED_POSITIONS, dtype=torch.int32),
    )


def _graph_step(deferred_input_prep, monkeypatch):
    """An engaged step whose runner has the decode graph's static input buffers."""
    st, case = _engaged(deferred_input_prep, monkeypatch)
    runner = case["runner"]
    for name in (
        "position_ids_cuda",
        "draft_tokens_cuda",
        "previous_batch_indices_cuda",
        "previous_pos_indices_cuda",
        "previous_pos_id_offsets_cuda",
        "previous_kv_lens_offsets_cuda",
    ):
        setattr(runner, name, torch.zeros(16, dtype=torch.int32))
    static = {
        "input_ids": torch.zeros(16, dtype=torch.int32),
        "position_ids": torch.zeros((1, 16), dtype=torch.int32),
    }
    runner.cuda_graph_runner = SimpleNamespace(shared_static_tensors=static)
    return st, runner, static


def _graph_inputs(st, runner, static):
    """``input_ids`` / ``position_ids`` as ``CUDAGraphRunner.replay`` hands them to the graph."""
    n = STEP_TOKENS
    if st.static_inputs_tokens() != n:
        static["input_ids"][:n].copy_(runner.input_ids_cuda[:n])
        static["position_ids"][0, :n].copy_(runner.position_ids_cuda[:n])
    return static["input_ids"][:n].tolist(), static["position_ids"][0, :n].tolist()


def test_staged_step_writes_the_graph_inputs(deferred_input_prep, monkeypatch):
    """The commit writes the static inputs too, so the replay skips its own copy."""
    st, runner, static = _graph_step(deferred_input_prep, monkeypatch)
    _stage_input_ids(st, runner)
    _stage_position_ids(st, runner)
    st.end()
    assert st.static_inputs_tokens() == STEP_TOKENS
    assert runner.input_ids_cuda[:STEP_TOKENS].tolist() == STAGED_INPUT_IDS
    assert _graph_inputs(st, runner, static) == (STAGED_INPUT_IDS, STAGED_POSITIONS)


def test_native_writes_after_flush_reach_the_graph_inputs(deferred_input_prep, monkeypatch):
    """flush, then native writes to input_ids_cuda / position_ids_cuda: the graph sees them."""
    st, runner, static = _graph_step(deferred_input_prep, monkeypatch)
    _stage_input_ids(st, runner)
    _stage_position_ids(st, runner)
    st.flush()
    runner.input_ids_cuda[:STEP_TOKENS].copy_(torch.tensor([101, 102, 103, 104]))
    runner.position_ids_cuda[:STEP_TOKENS].add_(100)
    st.end()
    assert st.static_inputs_tokens() == 0
    assert _graph_inputs(st, runner, static) == (
        [101, 102, 103, 104],
        [p + 100 for p in STAGED_POSITIONS],
    )


def test_flush_between_input_ids_and_position_ids(deferred_input_prep, monkeypatch):
    """Only position_ids is staged again after the flush: the replay copies both inputs."""
    st, runner, static = _graph_step(deferred_input_prep, monkeypatch)
    _stage_input_ids(st, runner)
    st.flush()
    runner.input_ids_cuda[:STEP_TOKENS].add_(100)
    _stage_position_ids(st, runner)
    st.end()
    assert st.static_inputs_tokens() == 0
    assert _graph_inputs(st, runner, static) == (
        [t + 100 for t in STAGED_INPUT_IDS],
        STAGED_POSITIONS,
    )


def test_staged_write_to_input_ids_does_not_reach_the_graph(deferred_input_prep, monkeypatch):
    """Why the window contract sends new input_ids_cuda writes through flush.

    The static inputs staged by ``overlap_inputs`` are what the graph replays;
    a later staged write to ``input_ids_cuda`` changes only the runner buffer.
    """
    st, runner, static = _graph_step(deferred_input_prep, monkeypatch)
    _stage_input_ids(st, runner)
    _stage_position_ids(st, runner)
    st.copy(runner.input_ids_cuda[:STEP_TOKENS], torch.tensor([101, 102, 103, 104]))
    st.end()
    assert runner.input_ids_cuda[:STEP_TOKENS].tolist() == [101, 102, 103, 104]
    assert _graph_inputs(st, runner, static)[0] == STAGED_INPUT_IDS
