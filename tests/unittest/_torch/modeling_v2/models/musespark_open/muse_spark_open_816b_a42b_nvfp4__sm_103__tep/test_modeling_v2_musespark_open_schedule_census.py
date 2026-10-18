# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host test: the schedule test's census of the paths between consecutive hop calls.

``_census`` (in ``test_modeling_v2_musespark_open_schedule.py``) looks, between
two consecutive hop calls of a captured graph, for a path whose programmatically entered
kernels each wait before they exit (``WAIT_BEFORE_EXIT``) -- or are entered through the
programmatic port of a kernel that never triggers its dependents early
(``NO_EARLY_TRIGGER``): such a kernel launches only once every thread block of its
predecessor has exited, after that predecessor's own wait. On a full-attention layer of the
tier-4 target set this is the trtllm-gen decode FMHA after ``prep_append``'s prep-append, whose thread
blocks with an empty KV range exit without waiting. These graphs are built by hand.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FMHA = (
    "fmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvCausalP32MultiCtasKvCgaVarSeqQ8Kv256"
    "StaticGroupedSwapsAbForGen"
)
CONSUMER = "kernel_cutlass_kernel_ops_allreduce_post_attn_hop_PostAttnConsumerKernel_object_at_0"
NVJET = "nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"


@pytest.fixture(scope="module")
def schedule():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(
        "_schedule_for_census",
        os.path.join(HERE, "test_modeling_v2_musespark_open_schedule.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(schedule, prep="_preattn_prep_append_kernel", port=None):
    """hop -> router shard -> prep -> FMHA -> norm-gate -> o_proj -> consumer hop, one stream."""
    raws = {
        1: "kernel_cutlass_kernel_ops_allreduce_post_attn_hop_PostAttnHopKernel_object_at_0",
        2: "_router_shard_kernel",
        3: prep,
        4: FMHA,
        5: "_attention_norm_gate_pdl_kernel",
        6: NVJET,
        7: CONSUMER,
    }
    names = {node: schedule._short_name(raw) for node, raw in raws.items()}
    programmatic = schedule.PORT_PROGRAMMATIC
    edges = [(a, a + 1, programmatic) for a in range(1, 7)]
    if port is not None:
        edges[2] = (3, 4, port)
    return names, raws, edges


def test_the_fmha_after_a_trigger_free_prep_append_needs_no_wait(schedule):
    names, raws, edges = _graph(schedule)
    assert names[4] == "Sm100fKernel" and names[7] == "PostAttnConsumerKernel"
    row = schedule._census(names, raws, edges)["hop"][0]
    assert row["pair"] == "PostAttnHopKernel -> PostAttnConsumerKernel" and row["all_pdl"]
    assert row["audited_path"] is not None
    assert row["audited_no_trigger"] == {
        "_preattn_prep_append_kernel": "_preattn_prep_append_kernel"
    }
    assert set(row["audited_waits"].values()) == {
        "_router_shard_kernel",
        "_preattn_prep_append_kernel",
        "_attention_norm_gate_pdl_kernel",
        NVJET,
    }
    # The FMHA is relied on for nothing; it is still the one wait the fewest-waits path
    # has that WAIT_BEFORE_EXIT does not name.
    assert FMHA not in row["audited_waits"] and row["unaudited"] == ["Sm100fKernel"]


@pytest.mark.parametrize(
    "prep, port",
    [
        ("_preattn_prep_append_kernel", 2),
        ("_preattn_prep_append_kernel", -1),
        ("_preattn_prep_kernel", None),
    ],
    ids=("launch-order-port", "no-port", "a-predecessor-that-may-trigger"),
)
def test_only_the_programmatic_port_of_a_trigger_free_kernel_carries_it(schedule, prep, port):
    names, raws, edges = _graph(schedule, prep, port)
    row = schedule._census(names, raws, edges)["hop"][0]
    assert row["all_pdl"] and row["audited_path"] is None
    assert row["audited_waits"] == {} and row["audited_no_trigger"] == {}
    assert "Sm100fKernel" in row["unaudited"]


def test_a_full_edge_into_the_fmha_leaves_no_all_pdl_pair(schedule):
    names, raws, edges = _graph(schedule, port=0)
    row = schedule._census(names, raws, edges)["hop"][0]
    assert not row["all_pdl"] and row["plain"] == {"Sm100fKernel": 1}
    assert row["audited_path"] is not None and row["audited_no_trigger"] == {}
