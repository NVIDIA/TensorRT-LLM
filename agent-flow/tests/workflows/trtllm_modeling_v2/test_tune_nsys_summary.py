# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nsys_summary on a synthetic sqlite export: step classes, graph-replay placement, families."""

import sqlite3

from agent_flow.workflows.trtllm_modeling_v2.tune.nsys_summary import (
    family,
    render,
    summarize,
    union_us,
)

US = 1_000  # ns per us


def _capture(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    cur.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, correlationId INTEGER, nameId INTEGER)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, "
        "shortName INTEGER, graphId INTEGER, graphNodeId INTEGER)"
    )
    names = {
        1: "cudaGraphLaunch_v10000",
        2: "cudaLaunchKernel_v7000",
        10: "flashinferNormKernels::rmsnorm",
        11: "nvjet_tst_64x8",
        12: "fmha_v2_flash_attention",
        13: "chunk_gated_delta_rule_fwd",
    }
    cur.executemany("INSERT INTO StringIds VALUES (?, ?)", names.items())
    cur.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)",
        [
            (0, 1000 * US, "[Executor] _forward_step 1: 0 ctx reqs, 0 ctx tokens, 4 gen reqs"),
            (
                1200 * US,
                3000 * US,
                "[Executor] _forward_step 2: 1 ctx reqs, 8192 ctx tokens, 0 gen reqs",
            ),
            (1300 * US, 1350 * US, "_prepare_inputs"),
        ],
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
        [
            (100 * US, 1, 1),  # step 1 launches its graph
            (1300 * US, 2, 2),
            (1400 * US, 3, 2),
        ],
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
        [
            # step 1's replay runs on the GPU while the host is already inside step 2 (overlap scheduler)
            (1500 * US, 1600 * US, 0, 10, 7, 1),
            (1600 * US, 1800 * US, 0, 11, 7, 2),
            (2000 * US, 2500 * US, 2, 12, None, None),
            (2500 * US, 2600 * US, 3, 13, None, None),
        ],
    )
    con.commit()
    con.close()


def test_a_graph_replay_is_charged_to_the_step_that_launched_it(tmp_path):
    db = tmp_path / "cap.sqlite"
    _capture(db)
    s = summarize(str(db))
    decode, prefill = s["classes"]["decode"], s["classes"]["prefill"]
    assert decode["graph_replay"] and decode["kernels_per_step"] == 2
    assert decode["gpu_busy_us"] == 300.0 and decode["idle_in_span_pct"] == 0.0
    assert decode["family_us_per_step"] == {"gemm": 200.0, "norm": 100.0}
    assert prefill["kernels_per_step"] == 2 and not prefill["graph_replay"]
    assert prefill["family_us_per_step"] == {"attention": 500.0, "recurrence": 100.0}
    assert prefill["top_kernels"][0]["name"] == "fmha_v2_flash_attention"
    assert s["host_us"] == {"_prepare_inputs": 50.0}
    assert "## decode: 1 steps" in render(s)


def test_busy_time_is_the_union_of_kernel_intervals():
    assert union_us([(0, 1000), (500, 1500), (3000, 4000)]) == 2.5
    assert (
        family("void cutlass::Kernel<moe_gemm>") == "gemm"
        and family("topk_routing_kernel") == "moe"
    )
