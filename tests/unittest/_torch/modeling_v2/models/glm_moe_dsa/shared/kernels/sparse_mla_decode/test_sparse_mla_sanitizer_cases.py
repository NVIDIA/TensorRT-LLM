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
"""Small sparse_mla_decode cases for compute-sanitizer (memcheck, racecheck, synccheck, initcheck).

Run under ``compute-sanitizer --tool <tool> --kernel-name kns=sparseMlaDecode``
so that only the sparse_mla_decode kernel is instrumented. The cases cover every code path of
the kernel at shapes small enough for instrumented execution: short rows with
fewer usable positions than the top-k (zero-filled gather slots), rows that
straddle 2048 usable positions, full rows, R = 6 and R = 1, several sequences,
empty rows, negative and repeated ids, and CUDA-graph capture plus replay with
rewritten inputs. Each case also checks the output against the native library
and the FP64 oracle with the frozen error budget of ``test_sparse_mla_sparse_mla.py``.
"""

import pytest
import torch
from sparse_mla_harness import DsaGenHarness, compare, error_stats, oracle_attention, record

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="needs an SM10x GPU and the sparse_mla_decode kernel",
)

RESULTS = "sparse_mla_sanitizer_cases.jsonl"


@pytest.mark.parametrize(
    "ctx_lens,seq_len_q",
    [
        ([100], 6),
        ([100], 1),
        ([2043], 6),
        ([2043], 1),
        ([8192], 6),
        ([8192], 1),
        ([3000, 100, 9000], 2),
    ],
)
def test_sparse_mla_small_matches_native(ctx_lens, seq_len_q):
    harness = DsaGenHarness(ctx_lens, seq_len_q=seq_len_q, seed=11)
    try:
        topk = harness.random_topk(seed=12, overlap=0.7, recent=32)
        step = harness.make_step(0, seed=13)
        compare(
            harness,
            step,
            topk,
            "small_matches_native",
            RESULTS,
            ctx_lens=ctx_lens,
            seq_len_q=seq_len_q,
        )
    finally:
        harness.shutdown()


def test_sparse_mla_small_empty_negative_and_repeated():
    harness = DsaGenHarness([4096, 40], seq_len_q=6, seed=14)
    try:
        topk = harness.random_topk(seed=15)
        topk[1, :] = -1
        topk[3, :1500] = 17
        topk[3, 1500:] = 4242
        for row in range(6, 12):
            for pos in (300, 1500, 2047):
                topk[row, pos] = 3
        step = harness.make_step(0, seed=16)
        compare(harness, step, topk, "small_empty_negative_repeated", RESULTS)
        # Negative ids inside the causal range: sparse_mla_decode skips them. Only
        # sparse_mla_decode's output is checked here (the native kernel reads the row that -1
        # addresses).
        topk = harness.random_topk(seed=17)
        topk[2, ::3] = -1
        topk[5, :1024] = -1
        harness.select_library(0, use_sparse_mla=True)
        sparse_out = harness.forward(0, step, topk).clone()
        global_idx = harness.global_indices(0, topk)
        oracle = oracle_attention(
            step.quant_q,
            harness.pool_rows(),
            global_idx,
            float(step.bmm1_scale[0]),
            float(step.bmm2_scale[0]),
            list_limit=harness.list_limits(),
        )
        err = error_stats(sparse_out.double(), oracle)
        record(RESULTS, "small_negative_inside_range_sparse_mla", {"sparse_mla_vs_oracle": err})
        assert err["nonfinite"] == 0 and err["nrms"] < 0.05, err
    finally:
        harness.shutdown()


def test_sparse_mla_small_graph_replay():
    harness = DsaGenHarness([8192], seq_len_q=6, seed=18)
    try:
        step = harness.make_step(0, seed=19)
        topk = harness.random_topk(seed=20, overlap=0.7, recent=32)
        harness.select_library(0, use_sparse_mla=True)
        harness.forward(0, step, topk)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = harness.forward(0, step, topk)
        pool_rows = harness.pool_rows()
        for it in range(3):
            topk.copy_(harness.random_topk(seed=300 + it, overlap=0.5, recent=16))
            global_idx = harness.global_indices(0, topk)
            rows = global_idx[global_idx >= 0].unique().long()
            fresh = (
                (torch.randn(rows.numel(), pool_rows.shape[1], device="cuda") * (1 + it))
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            pool_rows.view(torch.uint8)[rows] = fresh.view(torch.uint8)
            fresh_step = harness.make_step(0, seed=400 + it)
            for name in ("fused_q", "q_pe", "latent", "quant_q", "bmm1_scale", "bmm2_scale"):
                getattr(step, name).copy_(getattr(fresh_step, name))
            graph.replay()
            torch.cuda.synchronize()
            oracle = oracle_attention(
                step.quant_q,
                pool_rows,
                global_idx,
                float(step.bmm1_scale[0]),
                float(step.bmm2_scale[0]),
                list_limit=harness.list_limits(),
            )
            err = error_stats(out.double(), oracle)
            record(RESULTS, "small_graph_replay", {"iteration": it, "sparse_mla_vs_oracle": err})
            assert err["nonfinite"] == 0 and err["nrms"] < 0.05, err
    finally:
        harness.shutdown()
