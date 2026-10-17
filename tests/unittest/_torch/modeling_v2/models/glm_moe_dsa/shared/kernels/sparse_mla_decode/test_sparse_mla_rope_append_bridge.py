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
"""rope_kv_append's fused top-k bridge as the producer of sparse_mla_decode's global indices.

With the plan running ``sparse_mla_decode`` and ``rope_kv_append`` (installed by this test) the DSA
predictor's local-to-global remap is ``remap_topk_to_global``
(``trtllm::glm_moe_dsa_topk_index_bridge``), and its output is the index list sparse_mla_decode
reads. sparse_mla_decode differs from the native kernel on one
input class: a ``-1`` at an index-list position below the row's causal length (sparse_mla_decode
skips it, the native kernel reads the pool row that ``-1`` addresses). For every
generation call the bridge serves, this test asserts that

* the bridge ran (counted per call, not read from the capability log, whose
  first ``rope_kv_append/bridge`` decision is the declined context step),
* its global indices equal the native remap of the same local top-k bit for
  bit and contain no ``-1`` below the row's causal length,
* sparse_mla_decode received exactly those indices, and sparse_mla_decode stays within the frozen
  error budget against the FP64 oracle,
* CUPTI shows rope_kv_append's ``topkBridgeKernel`` and sparse_mla_decode's
  ``sparseMlaDecodeKernel`` once per generation call, and neither the native remap kernel nor the
  native sparse FMHA kernel.

A negative control puts ``-1`` inside the causal range of one row: the bridge
passes it through exactly like the native remap, and the check above reports
it, so the assertion is not vacuous.

Runs only when both ops are in the build.
"""

import os
import sys

import pytest
import torch
from sparse_mla_harness import (
    SPARSE_MLA_KERNEL,
    DsaGenHarness,
    capture_rope_append_bridge,
    check_budget,
    is_native_sparse_fmha,
    kernel_trace,
    negative_inside_causal,
    oracle_for,
    record,
    rope_append_bridge_report,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available()
    or not hasattr(torch.ops.trtllm, "glm_moe_dsa_topk_index_bridge"),
    reason="needs an SM10x build with the sparse_mla_decode and rope_kv_append kernels",
)


@pytest.fixture(autouse=True)
def _plan(monkeypatch):
    plan_testlib.select(dispatch, "sparse_mla_decode,rope_kv_append", monkeypatch)


RESULTS = "sparse_mla_rope_append_bridge.jsonl"
CASES = {
    "ctx100_r6": ([100], 6),  # every row ends in -1 entries past its causal length
    "ctx2042_r6": ([2042], 6),  # the last row sees exactly 2048 tokens
    "ctx2043_r6": ([2043], 6),  # rows straddle 2048 visible tokens
    "two_seqs_r6": ([5000, 300], 6),  # the second sequence has -1 tails
    "four_seqs_r2": ([20000, 20000, 20000, 20000], 2),
    "ctx32768_r6": ([32768], 6),  # target verify, MTP K = 5
    "ctx131072_r1": ([131072], 1),  # draft steps 1-4
}


def _generation_kernels(harness, step, topk):
    kernels = kernel_trace(lambda: harness.forward(0, step, topk))
    names = [k["name"] for k in kernels]
    return {
        "rope_append_bridge": sum("topkBridgeKernel" in n for n in names),
        "native_remap": sum("convertReqIndexToGlobal" in n for n in names),
        "sparse_mla_decode": sum(SPARSE_MLA_KERNEL in n for n in names),
        "native_sparse_fmha": sum(is_native_sparse_fmha(n) for n in names),
    }


@pytest.mark.parametrize("case", list(CASES))
def test_rope_append_bridge_feeds_sparse_mla(case):
    ctx_lens, seq_len_q = CASES[case]
    harness = DsaGenHarness(ctx_lens, seq_len_q=seq_len_q, seed=31, mtp_draft_len=seq_len_q - 1)
    try:
        step = harness.make_step(0, seed=32)
        topk = harness.random_topk(seed=33, overlap=0.7, recent=32)
        seq_lens_kv = torch.tensor(
            [c + seq_len_q for c in ctx_lens], dtype=torch.int32, device="cuda"
        )
        with (
            capture_rope_append_bridge() as bridge,
            harness.capture_sparse_mla_inputs() as sparse_mla_calls,
        ):
            harness.select_library(0, use_sparse_mla=False)
            native = harness.forward(0, step, topk).clone()
            harness.select_library(0, use_sparse_mla=True)
            sparse_out = harness.forward(0, step, topk).clone()
            torch.cuda.synchronize()
        assert harness.selected_library(0) == ["SparseMlaDecodeFmha"]
        # One bridge call per generation forward (native arm, then sparse_mla_decode arm).
        assert len(bridge) == 2, len(bridge)
        assert len(sparse_mla_calls) == 1, len(sparse_mla_calls)
        received = sparse_mla_calls[0]
        assert torch.equal(received["seq_lens_kv"], seq_lens_kv)
        reports = [rope_append_bridge_report(call, seq_lens_kv, seq_len_q) for call in bridge]
        same_as_bridge = torch.equal(received["indices"], bridge[1]["global"])
        global_idx = harness.global_indices(0, topk)
        oracle = oracle_for(harness, step, global_idx)
        kernels = _generation_kernels(harness, step, topk)
        check_budget(
            harness,
            native,
            sparse_out,
            global_idx,
            oracle,
            "rope_append_bridge_feeds_sparse_mla",
            RESULTS,
            shape=case,
            bridge=reports,
            sparse_mla_received_bridge_output=same_as_bridge,
            generation_kernels=kernels,
        )
        for report in reports:
            assert report["equals_native_remap"], report
            assert sum(report["negative_below_causal"]) == 0, report
        assert same_as_bridge
        assert kernels == {
            "rope_append_bridge": 1,
            "native_remap": 0,
            "sparse_mla_decode": 1,
            "native_sparse_fmha": 0,
        }, kernels
    finally:
        harness.shutdown()


def test_rope_append_bridge_negative_control():
    """``-1`` injected inside the causal range: passed through as by the native remap, detected."""
    ctx_lens, seq_len_q = [32768], 6
    harness = DsaGenHarness(ctx_lens, seq_len_q=seq_len_q, seed=41, mtp_draft_len=seq_len_q - 1)
    try:
        step = harness.make_step(0, seed=42)
        topk = harness.random_topk(seed=43, overlap=0.7, recent=32)
        topk[2, ::3] = -1
        injected = int((topk[2] < 0).sum().item())
        seq_lens_kv = torch.tensor([ctx_lens[0] + seq_len_q], dtype=torch.int32, device="cuda")
        with (
            capture_rope_append_bridge() as bridge,
            harness.capture_sparse_mla_inputs() as sparse_mla_calls,
        ):
            harness.select_library(0, use_sparse_mla=True)
            harness.forward(0, step, topk)
            torch.cuda.synchronize()
        assert len(bridge) == 1 and len(sparse_mla_calls) == 1
        report = rope_append_bridge_report(bridge[0], seq_lens_kv, seq_len_q)
        received = negative_inside_causal(
            sparse_mla_calls[0]["indices"], seq_lens_kv, seq_len_q
        ).tolist()
        record(
            RESULTS,
            "rope_append_bridge_negative_control",
            {"injected_row2": injected, "bridge": report, "sparse_mla_received_negative": received},
        )
        assert report["equals_native_remap"], report
        expected = [0] * harness.num_rows
        expected[2] = injected
        assert report["negative_below_causal"] == expected, report
        assert received == expected
    finally:
        harness.shutdown()
