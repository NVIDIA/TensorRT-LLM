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
"""Correctness of the sparse_mla_decode sparse MLA library against the native path.

Every case runs the native ``mla_rope_generation`` once and then the same
``TrtllmAttention.forward`` twice: with the native FMHA library and with sparse_mla_decode
first in the layer's library list. Both outputs are compared with an FP64
oracle over the same e4m3 query and pool rows that applies the native masking
rules (index-list positions past a row's causal length are ignored, negative
entries skipped, repeats counted).

Error budget, fixed before any tuning:
  * nrms(sparse_mla_decode - oracle) <= 1.25 * nrms(native - oracle) + 2e-4
  * max|sparse_mla_decode - oracle| <= 2 * max|native - oracle| + 1e-3
  * the existing FP8-KV sparse MLA tolerance: allclose(sparse_mla_decode, oracle, atol=0.12,
    rtol=0.01)
  * rows without a usable entry and non-finite values must match the native output exactly.
"""

import pytest
import torch
from sparse_mla_harness import (
    DsaGenHarness,
    compare,
    error_stats,
    oracle_attention,
    record,
    run_both,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E501
    SparseMlaDecodeFmha,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="needs an SM10x GPU and the sparse_mla_decode kernel",
)


@pytest.mark.parametrize(
    "ctx_lens,seq_len_q,q_std",
    [
        ([32768], 6, 1.0),  # target verify, MTP K=5
        ([32768], 1, 1.0),  # draft steps 1-4
        ([32768], 6, 4.0),  # peaky softmax
        ([32768], 6, 0.1),  # nearly flat softmax
        ([100], 6, 1.0),  # fewer visible tokens than top-k
        ([2042], 6, 1.0),  # last row sees exactly 2048 tokens
        ([2043], 6, 1.0),  # rows straddle 2048 visible tokens
        ([5000, 300], 6, 1.0),  # two sequences, one short
        ([20000, 20000, 20000, 20000], 2, 1.0),  # four sequences, MTP K=1
        ([131072], 6, 1.0),
        ([391000], 6, 1.0),  # served-length context
        ([491061], 6, 1.0),  # controlled-decode context
    ],
)
def test_sparse_mla_matches_native(ctx_lens, seq_len_q, q_std):
    harness = DsaGenHarness(ctx_lens, seq_len_q=seq_len_q, seed=21)
    try:
        topk = harness.random_topk(seed=22, overlap=0.7, recent=32)
        step = harness.make_step(0, seed=23, q_std=q_std)
        compare(
            harness,
            step,
            topk,
            "matches_native",
            ctx_lens=ctx_lens,
            seq_len_q=seq_len_q,
            q_std=q_std,
        )
    finally:
        harness.shutdown()


def test_sparse_mla_empty_rows_tails_and_duplicates():
    """Index lists as the native indexer shapes them, plus repeated and empty rows."""
    harness = DsaGenHarness([8192, 40], seq_len_q=6, seed=31)
    try:
        topk = harness.random_topk(seed=32)
        topk[1, :] = -1  # no usable entry
        topk[3, :1500] = 17  # one row repeated 1500 times ...
        topk[3, 1500:] = 4242  # ... and another one 548 times
        for row in range(6, 12):  # short sequence: valid ids past the causal length are ignored
            for pos in (300, 1500, 2047):
                topk[row, pos] = 3
        step = harness.make_step(0, seed=33)
        compare(harness, step, topk, "empty_rows_tails_and_duplicates")
    finally:
        harness.shutdown()


def test_sparse_mla_causal_clamp_is_per_row():
    """Valid ids at list positions between a row's causal length and the sequence length."""
    harness = DsaGenHarness([100], seq_len_q=6, seed=41)
    try:
        topk = harness.random_topk(seed=42)
        for row in range(harness.num_rows):
            for pos in range(harness.causal_len(row), 106):
                topk[row, pos] = 5
        step = harness.make_step(0, seed=43)
        result = compare(harness, step, topk, "causal_clamp_per_row")
        global_idx = harness.global_indices(0, topk)
        per_sequence = oracle_attention(
            step.quant_q,
            harness.pool_rows(),
            global_idx,
            float(step.bmm1_scale[0]),
            float(step.bmm2_scale[0]),
            list_limit=[106] * harness.num_rows,
        )
        native_rows = result["native"].double().reshape(result["oracle"].shape)
        # Rows 0-4 differ between the two rules; the native kernel must follow the per-row one.
        per_row_err = error_stats(native_rows[:5], result["oracle"][:5])
        per_seq_err = error_stats(native_rows[:5], per_sequence[:5])
        record(
            "sparse_mla_correctness.jsonl",
            "causal_clamp_native_rule",
            {"per_row": per_row_err, "per_sequence": per_seq_err},
        )
        assert per_row_err["nrms"] < per_seq_err["nrms"], (per_row_err, per_seq_err)
    finally:
        harness.shutdown()


def test_sparse_mla_skips_negative_entries_inside_range():
    """Negative ids before the causal length are skipped (the native kernel does not check them)."""
    harness = DsaGenHarness([8192], seq_len_q=6, seed=51)
    try:
        topk = harness.random_topk(seed=52)
        topk[2, ::3] = -1
        topk[4, 1000:] = -1
        topk[5, :1024] = -1
        step = harness.make_step(0, seed=53)
        native, decode_out, global_idx, oracle = run_both(harness, step, topk)
        per_row = {
            r: {
                "sparse_mla_decode": error_stats(decode_out[r].double(), oracle[r]),
                "native": error_stats(native[r].double(), oracle[r]),
            }
            for r in range(harness.num_rows)
        }
        record("sparse_mla_correctness.jsonl", "negative_entries_inside_range", {"rows": per_row})
        clean = max(per_row[r]["sparse_mla_decode"]["nrms"] for r in (0, 1, 3))
        for r in (2, 4, 5):
            assert per_row[r]["sparse_mla_decode"]["nrms"] <= 2.0 * clean, (r, per_row[r], clean)
    finally:
        harness.shutdown()


def test_sparse_mla_cuda_graph_replay():
    """Capture both libraries once; replay with new indices, queries and rewritten pool rows."""
    harness = DsaGenHarness([65536], seq_len_q=6, seed=61)
    try:
        step = harness.make_step(0, seed=62)
        topk = harness.random_topk(seed=63, overlap=0.7, recent=32)
        graphs = {}
        outputs = {}
        for use_sparse_mla in (False, True):
            harness.select_library(0, use_sparse_mla=use_sparse_mla)
            harness.forward(0, step, topk)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[use_sparse_mla] = harness.forward(0, step, topk)
            graphs[use_sparse_mla] = graph
        pool_rows = harness.pool_rows()
        for it in range(6):
            topk.copy_(harness.random_topk(seed=100 + it, overlap=0.5, recent=16))
            if it % 2 == 1:
                # Page reuse: overwrite the rows the next replay reads.
                global_idx = harness.global_indices(0, topk)
                rows = global_idx[global_idx >= 0].unique().long()
                fresh = (
                    (torch.randn(rows.numel(), pool_rows.shape[1], device="cuda") * (1 + it))
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                pool_rows.view(torch.uint8)[rows] = fresh.view(torch.uint8)
            fresh_step = harness.make_step(0, seed=200 + it)
            for name in ("fused_q", "q_pe", "latent", "quant_q", "bmm1_scale", "bmm2_scale"):
                getattr(step, name).copy_(getattr(fresh_step, name))
            for use_sparse_mla in (False, True):
                graphs[use_sparse_mla].replay()
            torch.cuda.synchronize()
            global_idx = harness.global_indices(0, topk)
            oracle = oracle_attention(
                step.quant_q,
                pool_rows,
                global_idx,
                float(step.bmm1_scale[0]),
                float(step.bmm2_scale[0]),
                list_limit=harness.list_limits(),
            )
            native_err = error_stats(outputs[False].double(), oracle)
            sparse_mla_err = error_stats(outputs[True].double(), oracle)
            record(
                "sparse_mla_correctness.jsonl",
                "graph_replay",
                {
                    "iteration": it,
                    "native_vs_oracle": native_err,
                    "sparse_mla_vs_oracle": sparse_mla_err,
                },
            )
            assert sparse_mla_err["nonfinite"] == 0
            assert sparse_mla_err["nrms"] <= 1.25 * native_err["nrms"] + 2e-4, (
                it,
                sparse_mla_err,
                native_err,
            )
    finally:
        harness.shutdown()


def test_sparse_mla_falls_back_for_other_top_k():
    """A request outside the domain (top-k 1024) must run the native library."""
    harness = DsaGenHarness([16384], seq_len_q=6, index_topk=1024, seed=71)
    try:
        gen = torch.Generator(device="cpu").manual_seed(72)
        topk = (
            torch.stack(
                [
                    torch.randperm(harness.causal_len(r), generator=gen)[:1024]
                    for r in range(harness.num_rows)
                ]
            )
            .to(torch.int32)
            .cuda()
        )
        step = harness.make_step(0, seed=73)
        harness.select_library(0, use_sparse_mla=False)
        native = harness.forward(0, step, topk).clone()
        harness.select_library(0, use_sparse_mla=True)
        fallback = harness.forward(0, step, topk).clone()
        assert "SparseMlaDecodeFmha" not in harness.selected_library(0)
        torch.testing.assert_close(fallback, native, rtol=0, atol=0)
    finally:
        harness.shutdown()


def test_sparse_mla_static_eligibility():
    fp8 = DsaGenHarness([256], seq_len_q=1, seed=81)
    bf16 = DsaGenHarness([256], seq_len_q=1, kv_dtype=torch.bfloat16, seed=82)
    try:
        assert SparseMlaDecodeFmha.is_available(fp8.layers[0])
        assert not SparseMlaDecodeFmha.is_available(bf16.layers[0])
        assert sparse_mla.static_ineligibility(bf16.layers[0]) == "KV cache is not FP8"
    finally:
        fp8.shutdown()
        bf16.shutdown()


def test_sparse_mla_rejects_context_phase():
    harness = DsaGenHarness([256], seq_len_q=1, seed=91)
    try:
        q = torch.empty(1, 8 * 576, dtype=torch.bfloat16, device="cuda")
        args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
        assert (
            sparse_mla.request_ineligibility(q, None, None, harness.metadata, args)
            == "not generation-only"
        )
    finally:
        harness.shutdown()
