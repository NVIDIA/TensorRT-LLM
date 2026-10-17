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
"""sparse_mla_decode under DSA ragged verification (per-request verify widths).

With ragged verification enabled in the generation metadata, ``TrtllmAttention``
selects and runs the FMHA library inside the token-major presentation: one row
per query token, ``seq_len_q`` 1, and that row's causal KV length. sparse_mla_decode engages
there; these tests check it against the native library and the FP64 oracle with
the frozen error budget of ``test_sparse_mla_sparse_mla.py``, check the lengths sparse_mla_decode
receives, and check that a ragged step whose widths are uniform gives the same
sparse_mla_decode output as the uniform layout.
"""

import pytest
import torch
from sparse_mla_harness import DsaGenHarness, compare

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="needs an SM10x GPU and the sparse_mla_decode kernel",
)


@pytest.mark.parametrize(
    "ctx_lens,verify_lens",
    [
        ([8192, 300, 40000], [6, 2, 4]),  # mixed widths, one short request
        ([391000], [3]),  # one request with a partial verify window
        ([100, 2043], [5, 6]),  # rows straddling the top-k length
        ([20000, 20000, 20000], [1, 6, 1]),
    ],
)
def test_sparse_mla_ragged_verification_matches_native(ctx_lens, verify_lens):
    harness = DsaGenHarness(
        ctx_lens, seq_len_q=max(verify_lens), seed=121, ragged_verify_lens=verify_lens
    )
    try:
        assert harness.metadata.token_major_gen_view() is not None
        topk = harness.random_topk(seed=122, overlap=0.7, recent=32)
        step = harness.make_step(0, seed=123)
        with harness.capture_sparse_mla_inputs() as calls:
            compare(
                harness,
                step,
                topk,
                "ragged_verify_matches_native",
                ctx_lens=ctx_lens,
                verify_lens=verify_lens,
            )
        assert len(calls) == 1, len(calls)
        call = calls[0]
        # One presented row per query token, each with its causal length.
        assert call["ragged"] and call["seq_len_q"] == 1, call
        expected = torch.tensor(harness.list_limits(), dtype=torch.int32)
        assert torch.equal(call["seq_lens_kv"].cpu(), expected), (call["seq_lens_kv"], expected)
    finally:
        harness.shutdown()


def test_sparse_mla_ragged_presentation_equals_uniform():
    """Uniform widths through the ragged presentation give the uniform layout's sparse_mla_decode output."""
    ctx_lens = [32768, 5000]
    outputs = {}
    queries = {}
    for ragged in (False, True):
        harness = DsaGenHarness(
            ctx_lens, seq_len_q=6, seed=131, ragged_verify_lens=[6, 6] if ragged else None
        )
        try:
            assert (harness.metadata.token_major_gen_view() is not None) == ragged
            topk = harness.random_topk(seed=132, overlap=0.7, recent=32)
            step = harness.make_step(0, seed=133)
            harness.select_library(0, use_sparse_mla=True)
            outputs[ragged] = harness.forward(0, step, topk).clone()
            assert harness.selected_library(0) == ["SparseMlaDecodeFmha"]
            queries[ragged] = (step.quant_q.clone(), step.bmm1_scale.clone())
        finally:
            harness.shutdown()
    torch.testing.assert_close(queries[True][0], queries[False][0], rtol=0, atol=0)
    torch.testing.assert_close(queries[True][1], queries[False][1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[True], outputs[False], rtol=0, atol=0)
