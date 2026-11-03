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
"""The launch-release q_a RMSNorm (``kimi_k3_mla.q_norm_pdl``) equals FlashInfer's RMSNorm."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)


@pytest.mark.parametrize("hidden", [1536])
@pytest.mark.parametrize("m", [1, 7, 14, 32])
@pytest.mark.parametrize("strided", [False, True], ids=["contig", "strided"])
def test_matches_flashinfer_rmsnorm(m, hidden, strided):
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

    gen = torch.Generator(device="cuda").manual_seed(m * hidden)
    width = hidden + 576 if strided else hidden
    full = (torch.randn((m, width), device="cuda", generator=gen) * 3).to(torch.bfloat16)
    x = full[:, :hidden]
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen)).to(torch.bfloat16)
    assert q_norm_pdl.can_use(x, weight)
    out = q_norm_pdl.rmsnorm_release_at_launch(x, weight, 1e-6)
    ref = rmsnorm(x, weight, 1e-6, enable_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_graph_chain_with_pdl_gemms():
    """In a CUDA graph between two PDL GEMMs (q_a -> norm -> q_b), the result is unchanged."""
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic, run_splitk_dense
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch.kernels.blackwell.dspark_draft_gemm_push import ensure_cute_runtime
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl

    ensure_cute_runtime()
    gen = torch.Generator(device="cuda").manual_seed(5)
    m, k, n_a, n_b = 14, 7168, 2112, 18432
    # Split-K BF16 GEMM tactic (mma_m, mma_n, split_k, ab_stages) for 14 rows of both weights.
    tactic = SplitKTactic(64, 16, 4, 8)

    def weight(n, kk):
        return (torch.randn((n, kk), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)

    w_a = weight(n_a, k)
    w_b = weight(n_b, 1536)
    x = torch.randn((m, k), device="cuda", generator=gen).to(torch.bfloat16)
    norm_w = (1 + 0.1 * torch.randn(1536, device="cuda", generator=gen)).to(torch.bfloat16)
    qa = torch.empty((m, n_a), device="cuda", dtype=torch.bfloat16)
    qb = torch.empty((m, n_b), device="cuda", dtype=torch.bfloat16)

    def step(norm):
        run_splitk_dense(x, w_a.t(), None, qa, True, tactic)
        run_splitk_dense(norm(qa[:, :1536]), w_b.t(), None, qb, True, tactic)

    step(lambda t: rmsnorm(t, norm_w, 1e-6, enable_pdl=True))
    torch.cuda.synchronize()
    ref = qb.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(lambda t: q_norm_pdl.rmsnorm_release_at_launch(t, norm_w, 1e-6))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(lambda t: q_norm_pdl.rmsnorm_release_at_launch(t, norm_w, 1e-6))
    for _ in range(5):
        qb.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(qb, ref)
