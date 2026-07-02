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
"""Unit tests for the Qwen-Image NVFP4 SVDQuant fused kernels (SM100 / Blackwell):
- nvfp4_svdquant_gemm   : residual NVFP4 GEMM + rank-r LoRA-up == fp4_gemm + D @ L1ᵀ
- nvfp4_quantize_smooth : NVFP4-quantize(x * pre_quant_scale) byte-identical to fp4_quantize(x*s)
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm torch ops)

_IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
skip_sm100 = pytest.mark.skipif(
    not _IS_SM100, reason="NVFP4 SVDQuant kernels require SM100 (Blackwell)"
)

_RANK = 32  # SVDQuant low-rank r (== the kernel's LoRA K)

_GEMM_CASES = [
    (m, n, k, use_bias, tactic)
    for m, n, k in [
        (256, 3072, 3072),
        (6912, 3072, 3072),
        (6912, 3072, 12288),
    ]
    for use_bias in [False, True]
    for tactic in [0, 1]
]
# Exercise every dynamic cluster shape on one image-sized problem. The broader
# shape/bias matrix above stays on the two base kernel shapes to keep SM100 CI bounded.
_GEMM_CASES.extend((6912, 3072, 3072, True, tactic) for tactic in range(2, 9))
# Cover both runtime clusters of each new narrow tile. M=44 exercises the
# token-tail regime where N=128 adds useful parallelism; M=6912 exercises the
# N=192 scale-factor layout and odd/even N-tile addressing at image size.
_GEMM_CASES.extend(
    [
        (44, 3072, 3072, True, 9),
        (44, 3072, 3072, True, 10),
        (6912, 3072, 3072, True, 11),
        (6912, 3072, 3072, True, 12),
    ]
)
# N=64 closes the small-M tactic gap with the stock CUTLASS NVFP4 runner.
# Exercise every retained 1SM runtime cluster without changing existing IDs.
_GEMM_CASES.extend((44, 3072, 3072, True, tactic) for tactic in range(13, 16))
# K=256 residual tiles use a BF16 K=64 storage/TMA box with logical rank 32;
# cover every runtime cluster, including 2SM multicast, on a non-multiple M tail.
_GEMM_CASES.extend((129, 3072, 3072, True, tactic) for tactic in range(16, 23))
# Appended runtime-cluster variants (23-26) mirroring the stock runner's per-shape
# winners: 256x256x256 at 4x2/2x4 (image-size M), 128x128x256 at 1x1 and
# 256x128x256 at 2x2 (token-tail M).
_GEMM_CASES.extend((6912, 3072, 3072, True, tactic) for tactic in [23, 24])
_GEMM_CASES.extend([(44, 3072, 3072, True, 25), (129, 3072, 12288, True, 26)])


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = (ref - got).float()
    noise = (err**2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10((ref.float() ** 2).mean() / noise))


def _make_svdquant_operator_chain():
    """Build the production single-linear operator chain and one representative input."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    m, n, k = 129, 3072, 3072
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    pqs = (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device=device)).abs().contiguous()
    smoothed = (x * pqs).to(torch.bfloat16)
    act_scale = (448.0 * 6.0 / smoothed.float().abs().amax()).reshape(1)

    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    weight_scale = (448.0 * 6.0 / weight.float().abs().amax()).reshape(1)
    weight_fp4, weight_sf = torch.ops.trtllm.fp4_quantize(weight, weight_scale, 16, False, True)
    alpha = (1.0 / (act_scale * weight_scale)).reshape(1).float()

    lora_a = torch.randn((_RANK, k), dtype=torch.bfloat16, device=device)
    l2t_smoothed = (pqs.unsqueeze(1) * lora_a.t()).contiguous()
    lora_b = torch.randn((n, _RANK), dtype=torch.bfloat16, device=device)
    lora_b_scaled = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn((n,), dtype=torch.bfloat16, device=device).contiguous()

    def prepare(activation: torch.Tensor):
        act_fp4, act_sf = torch.ops.trtllm.nvfp4_quantize_smooth(activation, pqs, act_scale)
        down = activation @ l2t_smoothed
        return act_fp4, act_sf, down

    def fused_chain(activation: torch.Tensor) -> torch.Tensor:
        act_fp4, act_sf, down = prepare(activation)
        return torch.ops.trtllm.nvfp4_svdquant_gemm_tuned(
            act_fp4.view(torch.uint8),
            weight_fp4.view(torch.uint8),
            act_sf.view(torch.uint8),
            weight_sf.view(torch.uint8),
            alpha,
            down,
            lora_b_scaled,
            torch.bfloat16,
            bias,
        )

    def reference_chain(activation: torch.Tensor) -> torch.Tensor:
        act_fp4, act_sf, down = prepare(activation)
        residual = torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            weight_fp4,
            act_sf,
            weight_sf,
            alpha,
            torch.bfloat16,
            allowed_backends="cutlass",
            bias=bias,
        )
        return residual + torch.mm(down, lora_b.t()).to(residual.dtype)

    return {"fused": fused_chain, "reference": reference_chain}, x


@skip_sm100
@pytest.mark.parametrize("m, n, k, use_bias, tactic", _GEMM_CASES)
def test_nvfp4_svdquant_gemm(m, n, k, use_bias, tactic):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    gx = ((448.0 * 6.0) / x.float().abs().max()).reshape(1).contiguous().cuda()
    gw = ((448.0 * 6.0) / w.float().abs().max()).reshape(1).contiguous().cuda()
    xq, x_sf = torch.ops.trtllm.fp4_quantize(x, gx, 16, False, True)
    wq, w_sf = torch.ops.trtllm.fp4_quantize(w, gw, 16, False, True)
    alpha = (1.0 / (gx * gw)).reshape(1).contiguous()
    D = torch.randn(m, _RANK, dtype=torch.bfloat16, device=dev) / (_RANK**0.25)
    L1 = torch.randn(n, _RANK, dtype=torch.bfloat16, device=dev) / (_RANK**0.25)
    # 1/alpha is folded into L1 so the epilogue out = alpha*acc yields the unscaled D @ L1ᵀ.
    L1_scaled = (L1.float() / alpha.reshape(-1)[:1]).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, dtype=torch.bfloat16, device=dev) if use_bias else None

    ref = torch.ops.trtllm.fp4_gemm(xq, wq, x_sf, w_sf, alpha, 0, torch.bfloat16).float()
    ref = ref + D.float() @ L1.float().t()
    if bias is not None:
        ref = ref + bias.float()
    out = torch.ops.trtllm.nvfp4_svdquant_gemm(
        xq.view(torch.uint8),
        wq.view(torch.uint8),
        x_sf.view(torch.uint8),
        w_sf.view(torch.uint8),
        alpha,
        D,
        L1_scaled,
        torch.bfloat16,
        bias,
        tactic,
    )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _sqnr_db(ref, out.float()) > 40.0

    if tactic == 0 and use_bias:
        from tensorrt_llm._torch.autotuner import autotune

        with autotune(tune_mode=True, skip_dynamic_tuning_buckets=True):
            tuned_out = torch.ops.trtllm.nvfp4_svdquant_gemm_tuned(
                xq.view(torch.uint8),
                wq.view(torch.uint8),
                x_sf.view(torch.uint8),
                w_sf.view(torch.uint8),
                alpha,
                D,
                L1_scaled,
                torch.bfloat16,
                bias,
            )
        assert _sqnr_db(ref, tuned_out.float()) > 40.0


@skip_sm100
@pytest.mark.parametrize("m, k", [(256, 3072), (6912, 12288)])
def test_nvfp4_quantize_smooth(m, k):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    pqs = (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device=dev)).abs()
    gs = ((448.0 * 6.0) / (x.float() * pqs.float()).abs().max()).reshape(1).contiguous()
    # Reference: quantize the pre-smoothed activation with the stock fp4_quantize.
    xq_ref, sf_ref = torch.ops.trtllm.fp4_quantize(
        (x * pqs).to(torch.bfloat16), gs, 16, False, True
    )
    xq, sf = torch.ops.trtllm.nvfp4_quantize_smooth(x, pqs, gs)
    assert torch.equal(xq.view(torch.uint8), xq_ref.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), sf_ref.view(torch.uint8))


@skip_sm100
def test_nvfp4_svdquant_fused_matches_reference():
    """The fused operator preserves the unfused single-linear computation."""
    from tensorrt_llm._torch.autotuner import autotune

    operator_chains, x = _make_svdquant_operator_chain()
    with torch.no_grad(), autotune(tune_mode=True, skip_dynamic_tuning_buckets=True):
        expected = operator_chains["reference"](x)
        actual = operator_chains["fused"](x)

    assert _sqnr_db(expected, actual) > 40.0


@skip_sm100
@pytest.mark.parametrize("implementation", ["fused", "reference"])
def test_nvfp4_svdquant_operator_cuda_graph(implementation):
    """The ordinary single-stream VisualGen runner captures and replays the full operator."""
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
        CUDAGraphRunner,
        CUDAGraphRunnerConfig,
    )

    operator_chains, x = _make_svdquant_operator_chain()
    operator_chain = operator_chains[implementation]
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    graph_chain = runner.wrap(operator_chain)

    with torch.no_grad(), autotune(tune_mode=True, skip_dynamic_tuning_buckets=True):
        eager = operator_chain(x).clone()
        captured = graph_chain(x).clone()
        assert len(runner.graphs) == 1

        replay_input = x + 0.125
        replay_eager = operator_chain(replay_input).clone()
        replay_actual = graph_chain(replay_input).clone()
        assert len(runner.graphs) == 1

    assert _sqnr_db(eager, captured) > 80.0
    assert _sqnr_db(replay_eager, replay_actual) > 80.0


@skip_sm100
@pytest.mark.parametrize("implementation", ["fused", "reference"])
def test_nvfp4_svdquant_operator_torch_compile(implementation):
    """The full operator compiles without graph breaks when CUDA graph mode is not used."""
    from tensorrt_llm._torch.autotuner import autotune

    operator_chains, x = _make_svdquant_operator_chain()
    operator_chain = operator_chains[implementation]
    compiled_chain = torch.compile(operator_chain, fullgraph=True)

    with torch.no_grad(), autotune(tune_mode=True, skip_dynamic_tuning_buckets=True):
        eager = operator_chain(x).clone()
        actual = compiled_chain(x).clone()

    assert _sqnr_db(eager, actual) > 80.0
