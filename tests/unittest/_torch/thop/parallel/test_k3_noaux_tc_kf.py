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
"""``trtllm::kimi_k3_noaux_tc_kf_op`` must equal ``trtllm::noaux_tc_op(.., 1, 1, 16, 1.0)`` bit for bit.

The generated Kimi-K3 routing kernel selects the top 16 of 896 experts by
``sigmoid(logit) + bias`` and renormalizes the selected scores. Ids, their order and the FP32
weights must be identical to ``noaux_tc_op``'s on: router-like logits, exact key ties (broken by
expert id) among few candidates (the kernel's fast path) and among many (its general path),
saturated sigmoids, selected-score sums just above 2^-13 and below it (the FP64 weight path), both
sides of the 2048-token schedule switch, non-contiguous and misaligned inputs, under CUDA graph
replay and right after the router GEMM with PDL. Rows with NaN logits must still get valid expert
ids. ``_KimiK3RoutingMethod`` must route through it only when enabled and on K3's routing shape.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103/SM107 GPU",
)

NUM_EXPERTS = 896
TOP_K = 16
MS = [1, 2, 7, 16, 18, 31, 64, 100, 128, 213, 256, 257, 777, 2048, 2049, 3001, 8192, 13356, 32768]
MODES = ["realistic", "rand", "ties", "fast_ties", "saturated", "near_threshold", "negative"]
# Expert pairs given identical logit columns and bias (and lifted into every top 16) by "fast_ties".
TIED_PAIRS = [(10, 500), (200, 701), (333, 334)]


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)


def _inputs(m, seed, mode):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if mode in ("realistic", "fast_ties"):
        # Router-like logits (as in test_k3_route_tail.py): unit hidden states against unit expert
        # directions plus per-expert popularity; tokens of one request share most of their hidden
        # state; bias spread ~0.19 as in Kimi-K3.
        dim = 512
        directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
        directions = directions / directions.norm(dim=1, keepdim=True)
        popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
        requests = torch.randn(((m + 6) // 7, dim), device="cuda", generator=gen)
        hidden = requests.repeat_interleave(7, dim=0)[:m] * 0.85
        hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
        hidden = hidden / hidden.norm(dim=1, keepdim=True)
        logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
        bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
        if mode == "fast_ties":
            # Three pairs of exactly equal keys inside every top 16, among few candidates.
            for e1, e2 in TIED_PAIRS:
                logits[:, e1] += 6.0
                logits[:, e2] = logits[:, e1]
                bias[e2] = bias[e1]
    elif mode == "rand":
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) * 1.5
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.1
    elif mode == "ties":
        # Seven distinct logits and no bias: exact key ties everywhere, broken by expert id.
        logits = torch.randint(-3, 4, (m, NUM_EXPERTS), device="cuda", generator=gen).float() * 0.5
        bias = torch.zeros((NUM_EXPERTS,), device="cuda")
    elif mode == "saturated":
        # About 40% of the sigmoids round to 1.0: dozens of candidates within the screening margin.
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) * 60.0
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 1e-3
    elif mode == "near_threshold":
        # The selected scores sum to just above 2^-13 (the smallest sums of the float weight path).
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) * 0.05 - 11.5
        bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
    else:  # negative: the selected scores sum to less than 2^-13
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) - 14.0
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.1
    return logits.contiguous(), bias.contiguous()


def _assert_bit_equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    a = actual.view(torch.int32) if actual.dtype == torch.float32 else actual
    e = expected.view(torch.int32) if expected.dtype == torch.float32 else expected
    mismatch = (a != e).nonzero()
    assert mismatch.numel() == 0, f"{what}: {mismatch.shape[0]} mismatches, first {mismatch[:4]}"


def _check(logits, bias, what):
    # noaux_tc_op reads its inputs as if contiguous (it does not check strides): give it contiguous copies.
    ref_w, ref_i = torch.ops.trtllm.noaux_tc_op(
        logits.contiguous(), bias.contiguous(), 1, 1, TOP_K, 1.0
    )
    w, i = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(logits, bias)
    _assert_bit_equal(i, ref_i, f"{what} ids")
    _assert_bit_equal(w, ref_w, f"{what} weights")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("m", MS)
def test_bit_identical_to_noaux_tc(m, mode):
    for seed in (0, 1):
        logits, bias = _inputs(m, seed, mode)
        _check(logits, bias, f"m={m} mode={mode} seed={seed}")


def _selected_score_sum(mode):
    logits, bias = _inputs(4096, 0, mode)
    _, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    return torch.sigmoid(logits).gather(1, ids.long()).sum(dim=1)


def test_modes_reach_their_paths():
    assert bool((_selected_score_sum("negative") < 2.0**-13).all())
    near = _selected_score_sum("near_threshold")
    assert bool((near >= 2.0**-13).all()) and bool((near < 2.0**-12).all())
    logits, bias = _inputs(64, 0, "fast_ties")
    _, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    for e1, e2 in TIED_PAIRS:
        assert bool((ids == e1).any(dim=1).all()) and bool((ids == e2).any(dim=1).all())


def test_nan_rows_get_valid_ids():
    m = 300
    logits, bias = _inputs(m, 4, "realistic")
    logits[0] = float("nan")
    logits[1, ::7] = float("nan")
    logits[2, 5] = float("nan")
    w, i = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(logits, bias)
    assert bool(((i >= 0) & (i < NUM_EXPERTS)).all())
    ref_w, ref_i = torch.ops.trtllm.noaux_tc_op(logits[3:].contiguous(), bias, 1, 1, TOP_K, 1.0)
    _assert_bit_equal(i[3:], ref_i, "finite rows next to NaN rows: ids")
    _assert_bit_equal(w[3:], ref_w, "finite rows next to NaN rows: weights")


@pytest.mark.parametrize("m", [8, 64, 256, 4096])
def test_after_router_gemm_with_pdl(m):
    # The op follows the router GEMM directly (PDL on by default on SM90+): it must wait for the logits.
    gen = torch.Generator(device="cuda").manual_seed(11)
    weight = (torch.randn((NUM_EXPERTS, 7168), device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    _, bias = _inputs(1, 12, "realistic")
    for seed in range(3):
        hidden = torch.randn((m, 7168), device="cuda", generator=gen).to(torch.bfloat16)
        logits = torch.ops.trtllm.dsv3_router_gemm_op(hidden, weight.t(), None, torch.float32)
        w, i = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(logits, bias)
        torch.cuda.synchronize()
        ref_w, ref_i = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
        _assert_bit_equal(i, ref_i, f"after router GEMM m={m} seed={seed} ids")
        _assert_bit_equal(w, ref_w, f"after router GEMM m={m} seed={seed} weights")


def test_strided_and_misaligned_inputs():
    m = 1000
    logits, bias = _inputs(m, 3, "realistic")
    wide = torch.zeros((m, NUM_EXPERTS + 3), device="cuda")
    wide[:, 3:] = logits
    _check(wide[:, 3:], bias, "non-contiguous rows")
    flat = torch.zeros((m * NUM_EXPERTS + 1,), device="cuda")
    flat[1:] = logits.reshape(-1)
    misaligned = flat[1:].view(m, NUM_EXPERTS)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    _check(misaligned, bias, "misaligned rows")
    big_bias = torch.zeros((2 * NUM_EXPERTS,), device="cuda")
    big_bias[::2] = bias
    _check(logits, big_bias[::2], "strided bias")


def test_empty_batch():
    logits = torch.empty((0, NUM_EXPERTS), device="cuda")
    bias = torch.zeros((NUM_EXPERTS,), device="cuda")
    w, i = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(logits, bias)
    assert w.shape == (0, TOP_K) and i.shape == (0, TOP_K)
    assert w.dtype == torch.float32 and i.dtype == torch.int32


def test_rejects_other_shapes_and_dtypes():
    bias = torch.zeros((NUM_EXPERTS,), device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(torch.zeros((4, 256), device="cuda"), bias)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(
            torch.zeros((4, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16), bias
        )


@pytest.mark.parametrize("m", [18, 256, 13356])
def test_cuda_graph_replay(m):
    logits, bias = _inputs(m, 5, "realistic")
    static_logits = logits.clone()
    torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(static_logits, bias)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, i = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op(static_logits, bias)
    for seed in (6, 7):
        new_logits, _ = _inputs(m, seed, "realistic")
        static_logits.copy_(new_logits)
        graph.replay()
        ref_w, ref_i = torch.ops.trtllm.noaux_tc_op(static_logits, bias, 1, 1, TOP_K, 1.0)
        _assert_bit_equal(i, ref_i, f"graph m={m} seed={seed} ids")
        _assert_bit_equal(w, ref_w, f"graph m={m} seed={seed} weights")


def _routing(modeling, monkeypatch, enabled, scale=1.0, bias=None):
    monkeypatch.setattr(modeling, "_K3_NOAUX_TC_KF", enabled)
    return modeling._KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )


def test_routing_method_wiring(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear as modeling

    logits, bias = _inputs(3001, 9, "realistic")
    calls = []
    kf_op = torch.ops.trtllm.kimi_k3_noaux_tc_kf_op

    def counting_op(*args):
        calls.append(args[0].shape[0])
        return kf_op(*args)

    monkeypatch.setattr(torch.ops.trtllm, "kimi_k3_noaux_tc_kf_op", counting_op, raising=False)

    off = _routing(modeling, monkeypatch, False, bias=bias)
    ids_off, w_off = off.apply(logits)
    assert calls == []

    on = _routing(modeling, monkeypatch, True, bias=bias)
    ids_on, w_on = on.apply(logits)
    assert calls == [3001]
    _assert_bit_equal(ids_on, ids_off, "routing method ids")
    _assert_bit_equal(w_on, w_off, "routing method weights")

    # Not K3's routing shape: DeepSeek-V3's path.
    scaled = _routing(modeling, monkeypatch, True, scale=2.5, bias=bias)
    scaled.apply(logits)
    on.apply(logits.to(torch.bfloat16))
    assert calls == [3001]

    monkeypatch.setattr(modeling, "_K3_NOAUX_TC_KF_MIN_TOKENS", 4096)
    on.apply(logits)
    assert calls == [3001]
