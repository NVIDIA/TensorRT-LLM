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
"""Tests of package cross_layer_expert_prefetch (cross-layer speculative routed-expert L2 prefetch)."""

import pytest
import torch
from torch import nn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NE = 256
HIDDEN = 6144
KMAX = 32
STAT_KS = (8, 12, 18, 24, 32)


def _ref_union(logits, bias, rows):
    s = torch.sigmoid(logits[:rows].float()) + bias.float()[None, :]
    thr = s.sort(dim=1, descending=True).values[:, 7:8]
    sel = s >= thr
    return sel.sum(0).to(torch.int32), s.max(0).values


def _ref_rank(logits, bias, rows):
    cnt, mx = _ref_union(logits, bias, rows)
    order = sorted(range(NE), key=lambda e: (-int(cnt[e]), -float(mx[e]), e))
    return order[:KMAX]


@pytest.fixture()
def spec():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.cross_layer_expert_prefetch import (  # noqa: E501
        spec_experts,
    )

    return spec_experts


def test_the_served_config(spec):
    cfg = spec.config()
    assert cfg is spec.SERVED_CONFIG
    assert (
        cfg.k,
        cfg.start,
        cfg.pred,
        cfg.fused,
        cfg.delay_ns,
        cfg.chunk,
        cfg.ctas,
        cfg.issue,
        cfg.xc_ctas,
    ) == (8, "moe", "self", True, 0, 16384, 16, "xc", 32)
    assert cfg.start in spec.STARTS and cfg.pred in spec.PREDS and cfg.issue in spec.ISSUES
    assert spec.ExpertPrefetchConfig(8, "moe", "self", True, 0, 16384, 16) == cfg


@cuda
@pytest.mark.parametrize("rows,ties", [(1, False), (6, False), (8, False), (1, True), (6, True)])
def test_rank_and_stats_match_reference(rows, ties):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.cross_layer_expert_prefetch import (  # noqa: E501
        kernels,
    )

    g = torch.Generator(device="cuda").manual_seed(rows)
    dev = "cuda"
    nxt = torch.randn(8, NE, device=dev, generator=g) * 3
    nbias = (torch.randn(NE, device=dev, generator=g) * 0.5).bfloat16()
    cur = torch.randn(8, NE, device=dev, generator=g) * 3
    cbias = (torch.randn(NE, device=dev, generator=g) * 0.5).bfloat16()
    if ties:
        # 13 integer logits and no bias: many equal scores within and across
        # rows (resolved by ID), distinct values far apart
        nxt = nxt.round().clamp(-6, 6)
        cur = cur.round().clamp(-6, 6)
        nbias.zero_()
        cbias.zero_()
    cur_ids = torch.randperm(NE, device=dev, generator=g)[:KMAX].to(torch.int32)
    out_ids = torch.zeros(KMAX, dtype=torch.int32, device=dev)
    counters = torch.zeros(kernels.NUM_COUNTERS, dtype=torch.int64, device=dev)
    kernels.rank_and_stats(nxt, nbias, out_ids, cur, cbias, cur_ids, counters, rows, True, True)
    torch.cuda.synchronize()
    assert out_ids.tolist() == _ref_rank(nxt, nbias, rows)
    cnt, _ = _ref_union(cur, cbias, rows)
    actual = cnt > 0
    pos = {int(e): j for j, e in enumerate(cur_ids.tolist())}
    exp = [1, int(actual.sum())]
    for k in STAT_KS:
        exp.append(sum(1 for e in range(NE) if actual[e] and pos.get(e, KMAX) < k))
    assert counters.tolist() == exp


def _exact_weight(g):
    """Gate weights in multiples of 1/256: with hidden values in {-1, 0, 1} every
    logit is exact in FP32 whatever the summation order."""
    return (torch.randint(-2, 3, (NE, HIDDEN), device="cuda", generator=g) / 256).bfloat16()


def _exact_hidden(rows, g):
    return torch.randint(-1, 2, (rows, HIDDEN), device="cuda", generator=g).bfloat16()


@cuda
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_gemm_rank_matches_reference(rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.cross_layer_expert_prefetch import (  # noqa: E501
        kernels,
    )

    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    h = _exact_hidden(rows, g)
    w = _exact_weight(g)
    bias = (torch.randn(NE, device="cuda", generator=g) * 0.1).bfloat16()
    scores = torch.zeros(kernels.MAX_ROWS, NE, device="cuda")
    ticket = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(KMAX, dtype=torch.int32, device="cuda")
    ref = _ref_rank(h.float() @ w.float().t(), bias, rows)
    for _ in range(3):  # every launch resets the ticket for the next one
        out.zero_()
        kernels.gemm_rank(h, w, bias, scores, ticket, out, rows)
        torch.cuda.synchronize()
        assert out.tolist() == ref
        assert ticket.item() == 0
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            kernels.gemm_rank(h, w, bias, scores, ticket, out, rows)
    torch.cuda.current_stream().wait_stream(s)
    out.zero_()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == ref and ticket.item() == 0


class _Gate(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = nn.Parameter(_exact_weight(g), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(
            (torch.randn(NE, device="cuda", generator=g) * 0.1).bfloat16(), requires_grad=False
        )

    def forward(self, h):
        return h.float() @ self.weight.float().t()


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        u8 = dict(dtype=torch.uint8, device="cuda")
        self.w3_w1_weight = nn.Parameter(
            torch.zeros(NE, 512, HIDDEN // 2, **u8), requires_grad=False
        )
        self.w3_w1_weight_scale = nn.Parameter(
            torch.zeros(NE, 512, HIDDEN // 16, **u8), requires_grad=False
        )
        self.w2_weight = nn.Parameter(torch.zeros(NE, HIDDEN, 128, **u8), requires_grad=False)
        self.w2_weight_scale = nn.Parameter(torch.zeros(NE, HIDDEN, 16, **u8), requires_grad=False)


class _Mlp(nn.Module):
    def __init__(self, seed, experts):
        super().__init__()
        self.gate = _Gate(seed)
        self.experts = experts


class _Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


@cuda
def test_link_layers_binds_routed_kernel_hooks(spec, monkeypatch):
    """At the fixed depth, predictor, issuing kernel and CTA count, every MoE layer's routed
    kernel prefetches the experts its own prediction names."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe,
    )

    monkeypatch.setattr(routed_moe, "resolve_expert_owner", lambda m: (m, ""))
    monkeypatch.setattr(spec.dispatch, "is_enabled", lambda f: True)
    spec._STATES.clear()
    gates = [_Gate(i) for i in range(3)]
    leaves = [nn.Module() for _ in range(3)]
    layers = [nn.Identity()]
    for gate, leaf in zip(gates, leaves):
        mlp = nn.Module()
        mlp.gate, mlp.experts = gate, leaf
        layers.append(_Layer(mlp))
    assert spec.link_layers(layers, 4) == 3
    bound = [hasattr(leaf, "_expert_prefetch_spec") for leaf in leaves]
    # Every layer's prediction feeds its own routed kernel.
    assert bound == [True, True, True]
    assert leaves[2]._expert_prefetch_spec() is None  # before the device state exists
    rows = 6
    h = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(rows, NE, device="cuda", dtype=torch.float32)
    for i in (1, 2, 3):
        link = layers[i]._expert_prefetch_link
        link.on_hop(h, logits)
        link.after_moe()
        link.after_post_moe()
    spec.join()
    torch.cuda.synchronize()
    st = spec._STATES[torch.cuda.current_device()]
    ids, k, ctas, self_store = leaves[2]._expert_prefetch_spec()
    assert ids.data_ptr() == st.pred_ids[3].data_ptr() and (k, ctas) == (8, 32)
    assert self_store
    # the routed kernel issues the prefetch: no prefetch kernel is launched
    assert st.prefetch_calls == 0
    assert not st.forked and not st.counters.any()
    # without routed_moe nothing is bound
    spec._STATES.clear()
    for leaf in leaves:
        leaf.__dict__.pop("_expert_prefetch_spec", None)
    monkeypatch.setattr(spec.dispatch, "is_enabled", lambda f: f != "routed_moe")
    spec.link_layers(layers, 4)
    assert not any(hasattr(leaf, "_expert_prefetch_spec") for leaf in leaves)
    spec._STATES.clear()


@cuda
def test_declines_outside_decode_shape(spec):
    spec._STATES.clear()
    experts = _Experts()
    layers = [_Layer(_Mlp(1, experts)), _Layer(_Mlp(2, experts))]
    spec.link_layers(layers, 2)
    link = layers[0]._expert_prefetch_link
    h = torch.zeros(16, HIDDEN, device="cuda", dtype=torch.bfloat16)
    link.on_hop(h, None)
    link.after_moe()
    link.after_post_moe()
    spec.join()
    assert not spec._STATES


@cuda
def test_delay_kernel_waits():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.cross_layer_expert_prefetch import (  # noqa: E501
        kernels,
    )

    kernels.delay(1000)  # compile
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    kernels.delay(50_000)
    t1.record()
    torch.cuda.synchronize()
    assert t0.elapsed_time(t1) >= 0.045  # ms
