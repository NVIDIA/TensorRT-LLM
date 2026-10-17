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
"""The cross_layer_expert_prefetch expert prefetch in the shared_expert_prefetch fork.

``cross_layer_expert_prefetch/test_expert_prefetch_composition_hop_routing.py`` checks that every
routed-MoE runner class of the tree, shared_expert_prefetch's included, compiles the prefetch from a
key with a cross_layer_expert_prefetch spec. This file checks what is specific to
shared_expert_prefetch's port:

* the merged pre-wait prologue of ``SharedPrefetchExpertClusterKernel``: the shared-expert chunks
  are spread over cross_layer_expert_prefetch's issuing CTAs (``spec_ctas``) when
  cross_layer_expert_prefetch restricts them, over the whole grid otherwise
  (shared_expert_prefetch's mapping without cross_layer_expert_prefetch), and the chunks cover the
  shared expert's bytes;
* ``SharedPrefetchRoutedMoe.__call__`` passes cross_layer_expert_prefetch's row of predicted experts
  as the 18th routed_moe address (routed_moe's dummy without a hook) before the two shared-expert
  addresses, i.e. ``_NUM_XC_ADDRS + 2`` addresses in the kernel's order.
  Both need no GPU (kernel compilation and launch are stubbed);
* on an SM100-family GPU with real GLM-5.2 weights (``GLM_MOE_DSA_REAL_MOE_INPUTS_DIR``,
  ``GLM_MOE_DSA_ROUTED_MOE_CKPT_DIR``): shared_expert_prefetch's FP32 accumulator with the prefetch (K = 8 / 12, all
  or 32 issuing CTAs, valid and out-of-range IDs) is the exact sum of routed_moe's weighted rows within the
  FP32 reorder bound (routed_moe_into_hop's fp32 class, as shared_expert_prefetch without
  cross_layer_expert_prefetch); the self-store writes the call's distinct experts ranked by row
  count; a CUDA graph with the prefetch replays within the bound and keeps storing.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _sub in ("routed_moe", "hop_routing", "shared_expert_prefetch"):
    sys.path.insert(0, os.path.join(_ROOT, _sub))

KMAX = 32
REORDER_REL = 2.0**-20
LAYER = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10"))

sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


# ---------------------------------------------------------------------------
# The merged prologue's issuers (no compilation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_ctas,spec_k,spec_ctas,want",
    [
        (96, 0, 0, 96),  # shared_expert_prefetch without cross_layer_expert_prefetch: every CTA
        (96, 0, 32, 96),  # spec_ctas, prefetch not compiled in: shared_expert_prefetch's mapping
        (96, 8, 0, 96),  # cross_layer_expert_prefetch on every CTA
        (96, 8, 32, 96 // 3),  # the prefetch on the early-resident CTAs: the same CTAs
        (96, 12, 40, 40),
        (96, 8, 200, 96),
        (144, 8, 56, 56),
    ],
)
def test_prologue_issuers_and_coverage(num_ctas, spec_k, spec_ctas, want):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch import (  # noqa: E501
        kernel_xc as zk,
    )

    kern = zk.SharedPrefetchExpertClusterKernel(
        num_ctas, spec_k=spec_k, spec_ctas=spec_ctas, spec_self=True, red_out=True
    )
    assert kern.se_ctas == want
    assert (kern.spec_k, kern.spec_ctas) == (spec_k, spec_ctas)
    for chunk, total in ((kern.se_chunk1, zk.SE_W1_BYTES), (kern.se_chunk2, zk.SE_W2_BYTES)):
        assert chunk % 16 == 0
        lanes = kern.se_ctas * zk.PF_LANES
        assert lanes * chunk >= total, (chunk, total)
        # every issuing CTA owns a chunk (no CTA past se_ctas is needed)
        assert (lanes - zk.PF_LANES) * chunk < total
    if spec_k == 0:
        assert (kern.se_chunk1, kern.se_chunk2) == (
            zk._chunk(zk.SE_W1_BYTES, num_ctas),
            zk._chunk(zk.SE_W2_BYTES, num_ctas),
        )


# ---------------------------------------------------------------------------
# SharedPrefetchRoutedMoe.__call__ passes cross_layer_expert_prefetch's row (no launch)
# ---------------------------------------------------------------------------
class _Stream:
    cuda_stream = 0


class _Owner:
    """The NVFP4 expert leaf: only what the call path reads."""

    def __init__(self):
        for name in (
            "w3_w1_weight",
            "w3_w1_weight_scale",
            "w2_weight",
            "w2_weight_scale",
            "fc31_alpha",
            "fc31_scale_c",
            "fc2_alpha",
            "fc31_input_scale",
        ):
            setattr(self, name, torch.zeros(64, dtype=torch.uint8))


class _Ws:
    def __init__(self):
        for name in ("rows", "scales", "dummy", "tile_count"):
            setattr(self, name, torch.zeros(16, dtype=torch.uint8))
        self.num_ctas = 152

    def order_on(self, stream):
        pass


@pytest.mark.parametrize("hooked", [True, False])
def test_call_passes_the_expert_prefetch_row(monkeypatch, hooked):
    import cutlass

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as routed_moe_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch.routed_moe import (  # noqa: E501
        SharedPrefetchRoutedMoe,
    )

    calls = []
    ws = _Ws()
    monkeypatch.setattr(routed_moe_mod, "_bulk_copy_ready", lambda x: x)
    monkeypatch.setattr(routed_moe_mod, "_workspace", lambda device, n: ws)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Stream())
    # 24 co-resident four-CTA clusters: a 96-CTA expert-cluster grid, as on the served GPU.
    monkeypatch.setattr(routed_moe_mod, "_max_active_clusters", lambda device, cluster: 24)
    owner = _Owner()
    ids = torch.zeros(KMAX, dtype=torch.int32)
    if hooked:
        owner._expert_prefetch_spec = lambda: (ids, 8, 32, True)
    runner = SharedPrefetchRoutedMoe.__new__(SharedPrefetchRoutedMoe)
    runner.module = owner
    runner.bias = torch.zeros(256, dtype=torch.bfloat16)
    runner.device = torch.device("cpu")
    runner.num_ctas = 152
    monkeypatch.setattr(runner, "_compiled", lambda *a, **k: (lambda *args: calls.append(args)))
    x = torch.zeros(6, 16, dtype=torch.bfloat16)
    routing = torch.zeros(2, 6, 8, dtype=torch.int32)
    acc = torch.zeros(6, 16, dtype=torch.float32)
    se = (torch.zeros(8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16))
    assert runner(x, routing, acc, se) == [acc]
    assert len(calls) == 1
    ints = [a for a in calls[0] if isinstance(a, cutlass.Int64)]
    addrs = [int(a.value) if hasattr(a, "value") else int(a) for a in ints]
    assert len(addrs) == routed_moe_mod._NUM_XC_ADDRS + 2 == 20
    assert (
        addrs[0] == x.data_ptr() and addrs[1] == routing.data_ptr() and addrs[10] == acc.data_ptr()
    )
    assert addrs[17] == (ids.data_ptr() if hooked else ws.dummy.data_ptr())
    assert addrs[18:] == [se[0].data_ptr(), se[1].data_ptr()]


# ---------------------------------------------------------------------------
# Real weights: shared_expert_prefetch's fork with and without the prefetch
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def layer():
    import routed_moe_fixtures as fx

    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=LAYER, tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0"))
    )


@pytest.fixture(scope="module")
def runners(layer):
    import shared_prefetch_fixtures as zf

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch.routed_moe import (  # noqa: E501
        SharedPrefetchRoutedMoe,
        shared_weights,
    )

    mlp = zf.build_shared_mlp(LAYER, int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0")))
    se, reason = shared_weights(mlp)
    assert se is not None, reason
    shared_expert_prefetch = SharedPrefetchRoutedMoe(layer.module, layer.bias)
    if shared_expert_prefetch.schedule(8, False, True) != "xc":
        pytest.skip("the prefetch is part of the expert-cluster schedule")
    return RoutedMoe.from_native(layer.module, layer.bias), shared_expert_prefetch, se, mlp


class _Spec:
    """Install a cross_layer_expert_prefetch hook on the fork's expert leaf."""

    def __init__(self, runner, ids, k, ctas=0, self_store=False):
        self.owner = runner.module
        self.spec = (ids, k, ctas, self_store)

    def __enter__(self):
        self.owner._expert_prefetch_spec = lambda: self.spec
        return self

    def __exit__(self, *exc):
        del self.owner._expert_prefetch_spec


def _pack(logits, bias):
    import hop_routing_oracle as orc

    ids, w = orc.native_routing(logits, bias)
    routing = torch.empty(2, ids.shape[0], 8, dtype=torch.int32, device="cuda")
    routing[0].copy_(ids)
    routing[1].copy_(w.float().view(torch.int32))
    return routing


def _windows(layer, rows, count, offset=0):
    import routed_moe_fixtures as fx

    win = [
        (x, lg)
        for x, lg in fx.real_windows(fx.load_real_capture(layer.layer), rows)
        if x is not None
    ]
    if not win:
        pytest.skip(
            f"no captured hidden states of layer {layer.layer} (GLM_MOE_DSA_REAL_MOE_INPUTS_DIR)"
        )
    step = max(1, len(win) // count)
    out = []
    for i in range(count):
        x, lg = win[(offset + i * step) % len(win)]
        x = x.cuda().contiguous()
        lg = lg.float().cuda().contiguous()
        out.append((x, lg, _pack(lg, layer.bias)))
    return out


def _ids(kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "valid":
        ids = torch.randperm(256, generator=g)[:KMAX]
    else:  # the kernel masks each ID into [0, 256)
        ids = torch.randint(-(2**31), 2**31 - 1, (KMAX,), generator=g)
    return ids.to(torch.int32).cuda()


def _exact(routed_runner, x, lg):
    fc2, scale, idx = routed_runner(x, lg, do_finalize=False)
    torch.cuda.synchronize()
    terms = scale.double()[:, :, None] * fc2[idx.long()].double()
    return terms.sum(1), REORDER_REL * terms.abs().sum(1)


def _shared_prefetch_acc(shared_expert_prefetch, se, x, routing):
    acc = torch.zeros(x.shape[0], 6144, dtype=torch.float32, device="cuda")
    shared_expert_prefetch(x, routing, acc, se)
    torch.cuda.synchronize()
    return acc


def _ranked_experts(routing):
    ids = routing[0].flatten().long().cpu()
    counts = torch.bincount(ids, minlength=256)
    distinct = [e for e in range(256) if counts[e] > 0]
    return sorted(distinct, key=lambda e: (-int(counts[e]), e))


@sm100
@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("k,ctas", [(8, 32), (12, 0), (8, 0), (12, 40)])
@pytest.mark.parametrize("ids_kind", ["valid", "out_of_range"])
def test_shared_prefetch_accumulator_within_bound(layer, runners, rows, k, ctas, ids_kind):
    routed_runner, shared_expert_prefetch, se, _ = runners
    for i, (x, lg, routing) in enumerate(_windows(layer, rows, 2, offset=rows)):
        exact, bound = _exact(routed_runner, x, lg)
        ref = _shared_prefetch_acc(shared_expert_prefetch, se, x, routing)
        assert ((ref.double() - exact).abs() <= bound).all(), ("no prefetch", rows, i)
        ids = _ids(ids_kind, seed=rows * 100 + i)
        with _Spec(shared_expert_prefetch, ids, k, ctas):
            got = _shared_prefetch_acc(shared_expert_prefetch, se, x, routing)
        assert ((got.double() - exact).abs() <= bound).all(), (rows, k, ctas, ids_kind, i)


@sm100
@pytest.mark.parametrize("rows", [1, 3, 6, 8])
@pytest.mark.parametrize("k,ctas", [(8, 32), (12, 0)])
def test_shared_prefetch_self_store_ranks_by_row_count(layer, runners, rows, k, ctas):
    routed_runner, shared_expert_prefetch, se, _ = runners
    for i, (x, lg, routing) in enumerate(_windows(layer, rows, 3, offset=5 * rows)):
        exact, bound = _exact(routed_runner, x, lg)
        ids = torch.full((KMAX,), 999, dtype=torch.int32, device="cuda")
        with _Spec(shared_expert_prefetch, ids, k, ctas, self_store=True):
            acc = _shared_prefetch_acc(shared_expert_prefetch, se, x, routing)
            got = ids.tolist()
        assert ((acc.double() - exact).abs() <= bound).all(), (rows, i)
        want = _ranked_experts(routing)[:KMAX]
        assert got[: len(want)] == want, (rows, i, got, want)
        assert all(v == 999 for v in got[len(want) :]), (rows, i, got)


@sm100
def test_shared_prefetch_cuda_graph_replay(layer, runners):
    routed_runner, shared_expert_prefetch, se, _ = runners
    rows = 6
    wins = _windows(layer, rows, 6, offset=3)
    refs = [_exact(routed_runner, x, lg) for x, lg, _ in wins]
    x_static = wins[0][0].clone()
    r_static = wins[0][2].clone()
    acc = torch.zeros(rows, 6144, dtype=torch.float32, device="cuda")
    ids = torch.zeros(KMAX, dtype=torch.int32, device="cuda")
    with _Spec(shared_expert_prefetch, ids, 8, 32, self_store=True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            shared_expert_prefetch(x_static, r_static, acc, se)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # the eager call compiled the prefetch variant: the capture records it
        assert shared_expert_prefetch.ready_for_capture(False, rows=rows, red=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            shared_expert_prefetch(x_static, r_static, acc, se)
    for it, (x, _, routing) in enumerate(wins):
        x_static.copy_(x)
        r_static.copy_(routing)
        acc.zero_()
        graph.replay()
        torch.cuda.synchronize()
        exact, bound = refs[it]
        assert ((acc.double() - exact).abs() <= bound).all(), it
        want = _ranked_experts(routing)[:KMAX]
        assert ids.tolist()[: len(want)] == want, it
    del graph
