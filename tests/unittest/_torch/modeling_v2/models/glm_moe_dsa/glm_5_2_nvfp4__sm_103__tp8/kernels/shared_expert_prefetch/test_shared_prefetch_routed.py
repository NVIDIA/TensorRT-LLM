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
"""shared_expert_prefetch's routed fork (``SharedPrefetchExpertClusterKernel``: hop_routing's fork
plus the pre-wait L2 prefetch of the shared expert's weights, on the shared_expert_prefetch grid) on
one GPU, real GLM-5.2 NVFP4 weights.

Given the same routing, neither the grid nor the prefetch may change any expert row:

* raw FC2 rows and scales of hop_routing's fork at the shared_expert_prefetch grid (the fork's
  grid set to 4 x shared_expert_prefetch's clusters) equal those at F's grid (24 clusters,
  raised to ``ceil(R * 8 / 2)`` for R > 6) bit for bit, for every D bucket 8..48 at R = 6 and
  for R = 1..8 (real windows);
* shared_expert_prefetch's FP32 accumulator equals the exact sum of routed_moe's weighted rows
  within the FP32 reorder bound (routed_moe_into_hop's fp32 class), for the same cases and in every
  test below. The accumulator's atomic adds come in any order, so two runs
  need not be bit-identical; every check is against that bound;
* CUDA-graph replay with fresh routing and interleaved row counts (one graph
  per R, replayed in alternating order);
* a delayed producer: a kernel that releases its dependents first and writes
  the routing and the activation only after a delay, so a read before the
  dependency wait would see stale data;
* the native shared expert co-running on a second stream (as served) changes
  no routed result, and its own output stays bit-identical while the routed
  kernel prefetches its weights;
* host-only declines.

Routings: constructed expert sets with exactly D distinct experts (fresh
experts per case), routed through the native routing of logits built for
them, and real captured windows of layer ``GLM_MOE_DSA_ROUTED_MOE_LAYER``.
"""

import contextlib
import os
import sys
from unittest import mock

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
for sub in ("routed_moe", "hop_routing"):
    sys.path.insert(0, os.path.join(_ROOT, sub))
import hop_routing_oracle as orc  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

REORDER_REL = 2.0**-20
WIDE_CLUSTERS = 36
LAYER = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10"))
F_CLUSTERS = 24
D_CASES = (8, 12, 16, 20, 23, 24, 25, 26, 28, 30, 31, 32, 33, 35, 36, 37, 40, 44, 48)


def _grid(runner, ctas):
    """``runner``'s expert-cluster grid set to ``ctas`` CTAs inside the block."""
    return mock.patch.object(runner, "_xc_ctas", lambda *args, **kwargs: ctas)


def _clusters(n):
    """shared_expert_prefetch's cluster count set to ``n`` inside the block (None: unchanged)."""
    if n is None:
        return contextlib.nullcontext()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch import (  # noqa: E501
        routed_moe,
    )

    return mock.patch.object(routed_moe, "DEFAULT_CLUSTERS", n)


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=LAYER, tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0"))
    )


class _SHARED_PREFETCH:
    """shared_expert_prefetch's runner with the layer's real shared expert ``mlp`` (its weights are
    the prefetch target)."""

    def __init__(self, runner, mlp, se):
        self.runner, self.mlp, self.se = runner, mlp, se

    def __call__(self, x, routing, red_out):
        return self.runner(x, routing, red_out, self.se)

    def __getattr__(self, name):
        return getattr(self.runner, name)


@pytest.fixture(scope="module")
def runners(layer):
    import shared_prefetch_fixtures as zf

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.hop_routing.routed_moe import (  # noqa: E501
        HopRoutedMoe,
    )
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
    return (
        RoutedMoe.from_native(layer.module, layer.bias),
        HopRoutedMoe(layer.module, layer.bias),
        _SHARED_PREFETCH(SharedPrefetchRoutedMoe(layer.module, layer.bias), mlp, se),
    )


def _pack(logits, bias):
    ids, w = orc.native_routing(logits, bias)
    routing = torch.empty(2, ids.shape[0], 8, dtype=torch.int32, device="cuda")
    routing[0].copy_(ids)
    routing[1].copy_(w.float().view(torch.int32))
    return routing


def _constructed(layer, rows, d, seed):
    sets = fx.expert_sets_with_distinct(rows, d, seed=seed)
    perm = torch.randperm(
        fx.NUM_EXPERTS, generator=torch.Generator().manual_seed(1000 + seed)
    ).tolist()
    uniq = sorted(set(e for row in sets for e in row))
    remap = {e: perm[j] for j, e in enumerate(uniq)}
    sets = [[remap[e] for e in row] for row in sets]
    lg = fx.logits_for_expert_sets(sets, layer.bias.cpu(), seed=seed).float().cuda().contiguous()
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    routing = _pack(lg, layer.bias)
    assert len(set(routing[0].flatten().tolist())) == d
    return x, lg, routing


def _real(layer, rows, count, offset=0):
    win = fx.real_windows(fx.load_real_capture(layer.layer), rows, stride=rows * 7)
    out = []
    for x, lg in win[offset : offset + count]:
        x, lg = x.cuda().contiguous(), lg.float().cuda().contiguous()
        out.append((x, lg, _pack(lg, layer.bias)))
    return out


def _cases(layer):
    out = [("D%d" % d, 6) + _constructed(layer, 6, d, seed=d) for d in D_CASES]
    for rows in range(1, 9):
        out += [
            ("real R%d #%d" % (rows, i), rows) + c
            for i, c in enumerate(_real(layer, rows, 2, offset=rows))
        ]
    return out


def _shared_prefetch_ctas(runner, rows):
    return runner._xc_ctas(rows, False, True)


def _f_ctas(runner, rows):
    """F's FP32-accumulator grid for ``rows`` rows: 24 clusters, raised to cover rows x 8 experts in two rounds."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as routed_moe_mod,
    )

    co = routed_moe_mod._max_active_clusters(runner.device, 4)
    return 4 * min(co, max(F_CLUSTERS, -(-rows * fx.TOP_K // 2)))


def _acc_within_bound(routed_runner, x, lg, acc):
    """Whether ``acc`` is the exact FP32-class sum of routed_moe's weighted raw rows for ``(x,
    lg)``: the atomics add the top-k contributions in any order, so the check is the reorder bound
    against the exact sum, not run-to-run equality."""
    fc2, scale, idx = routed_runner(x, lg, do_finalize=False)
    terms = scale.double()[:, :, None] * fc2[idx.long()].double()
    exact = terms.sum(1)
    bound = REORDER_REL * terms.abs().sum(1)
    return bool(((acc.double() - exact).abs() <= bound).all())


def test_grid(layer, runners):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch.routed_moe import (  # noqa: E501
        DEFAULT_CLUSTERS,
    )

    _, _, shared_expert_prefetch = runners
    co = __import__(
        "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe",
        fromlist=["x"],
    )
    clusters = co._max_active_clusters(shared_expert_prefetch.device, 4)
    print(
        f"GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_GRID co-resident 4-CTA clusters {clusters}; "
        f"shared_expert_prefetch grid R6 {_shared_prefetch_ctas(shared_expert_prefetch, 6)} CTAs",
        flush=True,
    )
    assert _shared_prefetch_ctas(shared_expert_prefetch, 6) == 4 * min(clusters, DEFAULT_CLUSTERS)
    for want in (28, 32, 36):
        with _clusters(want):
            assert _shared_prefetch_ctas(shared_expert_prefetch, 6) == 4 * min(clusters, want)
            assert _shared_prefetch_ctas(shared_expert_prefetch, 8) == 4 * min(
                clusters, max(want, 32)
            )  # two rounds cover 8 x 8 experts
    assert _shared_prefetch_ctas(shared_expert_prefetch, 8) == 4 * min(
        clusters, max(DEFAULT_CLUSTERS, 32)
    )
    assert shared_expert_prefetch.schedule(6, False, True) == "xc"


def test_raw_rows_grid_independent_and_fp32_class(layer, runners):
    routed_runner, hop_routing, shared_expert_prefetch = runners
    bad = []
    for name, rows, x, lg, routing in _cases(layer):
        with _grid(hop_routing, _f_ctas(shared_expert_prefetch, rows)):
            raw_f = hop_routing(x, routing, do_finalize=False)
        with _clusters(WIDE_CLUSTERS):
            wide = _shared_prefetch_ctas(shared_expert_prefetch, rows)
        with _grid(hop_routing, wide):
            raw_7 = hop_routing(x, routing, do_finalize=False)
        fc2, scale, idx = routed_runner(x, lg, do_finalize=False)
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(raw_f, raw_7))
        same_routed_moe = all(torch.equal(a, b) for a, b in zip(raw_f, (fc2, scale, idx)))
        terms = scale.double()[:, :, None] * fc2[idx.long()].double()
        exact = terms.sum(1)
        bound = REORDER_REL * terms.abs().sum(1)
        ok_acc = True
        for clusters in (None, WIDE_CLUSTERS):
            with _clusters(clusters):
                for _ in range(2):
                    acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
                    shared_expert_prefetch(x, routing, red_out=acc)
                    torch.cuda.synchronize()
                    ok_acc = ok_acc and bool(((acc.double() - exact).abs() <= bound).all())
        print(
            f"GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_ROUTED {name}: raw F-grid == {wide}-CTA grid {same}, == "
            f"routed_moe {same_routed_moe}, "
            f"fp32 within bound (shared_expert_prefetch default / wide) {ok_acc}",
            flush=True,
        )
        if not (same and same_routed_moe and ok_acc):
            bad.append(name)
    assert not bad, bad


def test_graph_replay_interleaved_rows(layer, runners):
    routed_runner, _, shared_expert_prefetch = runners
    graphs, state = {}, {}
    for rows in (1, 3, 6, 8):
        cases = _real(layer, rows, 3, offset=1)
        x = cases[0][0].clone()
        routing = cases[0][2].clone()
        acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
        shared_expert_prefetch(x, routing, red_out=acc)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            shared_expert_prefetch(x, routing, red_out=acc)
        graphs[rows] = g
        state[rows] = (x, routing, acc, cases)
    for step in range(2):
        for rows in (1, 3, 6, 8) if step == 0 else (8, 6, 3, 1):
            x, routing, acc, cases = state[rows]
            xn, ln, rn = cases[1 + step]
            x.copy_(xn)
            routing.copy_(rn)
            acc.zero_()
            graphs[rows].replay()
            torch.cuda.synchronize()
            assert _acc_within_bound(routed_runner, xn, ln, acc), (rows, step)
    del graphs


def _delay_kernel():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32, Int64

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        ptx as wptx,
    )

    class DelayedCopy:
        """Releases its dependents at entry, spins ``ns``, then copies ``words`` 16-byte words."""

        @cute.jit
        def run(self, src: Int64, dst: Int64, words: Int32, ns: Int64, stream):
            self.kernel(src, dst, words, ns).launch(
                grid=[1, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, src: Int64, dst: Int64, words: Int32, ns: Int64):
            tidx, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_launch_dependents()
            t0 = wptx.globaltimer_u64()
            tn = t0
            while tn - t0 < ns:
                tn = wptx.globaltimer_u64()
            i = Int32(tidx)
            while i < words:
                v = wptx.ld_v4_u32(src + Int64(i) * 16)
                wptx.st_v4_u32(dst + Int64(i) * 16, v[0], v[1], v[2], v[3])
                i = i + 256

    from cuda.bindings import driver as cuda

    z = cutlass.Int64(0)
    fn = cute.compile(
        DelayedCopy().run,
        z,
        z,
        cutlass.Int32(1),
        z,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    def launch(src, dst, ns):
        assert src.nbytes % 16 == 0 and src.nbytes == dst.nbytes
        fn(
            cutlass.Int64(src.data_ptr()),
            cutlass.Int64(dst.data_ptr()),
            cutlass.Int32(src.nbytes // 16),
            cutlass.Int64(ns),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    return launch


def test_delayed_producer(layer, runners):
    """Routing and activation written 20 us after the producer releases the routed kernel."""
    routed_runner, _, shared_expert_prefetch = runners
    launch = _delay_kernel()
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    for name, rows, x, lg, routing in _cases(layer)[::3]:
        # packed source: [routing | x] so that one delayed copy writes both
        src = torch.cat([routing.flatten().view(torch.uint8), x.flatten().view(torch.uint8)])
        pad = (-src.numel()) % 16
        src = torch.cat([src, torch.zeros(pad, dtype=torch.uint8, device="cuda")])
        # stale bytes (huge activations, invalid expert ids), so that a read before the dependency wait changes
        # the result
        dst = torch.full_like(src, 0x7F)
        r_view = dst[: routing.nbytes].view(torch.int32).view(2, rows, 8)
        x_view = (
            dst[routing.nbytes : routing.nbytes + x.nbytes]
            .view(torch.bfloat16)
            .view(rows, fx.HIDDEN)
        )
        acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
        launch(src, dst, 20000)
        shared_expert_prefetch(x_view, r_view, red_out=acc)
        torch.cuda.synchronize()
        assert _acc_within_bound(routed_runner, x, lg, acc), name


def test_corun_shared_expert(layer, runners):
    """shared_expert_prefetch's routed fork and the native shared expert (whose weights it
    prefetches) on two streams, as served: same results as alone."""
    routed_runner, _, shared_expert_prefetch = runners
    mlp = shared_expert_prefetch.mlp
    side = torch.cuda.Stream()
    for name, rows, x, lg, routing in _cases(layer)[::2]:
        want_acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
        shared_expert_prefetch(x, routing, red_out=want_acc)
        want_se = mlp(x)
        torch.cuda.synchronize()
        acc = torch.zeros_like(want_acc)
        side.wait_stream(torch.cuda.current_stream())
        shared_expert_prefetch(x, routing, red_out=acc)
        with torch.cuda.stream(side):
            se = mlp(x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        assert _acc_within_bound(routed_runner, x, lg, acc), name
        assert torch.equal(se.view(torch.int16), want_se.view(torch.int16)), name


def test_declines_before_device_work(layer):
    import shared_prefetch_fixtures as zf

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.shared_expert_prefetch import (  # noqa: E501
        routed_moe as shared_prefetch_mod,
    )

    class _Hop:
        _routed_enabled = True

        def __init__(self, mode):
            self.routed_mode = mode

        def can_run(self, rows):
            return rows <= 8

    mlp = zf.build_shared_mlp(LAYER, 0)
    x, lg, routing = _constructed(layer, 6, 30, seed=5)
    torch.cuda.synchronize()
    cases = (
        (_Hop("identity"), routing, mlp),
        (_Hop("row"), routing, mlp),
        (_Hop("fp32"), None, mlp),
        (_Hop("fp32"), routing[:, :5].contiguous(), mlp),
        (None, routing, mlp),
        (_Hop("fp32"), routing, None),
        (_Hop("fp32"), routing, torch.nn.Linear(4, 4)),
    )
    for hop, r, shared in cases:
        before = torch.cuda.memory_allocated()
        assert (
            shared_prefetch_mod.routed_output_into_hop(
                layer.module, layer.bias, x, lg, r, hop, shared
            )
            is None
        )
        assert torch.cuda.memory_allocated() == before


@pytest.mark.parametrize("live", [1, 5, 6])
def test_padded_rows_do_not_change_live_rows(layer, runners, live):
    """An 8-row call whose rows past ``live`` are padding: the live rows' FP32 accumulator equals the live-row call's
    within the reorder bound."""
    routed_runner, _, shared_expert_prefetch = runners
    ((x, lg, routing),) = _real(layer, 8, 1, offset=4)
    full = torch.zeros(8, fx.HIDDEN, dtype=torch.float32, device="cuda")
    shared_expert_prefetch(x, routing, red_out=full)
    xl = x[:live].contiguous()
    rl = _pack(lg[:live].contiguous(), layer.bias)
    part = torch.zeros(live, fx.HIDDEN, dtype=torch.float32, device="cuda")
    shared_expert_prefetch(xl, rl, red_out=part)
    torch.cuda.synchronize()
    assert _acc_within_bound(routed_runner, xl, lg[:live].contiguous(), part)
    assert _acc_within_bound(routed_runner, xl, lg[:live].contiguous(), full[:live])
