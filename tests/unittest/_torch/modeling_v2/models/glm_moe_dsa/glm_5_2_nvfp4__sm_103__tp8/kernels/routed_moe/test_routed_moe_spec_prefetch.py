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
"""routed_moe expert-cluster kernel with package cross_layer_expert_prefetch's speculative expert prefetch.

The prefetch (``spec_k`` > 0) only issues ``cp.async.bulk.prefetch.L2``, so
every output must equal the prefetch-off kernel bit for bit, for any content
of the expert-ID row (including out-of-range IDs). The self-prediction store
(``spec_self``) must write the call's distinct experts ranked by row count.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

KMAX = 32
REORDER_REL = 2.0**-20


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10")),
        tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0")),
    )


@pytest.fixture(scope="module")
def runner(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    r = RoutedMoe.from_native(layer.module, layer.bias)
    if r.schedule(8, False, True) != "xc":
        pytest.skip("the speculative prefetch is part of the expert-cluster schedule")
    return r


class _Spec:
    """Install a cross_layer_expert_prefetch hook on the runner's expert leaf."""

    def __init__(self, runner, ids, k, ctas=0, self_store=False):
        self.owner = runner.module
        self.spec = (ids, k, ctas, self_store)

    def __enter__(self):
        self.owner._expert_prefetch_spec = lambda: self.spec
        return self

    def __exit__(self, *exc):
        del self.owner._expert_prefetch_spec


def _windows(layer, rows, count, offset=0):
    win = [x for x, _ in fx.real_windows(fx.load_real_capture(layer.layer), rows)]
    if not win:
        pytest.skip(
            f"no captured hidden states of layer {layer.layer} (GLM_MOE_DSA_REAL_MOE_INPUTS_DIR)"
        )
    step = max(1, len(win) // count)
    xs = [win[(offset + i * step) % len(win)].cuda().contiguous() for i in range(count)]
    return [(x, layer.router_logits(x)) for x in xs]


def _ids(kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "valid":
        ids = torch.randperm(fx.NUM_EXPERTS, generator=g)[:KMAX]
    else:  # out of range: the kernel masks each ID into [0, 256)
        ids = torch.randint(-(2**31), 2**31 - 1, (KMAX,), generator=g)
    return ids.to(torch.int32).cuda()


def _outputs(runner, x, logits):
    fin = runner(x, logits)
    fc2, scale, _ = runner(x, logits, do_finalize=False)
    acc = torch.zeros(x.shape[0], fx.HIDDEN, dtype=torch.float32, device="cuda")
    runner(x, logits, red_out=acc)
    torch.cuda.synchronize()
    return fin.clone(), fc2.clone(), scale.clone(), acc


def _check_same(ref, got, what):
    fin, fc2, scale, _ = ref
    gfin, gfc2, gscale, gacc = got
    assert torch.equal(gfin, fin), f"{what}: finalized output differs"
    assert torch.equal(gfc2, fc2), f"{what}: raw FC2 rows differ"
    assert torch.equal(gscale, scale), f"{what}: routing weights differ"
    # FP32 accumulator: atomic adds, compare within the FP32 reorder bound of
    # the exact sum of the same raw products
    rows = gacc.shape[0]
    terms = scale.double()[:, :, None] * fc2.view(rows, 8, -1).double()
    exact = terms.sum(1)
    bound = REORDER_REL * terms.abs().sum(1)
    assert ((gacc.double() - exact).abs() <= bound).all(), f"{what}: accumulator"


@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("k,ctas", [(12, 0), (32, 48), (1, 4)])
@pytest.mark.parametrize("ids_kind", ["valid", "out_of_range"])
def test_prefetch_outputs_bit_identical(layer, runner, rows, k, ctas, ids_kind):
    for i, (x, logits) in enumerate(_windows(layer, rows, 2, offset=rows)):
        ref = _outputs(runner, x, logits)
        ids = _ids(ids_kind, seed=rows * 100 + i)
        with _Spec(runner, ids, k, ctas):
            got = _outputs(runner, x, logits)
        _check_same(ref, got, f"rows={rows} k={k} ctas={ctas} {ids_kind} window {i}")


def test_prefetch_cuda_graph_replay(layer, runner):
    rows = 6
    wins = _windows(layer, rows, 6, offset=3)
    x_static, l_static = (t.clone() for t in wins[0])
    ids = _ids("valid", seed=7)
    with _Spec(runner, ids, 12):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner(x_static, l_static, do_finalize=False)
        torch.cuda.current_stream().wait_stream(stream)
        # the eager call compiled the prefetch variant: the capture records it
        assert runner.ready_for_capture(False, rows=rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fc2_s, scale_s, _ = runner(x_static, l_static, do_finalize=False)
    for it, (x, logits) in enumerate(wins):
        x_static.copy_(x)
        l_static.copy_(logits)
        ids.copy_(_ids("valid", seed=100 + it))
        graph.replay()
        torch.cuda.synchronize()
        fc2, scale, _ = runner(x, logits, do_finalize=False)
        torch.cuda.synchronize()
        assert torch.equal(fc2_s, fc2) and torch.equal(scale_s, scale), it


def _ranked_experts(logits, bias):
    ids, _ = fx.routing_reference(logits, bias)
    counts = torch.bincount(ids.flatten().long().cpu(), minlength=fx.NUM_EXPERTS)
    distinct = [e for e in range(fx.NUM_EXPERTS) if counts[e] > 0]
    return sorted(distinct, key=lambda e: (-int(counts[e]), e))


@pytest.mark.parametrize("rows", [1, 3, 6, 8])
def test_self_store_ranks_by_row_count(layer, runner, rows):
    for i, (x, logits) in enumerate(_windows(layer, rows, 3, offset=5 * rows)):
        ref = _outputs(runner, x, logits)
        ids = torch.full((KMAX,), 999, dtype=torch.int32, device="cuda")
        with _Spec(runner, ids, 12, self_store=True):
            fc2, scale, _ = runner(x, logits, do_finalize=False)
            torch.cuda.synchronize()
            got = ids.tolist()
            # the store is idempotent for the same call (the prefetch reads
            # the previous call's row first)
            gfin = runner(x, logits)
            torch.cuda.synchronize()
        assert torch.equal(fc2, ref[1]) and torch.equal(scale, ref[2]), (rows, i)
        assert torch.equal(gfin, ref[0]), (rows, i)
        want = _ranked_experts(logits, layer.bias)[:KMAX]
        assert got[: len(want)] == want, (rows, i, got, want)
        assert all(v == 999 for v in got[len(want) :]), (rows, i, got)
