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
"""routed_moe's raw tuple through the native unfinalized consumer (one 4-GPU node).

On a single-node TP group the served DeepSeek-V3 layer hands the
``do_finalize=False`` MoE output to ``MoEAllReduce``, i.e. the native
``moe_finalize_allreduce`` op: a local finalize that follows the
permutation, the shared-expert add, the all-reduce, the residual add and the
RMSNorm. Run every test on the 4 GPUs of one node::

    srun -N1 --ntasks-per-node=4 --mpi=pmix python3 -m pytest ...
    test_routed_moe_native_consumer_mpi.py

Rank ``r`` holds TP8 shard ``r`` of a real GLM-5.2 NVFP4 layer (``GLM_MOE_DSA_ROUTED_MOE_LAYER``,
default 10) and runs captured real hidden-state windows of that layer. Per call
every rank feeds the same consumer three raw tuples:

* ``routed_moe``: routed_moe's token-major rows (row ``t * 8 + k``) with the identity
  permutation;
* ``routed_moe_relocated``: the same rows written at the native runner's permuted
  positions, with the native permutation and the native expert scales;
* ``native``: the native runner's raw tuple.

``routed_moe`` and ``routed_moe_relocated`` must give identical outputs bit for bit: the
consumer reads the token-major layout exactly like the native expert-grouped
layout. Where every rank's routed_moe rows equal the native rows bit for bit, ``routed_moe``
and ``native`` must give identical outputs as well; the other calls (an FP4
rounding flip inside routed_moe's frozen error budget) are counted and must stay
close.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(
    0, os.path.join(os.path.dirname(HERE), "..", "..", "shared", "kernels", "post_moe_hop")
)
import post_moe_hop_mpi_utils as U  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

LAYER = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10"))
WINDOWS = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_CONSUMER_WINDOWS", "8"))


@pytest.fixture(scope="module")
def env():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100-family GPU")
    e = U.mpi_env()
    if e.world < 2 or e.multi_node:
        pytest.skip("needs a single-node TP group of at least 2 ranks")
    return e


@pytest.fixture(scope="module")
def setup(env):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )
    from tensorrt_llm._torch.distributed import MoEAllReduce

    layer = fx.build_native_layer(layer=LAYER, tp_rank=env.rank)
    runner = RoutedMoe.from_native(layer.module, layer.bias)
    consumer = MoEAllReduce(env.mapping)
    # compile routed_moe's variants on every rank before the first collective call
    for rows in (1, 6, 8):
        x = torch.zeros(rows, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
        logits = layer.router_logits(x)
        runner(x, logits, do_finalize=False)
        layer.native_forward(x, logits, do_finalize=False)
    torch.cuda.synchronize()
    env.comm.barrier()
    return layer, runner, consumer


def _windows(rows: int, count: int):
    win = [x for x, _ in fx.real_windows(fx.load_real_capture(LAYER), rows)]
    assert win and win[0] is not None, f"no captured hidden states for layer {LAYER}"
    step = max(1, len(win) // count)
    return [win[i * step].cuda().contiguous() for i in range(count)]


def _consume(consumer, raw, shared, residual, gamma):
    from tensorrt_llm._torch.distributed import MoEAllReduceParams

    fc2, scale, idx = raw
    params = MoEAllReduceParams(
        expanded_idx_to_permuted_idx=idx,
        expert_scale_factor=scale,
        shared_expert_output=shared,
        residual=residual,
        norm_weight=gamma,
        eps=fx.RMS_EPS,
        is_cutlass_min_latency=False,
    )
    return consumer(fc2, all_reduce_params=params)


def _relocate(routed_moe_raw, native_raw):
    """routed_moe's rows at the native runner's permuted positions."""
    fc2_w, _, idx_w = routed_moe_raw
    fc2_n, scale_n, idx_n = native_raw
    assert fc2_n.shape[1] == fc2_w.shape[1], (fc2_n.shape, fc2_w.shape)
    out = torch.zeros_like(fc2_n)
    out[idx_n.reshape(-1).long()] = fc2_w[idx_w.reshape(-1).long()]
    return out, scale_n, idx_n


def _same(a, b) -> bool:
    return all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b))


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_raw_tuple_through_moe_finalize_allreduce(env, setup, rows):
    layer, runner, consumer = setup
    g_rank = torch.Generator(device="cuda").manual_seed(300 + 17 * env.rank + rows)
    g_common = torch.Generator(device="cuda").manual_seed(500 + rows)
    gamma = layer.norm_weight
    exact_calls, calls, worst = 0, 0, [0, 0.0]
    for x in _windows(rows, WINDOWS):
        logits = layer.router_logits(x)
        shared = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g_rank) * 0.1).bfloat16()
        residual = torch.randn(rows, fx.HIDDEN, device="cuda", generator=g_common) * 4.0
        residual = residual.bfloat16()
        native_raw = layer.native_forward(x, logits, do_finalize=False)
        routed_moe_raw = runner(x, logits, do_finalize=False)
        torch.cuda.synchronize()
        assert routed_moe_raw[0].shape == (rows * fx.TOP_K, fx.HIDDEN)
        ident = torch.arange(rows * fx.TOP_K, dtype=torch.int32)
        assert torch.equal(routed_moe_raw[2].view(-1).cpu(), ident), (
            "routed_moe permutation is not t * 8 + k"
        )
        assert (native_raw[2] >= 0).all(), "TP8/EP1: every (t, k) has a local native row"
        assert torch.equal(routed_moe_raw[1], native_raw[1]), "expert scales differ from native"
        rep = fx.raw_report(routed_moe_raw, native_raw)
        assert rep["same_expert"], rep
        rows_exact = U.all_ranks(rep["rows_exact"] == rep["rows"])

        out_w = _consume(consumer, routed_moe_raw, shared, residual, gamma)
        out_r = _consume(consumer, _relocate(routed_moe_raw, native_raw), shared, residual, gamma)
        out_n = _consume(consumer, native_raw, shared, residual, gamma)
        torch.cuda.synchronize()
        assert _same(out_w, out_r), "token-major layout read differently from the native layout"
        if rows_exact:
            assert _same(out_w, out_n), "identical rows, different consumer output"
            exact_calls += 1
        st = U.compare_stats(out_w[1], out_n[1])
        worst[0] = max(worst[0], st["max_ulp"])
        worst[1] = max(worst[1], st["max_abs"])
        torch.testing.assert_close(out_w[0], out_n[0], rtol=0.05, atol=0.15)
        calls += 1
    allw = env.comm.allgather((exact_calls, calls, worst))
    if env.rank == 0:
        print(
            f"[routed_moe] native consumer R={rows}: routed_moe raw == routed_moe relocated to the native layout "
            f"bit for bit on {env.world} ranks; routed_moe == native in {[a[0] for a in allw]} of "
            f"{calls} calls (all-rank-exact rows); residual vs native worst ulp/abs "
            f"{[a[2] for a in allw]}"
        )
