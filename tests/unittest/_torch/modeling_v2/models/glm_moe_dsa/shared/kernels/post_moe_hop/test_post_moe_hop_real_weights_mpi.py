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
"""post_moe_hop fed by the native NVFP4 runner with real GLM-5.2 weights (collective).

Fixture: real weights of target layers (router, 256 routed NVFP4 experts,
BF16 shared expert, next-layer RMSNorm weight) loaded on every TP rank through
the native modules; synthetic activations (see ``post_moe_hop_real_fixture``).

For each case the same input goes through

* native chain: runner ``do_finalize=True`` -> ``shared + routed`` -> native
  ``AllReduce`` RESIDUAL_RMS_NORM (MNNVL one-shot across hosts), i.e. the
  served GEN multi-node path;
* candidate chain: runner ``do_finalize=False`` -> ``PostMoeHop``;

and must agree: the runner's own finalize equals the torch finalize oracle on
its raw tuple, the residual outputs are bit-exact, the normalized outputs are
within the error budget frozen in ``test_post_moe_hop_post_moe_mpi.py``. A CUDA-graph
case replays the captured candidate chain with fresh inputs.

Set ``GLM_MOE_DSA_POST_MOE_HOP_REAL_LAYERS`` (comma-separated layer ids, default ``10,77``).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_moe_hop_mpi_utils as U  # noqa: E402

LAYERS = tuple(
    int(x) for x in os.environ.get("GLM_MOE_DSA_POST_MOE_HOP_REAL_LAYERS", "10,77").split(",")
)
NORM_MAX_ULP = 2


def _log(env, message):
    if env.rank == 0:
        print(f"[post_moe_hop-real] {message}", flush=True)


@pytest.fixture(scope="module")
def env():
    e = U.mpi_env()
    if e.world < 2:
        pytest.skip("needs a TP group of at least two ranks under srun --mpi=pmix")
    return e


@pytest.fixture(scope="module", params=("triton", "cluster"))
def hop(env, request):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    U.native_allreduce()
    hop = PostMoeHop(
        env.mapping, U.GLM_HIDDEN, U.GLM_TOP_K, require_multi_node=False, impl=request.param
    )
    assert U.all_ranks(hop.impl == request.param), f"{request.param} kernel unavailable"
    return hop


@pytest.fixture(scope="module", params=LAYERS, ids=lambda layer: f"layer{layer}")
def real_layer(env, request):
    from post_moe_hop_real_fixture import build_real_moe_layer

    layer = build_real_moe_layer(env.mapping, request.param)
    torch.cuda.synchronize()
    env.comm.barrier()
    yield layer
    layer.moe.destroy()


def test_native_producer_declares_raw_contract(env, hop, real_layer):
    """The native NVFP4 MoE module passes the pre-runner producer check."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
        producer_capability,
    )

    declared, reason = producer_capability(
        real_layer.moe, U.GLM_HIDDEN, U.GLM_TOP_K, torch.bfloat16
    )
    bound = PostMoeHop(
        env.mapping,
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
        impl=hop.impl,
        raw_producer=real_layer.moe,
    )
    backend = type(getattr(real_layer.moe, "backend", real_layer.moe)).__name__
    _log(
        env,
        f"[{hop.impl}] layer {real_layer.layer} producer {type(real_layer.moe).__name__}/"
        f"{backend}: declared={declared} {reason!r} engaged={bound.engaged}",
    )
    assert U.all_ranks(declared and bound.engaged and bound.workspace is hop.workspace)


def _native_chain(layer, x, logits, shared, residual):
    routed = layer.run_moe(x, logits, do_finalize=True)
    local = shared + routed
    norm, new_residual = U.native_fused_allreduce(local, residual, layer.next_gamma, U.GLM_EPS)
    return routed, norm, new_residual


def _candidate_chain(layer, hop, x, logits, shared, residual):
    fc2, scales, perm = layer.run_moe(x, logits, do_finalize=False)
    out, new_residual = hop(fc2, scales, perm, shared, residual, layer.next_gamma, U.GLM_EPS)
    return (fc2, scales, perm), out, new_residual


@pytest.mark.parametrize("rows", (1, 6, 16))
def test_real_runner_native_vs_post_moe_hop(env, hop, real_layer, rows):
    with torch.inference_mode():
        x = real_layer.hidden_states(rows, seed=1000 + rows)
        residual = real_layer.residual(rows, seed=2000 + rows)
        logits = real_layer.gate(x)
        shared = real_layer.shared_output(x)
        routed, native_norm, native_res = _native_chain(real_layer, x, logits, shared, residual)
        raw, out, new_residual = _candidate_chain(real_layer, hop, x, logits, shared, residual)
        torch.cuda.synchronize()

    fc2, scales, perm = raw
    layout = {
        "fc2_base_mod16": fc2.data_ptr() % 16,
        "fc2_strides": tuple(fc2.stride()),
        "scales_contig_base_mod16": (scales.is_contiguous(), scales.data_ptr() % 16),
        "perm_contig_base_mod16": (perm.is_contiguous(), perm.data_ptr() % 16),
    }
    finalize_oracle = U.reference_local_partial(fc2, scales, perm, None)
    finalize_ulps = U.bf16_ulp_distance(finalize_oracle, routed)
    res_exact = torch.equal(new_residual.view(torch.int16), native_res.view(torch.int16))
    norm_stats = U.compare_stats(out, native_norm)
    res_stats = U.compare_stats(new_residual, native_res)
    try:
        torch.testing.assert_close(out, native_norm, rtol=0.05, atol=0.15)
        torch.testing.assert_close(new_residual, native_res, rtol=0.05, atol=0.15)
        within_tolerance = True
    except AssertionError:
        within_tolerance = False
    distinct = int(perm[perm >= 0].numel())
    fixture = {
        "layer": real_layer.layer,
        "rows": rows,
        "scale_dtype": str(scales.dtype),
        "fc2_shape": list(fc2.shape),
        "valid_slots": distinct,
        "finalize_max_ulp": int(finalize_ulps.max()),
        "finalize_frac_ulp_gt0": float((finalize_ulps > 0).float().mean()),
    }
    _log(
        env,
        f"[{hop.impl}] {fixture} raw_layout={layout} res_exact={res_exact} "
        f"res_vs_native={res_stats} norm_vs_native={norm_stats} tol={within_tolerance}",
    )
    # The native producer needs no layout repair.
    ok = (
        int(finalize_ulps.max()) == 0
        and res_exact
        and within_tolerance
        and norm_stats["max_ulp"] <= NORM_MAX_ULP
        and hop.error_bits() == 0
        and hop.workspace.layout_repairs == 0
    )
    assert U.all_ranks(ok), f"rank {env.rank}: {fixture} {res_stats} {norm_stats}"


def test_real_runner_graph_replay(env, hop, real_layer):
    rows = 6
    with torch.inference_mode():
        x = real_layer.hidden_states(rows, seed=3000)
        residual = real_layer.residual(rows, seed=3001)
        logits = real_layer.gate(x)
        shared = real_layer.shared_output(x)
        _candidate_chain(real_layer, hop, x, logits, shared, residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, graph_out, graph_res = _candidate_chain(real_layer, hop, x, logits, shared, residual)
        torch.cuda.synchronize()
        env.comm.barrier()
        ok = True
        worst = {}
        for replay in range(8):
            x.copy_(real_layer.hidden_states(rows, seed=3100 + replay))
            residual.copy_(real_layer.residual(rows, seed=3200 + replay))
            logits.copy_(real_layer.gate(x))
            shared.copy_(real_layer.shared_output(x))
            graph.replay()
            torch.cuda.synchronize()
            _, native_norm, native_res = _native_chain(real_layer, x, logits, shared, residual)
            torch.cuda.synchronize()
            res_exact = torch.equal(graph_res.view(torch.int16), native_res.view(torch.int16))
            stats = U.compare_stats(graph_out, native_norm)
            call_ok = res_exact and stats["max_ulp"] <= NORM_MAX_ULP
            ok &= call_ok
            if not call_ok and not worst:
                worst = {"replay": replay, "res_exact": res_exact, **stats}
        ok &= hop.error_bits() == 0
    _log(env, f"[{hop.impl}] layer {real_layer.layer} graph replay x8 ok={ok} {worst}")
    assert U.all_ranks(ok), worst
