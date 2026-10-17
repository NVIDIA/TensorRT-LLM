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
"""routed_moe_into_hop on the two-host TP8 group.

routed_moe writes its finalized routed output into the post_moe_hop, which adds the shared
output, reduces across the group and normalizes.

Run every test on all 8 ranks (``srun --mpi=pmix``, two hosts). The MoE is the built-in
``Deepseekv3MoE`` adopted by the target's ``GlmMoE``, and every test runs as a decode step.
Real GLM-5.2 weights (layers 10 and 77) and captured real hidden-state windows.

Oracles, per call (identical inputs on every arm):

* ``raw``: ``Deepseekv3MoE.forward(do_finalize=False)`` with routed_moe (raw tuple)
  -> post_moe_hop on the raw tuple, i.e. the composition without routed_moe_into_hop;
* ``native``: native MoE finalized -> native MNNVL ``AllReduce`` with
  RESIDUAL_RMS_NORM.

routed_moe_into_hop must equal ``raw`` bit for bit (same routed values, same post_moe_hop arithmetic
after the finalize) and match ``native`` like the composition does.

The hop is built, and its routed input enabled, inside the served model
loader's ``MetaInitMode``, as the target's decoder-layer adoption does in the
server (``routed_into_hop_checks.served_construction``).

Tests of the routed-input variant and the routed_moe finalize -> post_moe_hop
hand-off (helpers in ``routed_into_hop_checks.py``):

* ``test_routed_into_hop_served_construction_buffers``: per mode, a hop built inside the
  served loader's ``MetaInitMode`` has its routed-input buffers in device
  memory, and its routed_moe_into_hop calls equal the raw variant;
* ``test_routed_into_hop_shared_absent_variant``: the ``has_shared=False`` routed variant;
* ``test_routed_into_hop_mixed_rank_variants``: some ranks keep the raw tuple (routed_moe raw or
  the native runner) while the others use routed_moe_into_hop, on the shared post_moe_hop workspace;
  no hang, bit-equal to every rank on the raw variant, published partials
  and outputs equal to the native-arithmetic emulation;
* ``test_routed_into_hop_mixed_rank_graph``: the same mix inside a multi-stream CUDA
  graph, replayed on fresh windows;
* ``test_routed_into_hop_delayed_stress``: back-to-back batches with random per-rank
  producers, shared output and device delays, plus delay injection inside
  the routed_moe kernels (``routed_moe.DELAY_INJECTION``: CTA entry of the
  routed kernel and the finalize kernel's stores).
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routed_moe"))
sys.path.insert(
    0, os.path.join(os.path.dirname(HERE), "..", "..", "shared", "kernels", "post_moe_hop")
)
sys.path.insert(0, HERE)
import post_moe_hop_mpi_utils as U  # noqa: E402
import routed_into_hop_checks as G  # noqa: E402
import routed_moe_chain_bench as CB  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

LAYERS = [
    int(v) for v in os.environ.get("GLM_MOE_DSA_ROUTED_MOE_INTO_HOP_LAYERS", "10,77").split(",")
]


@pytest.fixture(autouse=True)
def decode_step():
    """Every test runs as a decode step, where the target's MoE takes its routed fast paths."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    with dispatch.step(dispatch.Phase.DECODE):
        yield


@pytest.fixture(scope="module")
def env():
    e = U.mpi_env()
    if e.world != 8 or not e.multi_node:
        pytest.skip("needs the two-host TP8 group")
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    return e


@pytest.fixture(scope="module")
def stack(env):
    # as the served model builds it: inside the loader's MetaInitMode, with the
    # layer's MoE module declaring the raw-tuple contract the hop consumes
    moes, gammas, hop = G.build_stack(env, LAYERS)
    eng = CB.Engagement()
    return moes, gammas, hop, eng


@pytest.mark.parametrize("mode", ["row", "identity", "fp32"])
def test_routed_into_hop_served_construction_buffers(env, stack, mode):
    """Every mode's routed-input buffers are device memory when the hop is
    built inside the served loader's ``MetaInitMode``, and a routed_moe_into_hop call in that
    mode equals the raw variant bit for bit (modes "row", "identity") or within
    the FP32 reorder bound ("fp32")."""
    moes, gammas, _, _ = stack
    with G.served_construction():
        probe = torch.empty(4, device="cuda")
    assert probe.is_meta, "the harness does not reproduce the served loader's MetaInitMode"
    layer = LAYERS[0]
    hop, enabled = G.build_hop(env, moes[layer].experts, mode)
    assert hop.engaged and enabled and hop.routed_mode == mode
    problems = G.buffer_problems(hop)
    assert U.all_ranks(not problems), f"rank {env.rank}: {problems}"
    saved = {m: m._routed_into_hop for m in moes.values()}
    try:
        for m in moes.values():
            m._routed_into_hop = hop
        G.warm(env, {layer: moes[layer]}, [6], producers=("routed_moe_into_hop", "routed_moe"))
        ok, n, fails = G.run_mixed(
            env, {layer: moes[layer]}, gammas, hop, "all-routed_moe_into_hop", True, [6], 2
        )
        hop.raise_if_error()
    finally:
        for m, h in saved.items():
            m._routed_into_hop = h
        CB.set_features("")
    if env.rank == 0:
        per_hop = G.buffer_bytes(hop)
        print(
            f"[routed_moe_into_hop] served construction mode={mode}: buffers {G.buffer_report(hop)}, {n} calls; "
            f"{per_hop} bytes per hop, {per_hop * G.MOE_LAYERS / 2**20:.1f} MiB per rank over "
            f"{G.MOE_LAYERS} MoE layers"
        )
    assert U.all_ranks(ok), f"rank {env.rank}: {fails}"


@pytest.mark.parametrize("bad_rank", [0, 5])
def test_routed_into_hop_buffer_problem_turns_routed_into_hop_off_on_every_rank(
    env, stack, monkeypatch, bad_rank
):
    """The buffer check's failure path: if one rank's routed-input buffer is not
    device memory (as an allocation under ``MetaInitMode`` leaves it),
    ``enable_routed_input`` returns False on every rank and the MoE keeps the
    raw tuple, before any device work touches the buffer."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        hop as hop_mod,
    )

    moes, gammas, _, _ = stack
    layer = LAYERS[0]
    if env.rank == bad_rank:
        monkeypatch.setattr(
            hop_mod.PostMoeHop,
            "_routed_device_buffer",
            lambda self, shape, dtype: torch.empty(shape, dtype=dtype, device="meta"),
        )
    hop, enabled = G.build_hop(env, moes[layer].experts, "identity")
    monkeypatch.undo()
    assert hop.engaged
    assert U.all_ranks(not enabled and not hop._routed_enabled), (
        "routed_moe_into_hop stayed on somewhere"
    )
    moe = moes[layer]
    saved = moe._routed_into_hop
    try:
        moe._routed_into_hop = hop
        CB.set_features("post_moe_hop,routed_moe,routed_moe_into_hop")
        out = moe.forward(_windows(layer, 6, 1)[0], do_finalize=False)
        torch.cuda.synchronize()
        assert len(out) == 4, (
            "routed_moe_into_hop must keep the raw tuple when its hop declined routed input"
        )
    finally:
        moe._routed_into_hop = saved
        CB.set_features("")
    if env.rank == 0:
        print(
            f"[routed_moe_into_hop] buffer problem on rank {bad_rank}: routed_moe_into_hop off on all {env.world} ranks"
        )


@pytest.mark.parametrize("odd_rank", [0, 6])
def test_routed_into_hop_modes_differing_across_ranks_turn_routed_into_hop_off_everywhere(
    env, stack, monkeypatch, odd_rank
):
    """``TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION`` is read per rank.

    It comes from each rank's environment. If one rank selects another mode
    than its peers, ``enable_routed_input`` agrees on that and turns routed_moe_into_hop off on
    every rank (the MoE keeps the raw tuple), instead of enabling a different
    routed-input variant per rank.
    """
    moes, _, _, _ = stack
    layer = LAYERS[0]
    monkeypatch.setenv(
        "TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION",
        "identity" if env.rank == odd_rank else "fp32",
    )
    hop, enabled = G.build_hop(env, moes[layer].experts)
    monkeypatch.delenv("TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION")
    assert hop.engaged
    assert U.all_ranks(not enabled and not hop._routed_enabled), (
        "routed_moe_into_hop stayed on somewhere"
    )
    # The same environment on every rank, spelled as the selector accepts it,
    # enables routed_moe_into_hop everywhere with the mode the feature record names.
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    monkeypatch.setenv("TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION", " Identity ")
    hop2, enabled2 = G.build_hop(env, moes[layer].experts)
    record = dispatch.feature_record({"post_moe_hop", "routed_moe", "routed_moe_into_hop"})
    monkeypatch.delenv("TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION")
    assert U.all_ranks(enabled2 and hop2.routed_mode == "identity")
    assert "routed_moe_into_hop[identity]" in record
    if env.rank == 0:
        print(
            f"[routed_moe_into_hop] mode differs on rank {odd_rank}: routed_moe_into_hop off on all {env.world} ranks"
        )


def _windows(layer: int, rows: int, count: int):
    win = [x for x, _ in fx.real_windows(fx.load_real_capture(layer), rows)]
    step = max(1, len(win) // count)
    return [win[i * step].cuda() for i in range(count)]


def _arm(moe, hop, x, res, gamma, features: str):
    from tensorrt_llm._torch.distributed import AllReduceParams

    CB.set_features(features)
    if features == "":
        h = moe.forward(
            x, do_finalize=True, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
        )
        return U.native_fused_allreduce(h, res, gamma, fx.RMS_EPS), None
    out = moe.forward(x, do_finalize=False)
    if len(out) == 2:
        shared, routed = out
        return hop.hop_routed(routed, shared, res, gamma, fx.RMS_EPS), routed
    shared, fc2, scale, perm = out
    return hop(fc2, scale, perm, shared, res, gamma, fx.RMS_EPS), None


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_routed_into_hop_equals_composition_and_matches_native(env, stack, rows):
    moes, gammas, hop, eng = stack
    mode = hop.routed_mode
    buffers = {
        "row": lambda: hop.routed_input_buffer(rows),
        "identity": lambda: hop.routed_raw_buffers(rows)[0],
        "fp32": lambda: hop.routed_acc_buffer(rows),
    }
    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    worst_native = [0, 0.0]
    exact_native = 0
    total = 0
    reordered = 0
    for layer in LAYERS:
        for x in _windows(layer, rows, 6):
            res = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 4.0).bfloat16()
            e0 = eng.engaged("routed_output_into_hop")
            CB.set_features("post_moe_hop,routed_moe,routed_moe_into_hop")
            out = moes[layer].forward(x, do_finalize=False)
            assert eng.engaged("routed_output_into_hop") == e0 + 1, (
                "routed_moe_into_hop did not engage"
            )
            assert len(out) == 2 and out[1].data_ptr() == buffers[mode]().data_ptr()
            # the hop clears the FP32 accumulator after reading: keep a copy
            seen = [out[0], out[1].clone() if mode == "fp32" else out[1]]
            w_out, w_res = G.hop_call(hop, "routed", out, res, gammas[layer], True)
            mine = G.local_partial("routed", seen, True, hop=hop)
            (r_out, r_res), _, r_payload = G.call(
                moes[layer], hop, x, res, gammas[layer], "routed_moe", True
            )
            (n_out, n_res), _ = _arm(moes[layer], hop, x, res, gammas[layer], "")
            torch.cuda.synchronize()
            if G.exact_mode(hop):
                assert torch.equal(w_res, r_res), (
                    f"L{layer} R={rows}: residual differs from routed_moe raw -> post_moe_hop"
                )
                assert torch.equal(w_out, r_out), (
                    f"L{layer} R={rows}: output differs from routed_moe raw -> post_moe_hop"
                )
            else:
                ok, n_diff = G.reorder_ok(mine, r_payload, True)
                assert ok, f"L{layer} R={rows}: partial outside the FP32 reorder bound"
                reordered += n_diff
            st = U.compare_stats(w_res, n_res)
            exact_native += int(st["max_ulp"] == 0)
            worst_native[0] = max(worst_native[0], st["max_ulp"])
            worst_native[1] = max(worst_native[1], st["max_abs"])
            torch.testing.assert_close(w_out, n_out, rtol=0.05, atol=0.15)
            total += 1
    CB.set_features("")
    allw = env.comm.allgather((exact_native, total, worst_native, reordered))
    if env.rank == 0:
        rel = "bit for bit" if G.exact_mode(hop) else "within the FP32 reorder bound"
        print(
            f"[routed_moe_into_hop] mode={mode} R={rows}: routed_moe_into_hop == composition (routed_moe "
            f"raw -> post_moe_hop) {rel} on all ranks; "
            f"residual vs native exact in {[a[0] for a in allw]} of {total} windows per rank, "
            f"worst ulp/abs {[a[2] for a in allw]}, reordered partial elements {[a[3] for a in allw]}"
        )


def test_routed_into_hop_cuda_graph_replay(env, stack):
    moes, gammas, hop, eng = stack
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    rows = 6
    layer = LAYERS[0]
    wins = _windows(layer, rows, 5)
    x = wins[0].clone()
    g = torch.Generator(device="cuda").manual_seed(7)
    res = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 4.0).bfloat16()
    CB.set_features("post_moe_hop,routed_moe,routed_moe_into_hop")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _arm(
                moes[layer],
                hop,
                x,
                res,
                gammas[layer],
                "post_moe_hop,routed_moe,routed_moe_into_hop",
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    env.comm.barrier()
    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        with torch.cuda.graph(graph, stream=stream):
            (g_out, g_res), _ = _arm(
                moes[layer],
                hop,
                x,
                res,
                gammas[layer],
                "post_moe_hop,routed_moe,routed_moe_into_hop",
            )
    torch.cuda.synchronize()
    for w in wins[1:]:
        x.copy_(w)
        env.comm.barrier()
        graph.replay()
        torch.cuda.synchronize()
        (e_out, e_res), _ = _arm(
            moes[layer], hop, x, res, gammas[layer], "post_moe_hop,routed_moe,routed_moe_into_hop"
        )
        torch.cuda.synchronize()
        assert G.same_or_close(hop, (g_out, g_res), (e_out, e_res)), "graph replay differs"
    CB.set_features("")
    hop.raise_if_error()


def test_routed_into_hop_falls_back_to_raw_without_routed_input(env, stack):
    moes, gammas, hop, eng = stack
    layer = LAYERS[0]
    x = _windows(layer, 6, 1)[0]
    moe = moes[layer]
    saved = moe._routed_into_hop
    try:
        moe._routed_into_hop = None
        CB.set_features("post_moe_hop,routed_moe,routed_moe_into_hop")
        out = moe.forward(x, do_finalize=False)
        assert len(out) == 4, (
            "without the hop's routed input routed_moe_into_hop must keep the raw tuple"
        )
    finally:
        moe._routed_into_hop = saved
        CB.set_features("")


_WARM = set()


def _warm(env, moes, rows_list):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as routed_moe_mod,
    )

    key = (tuple(rows_list), routed_moe_mod.DELAY_INJECTION)
    if key not in _WARM:
        G.warm(env, moes, rows_list)
        _WARM.add(key)


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_routed_into_hop_shared_absent_variant(env, stack, rows):
    moes, gammas, hop, eng = stack
    _warm(env, moes, [1, 6, 8])
    ok, n, fails = G.run_mixed(
        env, moes, gammas, hop, "all-routed_moe_into_hop", False, [rows], 3, seed=rows
    )
    hop.raise_if_error()
    if env.rank == 0:
        print(
            f"[routed_moe_into_hop] shared absent R={rows}: {n} calls, routed variant == raw variant"
        )
    assert U.all_ranks(ok), f"rank {env.rank}: {fails}"


@pytest.mark.parametrize("with_shared", [True, False], ids=["shared", "no_shared"])
@pytest.mark.parametrize(
    "pattern", ["odd-routed_moe", "rank0-routed_moe", "host1-routed_moe", "odd-native"]
)
def test_routed_into_hop_mixed_rank_variants(env, stack, pattern, with_shared):
    moes, gammas, hop, eng = stack
    _warm(env, moes, [1, 6, 8])
    ok, n, fails = G.run_mixed(env, moes, gammas, hop, pattern, with_shared, [1, 6, 8], 2)
    hop.raise_if_error()
    producers = env.comm.allgather(G.mixed_pattern(env, pattern))
    if env.rank == 0:
        print(
            f"[routed_moe_into_hop] mixed {pattern} shared={with_shared}: producers {producers}, {n} calls "
            f"bit-equal to the all-raw variant"
        )
    assert U.all_ranks(ok), f"rank {env.rank}: {fails}"


@pytest.mark.parametrize("pattern", ["odd-routed_moe", "odd-native"])
def test_routed_into_hop_mixed_rank_graph(env, stack, pattern):
    moes, gammas, hop, eng = stack
    _warm(env, moes, [1, 6, 8])
    ok, replays = G.run_graph(env, moes, gammas, hop, pattern)
    hop.raise_if_error()
    if env.rank == 0:
        print(f"[routed_moe_into_hop] graph mixed {pattern}: {replays} replays x 4 calls bit-equal")
    assert U.all_ranks(ok)


@pytest.mark.parametrize("kernel_delay", [False, True], ids=["no_kernel_delay", "kernel_delay"])
def test_routed_into_hop_delayed_stress(env, stack, monkeypatch, kernel_delay):
    moes, gammas, hop, eng = stack
    if kernel_delay:
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
            routed_moe as routed_moe_mod,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.kernel import (  # noqa: E501
            DELAY_ENTRY,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.kernel_xc import (  # noqa: E501
            DELAY_FINALIZE,
        )

        monkeypatch.setattr(routed_moe_mod, "DELAY_INJECTION", (DELAY_ENTRY | DELAY_FINALIZE, 3000))
    _warm(env, moes, [1, 6, 8])
    ok, stats = G.run_stress(
        env, moes, gammas, hop, batches=6, batch=16, max_delay_us=1500.0, seed=int(kernel_delay)
    )
    allstats = env.comm.allgather(stats)
    if env.rank == 0:
        print(f"[routed_moe_into_hop] delayed stress kernel_delay={kernel_delay}: {allstats}")
    hop.raise_if_error()
    assert U.all_ranks(ok), f"rank {env.rank}: {stats}"
