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
"""Collective correctness tests of the post_moe_hop (the post_moe_hop contract).

Run one pytest process per GPU of the TP group, e.g. two 4-GPU hosts:

    srun --mpi=pmix -N2 --ntasks-per-node=4 ... \
        python3 -m pytest --import-mode=importlib -p no:cacheprovider -x -q \
        tests/unittest/_torch/modeling_v2/models/glm_moe_dsa/shared/kernels/post_moe_hop/test_post_moe_hop_post_moe_mpi.py

Every test is collective: all ranks run the same tests in the same order and
each verdict is AND-reduced across ranks before it is asserted, so a failure
on one rank fails the test everywhere instead of deadlocking the next test.
Inputs are synthetic raw tuples (rank-specific FC2 rows/weights/permutation,
replicated residual and norm weight); the real-weight native-runner check is
in ``test_post_moe_hop_real_weights_mpi.py``.

Oracles:
* the native finalize arithmetic in torch (``native_finalize_reference``) for
  the rank-local partial, which must equal what the kernel published;
* an exact emulation of the native MNNVL one-shot arithmetic applied to the
  partials every rank published (residual output must be bit-exact);
* the native TRT-LLM ``AllReduce`` (MNNVL across hosts) with RESIDUAL_RMS_NORM
  on the same partial (residual output bit-exact, normalized output compared
  in bf16 ulps and with the existing MNNVL test tolerance rtol=0.05/atol=0.15).
"""

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_moe_hop_mpi_utils as U  # noqa: E402

ROW_CASES = (1, 2, 6, 16)
NORM_MAX_ULP = 2


def _log(env, message):
    if env.rank == 0:
        print(f"[post_moe_hop-test] {message}", flush=True)


@pytest.fixture(scope="module")
def env():
    e = U.mpi_env()
    if e.world < 2:
        pytest.skip("needs a TP group of at least two ranks under srun --mpi=pmix")
    _log(e, f"world={e.world} gpus_per_node={e.gpus_per_node} multi_node={e.multi_node}")
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
    _log(env, f"--- hop impl={hop.impl}")
    return hop


def _check_call(env, hop, fc2, scales, perm, shared, residual, gamma, out, res_out, published=None):
    """Return (ok, stats) for one completed call against all three oracles."""
    ref_local = U.sanitize_bf16(U.reference_local_partial(fc2, scales, perm, shared))
    expected_slots = U.gather_bf16(ref_local).permute(1, 0, 2).cuda()  # [R, world, H]
    stats = {}
    ok = True
    if published is not None:
        slots_equal = torch.equal(published.view(torch.int16), expected_slots.view(torch.int16))
        stats["slots_exact"] = slots_equal
        ok &= slots_equal
    emu_norm, emu_res = U.emulate_post_reduction(expected_slots, residual, gamma, U.GLM_EPS)
    res_exact = torch.equal(res_out.view(torch.int16), emu_res.view(torch.int16))
    norm_stats = U.compare_stats(out, emu_norm)
    stats.update(res_exact=res_exact, norm_vs_emulation=norm_stats)
    ok &= res_exact and norm_stats["max_ulp"] <= NORM_MAX_ULP
    return ok, stats


class _Backend:
    def __init__(self, declares: bool):
        self._declares = declares

    def supports_unfinalized_output(self) -> bool:
        return self._declares


class _Routing:
    def __init__(self, top_k: int):
        self.experts_per_token = top_k


class _Producer:
    """Stand-in for a ``ConfigurableMoE``: the attributes ``producer_capability`` reads."""

    def __init__(
        self,
        declares=True,
        hidden=U.GLM_HIDDEN,
        top_k=U.GLM_TOP_K,
        dtype=torch.bfloat16,
        reduce_results=False,
    ):
        self.backend = _Backend(declares)
        self.routing_method = _Routing(top_k)
        self.hidden_size = hidden
        self.dtype = dtype
        self.reduce_results = reduce_results


def _reset_all(env, hop):
    assert U.all_ranks(hop.reset())
    assert U.all_ranks(hop.error_bits() == 0)


def test_capability_and_dispatch(env, hop):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
        check_health,
    )
    from tensorrt_llm.mapping import Mapping

    assert dispatch.get_impl("post_moe_hop", "post_moe_hop") is PostMoeHop
    assert dispatch.get_impl("post_moe_hop", "health_check") is check_health
    assert U.all_ranks(hop.engaged), f"hop not engaged: {hop.reason}"
    assert U.all_ranks(
        not hop.can_run(0)
        and hop.can_run(6)
        and hop.can_run(hop.max_rows)
        and not hop.can_run(hop.max_rows + 1)
    )

    # One vetoing rank must disengage every rank, without a hang.
    veto = "test veto" if env.rank == env.world - 1 else None
    vetoed = PostMoeHop(
        env.mapping, U.GLM_HIDDEN, U.GLM_TOP_K, require_multi_node=False, local_veto=veto
    )
    assert U.all_ranks(not vetoed.engaged and not vetoed.can_run(6))

    # A TP1 mapping never negotiates (no collective) and stays native.
    single = PostMoeHop(
        Mapping(world_size=1, rank=0, tp_size=1),
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
    )
    assert not single.engaged

    # The default domain is multi-node only; the default kernel is the cluster kernel.
    default = PostMoeHop(env.mapping, U.GLM_HIDDEN, U.GLM_TOP_K)
    assert U.all_ranks(default.engaged == env.multi_node)
    assert U.all_ranks(not default.engaged or default.impl == "cluster")

    # A cluster width that does not split H into whole warps falls back to Triton.
    fallback = PostMoeHop(
        env.mapping, U.GLM_HIDDEN, U.GLM_TOP_K, require_multi_node=False, impl="cluster", cluster=5
    )
    assert U.all_ranks(fallback.engaged and fallback.impl == "triton")
    _log(env, f"capability ok; multi_node={env.multi_node}")


@pytest.mark.parametrize(
    "case", ("declares", "no_declaration", "reduces", "top_k_one_rank", "hidden", "dtype")
)
def test_producer_contract_decided_before_runner(env, hop, case):
    """The producer's declared raw-tuple contract is part of the collective decision."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    producer = {
        "declares": _Producer(),
        "no_declaration": _Producer(declares=False),
        "reduces": _Producer(reduce_results=True),
        # A mismatch on one rank only must disengage every rank.
        "top_k_one_rank": _Producer(top_k=6 if env.rank == 1 else U.GLM_TOP_K),
        "hidden": _Producer(hidden=U.GLM_HIDDEN // 2),
        "dtype": _Producer(dtype=torch.float16),
    }[case]
    candidate = PostMoeHop(
        env.mapping,
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
        impl=hop.impl,
        raw_producer=producer,
    )
    expected = case == "declares"
    _log(
        env,
        f"[{hop.impl}] producer {case}: engaged={candidate.engaged} reason={candidate.reason!r}",
    )
    assert U.all_ranks(candidate.engaged == expected and candidate.can_run(6) == expected)


@pytest.mark.parametrize("scale_dtype", (torch.bfloat16, torch.float32), ids=("w_bf16", "w_fp32"))
@pytest.mark.parametrize("with_shared", (True, False), ids=("shared", "no_shared"))
@pytest.mark.parametrize("rows", ROW_CASES)
def test_matches_oracles(env, hop, rows, with_shared, scale_dtype):
    seed = 100 + rows * 4 + 2 * int(with_shared) + int(scale_dtype == torch.float32)
    fc2, scales, perm, shared, residual, gamma = U.make_raw_tuple(
        rows, seed=seed, scale_dtype=scale_dtype, with_shared=with_shared
    )
    out, res_out = hop(fc2, scales, perm, shared, residual, gamma, U.GLM_EPS)
    published = U.published_partials(hop, rows)
    ok, stats = _check_call(
        env, hop, fc2, scales, perm, shared, residual, gamma, out, res_out, published
    )

    # Native TRT-LLM hop on the identical rank partial.
    native_norm, native_res = U.native_fused_allreduce(
        published[:, env.rank].contiguous(), residual, gamma, U.GLM_EPS
    )
    torch.cuda.synchronize()
    native_res_exact = torch.equal(native_res.view(torch.int16), res_out.view(torch.int16))
    native_stats = U.compare_stats(out, native_norm)
    try:
        torch.testing.assert_close(out, native_norm, rtol=0.05, atol=0.15)
        torch.testing.assert_close(res_out, native_res, rtol=0.05, atol=0.15)
        within_tolerance = True
    except AssertionError:
        within_tolerance = False
    ok &= native_res_exact and within_tolerance and native_stats["max_ulp"] <= NORM_MAX_ULP
    ok &= hop.error_bits() == 0
    _log(
        env,
        f"[{hop.impl}] rows={rows} shared={with_shared} w={scale_dtype}: {stats} "
        f"native_res_exact={native_res_exact} norm_vs_native={native_stats}",
    )
    assert U.all_ranks(ok), f"rank {env.rank}: {stats} native={native_stats}"


@pytest.mark.parametrize("with_shared", (True, False), ids=("shared", "no_shared"))
def test_empty_and_padded_rows(env, hop, with_shared):
    rows = 6
    fc2, scales, perm, shared, residual, gamma = U.make_raw_tuple(
        rows, seed=300 + int(with_shared), with_shared=with_shared, empty_rows=(1, 4)
    )
    residual[2].zero_()  # a padded token row
    if shared is not None:
        shared[2].zero_()
    out, res_out = hop(fc2, scales, perm, shared, residual, gamma, U.GLM_EPS)
    published = U.published_partials(hop, rows)
    ok, stats = _check_call(
        env, hop, fc2, scales, perm, shared, residual, gamma, out, res_out, published
    )
    if not with_shared:
        ok &= bool((published[1].view(torch.int16) == 0).all())
        ok &= bool((published[4].view(torch.int16) == 0).all())
    ok &= bool(torch.isfinite(out.float()).all())
    _log(env, f"empty/padded rows shared={with_shared}: {stats}")
    assert U.all_ranks(ok), stats


@pytest.mark.parametrize(
    "block,warps,pdl", ((512, 2, False), (2048, 8, True), (8192, 16, True), (8192, 32, False))
)
def test_launch_configs(env, hop, block, warps, pdl):
    """Chunked and single-CTA-per-row grids, with and without PDL, share one workspace."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    variant = PostMoeHop(
        env.mapping,
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
        block=block,
        num_warps=warps,
        use_pdl=pdl,
        impl="triton",
    )
    ok = variant.engaged and variant.workspace is hop.workspace
    for rows in (1, 6):
        tensors = U.make_raw_tuple(rows, seed=350 + rows + block)
        out, res_out = variant(*tensors, U.GLM_EPS)
        call_ok, stats = _check_call(
            env, variant, *tensors, out, res_out, U.published_partials(variant, rows)
        )
        ok &= call_ok
    ok &= variant.error_bits() == 0
    _log(env, f"launch config block={block} warps={warps} pdl={pdl}: ok={ok}")
    assert U.all_ranks(ok), stats


def test_narrow_launch_config_disengages(env, hop):
    """A Triton program whose column vector does not span every warp is refused."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    variant = PostMoeHop(
        env.mapping,
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
        block=1024,
        num_warps=8,
        impl="triton",
    )
    _log(env, f"block=1024 num_warps=8: engaged={variant.engaged} reason={variant.reason!r}")
    assert U.all_ranks(not variant.engaged and "256 * num_warps" in variant.reason)


def _run_back_to_back(env, hops, sequence, seed):
    """Launch every call without a host sync, then check each one."""
    calls_before = hops[0].workspace.calls()
    inputs, outputs = [], []
    for i, rows in enumerate(sequence):
        tensors = U.make_raw_tuple(
            rows,
            seed=seed + i,
            with_shared=(i % 4 != 3),
            scale_dtype=torch.float32 if i % 5 == 0 else torch.bfloat16,
        )
        inputs.append(tensors)
        outputs.append(hops[i % len(hops)](*tensors, U.GLM_EPS))
    torch.cuda.synchronize()
    ok = hops[0].workspace.calls() - calls_before == len(sequence)
    failed = []
    for i, (tensors, (out, res_out)) in enumerate(zip(inputs, outputs)):
        call_ok, stats = _check_call(env, hops[0], *tensors, out, res_out)
        ok &= call_ok
        if not call_ok:
            failed.append(
                (i, sequence[i], stats["res_exact"], stats["norm_vs_emulation"]["max_ulp"])
            )
    ok &= hops[0].error_bits() == 0
    return ok, failed


def test_rapid_reuse_varying_rows(env, hop):
    sequence = [6, 1, 16, 3, 6, 6, 2, 16, 5, 6] * 3
    ok, failed = _run_back_to_back(env, [hop], sequence, seed=400)
    _log(env, f"[{hop.impl}] rapid reuse over {len(sequence)} calls ok={ok} failed={failed[:6]}")
    assert U.all_ranks(ok), failed


def test_mixed_launch_configs_back_to_back(env, hop):
    """The fixture's kernel interleaved with chunked and single-program Triton hops."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    variants = [hop] + [
        PostMoeHop(
            env.mapping,
            U.GLM_HIDDEN,
            U.GLM_TOP_K,
            require_multi_node=False,
            block=block,
            num_warps=warps,
            use_pdl=pdl,
            impl="triton",
        )
        for block, warps, pdl in ((8192, 16, True), (512, 2, False))
    ]
    sequence = [6, 16, 1, 6, 3, 16, 6, 2, 6, 5, 16, 6] * 2
    ok, failed = _run_back_to_back(env, variants, sequence, seed=450)
    _log(env, f"mixed configs over {len(sequence)} calls ok={ok} failed={failed[:6]}")
    assert U.all_ranks(ok), failed


def test_cuda_graph_replay_with_native_interleave(env, hop):
    num_layers, rows = 4, 6
    static = [U.make_raw_tuple(rows, seed=500 + layer) for layer in range(num_layers)]
    native_in = torch.randn(rows, U.GLM_HIDDEN, device="cuda").bfloat16()
    native_residual, native_gamma = static[0][4].clone(), static[0][5]

    def body():
        results = []
        native_out = None
        for layer, tensors in enumerate(static):
            results.append(hop(*tensors, U.GLM_EPS))
            if layer == 1:
                native_out = U.native_fused_allreduce(
                    native_in, native_residual, native_gamma, U.GLM_EPS
                )
        return results, native_out

    body()  # eager warmup: Triton compile and native workspaces
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_results, graph_native = body()
    torch.cuda.synchronize()
    env.comm.barrier()

    ok = True
    calls_before = hop.workspace.calls()
    for replay in range(20):
        for layer, tensors in enumerate(static):
            fresh = U.make_raw_tuple(
                rows, seed=600 + 10 * replay + layer, padded_rows=tensors[0].shape[0]
            )
            for dst, src in zip(tensors, fresh):
                if dst is not None:
                    dst.copy_(src)
        native_in.copy_(torch.randn_like(native_in.float()).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        for tensors, (out, res_out) in zip(static, graph_results):
            call_ok, stats = _check_call(env, hop, *tensors, out, res_out)
            ok &= call_ok
        gathered = U.gather_bf16(native_in).permute(1, 0, 2).cuda()
        _, native_res_ref = U.emulate_post_reduction(
            gathered, native_residual, native_gamma, U.GLM_EPS
        )
        ok &= torch.equal(graph_native[1].view(torch.int16), native_res_ref.view(torch.int16))
    ok &= hop.workspace.calls() - calls_before == 20 * num_layers
    ok &= hop.error_bits() == 0
    _log(env, f"[{hop.impl}] graph replay x20 with native interleave ok={ok}")
    assert U.all_ranks(ok)


def test_delayed_rank_stress(env, hop):
    rows = 6
    ok = True
    for it in range(12):
        victim = it % env.world
        tensors = U.make_raw_tuple(rows, seed=700 + it)
        env.comm.barrier()
        if env.rank == victim:
            if it % 3 == 0:
                time.sleep(0.02)  # host-side launch delay
            else:
                torch.cuda._sleep(int(2e6 * (1 + it % 4)))  # ~1-4 ms of device delay
        out, res_out = hop(*tensors, U.GLM_EPS)
        torch.cuda.synchronize()
        call_ok, stats = _check_call(env, hop, *tensors, out, res_out)
        ok &= call_ok
    ok &= hop.error_bits() == 0
    _log(env, f"[{hop.impl}] delayed-rank stress ok={ok}")
    assert U.all_ranks(ok)


def _missing_rank_trial(env, hop, impatient, missing, seed):
    """One call that rank ``missing`` skips; returns this rank's verdict."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )

    rows, timeout_s = 6, impatient.timeout_ns / 1e9
    tensors = U.make_raw_tuple(rows, seed=seed)
    before = U.lamport_snapshot(hop)
    _log(
        env,
        f"[{hop.impl}] Lamport invariant before rank {missing} skips, on every rank "
        f"(rank, epoch, prev_rows, ok): "
        f"{env.comm.allgather((env.rank, before['epoch'], before['prev_rows'], before['ok']))}",
    )
    assert U.all_ranks(before["ok"]), before
    cur = before["epoch"] % 3
    env.comm.barrier()
    start = time.monotonic()
    out = res_out = None
    if env.rank != missing:
        out, res_out = impatient(*tensors, U.GLM_EPS)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    counters = hop.workspace.counters()
    bits, polls = counters["error_bits"], counters["timeout_polls"]
    nan_chunks = slot_filled = None
    if env.rank == missing:
        reported = bits == 0 and polls == 0
    else:
        # Every program of the call must time out (the skipped rank's slot
        # stays empty) and therefore write NaN for all of its columns.
        slot_filled = U.lamport_snapshot(hop)["filled"][cur, :rows, missing].tolist()
        nan_chunks = [
            torch.isnan(t.float()).view(rows, U.GLM_HIDDEN // 1024, 1024).all(-1).sum(-1).tolist()
            for t in (out, res_out)
        ]
        reported = (
            bool(bits & 1)
            and polls > 0
            and timeout_s <= elapsed < 30.0
            and bool(torch.isnan(out.float()).all())
            and bool(torch.isnan(res_out.float()).all())
        )
    stats = env.comm.allgather((env.rank, round(elapsed, 3), bits, polls, reported))
    _log(
        env,
        f"[{hop.impl}] rank {missing} skipped; per rank (rank, elapsed_s, error_bits, "
        f"timeout_polls, ok): {stats}",
    )
    if not U.all_ranks(reported):
        detail = env.comm.allgather((env.rank, nan_chunks, slot_filled))
        _log(
            env,
            f"[{hop.impl}] per rank (rank, all-NaN 1024-column chunks per row of "
            f"[out, residual_out], filled elements per row of the skipped rank's slot): {detail}",
        )
        return False

    try:
        hop.raise_if_error()
        raised = False
    except RuntimeError:
        raised = True
    try:
        check_health(f"missing-rank{missing}-local")
        local_raised = False
    except PostMoeHopError:
        local_raised = True
    try:
        check_health(f"missing-rank{missing}-collective", collective=True)
        group_raised = False
    except PostMoeHopError:
        group_raised = True
    expected = env.rank != missing
    return raised == expected and local_raised == expected and group_raised


def test_bounded_error_and_collective_reset(env, hop):
    """A rank that never joins makes its peers time out, poison and report, not hang."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
        check_health,
    )

    impatient = PostMoeHop(
        env.mapping,
        U.GLM_HIDDEN,
        U.GLM_TOP_K,
        require_multi_node=False,
        timeout_s=0.2,
        impl=hop.impl,
    )
    assert U.all_ranks(impatient.workspace is hop.workspace)
    for trial, missing in enumerate(sorted({0, env.world // 2 + 1, env.world - 1})):
        ok = _missing_rank_trial(env, hop, impatient, missing, seed=800 + trial)
        assert U.all_ranks(ok), f"trial {trial}: rank {missing} skipped"
        _reset_all(env, hop)

    rows = 6
    tensors = U.make_raw_tuple(rows, seed=810)
    out, res_out = hop(*tensors, U.GLM_EPS)
    torch.cuda.synchronize()
    call_ok, stats = _check_call(env, hop, *tensors, out, res_out, U.published_partials(hop, rows))
    assert U.all_ranks(call_ok and hop.error_bits() == 0), stats
    health = check_health("after-reset", collective=True)
    assert U.all_ranks(not health.group_error and health.workspaces == 1)


def test_error_poisons_every_rank_until_reset(env, hop):
    """After an error recorded on one rank, every rank's next outputs are NaN."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop.kernels import (  # noqa: E501
        STATE_EPOCH,
        STATE_ERROR_WORDS,
    )

    victim = env.world - 2
    torch.cuda.synchronize()
    if env.rank == victim:
        # As if the previous call of this rank had timed out.
        epoch = int(hop.workspace.state[STATE_EPOCH].item())
        hop.workspace.state[STATE_ERROR_WORDS[(epoch + 2) % 3]] = 1
    torch.cuda.synchronize()
    env.comm.barrier()
    poisoned = True
    # Three calls cover every Lamport buffer, so the bits must be carried from
    # each buffer's error word into the next one.
    for call in range(3):
        tensors = U.make_raw_tuple(6, seed=850 + call)
        out, res_out = hop(*tensors, U.GLM_EPS)
        torch.cuda.synchronize()
        poisoned &= bool(torch.isnan(out.float()).all()) and bool(
            torch.isnan(res_out.float()).all()
        )
    words = hop.workspace.state.cpu().tolist()
    carried = all(words[w] == 1 for w in STATE_ERROR_WORDS)
    _log(
        env,
        f"[{hop.impl}] victim rank {victim} poisoned; three calls all NaN on this rank: "
        f"{poisoned}; per-rank error words: "
        f"{env.comm.allgather([words[w] for w in STATE_ERROR_WORDS])}",
    )
    assert U.all_ranks(
        poisoned
        and (carried if env.rank == victim else not any(words[w] for w in STATE_ERROR_WORDS))
    )
    with pytest.raises(PostMoeHopError):
        check_health("poisoned", collective=True)
    _reset_all(env, hop)
    tensors = U.make_raw_tuple(6, seed=860)
    out, res_out = hop(*tensors, U.GLM_EPS)
    torch.cuda.synchronize()
    call_ok, stats = _check_call(env, hop, *tensors, out, res_out, U.published_partials(hop, 6))
    assert U.all_ranks(call_ok), stats


def test_permutation_out_of_range_is_skipped_and_reported(env, hop):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )

    culprit = 2 % env.world
    rows = 6
    tensors = U.make_raw_tuple(rows, seed=870)
    if env.rank == culprit:
        tensors[2][1, 3] = tensors[0].shape[0]
        tensors[2][4, 0] = tensors[0].shape[0] + 1000
    out, res_out = hop(*tensors, U.GLM_EPS)
    published = U.published_partials(hop, rows)
    # The oracle skips entries outside [0, N_padded) like the kernels.
    call_ok, stats = _check_call(env, hop, *tensors, out, res_out, published)
    bits = hop.error_bits()
    call_ok &= bits == (4 if env.rank == culprit else 0)
    _log(env, f"[{hop.impl}] out-of-range permutation on rank {culprit}: {stats} bits={bits}")
    assert U.all_ranks(call_ok), (stats, bits)
    with pytest.raises(PostMoeHopError):
        check_health("perm-range", collective=True)
    _reset_all(env, hop)


def _misaligned(t: torch.Tensor, offset_elems: int) -> torch.Tensor:
    """Same values in a view whose base is ``offset_elems`` past a fresh allocation."""
    storage = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    view = storage[offset_elems : offset_elems + t.numel()].view(t.shape)
    view.copy_(t)
    return view


def _transposed(t: torch.Tensor) -> torch.Tensor:
    """Same values, non-contiguous (column-major) layout."""
    view = torch.empty(tuple(reversed(t.shape)), dtype=t.dtype, device=t.device).t()
    view.copy_(t)
    return view


def _padded_rows(t: torch.Tensor, extra_cols: int) -> torch.Tensor:
    """Same values in a ``[:, :H]`` view of a wider row-major allocation."""
    storage = torch.zeros((t.shape[0], t.shape[1] + extra_cols), dtype=t.dtype, device=t.device)
    view = storage[:, : t.shape[1]]
    view.copy_(t)
    return view


def _needs_copy(t: torch.Tensor, row_major_2d: bool = False) -> bool:
    """The documented 16-byte vector-access contract of the post_moe_hop kernels."""
    if t.data_ptr() % 16 or t.stride(-1) != 1:
        return True
    if row_major_2d:
        return (t.stride(0) * t.element_size()) % 16 != 0
    return not t.is_contiguous()


def _relayout(env, tensors):
    """Rank-dependent layouts; returns (tensors, expected repairs on this rank)."""
    fc2, scales, perm, shared, residual, gamma = tensors
    kind = env.rank % 4
    if kind == 0:
        fc2 = _misaligned(fc2, 1)  # base % 16 == 2
        perm = _misaligned(perm, 1)  # base % 16 == 4
        gamma = _misaligned(gamma, 2)  # base % 16 == 4
    elif kind == 1:
        fc2 = _padded_rows(fc2, 4)  # row pitch % 16 == 8
        scales = _transposed(scales)  # not contiguous (for R > 1)
        residual = _misaligned(residual, 4)  # base % 16 == 8
    elif kind == 2:
        fc2 = _padded_rows(fc2, 8)  # aligned row pitch: read in place
        if shared is not None:
            shared = _transposed(shared)  # inner stride != 1 (for R > 1)
    repairs = int(_needs_copy(fc2, row_major_2d=True)) + sum(
        int(_needs_copy(t)) for t in (scales, perm, shared, residual, gamma) if t is not None
    )
    return (fc2, scales, perm, shared, residual, gamma), repairs


def test_unaligned_layouts_are_repaired_locally(env, hop):
    """Per-rank layout repairs keep the collective and the arithmetic unchanged."""
    ok = True
    for i, rows in enumerate((6, 1, 16)):
        reference = U.make_raw_tuple(rows, seed=880 + i)
        tensors, expected = _relayout(env, reference)
        before = hop.workspace.layout_repairs
        out, res_out = hop(*tensors, U.GLM_EPS)
        call_ok, stats = _check_call(
            env, hop, *reference, out, res_out, U.published_partials(hop, rows)
        )
        repaired = hop.workspace.layout_repairs - before
        call_ok &= repaired == expected and (expected > 0) == (
            env.rank % 4 in (0, 1) or (rows > 1 and env.rank % 4 == 2)
        )
        ok &= call_ok
        _log(
            env,
            f"[{hop.impl}] relayout rows={rows}: {stats} repairs={repaired} expected={expected}",
        )

    # The repaired call inside a CUDA graph, replayed with fresh values.
    rows = 6
    static_ref = U.make_raw_tuple(rows, seed=890)
    static, expected = _relayout(env, static_ref)
    hop(*static, U.GLM_EPS)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_res = hop(*static, U.GLM_EPS)
    torch.cuda.synchronize()
    env.comm.barrier()
    for replay in range(4):
        fresh = U.make_raw_tuple(rows, seed=900 + replay, padded_rows=static_ref[0].shape[0])
        for dst, src in zip(static, fresh):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        call_ok, stats = _check_call(env, hop, *fresh, graph_out, graph_res)
        ok &= call_ok
    ok &= hop.error_bits() == 0
    _log(env, f"[{hop.impl}] relayout graph replay x4 ok={ok}")
    assert U.all_ranks(ok)


@pytest.mark.parametrize(
    "case",
    ("fc2_dtype", "scale_shape", "perm_dtype", "residual_shape", "shared_dtype", "gamma_shape"),
)
def test_contract_violation_raises_before_any_device_work(env, hop, case):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopContractError,
    )

    rows = 6
    fc2, scales, perm, shared, residual, gamma = U.make_raw_tuple(rows, seed=920)
    if case == "fc2_dtype":
        fc2 = fc2.half()
    elif case == "scale_shape":
        scales = scales[:, : U.GLM_TOP_K - 1]
    elif case == "perm_dtype":
        perm = perm.long()
    elif case == "residual_shape":
        residual = residual[:, : U.GLM_HIDDEN // 2]
    elif case == "shared_dtype":
        shared = shared.float()
    else:
        gamma = gamma[:-8]
    calls_before = hop.workspace.calls()
    with pytest.raises(PostMoeHopContractError):
        hop(fc2, scales, perm, shared, residual, gamma, U.GLM_EPS)
    assert U.all_ranks(hop.workspace.calls() == calls_before)
    tensors = U.make_raw_tuple(rows, seed=921)
    out, res_out = hop(*tensors, U.GLM_EPS)
    call_ok, stats = _check_call(env, hop, *tensors, out, res_out, U.published_partials(hop, rows))
    assert U.all_ranks(call_ok and hop.error_bits() == 0), stats


def test_health_check_reports_engagement(env, hop):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        check_health,
    )

    before = check_health("engagement-before", collective=True)
    for i in range(5):
        hop(*U.make_raw_tuple(6, seed=940 + i), U.GLM_EPS)
    after = check_health("engagement-after", collective=True)
    ok = (
        after.calls - before.calls == 5
        and after.error_bits == 0
        and not after.group_error
        and after.workspaces == 1
        and after.calls == hop.workspace.calls()
    )
    _log(env, f"[{hop.impl}] health {before} -> {after}")
    assert U.all_ranks(ok)
