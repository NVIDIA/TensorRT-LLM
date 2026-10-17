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
"""Collective tests of the deferred_norm deferred hop (8 ranks over 2 hosts, TP8).

Run every test file process under ``srun --mpi=pmix`` (one process per GPU);
every test is collective and runs in the same order on every rank.

Exact invariants (the deferred_norm hop against the post_moe_hop on the same raw tuple and
the same workspace):

* ``residual_out`` bit-identical to post_moe_hop's;
* the fragment-layout ``a_frag`` holds ``bf16(x * gamma)`` bitwise (torch);
* ``sumsq`` bit-identical to the torch emulation of post_moe_hop's per-thread /
  butterfly order;
* ``materialize(packet)`` bit-identical to post_moe_hop's normalized output (this
  checks that the deferred ``r`` equals post_moe_hop's ``r`` bit for bit).

Workspace protocol shared with post_moe_hop (``post_moe_hop/kernels.py``; post_moe_hop's
error handling), with deferred_norm and post_moe_hop hops on one workspace:

* a rank that never joins makes every other rank's deferred_norm call time out after
  the post_moe_hop's deadline, record the Lamport-timeout bit and poll count, and
  return NaN (residual, activation, square sums, materialized output); post_moe_hop's
  health check fails, and after a collective reset the next call is exact;
* an error word left by an earlier call poisons every later deferred_norm and post_moe_hop call
  of the group (NaN) and is carried from buffer to buffer until the reset;
* a permutation entry at or beyond the FC2 row count is skipped (output as
  post_moe_hop's) and reported (bit 4);
* scrambled store order (fault-injected deferred_norm and post_moe_hop builds, chosen per rank
  and per call, mixed with plain builds and random device delays) changes
  no output bit;
* contract violations raise ``PostMoeHopContractError`` before any device work;
  layouts the kernel cannot read with 16-byte vectors are repaired on the
  rank that has them (counted in the workspace), also in a CUDA graph;
* the health check counts deferred_norm calls as hop calls.
"""

import functools
import os
import random
import sys
import time

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "post_moe_hop"))

import deferred_norm_utils as U  # noqa: E402
import post_moe_hop_mpi_utils as M  # noqa: E402


def _env():
    env = M.mpi_env()
    if env.world != 8:
        pytest.skip("the deferred_norm hop is specialized to TP8")
    return env


_HOPS = {}


def _hops(trigger_early=True):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    if "post_moe_hop" not in _HOPS:
        _HOPS["post_moe_hop"] = PostMoeHop(
            env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False
        )
    post_moe_hop = _HOPS["post_moe_hop"]
    key = ("deferred_norm", trigger_early)
    if key not in _HOPS:
        _HOPS[key] = DeferredNormHop(post_moe_hop, trigger_early=trigger_early)
    return post_moe_hop, _HOPS[key]


def _verdict(packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, tag):
    """This rank's packet invariants against post_moe_hop's outputs (not collective)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        materialize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    torch.cuda.synchronize()
    ok_res = torch.equal(res_deferred_norm, res_post_moe_hop)
    scaled_ref = (res_deferred_norm.float() * gamma.float()).bfloat16()
    ok_scaled = torch.equal(
        rows_from_frag(packet.a_frag, res_deferred_norm.shape[0], res_deferred_norm.shape[1]),
        scaled_ref,
    )
    ok_ss = torch.equal(packet.sumsq, U.hop_sumsq_partials(res_deferred_norm))
    h = materialize(packet)
    torch.cuda.synchronize()
    ok_h = torch.equal(h, h_post_moe_hop)
    ok = ok_res and ok_scaled and ok_ss and ok_h
    if not ok:
        print(
            f"GLM_MOE_DSA_DEFERRED_NORM_HOP_FAIL {tag} rank={M.mpi_env().rank} res={ok_res} scaled={ok_scaled} "
            f"sumsq={ok_ss} materialize={ok_h} "
            f"max_ulp_h={int(U.bf16_ulp_distance(h, h_post_moe_hop).max())}",
            flush=True,
        )
    return ok


def _check(packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, tag):
    """Collective: ``_verdict`` on every rank."""
    return M.all_ranks(
        _verdict(packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, tag)
    )


def _all_nan_packet(packet, res):
    """Every value a consumer can derive from the packet is NaN."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        materialize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    torch.cuda.synchronize()
    a = rows_from_frag(packet.a_frag, res.shape[0], res.shape[1])
    h = materialize(packet)
    torch.cuda.synchronize()
    return all(bool(torch.isnan(t.float()).all()) for t in (res, a, packet.sumsq, h))


def _reset(post_moe_hop):
    assert M.all_ranks(post_moe_hop.reset())
    assert M.all_ranks(post_moe_hop.error_bits() == 0)


def test_capability_engaged():
    post_moe_hop, deferred_norm = _hops()
    assert M.all_ranks(post_moe_hop.engaged and deferred_norm.engaged), deferred_norm.reason


def test_hop_precompiles_served_projections():
    """An engaged hop has compiled both served projection kernels on every
    rank (the consumer's compile is a cache hit)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.proj_kernel import (  # noqa: E501
        is_compiled,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        SERVED_N1,
        served_kernel_configs,
    )

    _, deferred_norm = _hops()
    cfgs = served_kernel_configs(torch.cuda.current_device())
    ok = (
        deferred_norm.engaged
        and sorted(c["n1"] for c in cfgs) == sorted(SERVED_N1)
        and all(is_compiled(**c) for c in cfgs)
    )
    assert M.all_ranks(ok)


@pytest.mark.parametrize("failing_rank", [3, 7])
def test_projection_compile_failure_disengages_every_rank(monkeypatch, failing_rank):
    """A projection compile error on one rank disengages the hop on all
    ranks (collective decision), so no rank produces a packet."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
        projection,
    )

    env = _env()
    post_moe_hop, _ = _hops()
    if env.rank == failing_rank:

        def _fail(device):
            raise RuntimeError("injected projection compile failure")

        monkeypatch.setattr(projection, "precompile_served", _fail)
    hop = DeferredNormHop(post_moe_hop)
    if env.rank == failing_rank:
        local = "injected projection compile failure" in hop.reason
    else:
        local = hop.reason == "another rank is not capable"
    assert M.all_ranks(not hop.engaged and local), hop.reason


def test_unsupported_projection_disengages_every_rank(monkeypatch):
    """A served projection shape the kernel does not support (static,
    rank-identical) disengages the hop on every rank."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501  # noqa: E501
        DeferredNormHop,
        projection,
    )

    post_moe_hop, _ = _hops()
    monkeypatch.setattr(
        projection,
        "DeferredNormProjConfig",
        functools.partial(projection.DeferredNormProjConfig, smem_rows=8),
    )
    hop = DeferredNormHop(post_moe_hop)
    assert M.all_ranks(not hop.engaged and "staged rows" in hop.reason), hop.reason


@pytest.mark.parametrize("trigger_early", [True, False])
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_shared", [True, False])
@pytest.mark.parametrize("rows", [1, 2, 6, 8])
def test_deferred_matches_post_moe_hop(rows, with_shared, scale_dtype, trigger_early):
    post_moe_hop, deferred_norm = _hops(trigger_early)
    with torch.inference_mode():
        fc2, scales, perm, shared, residual, gamma = M.make_raw_tuple(
            rows, seed=40 + rows, scale_dtype=scale_dtype, with_shared=with_shared
        )
        h_post_moe_hop, res_post_moe_hop = post_moe_hop(
            fc2, scales, perm, shared, residual, gamma, M.GLM_EPS
        )
        packet, res_deferred_norm = deferred_norm(
            fc2, scales, perm, shared, residual, gamma, M.GLM_EPS
        )
        assert _check(
            packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, f"rows={rows}"
        )


def test_empty_rows_and_large_residual():
    post_moe_hop, deferred_norm = _hops()
    with torch.inference_mode():
        fc2, scales, perm, shared, residual, gamma = M.make_raw_tuple(6, seed=77, empty_rows=(1, 4))
        residual = (residual.float() * 50.0).bfloat16()
        h_post_moe_hop, res_post_moe_hop = post_moe_hop(
            fc2, scales, perm, shared, residual, gamma, M.GLM_EPS
        )
        packet, res_deferred_norm = deferred_norm(
            fc2, scales, perm, shared, residual, gamma, M.GLM_EPS
        )
        assert _check(
            packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, "empty-rows"
        )


def test_back_to_back_mixed_with_post_moe_hop():
    """deferred_norm and post_moe_hop hops interleaved on one workspace without host syncs."""
    post_moe_hop, deferred_norm = _hops()
    rows_seq = [6, 1, 8, 3, 6, 2, 6, 8, 5, 6, 1, 6]
    with torch.inference_mode():
        inputs = [M.make_raw_tuple(r, seed=900 + i) for i, r in enumerate(rows_seq)]
        outs = []
        for i, t in enumerate(inputs):
            if i % 2 == 0:
                outs.append(("deferred_norm", deferred_norm(*t, M.GLM_EPS)))
            else:
                outs.append(("post_moe_hop", post_moe_hop(*t, M.GLM_EPS)))
        torch.cuda.synchronize()
        # Reference: each call again, alone, through post_moe_hop.
        failures = []
        for i, (t, (kind, out)) in enumerate(zip(inputs, outs)):
            h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
            if kind == "post_moe_hop":
                ok = torch.equal(out[0], h_ref) and torch.equal(out[1], res_ref)
            else:
                ok = _check(out[0], out[1], h_ref, res_ref, t[5], f"b2b{i}")
            if not ok:
                failures.append(i)
        assert M.all_ranks(not failures), failures
        assert post_moe_hop.error_bits() == 0


def test_cuda_graph_replay():
    post_moe_hop, deferred_norm = _hops()
    with torch.inference_mode():
        static = [M.make_raw_tuple(6, seed=1200 + i) for i in range(3)]
        bufs = [[x.clone() if x is not None else None for x in t] for t in static]

        def body():
            a = deferred_norm(*bufs[0], M.GLM_EPS)
            b = post_moe_hop(*bufs[1], M.GLM_EPS)
            c = deferred_norm(*bufs[2], M.GLM_EPS)
            return a, b, c

        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ga, gb, gc = body()
        for trial, replays in enumerate((1, 1, 20)):
            fresh = [M.make_raw_tuple(6, seed=1300 + 10 * trial + i) for i in range(3)]
            for dst, src in zip(bufs, fresh):
                for d, s in zip(dst, src):
                    if d is not None:
                        d.copy_(s)
            for _ in range(replays):
                M.mpi_env().comm.barrier()
                graph.replay()
            torch.cuda.synchronize()
            ok = True
            for (kind, got), t in zip(
                (("deferred_norm", ga), ("post_moe_hop", gb), ("deferred_norm", gc)), fresh
            ):
                h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
                if kind == "post_moe_hop":
                    ok &= torch.equal(got[0], h_ref) and torch.equal(got[1], res_ref)
                else:
                    ok &= _check(got[0], got[1], h_ref, res_ref, t[5], f"graph{trial}")
            assert M.all_ranks(ok), f"trial {trial}"


def test_delayed_rank_stress():
    """Random per-rank device delays inside back-to-back batches (no host sync)."""
    post_moe_hop, deferred_norm = _hops()
    env = M.mpi_env()
    gen = torch.Generator().manual_seed(4242)
    with torch.inference_mode():
        for batch in range(4):
            inputs = []
            outs = []
            for i in range(12):
                rows = int(torch.randint(1, 9, (1,), generator=gen))
                t = M.make_raw_tuple(rows, seed=5000 + 100 * batch + i)
                local = torch.Generator().manual_seed(77 * batch + 13 * i + env.rank)
                if float(torch.rand(1, generator=local)) < 0.3:
                    torch.cuda._sleep(int(float(torch.rand(1, generator=local)) * 2_000_000))
                inputs.append(t)
                outs.append(deferred_norm(*t, M.GLM_EPS))
            torch.cuda.synchronize()
            ok = True
            for t, (packet, res) in zip(inputs, outs):
                h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
                ok &= _check(packet, res, h_ref, res_ref, t[5], f"stress{batch}")
            assert M.all_ranks(ok), f"batch {batch}"
        assert post_moe_hop.error_bits() == 0


@pytest.mark.parametrize("layer", [10, 76])
def test_real_weights(layer):
    """Real routed experts / router / shared expert / next-layer gain (synthetic activations)."""
    from post_moe_hop_real_fixture import build_real_moe_layer

    post_moe_hop, deferred_norm = _hops()
    env = M.mpi_env()
    with torch.inference_mode():
        real = build_real_moe_layer(env.mapping, layer, max_num_tokens=64)
        ok = True
        for rows in (1, 6, 8):
            x = real.hidden_states(rows, seed=31 + rows)
            logits = real.gate(x)
            shared = real.shared_output(x)
            residual = real.residual(rows, seed=41 + rows)
            fc2, scales, perm = real.run_moe(x, logits, False)
            h_post_moe_hop, res_post_moe_hop = post_moe_hop(
                fc2, scales, perm, shared, residual, real.next_gamma, M.GLM_EPS
            )
            packet, res_deferred_norm = deferred_norm(
                fc2, scales, perm, shared, residual, real.next_gamma, M.GLM_EPS
            )
            ok &= _check(
                packet,
                res_deferred_norm,
                h_post_moe_hop,
                res_post_moe_hop,
                real.next_gamma,
                f"L{layer}R{rows}",
            )
        real.moe.destroy()
        assert M.all_ranks(ok)


# ---------------------------------------------------------------------------
# Workspace protocol shared with post_moe_hop: bounded errors, sticky poison, permutation
# range, scrambled store order, raw-tuple contract and layout repairs.
# ---------------------------------------------------------------------------

_MISSING_TIMEOUT_S = 0.2


def _impatient():
    """A post_moe_hop with a short poll deadline on the same workspace, and the deferred_norm
    hop built on it (collective; the deferred_norm kernels are compile-cache hits)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    if "impatient" not in _HOPS:
        post_moe_hop = PostMoeHop(
            env.mapping,
            M.GLM_HIDDEN,
            M.GLM_TOP_K,
            require_multi_node=False,
            timeout_s=_MISSING_TIMEOUT_S,
        )
        _HOPS["impatient"] = (post_moe_hop, DeferredNormHop(post_moe_hop))
    return _HOPS["impatient"]


@pytest.mark.parametrize("missing", [0, 5, 7])
def test_missing_rank_times_out_with_nan_and_reset(missing):
    """A rank that never joins makes every other rank's deferred_norm call time out,
    report and return NaN instead of hanging; a collective reset recovers."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )

    post_moe_hop, deferred_norm = _hops()
    imp_post_moe_hop, imp_deferred_norm = _impatient()
    env = M.mpi_env()
    assert M.all_ranks(
        imp_deferred_norm.engaged and imp_post_moe_hop.workspace is post_moe_hop.workspace
    )
    assert M.all_ranks(imp_deferred_norm.timeout_ns == int(_MISSING_TIMEOUT_S * 1e9))
    rows = 6
    with torch.inference_mode():
        tensors = M.make_raw_tuple(rows, seed=600 + missing)
        before = M.lamport_snapshot(post_moe_hop)
        assert M.all_ranks(before["ok"]), before
        env.comm.barrier()
        start = time.monotonic()
        packet = res = None
        if env.rank != missing:
            packet, res = imp_deferred_norm(*tensors, M.GLM_EPS)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        counters = post_moe_hop.workspace.counters()
        bits, polls = counters["error_bits"], counters["timeout_polls"]
        if env.rank == missing:
            reported = bits == 0 and polls == 0
        else:
            reported = (
                bool(bits & 1)
                and polls > 0
                and _MISSING_TIMEOUT_S <= elapsed < 30.0
                and _all_nan_packet(packet, res)
            )
        stats = env.comm.allgather((env.rank, round(elapsed, 3), bits, polls, reported))
        if env.rank == 0:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_MISSING rank {missing} skipped: (rank, elapsed_s, error_bits, "
                f"timeout_polls, ok) {stats}",
                flush=True,
            )
        assert M.all_ranks(reported), stats
        try:
            check_health(f"deferred_norm-missing-rank{missing}-local")
            local_raised = False
        except PostMoeHopError:
            local_raised = True
        try:
            check_health(f"deferred_norm-missing-rank{missing}-collective", collective=True)
            group_raised = False
        except PostMoeHopError:
            group_raised = True
        assert M.all_ranks(local_raised == (env.rank != missing) and group_raised)
        _reset(post_moe_hop)
        t = M.make_raw_tuple(rows, seed=650 + missing)
        packet, res = deferred_norm(*t, M.GLM_EPS)
        h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
        assert _check(packet, res, h_ref, res_ref, t[5], f"after-reset{missing}")
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize(
    "sequence",
    [
        "deferred_norm,deferred_norm,deferred_norm",
        "deferred_norm,post_moe_hop,deferred_norm",
        "post_moe_hop,deferred_norm,post_moe_hop",
    ],
)
def test_error_poisons_every_rank_until_reset(sequence):
    """An error word left by an earlier call on one rank makes the next calls
    of every rank NaN, deferred_norm and post_moe_hop alike, and is carried through all three
    Lamport buffers until the collective reset."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop.kernels import (  # noqa: E501
        STATE_EPOCH,
        STATE_ERROR_WORDS,
    )

    post_moe_hop, deferred_norm = _hops()
    env = M.mpi_env()
    victim = env.world - 2
    with torch.inference_mode():
        torch.cuda.synchronize()
        if env.rank == victim:
            # As if this rank's previous call had timed out.
            epoch = int(post_moe_hop.workspace.state[STATE_EPOCH].item())
            post_moe_hop.workspace.state[STATE_ERROR_WORDS[(epoch + 2) % 3]] = 1
        torch.cuda.synchronize()
        env.comm.barrier()
        poisoned = True
        for call, kind in enumerate(sequence.split(",")):
            t = M.make_raw_tuple(6, seed=860 + call)
            if kind == "deferred_norm":
                packet, res = deferred_norm(*t, M.GLM_EPS)
                poisoned &= _all_nan_packet(packet, res)
            else:
                out, res = post_moe_hop(*t, M.GLM_EPS)
                torch.cuda.synchronize()
                poisoned &= bool(torch.isnan(out.float()).all() and torch.isnan(res.float()).all())
        words = post_moe_hop.workspace.state.cpu().tolist()
        carried = all(words[w] == 1 for w in STATE_ERROR_WORDS)
        clean = not any(words[w] for w in STATE_ERROR_WORDS)
        if env.rank == 0:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_POISON {sequence}: per-rank error words "
                f"{env.comm.allgather([words[w] for w in STATE_ERROR_WORDS])}",
                flush=True,
            )
        else:
            env.comm.allgather([words[w] for w in STATE_ERROR_WORDS])
        assert M.all_ranks(poisoned and (carried if env.rank == victim else clean))
        with pytest.raises(PostMoeHopError):
            check_health("deferred_norm-poisoned", collective=True)
        _reset(post_moe_hop)
        t = M.make_raw_tuple(6, seed=870)
        packet, res = deferred_norm(*t, M.GLM_EPS)
        h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
        assert _check(packet, res, h_ref, res_ref, t[5], "after-poison-reset")


def test_permutation_out_of_range_is_skipped_and_reported():
    """Entries at or beyond the FC2 row count are skipped (as by post_moe_hop and the
    torch oracle) and set error bit 4 on the rank that has them. FC2 holds
    exactly the referenced rows, so under memcheck without the caching
    allocator a read of a skipped entry would be an invalid access."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop.kernels import (  # noqa: E501
        ERROR_PERM_RANGE,
    )

    post_moe_hop, deferred_norm = _hops()
    env = M.mpi_env()
    culprit = 2 % env.world
    rows = 6
    with torch.inference_mode():
        t = list(M.make_raw_tuple(rows, seed=880, padded_rows=rows * M.GLM_TOP_K))
        if env.rank == culprit:
            t[2] = t[2].clone()
            t[2][1, 3] = t[0].shape[0]
            t[2][4, 0] = t[0].shape[0] + 1000
        packet, res = deferred_norm(*t, M.GLM_EPS)
        torch.cuda.synchronize()
        bits = post_moe_hop.error_bits()
        reported = bits == (ERROR_PERM_RANGE if env.rank == culprit else 0)
        with pytest.raises(PostMoeHopError):
            check_health("deferred_norm-perm-range", collective=True)
        _reset(post_moe_hop)
        h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
        torch.cuda.synchronize()
        ok = _verdict(packet, res, h_ref, res_ref, t[5], "perm-range") and reported
        # The oracle skips entries outside [0, N_padded) like the kernels.
        ref_local = M.sanitize_bf16(M.reference_local_partial(t[0], t[1], t[2], t[3]))
        slots = M.gather_bf16(ref_local).permute(1, 0, 2).cuda()
        _, emu_res = M.emulate_post_reduction(slots, t[4], t[5], M.GLM_EPS)
        ok &= torch.equal(res.view(torch.int16), emu_res.view(torch.int16))
        assert M.all_ranks(ok), bits
        _reset(post_moe_hop)


_FAULT_DELAYS_NS = (3000, 60000)


def _fault_hops():
    """post_moe_hop and deferred_norm hops with fault-injected kernel builds on the same workspace."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    post_moe_hop, deferred_norm = _hops()
    if "fault" not in _HOPS:
        hops = {"post_moe_hop": post_moe_hop, "deferred_norm": deferred_norm}
        for d in _FAULT_DELAYS_NS:
            fp6 = PostMoeHop(
                env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False, fault_delay_ns=d
            )
            hops[f"post_moe_hop_fault{d}"] = fp6
            hops[f"deferred_norm_fault{d}"] = DeferredNormHop(fp6)
        _HOPS["fault"] = hops
    return _HOPS["fault"]


def test_fault_injected_store_order_mixed_with_post_moe_hop():
    """Scrambled store order inside the kernels (fault-injected builds), a
    per-rank choice of deferred_norm or post_moe_hop kernel for the same collective call, and
    random per-rank device delays: every output bit as in an undisturbed post_moe_hop
    call, the call counter advances once per call, no error bit."""
    hops = _fault_hops()
    post_moe_hop = hops["post_moe_hop"]
    env = M.mpi_env()
    names = sorted(hops)
    assert M.all_ranks(all(h.engaged for h in hops.values()))
    common = random.Random(4711)
    local = random.Random(1000 + env.rank)
    calls_before = post_moe_hop.workspace.calls()
    failed, used, total = [], {}, 0
    with torch.inference_mode():
        for batch in range(6):
            plan = [
                (
                    common.randint(1, 8),
                    common.random() < 0.75,
                    torch.float32 if common.random() < 0.2 else torch.bfloat16,
                )
                for _ in range(12)
            ]
            inputs = [
                M.make_raw_tuple(r, seed=9000 + 100 * batch + j, with_shared=sh, scale_dtype=sd)
                for j, (r, sh, sd) in enumerate(plan)
            ]
            picks = [local.choice(names) for _ in plan]
            torch.cuda.synchronize()
            env.comm.barrier()
            outs = []
            for name, t in zip(picks, inputs):
                if local.random() < 0.3:
                    torch.cuda._sleep(int(local.uniform(0.0, 50.0) * 2000))
                used[name] = used.get(name, 0) + 1
                outs.append(hops[name](*t, M.GLM_EPS))
            torch.cuda.synchronize()
            for j, (name, t, out) in enumerate(zip(picks, inputs, outs)):
                h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
                if name.startswith("deferred_norm"):
                    ok = _verdict(out[0], out[1], h_ref, res_ref, t[5], f"fault{batch}.{j}")
                else:
                    torch.cuda.synchronize()
                    ok = torch.equal(out[0], h_ref) and torch.equal(out[1], res_ref)
                if not ok:
                    failed.append((batch, j, name))
            total += len(plan)
        counted = post_moe_hop.workspace.calls() - calls_before
    bits = post_moe_hop.error_bits()
    if env.rank == 0:
        print(
            f"GLM_MOE_DSA_DEFERRED_NORM_FAULT calls={total} (+{total} reference calls) used_on_rank0={used}",
            flush=True,
        )
    assert M.all_ranks(not failed and counted == 2 * total and bits == 0), (
        failed[:4],
        counted,
        bits,
    )


@pytest.mark.parametrize(
    "case",
    ("fc2_dtype", "scale_shape", "perm_dtype", "residual_shape", "shared_dtype", "gamma_shape"),
)
def test_contract_violation_raises_before_any_device_work(case):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopContractError,
    )

    post_moe_hop, deferred_norm = _hops()
    rows = 6
    with torch.inference_mode():
        fc2, scales, perm, shared, residual, gamma = M.make_raw_tuple(rows, seed=920)
        if case == "fc2_dtype":
            fc2 = fc2.half()
        elif case == "scale_shape":
            scales = scales[:, : M.GLM_TOP_K - 1]
        elif case == "perm_dtype":
            perm = perm.long()
        elif case == "residual_shape":
            residual = residual[:, : M.GLM_HIDDEN // 2]
        elif case == "shared_dtype":
            shared = shared.float()
        else:
            gamma = gamma[:-8]
        calls_before = post_moe_hop.workspace.calls()
        with pytest.raises(PostMoeHopContractError):
            deferred_norm(fc2, scales, perm, shared, residual, gamma, M.GLM_EPS)
        assert M.all_ranks(post_moe_hop.workspace.calls() == calls_before)
        t = M.make_raw_tuple(rows, seed=921)
        packet, res = deferred_norm(*t, M.GLM_EPS)
        h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
        assert _check(packet, res, h_ref, res_ref, t[5], f"after-contract-{case}")


def _misaligned(t, offset_elems):
    storage = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    view = storage[offset_elems : offset_elems + t.numel()].view(t.shape)
    view.copy_(t)
    return view


def _transposed(t):
    view = torch.empty(tuple(reversed(t.shape)), dtype=t.dtype, device=t.device).t()
    view.copy_(t)
    return view


def _padded_rows(t, extra_cols):
    storage = torch.zeros((t.shape[0], t.shape[1] + extra_cols), dtype=t.dtype, device=t.device)
    view = storage[:, : t.shape[1]]
    view.copy_(t)
    return view


def _needs_copy(t, row_major_2d=False):
    if t.data_ptr() % 16 or t.stride(-1) != 1:
        return True
    if row_major_2d:
        return (t.stride(0) * t.element_size()) % 16 != 0
    return not t.is_contiguous()


def _relayout(rank, tensors):
    """Rank-dependent layouts; returns (tensors, expected repairs on this rank)."""
    fc2, scales, perm, shared, residual, gamma = tensors
    kind = rank % 4
    if kind == 0:
        fc2 = _misaligned(fc2, 1)
        perm = _misaligned(perm, 1)
        gamma = _misaligned(gamma, 2)
    elif kind == 1:
        fc2 = _padded_rows(fc2, 4)
        scales = _transposed(scales)
        residual = _misaligned(residual, 4)
    elif kind == 2:
        fc2 = _padded_rows(fc2, 8)  # aligned row pitch: read in place
        if shared is not None:
            shared = _transposed(shared)
    repairs = int(_needs_copy(fc2, row_major_2d=True)) + sum(
        int(_needs_copy(t)) for t in (scales, perm, shared, residual, gamma) if t is not None
    )
    return (fc2, scales, perm, shared, residual, gamma), repairs


def test_unaligned_layouts_are_repaired_locally():
    """Per-rank layout repairs keep the collective and every output bit."""
    post_moe_hop, deferred_norm = _hops()
    env = M.mpi_env()
    ok = True
    with torch.inference_mode():
        for i, rows in enumerate((6, 1, 8)):
            reference = M.make_raw_tuple(rows, seed=880 + i)
            tensors, expected = _relayout(env.rank, reference)
            before = post_moe_hop.workspace.layout_repairs
            packet, res = deferred_norm(*tensors, M.GLM_EPS)
            repaired = post_moe_hop.workspace.layout_repairs - before
            h_ref, res_ref = post_moe_hop(*reference, M.GLM_EPS)
            ok &= _verdict(packet, res, h_ref, res_ref, reference[5], f"relayout{rows}")
            ok &= repaired == expected
        # The repaired call inside a CUDA graph, replayed with fresh values.
        rows = 6
        static_ref = M.make_raw_tuple(rows, seed=890)
        static, _ = _relayout(env.rank, static_ref)
        deferred_norm(*static, M.GLM_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_packet, g_res = deferred_norm(*static, M.GLM_EPS)
        torch.cuda.synchronize()
        env.comm.barrier()
        for replay in range(4):
            fresh = M.make_raw_tuple(rows, seed=900 + replay, padded_rows=static_ref[0].shape[0])
            for dst, src in zip(static, fresh):
                if dst is not None:
                    dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            h_ref, res_ref = post_moe_hop(*fresh, M.GLM_EPS)
            ok &= _verdict(g_packet, g_res, h_ref, res_ref, fresh[5], f"relayout-graph{replay}")
    assert M.all_ranks(ok and post_moe_hop.error_bits() == 0)


def test_health_check_counts_deferred_norm_calls():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        check_health,
    )

    post_moe_hop, deferred_norm = _hops()
    with torch.inference_mode():
        before = check_health("deferred_norm-engagement-before", collective=True)
        for i in range(5):
            deferred_norm(*M.make_raw_tuple(6, seed=940 + i), M.GLM_EPS)
        after = check_health("deferred_norm-engagement-after", collective=True)
    assert M.all_ranks(
        after.calls - before.calls == 5
        and after.error_bits == 0
        and not after.group_error
        and after.calls == post_moe_hop.workspace.calls()
    )


def test_memcheck_positive_control():
    """Tests only, run under compute-sanitizer memcheck by the sanitizer
    suite (``GLM_MOE_DSA_DEFERRED_NORM_MEMCHECK_CONTROL=1``): the kernel is told FC2 has one more
    row than it does and the permutation references that row, so memcheck
    must report an invalid global read in the deferred_norm hop."""
    if os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_MEMCHECK_CONTROL") != "1":
        pytest.skip("memcheck positive control (sanitizer suite only)")
    post_moe_hop, deferred_norm = _hops()
    rows = 1
    with torch.inference_mode():
        t = list(M.make_raw_tuple(rows, seed=990, padded_rows=rows * M.GLM_TOP_K))
        t[2] = t[2].clone()
        t[2][0, 0] = t[0].shape[0]
        deferred_norm.launch(*t, M.GLM_EPS, fc2_rows=t[0].shape[0] + 1)
        torch.cuda.synchronize()
    _reset(post_moe_hop)
