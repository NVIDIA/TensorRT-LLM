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
"""Collective tests of the deferred_norm deferred hop on routed_moe_into_hop's routed input (8 ranks, 2 hosts).

With ``routed_moe_into_hop`` selected, routed_moe writes its per-rank routed result into the
post_moe_hop hop's routed-input buffer (``PostMoeHop.enable_routed_input``; modes
``fp32`` (default), ``identity``, ``row``) and the layer calls
``PostMoeHop.hop_routed``. ``DeferredNormHop.hop_routed`` takes the same
inputs and returns the deferred packet. The routed content is written here
as routed_moe would write it (per rank), from a synthetic raw tuple.

Exact invariants, per mode, against ``PostMoeHop.hop_routed`` on the same
routed content and the same workspace:

* ``residual_out`` bit-identical;
* ``a_frag`` holds ``bf16(x * gamma)`` bitwise; ``sumsq`` equals the torch
  emulation of the hop's order; ``materialize(packet)`` equals
  ``hop_routed``'s normalized output bit for bit;
* ``fp32`` mode: the hop clears the accumulator rows it read (the buffer is
  zero after every call, as after post_moe_hop's).

post_moe_hop's error protocol on the routed variant, mixed with post_moe_hop routed, post_moe_hop
raw and deferred_norm raw calls on one workspace: missing rank (deadline, NaN, health check,
reset), sticky poison carried through every Lamport buffer, fault-injected
store order with per-rank kernel choice, contract violations before any
device work, per-rank layout repairs (also in a CUDA graph), graph replay,
the health check's call count, and the collective enable (post_moe_hop routed input
off, compile failure on one rank, modes that differ across ranks, mode
switch).
"""

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

MODES = ("fp32", "identity", "row")


def _env():
    env = M.mpi_env()
    if env.world != 8:
        pytest.skip("the deferred_norm hop is specialized to TP8")
    return env


def _routed_api():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    if not hasattr(PostMoeHop, "enable_routed_input"):
        pytest.skip("the installed post_moe_hop hop has no routed_moe_into_hop routed input")


_HOPS = {}


def _hops(mode, trigger_early=True, timeout_s=None, fault_delay_ns=0):
    """A post_moe_hop with its routed input in ``mode`` and a deferred_norm hop on it (collective)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    _routed_api()
    key = (mode, timeout_s, fault_delay_ns)
    if key not in _HOPS:
        kw = {}
        if timeout_s is not None:
            kw["timeout_s"] = timeout_s
        if fault_delay_ns:
            kw["fault_delay_ns"] = fault_delay_ns
        post_moe_hop = PostMoeHop(
            env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False, **kw
        )
        assert M.all_ranks(post_moe_hop.engaged and post_moe_hop.enable_routed_input(mode)), (
            post_moe_hop.reason
        )
        _HOPS[key] = {"post_moe_hop": post_moe_hop}
    entry = _HOPS[key]
    if trigger_early not in entry:
        deferred_norm = DeferredNormHop(entry["post_moe_hop"], trigger_early=trigger_early)
        assert M.all_ranks(deferred_norm.engaged and deferred_norm.enable_routed_input()), (
            deferred_norm.reason,
            deferred_norm.routed_reason,
        )
        entry[trigger_early] = deferred_norm
    return entry["post_moe_hop"], entry[trigger_early]


def _source(mode, rows, seed, with_shared=True):
    """Per-rank routed content as routed_moe writes it in ``mode``, plus shared /
    residual / gamma (the residual and gamma are replicated)."""
    fc2, scales, perm, shared, residual, gamma = M.make_raw_tuple(
        rows, seed=seed, with_shared=with_shared
    )
    if mode == "row":
        src = M.reference_local_partial(fc2, scales, perm, None)
    elif mode == "identity":
        valid = perm >= 0
        safe = torch.where(valid, perm, torch.zeros_like(perm)).long()
        src = (
            fc2[safe.view(-1)].contiguous(),
            torch.where(valid, scales, torch.zeros_like(scales)).bfloat16().contiguous(),
        )
    else:
        valid = (perm >= 0).unsqueeze(-1)
        safe = torch.where(perm >= 0, perm, torch.zeros_like(perm)).long()
        prod = fc2[safe.view(-1)].view(rows, M.GLM_TOP_K, -1).float() * scales.float().unsqueeze(-1)
        src = torch.where(valid, prod, torch.zeros_like(prod)).sum(1).contiguous()
    return src, shared, residual, gamma


def _fill(post_moe_hop, mode, src, rows):
    """Write ``src`` into the post_moe_hop's routed buffer (stream-ordered); return the view."""
    if mode == "row":
        buf = post_moe_hop.routed_input_buffer(rows)
        buf.copy_(src)
        return buf
    if mode == "identity":
        buf, scales = post_moe_hop.routed_raw_buffers(rows)
        buf.copy_(src[0])
        scales.copy_(src[1])
        return buf
    buf = post_moe_hop.routed_acc_buffer(rows)
    buf.copy_(src)
    return buf


def _post_moe_hop_reference(post_moe_hop, mode, src, rows, shared, residual, gamma):
    routed = _fill(post_moe_hop, mode, src, rows)
    return post_moe_hop.hop_routed(routed, shared, residual, gamma, M.GLM_EPS)


def _deferred_norm_call(post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma):
    routed = _fill(post_moe_hop, mode, src, rows)
    return deferred_norm.hop_routed(routed, shared, residual, gamma, M.GLM_EPS)


def _acc_clear(post_moe_hop, mode):
    if mode != "fp32":
        return True
    torch.cuda.synchronize()
    return bool((post_moe_hop.routed_acc_buffer(post_moe_hop.max_rows) == 0).all())


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
            f"GLM_MOE_DSA_DEFERRED_NORM_ROUTED_FAIL {tag} rank={M.mpi_env().rank} res={ok_res} scaled={ok_scaled} "
            f"sumsq={ok_ss} materialize={ok_h} "
            f"max_ulp_h={int(U.bf16_ulp_distance(h, h_post_moe_hop).max())}",
            flush=True,
        )
    return ok


def _check(packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, tag):
    return M.all_ranks(
        _verdict(packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, tag)
    )


def _all_nan_packet(packet, res):
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


# ---------------------------------------------------------------------------
# Collective enable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_routed_engaged(mode):
    post_moe_hop, deferred_norm = _hops(mode)
    assert M.all_ranks(deferred_norm.routed_engaged and deferred_norm.routed_mode == mode)
    md = type("Md", (), {"num_contexts": 0, "num_generations": 6})()
    ctx = type("Md", (), {"num_contexts": 1, "num_generations": 0})()
    assert M.all_ranks(
        deferred_norm.can_run_routed(6, md)
        and not deferred_norm.can_run_routed(6, ctx)
        and not deferred_norm.can_run_routed(9, md)
        and not deferred_norm.can_run_routed(0, md)
    )


def test_routed_requires_post_moe_hop_routed_input():
    """Without the post_moe_hop's routed input the routed variant stays off on every rank."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    _routed_api()
    post_moe_hop = PostMoeHop(env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False)
    deferred_norm = DeferredNormHop(post_moe_hop)
    ok = (
        deferred_norm.engaged
        and not deferred_norm.enable_routed_input()
        and not deferred_norm.routed_engaged
    )
    ok &= "routed input is not enabled" in deferred_norm.routed_reason
    md = type("Md", (), {"num_contexts": 0, "num_generations": 6})()
    ok &= not deferred_norm.can_run_routed(6, md)
    with torch.inference_mode():
        src, shared, residual, gamma = _source("fp32", 6, seed=31)
        with pytest.raises(ValueError):
            deferred_norm.hop_routed(src, shared, residual, gamma, M.GLM_EPS)
    assert M.all_ranks(ok), deferred_norm.routed_reason


@pytest.mark.parametrize("failing_rank", [2, 7])
def test_routed_compile_failure_disengages_every_rank(monkeypatch, failing_rank):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
        hop_kernel,
    )

    env = _env()
    post_moe_hop, _ = _hops("fp32")
    deferred_norm = DeferredNormHop(post_moe_hop)
    if env.rank == failing_rank:
        real = hop_kernel.compiled_hop

        def _fail(*args, **kwargs):
            if kwargs.get("routed_input"):
                raise RuntimeError("injected routed compile failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(hop_kernel, "compiled_hop", _fail)
    engaged = deferred_norm.enable_routed_input()
    if env.rank == failing_rank:
        local = "injected routed compile failure" in deferred_norm.routed_reason
    else:
        local = deferred_norm.routed_reason == "another rank is not capable"
    assert M.all_ranks(
        deferred_norm.engaged and not engaged and not deferred_norm.routed_engaged and local
    ), deferred_norm.routed_reason


def test_routed_modes_must_agree_across_ranks():
    """post_moe_hop hops whose routed-input modes differ across ranks (environment)
    leave the deferred_norm routed variant off on every rank."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    _routed_api()
    post_moe_hop = PostMoeHop(env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False)
    assert M.all_ranks(post_moe_hop.enable_routed_input("fp32" if env.rank % 2 else "identity"))
    deferred_norm = DeferredNormHop(post_moe_hop)
    engaged = deferred_norm.enable_routed_input()
    assert M.all_ranks(not engaged and "modes differ" in deferred_norm.routed_reason), (
        deferred_norm.routed_reason
    )


def test_routed_mode_switch_needs_reenable():
    """After the post_moe_hop switches modes (collective) the routed variant
    declines until it is enabled again, then serves the new mode exactly."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    _routed_api()
    post_moe_hop = PostMoeHop(env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False)
    assert M.all_ranks(post_moe_hop.enable_routed_input("fp32"))
    deferred_norm = DeferredNormHop(post_moe_hop)
    assert M.all_ranks(deferred_norm.enable_routed_input() and deferred_norm.routed_mode == "fp32")
    md = type("Md", (), {"num_contexts": 0, "num_generations": 6})()
    assert M.all_ranks(post_moe_hop.enable_routed_input("identity"))
    ok = not deferred_norm.can_run_routed(6, md)
    ok &= deferred_norm.enable_routed_input() and deferred_norm.routed_mode == "identity"
    ok &= deferred_norm.can_run_routed(6, md)
    with torch.inference_mode():
        src, shared, residual, gamma = _source("identity", 6, seed=33)
        h_post_moe_hop, res_post_moe_hop = _post_moe_hop_reference(
            post_moe_hop, "identity", src, 6, shared, residual, gamma
        )
        packet, res = _deferred_norm_call(
            post_moe_hop, deferred_norm, "identity", src, 6, shared, residual, gamma
        )
        ok &= _verdict(packet, res, h_post_moe_hop, res_post_moe_hop, gamma, "mode-switch")
    assert M.all_ranks(ok)


# ---------------------------------------------------------------------------
# Exactness against post_moe_hop's hop_routed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger_early", [True, False])
@pytest.mark.parametrize("with_shared", [True, False])
@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("mode", MODES)
def test_routed_packet_matches_hop_routed(mode, rows, with_shared, trigger_early):
    post_moe_hop, deferred_norm = _hops(mode, trigger_early)
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=50 + rows, with_shared=with_shared)
        h_post_moe_hop, res_post_moe_hop = _post_moe_hop_reference(
            post_moe_hop, mode, src, rows, shared, residual, gamma
        )
        cleared_post_moe_hop = _acc_clear(post_moe_hop, mode)
        packet, res_deferred_norm = _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma
        )
        cleared_deferred_norm = _acc_clear(post_moe_hop, mode)
        ok = _verdict(
            packet, res_deferred_norm, h_post_moe_hop, res_post_moe_hop, gamma, f"{mode} R={rows}"
        )
        assert M.all_ranks(ok and cleared_post_moe_hop and cleared_deferred_norm), (
            cleared_post_moe_hop,
            cleared_deferred_norm,
        )
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize("mode", MODES)
def test_routed_large_residual_and_zero_routed(mode):
    post_moe_hop, deferred_norm = _hops(mode)
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, 6, seed=77)
        residual = (residual.float() * 50.0).bfloat16()
        if mode == "identity":
            src = (src[0], torch.zeros_like(src[1]))
        else:
            src = torch.zeros_like(src)
        h_post_moe_hop, res_post_moe_hop = _post_moe_hop_reference(
            post_moe_hop, mode, src, 6, shared, residual, gamma
        )
        packet, res = _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, src, 6, shared, residual, gamma
        )
        assert _check(packet, res, h_post_moe_hop, res_post_moe_hop, gamma, f"{mode}-large-res")


@pytest.mark.parametrize("mode", MODES)
def test_back_to_back_routed_and_raw_mixed_with_post_moe_hop(mode):
    """deferred_norm routed, post_moe_hop routed, deferred_norm raw and post_moe_hop raw calls interleaved on one
    workspace without host syncs (each routed call refills the buffer on
    the stream, as routed_moe does)."""
    post_moe_hop, deferred_norm = _hops(mode)
    kinds = [
        "routed_deferred_norm",
        "routed_post_moe_hop",
        "deferred_norm",
        "post_moe_hop",
        "routed_deferred_norm",
        "routed_deferred_norm",
        "post_moe_hop",
        "deferred_norm",
        "routed_post_moe_hop",
        "routed_deferred_norm",
        "deferred_norm",
        "routed_post_moe_hop",
    ]
    rows_seq = [6, 1, 8, 3, 6, 2, 6, 8, 5, 6, 1, 6]
    with torch.inference_mode():
        inputs = []
        for i, (kind, r) in enumerate(zip(kinds, rows_seq)):
            if kind.startswith("routed_"):
                inputs.append(_source(mode, r, seed=900 + i))
            else:
                inputs.append(M.make_raw_tuple(r, seed=900 + i))
        outs = []
        for kind, r, t in zip(kinds, rows_seq, inputs):
            if kind == "routed_deferred_norm":
                outs.append(_deferred_norm_call(post_moe_hop, deferred_norm, mode, t[0], r, *t[1:]))
            elif kind == "routed_post_moe_hop":
                outs.append(_post_moe_hop_reference(post_moe_hop, mode, t[0], r, *t[1:]))
            elif kind == "deferred_norm":
                outs.append(deferred_norm(*t, M.GLM_EPS))
            else:
                outs.append(post_moe_hop(*t, M.GLM_EPS))
        torch.cuda.synchronize()
        failures = []
        for i, (kind, r, t, out) in enumerate(zip(kinds, rows_seq, inputs, outs)):
            if kind.startswith("routed_"):
                h_ref, res_ref = _post_moe_hop_reference(post_moe_hop, mode, t[0], r, *t[1:])
                gamma = t[3]
            else:
                h_ref, res_ref = post_moe_hop(*t, M.GLM_EPS)
                gamma = t[5]
            torch.cuda.synchronize()
            if kind.endswith("post_moe_hop"):
                ok = torch.equal(out[0], h_ref) and torch.equal(out[1], res_ref)
            else:
                ok = _verdict(out[0], out[1], h_ref, res_ref, gamma, f"b2b{i}-{kind}")
            if not ok:
                failures.append((i, kind))
        assert M.all_ranks(not failures and _acc_clear(post_moe_hop, mode)), failures
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize("mode", MODES)
def test_routed_cuda_graph_replay(mode):
    """Fill + deferred_norm routed, fill + post_moe_hop routed and a deferred_norm raw call in one CUDA
    graph, replayed with fresh routed content."""
    post_moe_hop, deferred_norm = _hops(mode)
    rows = 6
    with torch.inference_mode():
        static = [_source(mode, rows, seed=1200 + i) for i in range(2)]
        raw = [x.clone() if x is not None else None for x in M.make_raw_tuple(rows, seed=1250)]

        def clone_src(s):
            return tuple(x.clone() for x in s) if isinstance(s, tuple) else s.clone()

        bufs = [(clone_src(s[0]),) + tuple(s[1:]) for s in static]

        def body():
            a = _deferred_norm_call(
                post_moe_hop, deferred_norm, mode, bufs[0][0], rows, *bufs[0][1:]
            )
            b = _post_moe_hop_reference(post_moe_hop, mode, bufs[1][0], rows, *bufs[1][1:])
            c = deferred_norm(*raw, M.GLM_EPS)
            return a, b, c

        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ga, gb, gc = body()
        for trial, replays in enumerate((1, 1, 20)):
            fresh = [_source(mode, rows, seed=1300 + 10 * trial + i) for i in range(2)]
            fresh_raw = M.make_raw_tuple(rows, seed=1350 + trial)
            for dst, src in zip(bufs, fresh):
                if isinstance(dst[0], tuple):
                    for d, s in zip(dst[0], src[0]):
                        d.copy_(s)
                else:
                    dst[0].copy_(src[0])
                for d, s in zip(dst[1:], src[1:]):
                    if d is not None:
                        d.copy_(s)
            for d, s in zip(raw, fresh_raw):
                if d is not None:
                    d.copy_(s)
            for _ in range(replays):
                M.mpi_env().comm.barrier()
                graph.replay()
            torch.cuda.synchronize()
            ok = True
            h0, r0 = _post_moe_hop_reference(post_moe_hop, mode, fresh[0][0], rows, *fresh[0][1:])
            ok &= _verdict(ga[0], ga[1], h0, r0, fresh[0][3], f"graph{trial}-routed_deferred_norm")
            h1, r1 = _post_moe_hop_reference(post_moe_hop, mode, fresh[1][0], rows, *fresh[1][1:])
            torch.cuda.synchronize()
            ok &= torch.equal(gb[0], h1) and torch.equal(gb[1], r1)
            h2, r2 = post_moe_hop(*fresh_raw, M.GLM_EPS)
            ok &= _verdict(gc[0], gc[1], h2, r2, fresh_raw[5], f"graph{trial}-deferred_norm")
            assert M.all_ranks(ok and _acc_clear(post_moe_hop, mode)), f"trial {trial}"


def test_routed_delayed_rank_stress():
    """Random per-rank device delays inside back-to-back routed batches."""
    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    env = M.mpi_env()
    gen = torch.Generator().manual_seed(4243)
    with torch.inference_mode():
        for batch in range(3):
            inputs, outs = [], []
            for i in range(12):
                rows = int(torch.randint(1, 9, (1,), generator=gen))
                t = _source(mode, rows, seed=5000 + 100 * batch + i)
                local = torch.Generator().manual_seed(77 * batch + 13 * i + env.rank)
                if float(torch.rand(1, generator=local)) < 0.3:
                    torch.cuda._sleep(int(float(torch.rand(1, generator=local)) * 2_000_000))
                inputs.append((rows, t))
                outs.append(
                    _deferred_norm_call(post_moe_hop, deferred_norm, mode, t[0], rows, *t[1:])
                )
            torch.cuda.synchronize()
            ok = True
            for (rows, t), (packet, res) in zip(inputs, outs):
                h_ref, res_ref = _post_moe_hop_reference(post_moe_hop, mode, t[0], rows, *t[1:])
                ok &= _verdict(packet, res, h_ref, res_ref, t[3], f"stress{batch}")
            assert M.all_ranks(ok), f"batch {batch}"
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


# ---------------------------------------------------------------------------
# post_moe_hop's error protocol on the routed variant
# ---------------------------------------------------------------------------

_MISSING_TIMEOUT_S = 0.2


@pytest.mark.parametrize("missing", [0, 5])
@pytest.mark.parametrize("mode", ["fp32", "identity"])
def test_routed_missing_rank_times_out_with_nan_and_reset(mode, missing):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )

    post_moe_hop, deferred_norm = _hops(mode)
    imp_post_moe_hop, imp_deferred_norm = _hops(mode, timeout_s=_MISSING_TIMEOUT_S)
    env = M.mpi_env()
    assert M.all_ranks(
        imp_deferred_norm.routed_engaged and imp_post_moe_hop.workspace is post_moe_hop.workspace
    )
    rows = 6
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=600 + missing)
        before = M.lamport_snapshot(post_moe_hop)
        assert M.all_ranks(before["ok"]), before
        env.comm.barrier()
        start = time.monotonic()
        packet = res = None
        if env.rank != missing:
            packet, res = _deferred_norm_call(
                imp_post_moe_hop, imp_deferred_norm, mode, src, rows, shared, residual, gamma
            )
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
                and _acc_clear(imp_post_moe_hop, mode)
            )
        stats = env.comm.allgather((env.rank, round(elapsed, 3), bits, polls, reported))
        if env.rank == 0:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_ROUTED_MISSING mode={mode} rank {missing} skipped: (rank, elapsed_s, "
                f"error_bits, timeout_polls, ok) {stats}",
                flush=True,
            )
        assert M.all_ranks(reported), stats
        try:
            check_health(f"routed_deferred_norm-missing-rank{missing}-local")
            local_raised = False
        except PostMoeHopError:
            local_raised = True
        try:
            check_health(f"routed_deferred_norm-missing-rank{missing}-collective", collective=True)
            group_raised = False
        except PostMoeHopError:
            group_raised = True
        assert M.all_ranks(local_raised == (env.rank != missing) and group_raised)
        _reset(post_moe_hop)
        src, shared, residual, gamma = _source(mode, rows, seed=650 + missing)
        packet, res = _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma
        )
        h_ref, res_ref = _post_moe_hop_reference(
            post_moe_hop, mode, src, rows, shared, residual, gamma
        )
        assert _check(packet, res, h_ref, res_ref, gamma, f"after-reset{missing}")
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize(
    "sequence",
    [
        "routed_deferred_norm,routed_deferred_norm,routed_deferred_norm",
        "routed_deferred_norm,routed_post_moe_hop,deferred_norm",
        "routed_post_moe_hop,routed_deferred_norm,post_moe_hop",
        "deferred_norm,routed_deferred_norm,routed_post_moe_hop",
    ],
)
def test_routed_error_poisons_every_rank_until_reset(sequence):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop.kernels import (  # noqa: E501
        STATE_EPOCH,
        STATE_ERROR_WORDS,
    )

    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    env = M.mpi_env()
    victim = env.world - 3
    with torch.inference_mode():
        torch.cuda.synchronize()
        if env.rank == victim:
            epoch = int(post_moe_hop.workspace.state[STATE_EPOCH].item())
            post_moe_hop.workspace.state[STATE_ERROR_WORDS[(epoch + 2) % 3]] = 1
        torch.cuda.synchronize()
        env.comm.barrier()
        poisoned = True
        for call, kind in enumerate(sequence.split(",")):
            if kind.startswith("routed_"):
                src, shared, residual, gamma = _source(mode, 6, seed=860 + call)
                if kind == "routed_deferred_norm":
                    packet, res = _deferred_norm_call(
                        post_moe_hop, deferred_norm, mode, src, 6, shared, residual, gamma
                    )
                    poisoned &= _all_nan_packet(packet, res)
                else:
                    out, res = _post_moe_hop_reference(
                        post_moe_hop, mode, src, 6, shared, residual, gamma
                    )
                    torch.cuda.synchronize()
                    poisoned &= bool(
                        torch.isnan(out.float()).all() and torch.isnan(res.float()).all()
                    )
            else:
                t = M.make_raw_tuple(6, seed=860 + call)
                if kind == "deferred_norm":
                    packet, res = deferred_norm(*t, M.GLM_EPS)
                    poisoned &= _all_nan_packet(packet, res)
                else:
                    out, res = post_moe_hop(*t, M.GLM_EPS)
                    torch.cuda.synchronize()
                    poisoned &= bool(
                        torch.isnan(out.float()).all() and torch.isnan(res.float()).all()
                    )
        words = post_moe_hop.workspace.state.cpu().tolist()
        carried = all(words[w] == 1 for w in STATE_ERROR_WORDS)
        clean = not any(words[w] for w in STATE_ERROR_WORDS)
        per_rank = env.comm.allgather([words[w] for w in STATE_ERROR_WORDS])
        if env.rank == 0:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_ROUTED_POISON {sequence}: per-rank error words {per_rank}",
                flush=True,
            )
        assert M.all_ranks(
            poisoned
            and (carried if env.rank == victim else clean)
            and _acc_clear(post_moe_hop, mode)
        )
        with pytest.raises(PostMoeHopError):
            check_health("routed_deferred_norm-poisoned", collective=True)
        _reset(post_moe_hop)
        src, shared, residual, gamma = _source(mode, 6, seed=870)
        packet, res = _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, src, 6, shared, residual, gamma
        )
        h_ref, res_ref = _post_moe_hop_reference(
            post_moe_hop, mode, src, 6, shared, residual, gamma
        )
        assert _check(packet, res, h_ref, res_ref, gamma, "after-poison-reset")


_FAULT_DELAYS_NS = (3000, 60000)


@pytest.mark.parametrize("mode", ["fp32", "identity"])
def test_routed_fault_injected_store_order_mixed_with_post_moe_hop(mode):
    """Fault-injected deferred_norm / post_moe_hop routed builds chosen per rank and per call for
    the same collective call, plus random device delays: every output bit as
    in an undisturbed post_moe_hop routed call, one call count per call, no error."""
    post_moe_hop, deferred_norm = _hops(mode)
    hops = {
        "post_moe_hop": (post_moe_hop, post_moe_hop),
        "deferred_norm": (post_moe_hop, deferred_norm),
    }
    for d in _FAULT_DELAYS_NS:
        faulty_post_moe, faulty_deferred_norm = _hops(mode, fault_delay_ns=d)
        hops[f"post_moe_hop_fault{d}"] = (faulty_post_moe, faulty_post_moe)
        hops[f"deferred_norm_fault{d}"] = (faulty_post_moe, faulty_deferred_norm)
    env = M.mpi_env()
    names = sorted(hops)
    common = random.Random(4712)
    local = random.Random(2000 + env.rank)
    calls_before = post_moe_hop.workspace.calls()
    failed, used, total = [], {}, 0
    with torch.inference_mode():
        for batch in range(5):
            plan = [(common.randint(1, 8), common.random() < 0.75) for _ in range(12)]
            inputs = [
                _source(mode, r, seed=9100 + 100 * batch + j, with_shared=sh)
                for j, (r, sh) in enumerate(plan)
            ]
            picks = [local.choice(names) for _ in plan]
            torch.cuda.synchronize()
            env.comm.barrier()
            outs = []
            for name, (r, _), t in zip(picks, plan, inputs):
                if local.random() < 0.3:
                    torch.cuda._sleep(int(local.uniform(0.0, 50.0) * 2000))
                used[name] = used.get(name, 0) + 1
                owner, hop = hops[name]
                routed = _fill(owner, mode, t[0], r)
                outs.append(hop.hop_routed(routed, t[1], t[2], t[3], M.GLM_EPS))
            torch.cuda.synchronize()
            for j, (name, (r, _), t, out) in enumerate(zip(picks, plan, inputs, outs)):
                h_ref, res_ref = _post_moe_hop_reference(post_moe_hop, mode, t[0], r, *t[1:])
                if name.startswith("deferred_norm"):
                    ok = _verdict(out[0], out[1], h_ref, res_ref, t[3], f"fault{batch}.{j}")
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
            f"GLM_MOE_DSA_DEFERRED_NORM_ROUTED_FAULT mode={mode} calls={total} (+{total} reference calls) "
            f"used_on_rank0={used}",
            flush=True,
        )
    assert M.all_ranks(not failed and counted == 2 * total and bits == 0), (
        failed[:4],
        counted,
        bits,
    )


@pytest.mark.parametrize(
    "case", ("foreign_buffer", "row_count", "residual_dtype", "shared_shape", "gamma_shape")
)
def test_routed_contract_violation_raises_before_any_device_work(case):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopContractError,
    )

    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    rows = 6
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=920)
        routed = _fill(post_moe_hop, mode, src, rows)
        if case == "foreign_buffer":
            routed = routed.clone()
        elif case == "row_count":
            routed = post_moe_hop.routed_acc_buffer(rows + 1)
        elif case == "residual_dtype":
            residual = residual.half()
        elif case == "shared_shape":
            shared = shared[:, : M.GLM_HIDDEN // 2]
        else:
            gamma = gamma[:-8]
        calls_before = post_moe_hop.workspace.calls()
        with pytest.raises(PostMoeHopContractError):
            deferred_norm.hop_routed(routed, shared, residual, gamma, M.GLM_EPS)
        assert M.all_ranks(post_moe_hop.workspace.calls() == calls_before)
        post_moe_hop.routed_acc_buffer(post_moe_hop.max_rows).zero_()
        src, shared, residual, gamma = _source(mode, rows, seed=921)
        packet, res = _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma
        )
        h_ref, res_ref = _post_moe_hop_reference(
            post_moe_hop, mode, src, rows, shared, residual, gamma
        )
        assert _check(packet, res, h_ref, res_ref, gamma, f"after-contract-{case}")


def _misaligned(t, offset_elems):
    storage = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    view = storage[offset_elems : offset_elems + t.numel()].view(t.shape)
    view.copy_(t)
    return view


def _transposed(t):
    view = torch.empty(tuple(reversed(t.shape)), dtype=t.dtype, device=t.device).t()
    view.copy_(t)
    return view


def _needs_copy(t):
    """The hop's 16-byte vector rule for a 1-D / row-major operand."""
    return bool(t.data_ptr() % 16 or t.stride(-1) != 1 or not t.is_contiguous())


def _relayout(rank, shared, residual, gamma):
    """Rank-dependent layouts; returns them and the repairs expected on this
    rank (a one-row "transposed" tensor is still contiguous)."""
    kind = rank % 4
    if kind == 0:
        gamma = _misaligned(gamma, 2)
    elif kind == 1:
        residual = _misaligned(residual, 4)
    elif kind == 2 and shared is not None:
        shared = _transposed(shared)
    repairs = sum(int(_needs_copy(t)) for t in (shared, residual, gamma) if t is not None)
    return shared, residual, gamma, repairs


def test_routed_unaligned_layouts_are_repaired_locally():
    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    env = M.mpi_env()
    ok = True
    with torch.inference_mode():
        for i, rows in enumerate((6, 1, 8)):
            src, shared, residual, gamma = _source(mode, rows, seed=880 + i)
            r_shared, r_residual, r_gamma, expected = _relayout(env.rank, shared, residual, gamma)
            before = post_moe_hop.workspace.layout_repairs
            packet, res = _deferred_norm_call(
                post_moe_hop, deferred_norm, mode, src, rows, r_shared, r_residual, r_gamma
            )
            repaired = post_moe_hop.workspace.layout_repairs - before
            h_ref, res_ref = _post_moe_hop_reference(
                post_moe_hop, mode, src, rows, shared, residual, gamma
            )
            ok &= _verdict(packet, res, h_ref, res_ref, gamma, f"relayout{rows}")
            ok &= repaired == expected
        rows = 6
        src, shared, residual, gamma = _source(mode, rows, seed=890)
        s_src = src.clone()
        s_shared, s_residual, s_gamma, _ = _relayout(env.rank, shared, residual, gamma)
        _deferred_norm_call(
            post_moe_hop, deferred_norm, mode, s_src, rows, s_shared, s_residual, s_gamma
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_packet, g_res = _deferred_norm_call(
                post_moe_hop, deferred_norm, mode, s_src, rows, s_shared, s_residual, s_gamma
            )
        torch.cuda.synchronize()
        env.comm.barrier()
        for replay in range(4):
            f_src, f_shared, f_residual, _ = _source(mode, rows, seed=900 + replay)
            s_src.copy_(f_src)
            s_shared.copy_(f_shared)
            s_residual.copy_(f_residual)
            graph.replay()
            torch.cuda.synchronize()
            h_ref, res_ref = _post_moe_hop_reference(
                post_moe_hop, mode, f_src, rows, f_shared, f_residual, gamma
            )
            ok &= _verdict(g_packet, g_res, h_ref, res_ref, gamma, f"relayout-graph{replay}")
    assert M.all_ranks(ok and post_moe_hop.error_bits() == 0)


def test_health_check_counts_deferred_norm_routed_calls():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        check_health,
    )

    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    with torch.inference_mode():
        before = check_health("routed_deferred_norm-engagement-before", collective=True)
        for i in range(5):
            src, shared, residual, gamma = _source(mode, 6, seed=940 + i)
            _deferred_norm_call(post_moe_hop, deferred_norm, mode, src, 6, shared, residual, gamma)
        after = check_health("routed_deferred_norm-engagement-after", collective=True)
    assert M.all_ranks(
        after.calls - before.calls == 5
        and after.error_bits == 0
        and not after.group_error
        and after.calls == post_moe_hop.workspace.calls()
    )


def test_routed_memcheck_positive_control():
    """Tests only, run under compute-sanitizer memcheck by the sanitizer
    suite (``GLM_MOE_DSA_DEFERRED_NORM_MEMCHECK_CONTROL=1``): the routed variant reads (and, in
    fp32 mode, clears) an exactly sized accumulator with twice its row
    pitch, so row 1 is out of bounds and memcheck must report it."""
    if os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_MEMCHECK_CONTROL") != "1":
        pytest.skip("memcheck positive control (sanitizer suite only)")
    mode = "fp32"
    post_moe_hop, deferred_norm = _hops(mode)
    rows = 2
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=990)
        acc = src.clone()
        deferred_norm.launch_routed(
            acc,
            shared,
            residual,
            gamma,
            M.GLM_EPS,
            row_bytes=2 * acc.stride(0) * acc.element_size(),
        )
        torch.cuda.synchronize()
    _reset(post_moe_hop)
