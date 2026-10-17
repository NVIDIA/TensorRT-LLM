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
"""Collective tests of the deferred_norm_evict_hints hop against F's deferred_norm hop (8 ranks, 2 hosts).

For every deferred_norm_evict_hints hop configuration (``CONFIGS``: four experiment arms, the
defaults ``carveout=-1,poll_prefetch=none`` through TVM-FFI, and the
co-resident executor hop ``carveout=100,poll_prefetch=none,ffi=0``), on the
same post_moe_hop workspace and the same inputs as deferred_norm's hop (routed_moe_into_hop routed
input in each mode, and the raw tuple), every test below runs per configuration:

* ``residual_out``, ``a_frag``, ``sumsq`` and ``materialize()`` are bit-identical
  to deferred_norm's (and so to post_moe_hop's ``hop_routed`` / ``__call__``, which deferred_norm's
  own suite checks); the fp32 accumulator is zero after the call;
* deferred_norm_evict_hints, deferred_norm and post_moe_hop calls interleave on one workspace
  without host syncs;
* a CUDA graph of deferred_norm_evict_hints / deferred_norm / post_moe_hop calls replays with fresh
  inputs;
* random per-rank device delays and fault-injected store order;
* a missing rank times out with NaN, error bits and the post_moe_hop health check,
  and a collective reset restores exact results.
"""

import os
import sys
import time

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "post_moe_hop"))
sys.path.insert(0, os.path.join(_HERE, "..", "deferred_norm"))

import post_moe_hop_mpi_utils as M  # noqa: E402
from test_deferred_norm_routed_mpi import _fill, _source  # noqa: E402

CONFIGS = os.environ.get(
    "GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_TEST_HOP_CFGS",
    "carveout=100,poll_prefetch=wait3,ffi=0;"
    "carveout=100,poll_prefetch=cur,ffi=0;carveout=100,poll_prefetch=pre3,ffi=0;"
    "carveout=100,poll_prefetch=wait3,clear_hint=evict_last,ffi=0;"
    "carveout=-1,poll_prefetch=none;carveout=100,poll_prefetch=none,ffi=0",
).split(";")
MODES = ("fp32", "identity", "row")
_HOPS = {}


def _env():
    env = M.mpi_env()
    if env.world != 8:
        pytest.skip("the hop is specialized to TP8")
    return env


def _hops(mode, cfg, timeout_s=None, fault_delay_ns=0):
    """post_moe_hop (routed input ``mode``), deferred_norm hop and deferred_norm_evict_hints hop
    ``cfg`` on one workspace."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints import (  # noqa: E501
        EvictHintsDeferredHop,
        EvictHintsHopConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        parse_cfg,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
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
        deferred_norm = DeferredNormHop(post_moe_hop, trigger_early=True)
        assert M.all_ranks(deferred_norm.engaged and deferred_norm.enable_routed_input())
        _HOPS[key] = {"post_moe_hop": post_moe_hop, "deferred_norm": deferred_norm}
    entry = _HOPS[key]
    if cfg not in entry:
        hop_config = EvictHintsDeferredHop(
            entry["post_moe_hop"], trigger_early=True, config=parse_cfg(cfg, EvictHintsHopConfig)
        )
        assert M.all_ranks(hop_config.engaged and hop_config.enable_routed_input()), (
            hop_config.reason,
            hop_config.routed_reason,
        )
        entry[cfg] = hop_config
    return entry["post_moe_hop"], entry["deferred_norm"], entry[cfg]


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _outs(packet, res):
    """Residual, the R written rows of ``a_frag``, ``sumsq`` and the materialization."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        materialize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    h = materialize(packet)
    torch.cuda.synchronize()
    a = rows_from_frag(packet.a_frag, res.shape[0], res.shape[1])
    return [res.clone(), a.clone(), packet.sumsq.clone(), h.clone()]


def _same(a, b, tag):
    same = [torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b)]
    if not all(same):
        print(
            f"GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_HOP_FAIL {tag} rank={M.mpi_env().rank} "
            f"residual/a_frag/sumsq/h={same}",
            flush=True,
        )
    return all(same)


def _acc_clear(post_moe_hop, mode):
    if mode != "fp32":
        return True
    torch.cuda.synchronize()
    return bool((post_moe_hop.routed_acc_buffer(post_moe_hop.max_rows) == 0).all())


def _routed(post_moe_hop, hop, mode, src, rows, shared, residual, gamma):
    routed = _fill(post_moe_hop, mode, src, rows)
    return hop.hop_routed(routed, shared, residual, gamma, M.GLM_EPS)


@pytest.mark.parametrize("with_shared", [True, False])
@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", CONFIGS)
def test_routed_matches_deferred_norm(cfg, mode, rows, with_shared):
    post_moe_hop, deferred_norm, hop_config = _hops(mode, cfg)
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=50 + rows, with_shared=with_shared)
        ref = _outs(*_routed(post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma))
        got = _outs(*_routed(post_moe_hop, hop_config, mode, src, rows, shared, residual, gamma))
        ok = _same(got, ref, f"{cfg} {mode} R={rows}") and _acc_clear(post_moe_hop, mode)
        assert M.all_ranks(ok)
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_raw_tuple_matches_deferred_norm(cfg, rows):
    post_moe_hop, deferred_norm, hop_config = _hops("fp32", cfg)
    with torch.inference_mode():
        t = M.make_raw_tuple(rows, seed=300 + rows)
        ref = _outs(*deferred_norm(*t, M.GLM_EPS))
        got = _outs(*hop_config(*t, M.GLM_EPS))
        assert M.all_ranks(_same(got, ref, f"{cfg} raw R={rows}"))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_interleaved_with_deferred_norm_and_post_moe_hop(cfg):
    """deferred_norm_evict_hints / deferred_norm / post_moe_hop calls (routed and raw) back to back on one workspace."""
    mode = "fp32"
    post_moe_hop, deferred_norm, hop_config = _hops(mode, cfg)
    kinds = [
        "routed_deferred_norm_evict_hints",
        "routed_deferred_norm",
        "routed_post_moe_hop",
        "deferred_norm_evict_hints",
        "post_moe_hop",
        "routed_deferred_norm_evict_hints",
        "routed_deferred_norm_evict_hints",
        "deferred_norm",
        "routed_post_moe_hop",
        "routed_deferred_norm_evict_hints",
        "deferred_norm_evict_hints",
        "routed_deferred_norm",
    ]
    rows_seq = [6, 1, 8, 3, 6, 2, 6, 8, 5, 6, 1, 6]
    with torch.inference_mode():
        inputs = [
            (
                _source(mode, r, seed=900 + i)
                if k.startswith("routed_")
                else M.make_raw_tuple(r, seed=900 + i)
            )
            for i, (k, r) in enumerate(zip(kinds, rows_seq))
        ]
        outs = []
        for kind, r, t in zip(kinds, rows_seq, inputs):
            if kind.startswith("routed_"):
                hop = {
                    "routed_deferred_norm_evict_hints": hop_config,
                    "routed_deferred_norm": deferred_norm,
                }.get(kind)
                if hop is None:
                    outs.append(
                        post_moe_hop.hop_routed(
                            _fill(post_moe_hop, mode, t[0], r), *t[1:], M.GLM_EPS
                        )
                    )
                else:
                    outs.append(_routed(post_moe_hop, hop, mode, t[0], r, *t[1:]))
            else:
                hop = {
                    "deferred_norm_evict_hints": hop_config,
                    "deferred_norm": deferred_norm,
                    "post_moe_hop": post_moe_hop,
                }[kind]
                outs.append(hop(*t, M.GLM_EPS))
        torch.cuda.synchronize()
        snap = [
            [x.clone() for x in o] if kind.endswith("post_moe_hop") else _outs(*o)
            for kind, o in zip(kinds, outs)
        ]
        failures = []
        for i, (kind, r, t, got) in enumerate(zip(kinds, rows_seq, inputs, snap)):
            if kind.endswith("post_moe_hop"):
                if kind.startswith("routed_"):
                    ref = post_moe_hop.hop_routed(
                        _fill(post_moe_hop, mode, t[0], r), *t[1:], M.GLM_EPS
                    )
                else:
                    ref = post_moe_hop(*t, M.GLM_EPS)
                torch.cuda.synchronize()
                ok = torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
            else:
                if kind.startswith("routed_"):
                    ref = _outs(*_routed(post_moe_hop, deferred_norm, mode, t[0], r, *t[1:]))
                else:
                    ref = _outs(*deferred_norm(*t, M.GLM_EPS))
                ok = _same(got, ref, f"interleaved{i}-{kind}")
            if not ok:
                failures.append((i, kind))
        assert M.all_ranks(not failures and _acc_clear(post_moe_hop, mode)), failures
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_cuda_graph_replay(cfg):
    """Fill + deferred_norm_evict_hints routed, fill + deferred_norm routed, a
    deferred_norm_evict_hints raw call in one graph, replayed."""
    mode = "fp32"
    post_moe_hop, deferred_norm, hop_config = _hops(mode, cfg)
    rows = 6
    with torch.inference_mode():
        bufs = [list(_source(mode, rows, seed=1200 + i)) for i in range(2)]
        raw = [x.clone() if x is not None else None for x in M.make_raw_tuple(rows, seed=1250)]

        def body():
            a = _routed(post_moe_hop, hop_config, mode, bufs[0][0], rows, *bufs[0][1:])
            b = _routed(post_moe_hop, deferred_norm, mode, bufs[1][0], rows, *bufs[1][1:])
            c = hop_config(*raw, M.GLM_EPS)
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
                for d, s in zip(dst, src):
                    if d is not None:
                        d.copy_(s)
            for d, s in zip(raw, fresh_raw):
                if d is not None:
                    d.copy_(s)
            for _ in range(replays):
                M.mpi_env().comm.barrier()
                graph.replay()
            torch.cuda.synchronize()
            got = [_outs(*ga), _outs(*gb), _outs(*gc)]
            refs = [
                _outs(
                    *_routed(post_moe_hop, deferred_norm, mode, fresh[0][0], rows, *fresh[0][1:])
                ),
                _outs(
                    *_routed(post_moe_hop, deferred_norm, mode, fresh[1][0], rows, *fresh[1][1:])
                ),
                _outs(*deferred_norm(*fresh_raw, M.GLM_EPS)),
            ]
            ok = all(_same(g, r, f"graph{trial}-{j}") for j, (g, r) in enumerate(zip(got, refs)))
            assert M.all_ranks(ok and _acc_clear(post_moe_hop, mode)), f"trial {trial}"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_delayed_rank_stress(cfg):
    """Random per-rank device delays inside back-to-back deferred_norm_evict_hints routed batches."""
    mode = "fp32"
    post_moe_hop, deferred_norm, hop_config = _hops(mode, cfg)
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
                outs.append(_routed(post_moe_hop, hop_config, mode, t[0], rows, *t[1:]))
            torch.cuda.synchronize()
            snap = [_outs(*o) for o in outs]
            ok = True
            for (rows, t), got in zip(inputs, snap):
                ok &= _same(
                    got,
                    _outs(*_routed(post_moe_hop, deferred_norm, mode, t[0], rows, *t[1:])),
                    f"stress{batch}",
                )
            assert M.all_ranks(ok), f"batch {batch}"
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_fault_injected_store_order(cfg):
    """Scrambled store order (post_moe_hop's fault injection) in the deferred_norm_evict_hints kernel."""
    mode = "fp32"
    post_moe_hop, deferred_norm, hop_config = _hops(mode, cfg, fault_delay_ns=4000)
    with torch.inference_mode():
        inputs = [(r, _source(mode, r, seed=7000 + i)) for i, r in enumerate([6, 1, 8, 6, 3, 6])]
        outs = [_routed(post_moe_hop, hop_config, mode, t[0], r, *t[1:]) for r, t in inputs]
        torch.cuda.synchronize()
        snap = [_outs(*o) for o in outs]
        ok = True
        for (r, t), got in zip(inputs, snap):
            ok &= _same(
                got, _outs(*_routed(post_moe_hop, deferred_norm, mode, t[0], r, *t[1:])), "fault"
            )
        assert M.all_ranks(ok)
        assert M.all_ranks(post_moe_hop.error_bits() == 0)


_MISSING_TIMEOUT_S = 0.2


@pytest.mark.parametrize("missing", [0, 5])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_missing_rank_times_out_with_nan_and_reset(cfg, missing):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHopError,
        check_health,
    )

    mode = "fp32"
    post_moe_hop, deferred_norm, _ = _hops(mode, cfg)
    imp_post_moe_hop, _, imp_evict_hints = _hops(mode, cfg, timeout_s=_MISSING_TIMEOUT_S)
    env = M.mpi_env()
    assert M.all_ranks(imp_post_moe_hop.workspace is post_moe_hop.workspace)
    rows = 6
    with torch.inference_mode():
        src, shared, residual, gamma = _source(mode, rows, seed=600 + missing)
        env.comm.barrier()
        start = time.monotonic()
        snap = None
        if env.rank != missing:
            snap = _outs(
                *_routed(
                    imp_post_moe_hop, imp_evict_hints, mode, src, rows, shared, residual, gamma
                )
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        counters = post_moe_hop.workspace.counters()
        bits, polls = counters["error_bits"], counters["timeout_polls"]
        if env.rank == missing:
            reported = bits == 0 and polls == 0
        else:
            nan = all(bool(torch.isnan(t.float()).all()) for t in (snap[0], snap[2], snap[3]))
            reported = (
                bool(bits & 1)
                and polls > 0
                and _MISSING_TIMEOUT_S <= elapsed < 30.0
                and nan
                and _acc_clear(imp_post_moe_hop, mode)
            )
        stats = env.comm.allgather((env.rank, round(elapsed, 3), bits, polls, reported))
        if env.rank == 0:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_MISSING cfg={cfg} rank {missing} skipped: {stats}",
                flush=True,
            )
        assert M.all_ranks(reported), stats
        try:
            check_health(f"deferred_norm_evict_hints-missing-rank{missing}", collective=True)
            raised = False
        except PostMoeHopError:
            raised = True
        assert M.all_ranks(raised)
        assert M.all_ranks(post_moe_hop.reset())
        assert M.all_ranks(post_moe_hop.error_bits() == 0)
        src, shared, residual, gamma = _source(mode, rows, seed=650 + missing)
        _, _, hop_config = _hops(mode, cfg)
        got = _outs(*_routed(post_moe_hop, hop_config, mode, src, rows, shared, residual, gamma))
        ref = _outs(*_routed(post_moe_hop, deferred_norm, mode, src, rows, shared, residual, gamma))
        assert M.all_ranks(_same(got, ref, f"after-reset{missing}"))
