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
"""Kimi K3.1 column-parallel MoE tail (``moe_cp_tail``) over its TP group (TP2, TP4).

The tail's call runs on the workspace of the runtime builder the TEP8 target calls in
``post_load_weights`` (``rubin_ull_tail_workspace``), with the rank's columns and tactics of
``rubin_ull_cp_tail_share``. Every rank pushes its shared experts' partial into the call's slots
(``rubin_lamport_push``) and its columns of latent_up into the call's all-gather rows
(``rubin_ull_cp_tail``). The pre-attention mix finishes the call (``finish_ag``, the
round-to-nearest add, ``early_poll`` off and on), or the finish kernel does and a torch add
follows. Each result must be bitwise the replicated tail's: the same split-K GEMM's plain
products over every rank's columns (``run_gemm``, the values the push stores), the C++ MNNVL
all-reduce of the shared partials, and the MoE-residual mix (``rubin_attn_res_moe_add_rmsnorm``)
or ``bf16(prefix + bf16(latent_up + shared))``.

* Kernels (the entries): 1 to 32 rows, two and nine candidates, finished by the mix (with and
  without early poll) and the finish kernel, a C++ all-reduce on its own
  workspace between calls; then eleven calls of mixed row counts on a new workspace, so that its
  three buffers are reused after finishes of each kind re-armed them (the all-gather rows in
  place, and under the slots of a larger call); then four calls captured in one CUDA graph and
  replayed on new inputs (four calls per replay move the buffers on by one).
* Target: the tep8 target's own tail and slots (``_moe_cp_tail``, the latent all-reduce + norm
  and the push; ``moe_norm`` with ``moe_tail_early_poll`` off and on; ``moe_add``) on a stand-in
  expert partial, against the replicated data path with the C++ latent all-reduce + RMSNorm,
  eagerly and as two layers under CUDA graph replay with the shared push on the side stream.

Which steps take the tail (TEP8 decode steps of up to 32 rows) is checked on the CPU.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _SM107, reason="certified on SM107 only"),
]

HIDDEN = 7168
LATENT = 3584
EPS = 1e-6
#: The finishes of a tail's call: the mix, early poll off or on, or the finish kernel and the add.
_FINISHES = ("mix", "mix_early", "kernel")
#: Calls on a new workspace: (rows, finish). Call j runs on buffer j % 3, and its finish re-arms
#: the buffer of call j - 1 (slots and all-gather rows). Reused with the same rows (the
#: all-gather rows in place): 0 -> 3, 4 -> 7 and 6 -> 9 after the finish kernel re-armed the
#: buffer, 7 -> 10 after the mix with early poll did; with more rows (the slots over the earlier
#: all-gather rows): 2 -> 5 after the mix with early poll, 3 -> 6 after the mix.
_REUSE = (
    (8, "mix"),
    (32, "kernel"),
    (1, "mix_early"),
    (8, "mix_early"),
    (16, "mix"),
    (32, "kernel"),
    (16, "mix_early"),
    (16, "kernel"),
    (8, "mix_early"),
    (16, "mix"),
    (16, "mix"),
)


def _bf16(shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def _mix_weights(seed):
    """The mix's score projection, its norm and the output norm (``_MixW``'s fields)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    res_w = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    rms_w = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
    out_w = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
    return res_w, rms_w, out_w


def _differ(want, got):
    """Bytes that differ between two tensors or two tuples of them; a shape or dtype mismatch
    counts every byte (no exception: the other ranks may still be polling)."""
    if isinstance(want, torch.Tensor):
        want, got = (want,), (got,)
    if len(want) != len(got):
        return 1 << 30
    bad = 0
    for a, b in zip(want, got):
        if a.shape != b.shape or a.dtype != b.dtype:
            bad += a.numel() * a.element_size()
            continue
        bad += int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())
    return bad


def _setup(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == world_size
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=gpus)
    return comm, rank, mapping


def _plain_products(latent, up, world):
    """The values every rank's push stores into the all-gather rows: each rank's columns of
    ``latent @ up^T`` by the store-only variant of the push GEMM at that rank's tactic, ``-0``
    sent as ``+0``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_push_gemm as push_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
        rubin_draft_gemm_push as draft_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        _rubin_ull_common as ull_common,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        rubin_ull_cp_tail as cp_ops,
    )

    m = latent.shape[0]
    y = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=latent.device)
    for r in range(world):
        c0, n, tactics = cp_ops.rubin_ull_cp_tail_share(world, r)
        plain = torch.empty(m, n, dtype=torch.bfloat16, device=latent.device)
        draft_gemm.run_gemm(latent, up[c0 : c0 + n], plain, push_gemm.tactic_of(tactics[m]))
        y[:, c0 : c0 + n] = ull_common.sanitize_bf16(plain)
    return y


def _run_rank_kernels(world_size):
    """The tail's entries against the replicated tail, eagerly, over many calls and in a graph."""
    comm, rank, mapping = _setup(world_size)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_lamport_push as push_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        rubin_ull_cp_tail as cp_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
        rubin_attn_res_lamport_add_rmsnorm as mix_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
        rubin_attn_res_moe_add_rmsnorm as moe_mix_ops,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    try:
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        c0, n, tactics = cp_ops.rubin_ull_cp_tail_share(world_size, rank)
        spans = comm.allgather((c0, n))
        assert spans[0][0] == 0 and spans[-1][0] + spans[-1][1] == HIDDEN
        assert all(a[0] + a[1] == b[0] for a, b in zip(spans, spans[1:]))
        gen = torch.Generator(device="cuda").manual_seed(3)
        up = (torch.randn(HIDDEN, LATENT, device="cuda", generator=gen) * LATENT**-0.5).bfloat16()
        w_rows = up[c0 : c0 + n]
        weights = _mix_weights(5)
        bad = {"eager": 0, "reuse": 0, "graph": 0}
        checked = {"eager": 0, "reuse": 0, "graph": 0}

        def call(built, latent, part):
            """One call: this rank's shared partial into the slots (the shared experts' push),
            its columns of latent_up into the all-gather rows (the tail)."""
            push_ops.rubin_lamport_push(built["ws"], part, True, True)
            cp_ops.rubin_ull_cp_tail(latent, w_rows, built["ws"], tactics[latent.shape[0]], c0=c0)

        def mix_args(built, how, prefix, block):
            """The arguments of the target's ``moe_norm`` mix on a tail's call."""
            ws = built["ws"]
            return (
                prefix, ws.uc_local, ws.flags.data_ptr(), block, *weights, EPS, EPS, ws.world,
                built["chunk"], False, ws.mc_base, ws.rank, True, how.endswith("early"), False,
                True,
            )  # fmt: skip

        def finish(built, how, prefix, block):
            """The call finished as the target's slots finish it (``moe_norm`` / ``moe_add``)."""
            if how == "kernel":
                out = push_ops.rubin_lamport_finish(
                    built["ws"], torch.empty_like(prefix), True, True, False,
                    built["cpp_one_shot"], True,
                )  # fmt: skip
                return prefix + out, out
            return mix_ops.rubin_attn_res_lamport_add_rmsnorm(
                *mix_args(built, how, prefix, block), early_trigger=True
            )

        def reference(how, prefix, block, y, shared):
            if how == "kernel":
                fold = y + shared
                return prefix + fold, fold
            return moe_mix_ops.rubin_attn_res_moe_add_rmsnorm(
                prefix, y, shared, block, *weights, EPS, EPS, early_trigger=True
            )

        def case(built, stage, how, m, n_cand, seed):
            """A call of ``m`` rows on ``built``, finished by ``how``, against the reference."""
            latent = _bf16((m, LATENT), seed, 0.5)  # the all-reduced latent: the same everywhere
            part = _bf16((m, HIDDEN), seed + 1 + rank, 0.3)
            prefix = _bf16((m, HIDDEN), seed + 50)
            block = _bf16((n_cand - 1, m, HIDDEN), seed + 51)
            cp_ops.rubin_ull_cp_tail.is_valid(latent, w_rows, built["ws"], tactics[m], c0=c0)
            if how != "kernel":
                mix_ops.rubin_attn_res_lamport_add_rmsnorm.is_valid(
                    *mix_args(built, how, prefix, block), early_trigger=True
                )
            call(built, latent, part)
            got = finish(built, how, prefix, block)
            shared = allreduce(part)  # the C++ all-reduce, on its own workspace between calls
            want = reference(how, prefix, block, _plain_products(latent, up, world_size), shared)
            bad[stage] += _differ(want, got)
            checked[stage] += 1

        # The workspace the target builds in post_load_weights, for up to 32 rows.
        built = cp_ops.rubin_ull_tail_workspace(mapping, world_size)
        assert (built["cpp_one_shot"], built["chunk"]) == (True, world_size)
        for m in (1, 8, 14, 16, 32):
            for n_cand in (2, 9):
                for i, how in enumerate(_FINISHES):
                    case(built, "eager", how, m, n_cand, 1000 * m + 100 * n_cand + 10 * i)

        # Buffer reuse after the all-gather re-arm, on a new workspace: call j on buffer j % 3.
        fresh = cp_ops.rubin_ull_tail_workspace(mapping, world_size)
        for j, (m, how) in enumerate(_REUSE):
            case(fresh, "reuse", how, m, 5, 50000 + 100 * j)

        # CUDA graph: four calls back to back on one workspace, each finished differently and
        # feeding the next one's prefix, replayed on new inputs.
        m, n_cand = 8, 5
        hows = ("mix", "mix_early", "kernel", "mix_early")
        lat = [_bf16((m, LATENT), 60000 + i, 0.5) for i in range(4)]
        parts = [_bf16((m, HIDDEN), 61000 + i + 10 * rank, 0.3) for i in range(4)]
        prefix = _bf16((m, HIDDEN), 62000)
        block = _bf16((n_cand - 1, m, HIDDEN), 62001)

        def calls():
            outs, pre = [], prefix
            for i, how in enumerate(hows):
                call(built, lat[i], parts[i])
                outs.append(finish(built, how, pre, block))
                pre = outs[-1][0]
            return outs

        calls()  # every kernel compiled before the capture
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                lat[i].copy_(_bf16((m, LATENT), 63000 + 10 * rep + i, 0.5))
                parts[i].copy_(_bf16((m, HIDDEN), 64000 + 10 * rep + i + 100 * rank, 0.3))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            pre = prefix
            for i, how in enumerate(hows):
                y = _plain_products(lat[i], up, world_size)
                want = reference(how, pre, block, y, allreduce(parts[i]))
                bad["graph"] += _differ(want, results[i])
                checked["graph"] += 1
                pre = want[0]
        del graph
        torch.cuda.synchronize()
        bad = {k: comm.allreduce(v) for k, v in bad.items()}
    except Exception:
        traceback.print_exc()
        raise
    return bad, checked


def _run_rank_target(world_size):
    """The tep8 target's tail (``_moe_cp_tail``) and the slots that finish it, on its builders."""
    comm, rank, mapping = _setup(world_size)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm import (
        flashinfer_rmsnorm as rmsnorm_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8 import (  # noqa: E501
        modeling as tgt,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        allreduce_module as ar_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_comm as comm_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_lamport_push as push_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        rubin_ull_cp_tail as cp_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
        rubin_attn_res_moe_add_rmsnorm as moe_mix_ops,
    )
    from tensorrt_llm._torch.distributed import AllReduceStrategy

    try:
        c = tgt._CONSTANTS["generation"]
        eps = tgt._MOE_EPS
        # The tail's runtime resources as the target's post_load_weights builds them, for this
        # test's TP group.
        all_reduce = comm_ops.rubin_all_reduce(mapping, AllReduceStrategy.MNNVL, world_size)
        latent_ws = comm_ops.rubin_latent_ar_workspace(
            mapping,
            world_size,
            LATENT,
            c["latent_ar_own_ws_max_m"],
            c["latent_ar_oneshot_max_bytes"],
        )
        built = cp_ops.rubin_ull_tail_workspace(mapping, world_size, tgt._ULL_TAIL_MAX_ROWS)
        c0, n, tactics = cp_ops.rubin_ull_cp_tail_share(world_size, rank, tgt._ULL_TAIL_MAX_ROWS)
        moe_comm = tgt._MoeComm(
            all_reduce=all_reduce,
            dense_all_reduce=None,
            latent_ws=latent_ws,
            latent_stream=None,
            prezero_stream=None,
            memset_stream=None,
            **{f: torch.cuda.Event() for f in tgt._MoeComm._fields[6:]},
        )
        ull_tail = tgt._UllTail(
            ws=built["ws"],
            handle=built["handle"],
            c0=c0,
            n=n,
            tactics=tactics,
            cpp_one_shot=built["cpp_one_shot"],
            chunk=built["chunk"],
        )
        rt = tgt._Rt(
            **{
                **dict.fromkeys(tgt._Rt._fields),
                "aux_stream": torch.cuda.Stream(),
                "moe_comm": moe_comm,
                "ull_tail": ull_tail,
            }
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        up = (torch.randn(HIDDEN, LATENT, device="cuda", generator=gen) * LATENT**-0.5).bfloat16()
        norm_w = (torch.rand(LATENT, device="cuda", generator=gen) + 0.5).bfloat16()
        L = tgt._MoeW(
            **{**dict.fromkeys(tgt._MoeW._fields), "latent_norm": norm_w, "latent_up": up}
        )
        res_w, rms_w, out_w = _mix_weights(9)
        mw = tgt._MixW(proj=res_w, res_norm=rms_w, out_norm=out_w)
        slots = {poll: tgt._mix_slots({**c, "moe_tail_early_poll": poll}, eps) for poll in (0, 1)}
        bad = {"eager_norm": 0, "eager_add": 0, "eager_norm_no_bank": 0, "graph": 0}
        tails = 0

        def tail(routed, part):
            """A decode step's tail after its experts, on the current stream: the shared
            experts' partial pushed into the call, then the target's ``_moe_cp_tail``."""
            push_ops.rubin_lamport_push(rt.ull_tail.ws, part, True, True)
            return tgt._moe_cp_tail(L, routed, rt)

        def replicated(routed, part):
            """``(latent_up output, shared)`` of the replicated data path: the C++ latent
            all-reduce + RMSNorm, the plain products, the shared experts' C++ all-reduce."""
            rms_norm = tgt._MOE_AR_RMS_NORM
            latent = ar_ops.allreduce_module(routed, all_reduce, rms_norm, norm_w, eps)
            shared = ar_ops.allreduce_module(part, all_reduce, tgt._MOE_AR_NONE, None, None)
            return _plain_products(latent, up, world_size), shared

        def moe_mix(prefix, y, shared, bank):
            return moe_mix_ops.rubin_attn_res_moe_add_rmsnorm(
                prefix, y, shared, bank, res_w, rms_w, out_w, eps, eps, early_trigger=True
            )

        for m in (1, 8, 16, 32):
            routed = _bf16((m, LATENT), 1000 * m + rank, 0.5)  # this rank's experts' partial
            part = _bf16((m, HIDDEN), 1000 * m + 100 + rank, 0.3)  # its shared experts' one
            prefix = _bf16((m, HIDDEN), 1000 * m + 200)
            bank = _bf16((3, m, HIDDEN), 1000 * m + 201)
            y, shared = replicated(routed, part)
            want = moe_mix(prefix, y, shared, bank)
            for poll in (0, 1):
                out = tail(routed, part)
                tails += int(isinstance(out, tgt._MoeTail) and out.ws is ull_tail.ws)
                got = slots[poll]["moe_norm"](prefix, out, bank, mw)
                bad["eager_norm"] += _differ(want, got)
            # The finish kernel, then the add (no snapshot in the bank, or above the PDL rows).
            fold = prefix + (y + shared)
            bad["eager_add"] += _differ(fold, slots[1]["moe_add"](prefix, tail(routed, part)))
            got = slots[1]["moe_norm"](prefix, tail(routed, part), bank[:0], mw)
            want = (fold, rmsnorm_ops.flashinfer_rmsnorm(fold, out_w, eps))
            bad["eager_norm_no_bank"] += _differ(want, got)

        # CUDA graph: two MoE layers, each finished by the next layer's mix, in a decode step's
        # graph plan (the shared experts' push on the side stream beside the tail), replayed on
        # new inputs.
        m = 8
        routed = [_bf16((m, LATENT), 50000 + i + 10 * rank, 0.5) for i in range(2)]
        parts = [_bf16((m, HIDDEN), 51000 + i + 10 * rank, 0.3) for i in range(2)]
        prefix = _bf16((m, HIDDEN), 52000)
        bank = _bf16((3, m, HIDDEN), 52001)

        def layer(i, pre):
            moe_comm.shared_fork.record()
            with torch.cuda.stream(rt.aux_stream):
                moe_comm.shared_fork.wait()
                push_ops.rubin_lamport_push(rt.ull_tail.ws, parts[i], True, True)
                moe_comm.shared_join.record()
            out = tgt._moe_cp_tail(L, routed[i], rt)
            moe_comm.shared_join.wait()
            return slots[1]["moe_norm"](pre, out, bank, mw)

        def two_layers():
            first = layer(0, prefix)
            return first, layer(1, first[0])

        for _ in range(2):  # every kernel compiled before the capture
            two_layers()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = two_layers()
        for step in range(3):
            for i in range(2):
                routed[i].copy_(_bf16((m, LATENT), 60000 + 10 * step + i + 100 * rank, 0.5))
                parts[i].copy_(_bf16((m, HIDDEN), 61000 + 10 * step + i + 100 * rank, 0.3))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            first = moe_mix(prefix, *replicated(routed[0], parts[0]), bank)
            second = moe_mix(first[0], *replicated(routed[1], parts[1]), bank)
            bad["graph"] += _differ(first, graph_out[0]) + _differ(second, graph_out[1])
        del graph
        torch.cuda.synchronize()
        bad = {k: comm.allreduce(v) for k, v in bad.items()}
    except Exception:
        traceback.print_exc()
        raise
    return bad, tails


def _map(mpi_pool_executor, fn):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    return list(mpi_pool_executor.map(fn, *zip(*[(world_size,)] * world_size)))


@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_cp_tail_kernels(mpi_pool_executor):
    """The tail's entries: eagerly, over buffer reuse after the re-arm, in a CUDA graph."""
    for bad, checked in _map(mpi_pool_executor, _run_rank_kernels):
        assert checked == {"eager": 30, "reuse": len(_REUSE), "graph": 12}, checked
        assert not any(bad.values()), f"bytes that differ: {bad}"


@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_cp_tail_target(mpi_pool_executor):
    """The tep8 target's tail and slots against the replicated data path."""
    for bad, tails in _map(mpi_pool_executor, _run_rank_target):
        assert tails == 8, f"{tails} of 8 eager tails returned the call on the tail's workspace"
        assert not any(bad.values()), f"bytes that differ: {bad}"
