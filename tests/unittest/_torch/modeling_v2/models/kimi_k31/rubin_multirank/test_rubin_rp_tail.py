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
"""Kimi K3.1 fused Lamport finish + attn_res, and the targets' pushed attention all-reduce.

* ``rubin_attn_res``'s Lamport finish: the attn_res + RMSNorm kernel that sums the slots of a
  pushed one-shot all-reduce itself and advances the workspace is bitwise the finish kernel
  followed by the attn_res kernel on its result (both adds), interleaved with C++ all-reduces on
  the workspace and under CUDA graph replay.
* ``reduce_norm``: tep8's slot pushing a partial into its built Lamport push workspace, the mix
  kernel finishing the all-reduce, is bitwise the C++ all-reduce followed by ``add_norm``.

The column-parallel MoE tail of the targets is ``test_rubin_ull2_coll.py``.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

#: tep8's slots (reduce_norm) are certified on SM107 only.
_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)

HIDDEN = 7168
LATENT = 3584
EPS = 1e-6


def _differ(ref, out):
    return int((ref.view(torch.int16) != out.view(torch.int16)).sum())


def _differ_all(refs, outs):
    bad = 0
    for a, b in zip(refs, outs):
        bad += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
    return bad


def _bf16(shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def _attn_res_weights(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    res_w = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    rms_w = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
    out_w = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
    return res_w, rms_w, out_w


def _setup(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    return rank


def _run_rank_finish(world_size):
    """The fused finish against the finish kernel + the attn_res kernel on its result."""
    rank = _setup(world_size)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        rubin_lamport_ar as lamport,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import (
        rubin_attn_res,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, world_size, rank)
        weights = _attn_res_weights(5)
        bad = 0
        checked = 0
        early_checked = 0

        def reference(prefix, addend, block, add_rn):
            if add_rn:
                updated = prefix + addend
                out = rubin_attn_res.k3_attn_res_rmsnorm(updated, block, *weights, EPS, EPS)
            else:
                updated, out = rubin_attn_res.k3_attn_res_add_rmsnorm(
                    prefix, addend, block, *weights, EPS, EPS
                )
            return updated, out

        def fused(prefix, block, add_rn, finish_world, chunk, two_shot=False, early=False):
            return rubin_attn_res.k3_attn_res_lamport_add_rmsnorm(
                prefix,
                ws.uc_local,
                ws.flags.data_ptr(),
                block,
                *weights,
                EPS,
                EPS,
                finish_world,
                chunk,
                two_shot,
                ws.mc_base,
                ws.rank,
                add_rn,
                early,
            )

        # Both cluster sizes of the vectorized kernel (8 CTAs per token for up to 16 tokens with
        # at least 4 candidates, else 4), one-shot and two-shot C++ sizes.
        for m in (1, 14, 32, 64):
            cpp_one_shot = lamport.one_shot_for(
                m, HIDDEN, world_size, _MNNVL_ONE_SHOT_THRESHOLD_BYTES
            )
            chunk = lamport.reduce_chunk(world_size, cpp_one_shot)
            for n_cand in (2, 9):
                for add_rn in (False, True):
                    seed = 1000 * m + 100 * n_cand + add_rn
                    x = _bf16((m, HIDDEN), seed + rank, 0.25)
                    prefix = _bf16((m, HIDDEN), seed + 77)
                    block = _bf16((n_cand - 1, m, HIDDEN), seed + 78)
                    summed = allreduce(x)
                    ref = reference(prefix, summed, block, add_rn)
                    # One-shot / two-shot push, the attn_res kernel finishes the call (also
                    # before its grid wait: early_poll, the *_early_poll constants).
                    for one_shot in (True, False):
                        need = lamport.required_bytes(m, HIDDEN, world_size, one_shot)
                        if need > ws.bytes_per_buffer:
                            continue
                        for early in (False, True):
                            lamport.run_lamport_push(ws, x, one_shot)
                            got = fused(
                                prefix,
                                block,
                                add_rn,
                                world_size,
                                chunk,
                                not one_shot,
                                early,
                            )
                            bad += _differ_all(ref, got)
                            checked += 1
                            early_checked += int(early)
                            # A C++ all-reduce right after it on the workspace.
                            bad += _differ(summed, allreduce(x))
                    # Two-shot push finished in place, the kernel reads the broadcast rows.
                    lamport.run_lamport_push(ws, x, False)
                    lamport.run_lamport_finish_in_place(ws, m, HIDDEN, cpp_one_shot=cpp_one_shot)
                    got = fused(prefix, block, add_rn, 0, 0)
                    bad += _differ_all(ref, got)
                    # The finish kernel on a one-shot push.
                    lamport.run_lamport_push(ws, x, True)
                    out = lamport.run_lamport_finish(
                        ws, torch.empty_like(x), True, cpp_one_shot=cpp_one_shot
                    )
                    bad += _differ(allreduce(x), out)

        # CUDA graph: fused finishes, C++ all-reduces and pushes on one workspace, replayed with
        # new inputs.
        m, n_cand = 14, 5
        cpp_one_shot = lamport.one_shot_for(m, HIDDEN, world_size, _MNNVL_ONE_SHOT_THRESHOLD_BYTES)
        chunk = lamport.reduce_chunk(world_size, cpp_one_shot)
        xs = [_bf16((m, HIDDEN), 500 + i + 10 * rank, 0.25) for i in range(5)]
        prefix = _bf16((m, HIDDEN), 600)
        block = _bf16((n_cand - 1, m, HIDDEN), 601)

        def calls():
            lamport.run_lamport_push(ws, xs[0], True)
            first = fused(prefix, block, True, world_size, chunk)
            mid = allreduce(xs[1])
            lamport.run_lamport_push(ws, xs[2], False)
            second = fused(first[0], block, False, world_size, chunk, True)
            # early_poll, both transports, back to back with the calls above on the workspace.
            lamport.run_lamport_push(ws, xs[3], True)
            third = fused(second[0], block, True, world_size, chunk, False, True)
            lamport.run_lamport_push(ws, xs[4], False)
            fourth = fused(third[0], block, False, world_size, chunk, True, True)
            return first, mid, second, third, fourth

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(5):
                xs[i].copy_(_bf16((m, HIDDEN), 700 + 10 * rep + i + 100 * rank, 0.25))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            first = reference(prefix, allreduce(xs[0]), block, True)
            bad += _differ_all(first, results[0])
            bad += _differ(allreduce(xs[1]), results[1])
            second = reference(first[0], allreduce(xs[2]), block, False)
            bad += _differ_all(second, results[2])
            third = reference(second[0], allreduce(xs[3]), block, True)
            bad += _differ_all(third, results[3])
            fourth = reference(third[0], allreduce(xs[4]), block, False)
            bad += _differ_all(fourth, results[4])
        del graph
        torch.cuda.synchronize()
        assert checked > 0 and early_checked > 0
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} bytes differ"
        assert rubin_attn_res.MAX_TOKENS >= 64
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_reduce_norm(world_size):
    """tep8's ``reduce_norm`` on the built Lamport push workspace (the partial pushed, the mix
    kernel finishing the all-reduce) against the C++ all-reduce followed by ``add_norm``, for
    both transports, with and without ``oproj_ar_early_poll``; every mix takes its PDL kernel (the
    reference's add-and-mix and reduce_norm's Lamport finish, counted at the target's names)."""
    rank = _setup(world_size)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8 import (  # noqa: E501
        modeling as tgt,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_push_allreduce as push,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        c = tgt._CONSTANTS["generation"]
        max_m = c["attn_res_pdl_max_m"]
        ws = push.rubin_lamport_push_workspace(allreduce, max_m, HIDDEN)
        assert ws is not None, "the MNNVL all-reduce has a push workspace"
        res_w, rms_w, out_w = _attn_res_weights(3)
        mw = tgt._MixW(proj=res_w, res_norm=rms_w, out_norm=out_w)
        bad = 0
        one_shot = 0
        cases = [(m, sandwich) for sandwich in (False, True) for m in (1, 8, 14, 32, 64, max_m)]
        # The two PDL mix kernels, looked up by name when a slot runs: count their calls.
        calls = dict.fromkeys(
            ("rubin_attn_res_add_rmsnorm", "rubin_attn_res_lamport_add_rmsnorm"), 0
        )
        kernels = {name: getattr(tgt, name) for name in calls}

        def counted(name):
            def call(*args, **kwargs):
                calls[name] += 1
                return kernels[name](*args, **kwargs)

            return call

        for name in calls:
            setattr(tgt, name, counted(name))
        try:
            for m, sandwich in cases:
                consts = {**c, "oproj_ar_early_poll": sandwich}
                mix = tgt._mix_slots(consts, EPS)
                slots = tgt._tp_reduce_slots(consts, EPS, allreduce, ws, mix)
                partial = _bf16((m, HIDDEN), 9 * m + rank)
                prefix = _bf16((m, HIDDEN), 31 * m)
                bank = _bf16((4, m, HIDDEN), 32 * m)
                ref = mix["add_norm"](prefix, allreduce(partial), bank, mw)
                got = slots["reduce_norm"](prefix, partial, bank, mw)
                one_shot += int(ws.transport(m)[0])
                for a, b in zip(ref, got):
                    bad += _differ(a, b)
        finally:
            for name, kernel in kernels.items():
                setattr(tgt, name, kernel)
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
        assert 0 < one_shot < len(cases), "both transports"
        assert calls == dict.fromkeys(calls, len(cases)), calls
    except Exception:
        traceback.print_exc()
        raise
    return True


def _map(mpi_pool_executor, fn, *args):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    return list(mpi_pool_executor.map(fn, *zip(*[(world_size, *args)] * world_size)))


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_lamport_fused_finish(mpi_pool_executor):
    assert all(r is True for r in _map(mpi_pool_executor, _run_rank_finish))


@skip_pre_blackwell
@pytest.mark.skipif(not _SM107, reason="certified on SM107 only")
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_reduce_norm_fused_finish(mpi_pool_executor):
    assert all(r is True for r in _map(mpi_pool_executor, _run_rank_reduce_norm))
