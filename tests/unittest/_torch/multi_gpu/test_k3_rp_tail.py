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
"""Kimi K3 fused Lamport finish + attn_res.

* ``k3_attn_res`` with ``lamport_finish``: the attn_res + RMSNorm kernel that sums the slots of a
  pushed one-shot all-reduce itself and advances the workspace is bitwise the finish kernel
  followed by the attn_res kernel on its result (both adds), interleaved with C++ all-reduces on
  the workspace and under CUDA graph replay.
* A_mid's fused finish: o_proj push -> A_mid finishing the all-reduce is bitwise o_proj ->
  C++ all-reduce -> A_mid, through the model helpers.
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

HIDDEN = 7168
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
    from tensorrt_llm._torch.kernels.blackwell import k3_attn_res
    from tensorrt_llm._torch.kernels.blackwell import k3_lamport_ar as lamport

    # The ops by name: ``torch.ops`` in a function shipped to the workers cannot be pickled.
    attn_ops = k3_attn_res
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
                return updated, attn_ops.k3_attn_res_rmsnorm(updated, block, *weights, EPS, EPS)
            return attn_ops.k3_attn_res_add_rmsnorm(prefix, addend, block, *weights, EPS, EPS)

        def fused(prefix, block, add_rn, finish_world, chunk, two_shot=False, early=False):
            return attn_ops.k3_attn_res_lamport_add_rmsnorm(
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
                    # before its grid wait: early_poll, as A_mid's early poll runs it).
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
                    # The finish kernel on a one-shot push (LamportPending.materialize).
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
        assert k3_attn_res.MAX_TOKENS >= 64
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_oproj(world_size):
    """A_mid's fused finish: o_proj push -> A_mid finishing the all-reduce, against o_proj ->
    C++ all-reduce -> A_mid, through the model helpers."""
    rank = _setup(world_size)
    import importlib

    from torch import nn

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    dist_ops = importlib.import_module("tensorrt_llm._torch.distributed.ops")
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    switches = {
        (dist_ops, "_K3_AR_PUSH"): True,
        (kimi, "_K3_AR_PUSH_AMID"): True,
        (kimi, "_K3_ATTN_RES_PDL"): True,
        (kimi, "_K3_ULL_OPROJ_SANDWICH"): False,
    }
    saved = {key: getattr(*key) for key in switches}
    try:
        for (module, name), value in switches.items():
            setattr(module, name, value)
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        gen = torch.Generator(device="cuda").manual_seed(21 + rank)
        linear = nn.Linear(768, HIDDEN, bias=False, dtype=torch.bfloat16, device="cuda")
        linear.weight.data.copy_(torch.randn(HIDDEN, 768, device="cuda", generator=gen) * 0.05)
        gen = torch.Generator(device="cuda").manual_seed(3)
        proj = nn.Linear(HIDDEN, 1, bias=False, dtype=torch.bfloat16, device="cuda")
        proj.weight.data.copy_(torch.randn(1, HIDDEN, device="cuda", generator=gen) * 0.05)
        norm = kimi.KimiK3RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16, device="cuda")
        norm.weight.data.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen))
        out_norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
        out_norm.weight.data.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen))
        bad = 0
        pending = 0
        one_shot = 0
        limit = dist_ops._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
        # A_mid's early poll (_K3_ULL_OPROJ_SANDWICH): A_mid finishes the push before its grid wait.
        cases = [(m, sandwich) for sandwich in (False, True) for m in (1, 8, 14, 32, 64)]
        for m, sandwich in cases:
            kimi._K3_ULL_OPROJ_SANDWICH = sandwich
            x = _bf16((m, 768), 9 * m + rank)
            prefix = _bf16((m, HIDDEN), 31 * m)
            block = _bf16((4, m, HIDDEN), 32 * m)
            ref = kimi._apply_attn_res_add_and_rmsnorm(
                prefix, allreduce(linear(x)), block, proj, norm, out_norm
            )
            # The copy kernel pushes the o_proj output.
            target = allreduce.lamport_push_target(m, HIDDEN, in_place=True, deferred=True)
            assert target.deferred
            assert target.one_shot == (m * HIDDEN * world_size * 2 <= limit)
            one_shot += int(target.one_shot)
            allreduce.lamport_push_tensor(target, linear(x))
            result = allreduce.finish_lamport_push(target, in_place=True, deferred=True)
            pending += int(isinstance(result, dist_ops.LamportPending))
            got = kimi._apply_attn_res_add_and_rmsnorm(prefix, result, block, proj, norm, out_norm)
            for a, b in zip(ref, got):
                bad += _differ(a, b)
            # Unfused fallback of a pending result: the finish kernel into a tensor (before the
            # next all-reduce on the workspace).
            target = allreduce.lamport_push_target(m, HIDDEN, in_place=True, deferred=True)
            allreduce.lamport_push_tensor(target, linear(x))
            result = allreduce.finish_lamport_push(target, in_place=True, deferred=True)
            got = result.materialize().view(m, HIDDEN)
            bad += _differ(allreduce(linear(x)), got)
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
        # Both transports, finished by A_mid.
        assert pending == len(cases) and 0 < one_shot < len(cases)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    return True


def _map(mpi_pool_executor, fn, *args):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    return list(mpi_pool_executor.map(fn, *zip(*[(world_size, *args)] * world_size)))


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_fused_finish(mpi_pool_executor):
    assert all(r is True for r in _map(mpi_pool_executor, _run_rank_finish))


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_oproj_fused_finish(mpi_pool_executor):
    assert all(r is True for r in _map(mpi_pool_executor, _run_rank_oproj))
