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
"""Pushed MNNVL all-reduce of the Kimi K3 package (``kernels/k3r_comm/k3_lamport_ar.py``, the low-M GEMM
push epilogue and the ``layers.py`` API on main's ``AllReduce``): bitwise equal to the C++
MNNVL all-reduce, with pushed and C++ calls interleaved on one workspace.

``test_k3_lamport_entry_cells`` drives the CELLS of the collective entries on the shared MNNVL
workspace of the group, each call under ``validating`` and against the entry's reference, which
takes every rank's input in rank order:

* ``k3_lamport_push``: the slots of this rank's buffer (one-shot: every token's ``W`` slots;
  two-shot: the tokens this rank owns) hold every rank's sanitized rows, bit for bit;
* ``k3_lamport_finish``: the rank-order sum in the C++ kernel's order, bit for bit;
* ``k3_attn_res_lamport``: the updated prefix bit for bit, the normed mix within the family
  attn_res tests' cosine / relative-L2 band, the MXFP8 output bit for bit the low-M GEMM's
  quantization of the stored output;
* ``k3_latent_ar_rmsnorm``: the normed latent within the same band, its MXFP8 output (row-major
  or the mid-M GEMM's atoms) bit for bit the quantization of the stored output.
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


def _special(m, h, gen):
    """BF16 ``[m, h]`` with -0, +0, denormals, large values, inf and NaN lanes."""
    x = torch.randn(m, h, device="cuda", generator=gen) * torch.exp2(
        torch.randint(-20, 20, (m, 1), device="cuda", generator=gen).float()
    )
    x = x.to(torch.bfloat16)
    bits = x.view(torch.int16)
    r = torch.rand(m, h, device="cuda", generator=gen)
    bits[r < 0.05] = -32768
    bits[(r >= 0.05) & (r < 0.08)] = 0
    den = (r >= 0.08) & (r < 0.11)
    bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
        torch.int16
    )
    bits[(r >= 0.11) & (r < 0.115)] = 0x7F80
    bits[(r >= 0.115) & (r < 0.117)] = 0x7FC1
    return x


def _input(m, h, seed, rank, kind):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    if kind == 0:
        return (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    if kind == 1:
        return _special(m, h, gen)
    # Every rank sends denormals only: the sums stay denormal (no flush to zero anywhere).
    bits = torch.randint(-127, 127, (m, h), device="cuda", generator=gen).to(torch.int16)
    return bits.view(torch.bfloat16)


def _block_weights(n, k, seed):
    """E4M3 ``[n, k]`` weight and its power-of-two FP32 128x128 block scales."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.exp2(
        torch.randint(-8, -2, (n // 128, k // 128), device="cuda", generator=gen).float()
    )
    return w, scale


_K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
# tests/unittest/_torch/modeling_v2, where _validating lives (the MPI workers import it by path).
_MV2_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_ENTRIES = (
    "ops.comm.k3_lamport_push",
    "ops.comm.k3_lamport_finish",
    "ops.comm.k3_attn_res_lamport",
    "ops.comm.k3_latent_ar_rmsnorm",
)


def _weights(n, k, seed):
    import importlib

    lowm = importlib.import_module(f"{_K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")

    return lowm.prepare_k3_fp8w_lowm_weight(*_block_weights(n, k, seed))


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    # Imported by name: an attribute named ``ops`` next to the global ``torch`` would make
    # cloudpickle ship the ``torch.ops`` object with this function.
    from tensorrt_llm._torch.distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    lowm = importlib.import_module(f"{_K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
    lamport = importlib.import_module(f"{_K3}.kernels.k3r_comm.k3_lamport_ar")
    push = importlib.import_module(f"{_K3}.ops.comm.k3_lamport_push").k3_lamport_push
    finish = importlib.import_module(f"{_K3}.ops.comm.k3_lamport_finish").k3_lamport_finish
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, world_size, rank)
        threshold = _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        bad = 0

        def differ(ref, out):
            return int((ref.view(torch.int16) != out.view(torch.int16)).sum())

        # Push kernel + finish kernel against the C++ all-reduce, both transports reproducing
        # the C++ kernel the size selects, with C++ calls in between.
        for m, h in ((1, 7168), (14, 7168), (64, 7168), (300, 7168), (14, 3584), (5, 64)):
            cpp_one_shot = lamport.one_shot_for(m, h, world_size, threshold)
            for kind in (0, 1, 2):
                x = _input(m, h, 3 * m + kind, rank, kind)
                ref = allreduce(x)
                for one_shot in (False, True):
                    if lamport.required_bytes(m, h, world_size, one_shot) > ws.bytes_per_buffer:
                        continue
                    out = torch.empty_like(x)
                    push(ws, x, one_shot)
                    finish(ws, out, one_shot, cpp_one_shot=cpp_one_shot)
                    bad += differ(ref, out)
                    bad += differ(ref, allreduce(x))

        # GEMM push epilogue + finish against the GEMM followed by the C++ all-reduce.
        for n, k in ((7168, 768), (7168, 384)):
            w, w_scale = _weights(n, k, 11 + rank)
            for m in (1, 7, 14, 32, 64):
                cpp_one_shot = lamport.one_shot_for(m, n, world_size, threshold)
                x = _input(m, k, 5 * m, rank, 0)
                y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                lowm.run_k3_fp8w_lowm_gemm(x, w, w_scale, y)
                ref = allreduce(y)
                for one_shot in (False, True):
                    lowm.run_k3_fp8w_lowm_gemm_ar_push(x, w, w_scale, ws, one_shot)
                    out = finish(ws, torch.empty_like(y), one_shot, cpp_one_shot=cpp_one_shot)
                    bad += differ(ref, out)

        # One CUDA graph with pushed and C++ calls on the workspace, replayed with new inputs.
        n, k, m = 7168, 768, 14
        cpp_one_shot = lamport.one_shot_for(m, n, world_size, threshold)
        w, w_scale = _weights(n, k, 99 + rank)
        xs = [_input(m, k, 700 + i, rank, 0) for i in range(4)]
        ys = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        outs = [torch.empty_like(y) for y in ys]

        def calls():
            res = []
            for i in range(4):
                if i % 2 == 0:
                    lowm.run_k3_fp8w_lowm_gemm_ar_push(xs[i], w, w_scale, ws, cpp_one_shot)
                    res.append(finish(ws, outs[i], cpp_one_shot, cpp_one_shot=cpp_one_shot))
                else:
                    lowm.run_k3_fp8w_lowm_gemm(xs[i], w, w_scale, ys[i])
                    res.append(allreduce(ys[i]))
            return res

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                xs[i].copy_(_input(m, k, 900 + 10 * rep + i, rank, 0))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(4):
                y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                lowm.run_k3_fp8w_lowm_gemm(xs[i], w, w_scale, y)
                bad += differ(allreduce(y), results[i])
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_api(world_size):
    """``layers`` on main's ``AllReduce``: a pushed tensor finished as a tensor and deferred
    (``LamportPending``) against ``AllReduce.forward``, eagerly and in a CUDA graph."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    api = importlib.import_module(f"{_K3}.ops.comm.sm107_allreduce")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)

    def differ(ref, out):
        return int((ref.view(torch.int16) != out.view(torch.int16)).sum())

    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        bad = 0
        kinds = {"plain": 0, "deferred": 0}
        for m, h in ((1, 7168), (14, 7168), (64, 7168), (300, 7168), (14, 3584)):
            for kind in (0, 1):
                x = _input(m, h, 7 * m + kind, rank, kind)
                ref = allreduce(x)

                target = api.lamport_push_target(allreduce, m, h)
                assert target is not None
                try:
                    api.finish_lamport_push(allreduce, target)
                    raise AssertionError("finish_lamport_push accepted a target nobody pushed")
                except RuntimeError:
                    pass
                api.lamport_push_tensor(allreduce, target, x)
                out = api.finish_lamport_push(allreduce, target)
                bad += differ(ref, out)
                kinds["plain"] += 1

                target = api.lamport_push_target(allreduce, m, h, deferred=True)
                assert target.deferred
                api.lamport_push_tensor(allreduce, target, x)
                pending = api.finish_lamport_push(allreduce, target, deferred=True)
                assert isinstance(pending, api.LamportPending)
                bad += differ(ref, pending.materialize())
                kinds["deferred"] += 1

                bad += differ(ref, allreduce(x))

        # MNNVLAllReduce-level target with a forced transport (the drafter's embedding site).
        for one_shot in (True, False):
            x = _input(9, 7168, 50 + int(one_shot), rank, 0)
            ref = allreduce(x)
            target = api.mnnvl_lamport_push_target(
                allreduce.mnnvl_allreduce, 9, 7168, one_shot=one_shot
            )
            assert target.one_shot == one_shot
            api.lamport_push_tensor(allreduce, target, x)
            bad += differ(ref, api.finish_lamport_push(allreduce, target))

        # CUDA graph: plain push, C++ call, deferred push on one workspace, replayed.
        m, h = 14, 7168
        xs = [torch.zeros(m, h, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

        def calls():
            t0 = api.lamport_push_target(allreduce, m, h)
            api.lamport_push_tensor(allreduce, t0, xs[0])
            a = api.finish_lamport_push(allreduce, t0)
            b = allreduce(xs[1])
            t2 = api.lamport_push_target(allreduce, m, h, deferred=True)
            api.lamport_push_tensor(allreduce, t2, xs[2])
            c = api.finish_lamport_push(allreduce, t2, deferred=True).materialize()
            return a, b, c

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = calls()
        for rep in range(3):
            for i, x in enumerate(xs):
                x.copy_(_input(m, h, 600 + 10 * rep + i, rank, rep % 2))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for x, out in zip(xs, outs):
                bad += differ(allreduce(x), out)
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
        assert all(count == 10 for count in kinds.values()), kinds
    except Exception:
        traceback.print_exc()
        raise
    return True


def _similar(actual, expected):
    """The family attn_res tests' metric: cosine > 0.9999 and relative L2 < 5e-3."""
    a, e = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
    return cosine > 0.9999 and rel_l2 < 5e-3, f"cosine {cosine:.6f}, relative L2 {rel_l2:.3e}"


def _run_rank_cells(world_size):
    """Every cell of the four collective entries on the group's shared MNNVL workspace; returns
    this rank's failures."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib
    import zlib

    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.mapping import Mapping

    sys.path.insert(0, _MV2_TESTS)
    from _validating import validating

    lamport = importlib.import_module(f"{_K3}.kernels.k3r_comm.k3_lamport_ar")
    norm_ops = importlib.import_module(f"{_K3}.ops.norm.k3_attn_res_norm")
    push, finish, attn_res, latent = (
        getattr(importlib.import_module(f"{_K3}.{name}"), name.rpartition(".")[2])
        for name in _ENTRIES
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    W = world_size
    arch = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}.get(
        torch.cuda.get_device_capability()
    )
    failures = []

    def check(entry, cell, ok, what):
        if not ok:
            failures.append(f"rank {rank} {type(entry).__name__} [{cell.why}]: {what}")

    def bits(t):
        return t.contiguous().view(torch.int16)

    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def seed(cell, salt=0):
        return (zlib.crc32(cell.why.encode()) + salt) % 100003

    def sync():
        torch.cuda.synchronize()
        comm.Barrier()

    def rnd(gen, *shape, scale=1.0, shift=0.0):
        x = torch.randn(*shape, generator=gen, device="cuda") * scale + shift
        return x.to(torch.bfloat16)

    try:
        mapping = Mapping(world_size=W, tp_size=W, rank=rank)
        AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        # The largest call of the cells: 96 one-shot rows of 7168.
        need = lamport.required_bytes(96, 7168, W, True)
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            mapping, torch.bfloat16, buffer_size_bytes=need
        )
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, W, rank)
        slots = workspace["uc_buffer"].view(torch.int16)

        if arch in push.ARCHS:
            for cell in push.CELLS:
                m, h, one_shot = cell.spec["rows"], cell.spec["hidden"], cell.spec["one_shot"]
                xs = [_input(m, h, seed(cell), r, 0) for r in range(W)]
                sync()
                flags = ws.flags.view(torch.int32)
                base = int(flags[0].item()) * int(flags[2].item()) // 2
                with validating(push):
                    push(ws, xs[rank], one_shot)
                sync()
                want = torch.stack([bits(push.reference(ws, x, one_shot)) for x in xs])
                if one_shot:
                    got = slots[base : base + m * W * h].view(m, W, h).transpose(0, 1)
                else:
                    owned = torch.arange(rank, m, W, device="cuda")
                    rows = -(-m // W)
                    got = slots[base : base + rows * W * h].view(rows, W, h)[owned // W]
                    got, want = got.transpose(0, 1), want[:, owned]
                check(push, cell, torch.equal(got, want), "slot contents")
                out = finish(ws, torch.empty_like(xs[rank]), one_shot)
                check(push, cell, same(out, finish.reference(xs, out, one_shot)), "sum")

        if arch in finish.ARCHS:
            for cell in finish.CELLS:
                spec = cell.spec
                m, h, one_shot = spec["rows"], spec["hidden"], spec["one_shot"]
                xs = [_input(m, h, seed(cell), r, 0) for r in range(W)]
                sync()
                push(ws, xs[rank], one_shot)
                out = torch.empty_like(xs[rank])
                with validating(finish):
                    got = finish(ws, out, one_shot, cpp_one_shot=spec["cpp_one_shot"])
                want = finish.reference(xs, out, one_shot, cpp_one_shot=spec["cpp_one_shot"])
                check(finish, cell, got.data_ptr() == out.data_ptr(), "returned another tensor")
                check(finish, cell, same(got, want), "sum")

        if arch in attn_res.ARCHS:
            for cell in attn_res.CELLS:
                spec = cell.spec
                m, n, h = spec["rows"], spec["candidates"], spec["hidden"]
                gen = torch.Generator(device="cuda").manual_seed(seed(cell, 1))
                prefix = rnd(gen, m, h, scale=0.5)
                block = rnd(gen, n - 1, m, h, scale=0.5)
                weights = (
                    rnd(gen, h, scale=0.02),
                    rnd(gen, h, scale=0.02, shift=1.0),
                    rnd(gen, h, scale=0.02, shift=1.0),
                )
                xs = [_input(m, h, seed(cell), r, 0) for r in range(W)]
                two_shot = spec["two_shot"]
                chunk = lamport.reduce_chunk(W, spec["cpp_one_shot"])
                tail = (W, chunk, two_shot, ws.mc_base, rank, spec["add_rn"])
                tail += (spec["early_poll"], spec["entry_trigger"], False)
                eps = (spec["rms_eps"], spec["out_eps"])
                mxfp8 = spec["mxfp8"]
                updated = torch.empty_like(prefix) if spec["into"] else None
                sync()
                push(ws, xs[rank], not two_shot)
                ws_args = (ws.uc_local, ws.flags.data_ptr())
                with validating(attn_res):
                    got = attn_res(
                        prefix,
                        *ws_args,
                        block,
                        *weights,
                        *eps,
                        *tail,
                        mxfp8=mxfp8,
                        updated=updated,
                        early_trigger=spec["early_trigger"],
                    )
                ref = attn_res.reference(prefix, xs, block, *weights, *eps, *tail, mxfp8=mxfp8)
                check(attn_res, cell, len(got) == len(ref), "number of outputs")
                check(attn_res, cell, same(got[0], ref[0]), "updated prefix")
                if updated is not None:
                    check(attn_res, cell, got[0].data_ptr() == updated.data_ptr(), "not in updated")
                ok, detail = _similar(got[1], ref[1])
                check(attn_res, cell, ok, f"output: {detail}")
                if mxfp8:
                    want_q, want_s = norm_ops.mxfp8_rows_reference(got[1])
                    check(attn_res, cell, same(got[2], want_q), "E4M3 codes")
                    check(attn_res, cell, same(got[3], want_s), "UE8M0 scales")

        if arch in latent.ARCHS:
            for cell in latent.CELLS:
                spec = cell.spec
                m, h = spec["rows"], spec["hidden"]
                gen = torch.Generator(device="cuda").manual_seed(seed(cell, 1))
                gamma = rnd(gen, h, scale=0.02, shift=1.0)
                xs = [_input(m, h, seed(cell), r, 0) for r in range(W)]
                out = torch.empty_like(xs[rank])
                mx, sf_blocks = None, 0
                if spec["mxfp8"] is not None:
                    sf_blocks = 1 if spec["mxfp8"] == "atoms" else 0
                    scale_shape = ((h // 128) * 512,) if sf_blocks else (m, h // 32)
                    mx = (
                        torch.empty(m, h, dtype=torch.float8_e4m3fn, device="cuda"),
                        torch.empty(scale_shape, dtype=torch.uint8, device="cuda"),
                    )
                kwargs = dict(
                    uc_addr=ws.uc_local,
                    mc_addr=ws.mc_base,
                    flags=ws.flags,
                    rank=rank,
                    world_size=W,
                    mxfp8_out=mx,
                    sf_blocks=sf_blocks,
                    early_state=spec["early_state"],
                    row=spec["row"],
                )
                sync()
                with validating(latent):
                    got = latent(xs[rank], gamma, spec["eps"], out, **kwargs)
                ref = latent.reference(xs, gamma, spec["eps"], out, **kwargs)
                got_out, ref_out = (got, ref) if mx is None else (got[0], ref[0])
                check(latent, cell, got_out.data_ptr() == out.data_ptr(), "not in out")
                ok, detail = _similar(got_out, ref_out)
                check(latent, cell, ok, f"output: {detail}")
                if mx is not None:
                    want_q, want_s = norm_ops.mxfp8_rows_reference(got_out)
                    check(latent, cell, same(mx[0], want_q), "E4M3 codes")
                    if sf_blocks:
                        offsets = norm_ops.mx_atom_offsets(m, h, out.device)
                        check(latent, cell, same(mx[1][offsets], want_s.reshape(-1)), "atoms")
                    else:
                        check(latent, cell, same(mx[1], want_s), "UE8M0 scales")
        sync()
    except Exception:
        traceback.print_exc()
        raise
    return failures


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_entry_cells(mpi_pool_executor):
    """The CELLS of k3_lamport_push, k3_lamport_finish, k3_attn_res_lamport and
    k3_latent_ar_rmsnorm (see the module docstring)."""
    import importlib

    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch

    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    arch = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}.get(
        torch.cuda.get_device_capability()
    )
    ops = [
        getattr(importlib.import_module(f"{_K3}.{name}"), name.rpartition(".")[2])
        for name in _ENTRIES
    ]
    if not any(arch in op.ARCHS for op in ops):
        pytest.skip(f"{arch} is in none of the entries' ARCHS")
    results = mpi_pool_executor.map(_run_rank_cells, *zip(*[(world_size,)] * world_size))
    failures = [f for r in results for f in r]
    assert not failures, "\n".join(failures[:40])


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_ar(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_push_api(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_api, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
