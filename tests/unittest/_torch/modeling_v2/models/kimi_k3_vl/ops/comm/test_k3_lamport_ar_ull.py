# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's collective Lamport entries on their ``.CELLS``, on the shared MNNVL workspace of
a 2- or 4-rank group, each call under ``validating`` and against the entry's reference, which takes
every rank's input in rank order (the checks of ``kernels/k3r_comm/test_k3_lamport_ar.py``):

* ``k3_lamport_push``: the slots of this rank's buffer (one-shot: every token's ``W`` slots;
  two-shot: the tokens this rank owns) hold every rank's sanitized rows, bit for bit;
* ``k3_lamport_finish``: the rank-order sum in the C++ kernel's order, bit for bit;
* ``k3_attn_res_lamport_ull``: the updated prefix bit for bit, the normed mix within the family
  attn_res tests' cosine / relative-L2 band, the MXFP8 output bit for bit the low-M GEMM's
  quantization of the stored output.
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
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

_K3 = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"
# tests/unittest/_torch/modeling_v2, where _validating lives (the MPI workers import it by path).
_MV2_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_ENTRIES = (
    "ops.comm.k3_lamport_push",
    "ops.comm.k3_lamport_finish",
    "ops.comm.k3_attn_res_lamport_ull",
)


def _entry(name):
    import importlib

    return getattr(
        importlib.import_module(f"{_K3}.{name}"), name.rpartition(".")[2].removesuffix("_ull")
    )


def _input(m, h, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    return (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)


def _similar(actual, expected):
    """The family attn_res tests' metric: cosine > 0.9999 and relative L2 < 5e-3."""
    a, e = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
    return cosine > 0.9999 and rel_l2 < 5e-3, f"cosine {cosine:.6f}, relative L2 {rel_l2:.3e}"


def _run_rank_cells(world_size):
    """Every cell of the three collective entries on the group's shared MNNVL workspace; returns
    this rank's failures."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib
    import zlib

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm._torch.modeling_v2.catalog._op import Arch
    from tensorrt_llm.mapping import Mapping

    sys.path.insert(0, _MV2_TESTS)
    from _validating import validating

    lamport = importlib.import_module(f"{_K3}.kernels.k3r_comm.k3_lamport_ar")
    norm_ops = importlib.import_module(f"{_K3}.ops.norm.k3_attn_res_norm_ull")
    push, finish, attn_res = (_entry(name) for name in _ENTRIES)
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
                xs = [_input(m, h, seed(cell), r) for r in range(W)]
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
                xs = [_input(m, h, seed(cell), r) for r in range(W)]
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
                xs = [_input(m, h, seed(cell), r) for r in range(W)]
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
                        prefix, *ws_args, block, *weights, *eps, *tail, mxfp8=mxfp8, updated=updated
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
        sync()
    except Exception:
        traceback.print_exc()
        raise
    return failures


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_lamport_ull_entry_cells(mpi_pool_executor):
    """The CELLS of k3_lamport_push, k3_lamport_finish and k3_attn_res_lamport_ull."""
    from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    arch = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}.get(
        torch.cuda.get_device_capability()
    )
    if not any(arch in _entry(name).ARCHS for name in _ENTRIES):
        pytest.skip(f"{arch} is in none of the entries' ARCHS")
    results = mpi_pool_executor.map(_run_rank_cells, *zip(*[(world_size,)] * world_size))
    failures = [f for r in results for f in r]
    assert not failures, "\n".join(failures[:40])
