# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 low-latency MoE entries over their TP group on one node (one VR200 tray: TP2 and TP4).

* rubin_ull_cp_tail: every rank pushes its shared partial into the tail call's slots and its
  latent_up columns into the all-gather rows; the finish (``ag``) must equal
  ``bf16(y + bf16(all-reduced shared))`` bit for bit, ``y`` the same split-K GEMM's plain products.
* rubin_ull_front_bf16: the routing bitwise against the routing tail's reference on the same logits;
  the NVFP4 latent against the entry's reference.
* the tail finished by the next mix (the all-gather cells of rubin_attn_res_lamport_add_rmsnorm
  and of its ``_mx`` entry, 4 ranks; 8 where a node hosts them): the updated rows bit for bit, the
  MXFP8 form of the mix's own output bit for bit, and against the replicated tail (cuBLAS latent_up
  + the C++ all-reduce of the shared partials + the PDL MoE mix) within the A-class band, the
  shared sum bit for bit the C++ all-reduce's.
* one tail workspace over a decode sequence: in-band calls of varying rows, finished by the BF16 mix
  or by the finish kernel (a tapped layer's add), between steps that run the C++ all-reduce instead
  and leave the workspace alone; every finish bit for bit its reference.
* a block boundary's mix after attention (the ``replace`` cells of
  rubin_attn_res_lamport_add_rmsnorm, 8 ranks; the 4-rank ones run in test_rubin_comm_entries.py):
  every rank pushes its o_proj partial, the mix finishes the call as the updated prefix; both
  outputs bit for bit the C++ all-reduce followed by rubin_attn_res_rmsnorm.

Past 64 rows the push GEMM runs two N tiles of 64 rows; the plain reference runs the store-only
kernel 64 rows at a time.
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

HIDDEN, LATENT, NUM_EXPERTS = 7168, 3584, 896
# NVFP4 latent: a code differs only where the BF16 latent sits within an ulp of an E2M1 rounding
# boundary (split-K GEMM against the true-FP32 reference); 1 in 200 elements is the gate.
MAX_FLIPPED = 0.005
_M = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


def _entry(name):
    import importlib

    return getattr(importlib.import_module(f"{_M}.ops.moe.{name}"), name)


def _setup(rank, world):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm.mapping import Mapping

    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    return Mapping(world_size=world, tp_size=world, rank=rank, gpus_per_node=gpus)


def _bits(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


# The plain-torch finish of ops/moe/_rubin_ull_reference.py on ``c`` (ops.moe._rubin_ull_common),
# copied here: the MPI workers get this module by value, so it needs no import path to the tests.
def _allreduce(c, parts, cpp_one_shot):
    """The finish of a call: FP32 sums of the slots in rank order (chunks of ``c.reduce_chunk``
    ranks), rounded once to BF16, ``-0`` as ``+0``."""
    acc = torch.zeros(parts[0].shape, dtype=torch.float32)
    chunk = c.reduce_chunk(len(parts), cpp_one_shot)
    for c0 in range(0, len(parts), chunk):
        part = torch.zeros_like(acc)
        for p in parts[c0 : c0 + chunk]:
            part = part + c.sanitize_bf16(p).float()
        acc = acc + part
    return c.sanitize_bf16(acc.to(torch.bfloat16))


def _ag_finish(c, y, parts, cpp_one_shot):
    """The finish of a call with all-gather rows: ``bf16(y + bf16(sum of the slots))``."""
    return (y.float() + _allreduce(c, parts, cpp_one_shot).float()).to(torch.bfloat16)


def _plain_ag_rows(c, k, base, latent, up, world, tactic):
    """Every rank's all-gather columns as the push GEMM's plain store gives them (``-0`` as
    ``+0``): the same split-K kernel at the same tactic, 64 rows per launch (past 64 rows the
    push kernel's second N tile is the same tile over the next rows)."""
    m = latent.shape[0]
    y = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for r in range(world):
        r0, rn = c.block_slice(HIDDEN, world, r)
        w = up[r0 : r0 + rn].contiguous()
        for t0 in range(0, m, 64):
            rows = slice(t0, min(m, t0 + 64))
            plain = torch.empty(rows.stop - t0, rn, dtype=torch.bfloat16, device="cuda")
            base.run_gemm(latent[rows].contiguous(), w, plain, k.tactic_of(tactic))
            y[rows, r0 : r0 + rn] = c.sanitize_bf16(plain)
    return y


def _failure(spec):
    """The worker's exception as a RuntimeError holding its traceback as text.

    mpi4py returns a worker's exception without its traceback, and one that does not pickle
    comes back as the pickling error instead.
    """
    comm = MPI.COMM_WORLD
    text = f"rank {comm.Get_rank()} of {comm.Get_size()}, {spec}:\n{traceback.format_exc()}"
    print(text, file=sys.stderr, flush=True)
    return RuntimeError(text)


def _tail_rank(spec):
    try:
        import importlib

        c = importlib.import_module(f"{_M}.ops.moe._rubin_ull_common")
        k = importlib.import_module(f"{_M}.kernels.moe.rubin_ull_push_gemm")
        base = importlib.import_module(f"{_M}.kernels.spec.rubin_draft_gemm_push")
        lamport = importlib.import_module(f"{_M}.kernels.comm.rubin_lamport_ar")
        cp_mod = importlib.import_module(f"{_M}.ops.moe.rubin_ull_cp_tail")
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        mapping = _setup(rank, world)
        built = cp_mod.rubin_ull_tail_workspace(mapping, world)
        ws = built["ws"]
        m = spec["rows"]
        c0, n, tactics = cp_mod.rubin_ull_cp_tail_share(world, rank)
        tactic = tactics[m]
        g = torch.Generator(device="cuda").manual_seed(5)
        latent = torch.randn(m, LATENT, device="cuda", generator=g).bfloat16()
        up = (torch.randn(HIDDEN, LATENT, device="cuda", generator=g) * LATENT**-0.5).bfloat16()
        shared = []
        for r in range(world):
            gr = torch.Generator(device="cuda").manual_seed(50 + r)
            shared.append((torch.randn(m, HIDDEN, device="cuda", generator=gr) * 0.5).bfloat16())
        cpp_one_shot = built["cpp_one_shot"]
        bad = 0
        for _ in range(2):  # two calls: the second one runs on the next buffer
            lamport.run_lamport_push(ws, shared[rank], True)
            _entry("rubin_ull_cp_tail")(latent, up[c0 : c0 + n], ws, tactic, c0=c0)
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
            lamport.run_lamport_finish(ws, out, True, cpp_one_shot=cpp_one_shot, ag=True)
            y = _plain_ag_rows(c, k, base, latent, up, world, tactic)
            expect = _ag_finish(c, y.cpu(), [s.cpu() for s in shared], cpp_one_shot)
            torch.cuda.synchronize()
            bad += int(not _bits(out.cpu(), expect))
            # The finish of the replicated tail: bf16(latent_up + all-reduced shared) at the
            # chunk the targets' finish takes.
            assert built["chunk"] == c.reduce_chunk(world, cpp_one_shot)
        result = (bad,)
    except Exception:
        raise _failure(spec) from None
    return result


def _front_rank(spec):
    try:
        import importlib

        import tensorrt_llm._torch.custom_ops  # noqa: F401 -- dsv3_router_gemm_op

        fb = importlib.import_module(f"{_M}.ops.moe.rubin_ull_front_bf16")
        moe_common = importlib.import_module(f"{_M}.ops.moe._rubin_moe_common")
        route_tail = _entry("rubin_route_tail")
        # The router GEMM through its entry: with ``torch.ops`` in a function cloudpickle ships
        # by value, cloudpickle also ships the ``torch.ops`` object, which does not pickle, and
        # the task never reaches the workers.
        router_gemm = _entry("dsv3_router_gemm_op")
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        mapping = _setup(rank, world)
        ws = fb.rubin_ull_front_workspace(mapping)
        views = fb.rubin_ull_front_logits_views(ws)
        g = torch.Generator(device="cuda").manual_seed(7)
        down = (torch.randn(LATENT, HIDDEN, device="cuda", generator=g) * HIDDEN**-0.5).bfloat16()
        router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
        bias = (torch.rand(NUM_EXPERTS, device="cuda", generator=g) - 0.5) * 0.19
        rows = spec["rows"]
        x = torch.randn(rows, HIDDEN, device="cuda", generator=g).bfloat16()
        ep = NUM_EXPERTS // world
        logits = torch.cat(
            [
                router_gemm(
                    x[r : r + 16].contiguous(), router.t(), bias=None, out_dtype=torch.float32
                )
                for r in range(0, rows, 16)
            ]
        )
        op = fb.rubin_ull_front_bf16
        _, ref_latent = op.reference(x, logits, down, bias, 1.0, 1.0, rank * ep, ep)
        gsf = torch.tensor([2688.0 / ref_latent.abs().amax().item()], device="cuda")
        r0, n, tactics = fb.rubin_ull_front_share(world, rank)
        latent_rows = down[r0 : r0 + n]
        zero = torch.full((rows * LATENT // 2 + 37,), -1, dtype=torch.int32, device="cuda")
        args = (x, logits, latent_rows, views, bias, 1.0, gsf, rank * ep, ep, ws, tactics[rows])
        op.is_valid(*args, zero=zero)
        bad = 0
        for _ in range(3):  # both workspace halves, then the first again
            zero.fill_(-1)
            routing, codes, scales = op(*args, zero=zero)
            torch.cuda.synchronize()
            bad += int((zero != 0).sum())
            tail = route_tail.reference(logits, bias, 1.0, rank * ep, ep)
            count = int(tail[7].item())
            bad += int(not torch.equal(routing[0], tail[0]))
            bad += int(not torch.equal(routing[1].view(torch.int32), tail[1].view(torch.int32)))
            bad += int(not torch.equal(routing[4], tail[4])) + int(int(routing[7]) != count)
            bad += int(not torch.equal(routing[2][:count], tail[2][:count]))
            _, ref = op.reference(x, logits, down, bias, 1.0, float(gsf), rank * ep, ep)
            sf = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
            got = moe_common.e2m1_values(codes) * sf / float(gsf)
            flipped = (got != ref).float().mean().item()
            bad += int(flipped > MAX_FLIPPED)
            bad += int(((got - ref).abs() > 2 * sf / float(gsf) + 1e-30).any())
        result = (bad, flipped)
    except Exception:
        raise _failure(spec) from None
    return result


# The group sizes one node hosts: one VR200 tray holds 4 GPUs, so the 8-rank cells (helix8's MoE
# TP8 group, two trays) run where a node has 8 GPUs, or in an 8-rank run.
_WORLDS = [w for w in (2, 4, 8) if w <= max(torch.cuda.device_count(), 4)]


@pytest.mark.parametrize("mpi_pool_executor", _WORLDS, indirect=True)
@pytest.mark.parametrize("name", ["rubin_ull_cp_tail", "rubin_ull_front_bf16"])
def test_certified_cells(mpi_pool_executor, name):
    """The entry's CELLS of this group size."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    run = _front_rank if name == "rubin_ull_front_bf16" else _tail_rank
    for cell in [c for c in _entry(name).CELLS if c.spec["world"] == world]:
        spec = dict(cell.spec)
        results = list(mpi_pool_executor.map(run, *zip(*[(spec,)] * world)))
        assert all(r[0] == 0 for r in results), f"{name} {cell.why}: {results}"


# The tail finished by the next mix (the all-gather cells of rubin_attn_res_lamport_add_rmsnorm and
# of its _mx entry): updated rows bit for bit the reference over the push GEMM's plain columns and
# the shared partials, the MXFP8 form (_mx) bit for bit its rule's of the mix's own output; against
# the replicated
# tail (cuBLAS latent_up, the C++ all-reduce of the shared partials) within the A-class band, the
# shared sum itself bit for bit the C++ all-reduce's.
MIN_COSINE, MAX_RELATIVE_L2, MAX_WORDS = 0.9999, 2e-3, 0.01


def _close(a, b):
    x, y = a.float().flatten(), b.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    rel = ((x - y).norm() / (y.norm() + 1e-12)).item()
    return cosine, rel


def _tail_mix_rank(spec):
    try:
        import importlib

        from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

        c = importlib.import_module(f"{_M}.ops.moe._rubin_ull_common")
        k = importlib.import_module(f"{_M}.kernels.moe.rubin_ull_push_gemm")
        base = importlib.import_module(f"{_M}.kernels.spec.rubin_draft_gemm_push")
        lamport = importlib.import_module(f"{_M}.kernels.comm.rubin_lamport_ar")
        cp_mod = importlib.import_module(f"{_M}.ops.moe.rubin_ull_cp_tail")
        norm = importlib.import_module(f"{_M}.ops.norm.rubin_attn_res_lamport_add_rmsnorm")
        ncommon = importlib.import_module(f"{_M}.ops.norm._rubin_norm_common")
        ccommon = importlib.import_module(f"{_M}.ops.comm._rubin_comm_common")
        moe_mix = importlib.import_module(f"{_M}.ops.norm.rubin_attn_res_moe_add_rmsnorm")
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        mapping = _setup(rank, world)
        m, n_cand = spec["rows"], spec["candidates"]
        built = cp_mod.rubin_ull_tail_workspace(mapping, world, m)
        ws, chunk = built["ws"], built["chunk"]
        c0, n, tactics = cp_mod.rubin_ull_cp_tail_share(world, rank, m)
        g = torch.Generator(device="cuda").manual_seed(9 + m)

        def rn(*shape, sc=1.0):
            return (torch.randn(*shape, device="cuda", generator=g) * sc).bfloat16()

        latent = rn(m, LATENT)
        up = rn(HIDDEN, LATENT, sc=LATENT**-0.5)
        prefix, bank = rn(m, HIDDEN, sc=2.0), rn(n_cand - 1, m, HIDDEN)
        weights = (
            rn(HIDDEN, sc=0.05),
            (1.0 + rn(HIDDEN, sc=0.1).float()).bfloat16(),
            (1.0 + rn(HIDDEN, sc=0.1).float()).bfloat16(),
        )
        shared = []
        for r in range(world):
            gr = torch.Generator(device="cuda").manual_seed(70 + r)
            shared.append((torch.randn(m, HIDDEN, device="cuda", generator=gr) * 0.5).bfloat16())
        mx = "mx_atom" in spec
        entry = (
            norm.rubin_attn_res_lamport_add_rmsnorm_mx
            if mx
            else norm.rubin_attn_res_lamport_add_rmsnorm
        )
        mx_args = (spec["mx_atom"], 1, spec["mx_exact"]) if mx else ()
        args = (
            prefix, ws.uc_local, ws.flags.data_ptr(), bank, *weights, 1e-5, 1e-5,
            world, chunk, False, ws.mc_base, rank,
            spec["add_rn"], spec["early_poll"], False, spec["finish_ag"], *mx_args,
        )  # fmt: skip
        # A block boundary's mix (mix_bank_in_mix) stores the updated prefix into a bank row.
        bank_rows = torch.zeros(2, m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        entry.is_valid(*args, early_trigger=spec["early_trigger"])
        bad, out = {}, None
        for call in range(2):  # two calls: the second one runs on the next buffer
            lamport.run_lamport_push(ws, shared[rank], True)
            _entry("rubin_ull_cp_tail")(latent, up[c0 : c0 + n], ws, tactics[m], c0=c0)
            if spec.get("updated_out"):
                row = bank_rows[call]
                out = entry(*args, early_trigger=spec["early_trigger"], updated=row)
                bad[f"bank_row{call}"] = int(out[0].data_ptr() != row.data_ptr())
            else:
                out = entry(*args, early_trigger=spec["early_trigger"])
            torch.cuda.synchronize()
            y = _plain_ag_rows(c, k, base, latent, up, world, tactics[m])
            ref = entry.reference(
                prefix, shared, bank, *weights, 1e-5, 1e-5, chunk, spec["add_rn"], y, *mx_args,
            )  # fmt: skip
            bad[f"updated{call}"] = int(
                (out[0].view(torch.int16) != ref[0].view(torch.int16)).sum()
            )
            cos, rel = _close(out[1], ref[1])
            bad[f"output{call}"] = int(not (cos > MIN_COSINE and rel < MAX_RELATIVE_L2))
            if not mx:
                continue
            # The MXFP8 form of the mix's own output, in the cell's rule and layout.
            form = ncommon.mxfp8_exact_reference if spec["mx_exact"] else ncommon.mxfp8_reference
            xq, xs = form(out[1], spec["mx_atom"], 1)
            bad[f"codes{call}"] = int((out[2].view(torch.uint8) != xq.view(torch.uint8)).sum())
            if spec["mx_atom"]:
                row = torch.arange(m, device="cuda").view(m, 1)
                grp = torch.arange(HIDDEN // 32, device="cuda").view(1, -1)
                off = (
                    (grp // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + grp % 4
                ).flatten()
                bad[f"scales{call}"] = int((out[3][off] != xs[off]).sum())
            else:
                bad[f"scales{call}"] = int((out[3] != xs).sum())
        # Against the replicated tail: cuBLAS latent_up, the C++ all-reduce of the shared partials,
        # the PDL MoE mix (the same kernel's mix and norm).
        need = max(
            lamport.required_bytes(m, HIDDEN, world, True),
            lamport.required_bytes(m, HIDDEN, world, False),
        )
        get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16, buffer_size_bytes=need)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        shared_cpp = ar(shared[rank])
        shared_ref = ccommon.lamport_allreduce_reference(shared, chunk)
        y_cublas = latent @ up.t()
        replicated = moe_mix.rubin_attn_res_moe_add_rmsnorm(
            prefix, y_cublas, shared_cpp, bank, *weights, 1e-5, 1e-5, early_trigger=True
        )
        torch.cuda.synchronize()
        bad["shared_vs_cpp"] = int(
            (shared_cpp.view(torch.int16) != shared_ref.view(torch.int16)).sum()
        )
        cos, rel = _close(out[0], replicated[0])
        words = float((out[0].view(torch.int16) != replicated[0].view(torch.int16)).float().mean())
        y_words = float((y.view(torch.int16) != y_cublas.view(torch.int16)).float().mean())
        bad["vs_replicated"] = int(
            not (cos > MIN_COSINE and rel < MAX_RELATIVE_L2 and words < MAX_WORDS)
        )
        stats = dict(
            updated_cosine=cos, updated_rel_l2=rel, updated_words=words, latent_up_words=y_words
        )
        result = (sum(bad.values()), bad, stats)
    except Exception:
        raise _failure(spec) from None
    return result


def _tail_mix_cells(world):
    norm = __import__(
        f"{_M}.ops.norm.rubin_attn_res_lamport_add_rmsnorm",
        fromlist=["rubin_attn_res_lamport_add_rmsnorm", "rubin_attn_res_lamport_add_rmsnorm_mx"],
    )
    cells = (
        *norm.rubin_attn_res_lamport_add_rmsnorm.CELLS,
        *norm.rubin_attn_res_lamport_add_rmsnorm_mx.CELLS,
    )
    return [cell for cell in cells if cell.spec["world"] == world and cell.spec.get("finish_ag")]


@pytest.mark.parametrize("mpi_pool_executor", [w for w in _WORLDS if w >= 4], indirect=True)
def test_tail_finished_by_the_mix(mpi_pool_executor):
    """The all-gather cells of rubin_attn_res_lamport_add_rmsnorm and of its ``_mx`` entry
    (``.CELLS``) over the column-parallel tail's call: see the comment above ``_tail_mix_rank``."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    cells = _tail_mix_cells(world)
    assert cells, f"no all-gather cell at world {world}"
    for cell in cells:
        spec = dict(cell.spec)
        results = list(mpi_pool_executor.map(_tail_mix_rank, *zip(*[(spec,)] * world)))
        print(f"{cell.why}: {[r[2] for r in results]}")
        assert all(r[0] == 0 for r in results), f"{cell.why}: {results}"


def _replace_rank(spec):
    """A block boundary's mix after attention (``replace``): every rank pushes its partial (the
    copy push; o_proj's epilogue push lands the same words, test_rubin_mxfp8_linear_push.py), the
    mix finishes the call as the updated prefix. Then today's pair on the same workspace: the C++
    all-reduce and rubin_attn_res_rmsnorm of its result."""
    try:
        import importlib

        from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

        lamport = importlib.import_module(f"{_M}.kernels.comm.rubin_lamport_ar")
        norm = importlib.import_module(f"{_M}.ops.norm.rubin_attn_res_lamport_add_rmsnorm")
        plain = importlib.import_module(f"{_M}.ops.norm.rubin_attn_res_rmsnorm")
        push = importlib.import_module(f"{_M}.ops.comm.rubin_lamport_push")
        ccommon = importlib.import_module(f"{_M}.ops.comm._rubin_comm_common")
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        mapping = _setup(rank, world)
        m, n_cand, two_shot = spec["rows"], spec["candidates"], spec["two_shot"]
        need = max(
            lamport.required_bytes(m, HIDDEN, world, True),
            lamport.required_bytes(m, HIDDEN, world, False),
        )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            mapping, torch.bfloat16, buffer_size_bytes=need
        )
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, world, rank)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(31 + m + n_cand)

        def rn(*shape, sc=1.0):
            return (torch.randn(*shape, device="cuda", generator=g) * sc).bfloat16()

        stale, bank = rn(m, HIDDEN, sc=2.0), rn(n_cand - 1, m, HIDDEN)
        weights = (
            rn(HIDDEN, sc=0.05),
            (1.0 + rn(HIDDEN, sc=0.1).float()).bfloat16(),
            (1.0 + rn(HIDDEN, sc=0.1).float()).bfloat16(),
        )
        partials = []
        for r in range(world):
            gr = torch.Generator(device="cuda").manual_seed(90 + r)
            partials.append((torch.randn(m, HIDDEN, device="cuda", generator=gr) * 0.5).bfloat16())
        # Every world here sums the 8 (or 4) partials in one chunk on either C++ transport.
        chunk = ccommon.reduce_chunk(world, not two_shot)
        entry = norm.rubin_attn_res_lamport_add_rmsnorm
        args = (
            stale, ws.uc_local, ws.flags.data_ptr(), bank, *weights, 1e-5, 1e-5,
            world, chunk, two_shot, ws.mc_base, rank,
            False, spec["early_poll"], False, False,
        )  # fmt: skip
        out_buf = torch.empty_like(stale) if spec.get("updated_out") else None
        kw = dict(early_trigger=spec["early_trigger"], updated=out_buf, replace=True)
        entry.is_valid(*args, **kw)
        bad = {}
        for call in range(2):  # two calls: the second one runs on the next buffer
            push.rubin_lamport_push(ws, partials[rank], not two_shot, True)
            out = entry(*args, **kw)
            cpp = ar(partials[rank])
            twin = plain.rubin_attn_res_rmsnorm(
                cpp, bank, *weights, 1e-5, 1e-5, early_trigger=spec["early_trigger"]
            )
            ref = entry.reference(
                stale, partials, bank, *weights, 1e-5, 1e-5, chunk, False, None, replace=True
            )
            torch.cuda.synchronize()
            bad[f"updated{call}"] = int(not _bits(out[0], ref[0]))
            bad[f"cpp{call}"] = int(not _bits(cpp, ref[0]))
            bad[f"twin{call}"] = int(not (_bits(out[0], cpp) and _bits(out[1], twin)))
            cos, rel = _close(out[1], ref[1])
            bad[f"output{call}"] = int(not (cos > MIN_COSINE and rel < MAX_RELATIVE_L2))
            if out_buf is not None:
                bad[f"buffer{call}"] = int(out[0] is not out_buf)
        result = (sum(bad.values()), bad, {})
    except Exception:
        raise _failure(spec) from None
    return result


@pytest.mark.parametrize("mpi_pool_executor", [w for w in _WORLDS if w >= 8], indirect=True)
def test_boundary_replace_cells(mpi_pool_executor):
    """The 8-rank ``replace`` cells of rubin_attn_res_lamport_add_rmsnorm (``.CELLS``): see
    ``_replace_rank``."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    norm = __import__(
        f"{_M}.ops.norm.rubin_attn_res_lamport_add_rmsnorm",
        fromlist=["rubin_attn_res_lamport_add_rmsnorm"],
    )
    cells = [
        c
        for c in norm.rubin_attn_res_lamport_add_rmsnorm.CELLS
        if c.spec["world"] == world and c.spec.get("replace")
    ]
    assert cells, f"no replace cell at world {world}"
    for cell in cells:
        spec = dict(cell.spec)
        results = list(mpi_pool_executor.map(_replace_rank, *zip(*[(spec,)] * world)))
        assert all(r[0] == 0 for r in results), f"{cell.why}: {results}"


# One tail workspace over a decode sequence, as a helix8 GEN rank sees it: in-band calls of
# varying rows (the workspace sized to the bands' largest), each finished by the BF16 PDL mix (an
# untapped layer's moe_norm) or by the finish kernel (a tapped layer's moe_add), between
# out-of-band steps that run the C++ all-reduce and leave the tail's workspace alone. The three
# buffers rotate across calls of different rows, so a call reads only what its own pushes wrote.
_SEQUENCE = (
    (24, "mix"),
    (6, "finish"),
    (None, None),
    (30, "mix"),
    (12, "finish"),
    (None, None),
    (30, "finish"),
    (18, "mix"),
    (6, "mix"),
    (24, "finish"),
)
# The same over the tail's widest bands (1-72 rows: helix8's b1 to b12 k5 verify steps, the push
# GEMM in N tiles of 32 past 32 rows), with out-of-band steps of 78 rows (b13) between them.
_WIDE_SEQUENCE = (
    (66, "mix"),
    (36, "finish"),
    (None, None),
    (72, "mix"),
    (48, "finish"),
    (54, "mix"),
    (6, "mix"),
    (None, None),
    (60, "finish"),
    (42, "mix"),
    (72, "finish"),
    (30, "mix"),
)


def _tail_sequence_rank(spec):
    try:
        import importlib

        from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

        c = importlib.import_module(f"{_M}.ops.moe._rubin_ull_common")
        k = importlib.import_module(f"{_M}.kernels.moe.rubin_ull_push_gemm")
        base = importlib.import_module(f"{_M}.kernels.spec.rubin_draft_gemm_push")
        lamport = importlib.import_module(f"{_M}.kernels.comm.rubin_lamport_ar")
        cp_mod = importlib.import_module(f"{_M}.ops.moe.rubin_ull_cp_tail")
        norm = importlib.import_module(f"{_M}.ops.norm.rubin_attn_res_lamport_add_rmsnorm")
        push = importlib.import_module(f"{_M}.ops.comm.rubin_lamport_push")
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        mapping = _setup(rank, world)
        top, other = spec["max_rows"], spec["other_rows"]
        built = cp_mod.rubin_ull_tail_workspace(mapping, world, top)
        ws, chunk, cpp_one_shot = built["ws"], built["chunk"], built["cpp_one_shot"]
        c0, n, tactics = cp_mod.rubin_ull_cp_tail_share(world, rank, top)
        need = max(
            lamport.required_bytes(other, HIDDEN, world, True),
            lamport.required_bytes(other, HIDDEN, world, False),
        )
        get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16, buffer_size_bytes=need)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(5)

        def rn(gen, *shape, sc=1.0):
            return (torch.randn(*shape, device="cuda", generator=gen) * sc).bfloat16()

        up = rn(g, HIDDEN, LATENT, sc=LATENT**-0.5)
        weights = (
            rn(g, HIDDEN, sc=0.05),
            (1.0 + rn(g, HIDDEN, sc=0.1).float()).bfloat16(),
            (1.0 + rn(g, HIDDEN, sc=0.1).float()).bfloat16(),
        )
        mix = norm.rubin_attn_res_lamport_add_rmsnorm
        bad = {}
        for step, (m, kind) in enumerate(spec["sequence"]):
            gs = torch.Generator(device="cuda").manual_seed(100 + step)  # the same on every rank
            if m is None:
                part = rn(
                    torch.Generator(device="cuda").manual_seed(200 + 8 * step + rank), other, HIDDEN
                )
                ar(part)
                continue
            latent, prefix, bank = (
                rn(gs, m, LATENT),
                rn(gs, m, HIDDEN, sc=2.0),
                rn(gs, 4, m, HIDDEN),
            )
            shared = [
                rn(
                    torch.Generator(device="cuda").manual_seed(300 + 8 * step + r),
                    m,
                    HIDDEN,
                    sc=0.5,
                )
                for r in range(world)
            ]
            lamport.run_lamport_push(ws, shared[rank], True)
            _entry("rubin_ull_cp_tail")(latent, up[c0 : c0 + n], ws, tactics[m], c0=c0)
            if kind == "mix":
                out = mix(
                    prefix, ws.uc_local, ws.flags.data_ptr(), bank, *weights, 1e-5, 1e-5,
                    world, chunk, False, ws.mc_base, rank, True, True, False, True,
                    early_trigger=True,
                )  # fmt: skip
            else:
                got = push.rubin_lamport_finish(
                    ws, torch.empty_like(prefix), True, True, False, cpp_one_shot, True
                )
            torch.cuda.synchronize()
            y = _plain_ag_rows(c, k, base, latent, up, world, tactics[m])
            if kind == "mix":
                ref = mix.reference(prefix, shared, bank, *weights, 1e-5, 1e-5, chunk, True, y)
                bad[f"updated{step}"] = int(
                    (out[0].view(torch.int16) != ref[0].view(torch.int16)).sum()
                )
                cos, rel = _close(out[1], ref[1])
                bad[f"output{step}"] = int(not (cos > MIN_COSINE and rel < MAX_RELATIVE_L2))
            else:
                want = _ag_finish(c, y.cpu(), [s.cpu() for s in shared], cpp_one_shot)
                bad[f"finish{step}"] = int(not _bits(got.cpu(), want))
        result = (sum(bad.values()), bad)
    except Exception:
        raise _failure(spec) from None
    return result


@pytest.mark.parametrize("mpi_pool_executor", [w for w in _WORLDS if w >= 4], indirect=True)
def test_one_tail_workspace_serves_varying_rows_and_both_finishes(mpi_pool_executor):
    """See the comment above ``_SEQUENCE``: helix8's tail bands (1-30 rows) with out-of-band steps
    of 54 rows (b9) between them."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    spec = dict(world=world, max_rows=30, other_rows=54, sequence=_SEQUENCE)
    results = list(mpi_pool_executor.map(_tail_sequence_rank, *zip(*[(spec,)] * world)))
    assert all(r[0] == 0 for r in results), results


@pytest.mark.parametrize("mpi_pool_executor", [w for w in _WORLDS if w >= 4], indirect=True)
def test_one_tail_workspace_serves_its_widest_bands(mpi_pool_executor):
    """See the comments above ``_SEQUENCE`` and ``_WIDE_SEQUENCE``: 6 to 72 rows on one workspace
    sized to 72, with out-of-band steps of 78 rows between them."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    spec = dict(world=world, max_rows=72, other_rows=78, sequence=_WIDE_SEQUENCE)
    results = list(mpi_pool_executor.map(_tail_sequence_rank, *zip(*[(spec,)] * world)))
    assert all(r[0] == 0 for r in results), results
