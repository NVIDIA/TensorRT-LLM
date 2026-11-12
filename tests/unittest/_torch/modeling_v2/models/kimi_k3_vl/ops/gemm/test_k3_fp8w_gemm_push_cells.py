# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The collective FP8-weight GEMM entries of the K3 default target over their certified cells, on a
4-rank MNNVL group: ``k3_fp8w_lowm_gemm_push.CELLS``, ``k3_fp8w_mid_gemm_push.CELLS`` and
``k3_fp8w_lowm_mxfp8_push.CELLS``.

Every rank draws every rank's inputs from the same seeds, so each holds the group's inputs in rank
order and checks its own ``k3_lamport_ar.run_lamport_finish`` output against the entry's collective
``reference``. Each rank pushes under ``validating``; a cell that only SM107 admits (``sm_107`` in
its spec) is skipped elsewhere and reported.

Band: each rank's BF16 GEMM output can be one BF16 ulp off the reference's (see
``test_k3_fp8w_gemm_cells.ELEMENT_ULP``) and the finish sums four such parts in FP32 before its
own rounding; four one-ulp parts plus the rounding of the sum stay within 8 units of ``ULP``
(2^-8) of the row's largest element, and the RMS within 2.
"""

import os
import pickle
import sys
import traceback
from pathlib import Path

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
#: tests/unittest/_torch/modeling_v2, where ``_validating`` lives (imported by path in the ranks).
MV2_TESTS = str(Path(__file__).resolve().parents[4])
WORLD = 4
#: One Lamport buffer: the largest cell (128 x 7168 two-shot over 4 ranks) needs 3.7 MB.
BUFFER_BYTES = 8 << 20
ELEMENT_ULP, RMS_ULP = 8, 2
ENTRIES = ("k3_fp8w_lowm_gemm_push", "k3_fp8w_mid_gemm_push", "k3_fp8w_lowm_mxfp8_push")


def _weight(n, k, seed):
    """Random E4M3 ``[n, k]`` codes and UE8M0 block exponents ``uint8 [ceil(n/128), k/128]``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    exps = torch.randint(119, 125, (-(-n // 128), k // 128), device="cuda", generator=gen)
    return w, exps.to(torch.uint8)


def _rows(m, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=gen)
    mag = torch.randint(-6, 3, (m, k // 32), device="cuda", generator=gen).float()
    return (x * torch.exp2(mag).repeat_interleave(32, dim=1)).to(torch.bfloat16)


def _bf16(m, n, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, n, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def _run_rank(entry_name, world, mv2_tests):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import importlib

    sys.path.insert(0, mv2_tests)
    from _validating import assert_within_ulp, validating

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.mapping import Mapping

    lowm = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
    mx = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_mxfp8")
    lamport = importlib.import_module(f"{K3}.kernels.k3r_comm.k3_lamport_ar")
    op = getattr(importlib.import_module(f"{K3}.ops.gemm.{entry_name}"), entry_name)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    sm107 = torch.cuda.get_device_capability() == (10, 7)
    report = {"failed": [], "skipped": [], "passed": 0}
    try:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16, BUFFER_BYTES)
        ws = lamport.LamportWorkspace.from_mnnvl(workspace, world, rank)

        def finish(m, n, one_shot, ag=False):
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            lamport.run_lamport_finish(ws, out, one_shot, ag=ag)
            torch.cuda.synchronize()
            return out

        def lowm_push(i, s):
            n, k, m, one_shot = s["n"], s["k"], s["m"], s["one_shot"]
            strided = s.get("strided", False)
            k_total = s.get("k_total", k)
            xs, ws_, scales, gates, addends = [], [], [], [], []
            for r in range(world):
                x_full = _rows(m, k_total, 100 * i + r)
                w_full, exps = _weight(n, k_total, 200 * i + r)
                lo = r * k % k_total if strided else 0
                xs.append(x_full[:, lo : lo + k])
                ws_.append(w_full[:, lo : lo + k])
                scales.append(
                    lowm.pack_ue8m0_exponents(
                        exps[:, lo // 128 : (lo + k) // 128].contiguous(), n, k
                    )
                )
                gates.append(_bf16(m, k, 300 * i + r).sigmoid() if s.get("gate") else None)
                addends.append(_bf16(m, n, 400 * i + r) if s.get("addend") else None)
            kwargs = {"strided": strided}
            if s.get("gate"):
                kwargs["gate"] = gates[rank]
            if s.get("addend"):
                kwargs["addend"] = addends[rank]
            with validating(op):
                op(xs[rank], ws_[rank], scales[rank], ws, one_shot, **kwargs)
            got = finish(m, n, one_shot)
            if s.get("gate"):
                kwargs["gate"] = gates
            if s.get("addend"):
                kwargs["addend"] = addends
            return got, op.reference(xs, ws_, scales, ws, one_shot, **kwargs)

        def mid_push(i, s):
            n, k, m, one_shot = s["n"], s["k"], s["m"], s["one_shot"]
            tactic = lowm.mid_ar_push_tactic(m, n, k)
            assert tactic is not None, "a mid-M push cell must be above the low-M cut-over"
            xqs, xss, ws_, scales = [], [], [], []
            for r in range(world):
                xq, xs = lowm.quantize_mid_operand(_rows(m, k, 500 * i + r), tactic)
                w, exps = _weight(n, k, 600 * i + r)
                xqs.append(xq)
                xss.append(xs)
                ws_.append(w)
                scales.append(lowm.pack_ue8m0_exponents(exps, n, k))
            with validating(op):
                op(xqs[rank], xss[rank], ws_[rank], scales[rank], ws, one_shot, tactic)
            got = finish(m, n, one_shot)
            return got, op.reference(xqs, xss, ws_, scales, ws, one_shot, tactic)

        def mxfp8_push(i, s):
            variant, k, m = s["variant"], s["k"], s["m"]
            if variant == "ag":
                hidden = s["hidden"]
                n = hidden // world
                # The column-parallel tail: the same latent rows on every rank.
                xq, xs = mx.mxfp8_quantize_reference(_rows(m, k, 700 * i))
                w_full, exps = _weight(hidden, k, 800 * i)
                col0s = [r * n for r in range(world)]
                ws_ = [w_full[c : c + n].contiguous() for c in col0s]
                scales = [
                    lowm.pack_ue8m0_exponents(exps[c // 128 : (c + n) // 128].contiguous(), n, k)
                    for c in col0s
                ]
                tactic = mx.default_mx_tactic(m, hidden, k)
                zeros = torch.zeros(m, hidden, dtype=torch.bfloat16, device="cuda")
                lamport.run_lamport_push(ws, zeros, True)
                with validating(op):
                    op(
                        xq,
                        xs,
                        ws_[rank],
                        scales[rank],
                        ws,
                        col0=col0s[rank],
                        hidden=hidden,
                        tactic=tactic,
                    )
                got = finish(m, hidden, True, ag=True)
                want = op.reference(
                    [xq] * world,
                    [xs] * world,
                    ws_,
                    scales,
                    ws,
                    col0=col0s,
                    hidden=hidden,
                    tactic=tactic,
                )
                return got, want
            n = s["n"]
            one_shot = s.get("one_shot", True)
            k_total = s.get("k_total", k)
            xqs, xss, ws_, scales, addends, k0s = [], [], [], [], [], []
            for r in range(world):
                xq, xs = mx.mxfp8_quantize_reference(_rows(m, k_total, 900 * i + r))
                w, exps = _weight(n, k_total, 1000 * i + r)
                lo = r * k % k_total
                xqs.append(xq)
                xss.append(xs)
                ws_.append(w)
                k0s.append(lo)
                if variant == "slice":
                    sl = exps[:, lo // 128 : (lo + k) // 128].contiguous()
                    scales.append(lowm.pack_ue8m0_exponents(sl, n, k))
                    addends.append(_bf16(m, n, 1100 * i + r))
                else:
                    scales.append(lowm.pack_ue8m0_exponents(exps, n, k))
            if variant == "slice":
                with validating(op):
                    op(
                        xqs[rank],
                        xss[rank],
                        ws_[rank],
                        scales[rank],
                        ws,
                        one_shot,
                        k0=k0s[rank],
                        k1=k,
                        addend=addends[rank],
                    )
                got = finish(m, n, one_shot)
                want = op.reference(
                    xqs, xss, ws_, scales, ws, one_shot, k0=k0s, k1=k, addend=addends
                )
                return got, want
            with validating(op):
                op(xqs[rank], xss[rank], ws_[rank], scales[rank], ws, one_shot)
            got = finish(m, n, one_shot)
            return got, op.reference(xqs, xss, ws_, scales, ws, one_shot)

        run = {
            "k3_fp8w_lowm_gemm_push": lowm_push,
            "k3_fp8w_mid_gemm_push": mid_push,
            "k3_fp8w_lowm_mxfp8_push": mxfp8_push,
        }[entry_name]
        for i, cell in enumerate(op.CELLS):
            if cell.spec.get("sm_107") and not sm107:
                report["skipped"].append(cell.why)
                continue
            comm.Barrier()
            got, want = run(i, cell.spec)
            try:
                assert_within_ulp(got, want, ELEMENT_ULP, RMS_ULP)
                report["passed"] += 1
            except AssertionError as e:
                report["failed"].append(f"{cell.why}: {e}")
    except Exception:
        traceback.print_exc()
        raise
    return report


@skip_pre_blackwell
@pytest.mark.parametrize("entry_name", ENTRIES)
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_fp8w_gemm_push_cells(mpi_pool_executor, entry_name):
    """Drives ``k3_fp8w_lowm_gemm_push.CELLS``, ``k3_fp8w_mid_gemm_push.CELLS`` or
    ``k3_fp8w_lowm_mxfp8_push.CELLS`` on every rank."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    args = [(entry_name, world, MV2_TESTS)] * world
    reports = list(mpi_pool_executor.map(_run_rank, *zip(*args)))
    failed = [f"rank {r}: {msg}" for r, rep in enumerate(reports) for msg in rep["failed"]]
    assert not failed, "\n".join(failed)
    assert all(rep["passed"] for rep in reports), reports
    skipped = reports[0]["skipped"]
    if skipped:
        print(f"{entry_name}: skipped off SM107: {skipped}")
