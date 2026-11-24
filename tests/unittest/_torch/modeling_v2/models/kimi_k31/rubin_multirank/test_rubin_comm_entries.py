# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (four ranks, one VR200 tray) for the K3.1 Rubin comm entries.

Drives the cells of rubin_lamport_push (push, finish), the cells of
rubin_attn_res_lamport_add_rmsnorm and of its ``_mx`` entry (without all-gather rows) that finish
a pushed call (``world == 4``; a ``replace`` cell, where the all-reduce replaces the prefix, also
against the C++ all-reduce followed by rubin_attn_res_rmsnorm, bit for bit), and
rubin_latent_ar_norm_row, plus the runtime helper
rubin_push_allreduce's push workspace. Every rank builds the
engine MNNVL AllReduce on the shared workspace, regenerates every rank's partial from the cell's
seed, and gates its results against the entry's reference over all partials (bit for bit where
the kernel reproduces the C++ all-reduce) and against the C++ all-reduce run on the same
workspace afterwards. Needs the kernels at their family paths.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import os
import pickle
import sys
import traceback
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or torch.cuda.device_count() < WORLD,
        reason="sm_107 kernels on four GPUs (one VR200 tray)",
    ),
]

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_lamport_push import (  # noqa: E402,E501
    rubin_lamport_finish,
    rubin_lamport_push,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_latent_ar_norm import (  # noqa: E402,E501
    rubin_latent_ar_norm_row,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_lamport_add_rmsnorm import (  # noqa: E402,E501
    rubin_attn_res_lamport_add_rmsnorm,
    rubin_attn_res_lamport_add_rmsnorm_mx,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

EPS = 1e-5
MIN_COSINE, MAX_RELATIVE_L2 = 0.9999, 2e-3

LAMPORT_ENTRIES = {
    "push": rubin_lamport_push,
    "finish": rubin_lamport_finish,
}
LAMPORT_CELLS = [(name, c) for name, e in LAMPORT_ENTRIES.items() for c in e.CELLS]
# Each entry's cells of a call without all-gather rows (their all-gather cells, the column-parallel
# MoE tail's, run in test_rubin_ull_bf16_ranks.py over the tail's push GEMM).
ATTN_RES_CELLS = [
    c
    for c in rubin_attn_res_lamport_add_rmsnorm.CELLS
    if c.spec["world"] == WORLD and not c.spec.get("finish_ag")
]
ATTN_RES_MX_CELLS = [
    c
    for c in rubin_attn_res_lamport_add_rmsnorm_mx.CELLS
    if c.spec["world"] == WORLD and not c.spec.get("finish_ag")
]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _setup():
    """This rank's (rank, mapping, MNNVL AllReduce), built once per worker process."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    # Pickled by value, so this function's globals are fresh per call: keep the module in a
    # worker-process registry.
    reg = vars(sys.modules.setdefault("_kimi_k31_comm", types.ModuleType("_kimi_k31_comm")))
    if "allreduce" not in reg:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        reg["mapping"] = mapping
        reg["allreduce"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert reg["allreduce"].mnnvl_allreduce is not None
    return rank, reg["mapping"], reg["allreduce"]


def _workspace(mapping, allreduce, rows: int, hidden: int):
    """The shared MNNVL workspace grown for ``rows`` x ``hidden`` either way, and its view."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm.rubin_lamport_ar import (  # noqa: E501
        LamportWorkspace,
        required_bytes,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

    need = max(
        required_bytes(rows, hidden, WORLD, True),
        required_bytes(rows, hidden, WORLD, False),
        allreduce.mnnvl_allreduce.get_required_workspace_size(rows, hidden, WORLD, torch.bfloat16),
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping, torch.bfloat16, buffer_size_bytes=need
    )
    return workspace, LamportWorkspace.from_mnnvl(workspace, WORLD, mapping.tp_rank)


def _partials(rows: int, hidden: int, seed: int, scale: float = 0.25):
    """Every rank's bf16 partial, in rank order, identical on every rank."""
    out = []
    for r in range(WORLD):
        g = torch.Generator(device="cuda").manual_seed(seed * 1009 + r)
        out.append((torch.randn(rows, hidden, generator=g, device="cuda") * scale).bfloat16())
    return out


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    view = torch.uint8 if a.element_size() == 1 else torch.int16
    return int((a.view(view) != b.view(view)).sum())


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    x, y = a.float().flatten(), b.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    rel = ((x - y).norm() / (y.norm() + 1e-12)).item()
    return cosine > MIN_COSINE and rel < MAX_RELATIVE_L2


def _digest(t: torch.Tensor) -> int:
    bits = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    weights = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64) % 65521
    return int((bits * weights).sum().item())


def _lamport_rank(spec: dict, seed: int) -> dict:
    """push / finish: one call on every rank."""
    try:
        rank, mapping, allreduce = _setup()
        _, ws = _workspace(mapping, allreduce, spec["rows"], spec["hidden"])
        xs = _partials(spec["rows"], spec["hidden"], seed)
        x, out = xs[rank], torch.empty_like(xs[rank])
        one_shot = spec["one_shot"]
        cpp = spec.get("cpp_one_shot")
        rubin_lamport_push.is_valid(ws, x, one_shot, True)
        rubin_lamport_push(ws, x, one_shot, True)
        finish_args = (ws, out, one_shot, True, spec.get("wait_first", False), cpp, False)
        rubin_lamport_finish.is_valid(*finish_args)
        rubin_lamport_finish(*finish_args)
        ref = rubin_lamport_finish.reference(xs, one_shot, cpp)
        # Then the C++ all-reduce on the same workspace (the rotation state must be consistent).
        cpp_out = allreduce(x)
        torch.cuda.synchronize()
        return dict(
            rank=rank,
            ref=_mismatch(out, ref),
            cpp=_mismatch(out, cpp_out),
            digest=_digest(out),
            input_ok=bool(torch.equal(x, _partials(spec["rows"], spec["hidden"], seed)[rank])),
        )
    except Exception:
        traceback.print_exc()
        raise


def _attn_res_rank(spec: dict, seed: int) -> dict:
    """The mix after attention finishing the call every rank pushed."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm._rubin_comm_common import (  # noqa: E501
            lamport_allreduce_reference,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._rubin_norm_common import (  # noqa: E501
            mxfp8_reference,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_rmsnorm import (  # noqa: E501
            rubin_attn_res_add_rmsnorm,
            rubin_attn_res_rmsnorm,
        )

        mx = spec.get("mx_entry", False)
        entry = rubin_attn_res_lamport_add_rmsnorm_mx if mx else rubin_attn_res_lamport_add_rmsnorm
        rank, mapping, allreduce = _setup()
        rows, n, h = spec["rows"], spec["candidates"], 7168
        _, ws = _workspace(mapping, allreduce, rows, h)
        partials = _partials(rows, h, seed)
        g = torch.Generator(device="cuda").manual_seed(seed + 1)

        def rn(*shape, sc=1.0):
            return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

        prefix, block = rn(rows, h), rn(n - 1, rows, h)
        weights = (
            rn(h, sc=0.05),
            (1.0 + rn(h, sc=0.1).float()).to(torch.bfloat16),
            (1.0 + rn(h, sc=0.1).float()).to(torch.bfloat16),
        )
        two_shot, add_rn = spec["two_shot"], spec.get("add_rn", False)
        chunk = WORLD  # up to eight ranks: one partial sum, either C++ kernel
        rubin_lamport_push(ws, partials[rank], not two_shot, True)
        args = (
            prefix, ws.uc_local, ws.flags.data_ptr(), block, *weights, EPS, EPS,
            WORLD, chunk, two_shot, ws.mc_base, rank,
            add_rn, spec["early_poll"], spec.get("entry_trigger", False), False,
        )  # fmt: skip
        if mx:
            args += (False, 1, False)  # row-major scales, the low-M GEMM's rule
        replace = spec.get("replace", False)
        # A caller buffer for the updated prefix (a bank row in helix8's mix_bank_in_mix).
        out_buf = torch.empty_like(prefix) if spec.get("updated_out") else None
        kw = dict(early_trigger=spec["early_trigger"], updated=out_buf, replace=replace)
        entry.is_valid(*args, **kw)
        got = entry(*args, **kw)
        ref = entry.reference(
            prefix, partials, block, *weights, EPS, EPS, chunk, add_rn, None, replace=replace
        )[:2]
        result = dict(rank=rank, updated=_mismatch(got[0], ref[0]), close=_close(got[1], ref[1]))
        if out_buf is not None and got[0] is not out_buf:
            result["updated"] += 1  # the updated prefix must be the caller's buffer
        if not add_rn:
            addend = lamport_allreduce_reference(partials, chunk)
            if replace:
                # Today's boundary: the C++ all-reduce's result, then the mix without an add.
                twin = (
                    addend,
                    rubin_attn_res_rmsnorm(
                        addend, block, *weights, EPS, EPS, early_trigger=spec["early_trigger"]
                    ),
                )
            else:
                twin = rubin_attn_res_add_rmsnorm(
                    prefix, addend, block, *weights, EPS, EPS, early_trigger=spec["early_trigger"]
                )
            result["twin"] = sum(_mismatch(a, b) for a, b in zip(got[:2], twin))
        if len(got) == 4:
            xq, xs = mxfp8_reference(got[1])
            result["mx"] = _mismatch(got[2], xq) + _mismatch(got[3], xs)
        # The workspace went on rotating: the C++ all-reduce of the same partials matches.
        result["cpp"] = _mismatch(
            allreduce(partials[rank]), lamport_allreduce_reference(partials, chunk)
        )
        result["digest"] = _digest(got[1])
        torch.cuda.synchronize()
        return result
    except Exception:
        traceback.print_exc()
        raise


def _latent_rank(spec: dict, seed: int) -> dict:
    """The routed-latent all-reduce + RMSNorm on every rank."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_latent_ar_norm import (  # noqa: E501
            LATENT,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._rubin_norm_common import (  # noqa: E501
            mxfp8_reference,
        )
        from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES
        from tensorrt_llm.functional import AllReduceFusionOp

        rank, mapping, allreduce = _setup()
        rows, norm = spec["rows"], spec["norm"]
        workspace, ws = _workspace(mapping, allreduce, rows, LATENT)
        xs = _partials(rows, LATENT, seed, scale=1.0)
        x = xs[rank]
        g = torch.Generator(device="cuda").manual_seed(seed + 7)
        gamma = (torch.rand(LATENT, generator=g, device="cuda") + 0.5).bfloat16()
        out = torch.empty_like(x)
        sf_blocks = spec.get("sf_blocks", 0)
        mx = None
        if spec["mxfp8"]:
            xs_bytes = (LATENT // 128) * sf_blocks * 512 if sf_blocks else rows * (LATENT // 32)
            mx = (
                torch.empty(x.shape, dtype=torch.float8_e4m3fn, device="cuda"),
                torch.zeros(xs_bytes, dtype=torch.uint8, device="cuda"),
            )
        head = (x, gamma, EPS, out, ws.uc_local, ws.mc_base, ws.flags, rank, WORLD, True)
        tail = (spec["early_trigger"], norm, mx)
        entry = rubin_latent_ar_norm_row
        args = (*head, *tail, sf_blocks, False, spec.get("early_state", False))
        ref = entry.reference(xs, gamma, EPS, norm, spec["mxfp8"], sf_blocks)
        entry.is_valid(*args)
        entry(*args)
        ref_out = ref[0] if spec["mxfp8"] else ref
        result = dict(rank=rank, close=_close(out, ref_out), digest=_digest(out))
        if not norm:
            result["sum"] = _mismatch(out, ref_out)  # the bf16 sum is exact in the reference
        if mx is not None:
            codes, scales = mxfp8_reference(out, sf_blocks > 0, max(sf_blocks, 1))
            if sf_blocks:
                written = scales != 0  # the reference leaves unwritten atom bytes 0
                result["mx"] = _mismatch(mx[0], codes) + int(
                    (mx[1][written] != scales[written]).sum()
                )
            else:
                result["mx"] = _mismatch(mx[0], codes) + _mismatch(mx[1].view(rows, -1), scales)
        if rows * LATENT * WORLD * 2 <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
            # The one-shot sizes: bit for bit the C++ one-shot kernel on the same workspace.
            op = AllReduceFusionOp.RMS_NORM if norm else AllReduceFusionOp.NONE

            # Nested: a torch.ops lookup in this function's own body makes cloudpickle send the
            # torch.ops module with it, which cannot be pickled.
            def cpp():
                (cpp_out,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                    x,
                    gamma if norm else None,
                    None,
                    EPS if norm else None,
                    workspace["uc_buffer"].view(torch.bfloat16).view(3, -1),
                    workspace["buffer_flags"],
                    norm,
                    None,
                    int(op),
                )
                return cpp_out

            result["cpp"] = _mismatch(out, cpp())
        torch.cuda.synchronize()
        return result
    except Exception:
        traceback.print_exc()
        raise


def _helper_rank(seed: int) -> dict:
    """The built push workspace end to end: copy push, then the finish on the call's transport."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
            rubin_push_allreduce as push,
        )

        rank, mapping, allreduce = _setup()
        bad = 0
        built = push.rubin_lamport_push_workspace(allreduce, 96, 7168)
        assert built is not None
        for rows in (1, 8, 96):
            x = _partials(rows, 7168, seed + rows)[rank]
            expect = allreduce(x)
            one_shot, _ = built.transport(rows)
            cpp_one_shot = rows * 7168 * WORLD * 2 <= push._MNNVL_ONE_SHOT_THRESHOLD_BYTES
            rubin_lamport_push(built.view, x, one_shot, built.pdl)
            got = rubin_lamport_finish(
                built.view, torch.empty_like(x), one_shot, built.pdl, False, cpp_one_shot, False
            )
            bad += _mismatch(got, expect)
        torch.cuda.synchronize()
        return dict(rank=rank, bad=bad)
    except Exception:
        traceback.print_exc()
        raise


def _run(executor, fn, *args):
    results = list(executor.map(fn, *zip(*[args] * WORLD)))
    return sorted(results, key=lambda r: r["rank"])


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "name,cell", LAMPORT_CELLS, ids=[f"{n}:{c.why[:48]}" for n, c in LAMPORT_CELLS]
)
def test_lamport_push_cells(mpi_pool_executor, name, cell) -> None:
    """Every cell of rubin_lamport_push's two entries (``rubin_lamport_push.CELLS`` and
    ``rubin_lamport_finish.CELLS``): bit for bit the reference and the C++ all-reduce, the same
    bits on every rank."""
    results = _run(mpi_pool_executor, _lamport_rank, cell.spec, _seed(cell))
    for r in results:
        assert r["ref"] == 0 and r["cpp"] == 0 and r["input_ok"], r
    assert len({r["digest"] for r in results}) == 1


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", ATTN_RES_CELLS, ids=[c.why[:48] for c in ATTN_RES_CELLS])
def test_attn_res_lamport_finish_cells(mpi_pool_executor, cell) -> None:
    """The ``world == 4`` cells of rubin_attn_res_lamport_add_rmsnorm (``.CELLS``)."""
    results = _run(mpi_pool_executor, _attn_res_rank, cell.spec, _seed(cell))
    for r in results:
        assert r["updated"] == 0 and r["close"] and r["cpp"] == 0, r
        assert r.get("twin", 0) == 0 and r.get("mx", 0) == 0, r
    assert len({r["digest"] for r in results}) == 1


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", ATTN_RES_MX_CELLS, ids=[c.why[:48] for c in ATTN_RES_MX_CELLS])
def test_attn_res_lamport_mx_finish_cells(mpi_pool_executor, cell) -> None:
    """The ``world == 4`` cells of rubin_attn_res_lamport_add_rmsnorm_mx (``.CELLS``) without
    all-gather rows: the plain entry's results and the MXFP8 form of its output, bit for bit."""
    spec = dict(cell.spec, mx_entry=True)
    results = _run(mpi_pool_executor, _attn_res_rank, spec, _seed(cell))
    for r in results:
        assert r["updated"] == 0 and r["close"] and r["cpp"] == 0, r
        assert r.get("twin", 0) == 0 and r.get("mx", -1) == 0, r
    assert len({r["digest"] for r in results}) == 1


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", rubin_latent_ar_norm_row.CELLS, ids=[c.why[:48] for c in rubin_latent_ar_norm_row.CELLS]
)
def test_latent_ar_norm_cells(mpi_pool_executor, cell) -> None:
    """Every cell of rubin_latent_ar_norm_row (``.CELLS``)."""
    results = _run(mpi_pool_executor, _latent_rank, cell.spec, _seed(cell))
    for r in results:
        assert r["close"] and r.get("sum", 0) == 0 and r.get("mx", 0) == 0, r
        assert r.get("cpp", 0) == 0, r
    assert len({r["digest"] for r in results}) == 1


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_push_allreduce_helper(mpi_pool_executor) -> None:
    for r in _run(mpi_pool_executor, _helper_rank, 17):
        assert r["bad"] == 0, r
