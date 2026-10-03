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
"""The Kimi K3 decode collectives on the MNNVL all-reduce workspace, one process per GPU over the TP group of this
run (4 on one GB200 tray), at every M in 1..8 and at 16, 32, 64 tokens:
  trtllm::mnnvl_allreduce_attn_res (MNNVLAllReduce.allreduce_attn_res_rmsnorm): against the unfused path it replaces,
    the MNNVL all-reduce then trtllm::attn_res_add_rmsnorm_fwd (attn_res_rmsnorm_fwd without a prefix sum): the
    updated prefix sum bit for bit, the normed rows within 1e-2 (max |d| / max |ref|), both against an fp32 port of the
    attention-residual selection within 2e-2; 0, 1, 3, 8 and 11 snapshots, with and without the prefix;
each with run-to-run identical bits, the same bits on every rank, each M's rows bit-identical to the same rows of the
64-row call, and one rank's perturbed input changing every rank's result.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_mnnvl_comm.py [attn_res]
"""

import hashlib
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
H = 7168
EPS, OUT_EPS = 1e-6, 1e-5
M_CASES = list(range(1, 9)) + [16, 32, 64]
M_MAX = 64
SNAPSHOTS = (0, 1, 3, 8, 11)


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} SM100 GPUs with MNNVL and mpi4py"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _rel(a, b) -> float:
    """max |a - b| / max |b|: normalized like the GEMV checks (a 2-ulp rounding difference at the tail of a large M
    is within it; a wrong row is not)."""
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _context():
    """One process per GPU, ranks filling the nodes in order (gpus_per_node = the node's GPU count, so a rank's
    local_rank is its device on every node); the multicast buffers use fabric handles within a tray as across trays."""
    os.environ.setdefault("TRTLLM_FORCE_MNNVL_AR", "1")
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    return SimpleNamespace(
        comm=comm, rank=rank, world=world, mnnvl=MNNVLAllReduce(mapping, torch.bfloat16)
    )


def _allreduce(ctx, x):
    """The plain MNNVL all-reduce. Up to 8 ranks its one-shot and two-shot kernels both sum the ranks in rank order in
    fp32, the fused op's order, so the reference is exact whichever one the size picks."""
    from tensorrt_llm._torch.distributed import AllReduceParams

    assert ctx.world <= 8, "above 8 ranks the two-shot kernel sums the ranks in another order"
    return ctx.mnnvl(x, AllReduceParams())


def _all_ranks(ctx, good) -> bool:
    return all(ctx.comm.allgather(bool(good)))


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(_bits(t).cpu().numpy().tobytes()).hexdigest()


def _attn_res_inputs(ctx, snapshots, with_prefix):
    shared = torch.Generator(device="cuda").manual_seed(1000 + 17 * snapshots + int(with_prefix))
    prefix = (
        torch.randn(M_MAX, H, generator=shared, device="cuda").bfloat16() if with_prefix else None
    )
    scale = (1 + torch.arange(snapshots, device="cuda")).view(-1, 1, 1)
    block = (torch.randn(snapshots, M_MAX, H, generator=shared, device="cuda") * scale).bfloat16()
    res_w = (torch.randn(H, generator=shared, device="cuda") * 0.05).bfloat16()
    rms_w = (1 + 0.1 * torch.randn(H, generator=shared, device="cuda")).bfloat16()
    out_w = (1 + 0.1 * torch.randn(H, generator=shared, device="cuda")).bfloat16()
    own = torch.Generator(device="cuda").manual_seed(7 * (1000 + snapshots) + ctx.rank + 1)
    partial = (torch.randn(M_MAX, H, generator=own, device="cuda") * 0.5).bfloat16()
    return partial, prefix, block, res_w, rms_w, out_w


def _fp32_reference(updated, block, res_w, rms_w, out_w):
    """HF _apply_attn_res + RMSNorm in fp32 over [snapshots..., updated]."""
    v = torch.cat([block.float(), updated.float().unsqueeze(0)], 0)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    probs = (k * (rms_w.float() * res_w.float())).sum(-1).softmax(0)
    mixed = (probs.unsqueeze(-1) * v).sum(0).bfloat16().float()
    normalized = (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + OUT_EPS)).bfloat16()
    return (normalized.float() * out_w.float()).bfloat16()


def _unfused(ctx, partial, prefix, block, res_w, rms_w, out_w):
    """MNNVL all-reduce, then attn_res_add_rmsnorm_fwd (attn_res_rmsnorm_fwd at a block start); (normed, updated).
    With no snapshot (one candidate) the selection is the identity: normed is None (compared with fp32 only)."""
    m, s = partial.shape[0], block.shape[0]
    reduced = _allreduce(ctx, partial)
    if s == 0:
        return None, (reduced if prefix is None else (prefix.float() + reduced.float()).bfloat16())
    if prefix is None:
        out = torch.ops.trtllm.attn_res_rmsnorm_fwd(reduced.reshape(m, 1, H), block.reshape(s, m, 1, H), res_w, rms_w,
                                                    out_w, EPS, OUT_EPS)  # fmt: skip
        return out.reshape(m, H), reduced
    updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(prefix.reshape(m, 1, H), reduced.reshape(m, 1, H),
                                                             block.reshape(s, m, 1, H), res_w, rms_w, out_w, EPS,
                                                             OUT_EPS)  # fmt: skip
    return out.reshape(m, H), updated.reshape(m, H)


def check_attn_res(ctx):
    results = []
    for snapshots in SNAPSHOTS:
        for with_prefix in (True, False):
            partial64, prefix64, block64, res_w, rms_w, out_w = _attn_res_inputs(
                ctx, snapshots, with_prefix
            )

            def fused(rows, part=None):
                pre = prefix64[:rows].contiguous() if with_prefix else None
                return ctx.mnnvl.allreduce_attn_res_rmsnorm(
                    (part if part is not None else partial64[:rows]).contiguous(), pre,
                    block64[:, :rows].contiguous(), res_w, rms_w, out_w, EPS, OUT_EPS)  # fmt: skip

            n64, u64 = fused(M_MAX)
            for m in M_CASES:
                partial = partial64[:m].contiguous()
                prefix = prefix64[:m].contiguous() if with_prefix else None
                block = block64[:, :m].contiguous()
                n, u = fused(m)
                ref_n, ref_u = _unfused(ctx, partial, prefix, block, res_w, rms_w, out_w)
                fp32 = _fp32_reference(ref_u, block, res_w, rms_w, out_w)
                again = [fused(m) for _ in range(2)]
                bad = partial.clone()
                if ctx.rank == ctx.world - 1:
                    bad[0, 0] += 1.0
                bad_n, bad_u = fused(m, bad)
                row = dict(
                    op="mnnvl_allreduce_attn_res", case=f"S{snapshots}_{'prefix' if with_prefix else 'noprefix'}",
                    M=m, updated_exact=_same(u, ref_u),
                    normed_vs_unfused=_rel(n, ref_n) if ref_n is not None else 0.0, normed_vs_fp32=_rel(n, fp32),
                    det=all(_same(a, n) and _same(b, u) for a, b in again),
                    rows_as_m64=_same(n, n64[:m]) and _same(u, u64[:m]), control=not _same(bad_u, u),
                    ranks_agree=len(set(ctx.comm.allgather(_digest(n)))) == 1,
                )  # fmt: skip
                good = (row["updated_exact"] and row["normed_vs_unfused"] <= 1e-2 and row["normed_vs_fp32"] <= 2e-2
                        and row["det"] and row["rows_as_m64"] and row["control"] and row["ranks_agree"])  # fmt: skip
                row["ok"] = _all_ranks(ctx, good)
                results.append(row)
    return results


CHECKS = {"attn_res": check_attn_res}


def _run_checks(names):
    try:
        ctx = _context()
        with torch.inference_mode():
            return [row for name in names for row in CHECKS[name](ctx)]
    except Exception:
        traceback.print_exc()
        raise


def _report(rows):
    for row in rows:
        fields = " ".join(f"{k}={(f'{v:.3e}' if isinstance(v, float) else v)}" for k, v in row.items()
                          if k not in ("op", "case", "M"))  # fmt: skip
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("check", list(CHECKS))
def test_k3_mnnvl_comm(mpi_pool_executor, check):
    per_rank = list(mpi_pool_executor.map(_run_checks, [[check]] * WORLD))
    _report(per_rank[0])
    assert all(row["ok"] for rows in per_rank for row in rows)


def main() -> int:
    names = sys.argv[1:] or list(CHECKS)
    rows = _run_checks(names)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(rows)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
