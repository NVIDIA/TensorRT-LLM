# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``m3_allreduce`` entry, 4 ranks: its cells against the entry's reference, and the
fused draft loop's layer boundaries built on it (``M3Eagle3Draft.ar_norm``: the plain all-reduce, then FlashInfer's
fused add + RMSNorm) against stock Eagle3's, bit for bit.

Stock ``Eagle3DecoderLayer`` all-reduces each projection plainly (BF16 out) and runs the ``RMSNorm`` module with the
residual: FlashInfer's ``fused_add_rmsnorm``, which adds in FP32 and normalizes the FP32 sum. The draft loop's
boundaries must be that path exactly, under every all-reduce strategy the single-node drafter can run (AUTO,
ONESHOT, TWOSHOT, NCCL; MNNVL is multi-node only), at tp4 shapes (H = 6144, eps = 1e-6, 1 to 32 rows). The inputs are
random rows and rows built so a BF16 rounding of the residual sum would move the norm the most, including the
review's case: unit weights, sums 1 + 2^-8 on 6143 coordinates and 2 - 2^-8 on the last, both BF16 rounding
midpoints that round to even in opposite directions.

Asserts the normalized rows and the residual sum bitwise equal to the stock path's, and reports the max ulp of each
against an fp64 reference (ulps of BF16 at the larger magnitude; a correctly rounded output is within 0.5).

Its own 4-rank launcher, as ``_m3_mc_allreduce_norm_op_matrix.py`` is; the collected entry point is
``test_modeling_v2_m3_allreduce.py``.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _m3_allreduce_op_matrix.py
"""

import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "the fused all-reduce + RMSNorm requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

HIDDEN = 6144
EPS = 1e-6
TOKENS = (1, 4, 8, 32)
RESULT_TAG = "[ulp]"
"""Prefix of rank 0's result lines; the pytest entry reports them."""

RANK = 0
WORLD = 1

Inputs = Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]
"""(every rank's [m, H] BF16 partial rows, the [m, H] BF16 residual, the [H] BF16 norm weight)."""


# ---------------------------------------------------------------- inputs (every rank builds every rank's rows)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _random(seed: int, m: int, x_scale: float, r_scale: float, unit_w: bool) -> Inputs:
    g = _gen(seed)
    xs = [
        (torch.randn(m, HIDDEN, generator=g, device="cuda") * x_scale).bfloat16()
        for _ in range(WORLD)
    ]
    residual = (torch.randn(m, HIDDEN, generator=g, device="cuda") * r_scale).bfloat16()
    if unit_w:
        w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    else:
        w = (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).bfloat16()
    return xs, residual, w


def _on_rank0(m: int, x: torch.Tensor) -> List[torch.Tensor]:
    """``x`` as rank 0's partial, zeros elsewhere (the cross-rank sum is exact)."""
    return [x.bfloat16()] + [
        torch.zeros(m, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(WORLD - 1)
    ]


def _review_case(m: int) -> Inputs:
    """Sums 1 + 2^-8 on the first H - 1 coordinates and 2 - 2^-8 on the last, unit weights.

    Both sums are BF16 midpoints: 1 + 2^-8 rounds (to even) down to 1 and 2 - 2^-8 up to 2, so the fused path's
    norm sees a residual moved in opposite directions on the bulk and on the last coordinate.
    """
    residual = torch.ones(m, HIDDEN, device="cuda")
    residual[:, -1] = 2.0 - 2.0**-7
    x = torch.full((m, HIDDEN), 2.0**-8, device="cuda")
    return (
        _on_rank0(m, x),
        residual.bfloat16(),
        torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16),
    )


def _review_case_flipped(m: int) -> Inputs:
    """The mirror: the bulk on odd-mantissa midpoints (1 + 3 * 2^-8 rounds up), the last one rounding down."""
    residual = torch.full((m, HIDDEN), 1.0 + 2.0**-7, device="cuda")
    residual[:, -1] = 2.0 - 2.0 * 2.0**-7
    x = torch.full((m, HIDDEN), 2.0**-8, device="cuda")
    return (
        _on_rank0(m, x),
        residual.bfloat16(),
        torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16),
    )


def _midpoints(seed: int, m: int, outliers: int) -> Inputs:
    """Every sum a BF16 midpoint: residual r random in [1, 2) and x = half its ulp (+2^-8), split over the ranks.

    The bulk has an odd mantissa (rounds up); ``outliers`` coordinates per row carry an even one (rounds down) and
    a larger magnitude, so they weigh in the RMS while rounding the other way.
    """
    g = _gen(seed)
    mant = (
        torch.randint(0, 64, (m, HIDDEN), generator=g, device="cuda") * 2 + 1
    )  # odd 7-bit mantissa
    r = 1.0 + mant.float() * 2.0**-7
    x = torch.full((m, HIDDEN), 2.0**-8, device="cuda")
    if outliers:
        cols = torch.randperm(HIDDEN, generator=g, device="cuda")[:outliers]
        even = torch.randint(0, 64, (m, outliers), generator=g, device="cuda") * 2
        r[:, cols] = 8.0 * (1.0 + even.float() * 2.0**-7)
        x[:, cols] = 8.0 * 2.0**-8
    # Split x over the ranks in exact BF16 pieces: x/2 on rank 0, x/4 on ranks 1 and 2, 0 on rank 3.
    parts = [x / 2, x / 4, x / 4] + [torch.zeros_like(x)] * (WORLD - 3)
    return (
        [p.bfloat16() for p in parts[:WORLD]],
        r.bfloat16(),
        torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16),
    )


def _cancelling(seed: int, m: int) -> Inputs:
    """Large partials that cancel across the ranks (+A, -A, +B, -B + d) onto a unit residual."""
    g = _gen(seed)
    a = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 64).bfloat16().float()
    b = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 64).bfloat16().float()
    d = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 2.0**-6).bfloat16().float()
    parts = [a, -a, b, (-b + d)]
    residual = (1.0 + torch.rand(m, HIDDEN, generator=g, device="cuda")).bfloat16()
    w = (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).bfloat16()
    return [p.bfloat16() for p in parts[:WORLD]], residual, w


def _cases() -> List[Tuple[str, Callable[[int], Inputs]]]:
    cases: List[Tuple[str, Callable[[int], Inputs]]] = [
        ("review: 1+2^-8 x6143, 2-2^-8 x1", _review_case),
        ("review mirrored", _review_case_flipped),
        ("all midpoints, no outliers", lambda m: _midpoints(11, m, 0)),
        ("all midpoints, 1 outlier", lambda m: _midpoints(12, m, 1)),
        ("all midpoints, 64 outliers", lambda m: _midpoints(13, m, 64)),
        ("cancelling partials", lambda m: _cancelling(14, m)),
    ]
    for i, (xs, rs, unit) in enumerate(
        ((1.0, 4.0, False), (1.0, 1.0, True), (0.05, 8.0, False), (8.0, 0.5, False))
    ):
        cases.append(
            (
                f"random x{xs} r{rs}{' unit-w' if unit else ''}",
                lambda m, i=i, xs=xs, rs=rs, unit=unit: _random(100 + i, m, xs, rs, unit),
            )
        )
    return cases


# ---------------------------------------------------------------- the three paths


def _fp64(xs: List[torch.Tensor], residual: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    s = residual.double()
    for x in xs:
        s = s + x.double()
    rms = torch.rsqrt(s.square().mean(-1, keepdim=True) + EPS)
    return s * rms * w.double()


def _ulp_at(v: torch.Tensor) -> torch.Tensor:
    """BF16's ulp at |v| (fp64): 2^(e - 8) for |v| in [2^(e-1), 2^e), floored at the smallest normal's."""
    _, e = torch.frexp(v.abs().double())
    return torch.ldexp(torch.ones_like(v, dtype=torch.float64), torch.clamp(e - 8, min=-133))


def _ulps(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max |out - ref| in BF16 ulps at the larger magnitude of the two."""
    o, r = out.double(), ref.double()
    return ((o - r).abs() / _ulp_at(torch.maximum(o.abs(), r.abs()))).max().item()


def _ulps_at(out: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor) -> float:
    """Max |out - ref| in BF16 ulps at the magnitude ``scale`` (the operands'), not at the result's."""
    return ((out.double() - ref.double()).abs() / _ulp_at(scale.abs())).max().item()


def _target(ar: Any, x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor):
    """The draft loop's boundary (``M3Eagle3Draft.ar_norm``): ``m3_allreduce`` plain, then the catalog's
    ``flashinfer_fused_add_rmsnorm``."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
        flashinfer_fused_add_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm.m3_allreduce import (  # noqa: E501
        m3_allreduce,
    )

    return flashinfer_fused_add_rmsnorm(m3_allreduce(x, ar, None), residual.clone(), w, EPS)


def _check_cells(mapping: Any) -> None:
    """Every cell of ``m3_allreduce.CELLS`` (AUTO strategy, as the target builds it) against the entry's reference:
    rank-order fp32 sums with one BF16 rounding of the sum and one of the norm. A strategy may round the cross-rank sum
    to BF16 before adding the residual, so the residual sum is bounded at its operands' magnitude
    u = ulp(max(|residual|, |sum|)): half an ulp for that rounding plus half an output ulp for each of the two final
    roundings, an output ulp being at most 2u, so 2.5u (where the two cancel, the output's own ulp is far smaller).
    The normed rows are the norm of the op's own residual sum, within 2 ulps of the reference norm of that sum."""
    import sys as _sys
    from pathlib import Path as _Path

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm.m3_allreduce import (  # noqa: E501
        m3_allreduce,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
    from tensorrt_llm.functional import AllReduceStrategy

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
    from _validating import validating

    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
    for i, cell in enumerate(m3_allreduce.CELLS):
        spec = cell.spec
        xs, residual, w = _random(200 + i, spec["m"], 1.0, 4.0, False)
        params = (
            None
            if spec.get("plain")
            else AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual.clone(),
                norm_weight=w,
                eps=EPS,
                trigger_completion_at_end=False,
            )
        )
        # The reference first: the fused path may write the updated residual into ``params.residual``.
        ref = m3_allreduce.reference(xs, ar, params)
        with validating(m3_allreduce):
            out = m3_allreduce(xs[RANK], ar, params)
        torch.cuda.synchronize()
        total = sum(x.double() for x in xs)
        if params is None:
            err = _ulps_at(out, ref, total)
            assert err <= 1.0, (cell.why, err)
        else:
            err = _ulps_at(out[1], ref[1], torch.maximum(total.abs(), residual.double().abs()))
            assert err <= 2.5, (cell.why, "residual", err)
            h = out[1].double()
            own = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS) * w.double()
            assert _ulps(out[0], own) <= 2.0, (cell.why, "normed", _ulps(out[0], own))
        dist.barrier()
        if RANK == 0:
            print(f"{RESULT_TAG} cell {cell.why} passed", flush=True)


def _stock(ar: Any, x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor):
    """Stock Eagle3: ``o_proj``'s plain all-reduce (BF16 out), then ``RMSNorm(h, residual)``'s FlashInfer
    ``fused_add_rmsnorm`` (FP32 add, norm of the FP32 sum; in place on both)."""
    h = ar(x)
    res = residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(h, res, w, EPS)
    return h, res


# ---------------------------------------------------------------- the matrix


def _single_node_cluster_info() -> None:
    """``TorchDist`` (the communicator under ``TLLM_DISABLE_MPI``) asks Ray for each rank's node and GPU to build its
    node-local group. These ranks are spawned without Ray on one node, so every rank reports the same node and its
    own device."""
    from tensorrt_llm._torch.distributed.communicator import TorchDist

    def cluster_info(self):
        if self.cluster_info is None:
            self.cluster_info = [("127.0.0.1", r) for r in range(dist.get_world_size())]
        return self.cluster_info

    TorchDist._get_cluster_info = cluster_info


def _run_matrix() -> None:
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers trtllm::flashinfer_fused_add_rmsnorm
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    assert IS_FLASHINFER_AVAILABLE, "stock Eagle3's add + RMSNorm is FlashInfer's fused_add_rmsnorm"
    _single_node_cluster_info()
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=WORLD)
    _check_cells(mapping)
    strategies = (
        AllReduceStrategy.AUTO,
        AllReduceStrategy.ONESHOT,
        AllReduceStrategy.TWOSHOT,
        AllReduceStrategy.NCCL,
    )
    worst = 0.0
    mismatches: List[str] = []
    for strategy in strategies:
        ar = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
        assert ar.mnnvl_allreduce is None and ar.symm_mem_allreduce is None, strategy
        for name, build in _cases():
            ulp, res_ulp = 0.0, 0.0
            for m in TOKENS:
                xs, residual, w = build(m)
                target_out, target_res = _target(ar, xs[RANK], residual, w)
                stock_out, stock_res = _stock(ar, xs[RANK], residual, w)
                torch.cuda.synchronize()
                for what, a, b in (
                    ("normed", target_out, stock_out),
                    ("residual", target_res, stock_res),
                ):
                    bad = int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
                    if bad:
                        mismatches.append(
                            f"{strategy.name} {name} m={m}: {bad} {what} elements differ from stock"
                        )
                ref = _fp64(xs, residual, w)
                ref_res = residual.double() + sum(x.double() for x in xs)
                u, ur = _ulps(target_out, ref), _ulps(target_res, ref_res)
                assert math.isfinite(u) and math.isfinite(ur), (strategy, name, m)
                ulp, res_ulp = max(ulp, u), max(res_ulp, ur)
                dist.barrier()
            worst = max(worst, ulp)
            if RANK == 0:
                print(
                    f"{RESULT_TAG} {strategy.name:7s} | {name:34s} | target = stock vs fp64: normed {ulp:.3f}"
                    f" | residual {res_ulp:.3f}",
                    flush=True,
                )
    if RANK == 0:
        scope = f"all cases, m in {TOKENS}"
        print(
            f"{RESULT_TAG} MAX     | {scope:34s} | target = stock vs fp64: normed {worst:.3f}",
            flush=True,
        )
    assert not mismatches, "\n".join(mismatches)


def _run_one_rank() -> int:
    global RANK, WORLD
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    assert WORLD == 4, f"the drafter runs at tp4, got {WORLD} ranks"
    try:
        _run_matrix()
    except BaseException:
        import traceback

        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per claimed device, each in its own session; on the first nonzero exit or the deadline every
    session is killed (as ``_m3_mc_allreduce_norm_op_matrix.py`` does)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 4, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the matrix runs at 4"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure: Optional[str] = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
