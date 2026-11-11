# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix of this target's fused all-reduce + mHC boundary (`ops/comm/mhc_ar_boundary.py`).

Same shape as the catalog's collective matrices (`tests/unittest/_torch/modeling_v2/comm/`): run plainly, this
script is its own launcher. It spawns one rank per device named in CUDA_VISIBLE_DEVICES (the op is compiled for
4) and kills the whole process group on the first failure or at the deadline, because a broken collective hangs
rather than raising.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mhc_ar_boundary_op_matrix.py

Every rank can regenerate every other rank's partial from the seed, so the reference is arithmetic rather than a
second collective. The IPC workspace's one host-side exchange of memory handles goes through a gloo group, as in
the catalog's allreduce matrix.

Not a pytest module: the `check_*` bodies are one ordered sequence inside one job and read rank state that only
`_run_one_rank` binds.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "the fused boundary requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

# Bound inside the rank body, never in the launcher: importing the entry pulls in tensorrt_llm, and the launcher
# must carry no CUDA or MPI state.
entry: Any = None
library_allreduce: Any = None
assert_within_ulp: Any = None
WORKSPACE: Any = None
RANK = 0
WORLD = 1
HOST_GROUP: Any = None

MULT, HIDDEN, MIX = 4, 4096, 24
NORM_EPS, HC_EPS, SINKHORN_EPS, POST_MULT, LAYER_INPUT_EPS = 1e-6, 1e-6, 1e-6, 2.0, 1e-5

# The band `_compare` gates on, and how it was measured, is in its docstring.
BF16_ELEMENT_ULP, BF16_RMS_ULP, FP32_ATOL = 3.0, 0.5, 5e-3
OUTPUTS = ("residual_out", "post_out", "comb_out", "layer_input")
# 1: rank 0 prints each call's deviations in the band's units; 2: prints them and gates nothing.
REPORT = os.environ.get("MHC_AR_MATRIX_REPORT", "0")


class _HostExchange:
    """The one call `IpcMemory` makes on `Distributed`: a TP-group allgather of host objects."""

    def tp_allgather(self, obj: Any) -> List[Any]:
        gathered: List[Any] = [None] * WORLD
        dist.all_gather_object(gathered, obj, group=HOST_GROUP)
        return gathered


def _partial(rank: int, rows: int, seed: int) -> torch.Tensor:
    """One rank's share of a block output, at the scale an o_proj / MoE partial has."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return (torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)


def _boundary_inputs(rows: int, seed: int) -> dict:
    """The streams, the previous block's coefficients and the next block's mixer and norm (shared by the ranks)."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + 7)
    residual = torch.randn(rows, MULT, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    post = torch.rand(rows, MULT, generator=gen, device="cuda") * POST_MULT
    comb = torch.rand(rows, MULT, MULT, generator=gen, device="cuda")
    comb = comb / comb.sum(-2, keepdim=True)
    return dict(
        residual=residual,
        post=post,
        comb=comb.contiguous(),
        fn=torch.randn(MIX, MULT * HIDDEN, generator=gen, device="cuda") * 0.02,
        scale=torch.ones(3, device="cuda"),
        base=torch.randn(MIX, generator=gen, device="cuda") * 0.1,
        norm_weight=(1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(
            torch.bfloat16
        ),
    )


def _scalars(cell) -> dict:
    return dict(
        sinkhorn_iters=cell.spec["sinkhorn_iters"],
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_eps=SINKHORN_EPS,
        post_mult=POST_MULT,
        layer_input_eps=LAYER_INPUT_EPS,
    )


def _report(out, ref) -> None:
    """Print each output's deviation in the units `_compare` gates on."""
    # Imported here, not at module scope: _validating imports tensorrt_llm, which the launcher must not, and the
    # rank body puts it on sys.path.
    from _validating import ULP

    ulp = ULP[torch.bfloat16]
    for name, o, r in zip(OUTPUTS, out, ref):
        d = o.float() - r.float()
        if r.dtype != torch.bfloat16:
            print(f"[report] {name}: max abs {d.abs().max().item():.3g}", flush=True)
            continue
        flat_d, flat_r = d.reshape(-1, r.shape[-1]), r.float().reshape(-1, r.shape[-1])
        row_scale = flat_r.abs().amax(1, keepdim=True).clamp_min(1e-9)
        element = (flat_d.abs() / row_scale).max().item() / ulp
        rms = (flat_d.square().mean().sqrt() / flat_r.square().mean().sqrt()).item() / ulp
        print(f"[report] {name}: {element:.2f} ulp per element, {rms:.3f} ulp RMS", flush=True)


def _compare(out, ref) -> None:
    """The BF16 outputs within 3 ulp per element against the row scale and 0.5 ulp relative RMS, the FP32
    coefficients within 5e-3 absolute.

    Measured on B200 over the 15 calls of this file (4 ranks; the three cells x 3 seeds, 4 interleaved and 2
    graph-replayed calls):

    - residual_out: at most 1.74 ulp per element, 0.17 ulp RMS. The streams differ from the reference only by
      the order of their FP32 sums before one BF16 rounding; a flip of one ulp of a row's largest elements
      reads as 1 to 2 ulp of the row scale.
    - layer_input: at most 2.02 ulp per element, 0.25 ulp RMS. A stream's flip carries into the collapsed
      row on top of its own rounding, and `pre` comes out of the TF32 mixer.
    - post_out and comb_out: at most 1.9e-3 and 1.4e-3 absolute, on values in [0, 2] and [0, 1]. The mixer
      weights round to TF32 (the BF16 streams are exact in it).

    A reduction that drops a rank measures 148 to 175 ulp per element (120 to 130 ulp RMS) on the BF16
    outputs and 0.44 to 0.65 absolute on the coefficients; `check_the_gate_discriminates_a_wrong_reduction`
    checks the band refuses one.
    """
    if REPORT in ("1", "2") and RANK == 0:
        _report(out, ref)
    if REPORT == "2":
        return
    for name, o, r in zip(OUTPUTS, out, ref):
        if r.dtype == torch.bfloat16:
            assert_within_ulp(o, r, element_ulp=BF16_ELEMENT_ULP, rms_ulp=BF16_RMS_ULP)
        else:
            torch.testing.assert_close(
                o, r, rtol=0.0, atol=FP32_ATOL, msg=lambda m, n=name: f"{n}: {m}"
            )


def _run(rows: int, seed: int, cell, inputs: dict | None = None):
    inputs = _boundary_inputs(rows, seed) if inputs is None else inputs
    out = entry(_partial(RANK, rows, seed), WORKSPACE, RANK, **inputs, **_scalars(cell))
    ref = entry.reference(
        [_partial(r, rows, seed) for r in range(WORLD)], WORKSPACE, RANK, **inputs, **_scalars(cell)
    )
    return out, ref


def check_certified_cells() -> None:
    """Every cell, three seeds, against the entry's reference and this file's band."""
    for cell in entry.CELLS:
        for seed in range(3):
            out, ref = _run(cell.spec["rows"], seed, cell)
            _compare(out, ref)


def check_interleaves_with_the_library_allreduce() -> None:
    """The op and the library's one-shot all-reduce share the workspace's Lamport rotation: alternate them."""
    cell = entry.CELLS[-1]
    for seed in range(4):
        out, ref = _run(cell.spec["rows"], seed, cell)
        _compare(out, ref)
        x = _partial(RANK, 8, 100 + seed)
        summed = library_allreduce(x)
        expect = torch.stack([_partial(r, 8, 100 + seed).float() for r in range(WORLD)]).sum(0)
        assert_within_ulp(
            summed, expect.to(torch.bfloat16), element_ulp=BF16_ELEMENT_ULP, rms_ulp=BF16_RMS_ULP
        )


def check_cuda_graph_replay() -> None:
    """Captured once, replayed on new partials and streams written into the captured inputs."""
    cell = entry.CELLS[1]
    rows = cell.spec["rows"]
    inputs = _boundary_inputs(rows, 0)
    partial = _partial(RANK, rows, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        entry(partial, WORKSPACE, RANK, **inputs, **_scalars(cell))
    torch.cuda.current_stream().wait_stream(side)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = entry(partial, WORKSPACE, RANK, **inputs, **_scalars(cell))
    for seed in (5, 6):
        fresh = _boundary_inputs(rows, seed)
        for name, tensor in fresh.items():
            inputs[name].copy_(tensor)
        partial.copy_(_partial(RANK, rows, seed))
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        ref = entry.reference(
            [_partial(r, rows, seed) for r in range(WORLD)],
            WORKSPACE,
            RANK,
            **inputs,
            **_scalars(cell),
        )
        _compare(out, ref)


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference that drops one rank's partial must fail the band."""
    cell = entry.CELLS[1]
    rows, seed = cell.spec["rows"], 9
    inputs = _boundary_inputs(rows, seed)
    out = entry(_partial(RANK, rows, seed), WORKSPACE, RANK, **inputs, **_scalars(cell))
    wrong = entry.reference(
        [_partial(r, rows, seed) for r in range(WORLD - 1)],
        WORKSPACE,
        RANK,
        **inputs,
        **_scalars(cell),
    )
    try:
        _compare(out, wrong)
    except AssertionError:
        return
    if REPORT != "2":
        raise AssertionError("the band accepted a reduction that dropped a rank")


def check_is_valid_rejects_what_the_kernel_cannot_take() -> None:
    cell = entry.CELLS[0]
    inputs = _boundary_inputs(9, 0)
    try:
        entry.is_valid(_partial(RANK, 9, 0), WORKSPACE, RANK, **inputs, **_scalars(cell))
    except AssertionError:
        return
    raise AssertionError("is_valid accepted 9 rows")


CHECKS = (
    check_certified_cells,
    check_interleaves_with_the_library_allreduce,
    check_cuda_graph_replay,
    check_the_gate_discriminates_a_wrong_reduction,
    check_is_valid_rejects_what_the_kernel_cannot_take,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero and the launcher kills its
    siblings."""
    global RANK, WORLD, HOST_GROUP, WORKSPACE, entry, library_allreduce, assert_within_ulp
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    HOST_GROUP = dist.new_group(backend="gloo")

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as ar_entry
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.comm import (  # noqa: E501
        mhc_ar_boundary as boundary_module,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    assert world == boundary_module.WORLD, (
        f"the op is compiled for {boundary_module.WORLD} ranks, got {world}"
    )
    local_group = dist.new_group(ranks=list(range(world)), backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())
    communicator.Distributed.get = staticmethod(lambda mapping: _HostExchange())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)

    entry = boundary_module.mhc_ar_boundary
    group_args = ar_entry.allreduce_group_args(mapping, strategy=int(AllReduceStrategy.ONESHOT))
    WORKSPACE = group_args["workspace"]

    def library_allreduce(x):
        return ar_entry.allreduce(
            x,
            residual=None,
            norm_weight=None,
            scale=None,
            bias=None,
            op=int(AllReduceFusionOp.NONE),
            eps=1e-6,
            trigger_completion_at_end=False,
            **group_args,
        )

    # Spawned worker, not a pytest module: reach the modeling_v2 test helpers directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import assert_within_ulp as _assert_within_ulp

    assert_within_ulp = _assert_within_ulp

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves a collective early wedges its siblings; exit hard so the launcher kills them.
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per visible device, rendezvous over env://, all killed on the first nonzero exit or at the
    deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
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
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
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
    failure = None
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
