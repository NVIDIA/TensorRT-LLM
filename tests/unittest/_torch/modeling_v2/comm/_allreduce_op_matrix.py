# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the allreduce catalog entry.

The gpt-oss and deepseek cells (`RESIDUAL_RMS_NORM` in bf16) and Step-5's
plain sums (`NONE`) at hidden 4096 in bf16 and fp32. Same shape as the
allgather and reducescatter matrices beside it: run plainly, this script is
its own launcher. It spawns one rank per
device named in CUDA_VISIBLE_DEVICES and kills the whole process group on the
first failure or at the deadline, because a broken collective hangs rather
than raising.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py
    CUDA_VISIBLE_DEVICES=0,1 python _allreduce_op_matrix.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py --checks=lamport

The world size is the number of visible devices, and a run certifies the cells
whose `world` matches it. The collected entry point runs both sizes.
`--checks=lamport` runs instead the detector of the engine's Lamport slot
hazard (see the entry's note), which fails until that is fixed.

Not a pytest module: the `check_*` bodies are one ordered sequence inside one
job and read rank state that only `_run_one_rank` binds.

The IPC workspace needs one host-side exchange of memory handles across the
TP group. The engine does that through `Distributed.get(mapping)`, which under
MPI-disabled mode is `TorchDist` and requires Ray. These ranks rendezvous
through torch.distributed alone, so `_HostExchange` below stands in for it
with a gloo group. It carries only that exchange; the allreduce itself runs on
the real NCCL communicator and the real IPC buffers.

`TorchDist` also registers the world and node-local process groups with the
C++ side (`init_pg`). The `_pg` symbol reads both on its first call to find
which ranks share a node, and dereferences null when they are missing. The
rank body registers them the same way, with the backend pair a Ray worker uses.
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

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

# tests/unittest/_torch/modeling_v2, where `_validating.py` lives.
_MODELING_V2_TESTS = Path(__file__).resolve().parent.parent

# Bound inside the rank body, never in the launcher: importing the entry pulls
# in tensorrt_llm, and the launcher must carry no CUDA or MPI state.
entry: Any = None
validating: Any = None
# Communicator arguments per strategy, built once the way a target does.
GROUP_ARGS: dict = {}
RANK = 0
WORLD = 1
HOST_GROUP: Any = None

# Decode graph batch sizes a served default engine captures, sampled: the
# smallest, a power of two, a size that is not one, and the cap.
GRAPH_BATCH_SIZES = (1, 8, 13, 128)

# Bands, in ulps of the output dtype: (per element against the row's largest
# magnitude, relative RMS). Keyed by (op, dtype); derived in `_compare`.
_FUSED_BF16_BAND = (5.0, 1.25)
_SUM_BF16_BAND = (4.0, 1.0)
_SUM_FP32_BAND = (4.0, 1.0)


def _band(op: int, dtype: torch.dtype):
    if op == entry.NONE:
        return _SUM_FP32_BAND if dtype == torch.float32 else _SUM_BF16_BAND
    assert dtype == torch.bfloat16, f"no band is derived for the fused op in {dtype}"
    return _FUSED_BF16_BAND


def _compare(out, ref, op: int) -> None:
    """Every output within its band.

    RESIDUAL_RMS_NORM in bf16 (the gpt-oss and deepseek cells): 5 ulp per element
    against the row scale, 1.25 ulp relative RMS. Measured on GB300 with the
    gpt-oss payloads (hidden 2880, bf16, 20 seeds, 1 to 8192 rows):

    - world 2 and 4, up to 2048 rows: at most 2.56 ulp per element and
      0.91 ulp RMS. Two and four ranks measure the same here, so these
      kernels accumulate in fp32.
    - world 4, 8192 rows: 4.00 ulp per element, 0.97 ulp RMS. World 2 at
      8192 rows stays at 2.56. The error grows with the rank count only at
      this size, where AUTO takes the NCCL fallback, consistent with a path
      that rounds to bf16 at each rank's addition.

    The deepseek cells (hidden 7168) run under the same band. `norm_out`
    measures closer to a norm of the fp32 sum than to a norm of the
    bf16-rounded `residual_out`, so the reference's norm reads the fp32 sum.

    NONE (the Step-5 cells), against the correctly rounded sum. Measured on
    GB300 at world 4, hidden 4096, 8 seeds per row count, rows 1, 37, 128,
    512, 1023, 1024, 1029, 2048, 4095, 4096 and 8192, with the kernel AUTO
    picked read from the op's debug log (one-shot below 1024 rows, two-shot
    from 1024 to 4095, NCCL from 4096):

    - bf16: 4 ulp per element, 1 ulp RMS. The one-shot and two-shot kernels
      measured bit-equal to the reference at every row count: they sum the
      four bf16 words in fp32 and round once, which is the reference's own
      rounding. NCCL measured 2.60 ulp per element and 0.76 ulp RMS at 4096
      and 8192 rows: its ring rounds to bf16 after each of its three
      additions, and a partial can exceed the row's largest final magnitude.
    - fp32: 4 ulp per element, 1 ulp RMS, in fp32 ulps. The fp32 cells run
      NCCL (the entry refuses fp32 under a workspace strategy); every path,
      one-shot and two-shot included, measured at most 1.37 ulp per element
      and 0.39 ulp RMS: four fp32 words added in
      some order, three roundings each at most half an ulp of a partial
      bounded by the sum of magnitudes, against a reference summed in fp64.
      An output rounded through bf16 measured at least 16346 ulp away;
      `check_the_fp32_band_refuses_a_bf16_round_trip` checks the band
      refuses it.

    A dropped rank moves the result by tens of ulp under RESIDUAL_RMS_NORM,
    at least 129 bf16 ulp and 4.2e6 fp32 ulp under NONE (measured, same
    sweep); `check_the_gate_discriminates_a_wrong_reduction` checks every
    band refuses one.
    """
    # Imported here, not at module scope: _validating imports tensorrt_llm,
    # which the launcher must not, and the rank body puts it on sys.path.
    from _validating import assert_within_ulp

    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    assert len(outs) == len(refs), (len(outs), len(refs))
    for o, r in zip(outs, refs):
        element_ulp, rms_ulp = _band(op, r.dtype)
        assert_within_ulp(o, r, element_ulp=element_ulp, rms_ulp=rms_ulp)


class _HostExchange:
    """The one call `IpcMemory` makes on `Distributed`: a TP-group allgather of
    host objects."""

    def tp_allgather(self, obj: Any) -> List[Any]:
        gathered: List[Any] = [None] * WORLD
        dist.all_gather_object(gathered, obj, group=HOST_GROUP)
        return gathered


def _payload(
    rank: int, rows: int, hidden: int, seed: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """One rank's partial sum: values of the scale a projection produces.

    Every rank can regenerate every other rank's payload from the seed, which
    is what lets the reference be arithmetic rather than a second collective.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn(rows, hidden, generator=gen, device="cuda").to(dtype)


def _residual_and_weight(rows: int, hidden: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + 7)
    residual = torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16)
    return residual, weight


def _operands(spec, rows: int, seed: int):
    """This rank's input and the fusion operands the cell's op reads: none for
    NONE, which is how a Step-5 call site passes it."""
    x = _payload(RANK, rows, spec["hidden"], seed, spec["dtype"])
    if spec["op"] == entry.NONE:
        return x, None, None
    residual, weight = _residual_and_weight(rows, spec["hidden"], seed)
    return x, residual, weight


def _eps(spec) -> float:
    # NONE reads no eps; the op still takes one.
    return spec.get("eps", 0.0)


def _call(x, residual, weight, eps, op, strategy):
    return entry.allreduce(
        x,
        residual=residual,
        norm_weight=weight,
        scale=None,
        bias=None,
        op=op,
        eps=eps,
        trigger_completion_at_end=False,
        **GROUP_ARGS[strategy],
    )


def _reference(rows, spec, seed, residual, weight, world=None):
    world = WORLD if world is None else world
    inputs = [_payload(r, rows, spec["hidden"], seed, spec["dtype"]) for r in range(world)]
    return entry.allreduce.reference(inputs, residual, weight, _eps(spec), spec["op"])


def _cells():
    return [c for c in entry.allreduce.CELLS if c.spec["world"] == WORLD]


def _first_spec_of(op: int, dtype: torch.dtype):
    """The first cell at this world size for `(op, dtype)`, or None."""
    for c in _cells():
        if c.spec["op"] == op and c.spec["dtype"] == dtype:
            return c.spec
    return None


def _distinct_specs():
    """One spec per (op, dtype) this world size certifies, in cell order: the
    contract checks below run once per kind of call, not once per row count."""
    seen, specs = set(), []
    for c in _cells():
        key = (c.spec["op"], c.spec["dtype"])
        if key not in seen:
            seen.add(key)
            specs.append(c.spec)
    return specs


def check_certified_cells() -> None:
    """Every cell at this world size, inside `validating`, against the entry's
    reference and this file's band."""
    cells = _cells()
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        seed = 4000 + 17 * i
        x, residual, weight = _operands(spec, spec["rows"], seed)
        with validating(entry.allreduce):
            out = _call(x, residual, weight, _eps(spec), spec["op"], spec["strategy"])
        ref = _reference(spec["rows"], spec, seed, residual, weight)
        try:
            first = out[0] if isinstance(out, tuple) else out
            assert first.dtype == spec["dtype"], (first.dtype, spec["dtype"])
            _compare(out, ref, spec["op"])
        except AssertionError as exc:
            raise AssertionError(f"{cell.why}: {exc}") from exc
        dist.barrier()


def check_inputs_are_untouched() -> None:
    """Every form returns fresh tensors and writes none of its arguments --
    the fused form is the opposite contract from flashinfer_fused_add_rmsnorm,
    which a target swaps this entry in for."""
    for spec in _distinct_specs():
        x, residual, weight = _operands(spec, 32, 61)
        x0 = x.clone()
        r0 = None if residual is None else residual.clone()
        outs = _call(x, residual, weight, _eps(spec), spec["op"], spec["strategy"])
        outs = outs if isinstance(outs, tuple) else (outs,)
        torch.cuda.synchronize()
        assert torch.equal(x, x0), f"input was written ({spec['dtype']}, op {spec['op']})"
        if residual is not None:
            assert torch.equal(residual, r0), "residual was written"
        inputs = [x.data_ptr()] + ([] if residual is None else [residual.data_ptr()])
        for out in outs:
            assert out.dtype == spec["dtype"], (out.dtype, spec["dtype"])
            assert out.data_ptr() not in inputs, "output aliases an input"
        dist.barrier()


def _graph_replay(spec) -> None:
    hidden, eps, op, strategy = spec["hidden"], _eps(spec), spec["op"], spec["strategy"]
    pool = None
    graphs = {}
    for rows in GRAPH_BATCH_SIZES:
        x, residual, weight = _operands(spec, rows, 700 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _call(x, residual, weight, eps, op, strategy)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outs = _call(x, residual, weight, eps, op, strategy)
        outs = outs if isinstance(outs, tuple) else (outs,)
        if pool is None:
            pool = graph.pool()
        graphs[rows] = (graph, x, residual, weight, outs)
        dist.barrier()

    for step, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
        graph, x, residual, weight, outs = graphs[rows]
        seed = 900 + step
        x.copy_(_payload(RANK, rows, hidden, seed, spec["dtype"]))
        if residual is not None:
            new_residual, _ = _residual_and_weight(rows, hidden, seed)
            residual.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(rows, spec, seed, residual, weight)
        got = tuple(o.clone() for o in outs)
        try:
            _compare(got if len(got) > 1 else got[0], ref, op)
        except AssertionError as exc:
            raise AssertionError(f"replay rows={rows} ({spec['dtype']}, op {op}): {exc}") from exc
        dist.barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """One graph per decode batch size, sharing a pool, replayed after the
    input buffers are refilled: the engine's decode path. Once per kind of
    call this world size certifies."""
    for spec in _distinct_specs():
        _graph_replay(spec)


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference that drops one rank's contribution must fail the band;
    otherwise the band passes anything. Once per band."""
    for spec in _distinct_specs():
        rows = 64
        x, residual, weight = _operands(spec, rows, 81)
        out = _call(x, residual, weight, _eps(spec), spec["op"], spec["strategy"])
        wrong = _reference(rows, spec, 81, residual, weight, world=WORLD - 1)
        try:
            _compare(out, wrong, spec["op"])
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"the band accepted a sum that is missing one rank ({spec['dtype']}, op {spec['op']})"
            )
        dist.barrier()


def check_the_fp32_band_refuses_a_bf16_round_trip() -> None:
    """The fp32 cells certify that the sum stays fp32: an output rounded
    through bf16 -- a path that reduced in 16 bits -- must fail their band."""
    spec = _first_spec_of(entry.NONE, torch.float32)
    if spec is None:
        return
    x, _, _ = _operands(spec, 64, 83)
    out = _call(x, None, None, _eps(spec), spec["op"], spec["strategy"])
    ref = _reference(64, spec, 83, None, None)
    _compare(out, ref, spec["op"])
    try:
        _compare(out.to(torch.bfloat16).float(), ref, spec["op"])
    except AssertionError:
        pass
    else:
        raise AssertionError("the fp32 band accepted a sum rounded through bf16")
    dist.barrier()


def check_wrapper_guards() -> None:
    """The guard refuses a strided input, a missing workspace, and a 4-byte
    input under a strategy that reads the IPC workspace (the Lamport slot
    hazard `check_a_16_bit_call_after_an_fp32_call_is_right` shows); it
    admits that input under NCCL.

    `is_valid` is called directly rather than through a guarded op call: a
    guard that wrongly admitted the missing workspace would hand the kernel a
    null pointer.
    """
    spec = _cells()[0].spec
    hidden = spec["hidden"]
    wide = _payload(RANK, 8, 2 * hidden, 91)
    strided = wide[:, :hidden]
    residual, weight = _residual_and_weight(8, hidden, 91)
    fp32 = _payload(RANK, 8, hidden, 91, torch.float32)
    for bad, match in (
        (dict(input=strided), "contiguous"),
        (dict(workspace=None), "workspace"),
        (dict(input=fp32, op=entry.NONE), "half-cleared"),
    ):
        kwargs = dict(GROUP_ARGS[entry.AUTO])
        kwargs.update(
            input=_payload(RANK, 8, hidden, 91),
            residual=residual,
            norm_weight=weight,
            scale=None,
            bias=None,
            op=spec["op"],
            eps=_eps(spec),
            trigger_completion_at_end=False,
        )
        kwargs.update(bad)
        try:
            entry.allreduce.is_valid(**kwargs)
        except AssertionError as exc:
            assert match in str(exc), (match, str(exc))
        else:
            raise AssertionError(f"is_valid accepted {sorted(bad)}")
    entry.allreduce.is_valid(
        fp32,
        residual=None,
        norm_weight=None,
        scale=None,
        bias=None,
        op=entry.NONE,
        eps=0.0,
        trigger_completion_at_end=False,
        **GROUP_ARGS[entry.NCCL],
    )
    dist.barrier()


# A one-shot call the size of the sequence's last call needs every rank's
# words to have been cleared to the Lamport sentinel. 1000 rows of 4096 bf16
# is one-shot at world 4 and covers the bytes an fp32 call of 128 rows left.
_LAMPORT_LAST_ROWS = 1000
# Cycles rank 0 sleeps before its last call, so its words land after the
# other ranks have started polling for them.
_LAMPORT_DELAY_CYCLES = 400_000_000


def _lamport_sequence(first: dict, first_rows: int, seed: int) -> int:
    """`first` at `first_rows`, two one-row bf16 calls, then a bf16 call that
    reuses `first`'s Lamport slot (triple buffering: three calls on) with
    rank 0's launch delayed. Returns how many rows of that last call this rank
    got wrong. Every call is AUTO and unguarded: the sequence is what a target
    that alternates bf16 and fp32 sums on one workspace runs."""
    bf16 = dict(hidden=4096, dtype=torch.bfloat16, op=entry.NONE, strategy=entry.AUTO)
    dist.barrier()
    torch.cuda.synchronize()
    x, _, _ = _operands(first, first_rows, seed)
    _call(x, None, None, 0.0, entry.NONE, entry.AUTO)
    for k in (1, 2):
        x, _, _ = _operands(bf16, 1, seed + k)
        _call(x, None, None, 0.0, entry.NONE, entry.AUTO)
    x, _, _ = _operands(bf16, _LAMPORT_LAST_ROWS, seed + 3)
    if RANK == 0:
        torch.cuda._sleep(_LAMPORT_DELAY_CYCLES)
    out = _call(x, None, None, 0.0, entry.NONE, entry.AUTO)
    ref = _reference(_LAMPORT_LAST_ROWS, bf16, seed + 3, None, None)
    torch.cuda.synchronize()
    dist.barrier()
    # Three same-dtype calls clear every slot the sequence touched.
    for k in range(3):
        x, _, _ = _operands(bf16, _LAMPORT_LAST_ROWS, seed + 4 + k)
        _call(x, None, None, 0.0, entry.NONE, entry.AUTO)
    torch.cuda.synchronize()
    dist.barrier()
    element_ulp, _ = _band(entry.NONE, torch.bfloat16)
    from _validating import ULP

    err = (out.float() - ref.float()).abs() / ref.float().abs().amax(dim=1, keepdim=True)
    wrong = (err > element_ulp * ULP[torch.bfloat16]) | ~torch.isfinite(out.float())
    return int(wrong.any(dim=1).sum().item())


def check_a_16_bit_call_after_an_fp32_call_is_right() -> None:
    """The Lamport slot hazard, as the target's call pattern meets it.

    Control first: the same sequence opened by a bf16 call of the same byte
    count (256 rows) must be right on every rank. Then the sequence opened by
    an fp32 call of 128 rows, which the one-shot kernel's clear only half
    erases (see the entry's note): rows 512 to 999 of the last call read
    rank 0's slot before rank 0 wrote it. Fails until the engine records the
    clear size in a dtype-independent unit.
    """
    control = _lamport_sequence(
        dict(hidden=4096, dtype=torch.bfloat16, op=entry.NONE, strategy=entry.AUTO), 256, 5100
    )
    assert control == 0, f"the bf16-opened control got {control} wrong rows"
    wrong = _lamport_sequence(
        dict(hidden=4096, dtype=torch.float32, op=entry.NONE, strategy=entry.AUTO), 128, 5200
    )
    assert wrong == 0, (
        f"rank {RANK}: {wrong} of {_LAMPORT_LAST_ROWS} rows wrong in a bf16 one-shot call "
        "three calls after an fp32 one-shot call on the same workspace"
    )


CHECKS = (
    check_certified_cells,
    check_inputs_are_untouched,
    check_cuda_graph_replay_at_engine_batch_sizes,
    check_the_gate_discriminates_a_wrong_reduction,
    check_the_fp32_band_refuses_a_bf16_round_trip,
    check_wrapper_guards,
)

# The engine defect's detector, run as its own job: it leaves the workspace
# in a state only its own flush repairs, and it fails until the fix lands.
LAMPORT_CHECKS = (check_a_16_bit_call_after_an_fp32_call_is_right,)

_CHECK_SETS = {"cells": CHECKS, "lamport": LAMPORT_CHECKS}
_CHECKS_FLAG = "--checks="


def _bind_rank() -> None:
    """Rendezvous, register the process groups with the C++ side, and bind the
    entry and its communicator arguments for this rank."""
    global RANK, WORLD, HOST_GROUP, GROUP_ARGS, entry, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # The spawned rank owns device `rank`. Never LOCAL_RANK: every spawned
    # rank inherits the launching shell's one value, and NCCL then refuses
    # the group with "invalid usage".
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    HOST_GROUP = dist.new_group(backend="gloo")

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as entry
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.mapping import Mapping

    # Every rank is on this one node, so the node-local group is all of them.
    local_group = dist.new_group(ranks=list(range(world)), backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())

    communicator.Distributed.get = staticmethod(lambda mapping: _HostExchange())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
    # What a target does in __init__: build the communicator arguments once,
    # per strategy it passes.
    GROUP_ARGS = {s: entry.allreduce_group_args(mapping, s) for s in (entry.AUTO, entry.NCCL)}

    # Spawned worker, not a pytest module: reach the sibling helper directly.
    sys.path.insert(0, str(_MODELING_V2_TESTS))
    from _validating import validating as _validating_cm

    validating = _validating_cm


def _check_set(argv) -> str:
    """The check set named by `--checks=<name>`, `cells` by default."""
    names = [a[len(_CHECKS_FLAG) :] for a in argv if a.startswith(_CHECKS_FLAG)]
    name = names[-1] if names else "cells"
    assert name in _CHECK_SETS, f"unknown check set {name!r}; one of {sorted(_CHECK_SETS)}"
    return name


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check of the set; a failing rank
    exits nonzero and the launcher kills its siblings."""
    checks = _CHECK_SETS[_check_set(sys.argv)]
    _bind_rank()
    for check in checks:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves a collective early wedges its siblings; exit
            # hard so the launcher kills them at once.
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(checks)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks(script: Path = Path(__file__).resolve(), checks: str = "cells") -> None:
    """One worker per visible device, rendezvous over env://, all killed on
    the first nonzero exit or at the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"a collective test needs at least 2 devices, got {world_size}"
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
    # Inheriting the launching srun step's PMI namespace lets one failing rank
    # take the whole step down unreported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(script), _WORKER_FLAG, f"{_CHECKS_FLAG}{checks}"],
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
    _spawn_ranks(checks=_check_set(sys.argv))
    print("OK")
