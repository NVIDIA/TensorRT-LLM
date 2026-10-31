# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the allreduce catalog entry.

It covers both fusions the entry certifies: `RESIDUAL_RMS_NORM` (the gpt-oss
and deepseek default targets) and the plain sum `op=NONE` the qwen4 unified
target calls at hidden 2560, which has checks of its own (`check_none_*`).

Same shape as the allgather and reducescatter matrices beside it: run plainly,
this script is its own launcher. It spawns one rank per device named in
CUDA_VISIBLE_DEVICES and kills the whole process group on the first failure
or at the deadline, because a broken collective hangs rather than raising.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py
    CUDA_VISIBLE_DEVICES=0,1 python _allreduce_op_matrix.py

The world size is the number of visible devices, and a run certifies the cells
whose `world` matches it. The collected entry point runs every size, 2, 4 and
8, that the node has devices for. The `NONE` cells are world 4 only, so a
world-2 or world-8 run checks the `RESIDUAL_RMS_NORM` cells alone. Rank 0
prints the measured error of every cell and negative control.

Not a pytest module, for the same reason as its siblings: the `check_*`
bodies are one ordered sequence inside one job and read rank state that only
`_run_one_rank` binds.

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

import functools
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

# Bound inside the rank body, never in the launcher: importing the entry pulls
# in tensorrt_llm, and the launcher must carry no CUDA or MPI state.
allreduce: Any = None
entry: Any = None
validating: Any = None
GROUP_ARGS: dict = {}
RANK = 0
WORLD = 1
HOST_GROUP: Any = None

# Decode graph batch sizes a served default engine captures, sampled: the
# smallest, a power of two, a size that is not one, and the cap.
GRAPH_BATCH_SIZES = (1, 8, 13, 128)

# The row counts the qwen4 target captures its plain sum at under tp4 / tep4:
# the decode batches above, the 128-request batch at an MTP-3 verify (4 rows
# per request) and at an MTP-7 verify (8 rows per request). All of them take
# the one-shot kernel.
NONE_GRAPH_ROWS = (1, 8, 13, 128, 512, 1024)

# One row count per kernel AUTO picks for `NONE` at hidden 2560, world 4:
# the one-shot IPC kernel, and NCCL.
NONE_PATH_ROWS = (64, 8192)


def _compare(out, ref) -> None:
    """5 ulp per element against the row scale, 1.25 ulp relative RMS, on
    every output.

    Measured on GB300 with the gpt-oss payloads (hidden 2880, bf16, 20 seeds,
    1 to 8192 rows):

    - world 2 and 4, up to 2048 rows: at most 2.56 ulp per element and
      0.91 ulp RMS. Two and four ranks measure the same here, so these
      kernels accumulate in fp32.
    - world 4, 8192 rows: 4.00 ulp per element, 0.97 ulp RMS. World 2 at
      8192 rows stays at 2.56. The error grows with the rank count only at
      this size, where AUTO takes the NCCL fallback, consistent with a path
      that rounds to bf16 at each rank's addition.

    The deepseek cells (hidden 7168) run under the same band.

    The qwen4 `NONE` cells (world 4, hidden 2560) run under it too, measured
    on GB300 with one seed per cell. The one-shot kernel (1, 128 and 1024 rows)
    is bitwise the reference, an fp32 sum of four bf16 partials rounded once.
    The NCCL path (8192 rows) reads 2.59 ulp per element and 0.76 ulp RMS. A
    sum missing one rank reads 189 / 148 ulp on the one-shot kernel and
    230 / 148 ulp on NCCL, at least 37x the band on either gate.

    `norm_out` measures closer to a norm of the fp32 sum than to a norm of
    the bf16-rounded `residual_out`, so the reference's norm reads the fp32
    sum. Most of the RMS is the two bf16 roundings being compared: the
    reference's and the kernel's.

    A dropped or doubled rank moves the result by tens of ulp;
    `check_the_gate_discriminates_a_wrong_reduction` checks the band refuses
    one, and `check_none_gate_discriminates_a_missing_rank` does so for `NONE`
    on both of its kernels. World 8 runs only inside one B200 or B300 node,
    and its cells pass this band there; the per-element error was not
    recorded. Across GB300 trays every message takes the path whose error
    grows with the rank count, so a cross-tray cell needs a new measurement
    (`comm/mnnvl_allreduce` is the entry for that topology).
    """
    # Imported here, not at module scope: _validating imports tensorrt_llm,
    # which the launcher must not, and the rank body puts it on sys.path.
    from _validating import assert_within_ulp

    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    assert len(outs) == len(refs), (len(outs), len(refs))
    for o, r in zip(outs, refs):
        assert_within_ulp(o, r, element_ulp=5.0, rms_ulp=1.25)


def _measure(out: torch.Tensor, ref: torch.Tensor):
    """The two quantities `_compare` gates, in ulp, plus the cosine similarity:
    what rank 0 prints so the band's derivation can be read off the log."""
    from _validating import ULP

    ulp = ULP[ref.dtype]
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    row_scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row_scale).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    cosine = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    return element, rms, cosine


def _report(label: str, out, ref) -> None:
    if RANK != 0:
        return
    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    for k, (o, r) in enumerate(zip(outs, refs)):
        element, rms, cosine = _measure(o, r)
        print(
            f"[measure] world={WORLD} {label} out{k}: element {element:.3f} ulp, "
            f"rms {rms:.3f} ulp, cosine {cosine:.9f}",
            flush=True,
        )


class _HostExchange:
    """The one call `IpcMemory` makes on `Distributed`: a TP-group allgather of
    host objects."""

    def tp_allgather(self, obj: Any) -> List[Any]:
        gathered: List[Any] = [None] * WORLD
        dist.all_gather_object(gathered, obj, group=HOST_GROUP)
        return gathered


def _payload(rank: int, rows: int, hidden: int, seed: int) -> torch.Tensor:
    """One rank's partial sum: bf16 values of the scale a projection produces.

    Every rank can regenerate every other rank's payload from the seed, which
    is what lets the reference be arithmetic rather than a second collective.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)


def _residual_and_weight(rows: int, hidden: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + 7)
    residual = torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16)
    return residual, weight


def _norm_operands(spec, rows: int, seed: int):
    """`(residual, norm_weight, eps)` for a cell: real tensors under
    RESIDUAL_RMS_NORM, None under NONE, which reads neither."""
    if spec["op"] == entry.NONE:
        return None, None, 0.0
    residual, weight = _residual_and_weight(rows, spec["hidden"], seed)
    return residual, weight, spec["eps"]


def _call(x, residual, weight, eps, op):
    return allreduce(
        x,
        residual=residual,
        norm_weight=weight,
        scale=None,
        bias=None,
        op=op,
        eps=eps,
        trigger_completion_at_end=False,
    )


def _reference(rows, hidden, seed, residual, weight, eps, op):
    inputs = [_payload(r, rows, hidden, seed) for r in range(WORLD)]
    return entry.allreduce.reference(inputs, residual, weight, eps, op)


def _cells():
    return [c for c in entry.allreduce.CELLS if c.spec["world"] == WORLD]


def _norm_cells():
    return [c for c in _cells() if c.spec["op"] == entry.RESIDUAL_RMS_NORM]


def _none_cells():
    return [c for c in _cells() if c.spec["op"] == entry.NONE]


def check_certified_cells() -> None:
    """Every cell at this world size, inside `validating`, against the entry's
    reference and this file's band."""
    cells = _cells()
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        # The payloads are bf16 and the communicator arguments were built
        # once with the target's strategy: a cell naming anything else would
        # be certified at values it never ran.
        assert spec["dtype"] is torch.bfloat16, (cell.why, spec["dtype"])
        assert spec["strategy"] == GROUP_ARGS["strategy"], (cell.why, spec["strategy"])
        seed = 4000 + 17 * i
        x = _payload(RANK, spec["rows"], spec["hidden"], seed)
        residual, weight, eps = _norm_operands(spec, spec["rows"], seed)
        with validating(entry.allreduce):
            out = _call(x, residual, weight, eps, spec["op"])
        ref = _reference(spec["rows"], spec["hidden"], seed, residual, weight, eps, spec["op"])
        _report(cell.why, out, ref)
        try:
            _compare(out, ref)
        except AssertionError as exc:
            raise AssertionError(f"{cell.why}: {exc}") from exc
        dist.barrier()


def check_inputs_are_untouched() -> None:
    """The fused form returns fresh tensors and writes neither argument --
    the opposite contract from flashinfer_fused_add_rmsnorm, which a target
    swaps this entry in for."""
    spec = _norm_cells()[0].spec
    x = _payload(RANK, 32, spec["hidden"], 61)
    residual, weight = _residual_and_weight(32, spec["hidden"], 61)
    x0, r0 = x.clone(), residual.clone()
    norm_out, residual_out = _call(x, residual, weight, spec["eps"], spec["op"])
    torch.cuda.synchronize()
    assert torch.equal(x, x0), "input was written"
    assert torch.equal(residual, r0), "residual was written"
    for out in (norm_out, residual_out):
        assert out.data_ptr() not in (x.data_ptr(), residual.data_ptr()), "output aliases an input"
    dist.barrier()


def check_none_inputs_are_untouched() -> None:
    """`NONE` returns one fresh tensor and leaves `input` as it was, on both
    kernels AUTO picks for it -- the target reads its partial again only
    through the sum, but a kernel writing the partial in place would also
    alias the next layer's buffers."""
    if not _none_cells():
        return
    hidden = _none_cells()[0].spec["hidden"]
    for rows in NONE_PATH_ROWS:
        x = _payload(RANK, rows, hidden, 63 + rows)
        x0 = x.clone()
        out = _call(x, None, None, 0.0, entry.NONE)
        torch.cuda.synchronize()
        assert isinstance(out, torch.Tensor), f"NONE returned {type(out)}, not one tensor"
        assert out.shape == x.shape and out.dtype == x.dtype, (out.shape, out.dtype)
        assert torch.equal(x, x0), f"input was written at {rows} rows"
        assert out.data_ptr() != x.data_ptr(), f"output aliases the input at {rows} rows"
        dist.barrier()


def check_none_ranks_agree_bitwise() -> None:
    """Every rank gets the same bits back.

    Load-bearing for the qwen4 target, not a nicety: under tep4 each rank
    routes the reduced rows to its own 128-expert window from its own copy of
    the sum. If two ranks' copies differed by one bit, a top-10 tie could be
    broken differently on each, and a token would be served by 9 or 11
    experts with no error raised.
    """
    if not _none_cells():
        return
    hidden = _none_cells()[0].spec["hidden"]
    for rows in (1, 128, 1024, 8192):
        x = _payload(RANK, rows, hidden, 71 + rows)
        out = _call(x, None, None, 0.0, entry.NONE).contiguous()
        gathered = [torch.empty_like(out) for _ in range(WORLD)]
        dist.all_gather(gathered, out)
        for r, other in enumerate(gathered):
            differing = (other.view(torch.int16) != out.view(torch.int16)).sum().item()
            assert differing == 0, (
                f"{rows} rows: rank {RANK} and rank {r} differ in {differing} elements"
            )
        dist.barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """One graph per decode batch size, sharing a pool, replayed after the
    input buffers are refilled: the engine's decode path."""
    spec = _norm_cells()[0].spec
    hidden, eps, op = spec["hidden"], spec["eps"], spec["op"]
    pool = None
    graphs = {}
    for rows in GRAPH_BATCH_SIZES:
        x = _payload(RANK, rows, hidden, 700 + rows)
        residual, weight = _residual_and_weight(rows, hidden, 700 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _call(x, residual, weight, eps, op)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outs = _call(x, residual, weight, eps, op)
        persistent = tuple(o.clone() for o in outs)
        if pool is None:
            pool = graph.pool()
        graphs[rows] = (graph, x, residual, weight, outs, persistent)
        dist.barrier()

    for step, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
        graph, x, residual, weight, outs, _ = graphs[rows]
        seed = 900 + step
        x.copy_(_payload(RANK, rows, hidden, seed))
        new_residual, _ = _residual_and_weight(rows, hidden, seed)
        residual.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(rows, hidden, seed, residual, weight, eps, op)
        try:
            _compare(tuple(o.clone() for o in outs), ref)
        except AssertionError as exc:
            raise AssertionError(f"replay rows={rows}: {exc}") from exc
        dist.barrier()


def check_none_cuda_graph_replay_at_target_row_counts() -> None:
    """The qwen4 target's captured plain sums: one graph per row count it
    captures under tp4 / tep4 (`NONE_GRAPH_ROWS`), sharing a pool, each
    replayed twice after its input buffer is refilled -- a graph that baked
    in its capture-time payload, or a Lamport buffer left in the wrong phase
    by the previous replay, fails the second time."""
    if not _none_cells():
        return
    hidden = _none_cells()[0].spec["hidden"]
    pool = None
    graphs = {}
    for rows in NONE_GRAPH_ROWS:
        x = _payload(RANK, rows, hidden, 1700 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _call(x, None, None, 0.0, entry.NONE)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            out = _call(x, None, None, 0.0, entry.NONE)
        if pool is None:
            pool = graph.pool()
        graphs[rows] = (graph, x, out)
        dist.barrier()

    for rnd in range(2):
        for step, rows in enumerate(reversed(NONE_GRAPH_ROWS)):
            graph, x, out = graphs[rows]
            seed = 1900 + 50 * rnd + step
            x.copy_(_payload(RANK, rows, hidden, seed))
            graph.replay()
            torch.cuda.synchronize()
            ref = _reference(rows, hidden, seed, None, None, 0.0, entry.NONE)
            got = out.clone()
            if rnd == 0 and rows in (1, 1024):
                _report(f"NONE graph replay rows={rows}", got, ref)
            try:
                _compare(got, ref)
            except AssertionError as exc:
                raise AssertionError(f"NONE replay {rnd} rows={rows}: {exc}") from exc
            dist.barrier()


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference that drops one rank's contribution must fail the band;
    otherwise the band passes anything."""
    spec = _norm_cells()[0].spec
    rows, hidden = 64, spec["hidden"]
    x = _payload(RANK, rows, hidden, 81)
    residual, weight = _residual_and_weight(rows, hidden, 81)
    out = _call(x, residual, weight, spec["eps"], spec["op"])
    short = [_payload(r, rows, hidden, 81) for r in range(WORLD - 1)]
    wrong = entry.allreduce.reference(short, residual, weight, spec["eps"], spec["op"])
    try:
        _compare(out, wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("the band accepted a sum that is missing one rank")
    dist.barrier()


def check_none_gate_discriminates_a_missing_rank() -> None:
    """The same negative control for `NONE`, on each kernel AUTO picks for it:
    a sum missing the last rank's partial must fail the band. Rank 0 prints
    how far outside the band it reads."""
    if not _none_cells():
        return
    hidden = _none_cells()[0].spec["hidden"]
    for rows in NONE_PATH_ROWS:
        seed = 83 + rows
        x = _payload(RANK, rows, hidden, seed)
        out = _call(x, None, None, 0.0, entry.NONE)
        short = [_payload(r, rows, hidden, seed) for r in range(WORLD - 1)]
        wrong = entry.allreduce.reference(short, None, None, 0.0, entry.NONE)
        _report(f"NONE missing-rank control rows={rows}", out, wrong)
        try:
            _compare(out, wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"the band accepted a {rows}-row sum that is missing one rank")
        dist.barrier()


def check_wrapper_guards() -> None:
    """The guard refuses a strided input and a missing workspace, under each
    fusion this world certifies.

    `is_valid` is called directly rather than through a guarded op call: a
    guard that wrongly admitted the missing workspace would hand the kernel a
    null pointer.
    """
    specs = [_norm_cells()[0].spec] + [c.spec for c in _none_cells()[:1]]
    for spec in specs:
        hidden = spec["hidden"]
        wide = _payload(RANK, 8, 2 * hidden, 91)
        strided = wide[:, :hidden]
        residual, weight, eps = _norm_operands(spec, 8, 91)
        for bad, match in (
            (dict(input=strided), "contiguous"),
            (dict(workspace=None), "workspace"),
        ):
            kwargs = dict(GROUP_ARGS)
            kwargs.update(
                input=_payload(RANK, 8, hidden, 91),
                residual=residual,
                norm_weight=weight,
                scale=None,
                bias=None,
                op=spec["op"],
                eps=eps,
                trigger_completion_at_end=False,
            )
            kwargs.update(bad)
            try:
                entry.allreduce.is_valid(**kwargs)
            except AssertionError as exc:
                assert match in str(exc), (match, str(exc))
            else:
                raise AssertionError(f"is_valid accepted {sorted(bad)} under op {spec['op']}")
    dist.barrier()


CHECKS = (
    check_certified_cells,
    check_inputs_are_untouched,
    check_none_inputs_are_untouched,
    check_none_ranks_agree_bitwise,
    check_cuda_graph_replay_at_engine_batch_sizes,
    check_none_cuda_graph_replay_at_target_row_counts,
    check_the_gate_discriminates_a_wrong_reduction,
    check_none_gate_discriminates_a_missing_rank,
    check_wrapper_guards,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero
    and the launcher kills its siblings."""
    global RANK, WORLD, HOST_GROUP, GROUP_ARGS, allreduce, entry, validating
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

    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as entry
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.mapping import Mapping

    if RANK == 0:
        print(f"[rank 0] tensorrt_llm from {tensorrt_llm.__file__}", flush=True)
        print(f"[rank 0] entry from {entry.__file__}", flush=True)
        print(
            f"[rank 0] device {torch.cuda.get_device_name(rank)}, "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability(rank)))}",
            flush=True,
        )

    # Every rank is on this one node, so the node-local group is all of them.
    local_group = dist.new_group(ranks=list(range(world)), backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())

    communicator.Distributed.get = staticmethod(lambda mapping: _HostExchange())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
    # What a target does in __init__: build the communicator arguments once.
    # The partial is only the call surface, the way the allgather matrix uses
    # one: the guard is armed on the entry itself, and `reference` and `CELLS`
    # are read off `entry.allreduce` directly.
    GROUP_ARGS = entry.allreduce_group_args(mapping)
    allreduce = functools.partial(entry.allreduce, **GROUP_ARGS)

    # Spawned worker, not a pytest module: reach the sibling helper in
    # tests/unittest/_torch/modeling_v2 (comm -> modeling_v2) directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    for check in CHECKS:
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
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
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
    # See the allgather matrix: inheriting the launching srun step's PMI
    # namespace lets one failing rank take the whole step down unreported.
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
