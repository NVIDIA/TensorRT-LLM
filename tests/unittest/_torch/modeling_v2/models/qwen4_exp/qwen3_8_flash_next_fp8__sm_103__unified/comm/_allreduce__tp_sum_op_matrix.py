# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the qwen4 FP8 target's `allreduce__tp_sum` variant.

The shared entry's matrix (`modeling_v2/comm/_allreduce_op_matrix.py`), copied
with the variant's cells and the claims its `tp_sum` call site rests on. It
runs every check over both communicators `allreduce_group_args` can route the
call to:

  torch.distributed  one worker process per device, `TLLM_DISABLE_MPI=1`: the
                     `allreduce_pg` symbol on the TP process group, which the
                     target takes when `mpi_disabled()`.
  MPI session        the same checks under `mpirun`, `process_group=None`: the
                     plain `allreduce` symbol on `getComm`'s communicator, with
                     the IPC workspace exchanged by the engine's own `MPIDist`.
                     This is what the target calls when it serves under
                     `mpirun` + `trtllm-llmapi-launch`.

Run plainly, this script is its own launcher: it starts one rank per device,
first over torch.distributed and then under mpirun, and kills the whole process
group on the first failure or at the deadline, because a broken collective
hangs rather than raising. The MPI half is adapted from the qwen4 FP8 target's
allgather__dp_gather matrix (`_spawn_mpi_ranks`, `_run_one_mpi_rank`) and runs
its ranks through the shared `comm/_lockstep.py`.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce__tp_sum_op_matrix.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce__tp_sum_op_matrix.py --world-size 2

The world size is `--world-size`, else the number of visible devices; a run
certifies the cells whose `world` matches it, on the first that many devices.
The collected entry point (`test_fp8_unified_op_matrices.py` beside this
file) runs 2 and 4. Rank 0 prints each cell's measured error (bf16
ulp, max abs, max abs over max |ref|, the bit-exact fraction), which
`_rank_job` shows only on failure; run this file directly to read them.

Not a pytest module: the `check_*` bodies are one ordered sequence inside one
job and read rank state that only `_run_one_rank` binds. This file imports only
torch and the standard library at module scope: `_rank_job` executes it to read
`DEADLINE_S`.

The IPC workspace needs one host-side exchange of memory handles across the TP
group. The engine does that through `Distributed.get(mapping)`. Under MPI that
is `MPIDist`, which the MPI transport uses as it is. Under MPI-disabled mode it
is `TorchDist`, which requires Ray; the torch.distributed ranks rendezvous
through torch.distributed alone, so `_HostExchange` stands in for it with a
gloo group. It carries only that exchange; the allreduce itself runs on the
real NCCL communicator and the real IPC buffers. `TorchDist` also registers the
world and node-local process groups with the C++ side (`init_pg`), which the
`_pg` symbol reads on its first call; the rank body registers them the same way.
"""

import argparse
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
_MPI_WORKER_FLAG = "--mpi-rank-worker"
PG_DEADLINE_S = 900
MPI_DEADLINE_S = 900
# The whole job's budget, which `_rank_job` reads: both transports run in one job.
DEADLINE_S = PG_DEADLINE_S + MPI_DEADLINE_S

# Bound inside the rank body, never in the launcher: importing the entry pulls
# in tensorrt_llm, and the launcher must carry no CUDA or MPI state.
allreduce: Any = None
entry: Any = None
validating: Any = None
ULP: Any = None
GROUP_ARGS: dict = {}
RANK = 0
WORLD = 1
HOST_GROUP: Any = None
#: "pg" (torch.distributed, the `allreduce_pg` symbol) or "mpi" (MPI session, the
#: plain `allreduce` symbol).
TRANSPORT = "pg"
# The MPI-session rank (`_lockstep.Rank`); None on the torch.distributed transport.
MPI_RANK: Any = None

# Decode graph batch sizes a served gpt-oss or deepseek default engine
# captures, sampled: the smallest, a power of two, a size that is not one, and
# the cap.
GRAPH_BATCH_SIZES = (1, 8, 13, 128)

# The row counts this target captures `tp_sum` at. Every rank holds every row
# under attention TP, so a captured decode graph's row count is its batch size:
# the 19 batch sizes the tp2/tep2/tp4/tep4 engines capture (1, 2, 4, 8, then
# 16..128 by 8). Then the MTP verify steps at tep4: 128 requests x 4 rows
# (MTP3) and x 8 rows (MTP7).
ENGINE_BATCH_SIZES = (1, 2, 4, 8) + tuple(range(16, 129, 8))
TP_SUM_GRAPH_ROWS = ENGINE_BATCH_SIZES + (512, 1024)

#: tests/unittest/_torch/modeling_v2: `_validating` and the shared `comm/` helpers.
_MODELING_V2_TESTS = Path(__file__).resolve().parents[4]


def _compare(out, ref) -> None:
    """5 ulp per element against the row scale, 1.25 ulp relative RMS, on
    every output.

    The shared entry's band, measured there on GB300 with the gpt-oss payloads
    (hidden 2880, bf16, 20 seeds, 1 to 8192 rows): at most 2.56 ulp per element
    and 0.91 ulp RMS up to 2048 rows; at world 4 and 8192 rows, where AUTO takes
    the NCCL fallback, 4.00 and 0.97.

    The tp_sum cells (op NONE, hidden 2560) sit under the same band. The
    measured numbers are in the variant's `note`. NCCL rounds each partial sum
    to bf16, so its error grows with the rank count. The one-shot kernel sums
    in fp32 and rounds once, which leaves a plain sum at most one rounding from
    the reference.

    A dropped rank moves the result by tens of ulp;
    `check_the_gate_discriminates_a_wrong_reduction` holds that the band
    refuses one.
    """
    # Imported here, not at module scope: _validating imports tensorrt_llm,
    # which the launcher must not, and the rank body puts it on sys.path.
    from _validating import assert_within_ulp

    outs, refs = _as_tuple(out), _as_tuple(ref)
    assert len(outs) == len(refs), (len(outs), len(refs))
    for o, r in zip(outs, refs):
        assert_within_ulp(o, r, element_ulp=5.0, rms_ulp=1.25)


def _ulp_stats(out, ref) -> tuple[float, float]:
    """`_compare`'s two statistics, worst over the outputs: (per element against
    the row scale, relative RMS), both in ulp of the output dtype."""
    elt = rms = 0.0
    for o, r in zip(_as_tuple(out), _as_tuple(ref)):
        ulp = ULP[r.dtype]
        o2, r2 = o.float().reshape(-1, o.shape[-1]), r.float().reshape(-1, r.shape[-1])
        scale = r2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        elt = max(elt, ((o2 - r2).abs() / scale).max().item() / ulp)
        rel = (o2 - r2).pow(2).mean().sqrt() / r2.pow(2).mean().sqrt().clamp_min(1e-9)
        rms = max(rms, rel.item() / ulp)
    return elt, rms


def _abs_stats(out, ref) -> tuple[float, float, float]:
    """Worst over the outputs: the largest absolute error, that error over the
    reference's largest magnitude, and the fraction of elements bit-identical
    to the reference (the fp32 sum rounded once)."""
    max_abs = max_rel = 0.0
    exact = 1.0
    for o, r in zip(_as_tuple(out), _as_tuple(ref)):
        diff = (o.float() - r.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, (diff.max() / r.float().abs().max().clamp_min(1e-9)).item())
        same = (o.view(torch.int16) == r.view(torch.int16)).float().mean().item()
        exact = min(exact, same)
    return max_abs, max_rel, exact


def _as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def _barrier() -> None:
    """Every rank reaches this point, on the transport's own host-visible barrier."""
    if TRANSPORT == "mpi":
        MPI_RANK.barrier()
    else:
        dist.barrier()


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


def _residual_and_weight(rows: int, hidden: int, seed: int, op: int):
    """The fused form's residual and norm weight; the plain sum takes neither,
    and `tp_sum` passes None for both."""
    if op == entry.NONE:
        return None, None
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + 7)
    residual = torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16)
    return residual, weight


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


def _cells(op=None):
    return [
        c
        for c in entry.allreduce.CELLS
        if c.spec["world"] == WORLD and (op is None or c.spec["op"] == op)
    ]


def _first_spec_of_each_op():
    """One spec per op this world's cells hold: the fused form's and tp_sum's."""
    specs = {}
    for cell in _cells():
        specs.setdefault(cell.spec["op"], cell.spec)
    return list(specs.values())


def _report(what: str, out, ref) -> None:
    if RANK == 0:
        elt, rms = _ulp_stats(out, ref)
        max_abs, max_rel, exact = _abs_stats(out, ref)
        print(
            f"[world {WORLD} {TRANSPORT}] {what}: {elt:.2f} ulp element, {rms:.3f} ulp RMS; "
            f"max abs {max_abs:.3e}, max abs / max|ref| {max_rel:.3e}, "
            f"{100 * exact:.4f}% bit-exact",
            flush=True,
        )


def check_certified_cells() -> None:
    """Every cell at this world size, inside `validating`, against the entry's
    reference and this file's band."""
    cells = _cells()
    assert cells, f"no cell is certified at world size {WORLD}"
    assert _cells(entry.NONE), f"no tp_sum cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        seed = 4000 + 17 * i
        x = _payload(RANK, spec["rows"], spec["hidden"], seed)
        residual, weight = _residual_and_weight(spec["rows"], spec["hidden"], seed, spec["op"])
        with validating(entry.allreduce):
            out = _call(x, residual, weight, spec["eps"], spec["op"])
        ref = _reference(
            spec["rows"], spec["hidden"], seed, residual, weight, spec["eps"], spec["op"]
        )
        _report(cell.why, out, ref)
        try:
            _compare(out, ref)
        except AssertionError as exc:
            raise AssertionError(f"{cell.why}: {exc}") from exc
        _barrier()


def _same_bits_on_every_rank(out) -> bool:
    """Whether every rank holds bit-identical outputs, so -0/+0 and NaN payloads
    count. Over torch.distributed the outputs are gathered and compared as raw
    16-bit words; over the MPI session each rank's bytes are hashed and the
    digests compared (`_lockstep.Rank.same_on_ranks`)."""
    if TRANSPORT == "mpi":
        return MPI_RANK.same_on_ranks(*_as_tuple(out))
    for t in _as_tuple(out):
        gathered = [torch.empty_like(t) for _ in range(WORLD)]
        dist.all_gather(gathered, t.contiguous())
        words = [g.view(torch.int16) for g in gathered]
        if not all(torch.equal(words[0], w) for w in words[1:]):
            return False
    return True


def check_every_rank_gets_the_same_bits() -> None:
    """The plain sum hands every rank the same bits, at every tp_sum cell.

    `tp_sum` completes a partial whose result every rank then carries as its
    own copy of the hidden state, and the routed expert windows rely on every
    rank routing each row the same way: a last-bit difference between ranks
    can move a top-k choice on one rank only, and then a row's experts are
    summed twice or not at all. Every rank evaluates the same gathered
    comparison, so a mismatch fails every rank at once.
    """
    for i, cell in enumerate(_cells(entry.NONE)):
        spec = cell.spec
        seed = 6000 + 17 * i
        x = _payload(RANK, spec["rows"], spec["hidden"], seed)
        out = _call(x, None, None, spec["eps"], spec["op"])
        assert _same_bits_on_every_rank(out), f"{cell.why}: ranks got different bits"
        _barrier()


def check_inputs_are_untouched() -> None:
    """Both forms return fresh tensors and write no argument -- the fused form
    the opposite contract from flashinfer_fused_add_rmsnorm, which a target
    swaps it in for."""
    for spec in _first_spec_of_each_op():
        op = spec["op"]
        x = _payload(RANK, 32, spec["hidden"], 61)
        residual, weight = _residual_and_weight(32, spec["hidden"], 61, op)
        x0 = x.clone()
        r0 = None if residual is None else residual.clone()
        outs = _as_tuple(_call(x, residual, weight, spec["eps"], op))
        torch.cuda.synchronize()
        assert len(outs) == (1 if op == entry.NONE else 2), (op, len(outs))
        assert torch.equal(x, x0), f"op {op}: input was written"
        if residual is not None:
            assert torch.equal(residual, r0), f"op {op}: residual was written"
        held = {x.data_ptr()} | ({residual.data_ptr()} if residual is not None else set())
        for out in outs:
            assert out.data_ptr() not in held, f"op {op}: output aliases an input"
        _barrier()


def _graph_replay(spec, row_counts) -> None:
    """One graph per row count, sharing a pool, replayed after the input
    buffers are refilled: the engine's decode path."""
    hidden, eps, op = spec["hidden"], spec["eps"], spec["op"]
    pool = None
    graphs = {}
    for rows in row_counts:
        x = _payload(RANK, rows, hidden, 700 + rows)
        residual, weight = _residual_and_weight(rows, hidden, 700 + rows, op)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _call(x, residual, weight, eps, op)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _barrier()
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outs = _as_tuple(_call(x, residual, weight, eps, op))
        if pool is None:
            pool = graph.pool()
        graphs[rows] = (graph, x, residual, weight, outs)
        _barrier()

    for step, rows in enumerate(reversed(row_counts)):
        graph, x, residual, weight, outs = graphs[rows]
        seed = 900 + step
        x.copy_(_payload(RANK, rows, hidden, seed))
        if residual is not None:
            new_residual, _ = _residual_and_weight(rows, hidden, seed, op)
            residual.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(rows, hidden, seed, residual, weight, eps, op)
        got = tuple(o.clone() for o in outs)
        got = got if len(got) > 1 else got[0]
        _report(f"op {op} graph replay, {rows} rows", got, ref)
        try:
            _compare(got, ref)
        except AssertionError as exc:
            raise AssertionError(f"op {op} replay rows={rows}: {exc}") from exc
        if op == entry.NONE:
            assert _same_bits_on_every_rank(got), f"replay rows={rows}: ranks got different bits"
        _barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """The fused form at the default targets' decode batch sizes, and tp_sum at
    the row counts this target captures."""
    for spec in _first_spec_of_each_op():
        rows = TP_SUM_GRAPH_ROWS if spec["op"] == entry.NONE else GRAPH_BATCH_SIZES
        _graph_replay(spec, rows)


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference that drops one rank's contribution must fail the band, for
    both forms; otherwise the band passes anything."""
    for spec in _first_spec_of_each_op():
        rows, hidden, op = 64, spec["hidden"], spec["op"]
        x = _payload(RANK, rows, hidden, 81)
        residual, weight = _residual_and_weight(rows, hidden, 81, op)
        out = _call(x, residual, weight, spec["eps"], op)
        short = [_payload(r, rows, hidden, 81) for r in range(WORLD - 1)]
        wrong = entry.allreduce.reference(short, residual, weight, spec["eps"], op)
        if RANK == 0:
            elt, rms = _ulp_stats(out, wrong)
            print(
                f"[world {WORLD}] op {op}, one rank dropped from the reference: "
                f"{elt:.1f} ulp element, {rms:.1f} ulp RMS",
                flush=True,
            )
        try:
            _compare(out, wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"op {op}: the band accepted a sum that is missing one rank")
        _barrier()


def check_wrapper_guards() -> None:
    """The guard refuses a strided input and a missing workspace, for both
    forms.

    `is_valid` is called directly rather than through a guarded op call: a
    guard that wrongly admitted the missing workspace would hand the kernel a
    null pointer.
    """
    for spec in _first_spec_of_each_op():
        hidden, op = spec["hidden"], spec["op"]
        wide = _payload(RANK, 8, 2 * hidden, 91)
        strided = wide[:, :hidden]
        residual, weight = _residual_and_weight(8, hidden, 91, op)
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
                op=op,
                eps=spec["eps"],
                trigger_completion_at_end=False,
            )
            kwargs.update(bad)
            try:
                entry.allreduce.is_valid(**kwargs)
            except AssertionError as exc:
                assert match in str(exc), (match, str(exc))
            else:
                raise AssertionError(f"op {op}: is_valid accepted {sorted(bad)}")
        _barrier()


def check_the_call_reaches_this_transports_symbol() -> None:
    """The tp_sum call, with the arguments `allreduce_group_args` built for this
    transport, launches exactly one op: `torch.ops.trtllm.allreduce` (the MPI
    session, what a served engine runs) or `torch.ops.trtllm.allreduce_pg`
    (torch.distributed). Both symbols are wrapped with a counter for one call and
    restored afterwards, so the certified cells are known to have run the symbol
    this transport names, not only to have been routed toward it."""
    namespace = torch.ops.trtllm
    names = ("allreduce", "allreduce_pg")
    # Resolving each symbol first caches it on the namespace, which is what the
    # counter replaces and the `finally` puts back.
    originals = {name: getattr(namespace, name) for name in names}
    calls = dict.fromkeys(names, 0)

    def counting(name):
        original = originals[name]

        def call(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return call

    spec = _cells(entry.NONE)[0].spec
    x = _payload(RANK, 8, spec["hidden"], 71)
    try:
        for name in names:
            setattr(namespace, name, counting(name))
        out = _call(x, None, None, spec["eps"], spec["op"])
        torch.cuda.synchronize()
    finally:
        for name in names:
            setattr(namespace, name, originals[name])
    want = "allreduce" if TRANSPORT == "mpi" else "allreduce_pg"
    assert calls == {name: int(name == want) for name in names}, (TRANSPORT, calls)
    _compare(out, _reference(8, spec["hidden"], 71, None, None, spec["eps"], spec["op"]))
    _barrier()


CHECKS = (
    check_certified_cells,
    check_the_call_reaches_this_transports_symbol,
    check_every_rank_gets_the_same_bits,
    check_inputs_are_untouched,
    check_cuda_graph_replay_at_engine_batch_sizes,
    check_the_gate_discriminates_a_wrong_reduction,
    check_wrapper_guards,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero
    and the launcher kills its siblings."""
    global RANK, WORLD, HOST_GROUP, GROUP_ARGS, ULP, allreduce, entry, validating
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

    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        allreduce__tp_sum as entry,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.mapping import Mapping

    # Every rank is on this one node, so the node-local group is all of them.
    local_group = dist.new_group(ranks=list(range(world)), backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())

    communicator.Distributed.get = staticmethod(lambda mapping: _HostExchange())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
    # What the target does in `_tp_sum_op`: build the communicator arguments
    # once. The guard is armed on the entry itself, and `reference` and
    # `CELLS` are read off `entry.allreduce` directly.
    GROUP_ARGS = entry.allreduce_group_args(mapping)
    assert GROUP_ARGS["process_group"] is not None, "MPI disabled, yet no process group"
    allreduce = functools.partial(entry.allreduce, **GROUP_ARGS)

    # Spawned worker, not a pytest module: reach the shared helpers directly.
    sys.path.insert(0, str(_MODELING_V2_TESTS))
    from _validating import ULP as _ulp
    from _validating import validating as _validating_cm

    validating, ULP = _validating_cm, _ulp

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
    print(f"[rank {RANK}] {len(CHECKS)} checks passed over torch.distributed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _run_one_mpi_rank() -> int:
    """Body of one MPI rank (started by `_spawn_mpi_ranks`): every check on the
    plain symbol, through the shared `_lockstep` rank and check runner. A failing
    check aborts the MPI job, which is what ends its siblings."""
    global RANK, WORLD, GROUP_ARGS, ULP, TRANSPORT, MPI_RANK, allreduce, entry, validating
    os.environ.pop("TLLM_DISABLE_MPI", None)
    sys.path.insert(0, str(_MODELING_V2_TESTS))
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import _lockstep
    from _validating import ULP as _ulp
    from _validating import validating as _validating_cm

    TRANSPORT = "mpi"
    # Binds the MPI world, the rank's device (rank % visible devices) and the TP
    # Mapping over every rank, and imports tensorrt_llm, whose MPI session the
    # plain symbol builds its communicator on.
    MPI_RANK = _lockstep.Rank(_lockstep.parse_args(sys.argv[1:]))
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        allreduce__tp_sum as entry,
    )
    from tensorrt_llm._utils import mpi_disabled, mpi_world_size

    # `allreduce_group_args` picks the symbol by `mpi_disabled()`: this is the launch
    # under which it passes None, as the target's `_tp_sum_op` does when it serves
    # under mpirun, and the IPC workspace is exchanged by the engine's own MPIDist.
    assert not mpi_disabled(), "the MPI transport ran with MPI disabled"
    assert mpi_world_size() == MPI_RANK.world, (mpi_world_size(), MPI_RANK.world)
    RANK, WORLD = MPI_RANK.rank, MPI_RANK.world
    GROUP_ARGS = entry.allreduce_group_args(MPI_RANK.mapping)
    assert GROUP_ARGS["process_group"] is None and GROUP_ARGS["rank"] is None, GROUP_ARGS
    allreduce = functools.partial(entry.allreduce, **GROUP_ARGS)
    validating, ULP = _validating_cm, _ulp

    code = _lockstep.run_checks(MPI_RANK, list(CHECKS))
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed over the MPI session", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Finalize MPI, then leave without running static destructors: tearing down an
    # NCCL communicator that captured collectives into CUDA graphs is what the
    # torch.distributed rank above avoids too, and mpirun counts a rank that exits
    # without MPI_Finalize as a failed job.
    MPI_RANK.MPI.Finalize()
    os._exit(code)


def _spawn_ranks(world_size: int | None) -> None:
    """One worker per device, rendezvous over env://, all killed on the first
    nonzero exit or at the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    devices = [d for d in visible.split(",") if d.strip()]
    world_size = len(devices) if world_size is None else world_size
    assert 2 <= world_size <= len(devices), (
        f"a {world_size}-rank run needs 2..{len(devices)} of the visible devices {devices}"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(devices[:world_size]),
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

    deadline = time.monotonic() + PG_DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {PG_DEADLINE_S}s (wedged)"
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
        raise AssertionError(f"the {world_size}-rank torch.distributed run failed: {failure}")


#: The Slurm step variables an mpirun started inside a one-task Slurm step must not
#: inherit. With them OpenMPI sizes its slots from the step (one) and every rank reads
#: the step's own rank; without them it runs the ranks on this node.
_SLURM_STEP_VARS = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)


def _spawn_mpi_ranks(world_size: int | None) -> None:
    """Start one MPI rank per device under mpirun, with a deadline that kills the
    whole process group.

    Adapted from `_lockstep.spawn` (tests/unittest/_torch/modeling_v2/comm/_lockstep.py),
    with `--oversubscribe` and the Slurm step's variables dropped, so the same command
    runs on a CI node and inside a one-task Slurm step. `TLLM_DISABLE_MPI` is dropped
    too: the plain symbol needs tensorrt_llm's MPI session.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    devices = [d for d in visible.split(",") if d.strip()]
    world_size = len(devices) if world_size is None else world_size
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(world_size),
        sys.executable,
        str(Path(__file__).resolve()),
        _MPI_WORKER_FLAG,
        "--world-size",
        str(world_size),
    ]
    env = {
        k: v for k, v in os.environ.items() if k not in _SLURM_STEP_VARS and k != "TLLM_DISABLE_MPI"
    }
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:world_size])
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=MPI_DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world_size}-rank MPI run did not finish in {MPI_DEADLINE_S}s (wedged)"
        ) from None
    assert code == 0, f"the {world_size}-rank MPI run exited {code}"


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_mpi_rank())
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=None)
    world_size = parser.parse_args().world_size
    _spawn_ranks(world_size)
    print("OK: torch.distributed (allreduce_pg)", flush=True)
    _spawn_mpi_ranks(world_size)
    print("OK: MPI session (allreduce)", flush=True)
    print("OK")
