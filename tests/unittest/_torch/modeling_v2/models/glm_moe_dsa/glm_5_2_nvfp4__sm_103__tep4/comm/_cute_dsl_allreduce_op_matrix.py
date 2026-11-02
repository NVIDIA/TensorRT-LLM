# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the GLM-5.2 target's three Lamport all-reduce entries.

`cute_dsl_allreduce`, `cute_dsl_allreduce_add_rmsnorm` and
`cute_dsl_allreduce_add_rmsnorm_quant_nvfp4` share one workspace -- the
symmetric-memory peer buffers and the device slot counter every launch reads
and advances -- so they are certified together, on one workspace, with their
calls interleaved the way a forward interleaves them.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it spawns one worker process per device named in
CUDA_VISIBLE_DEVICES (four, the target's TP width), under a deadline the parent
enforces by killing every worker's session. A Lamport collective whose ranks
disagree hangs rather than raising, so the deadline is what keeps a broken
kernel from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _cute_dsl_allreduce_op_matrix.py

The workers are plain processes: no MPI. The workspace brings its process
group up from the rendezvous address this launcher puts in the environment.

Not a pytest module, despite the `check_*` bodies: they are one ordered
sequence inside a single 4-rank job, reading rank state only `_run_one_rank`
binds, and the slot-counter checks depend on every launch before them. The
collected entry point is `test_modeling_v2_cute_dsl_allreduce_op_matrix.py`
beside this file. This file imports only torch and the standard library at
module scope: `_rank_job` executes it to read `DEADLINE_S`.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
#: Four CuTe DSL compiles per rank, then the cells, the largest of which build
#: fp64 references over 8192 x 6144. Read by `_rank_job.run`.
DEADLINE_S = 1800

WORLD_SIZE = 4
HIDDEN = 6144
#: GLM-5.2's `rms_norm_eps`.
EPS = 1e-5
#: The target's token budget: the workspace is built for it, and it is the
#: largest row count any entry is called with.
MAX_ROWS = 8192
SF_VEC = 16

# The payload scales. Four partials of std 0.5 sum to std 1, the residual
# is std 1, so the sum and the residual are the same size and neither hides
# the other's rounding. The norm weight is 1 +- 1/8, the spread the band
# derivations below assume.
_PARTIAL_STD = 0.5
_RESIDUAL_STD = 1.0
_WEIGHT_SPREAD = 0.25

# Bound in the rank body, never at module scope: importing the entries pulls
# in tensorrt_llm, and `_rank_job` executes this file to read DEADLINE_S.
RANK = 0
WORLD = 1
WS: Any = None
TWOSHOT_ABOVE = 0
pure: Any = None
fused: Any = None
quant: Any = None
fp4_quantize: Any = None
fp4_quantize_mod: Any = None
validating: Any = None
assert_within_ulp: Any = None


# ---- the host mirror of the device slot counter --------------------------

# `state` is `[entry counter, current slot, rows*hidden/8 last published per
# slot (3), release epoch]`. Every launch of any of the four kernels takes
# the current slot, records how many 16-byte vectors it published there, and
# moves the slot on by one mod 3; the entry counter returns to 0 when the
# launch completes. The epoch moves only under the `strict` PDL policy, which
# no kernel of the workspace uses.
_slot = 0
_counts = [0, 0, 0]


def _launched(rows: int) -> None:
    global _slot
    _counts[_slot] = rows * HIDDEN // 8
    _slot = (_slot + 1) % 3


def _assert_state(where: str) -> None:
    torch.cuda.synchronize()
    got = WS.state.tolist()
    want = [0, _slot, *_counts, 0]
    assert got == want, f"{where}: device slot state {got}, expected {want}"


# ---- payloads -------------------------------------------------------------


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def _randn(rows: int, seed: int, std: float) -> torch.Tensor:
    # Every rank regenerates every rank's payload from the seed: the CUDA
    # generator is deterministic per (seed, shape), so the reference is
    # arithmetic on this rank rather than a second collective.
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, HIDDEN, generator=g, device="cuda") * std).to(torch.bfloat16)


def _partials(rows: int, seed: int) -> List[torch.Tensor]:
    """Every rank's partial, in rank order."""
    return [_randn(rows, seed * 977 + r + 1, _PARTIAL_STD) for r in range(WORLD)]


def _residual(rows: int, seed: int) -> torch.Tensor:
    """The residual stream: replicated, identical on every rank, as in the target."""
    return _randn(rows, seed * 977 + 101, _RESIDUAL_STD)


def _weight(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed * 977 + 202)
    u = torch.rand(HIDDEN, generator=g, device="cuda") - 0.5
    return (1.0 + u * _WEIGHT_SPREAD).to(torch.bfloat16)


def _global_scale(norm: torch.Tensor) -> torch.Tensor:
    """An fp32 `[1]` NVFP4 global scale the payload neither saturates nor sits on a grid of.

    The canonical `448 * 6 / amax` of this very batch makes every block scale
    `448 * block_amax / amax`, a ratio of two bf16 values times 448, which lands
    exactly on an e4m3 rounding midpoint in up to 6% of blocks; there the
    kernel's `gs * amax * (1/6)` and the reference's `gs * (amax / 6)` round
    the last fp32 bit differently and pick opposite neighbours. The target's
    scale comes from calibration, not from the batch, so it is no such
    fraction: 0.93 of the canonical value stands in for one (a calibration
    amax a little above this batch's). Computed from the reference norm, so it
    is the same on every rank.
    """
    return (448.0 * 6.0 * 0.93 / norm.float().abs().max()).reshape(1)


# ---- gates ----------------------------------------------------------------


def _assert_same_on_every_rank(t: torch.Tensor, where: str) -> None:
    """Every rank sums in the same order, so every rank's result is bit-identical."""
    bits = t.contiguous().view(torch.uint8).to(torch.int32)
    lo, hi = bits.clone(), bits.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    assert torch.equal(lo, hi), f"{where}: the ranks hold different results"


def _gate_pure(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    # Bit-exact: the kernel adds the four bf16 partials in fp32 in rank order
    # and rounds once, which is the reference's arithmetic step for step.
    assert torch.equal(out, ref), f"{where}: the sum is not bit-exact"


def _gate_fused(
    rows: int,
    norm: torch.Tensor,
    new_residual: torch.Tensor,
    ref_norm: torch.Tensor,
    ref_residual: torch.Tensor,
    where: str,
) -> None:
    """The bands, and why.

    One-shot (rows <= TWOSHOT_ABOVE): the residual is bf16(fp32 sum in rank
    order + residual), the reference's arithmetic exactly, so it is bit-exact.
    The norm differs from the fp64 reference only through the kernel's fp32
    sum of squares, its approximate rsqrt and its fp32 products, about 1e-6
    relative before the bf16 rounding. That moves an element only where it
    sits on a bf16 rounding boundary, and then by one ulp of that element:
    at most 2^-7 of it, 2 units of 2^-8 against the row's largest. So 2 ulp
    element-wise; and since about 1e-6 / 2^-8 of the elements move, 0.5 ulp
    relative RMS is some ten times what that rate gives.

    Two-shot (rows > TWOSHOT_ABOVE): the reduced rows are gathered as bf16, so
    the residual is bf16(bf16(sum) + residual), rounded twice. Against the
    single rounding that is at most half an ulp of the sum plus an ulp of the
    result: 1 + 2 units of the row's largest when the sum is no larger than
    the residual stream it lands in (both std 1 here), so 4 ulp element-wise.
    Double rounding moves roughly a quarter to a half of the elements by one
    ulp, about 1 ulp relative RMS, so 2. And the residual must NOT be
    bit-exact: that is what shows the two-shot kernel ran. The norm inherits
    the residual's difference scaled by the weight, at most 4 x (9/7) units
    with the weight in 1 +- 1/8, plus its own 2: 8 ulp element-wise, 2 RMS.
    """
    if rows <= TWOSHOT_ABOVE:
        assert torch.equal(new_residual, ref_residual), f"{where}: one-shot residual not exact"
        assert_within_ulp(norm, ref_norm, element_ulp=2.0, rms_ulp=0.5)
    else:
        assert_within_ulp(new_residual, ref_residual, element_ulp=4.0, rms_ulp=2.0)
        assert not torch.equal(new_residual, ref_residual), (
            f"{where}: a two-shot residual rounded once: the one-shot kernel ran above "
            f"{TWOSHOT_ABOVE} rows"
        )
        assert_within_ulp(norm, ref_norm, element_ulp=8.0, rms_ulp=2.0)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] packed bytes -> [M, K] e2m1 codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _scale_near_tie(norm: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """`[rows, hidden/16]` blocks whose e4m3 scale the gate declines to predict.

    The kernel forms `gs * amax * (1/6)` and the reference `gs * (amax / 6)`:
    a few fp32 ulps apart. A block whose scale lands within `TIE_WINDOW` of an
    e4m3 rounding midpoint can round either way; anywhere else the two match.
    """
    rows = norm.shape[0]
    vecmax = norm.float().reshape(rows, -1, SF_VEC).abs().amax(-1)
    unq = gs.float().reshape(()) * (vecmax / 6.0)
    w = fp4_quantize_mod.TIE_WINDOW
    lo = (unq * (1 - w)).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    hi = (unq * (1 + w)).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    return lo != hi


def _gate_quant(
    norm: torch.Tensor,
    gs: torch.Tensor,
    fp4_out: torch.Tensor,
    scale_out: torch.Tensor,
    ref: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    where: str,
) -> None:
    """The NVFP4 payload is `fp4_quantize` of the bf16 norm the kernel wrote.

    Gate 1, against the kernel's own norm: scales equal and codes equal, except
    in the two places no reference can call -- an element within `fp4_quantize`'s
    tie window of an e2m1 midpoint (the kernel's approximate reciprocals pick
    the side), and every element of a block whose scale is near an e4m3 tie.
    Gate 2, against the entry's reference: wherever the reference norm block
    equals the kernel's bit for bit the quantize input is identical, so the
    same exactness holds there. The excluded fraction is bounded: 2.5% of
    elements, twice the worst of 120 CPU draws of this payload, so the window
    cannot grow into a licence to be wrong.
    """
    rows = norm.shape[0]
    blocks = HIDDEN // SF_VEC
    sf_tie = _scale_near_tie(norm, gs)
    el_tie = fp4_quantize.near_tie(fp4_quantize.scaled(norm, gs, SF_VEC))
    excluded = el_tie | sf_tie.repeat_interleave(SF_VEC, dim=1)
    assert excluded.sum().item() <= 0.025 * excluded.numel(), (
        f"{where}: {excluded.sum().item()} of {excluded.numel()} elements in a tie window"
    )
    codes, sfs = _unpack(fp4_out), scale_out.view(torch.uint8)

    own_fp4, own_sf = fp4_quantize.reference(norm, gs, SF_VEC, False, False)
    own_sf = own_sf.view(rows, blocks)
    assert torch.equal(sfs[~sf_tie], own_sf[~sf_tie]), f"{where}: block scales differ"
    own_codes = _unpack(own_fp4)
    assert torch.equal(codes[~excluded], own_codes[~excluded]), f"{where}: codes differ"

    ref_norm, _, ref_fp4, ref_sf = ref
    same = (norm.view(rows, blocks, SF_VEC) == ref_norm.view(rows, blocks, SF_VEC)).all(-1)
    keep_sf = same & ~sf_tie
    assert torch.equal(sfs[keep_sf], ref_sf.view(torch.uint8)[keep_sf]), (
        f"{where}: block scales differ from the reference where the norms agree"
    )
    keep = keep_sf.repeat_interleave(SF_VEC, dim=1) & ~el_tie
    assert torch.equal(codes[keep], _unpack(ref_fp4)[keep]), (
        f"{where}: codes differ from the reference where the norms agree"
    )


# ---- one call of each entry, with its inputs and its check ----------------


class _Call:
    """One entry call: owns its buffers, so it can be refilled and replayed."""

    def __init__(self, kind: str, rows: int) -> None:
        self.kind, self.rows = kind, rows
        dev = "cuda"
        self.partial = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.residual = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.weight = torch.empty(HIDDEN, dtype=torch.bfloat16, device=dev)
        self.gs = torch.empty(1, dtype=torch.float32, device=dev)
        self.fp4 = torch.empty(rows, HIDDEN // 2, dtype=torch.uint8, device=dev)
        self.sf = torch.empty(rows, HIDDEN // SF_VEC, dtype=torch.uint8, device=dev)
        self._inputs: Optional[tuple] = None

    def fill(self, seed: int) -> None:
        """Fresh inputs for `seed`; NaN in the outputs, so an unwritten element fails."""
        partials = _partials(self.rows, seed)
        residual, weight = _residual(self.rows, seed), _weight(seed)
        self.partial.copy_(partials[RANK])
        self.residual.copy_(residual)
        self.weight.copy_(weight)
        self.out.fill_(float("nan"))
        self.fp4.fill_(0xAA)
        self.sf.fill_(0xAA)
        if self.kind == "quant":
            norm, _ = fused.reference(partials, residual, weight, self.out, WS)
            self.gs.copy_(_global_scale(norm))
        self._inputs = (partials, residual, weight)

    def launch(self) -> None:
        if self.kind == "pure":
            assert pure(self.partial, self.out, WS) is self.out
        elif self.kind == "fused":
            assert fused(self.partial, self.residual, self.weight, self.out, WS) is self.out
        else:
            got = quant(
                self.partial, self.residual, self.weight, self.gs, self.out, self.fp4, self.sf, WS
            )
            assert got is self.out

    def check(self, where: str) -> None:
        partials, residual, weight = self._inputs
        where = f"{where}: {self.kind} {self.rows} rows"
        if self.kind == "pure":
            _gate_pure(self.out, pure.reference(partials, self.out, WS), where)
        elif self.kind == "fused":
            ref_norm, ref_res = fused.reference(partials, residual, weight, self.out, WS)
            _gate_fused(self.rows, self.out, self.residual, ref_norm, ref_res, where)
        else:
            ref = quant.reference(
                partials, residual, weight, self.gs, self.out, self.fp4, self.sf, WS
            )
            _gate_fused(self.rows, self.out, self.residual, ref[0], ref[1], where)
            _gate_quant(self.out, self.gs, self.fp4, self.sf, ref, where)
            _assert_same_on_every_rank(self.fp4, where)
            _assert_same_on_every_rank(self.sf, where)
        _assert_same_on_every_rank(self.out, where)
        if self.kind != "pure":
            _assert_same_on_every_rank(self.residual, where)


def _entry(kind: str) -> Any:
    return {"pure": pure, "fused": fused, "quant": quant}[kind]


# ---- checks ---------------------------------------------------------------


def check_the_workspace_agreed_and_launched_nothing() -> None:
    """Every rank agreed the same multicast decision, and building launched no kernel."""
    flags: List[Any] = [None] * WORLD
    dist.all_gather_object(flags, WS.multicast_base != 0)
    assert len(set(flags)) == 1, f"ranks disagree on multicast: {flags}"
    assert WS.max_rows == MAX_ROWS and WS.hidden == HIDDEN and WS.world == WORLD
    _assert_state("after build")
    if RANK == 0:
        print(f"[rank 0] multicast publish {'on' if flags[0] else 'off'}", flush=True)


def check_every_cell_interleaved_on_one_workspace() -> None:
    """Drive every cell of the three entries, inside `validating`, round-robin.

    One call of each entry in turn, so consecutive launches change entry and
    row count -- a 1-row pure sum after an 8192-row fused one reuses a slot
    the larger call filled, which is what the per-slot vector counts in the
    device state exist to clear correctly. The slot state is checked after
    every call.
    """
    queues = [[("pure", c) for c in pure.CELLS], [("fused", c) for c in fused.CELLS]]
    queues.append([("quant", c) for c in quant.CELLS])
    order: List[Tuple[str, Any]] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    for kind, cell in order:
        rows = cell.spec["rows"]
        call = _Call(kind, rows)
        call.fill(_seed(cell.why))
        torch.cuda.synchronize()
        dist.barrier()
        with validating(_entry(kind)):
            call.launch()
        _launched(rows)
        _assert_state(f"cell {kind} {cell.why!r}")
        call.check(f"cell {cell.why!r}")
        dist.barrier()


def check_cuda_graph_replays_rotate_the_slot_counter() -> None:
    """One capture of five launches across the three entries, replayed four times.

    The kernels take their slot from the device counter and advance it
    themselves, so a captured sequence must land on a different slot each
    replay: five launches is not a multiple of three, so replay `k` starts on
    slot `(start + 5k) mod 3`. A slot read on the host at capture would replay
    the same three-slot pattern every time; the state check after each replay
    and the outputs, checked on fresh inputs every replay, tell the two apart.
    Then one eager call continues from wherever the graph left the counter.
    """
    plan = [("pure", 1), ("fused", 64), ("quant", 2), ("fused", 1), ("pure", 64)]
    calls = [_Call(kind, rows) for kind, rows in plan]
    base = _seed("cuda graph")
    for c in calls:
        c.fill(base)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        for c in calls:
            c.launch()
    _assert_state("after capture: capturing launches nothing")
    dist.barrier()
    for replay in range(4):
        for c in calls:
            c.fill(base + 1 + replay)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        for c in calls:
            _launched(c.rows)
        _assert_state(f"replay {replay}")
        for c in calls:
            c.check(f"replay {replay}")
        dist.barrier()

    eager = _Call("fused", 64)
    eager.fill(base + 100)
    torch.cuda.synchronize()
    dist.barrier()
    eager.launch()
    _launched(eager.rows)
    _assert_state("eager call after the replays")
    eager.check("eager call after the replays")
    del graph
    dist.barrier()


def check_guards_refuse_a_row_count_above_the_budget() -> None:
    """Above `max_rows` the kernel would write past every peer's buffer.

    The guard raises before anything launches, on every rank alike, so the
    group stays in step.
    """
    rows = MAX_ROWS + 1
    big = _Call("quant", rows)
    for kind in ("pure", "fused", "quant"):
        try:
            with validating(_entry(kind)):
                big.kind = kind
                big.launch()
        except AssertionError as exc:
            assert "budget" in str(exc), exc
        else:
            raise AssertionError(f"{kind} accepted {rows} rows")
    _assert_state("after the refused calls")
    dist.barrier()


def check_quant_guard_refuses_what_the_kernel_misreads() -> None:
    """A per-token global scale, and NVFP4 outputs that are not the compact
    `[rows, ...]` buffers the kernel indexes through raw pointers."""
    c = _Call("quant", 4)
    cases = (
        ("exactly one element", dict(gs=torch.ones(4, device="cuda"))),
        ("contiguous", dict(fp4=torch.empty(4, HIDDEN, dtype=torch.uint8, device="cuda")[:, ::2])),
        ("must be", dict(sf=torch.empty(2, HIDDEN // SF_VEC, dtype=torch.uint8, device="cuda"))),
    )
    for match, override in cases:
        args = {"gs": c.gs, "fp4": c.fp4, "sf": c.sf, **override}
        try:
            with validating(quant):
                quant(
                    c.partial, c.residual, c.weight, args["gs"], c.out, args["fp4"], args["sf"], WS
                )
        except AssertionError as exc:
            assert match in str(exc), exc
        else:
            raise AssertionError(f"the quant guard accepted a case it should refuse ({match})")
    _assert_state("after the refused quant calls")
    dist.barrier()


CHECKS = (
    check_the_workspace_agreed_and_launched_nothing,
    check_every_cell_interleaved_on_one_workspace,
    check_cuda_graph_replays_rotate_the_slot_counter,
    check_guards_refuse_a_row_count_above_the_budget,
    check_quant_guard_refuses_what_the_kernel_misreads,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero
    and the launcher kills its siblings."""
    global RANK, WORLD, WS, TWOSHOT_ABOVE, pure, fused, quant
    global fp4_quantize, fp4_quantize_mod, validating, assert_within_ulp
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization import (
        fp4_quantize as _fp4_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm import (  # noqa: E501
        _lamport,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm.cute_dsl_allreduce import (  # noqa: E501
        cute_dsl_allreduce,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm.cute_dsl_allreduce_add_rmsnorm import (  # noqa: E501
        cute_dsl_allreduce_add_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import (  # noqa: E501
        cute_dsl_allreduce_add_rmsnorm_quant_nvfp4,
    )

    # This file runs as a spawned worker, not under pytest, so the repo's
    # `__extra_import_path__` is not in play: reach the shared helper the
    # plain way. Confined to these worker processes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import assert_within_ulp as _within
    from _validating import validating as _validating_cm

    RANK, WORLD = rank, world
    assert WORLD == WORLD_SIZE, f"the entries are built for TP{WORLD_SIZE}, got {WORLD} ranks"
    # No MPI here: the group comes up from the address the launcher exported.
    _lamport.ensure_process_group(rank, world)
    assert dist.is_initialized() and dist.get_world_size() == world
    WS = _lamport.build_workspace(HIDDEN, EPS, MAX_ROWS, rank, world)
    TWOSHOT_ABOVE = _lamport.TWOSHOT_ABOVE
    pure = cute_dsl_allreduce
    fused = cute_dsl_allreduce_add_rmsnorm
    quant = cute_dsl_allreduce_add_rmsnorm_quant_nvfp4
    fp4_quantize_mod = _fp4_mod
    fp4_quantize = _fp4_mod.fp4_quantize
    validating, assert_within_ulp = _validating_cm, _within

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective
            # early wedges every sibling in it, and the launcher kills the
            # rest the moment it sees this nonzero exit.
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    # Exit without tearing the group or the symmetric memory down: these are
    # throwaway processes the launcher owns.
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """Start one worker process per claimed device; no external launcher.

    The workers find each other through the address and port exported here.
    Each child gets its own session; on the first nonzero exit or on the
    deadline, every session is killed, which is what keeps one wedged
    collective from outliving the run.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == WORLD_SIZE, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); these entries need {WORLD_SIZE}"
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
    # Drop a launching srun step's PMIx/PMI/OMPI namespace from the children:
    # inherited, it registers them as participants of that step, and the first
    # hard exit trips srun's errhandler, which kills the whole step before
    # anything is reported. These ranks rendezvous through torch.distributed.
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
