# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the GLM-5.2 target's mnnvl_hop entry.

A collective cannot be exercised in one process, so this script is its own launcher (see
``_mnnvl_ranks``): run plainly it spawns one rank per device in CUDA_VISIBLE_DEVICES,

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_hop_op_matrix.py

and ``--rank-worker`` runs one rank of a world started elsewhere (the 16-rank c1 world). The
checks are one fixed sequence on one MNNVL all-reduce workspace, which each check leaves rotated
for the next; collected as tests they would run at world size 1, so neither this file's name nor
its functions match pytest's patterns. The collected entry point is
``test_modeling_v2_mnnvl_hop_op_matrix.py`` beside this file.

Beyond the entry's cells: exact sums, the Lamport flags, -0.0 rows, the published shard,
interleaving with the built-in MNNVL all-reduce on its workspace, CUDA graph replays, ranks
arriving late, and the guards. On a world whose one-shot all-reduce stays under 1 MiB the built-in
op computes the same values in the same order, so the hop matches it bit for bit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import _mnnvl_ranks as ranks
import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "mnnvl_hop requires CUDA devices"

DEADLINE_S = 1200
HIDDEN = 6144
EPS = 1e-5

# Bound in the rank body: the entry and the workspace helpers import tensorrt_llm.
entry: Any = None
ws: Any = None
validating: Any = None
fusion_op: Any = None
RANK = 0
WORLD = 1
MAPPING: Any = None


def _rows(rows: int, seed: int, rank: int) -> torch.Tensor:
    """One rank's partial rows: their sum over the ranks has unit variance."""
    return ranks.normal((rows, HIDDEN), seed * 1009 + rank, scale=WORLD**-0.5)


def _residual(rows: int, seed: int, zero: bool = False) -> torch.Tensor:
    """The residual stream: the same on every rank, as on the target."""
    if zero:
        return torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    return ranks.normal((rows, HIDDEN), seed * 1009 + 500)


def _gamma(seed: int) -> torch.Tensor:
    """An RMSNorm weight near 1, the same on every rank."""
    return (1.0 + ranks.normal(HIDDEN, seed, scale=0.1, dtype=torch.float32)).to(torch.bfloat16)


def _compare(norm, residual_out, ref_norm, ref_residual, where: str) -> None:
    """Bands of the hop against the FP32 reference.

    residual_out: the kernel rounds the FP32 sum over the ranks to BF16 and adds the residual in
    BF16, two roundings of half an ulp, each of a value at most the row's largest (inputs scaled
    so that the sum and the residual are of one size): at most about 2 ulp of the row scale per
    element, 3 allowed; about 0.6 ulp of relative RMS, 1.5 allowed.

    norm_out: residual_out's error carried through the scale (gamma within 0.6-1.4), the sum of
    squares of the BF16 values (each square rounded to BF16, below 1/4 ulp on the scale) and the
    output rounding (half an ulp): at most about 3.5 ulp per element, 5 allowed; about 0.7 ulp of
    relative RMS, 1.5 allowed. Derived, not fitted: a misplaced slot or rank misses by orders.
    """
    from _validating import assert_within_ulp

    try:
        assert_within_ulp(residual_out, ref_residual, element_ulp=3.0, rms_ulp=1.5)
        assert_within_ulp(norm, ref_norm, element_ulp=5.0, rms_ulp=1.5)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def _builtin_one_shot(rows: int) -> bool:
    """Whether the built-in MNNVL all-reduce runs one-shot here, in the hops' reduction order."""
    return rows * HIDDEN * WORLD * torch.bfloat16.itemsize <= ranks.BUILTIN_ONE_SHOT_BYTES


def _builtin_hop(x, residual, gamma, comm_buffer, buffer_flags) -> Tuple[torch.Tensor, ...]:
    """The built-in MNNVL residual + RMSNorm all-reduce on the same workspace."""
    norm, residual_out = torch.ops.trtllm.mnnvl_fusion_allreduce(
        x, gamma, residual, EPS, comm_buffer, buffer_flags, True, None, fusion_op
    )
    return norm, residual_out


def _word_offsets(rows: int) -> torch.Tensor:
    """Offsets, in 32-bit words of a Lamport buffer, of this rank's slot of every row."""
    tokens = torch.arange(rows, device="cuda").view(-1, 1)
    pairs = torch.arange(0, HIDDEN, 2, device="cuda").view(1, -1)
    return (((tokens * WORLD + RANK) * HIDDEN + pairs) // 2).view(-1)


def _publish(x: torch.Tensor, offsets: torch.Tensor | None = None) -> None:
    """Store this rank's rows into every rank's current Lamport buffer, as the o_proj publish does.

    The GEMM's epilogue reads the current buffer from the flags and stores its BF16 output, -0.0
    as +0.0, at [token][rank][column]. Done here with 32-bit stores through the multicast view:
    a word is the unit the hop's poll tests, so a half-written slot is never taken for a full one.
    The index comes from the flags on the device, so this also runs inside a CUDA graph.
    """
    rows = x.shape[0]
    lamport, flags = ws.publish_target(MAPPING, rows, HIDDEN)
    words = lamport.view(torch.int32)
    if offsets is None:
        offsets = _word_offsets(rows)
    base = flags[0].long() * (flags[2].long() // 4)
    payload = torch.where(x == 0, torch.zeros_like(x), x).contiguous().view(torch.int32)
    words.index_copy_(0, base + offsets, payload.view(-1))


def _hop(x, residual, gamma, shard_published: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, x.shape[0], HIDDEN)
    if shard_published:
        _publish(x)
    return entry.mnnvl_hop(x, residual, gamma, EPS, comm_buffer, buffer_flags, shard_published)


def check_certified_cells() -> None:
    """Every cell: the band against the reference, the same bits on every rank, and the bits of
    the built-in one-shot all-reduce where it runs one-shot."""
    gamma = _gamma(7)
    cells = [(cell.spec, cell.why) for cell in entry.mnnvl_hop.CELLS]
    cells.extend(
        (
            dict(rows=rows, hidden=HIDDEN, shard_published=published, zero_residual=zero),
            f"TP8 rows={rows}, published={published}, zero_residual={zero}",
        )
        for rows in (2, 4)
        for published, zero in ((True, False), (False, False), (False, True))
    )
    for i, (spec, why) in enumerate(cells):
        rows, published, zero = spec["rows"], spec["shard_published"], spec["zero_residual"]
        assert spec["hidden"] == HIDDEN, why
        seed = 100 + 13 * i
        per_rank = [_rows(rows, seed, r) for r in range(WORLD)]
        x, residual = per_rank[RANK], _residual(rows, seed, zero)
        comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        if published:
            _publish(x)
        with validating(entry.mnnvl_hop):
            norm, residual_out = entry.mnnvl_hop(
                x, residual, gamma, EPS, comm_buffer, buffer_flags, published
            )
        ref_norm, ref_residual = entry.mnnvl_hop.reference(per_rank, residual, gamma, EPS)
        _compare(norm, residual_out, ref_norm, ref_residual, why)
        ranks.assert_same_on_every_rank(norm, f"{why}: norm_out")
        ranks.assert_same_on_every_rank(residual_out, f"{why}: residual_out")
        if _builtin_one_shot(rows):
            builtin_norm, builtin_residual = _builtin_hop(
                x, residual, gamma, comm_buffer, buffer_flags
            )
            ranks.assert_bitwise(norm, builtin_norm, f"{why}: norm_out vs built-in")
            ranks.assert_bitwise(residual_out, builtin_residual, f"{why}: residual_out")
        dist.barrier()


def check_exact_sums_are_exact() -> None:
    """Rows on the 1/8 grid sum exactly: residual_out equals the reference bit for bit.

    Any slot read from the wrong rank, row or column, or summed twice, moves an exact result.
    """
    gamma = _gamma(11)
    for rows, published in ((8, False), (8, True), (1, True), (3, False)):
        seed = 2000 + rows + 7 * published
        # |row| <= 1 per rank, |sum| <= 16 on 16 ranks, |residual| <= 8: all exact in BF16.
        per_rank = [ranks.eighths((rows, HIDDEN), seed * 31 + r, bound=8) for r in range(WORLD)]
        residual = ranks.eighths((rows, HIDDEN), seed * 31 + 97, bound=64)
        norm, residual_out = _hop(per_rank[RANK], residual, gamma, published)
        ref_norm, ref_residual = entry.mnnvl_hop.reference(per_rank, residual, gamma, EPS)
        where = f"exact rows={rows} published={published}"
        ranks.assert_bitwise(residual_out, ref_residual, where)
        _compare(norm, residual_out, ref_norm, ref_residual, where)
        dist.barrier()


def check_lamport_flags_rotate() -> None:
    """After each hop the current buffer is the dirty one, with the hop's bytes to clear."""
    gamma = _gamma(13)
    for step, rows in enumerate((8, 2, 4, 1, 8)):
        published = step % 2 == 1
        _, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        before = ranks.lamport_flags(buffer_flags)
        _hop(_rows(rows, 3000 + step, RANK), _residual(rows, 3000 + step), gamma, published)
        after = ranks.lamport_flags(buffer_flags)
        bytes_to_clear = rows * HIDDEN * WORLD * torch.bfloat16.itemsize
        ranks.assert_rotated(before, after, bytes_to_clear, f"step {step}")
        dist.barrier()


def check_negative_zero_rows_arrive_as_positive_zero() -> None:
    """-0.0 marks an empty slot; rows of -0.0 must neither hang the exchange nor sum to -0.0."""
    gamma = _gamma(17)
    for published in (False, True):
        x = torch.full((8, HIDDEN), -0.0, dtype=torch.bfloat16, device="cuda")
        residual = ranks.eighths((8, HIDDEN), 4000, bound=64)
        # A residual of -0.0 plus a sum of +0.0 is +0.0, and any other residual is itself.
        residual = torch.where(residual == 0, torch.ones_like(residual), residual)
        _, residual_out = _hop(x, residual, gamma, published)
        ranks.assert_bitwise(residual_out, residual, f"-0.0 rows published={published}")
        dist.barrier()


def check_interleaved_with_the_builtin_all_reduce() -> None:
    """The hops and the built-in MNNVL all-reduce (one-shot and two-shot) share one workspace.

    They read, clear and rotate the same flags; every result stays right whichever kernel ran
    before it.
    """
    gamma = _gamma(19)
    for round_ in range(3):
        seed = 5000 + 100 * round_
        for kind, rows in (
            ("hop", 8),
            ("builtin", 2),
            ("published", 1),
            ("builtin_two_shot", 64),
            ("published", 8),
            ("hop", 1),
        ):
            per_rank = [_rows(rows, seed + rows, r) for r in range(WORLD)]
            residual = _residual(rows, seed + rows)
            if kind.startswith("builtin"):
                # 64 rows are two-shot for the built-in all-reduce; a workspace that holds the
                # rows one-shot holds them two-shot too.
                comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
                norm, residual_out = _builtin_hop(
                    per_rank[RANK], residual, gamma, comm_buffer, buffer_flags
                )
            else:
                norm, residual_out = _hop(per_rank[RANK], residual, gamma, kind == "published")
            ref_norm, ref_residual = entry.mnnvl_hop.reference(per_rank, residual, gamma, EPS)
            _compare(norm, residual_out, ref_norm, ref_residual, f"round {round_} {kind} {rows}")
        dist.barrier()


def check_cuda_graph_replays() -> None:
    """Replay the production o_proj publish and current-workspace hops at full/partial batch."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.o_proj_publish import (  # noqa: E501
        o_proj_publish,
    )

    gamma = _gamma(23)
    k = 16384 // WORLD
    shapes = ((8, 2), (4, 1), (8, 1)) if WORLD == 8 else ((8, 1),)
    addresses = ranks.workspace_addresses(MAPPING, HIDDEN)
    states = []

    def chain(state):
        rows, draft_rows = state["shape"]
        lamport, flags = ws.publish_target(MAPPING, rows, HIDDEN)
        o_proj_publish(state["x"], state["weight"], state["layout"], lamport, flags, WORLD, RANK)
        comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        norm1, res1 = entry.mnnvl_hop(
            state["layout"], state["residual"], gamma, EPS, comm_buffer, buffer_flags, True
        )
        norm2, res2 = _hop(norm1, res1, gamma, False)
        lamport, flags = ws.publish_target(MAPPING, draft_rows, HIDDEN)
        o_proj_publish(
            state["draft_x"], state["weight"], state["draft_layout"], lamport, flags, WORLD, RANK
        )
        comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, draft_rows, HIDDEN)
        norm3, res3 = entry.mnnvl_hop(
            state["draft_layout"],
            res2[:draft_rows].contiguous(),
            gamma,
            EPS,
            comm_buffer,
            buffer_flags,
            True,
        )
        return norm1, res1, norm2, res2, norm3, res3

    for rows, draft_rows in shapes:
        state = dict(
            shape=(rows, draft_rows),
            x=ranks.normal((rows, k), 6000 + RANK),
            weight=ranks.normal((HIDDEN, k), 6010 + RANK, scale=(k * WORLD) ** -0.5),
            layout=torch.full((rows, HIDDEN), 71.0, dtype=torch.bfloat16, device="cuda"),
            residual=_residual(rows, 6000),
            draft_x=ranks.normal((draft_rows, k), 6001 + RANK),
            draft_layout=torch.full(
                (draft_rows, HIDDEN), 73.0, dtype=torch.bfloat16, device="cuda"
            ),
        )
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            chain(state)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        state["graph"] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(state["graph"]):
            state["outputs"] = chain(state)
        states.append(state)
        assert ranks.workspace_addresses(MAPPING, HIDDEN) == addresses

    for replay in range(8):
        state = states[replay % len(states)]
        rows, draft_rows = state["shape"]
        seed = 6100 + 10 * replay
        state["x"].copy_(ranks.normal((rows, k), seed + RANK))
        state["weight"].copy_(
            ranks.normal((HIDDEN, k), seed + 30 + RANK, scale=(k * WORLD) ** -0.5)
        )
        state["residual"].copy_(_residual(rows, seed))
        state["draft_x"].copy_(ranks.normal((draft_rows, k), seed + 1 + RANK))
        local_published = o_proj_publish.reference(
            state["x"], state["weight"], state["layout"], None, None, WORLD, RANK
        )
        per_rank = [torch.empty_like(local_published) for _ in range(WORLD)]
        dist.all_gather(per_rank, local_published)
        torch.cuda.synchronize()
        dist.barrier()
        state["graph"].replay()
        torch.cuda.synchronize()
        norm1, res1, norm2, res2, norm3, res3 = state["outputs"]
        ref = entry.mnnvl_hop.reference(per_rank, state["residual"], gamma, EPS)
        _compare(norm1, res1, *ref, f"replay {replay} production publish rows={rows}")
        assert torch.all(state["layout"] == 71.0), "o_proj_publish wrote its layout-only output"
        gathered = [torch.empty_like(norm1) for _ in range(WORLD)]
        dist.all_gather(gathered, norm1)
        ref = entry.mnnvl_hop.reference(gathered, res1, gamma, EPS)
        _compare(norm2, res2, *ref, f"replay {replay} hop")
        local_draft = o_proj_publish.reference(
            state["draft_x"], state["weight"], state["draft_layout"], None, None, WORLD, RANK
        )
        per_rank_draft = [torch.empty_like(local_draft) for _ in range(WORLD)]
        dist.all_gather(per_rank_draft, local_draft)
        assert torch.all(state["draft_layout"] == 73.0)
        ref = entry.mnnvl_hop.reference(per_rank_draft, res2[:draft_rows], gamma, EPS)
        _compare(norm3, res3, *ref, f"replay {replay} draft rows={draft_rows}")
        ranks.assert_same_on_every_rank(norm3, f"replay {replay} draft hop")
        assert ranks.workspace_addresses(MAPPING, HIDDEN) == addresses
        dist.barrier()
    del states
    torch.cuda.synchronize()


def check_late_ranks() -> None:
    """Ranks reach the hops at different times; back-to-back hops still exchange the right rows.

    Rank r sleeps on its stream before each hop, so the early ranks poll while the late ones
    have not published, and a fast rank's next hop publishes into a buffer a slow rank has only
    just emptied.
    """
    gamma = _gamma(29)
    issued: List[Tuple[Any, ...]] = []
    for step in range(6):
        rows = (8, 1)[step % 2]
        published = step % 3 == 0
        seed = 7000 + step
        per_rank = [_rows(rows, seed, r) for r in range(WORLD)]
        residual = _residual(rows, seed)
        torch.cuda._sleep(((RANK * 7 + step * 3) % WORLD) * 400_000)
        issued.append((per_rank, residual, *_hop(per_rank[RANK], residual, gamma, published)))
    torch.cuda.synchronize()
    for step, (per_rank, residual, norm, residual_out) in enumerate(issued):
        ref_norm, ref_residual = entry.mnnvl_hop.reference(per_rank, residual, gamma, EPS)
        _compare(norm, residual_out, ref_norm, ref_residual, f"late ranks step {step}")


def check_the_op_refuses_unsupported_shapes() -> None:
    """Shapes outside hop_supported are refused loudly, before any rank enters the exchange."""
    assert ws.hop_supported(8, HIDDEN, WORLD)
    assert ws.hop_supported(1, HIDDEN, WORLD)
    # On 16 ranks: 8 rows of 6144 are the 1.5 MiB one-shot limit; 9 rows are above it.
    assert ws.hop_supported(8, HIDDEN, 16) and not ws.hop_supported(9, HIDDEN, 16)
    # 6152 columns are 769 eight-element vectors: no cluster of 2 or more CTAs splits them.
    assert not ws.hop_supported(1, HIDDEN + 8, WORLD)
    assert not ws.hop_supported(1, HIDDEN, 32)
    too_many = 1
    while ws.hop_supported(too_many, HIDDEN, WORLD):
        too_many += 1
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, 1, HIDDEN)
    for rows, hidden in ((too_many, HIDDEN), (1, HIDDEN + 8)):
        x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device="cuda")
        g = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
        try:
            entry.mnnvl_hop(x, x, g, EPS, comm_buffer, buffer_flags, False)
        except RuntimeError as exc:
            assert "unsupported shape" in str(exc), exc
        else:
            raise AssertionError(f"the op took {rows} x {hidden} rows")


def check_the_guard_refuses_flags_of_another_workspace() -> None:
    """Flags that do not describe comm_buffer's buffers are refused by is_valid."""
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, 8, HIDDEN)
    foreign = buffer_flags.clone()
    foreign.view(torch.int32)[2] += 16
    x = _rows(8, 8000, RANK)
    try:
        with validating(entry.mnnvl_hop):
            entry.mnnvl_hop(x, x, _gamma(37), EPS, comm_buffer, foreign, False)
    except AssertionError as exc:
        assert "flags of comm_buffer's workspace" in str(exc), exc
    else:
        raise AssertionError("is_valid admitted foreign flags")


def check_the_guard_refuses_a_workspace_of_another_empty_value() -> None:
    """A workspace view that is not BF16 (an exchange workspace's words) is refused."""
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, 8, HIDDEN)
    x = _rows(8, 8100, RANK)
    try:
        with validating(entry.mnnvl_hop):
            entry.mnnvl_hop(
                x, x, _gamma(41), EPS, comm_buffer.view(torch.int32), buffer_flags, False
            )
    except AssertionError as exc:
        assert "BF16 view" in str(exc), exc
    else:
        raise AssertionError("is_valid admitted a non-BF16 workspace view")


CHECKS = (
    check_certified_cells,
    check_exact_sums_are_exact,
    check_lamport_flags_rotate,
    check_negative_zero_rows_arrive_as_positive_zero,
    check_interleaved_with_the_builtin_all_reduce,
    check_cuda_graph_replays,
    check_late_ranks,
    check_the_op_refuses_unsupported_shapes,
    check_the_guard_refuses_flags_of_another_workspace,
    check_the_guard_refuses_a_workspace_of_another_empty_value,
)


def _run_one_rank() -> None:
    global entry, ws, validating, fusion_op, RANK, WORLD, MAPPING
    RANK, WORLD, MAPPING = ranks.init_rank()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.comm import (  # noqa: E501
        _mnnvl_workspace,
        mnnvl_hop,
    )
    from tensorrt_llm.functional import AllReduceFusionOp

    entry, ws = mnnvl_hop, _mnnvl_workspace
    fusion_op = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)
    # Run as a script, not collected: the shared _validating module is reached the plain way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    ranks.grow_workspace_before_capture(MAPPING, HIDDEN)
    ranks.run_checks(CHECKS, RANK)


if __name__ == "__main__":
    if ranks.WORKER_FLAG in sys.argv:
        _run_one_rank()
    ranks.spawn_ranks(Path(__file__).resolve(), DEADLINE_S)
    print("OK")
