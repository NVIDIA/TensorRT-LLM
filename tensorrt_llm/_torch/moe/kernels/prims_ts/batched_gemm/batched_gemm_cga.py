# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host and device topology helpers for wider/flexible CGAs.

``umma_cta_count`` is the logical UMMA instruction atom width; the preferred and
fallback shapes are complete physical CUDA cluster shapes.  Supported wider
shapes use an even X dimension, so adjacent physical ranks are independent
2-CTA UMMA atoms.  For example, a 4x2 CGA is laid out as::

    physical Y
       1       [4 5] [6 7]       rank = x + dim_x * y
       0       [0 1] [2 3]       [a b] = one 2-CTA UMMA atom
                 0     1         logical atom X

There are three different masks for each TMA-to-UMMA operand.  They must not
be conflated:

``payload``
    CTAs whose SMEM receives the TMA data.
``full owners``
    Even ranks whose UMMA full barriers are armed.  Every payload rank maps to
    the leader of its adjacent atom: ``owner = rank & ~1``.
``empty release``
    Ranks signaled by UMMA consumers when a stage can be reused.

The following 4x2 examples use issuer rank 0. ``E`` is the number of consumer
arrivals expected by the producer's empty barrier::

    operand   payload P       full owners O   empty release R   E
    A         {0,4}           {0,4}           {0,1}             dim_y = 2
    B         {0,2}           {0,2}           {0,1}             dim_x / 2 = 2
    SFB       {0,1,2,3}       {0,2}           {0}               dim_x / 2 = 2

Mask reductions for the supported shapes make the local-versus-multicast
distinction explicit::

    shape   A(P/O/R/E)              B(P/O/R/E)              SFB(P/O/R/E)
    2x1     {0}/{0}/{0,1}/1         {0}/{0}/{0,1}/1         {0,1}/{0}/{0}/1
    4x1     {0}/{0}/{0,1}/1         {0,2}/{0,2}/{0,1}/2     {0..3}/{0,2}/{0}/2
    2x2     {0,2}/{0,2}/{0,1}/2     {0}/{0}/{0,1}/1         {0,1}/{0}/{0}/1
    4x2     {0,4}/{0,4}/{0,1}/2     {0,2}/{0,2}/{0,1}/2     {0..3}/{0,2}/{0}/2
    4x4     {0,4,8,12}/same/{0,1}/4 {0,2}/same/{0,1}/2      {0..3}/{0,2}/{0}/2

Rows other than Y=0 use the same formulas with their row base added.  A 2x1
SFB entry is relevant to the active fallback of a flexible wider kernel: that
compiled kernel still has one elected issuer.  A standalone legacy 2x1 kernel
does not compile the extended SFB path.

Only a flexible launch queries its active physical dimensions on the device;
fixed wider shapes use compile-time dimensions.  Keeping all rank, election,
and mask math here prevents task ownership, TMA issue, and pipeline barriers
from silently drifting apart.
"""

from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32

from .batched_gemm_config import SfLayout, SfSmemToTmemCopy


class CgaPolicy(IntEnum):
    """CTA participation policy for one task/resource.

    Selected ranks for the diagrammed 4x2 CGA and its 2x1 fallback are::

        policy                    4x2 selected ranks   2x1 selected ranks
        ALL                       0,1,2,3,4,5,6,7      0,1
        ATOM_LEADER               0,2,4,6              0
        ONE_Y_ROW                 0,1,2,3              0,1
        FIRST_X_ATOM_PER_Y        0,1,4,5              0,1
        ONE_X_COLUMN              0,4                  0
        CTA_ZERO                  0                    0
        MCAST_A                   0,1,2,3              0,1
        MCAST_B                   0,1,4,5              0,1
        MCAST_SFB                 0,4                  0

    The ``MCAST_*`` policies adapt to the active shape.  A and B become
    all-CTA policies when their multicast dimension collapses.  SFB remains
    one-X-column on a flexible 2x1 fallback because rank 0 multicasts the
    payload to rank 1.  Rank-only policies avoid querying active dimensions.
    """

    ALL = 0
    ATOM_LEADER = 1
    ONE_Y_ROW = 2
    FIRST_X_ATOM_PER_Y = 3
    ONE_X_COLUMN = 4
    CTA_ZERO = 5
    MCAST_A = 6
    MCAST_B = 7
    MCAST_SFB = 8


class CgaMcastDirection(IntEnum):
    """Logical relation used to release a multicast pipeline's empty barrier.

    This is the FlashInfer equivalent of TRT-LLM/CUTLASS ``McastDirection``::

        FlashInfer        TRT-LLM name   operands   expected empty arrivals
        ALONG_Y           Row            A, SFA     NY = dim_y
        ACROSS_X_ATOMS    Col            B, SFB     NX = dim_x / atom_x

    TRT-LLM's diagrams call Y-varying communication ``Row`` because they draw
    X vertically and Y horizontally.  The FlashInfer names state the physical
    coordinate directly to avoid that orientation ambiguity.

    There is intentionally no ``RowCol`` member. FlashInfer gives A, B, SFA,
    and SFB separate task pipelines, so each empty barrier has a one-sided
    release relation. A future combined A+B or SFA+SFB pipeline would require
    ``NY + NX - 1`` arrivals and the union of both release relations; it must
    add that direction explicitly rather than treating either one-sided mask
    as sufficient.
    """

    ALONG_Y = 0
    ACROSS_X_ATOMS = 1


def tma_mcast_direction(operand: str) -> CgaMcastDirection:
    """Map an operand to its empty-barrier multicast relation."""
    if operand == "a" or operand == "sfa":
        return CgaMcastDirection.ALONG_Y
    if operand == "b" or operand == "sfb":
        return CgaMcastDirection.ACROSS_X_ATOMS
    raise ValueError(f"Unknown CGA operand: {operand}")


def is_flexible_cga(cfg) -> bool:
    """Return whether one launch may use either configured physical shape."""
    return cfg.fallback_cluster_shape is not None


def has_multiple_umma_atoms(cfg) -> bool:
    """Return whether the preferred physical CGA contains multiple atoms."""
    dim_x, dim_y, dim_z = cfg.preferred_cluster_shape
    return dim_x * dim_y * dim_z > cfg.umma_cta_count


def _routed_operand(cfg, *, scale_factor: bool) -> str | None:
    """Return the logical operand supplied by the configured routed path."""
    has_route = cfg.has_routed_sfs if scale_factor else cfg.has_routed_act
    if not has_route:
        return None
    if scale_factor:
        return "sfb" if cfg.is_swap_ab else "sfa"
    return "b" if cfg.is_swap_ab else "a"


def _uses_compact_sfb_copy(cfg) -> bool:
    """Return whether SFB uses the per-CTA compact R8c4 LDS+STTM path."""
    return cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.LDS_STTM
    ) and cfg.smem_sfb_layout == int(SfLayout.R8c4)


def sfb_copy_uses_atom_leader(cfg) -> bool:
    """Return whether CopySfB is owned by one CTA per 2-CTA UMMA atom."""
    return cfg.uses_two_cta_mma and cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.UTCCP
    )


def needs_extended_tma_multicast(cfg, operand: str) -> bool:
    """Return whether one operand needs topology-aware TMA multicast.

    This compile-time boolean controls payload multicast and load-task
    election, not TMA-to-UMMA barrier routing.  Every multi-atom
    ``PipelineTmaUmma`` still needs atom-relative barriers even when this
    function returns ``False``.

    The decision table is::

        operand  topology trigger       additionally requires / excludes
        A        dim_y > 1              no CastA; no routed-LDGSTS A
        B        dim_x > 2              no routed-LDGSTS B
        SFA      dim_y > 1              SFA present; no CastA/routed SFA/
                                        local LDS+STTM copy
        SFB      dim_x > 1, multi-atom  SFs present; no routed SFB/compact
                                        R8c4 LDS+STTM copy

    Validation guarantees that a fallback divides the preferred shape.  An
    enabled flexible kernel therefore compiles one extended path whose masks
    and policies adapt to either active shape.  An unaffected operand retains
    its local payload/task path even when another operand multicasts.
    """
    dim_x, dim_y, _ = cfg.preferred_cluster_shape
    routed_operand = _routed_operand(cfg, scale_factor=operand.startswith("sf"))
    if operand == "a":
        return (
            dim_y > 1
            and not cfg.has_cast_a
            and not (routed_operand == "a" and not cfg.has_tma_route)
        )
    if operand == "b":
        return dim_x > 2 and not (routed_operand == "b" and not cfg.has_tma_route)
    if operand == "sfa":
        return (
            cfg.has_scale_factor_a
            and dim_y > 1
            and not cfg.has_cast_a
            and routed_operand != "sfa"
            and cfg.sfa_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
        )
    if operand == "sfb":
        return (
            cfg.has_input_block_scaling
            and has_multiple_umma_atoms(cfg)
            and dim_x > 1
            and routed_operand != "sfb"
            and not _uses_compact_sfb_copy(cfg)
        )
    raise ValueError(f"Unknown CGA operand: {operand}")


@dataclass(frozen=True)
class HostCgaTopology:
    """Pure-Python mirror of device rank, policy, and mask formulas.

    Operand masks are built from rank-bit, atom, row, column, and X-parity
    primitives. Tests use this model as an executable version of the module
    diagrams rather than duplicating mask arithmetic.
    """

    dim_x: int
    dim_y: int
    dim_z: int = 1

    def __post_init__(self) -> None:
        if self.dim_x <= 0 or self.dim_y <= 0 or self.dim_z != 1:
            raise ValueError(
                "CGA dimensions must have positive X/Y and Z=1, got "
                f"({self.dim_x}, {self.dim_y}, {self.dim_z})"
            )
        if self.dim_x * self.dim_y > 16:
            raise ValueError("CGA masks support at most 16 CTAs")

    def rank(self, x: int, y: int, z: int = 0) -> int:
        return x + self.dim_x * (y + self.dim_y * z)

    def rank_bit(self, x: int, y: int) -> int:
        """Return the mask bit for physical coordinate ``(x, y)``."""
        return 1 << self.rank(x, y)

    def atom_mask(self, x: int, y: int) -> int:
        """Return both ranks in the adjacent 2-CTA atom containing ``x``."""
        atom_x = x & ~1
        return self.rank_bit(atom_x, y) | self.rank_bit(atom_x + 1, y)

    def row_mask(self, y: int) -> int:
        """Return every CTA in physical Y row ``y``."""
        return sum(self.rank_bit(x, y) for x in range(self.dim_x))

    def column_mask(self, x: int) -> int:
        """Return every CTA at physical X coordinate ``x``."""
        return sum(self.rank_bit(x, y) for y in range(self.dim_y))

    def parity_mask(self, x: int, y: int) -> int:
        """Return one X-parity lane from every atom in physical row ``y``."""
        parity = x & 1
        return sum(
            self.rank_bit(atom_x + parity, y) for atom_x in range(0, self.dim_x, 2)
        )

    def selected(self, policy: CgaPolicy, x: int, y: int) -> bool:
        """Evaluate the documented policy table for one physical coordinate."""
        first_y_row = y == 0
        first_x_atom = x < min(self.dim_x, 2)
        first_x_column = x == 0
        if policy == CgaPolicy.ALL:
            return True
        if policy == CgaPolicy.ATOM_LEADER:
            return (x & 1) == 0
        if policy == CgaPolicy.ONE_Y_ROW:
            return first_y_row
        if policy == CgaPolicy.FIRST_X_ATOM_PER_Y:
            return first_x_atom
        if policy == CgaPolicy.ONE_X_COLUMN:
            return first_x_column
        if policy == CgaPolicy.CTA_ZERO:
            return first_x_column and first_y_row
        if policy == CgaPolicy.MCAST_A:
            return first_y_row if self.dim_y > 1 else True
        if policy == CgaPolicy.MCAST_B:
            return first_x_atom if self.dim_x > 2 else True
        if policy == CgaPolicy.MCAST_SFB:
            return first_x_column if self.dim_x > 1 else True
        raise ValueError(f"Unknown CGA policy: {policy}")

    def participant_count(self, policy: CgaPolicy) -> int:
        return sum(
            self.selected(policy, x, y)
            for y in range(self.dim_y)
            for x in range(self.dim_x)
        )

    def payload_mask_a(self, x: int) -> int:
        return self.column_mask(x)

    def payload_mask_b(self, x: int, y: int) -> int:
        return self.parity_mask(x, y)

    def payload_mask_sfb(self, y: int) -> int:
        return self.row_mask(y)

    @staticmethod
    def full_barrier_mask(payload_mask: int) -> int:
        # Fold each odd-rank payload bit onto its preceding even-rank owner,
        # then discard odd positions. This is the host form of the device
        # projection below.
        return (payload_mask | (payload_mask >> 1)) & 0x5555

    def release_mask_a(self, x: int) -> int:
        return self.atom_mask(x, 0)

    def release_mask_b(self, y: int) -> int:
        return self.atom_mask(0, y)

    def release_mask_sfb(self, y: int) -> int:
        return self.rank_bit(0, y)


@cute.jit
def runtime_cga_coords():
    """Return runtime physical CGA dimensions, rank, and `(x, y)` coordinates."""
    dim_x, dim_y, dim_z = cute.arch.block_in_cluster_dim()
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    x = rank % dim_x
    y = (rank // dim_x) % dim_y
    return dim_x, dim_y, dim_z, rank, x, y


@cute.jit
def active_cga_coords(cfg: cutlass.Constexpr[object]):
    """Return active physical dimensions and CTA coordinates.

    Flexible launches select preferred/fallback dimensions at device runtime.
    Fixed wider launches materialize the preferred dimensions as constants.
    Callers keep legacy 1x1/2x1 kernels on stock paths and never invoke this
    helper for them.
    """
    if cutlass.const_expr(is_flexible_cga(cfg)):
        dim_x, dim_y, dim_z = cute.arch.block_in_cluster_dim()
    else:
        host_x, host_y, host_z = cfg.preferred_cluster_shape
        dim_x, dim_y, dim_z = Int32(host_x), Int32(host_y), Int32(host_z)
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    x = rank % dim_x
    y = (rank // dim_x) % dim_y
    return dim_x, dim_y, dim_z, rank, x, y


@cute.jit
def cga_policy_selected(
    cfg: cutlass.Constexpr[object], policy: cutlass.Constexpr[int]
) -> Boolean:
    """Return whether the current CTA owns a task with ``policy``.

    ``policy`` is a compile-time constant stored on :class:`CgaTask`; only the
    active dimensions of a flexible launch are runtime values.  The function
    is deliberately split into three groups:

    * ``ALL`` has no topology dependency.
    * ``ATOM_LEADER`` and ``CTA_ZERO`` need only physical rank.  Avoiding
      :func:`active_cga_coords` here also avoids a runtime dimension query.
    * Row/column and ``MCAST_*`` policies need active X/Y.  Their primitive
      booleans are computed once and reused below.

    ``MCAST_A`` and ``MCAST_B`` expand to all CTAs when their multicast
    dimension collapses on a fallback. ``MCAST_SFB`` keeps one issuer column
    whenever X > 1 because its fallback rank 0 still multicasts to rank 1.
    See :class:`CgaPolicy` for the complete 4x2/2x1 truth table.
    """
    if cutlass.const_expr(policy == int(CgaPolicy.ALL)):
        return Boolean(True)

    # Rank-only ownership: even ranks lead adjacent 2-CTA atoms; rank zero is
    # the one physical-CGA-global owner.
    if cutlass.const_expr(policy == int(CgaPolicy.ATOM_LEADER)):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        return (rank & Int32(1)) == Int32(0)
    if cutlass.const_expr(policy == int(CgaPolicy.CTA_ZERO)):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        return rank == Int32(0)

    # Shape-dependent ownership. Fixed wider kernels receive constexpr
    # dimensions; only flexible preferred/fallback kernels query them.
    dim_x, dim_y, _, _, x, y = active_cga_coords(cfg)

    # Primitive selections in the 4x2 rank diagram (T executes the task):
    #
    #                        y=1             y=0
    #   first_y_row        F F | F F       T T | T T
    #   first_x_atom       T T | F F       T T | F F
    #   first_x_column     T F | F F       T F | F F
    #
    # ONE_* policies use these booleans directly. MCAST_* policies reuse them
    # conditionally so a flexible fallback can change task ownership without
    # compiling another kernel.
    first_y_row = y == Int32(0)
    first_x_atom = x < Int32(2)
    first_x_column = x == Int32(0)
    if cutlass.const_expr(policy == int(CgaPolicy.ONE_Y_ROW)):
        return first_y_row
    if cutlass.const_expr(policy == int(CgaPolicy.FIRST_X_ATOM_PER_Y)):
        return first_x_atom
    if cutlass.const_expr(policy == int(CgaPolicy.ONE_X_COLUMN)):
        return first_x_column

    # Adaptive multicast elections reuse the same row/atom/column predicates
    # but expand to all CTAs if no payload sharing remains on that dimension.
    if cutlass.const_expr(policy == int(CgaPolicy.MCAST_A)):
        selected = Boolean(True)
        if dim_y > Int32(1):
            selected = first_y_row
        return selected
    if cutlass.const_expr(policy == int(CgaPolicy.MCAST_B)):
        selected = Boolean(True)
        if dim_x > Int32(2):
            selected = first_x_atom
        return selected
    if cutlass.const_expr(policy == int(CgaPolicy.MCAST_SFB)):
        selected = Boolean(True)
        if dim_x > Int32(1):
            selected = first_x_column
        return selected

    # Config validation/task construction only emits known policies. Keep a
    # defensive false result so an invalid value cannot accidentally run work.
    return Boolean(False)


@cute.jit
def _rank_bit(rank: Int32) -> Int32:
    """Return one 16-bit CGA mask bit."""
    return Int32(1) << rank


@cute.jit
def _atom_mask(dim_x: Int32, x: Int32, y: Int32) -> Int32:
    """Return the adjacent 2-CTA atom containing physical ``(x, y)``."""
    atom_leader = (x & Int32(~1)) + dim_x * y
    return _rank_bit(atom_leader) | _rank_bit(atom_leader + Int32(1))


@cute.jit
def _column_mask(dim_x: Int32, dim_y: Int32, x: Int32) -> Int32:
    """Return the active physical X column used by A/SFA multicast."""
    result = Int32(0)
    for yy in cutlass.range_constexpr(4):
        if Int32(yy) < dim_y:
            result = result | _rank_bit(x + dim_x * Int32(yy))
    return result


@cute.jit
def _row_mask(dim_x: Int32, y: Int32) -> Int32:
    """Return the active physical Y row used by SFB multicast."""
    result = Int32(0)
    for xx in cutlass.range_constexpr(4):
        if Int32(xx) < dim_x:
            result = result | _rank_bit(Int32(xx) + dim_x * y)
    return result


@cute.jit
def _x_parity_mask(dim_x: Int32, x: Int32, y: Int32) -> Int32:
    """Return the issuer's X parity from every atom in one row for B."""
    result = Int32(0)
    parity = x & Int32(1)
    for atom_x in cutlass.range_constexpr(0, 4, 2):
        if Int32(atom_x) < dim_x:
            result = result | _rank_bit(Int32(atom_x) + parity + dim_x * y)
    return result


@cute.jit
def _project_payload_to_barrier_owners(payload_mask: Int32) -> Int32:
    """Project every payload bit to its adjacent even-rank barrier owner."""
    return (payload_mask | (payload_mask >> Int32(1))) & Int32(0x5555)


@cute.jit
def cga_tma_masks(cfg: cutlass.Constexpr[object], operand: cutlass.Constexpr[str]):
    """Return ``(payload, full owners, empty release, arrivals)``.

    ``operand`` is a compile-time string. It selects both a payload-mask kind
    and a :class:`CgaMcastDirection`; these are related but distinct. A and SFA
    share a physical X column, B uses one X parity across a row, and SFB uses
    the entire row. B and SFB therefore share a direction/arrival count but not
    a payload or release mask. All cases reuse the rank/atom primitives above
    and the owner projection documented in the module diagram.

    A/SFA
        The Y=0 issuer multicasts down its X column. Every Y atom releases to
        that column's Y=0 atom, so the empty barrier expects ``dim_y`` arrivals.
    B
        The first X atom in each row issues one parity lane across all atoms.
        Every atom releases to the row's first atom, so the count is
        ``dim_x / 2``.
    SFB
        The X=0 issuer multicasts across its row. Atom leaders release to X=0,
        again producing ``dim_x / 2`` arrivals.
    """
    dim_x, dim_y, _, _, x, y = active_cga_coords(cfg)
    direction = tma_mcast_direction(operand)
    if cutlass.const_expr(operand == "a" or operand == "sfa"):
        # A/SFA payload for issuer x=0 in a 4x2 CGA. The elected task CTAs are
        # in y=0, but the same X coordinate receives data in every Y row:
        #
        #     y=1   P . | . .      full owners: O . | . .
        #     y=0   I . | . .                   O . | . .
        #           ----+----
        #            atom0 atom1
        #
        # Both Y atoms release the empty barrier acquired by atom0 at y=0,
        # hence release={0,1} and arrival_count=dim_y.
        payload = _column_mask(dim_x, dim_y, x)
        release = _atom_mask(dim_x, x, Int32(0))
    elif cutlass.const_expr(operand == "b"):
        # B preserves the issuer's X parity across atoms in one Y row. For the
        # even issuer at (x=0,y=0) in 4x2:
        #
        #     y=1   . . | . .      full owners: . . | . .
        #     y=0   I . | P .                   O . | O .
        #           ----+----
        #            atom0 atom1
        #
        # Every X atom releases to the first atom in its row, so
        # release={0,1} and arrival_count=dim_x/2.
        payload = _x_parity_mask(dim_x, x, y)
        release = _atom_mask(dim_x, Int32(0), y)
    elif cutlass.const_expr(operand == "sfb"):
        # SFB has one X=0 issuer per Y row and fills that whole row. At y=0:
        #
        #     y=1   . . | . .      full owners: . . | . .
        #     y=0   I P | P P                   O . | O .
        #           ----+----
        #            atom0 atom1
        #
        # Each atom leader releases to the sole acquired owner at x=0, so
        # release={0} and arrival_count=dim_x/2.
        payload = _row_mask(dim_x, y)
        release = _rank_bit(dim_x * y)
    else:
        raise ValueError(f"Unknown CGA operand: {operand}")

    # Direction initializes the empty barrier's expected arrivals. Payload
    # kind remains independent: B and SFB share this count but use different
    # masks above, matching TRT-LLM's McastDir versus McastMaskKind split.
    if cutlass.const_expr(direction == CgaMcastDirection.ALONG_Y):
        arrival_count = dim_y
    else:
        arrival_count = dim_x // Int32(2)
    return (
        payload,
        _project_payload_to_barrier_owners(payload),
        release,
        arrival_count,
    )


@dataclass(frozen=True)
class TaskParticipation:
    """One real task's contribution to the persistent work queue."""

    name: str
    warps: int
    policy: CgaPolicy


def operand_cga_policy(cfg, operand: str) -> CgaPolicy:
    """Shared ownership decision for operand tasks and their barrier census."""
    if not needs_extended_tma_multicast(cfg, operand):
        return CgaPolicy.ALL
    return {
        "a": CgaPolicy.MCAST_A,
        "b": CgaPolicy.MCAST_B,
        "sfa": CgaPolicy.MCAST_A,
        "sfb": CgaPolicy.MCAST_SFB,
    }[operand]


def clc_task_participation(cfg) -> tuple[TaskParticipation, ...]:
    """Resolve actual tasks once; fused resource aliases add no participants."""
    w = cfg.warp_layout
    tasks = []

    def add(name, warps, policy=CgaPolicy.ALL):
        if warps:
            tasks.append(TaskParticipation(name, warps, policy))

    add("load_a", w.num_load_a_warps, operand_cga_policy(cfg, "a"))
    add("load_b", w.num_load_b_warps, operand_cga_policy(cfg, "b"))
    add("epilogue", w.num_epilogue_warps)
    add("padding", w.num_padding_warps)
    add("sync", w.num_sync_warps)
    add("cast_a", w.num_cast_a_warps)
    add("mma", w.num_mma_warps, CgaPolicy.ATOM_LEADER)
    add("workid", w.num_workid_warps, CgaPolicy.CTA_ZERO)
    if not cfg.fuse_operand_sf_loads:
        if cfg.has_scale_factor_a:
            add("load_sfa", w.num_load_sfa_warps, operand_cga_policy(cfg, "sfa"))
        if cfg.has_input_block_scaling:
            add("load_sfb", w.num_load_sfb_warps, operand_cga_policy(cfg, "sfb"))
    if cfg.use_deepseek_fp8:
        add("load_sfab", w.num_load_sfab_warps)
    if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
        if cfg.use_combined_sfab_copy:
            add("copy_sfab", w.num_copy_sfa_warps, CgaPolicy.ATOM_LEADER)
        else:
            add(
                "copy_sfa",
                w.num_copy_sfa_warps,
                CgaPolicy.ATOM_LEADER
                if cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.UTCCP)
                else CgaPolicy.ALL,
            )
            add(
                "copy_sfb",
                w.num_copy_sfb_warps,
                CgaPolicy.ATOM_LEADER
                if sfb_copy_uses_atom_leader(cfg)
                else CgaPolicy.ALL,
            )
    return tuple(tasks)


def host_clc_consumer_threads(cfg, dim_x: int, dim_y: int) -> int:
    """Exact arrivals for the physical CGA, from resolved task ownership."""
    topo = HostCgaTopology(dim_x, dim_y)
    return 32 * sum(
        t.warps * topo.participant_count(t.policy) for t in clc_task_participation(cfg)
    )


def verifier_clc_consumer_threads(cfg) -> int:
    """Project actual tasks onto the stock verifier's atom-only CTA model."""
    if not has_multiple_umma_atoms(cfg):
        return host_clc_consumer_threads(cfg, cfg.umma_cta_count, 1)
    return 32 * sum(
        t.warps * (1 if t.name == "workid" else cfg.umma_cta_count)
        for t in clc_task_participation(cfg)
    )


@cute.jit
def active_clc_consumer_threads(cfg: cutlass.Constexpr[object]) -> Int32:
    """Device evaluation of the same task list for the active physical CGA."""
    dim_x, dim_y, _, _, _, _ = active_cga_coords(cfg)
    tasks = clc_task_participation(cfg)
    total = Int32(0)
    for i in cutlass.range_constexpr(len(tasks)):
        participants = dim_x * dim_y
        if cutlass.const_expr(tasks[i].policy == CgaPolicy.ATOM_LEADER):
            participants = (dim_x // Int32(cfg.umma_cta_count)) * dim_y
        elif cutlass.const_expr(tasks[i].policy == CgaPolicy.CTA_ZERO):
            participants = Int32(1)
        elif cutlass.const_expr(tasks[i].policy == CgaPolicy.MCAST_A):
            participants = dim_x
        elif cutlass.const_expr(tasks[i].policy == CgaPolicy.MCAST_B):
            participants = Int32(cfg.umma_cta_count) * dim_y
        elif cutlass.const_expr(tasks[i].policy == CgaPolicy.MCAST_SFB):
            participants = dim_y
        total += Int32(tasks[i].warps) * participants
    return total * Int32(32)
