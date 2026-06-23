# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MoE scheduler utilities and online TMA descriptor helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import nvvm_wrappers
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import AddressSpace, Numeric, Pointer
from cutlass.cutlass_dsl import Boolean, Int32, T, dsl_user_op
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

TensormapDescBytes = 128
TensormapDescBytes = 64  # {$nv-internal-release}

# =============================================================================
# Pointer Utilities
# =============================================================================


@dsl_user_op
def _nanosleep(
    sleep_time: int,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Compatibility wrapper for wheels without ``cute.arch.nanosleep``."""
    if cutlass.const_expr(hasattr(cute.arch, "nanosleep")):
        cute.arch.nanosleep(sleep_time=sleep_time, loc=loc, ip=ip)
    else:
        llvm.inline_asm(
            res=None,
            operands_=[Int32(sleep_time).ir_value(loc=loc, ip=ip)],
            asm_string="nanosleep.u32 $0;",
            constraints="r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
@cute.jit
def spin_wait(
    ptr: Pointer,
    condition: Callable[[Int32], bool],
    fail_sleep_cycles: int = 100,
    peek_only: bool = False,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Boolean:
    """Spin until condition is true, or do one condition check with peek_only."""
    current = cute.arch.load(ptr, ptr.dtype, cop="cg", loc=loc, ip=ip)
    if cutlass.const_expr(peek_only):
        # One-shot peek: forward the condition Boolean to the caller.
        return Boolean(condition(current))
    while not condition(current):
        # Load with L1 cache bypass (ld.global.cg)
        if cutlass.const_expr(fail_sleep_cycles > 0):
            _nanosleep(fail_sleep_cycles, loc=loc, ip=ip)
        current = cute.arch.load(ptr, ptr.dtype, cop="cg", loc=loc, ip=ip)
    # Spin-path: condition was satisfied; uniformize return type with the
    # peek path so callers always see a Boolean.
    return Boolean(True)


# =============================================================================
# Cluster-DSMEM helpers (for atomic_counter dynamic scheduler)
# =============================================================================
#
# Ported from cute_dsl_kernel_library/dsl_kernels/moe/moe_persistent_scheduler.py
# (lines 79-145).  Used by the fused fc1+fc2 mega scheduler when
# load_balance_mode == 'atomic_counter' to
# implement the leader-CTA atom.add + DSMEM broadcast cluster-tile-idx
# fetch protocol.  ``atom.add`` itself uses cute.arch.atomic_add (the
# upstream cute_dsl wrapper) instead of a hand-rolled helper.


@dsl_user_op
def store_i32_to_peer_cluster_smem_async(
    smem_ptr,
    value: Int32,
    mbar_ptr,
    cta_rank_in_cluster,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Store one int32 to a peer CTA's SMEM via st.async.shared::cluster.

    Uses ``mapa.shared::cluster`` to translate ``smem_ptr`` / ``mbar_ptr``
    (this CTA's SMEM addresses) into the peer CTA's address space, then
    issues ``st.async.shared::cluster.mbarrier::complete_tx::bytes.u32``
    which both writes the int32 AND signals completion on the peer
    mbarrier.  The peer mbarrier's expect_tx must be set up beforehand
    (see ``mbarrier_arrive_expect_tx_on_peer``).
    """
    smem_addr = llvm.ptrtoint(T.i32(), smem_ptr.llvm_ptr, loc=loc, ip=ip)
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            smem_addr,
            value.ir_value(loc=loc, ip=ip),
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_addr;
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_addr, $0, $3;
            mapa.shared::cluster.u32 remote_mbar, $2, $3;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [remote_addr], $1, [remote_mbar];
        }}""",
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer(
    mbar_ptr,
    tx_count: Int32,
    cta_rank_in_cluster,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Set expect_tx on a peer CTA's mbarrier via inline PTX.

    Pairs with ``store_i32_to_peer_cluster_smem_async``: this side
    declares "I expect ``tx_count`` bytes via st.async on this peer
    mbarrier"; the store side then completes the transaction.
    """
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            tx_count.ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_mbar, $0, $1;
            mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_mbar], $2;
        }}""",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def gmem_ptr_to_generic(
    gmem_ptr: Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Pointer:
    if gmem_ptr.memspace != AddressSpace.gmem:
        raise ValueError(
            f"gmem_ptr_to_generic requires pointer in gmem address space, got {gmem_ptr.memspace}"
        )
    # Get LLVM pointer and cast to generic address space
    llvm_ptr = gmem_ptr.to_llvm_ptr(loc=loc,
                                    ip=ip)  # type: ignore[attr-defined]
    generic_llvm_ptr = llvm.addrspacecast(llvm.PointerType.get(
        AddressSpace.generic),
                                          llvm_ptr,
                                          loc=loc,
                                          ip=ip)
    # Create a new cute.Pointer with generic address space, preserving alignment
    return cute.make_ptr(
        gmem_ptr.dtype,
        generic_llvm_ptr,
        AddressSpace.generic,
        assumed_align=gmem_ptr.alignment,  # type: ignore[attr-defined]
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def generic_ptr_to_gmem(
    generic_ptr: Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Pointer:
    if generic_ptr.memspace != AddressSpace.generic:
        raise ValueError(
            f"generic_ptr_to_gmem requires pointer in generic address space, "
            f"got {generic_ptr.memspace}")
    # Get LLVM pointer and cast to gmem address space
    llvm_ptr = generic_ptr.to_llvm_ptr(loc=loc,
                                       ip=ip)  # type: ignore[attr-defined]
    gmem_llvm_ptr = llvm.addrspacecast(llvm.PointerType.get(AddressSpace.gmem),
                                       llvm_ptr,
                                       loc=loc,
                                       ip=ip)
    # Create a new cute.Pointer with gmem address space, preserving alignment
    return cute.make_ptr(
        generic_ptr.dtype,
        gmem_llvm_ptr,
        AddressSpace.gmem,
        assumed_align=generic_ptr.alignment,  # type: ignore[attr-defined]
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_tma_descriptor(
    tma_desc_ptr: Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """
    Prefetch a TMA descriptor from global memory.

    This function prefetches the TMA descriptor pointed to by tma_desc_ptr
    into the TMA descriptor cache. The pointer must be in generic or global
    address space. If a gmem pointer is passed, it will be automatically
    converted to generic address space.

    :param tma_desc_ptr: Pointer to the TMA descriptor in global or generic memory
    :type tma_desc_ptr: Pointer
    :raises ValueError: If pointer is not in generic or global address space
    """
    if tma_desc_ptr.memspace not in (AddressSpace.gmem, AddressSpace.generic):
        raise ValueError(
            f"prefetch_tma_descriptor requires pointer in gmem or generic address space, "
            f"got {tma_desc_ptr.memspace}")
    # Convert gmem pointer to generic if needed
    if tma_desc_ptr.memspace == AddressSpace.gmem:
        tma_desc_ptr = gmem_ptr_to_generic(tma_desc_ptr, loc=loc, ip=ip)
    # Convert cute.Pointer to LLVM pointer for prefetch
    llvm_ptr = tma_desc_ptr.to_llvm_ptr(loc=loc,
                                        ip=ip)  # type: ignore[attr-defined]
    nvvm_wrappers.prefetch(llvm_ptr, tensormap=True, loc=loc, ip=ip)


def ptr_offset_bytes(ptr: Pointer, byte_offset: int) -> Pointer:
    """Offset a pointer by a given number of bytes."""
    element_offset = byte_offset * 8 // ptr.dtype.width
    return ptr + element_offset


@dsl_user_op
def tensormap_ptr_for_copy(
    raw_ptr: Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Pointer:
    """
    Convert a raw TMA descriptor gmem pointer to the type expected by cute.copy.

    cute.copy requires the tma_desc_ptr to be in generic address space and
    recast to TmaDescriptorTiledType. This utility performs both conversions.

    :param raw_ptr: Raw pointer to TMA descriptor in gmem
    :type raw_ptr: Pointer
    :return: Pointer compatible with cute.copy's tma_desc_ptr parameter
    :rtype: Pointer
    """
    generic_ptr = gmem_ptr_to_generic(raw_ptr, loc=loc, ip=ip)
    tma_desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        generic_ptr.memspace,
        generic_ptr.alignment,
    )
    return _cute_ir.recast_iter(tma_desc_ptr_ty, generic_ptr.value)


# =============================================================================
# MoE Utilities
# =============================================================================


@dsl_user_op
@cute.jit
def compute_expert_token_range(
    offs: cute.Tensor,
    expert_idx: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Int32, Int32]:
    """
    Compute token offset and count for a given expert from the cumsum offs tensor.

    :param offs: Cumulative sum tensor of token counts per expert, shape (experts,)
    :param expert_idx: Index of the expert
    :return: (token_offset, tokens_i) where token_offset is the start position
             and tokens_i is the number of tokens for this expert
    """
    token_offset = Int32(0)
    if expert_idx > Int32(0):
        token_offset = offs[expert_idx - 1]  # type: ignore[assignment]
    tokens_i = offs[expert_idx] - token_offset
    return token_offset, tokens_i


@dsl_user_op
@cute.jit
def compute_expert_token_count_from_sizes(
    sizes: cute.Tensor,
    expert_idx: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """
    Read per-expert token count from a raw sizes tensor.

    This is the sizes-mode counterpart of ``compute_expert_token_range``: it
    returns *only* the count for ``expert_idx``; the cumulative token offset
    is the caller's responsibility (typically maintained as a running cumul in
    scheduler register state, updated when ``expert_idx`` advances).  Used by
    the MegaMoE-fused fc12 scheduler when sizes are exposed as a direct view
    of ``expert_recv_count_sum`` (e.g. via ``i32 stride=(2,)`` over an i64
    tensor) and no cumulative sum kernel was run on the host.
    """
    return sizes[expert_idx]


@dsl_user_op
def rewrite_tensor_shape(
    tensor: cute.Tensor,
    new_shape: Tuple,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Tensor:
    """
    Rewrite tensor shape while keeping the same stride and iterator.

    This is primarily for debug friendliness - shows the actual expert's shape
    instead of the fake global shape. No runtime overhead as it becomes
    dead code in non-debug builds.

    :param tensor: Source tensor whose stride and iterator to preserve
    :param new_shape: New shape to apply
    :return: New tensor with the given shape but original stride and iterator
    """
    new_layout = cute.make_layout(new_shape,
                                  stride=tensor.stride,
                                  loc=loc,
                                  ip=ip)
    return cute.make_tensor(tensor.iterator, new_layout, loc=loc, ip=ip)


# =============================================================================
# TMA Descriptor Workspace Helper
# =============================================================================


class TensormapWorkspace:
    """
    Helper for linear workspace layout of TMA descriptors.

    Manages address calculation for a workspace buffer containing TMA descriptors
    organized as: for each executor (e.g., expert or group), a fixed set of
    named descriptor slots.

    Layout: [slot_0_exec_0, slot_1_exec_0, ..., slot_0_exec_1, slot_1_exec_1, ...]

    Example:
        # 2Dx3D MoE: only C is expert-wise
        workspace = TensormapWorkspace(workspace_ptr, ["c"])

        # 2Dx2D MoE: A and B are expert-wise
        workspace = TensormapWorkspace(workspace_ptr, ["a", "b"])

        # General grouped GEMM: all three tensors
        workspace = TensormapWorkspace(workspace_ptr, ["a", "b", "c"])
    """

    def __init__(self, workspace_ptr: Pointer, slot_names: list):
        """
        :param workspace_ptr: Pointer to the beginning of the workspace buffer
        :param slot_names: Ordered list of tensor names, defining the slot layout
                           per executor. e.g., ["a", "b", "c"]
        """
        self.workspace_ptr = workspace_ptr
        self._name_to_slot = {name: i for i, name in enumerate(slot_names)}
        self._slots_per_executor = len(slot_names)

    @cute.jit
    def get_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        """
        Get the workspace pointer for a specific TMA descriptor.

        :param tensor_name: Name of the tensor (must be one of the slot_names)
        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx)
        :return: Aligned pointer to the TMA descriptor in workspace
        """
        if cutlass.const_expr(tensor_name not in self._name_to_slot):
            raise ValueError(
                f"Invalid tensor_name '{tensor_name}', "
                f"expected one of {list(self._name_to_slot.keys())}")
        slot = self._name_to_slot[tensor_name]
        byte_offset = (executor_idx * self._slots_per_executor +
                       slot) * TensormapDescBytes
        return ptr_offset_bytes(self.workspace_ptr,
                                byte_offset).align(TensormapDescBytes)

    @staticmethod
    def size_bytes(num_slots: int, num_executors: int) -> int:
        """
        Calculate workspace size in bytes.

        :param num_slots: Number of descriptor slots per executor
        :param num_executors: Total number of executors (e.g., expert_cnt or group_cnt)
        :return: Total workspace size in bytes
        """
        return num_slots * num_executors * TensormapDescBytes


# =============================================================================
# Online TMA Descriptor Creator (Abstract Base Class)
# =============================================================================


@dataclass(frozen=True)
class OnlineTensormapDescCreator(ABC):
    """
    Abstract base class for building TMA descriptors online (at kernel runtime).

    Subclasses store all needed parameters (both codegen-time configs and runtime
    values) as explicit instance attributes in __init__. No dict-based APIs.

    Subclasses must implement exactly 2 abstract methods:
    - construct_and_write: Build TMA descriptor(s) for one executor and write to workspace
    - get_desc_ptr: Return raw gmem pointer to a specific descriptor in workspace

    To convert the raw pointer for use with cute.copy, callers should use the
    standalone tensormap_ptr_for_copy() utility.
    """

    @abstractmethod
    def construct_and_write(self,
                            executor_idx: Int32,
                            dependency: Any = None) -> None:
        """
        Build TMA descriptor(s) for one executor and write to workspace.

        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx).
            Semantics may vary by subclass when ``dependency`` is provided.
        :param dependency: Optional pipeline consumer for inter-warp-group
            synchronization. When provided, the subclass decides when to wait
            (via ``dependency.wait_and_advance()``) and release. The subclass
            also decides how to interpret ``executor_idx`` in this mode.
        """
        ...

    @abstractmethod
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        """
        Get the raw gmem pointer to a specific TMA descriptor in workspace.

        :param tensor_name: Name identifying which tensor's descriptor
        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx)
        :return: Raw pointer (gmem) to the TMA descriptor
        """
        ...


# {$nv-internal-release begin}

# Internal example to show the general grouped gemm online desc construction.
# =============================================================================
# General Grouped GEMM TMA Descriptor Constructor
# =============================================================================


class GeneralGroupedGemmTensormapConstructor(OnlineTensormapDescCreator):
    """
    TMA descriptor constructor for general Grouped GEMM with pre-initialized descriptors.

    This class creates TMA descriptors for A, B, C tensors and writes them
    to a workspace buffer. Each group has its own set of descriptors.

    Uses cute.nvgpu.make_tiled_tma_atom_A/B for A and B tensors to ensure
    correct MMA projections.

    All parameters are stored as explicit instance attributes (no dicts).

    Workspace layout per group: [A(64B), B(64B), C(64B)]

    :param a_dtype: Data type for tensor A
    :param b_dtype: Data type for tensor B
    :param c_dtype: Data type for tensor C
    :param a_smem_layout: SMEM layout for A TMA
    :param b_smem_layout: SMEM layout for B TMA
    :param epi_smem_layout: SMEM layout for epilogue (C) TMA
    :param a_tma_op: TMA operation for A (G2S or G2S multicast)
    :param b_tma_op: TMA operation for B (G2S or G2S multicast)
    :param tiled_mma: TiledMma for correct MMA projections
    :param mma_tiler: MMA tiler shape (M, N, K)
    :param cluster_layout_vmnk_shape: Cluster layout shape for multicast
    :param epi_tile: Epilogue tile shape
    :param ptrs_abc: Tensor[num_groups, 3] (i64) - pointers to A, B, C
    :param problem_sizes_mnkl: Tensor[num_groups, 4] (i32) - M, N, K, L per group
    :param strides_abc: Tensor[num_groups, 3, 2] (i32) - strides for A, B, C
    :param group_cnt: Number of groups
    :param workspace_ptr: Pointer to workspace for TMA descriptors
    """

    def __init__(
        self,
        # Codegen-time configs
        a_dtype: Type[Numeric],
        b_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        a_tma_op: cute.CopyAtom,
        b_tma_op: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        cluster_layout_vmnk_shape: cute.Layout,
        epi_tile: cute.Tile,
        # Runtime params
        ptrs_abc: cute.Tensor,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        group_cnt: Int32,
        workspace_ptr: Pointer,
    ) -> None:
        super().__init__()
        # Codegen-time configs
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.a_smem_layout = a_smem_layout
        self.b_smem_layout = b_smem_layout
        self.epi_smem_layout = epi_smem_layout
        self.a_tma_op = a_tma_op
        self.b_tma_op = b_tma_op
        self.tiled_mma = tiled_mma
        self.mma_tiler = mma_tiler
        self.cluster_layout_vmnk_shape = cluster_layout_vmnk_shape
        self.epi_tile = epi_tile
        # Runtime params
        self.ptrs_abc = ptrs_abc
        self.problem_sizes_mnkl = problem_sizes_mnkl
        self.strides_abc = strides_abc
        self.group_cnt = group_cnt
        self.workspace = TensormapWorkspace(workspace_ptr, ["a", "b", "c"])

    @cute.jit
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        return self.workspace.get_ptr(tensor_name, executor_idx)

    @cute.jit
    def construct_and_write(self,
                            executor_idx: Int32,
                            dependency: Any = None) -> None:
        """
        Build TMA descriptors for A, B, C of one group and write to workspace.
        """
        group_idx = executor_idx

        if group_idx < self.group_cnt:
            # Read pointers
            ptr_a = self.ptrs_abc[group_idx, 0]
            ptr_b = self.ptrs_abc[group_idx, 1]
            ptr_c = self.ptrs_abc[group_idx, 2]

            # Read problem sizes
            M = self.problem_sizes_mnkl[group_idx, 0]
            N = self.problem_sizes_mnkl[group_idx, 1]
            K = self.problem_sizes_mnkl[group_idx, 2]

            # Read strides
            stride_a_0 = self.strides_abc[group_idx, 0, 0]
            stride_a_1 = self.strides_abc[group_idx, 0, 1]
            stride_b_0 = self.strides_abc[group_idx, 1, 0]
            stride_b_1 = self.strides_abc[group_idx, 1, 1]
            stride_c_0 = self.strides_abc[group_idx, 2, 0]
            stride_c_1 = self.strides_abc[group_idx, 2, 1]

            # Construct tensors with shape (mode0, mode1, L=1)
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int32(0)

            # Tensor A: (M, K, 1)
            a_ptr = cute.make_ptr(self.a_dtype, ptr_a, AddressSpace.gmem)
            a_layout = cute.make_layout((M, K, c1),
                                        stride=(stride_a_0, stride_a_1, c0))
            a_tensor = cute.make_tensor(a_ptr, a_layout)

            # Tensor B: (N, K, 1)
            b_ptr = cute.make_ptr(self.b_dtype, ptr_b, AddressSpace.gmem)
            b_layout = cute.make_layout((N, K, c1),
                                        stride=(stride_b_0, stride_b_1, c0))
            b_tensor = cute.make_tensor(b_ptr, b_layout)

            # Tensor C: (M, N, 1)
            c_ptr = cute.make_ptr(self.c_dtype, ptr_c, AddressSpace.gmem)
            c_layout = cute.make_layout((M, N, c1),
                                        stride=(stride_c_0, stride_c_1, c0))
            c_tensor = cute.make_tensor(c_ptr, c_layout)

            # Create TMA atom for A using make_tiled_tma_atom_A
            tma_atom_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
                self.a_tma_op,
                a_tensor,
                self.a_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk_shape,
            )
            cpasync.copy_tensormap(tma_atom_a,
                                   self.get_desc_ptr("a", group_idx))

            # Create TMA atom for B using make_tiled_tma_atom_B
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                self.b_tma_op,
                b_tensor,
                self.b_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk_shape,
            )
            cpasync.copy_tensormap(tma_atom_b,
                                   self.get_desc_ptr("b", group_idx))

            # Create TMA atom for C (S2G) using generic make_tiled_tma_atom
            tma_atom_c, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_tensor,
                self.epi_smem_layout,
                self.epi_tile,
            )
            cpasync.copy_tensormap(tma_atom_c,
                                   self.get_desc_ptr("c", group_idx))


# {$nv-internal-release end}

# =============================================================================
# MoE Grouped GEMM Tensormap Constructor
# =============================================================================


class MoEGroupedGemmTensormapConstructor(OnlineTensormapDescCreator):
    """
    Tensormap descriptor constructor for MoE Grouped GEMM (expert-wise descriptors only).

    Non-expert-wise descriptors are passed directly at kernel launch.
    This class only handles:
    - 2Dx3D: C descriptors (expert-wise, to avoid write conflicts)
    - 2Dx2D: A and B descriptors (expert-wise, tokens is reduction axis)

    All parameters are stored as explicit instance attributes (no dicts).

    Workspace layout:
    - 2Dx3D: [C_0, C_1, ..., C_{n-1}]
    - 2Dx2D: [A_0, A_1, ..., A_{n-1}, B_0, B_1, ..., B_{n-1}]
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        # Codegen-time configs
        a_dtype: Type[Numeric],
        b_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        a_tma_op: cute.CopyAtom,
        b_tma_op: cute.CopyAtom,
        c_tma_op: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        cluster_layout_vmnk_shape: cute.Layout,
        epi_tile: cute.Tile,
        # Runtime params
        a_tensor: cute.Tensor,  # fake GEMM domain A
        b_tensor: cute.Tensor,  # fake GEMM domain B
        c_tensor: cute.Tensor,  # fake GEMM domain C
        offs: cute.Tensor,  # (experts,) cumsum
        workspace_ptr: Pointer,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        # Codegen-time configs
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.a_smem_layout = a_smem_layout
        self.b_smem_layout = b_smem_layout
        self.epi_smem_layout = epi_smem_layout
        self.a_tma_op = a_tma_op
        self.b_tma_op = b_tma_op
        self.c_tma_op = c_tma_op
        self.tiled_mma = tiled_mma
        self.mma_tiler = mma_tiler
        self.cluster_layout_vmnk_shape = cluster_layout_vmnk_shape
        self.epi_tile = epi_tile
        # Runtime params
        self.a_tensor = a_tensor
        self.b_tensor = b_tensor
        self.c_tensor = c_tensor
        self.offs = offs
        # Workspace with scenario-specific slot layout
        if scenario == "2Dx3D":
            self.workspace = TensormapWorkspace(workspace_ptr, ["c"])
        else:
            self.workspace = TensormapWorkspace(workspace_ptr, ["a", "b"])

    @staticmethod
    def get_workspace_size(scenario: Literal["2Dx3D", "2Dx2D"],
                           expert_cnt: int) -> int:
        """Calculate workspace size in bytes for tensormap descriptors."""
        if scenario == "2Dx3D":
            return TensormapWorkspace.size_bytes(1, expert_cnt)  # only C
        else:
            return TensormapWorkspace.size_bytes(2, expert_cnt)  # A and B

    @cute.jit
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        return self.workspace.get_ptr(tensor_name, executor_idx)

    @cute.jit
    def construct_and_write(self,
                            executor_idx: Int32,
                            dependency: Any = None) -> None:
        """
        Create expert-wise tensormap descriptors for the given expert.

        - 2Dx3D: Creates C descriptor for this expert
        - 2Dx2D: Creates A and B descriptors for this expert
        """
        if cutlass.const_expr(self.scenario == "2Dx3D"):
            self._construct_c_desc_2dx3d(executor_idx)
        else:  # 2Dx2D
            self._construct_ab_descs_2dx2d(executor_idx)

    @cute.jit
    def _construct_c_desc_2dx3d(self, expert_idx: Int32) -> None:
        """
        2Dx3D: Create expert-wise C descriptor.
        C tensor: (fake_m, n, 1) = (tokens_sum, intermediate, 1)
        Slice fake_m -> (tokens_i, intermediate, 1) per expert.
        """
        token_offset, tokens_i = compute_expert_token_range(
            self.offs, expert_idx)

        c_ptr = self.c_tensor.iterator
        c_stride = self.c_tensor.stride
        intermediate = self.c_tensor.shape[1]  # type: ignore[index]

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        c_ptr_i = c_ptr + token_offset * c_stride[0]  # type: ignore[index]
        c_layout_i = cute.make_layout(
            (tokens_i, intermediate, c1),
            stride=(c_stride[0], c_stride[1], c0),  # type: ignore[index]
        )
        c_tensor_i = cute.make_tensor(c_ptr_i, c_layout_i)

        tma_atom_c, _ = cpasync.make_tiled_tma_atom(
            self.c_tma_op,
            c_tensor_i,
            self.epi_smem_layout,
            self.epi_tile,
        )
        cpasync.copy_tensormap(tma_atom_c, self.get_desc_ptr("c", expert_idx))

    @cute.jit
    def _construct_ab_descs_2dx2d(self, expert_idx: Int32) -> None:
        """
        2Dx2D: Create expert-wise A and B descriptors.
        A: (m, fake_k, 1) -> slice to (m, tokens_i, 1)
        B: (n, fake_k, 1) -> slice to (n, tokens_i, 1)
        """
        token_offset, tokens_i = compute_expert_token_range(
            self.offs, expert_idx)

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # A tensor: (m, fake_k, 1) -> (m, tokens_i, 1)
        a_ptr = self.a_tensor.iterator
        a_stride = self.a_tensor.stride
        a_m = self.a_tensor.shape[0]  # type: ignore[index]

        a_ptr_i = a_ptr + token_offset * a_stride[1]  # type: ignore[index]
        a_layout_i = cute.make_layout(
            (a_m, tokens_i, c1),
            stride=(a_stride[0], a_stride[1], c0),  # type: ignore[index]
        )
        a_tensor_i = cute.make_tensor(a_ptr_i, a_layout_i)

        tma_atom_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
            self.a_tma_op,
            a_tensor_i,
            self.a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
        )
        cpasync.copy_tensormap(tma_atom_a, self.get_desc_ptr("a", expert_idx))

        # B tensor: (n, fake_k, 1) -> (n, tokens_i, 1)
        b_ptr = self.b_tensor.iterator
        b_stride = self.b_tensor.stride
        b_n = self.b_tensor.shape[0]  # type: ignore[index]

        b_ptr_i = b_ptr + token_offset * b_stride[1]  # type: ignore[index]
        b_layout_i = cute.make_layout(
            (b_n, tokens_i, c1),
            stride=(b_stride[0], b_stride[1], c0),  # type: ignore[index]
        )
        b_tensor_i = cute.make_tensor(b_ptr_i, b_layout_i)

        tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
            self.b_tma_op,
            b_tensor_i,
            self.b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
        )
        cpasync.copy_tensormap(tma_atom_b, self.get_desc_ptr("b", expert_idx))


# =============================================================================
# MoE Scaled Grouped GEMM Tensormap Constructor
# =============================================================================


class MoEScaledGroupedGemmTensormapConstructor(OnlineTensormapDescCreator):
    """
    Tensormap descriptor constructor for MoE Scaled Grouped GEMM (block-scaled).

    .. py:attribute:: ChunkSize
        :value: 128

        Number of experts processed per chunk in the desc_init_kernel.
        Must match the warp-group width (4 warps × 32 threads).

    Extends MoEGroupedGemmTensormapConstructor with SFA/SFB descriptor support.

    Expert-wise descriptors only — non-expert-wise descriptors are passed
    directly at kernel launch.

    Workspace layout:
    - 2Dx3D: [C_0, C_1, ..., C_{n-1}]  (1 slot per expert)
    - 2Dx2D: [A_0, B_0, SFA_0, SFB_0, A_1, B_1, SFA_1, SFB_1, ...]  (4 slots per expert)

    :param scenario: "2Dx3D" or "2Dx2D"
    :param sf_vec_size: Scale factor vector size (32 for MXFP8/MXFP4, 16 for NVFP4)
    :param a_dtype: Data type for tensor A
    :param b_dtype: Data type for tensor B
    :param c_dtype: Data type for tensor C
    :param sf_dtype: Data type for scale factors (SFA/SFB)
    :param a_smem_layout: SMEM layout for A TMA
    :param b_smem_layout: SMEM layout for B TMA
    :param epi_smem_layout: SMEM layout for epilogue (C) TMA
    :param sfa_smem_layout: SMEM layout for SFA TMA
    :param sfb_smem_layout: SMEM layout for SFB TMA
    :param a_tma_op: TMA operation for A
    :param b_tma_op: TMA operation for B
    :param c_tma_op: TMA operation for C (S2G store or reduce)
    :param sfa_tma_op: TMA operation for SFA
    :param sfb_tma_op: TMA operation for SFB
    :param tiled_mma: TiledMma for A/B/SFA/C TMA atom construction
    :param tiled_mma_sfb: TiledMma for SFB (separate due to 2CTA replication)
    :param mma_tiler: MMA tiler shape (M, N, K)
    :param mma_tiler_sfb: MMA tiler shape for SFB
    :param cluster_layout_vmnk_shape: Cluster layout shape for A/B/SFA multicast
    :param cluster_layout_sfb_vmnk_shape: Cluster layout shape for SFB multicast
    :param epi_tile: Epilogue tile shape
    :param a_tensor: Fake GEMM domain A tensor
    :param b_tensor: Fake GEMM domain B tensor
    :param c_tensor: Fake GEMM domain C tensor
    :param sfa_tensor: Fake GEMM domain SFA tensor (atom-tiled layout)
    :param sfb_tensor: Fake GEMM domain SFB tensor (atom-tiled layout)
    :param offs: (experts,) cumsum offsets in data domain
    :param offs_padded: (experts,) cumsum offsets in padded scale domain
    :param workspace_ptr: Pointer to workspace for TMA descriptors
    :param expert_cnt: Total number of experts
    """

    ChunkSize = 128

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        sf_vec_size: int,
        # Codegen-time configs: dtypes
        a_dtype: Type[Numeric],
        b_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
        sf_dtype: Type[Numeric],
        # Codegen-time configs: SMEM layouts
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        epi_smem_layout: cute.Layout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        # Codegen-time configs: TMA ops
        a_tma_op: cute.CopyAtom,
        b_tma_op: cute.CopyAtom,
        c_tma_op: cute.CopyAtom,
        sfa_tma_op: cute.CopyAtom,
        sfb_tma_op: cute.CopyAtom,
        # Codegen-time configs: MMA / cluster / tile
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mma_tiler: cute.Tile,
        mma_tiler_sfb: cute.Tile,
        cluster_layout_vmnk_shape: cute.Layout,
        cluster_layout_sfb_vmnk_shape: cute.Layout,
        epi_tile: cute.Tile,
        # Runtime params
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        offs: cute.Tensor,
        offs_padded: cute.Tensor,
        workspace_ptr: Pointer,
        expert_cnt: Optional[Union[Int32, int]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.sf_vec_size = sf_vec_size
        # Dtypes
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.sf_dtype = sf_dtype
        # SMEM layouts
        self.a_smem_layout = a_smem_layout
        self.b_smem_layout = b_smem_layout
        self.epi_smem_layout = epi_smem_layout
        self.sfa_smem_layout = sfa_smem_layout
        self.sfb_smem_layout = sfb_smem_layout
        # TMA ops
        self.a_tma_op = a_tma_op
        self.b_tma_op = b_tma_op
        self.c_tma_op = c_tma_op
        self.sfa_tma_op = sfa_tma_op
        self.sfb_tma_op = sfb_tma_op
        # MMA / cluster / tile
        self.tiled_mma = tiled_mma
        self.tiled_mma_sfb = tiled_mma_sfb
        self.mma_tiler = mma_tiler
        self.mma_tiler_sfb = mma_tiler_sfb
        self.cluster_layout_vmnk_shape = cluster_layout_vmnk_shape
        self.cluster_layout_sfb_vmnk_shape = cluster_layout_sfb_vmnk_shape
        self.epi_tile = epi_tile
        # Runtime params
        self.a_tensor = a_tensor
        self.b_tensor = b_tensor
        self.c_tensor = c_tensor
        self.sfa_tensor = sfa_tensor
        self.sfb_tensor = sfb_tensor
        self.offs = offs
        self.offs_padded = offs_padded
        self.expert_cnt = expert_cnt
        # Workspace with scenario-specific slot layout
        if scenario == "2Dx3D":
            self.workspace = TensormapWorkspace(workspace_ptr, ["c"])
        else:
            self.workspace = TensormapWorkspace(workspace_ptr,
                                                ["a", "b", "sfa", "sfb"])

    @staticmethod
    def get_workspace_size(scenario: Literal["2Dx3D", "2Dx2D"],
                           expert_cnt: int) -> int:
        """Calculate workspace size in bytes for tensormap descriptors."""
        if scenario == "2Dx3D":
            return TensormapWorkspace.size_bytes(1, expert_cnt)  # C only
        else:
            return TensormapWorkspace.size_bytes(4,
                                                 expert_cnt)  # A, B, SFA, SFB

    @cute.jit
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        return self.workspace.get_ptr(tensor_name, executor_idx)

    @cute.jit
    def construct_and_write(self,
                            lane_in_group: Int32,
                            dependency: Any = None) -> None:
        """Create expert-wise tensormap descriptors for all experts."""
        consumer, smem_offs_padded = dependency
        assert self.expert_cnt is not None
        num_chunks = (self.expert_cnt + self.ChunkSize - 1) // self.ChunkSize

        chunk_idx = cutlass.Int32(0)
        while chunk_idx < num_chunks:
            expert_idx = chunk_idx * self.ChunkSize + lane_in_group
            in_bounds = expert_idx < self.expert_cnt

            # Phase 1: descriptors independent of offs_padded.
            if in_bounds:
                if cutlass.const_expr(self.scenario == "2Dx2D"):
                    self._construct_ab_descs_2dx2d(expert_idx)
                else:
                    self._construct_c_desc_2dx3d(expert_idx)

            # All threads participate in barrier (fixed arrive count)
            handle = consumer.wait_and_advance()

            # Phase 2: SF descriptors read padded offsets from SMEM.
            if in_bounds:
                if cutlass.const_expr(self.scenario == "2Dx2D"):
                    # smem_offs_padded layout: [carry, chunk[0], ..., chunk[127]]
                    # padded_offset = smem[lane]     (prev expert's cumulative)
                    # padded_end    = smem[lane + 1] (this expert's cumulative)
                    padded_offset = smem_offs_padded[lane_in_group]
                    padded_size_i = smem_offs_padded[lane_in_group +
                                                     1] - padded_offset
                    self._construct_sf_descs_2dx2d_direct(
                        expert_idx, padded_offset, padded_size_i)

            # All threads release (fixed arrive count)
            handle.release()

            chunk_idx += 1

    # -----------------------------------------------------------------
    # 2Dx3D: C descriptor (same as MoEGroupedGemmTensormapConstructor)
    # -----------------------------------------------------------------

    @cute.jit
    def _construct_c_desc_2dx3d(self, expert_idx: Int32) -> None:
        """
        2Dx3D: Create expert-wise C descriptor.
        C: (fake_m, n, 1) -> slice to (tokens_i, n, 1) per expert.
        """
        token_offset, tokens_i = compute_expert_token_range(
            self.offs, expert_idx)
        c1 = cutlass.Int32(1)

        c_i = cute.domain_offset((token_offset, 0, 0), self.c_tensor)
        c_i = rewrite_tensor_shape(
            c_i, (tokens_i, self.c_tensor.shape[1], c1))  # type: ignore[index]

        tma_atom_c, _ = cpasync.make_tiled_tma_atom(
            self.c_tma_op,
            c_i,
            self.epi_smem_layout,
            self.epi_tile,
        )
        cpasync.copy_tensormap(tma_atom_c, self.get_desc_ptr("c", expert_idx))

    # -----------------------------------------------------------------
    # 2Dx2D: A, B descriptors (same as MoEGroupedGemmTensormapConstructor)
    # -----------------------------------------------------------------

    @cute.jit
    def _construct_ab_descs_2dx2d(self, expert_idx: Int32) -> None:
        """
        2Dx2D: Create expert-wise A and B descriptors.
        A: (m, fake_k, 1) -> slice to (m, tokens_i, 1)
        B: (n, fake_k, 1) -> slice to (n, tokens_i, 1)
        """
        token_offset, tokens_i = compute_expert_token_range(
            self.offs, expert_idx)
        c1 = cutlass.Int32(1)

        # A: (m, fake_k, 1) -> domain_offset + rewrite shape
        a_i = cute.domain_offset((0, token_offset, 0), self.a_tensor)
        a_i = rewrite_tensor_shape(
            a_i, (self.a_tensor.shape[0], tokens_i, c1))  # type: ignore[index]

        tma_atom_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
            self.a_tma_op,
            a_i,
            self.a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
        )
        cpasync.copy_tensormap(tma_atom_a, self.get_desc_ptr("a", expert_idx))

        # B: (n, fake_k, 1) -> domain_offset + rewrite shape
        b_i = cute.domain_offset((0, token_offset, 0), self.b_tensor)
        b_i = rewrite_tensor_shape(
            b_i, (self.b_tensor.shape[0], tokens_i, c1))  # type: ignore[index]

        tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
            self.b_tma_op,
            b_i,
            self.b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
        )
        cpasync.copy_tensormap(tma_atom_b, self.get_desc_ptr("b", expert_idx))

    # -----------------------------------------------------------------
    # 2Dx2D: SFA, SFB descriptors (new for block-scaled)
    # -----------------------------------------------------------------

    @cute.jit
    def _construct_sf_descs_2dx2d_direct(
        self,
        expert_idx: Int32,
        padded_offset: Int32,
        padded_size_i: Int32,
    ) -> None:
        """
        2Dx2D: Create expert-wise SFA and SFB descriptors with pre-computed
        padded offset and size.

        This variant allows the caller to supply padded offsets from SMEM
        (in desc_init_kernel) instead of reading from ``self.offs_padded`` in GMEM.
        """
        c1 = cutlass.Int32(1)

        a_chunks_to_move = (padded_offset // self.sf_vec_size *
                            cute.size(self.sfa_tensor, mode=[0]) // 128)
        a_elems_to_move = cute.size(
            self.sfa_tensor, mode=[0]) * padded_offset // self.sf_vec_size
        b_chunks_to_move = (padded_offset // self.sf_vec_size *
                            cute.size(self.sfb_tensor, mode=[0]) // 128)
        b_elems_to_move = cute.size(
            self.sfb_tensor, mode=[0]) * padded_offset // self.sf_vec_size

        per_expert_sfa_shape = (self.sfa_tensor.shape[0], padded_size_i, c1
                                )  # type: ignore[index]
        sfa_layout_i = tile_atom_to_shape_SF(per_expert_sfa_shape,
                                             self.sf_vec_size)
        sfa_i = cute.make_tensor(self.sfa_tensor.iterator + a_elems_to_move,
                                 sfa_layout_i)

        tma_atom_sfa, _ = cute.nvgpu.make_tiled_tma_atom_A(
            self.sfa_tma_op,
            sfa_i,
            self.sfa_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
            internal_type=cutlass.Uint64,
        )
        cpasync.copy_tensormap(tma_atom_sfa,
                               self.get_desc_ptr("sfa", expert_idx))

        per_expert_sfb_shape = (self.sfb_tensor.shape[0], padded_size_i, c1
                                )  # type: ignore[index]
        sfb_layout_i = tile_atom_to_shape_SF(per_expert_sfb_shape,
                                             self.sf_vec_size)
        sfb_i = cute.make_tensor(self.sfb_tensor.iterator + b_elems_to_move,
                                 sfb_layout_i)

        tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
            self.sfb_tma_op,
            sfb_i,
            self.sfb_smem_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk_shape,
            internal_type=cutlass.Uint64,
        )
        cpasync.copy_tensormap(tma_atom_sfb,
                               self.get_desc_ptr("sfb", expert_idx))
