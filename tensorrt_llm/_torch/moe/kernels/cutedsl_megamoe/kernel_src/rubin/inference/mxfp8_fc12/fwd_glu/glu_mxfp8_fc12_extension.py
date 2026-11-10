# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Scheduling adapter for the MXFP8 GLU FC12 kernel."""

import dataclasses
from typing import ClassVar, List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute as cute_ir
from cutlass._mlir.dialects import cute_nvgpu as cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace, Pointer
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ......helpers.dsl_helpers import spin_peek, spin_wait
from .....schedulers.fc12_mapping import BlockPhase, NonSwapAbFc12WorkTileInfo, peek_ready_bit
from .....blackwell.custom_mix_cga_helpers import PipelineTmaUmmaMixedCga

# Forward GLU tensor roles: FC1 activation is token-indexed (M), FC1 weight is
# expert-indexed; "d"/"sfd" are the FC1 fp8 output + E8M0 plane, "c" the raw gate/up.
TensorRole = Literal[
    "fc1_activation",
    "fc1_weight",
    "fc1_activation_sf",
    "fc1_weight_sf",
    "c",
    "d",
    "sfd",
    "topk",
    "fc2_activation",
    "fc2_activation_sf",
    "fc2_weight",
    "fc2_weight_sf",
]
WeightStorageMode = Literal["contiguous", "discrete"]

_descriptor_byte_count = 128
_discrete_weight_descriptor_roles = (
    "fc1_weight",
    "fc1_weight_sf",
    "fc2_weight",
    "fc2_weight_sf",
)
_discrete_weight_descriptor_slot = {
    tensor_role: slot_index
    for slot_index, tensor_role in enumerate(_discrete_weight_descriptor_roles)
}


def discrete_weight_descriptor_workspace_size(expert_count: int, is_mixed_cga: bool = False) -> int:
    """Return storage bytes for four 128-byte TensorMaps per expert and CGA shape."""
    if not isinstance(expert_count, int) or isinstance(expert_count, bool) or expert_count <= 0:
        raise ValueError("expert_count must be a positive integer.")
    if not isinstance(is_mixed_cga, bool):
        raise ValueError(f"is_mixed_cga must be a bool, got {is_mixed_cga!r}.")
    return (
        expert_count
        * (1 + int(is_mixed_cga))
        * len(_discrete_weight_descriptor_roles)
        * _descriptor_byte_count
    )


def discrete_weight_descriptor_workspace_alignment() -> int:
    return _descriptor_byte_count


@cute.jit
def raw_discrete_weight_descriptor_pointer(
    descriptor_workspace: Pointer,
    tensor_role: TensorRole,
    expert_idx: Int32,
    cluster_variant: Int32 = cutlass.Int32(0),
    is_mixed_cga: cutlass.Constexpr = False,
) -> Pointer:
    """Return the 128-byte-aligned descriptor slot for one expert/cluster operand.

    Layout per expert is ``[preferred roles | fallback roles]`` when
    ``is_mixed_cga``; uniform launches keep only the preferred roles.
    """
    slot_index = _discrete_weight_descriptor_slot[tensor_role]
    byte_offset = (
        (expert_idx * Int32(1 + int(is_mixed_cga)) + cluster_variant)
        * Int32(len(_discrete_weight_descriptor_roles))
        + Int32(slot_index)
    ) * Int32(_descriptor_byte_count)
    return (descriptor_workspace + byte_offset).align(_descriptor_byte_count)


@dsl_user_op
def copy_discrete_weight_descriptor_pointer(
    descriptor_workspace: Pointer,
    tensor_role: TensorRole,
    expert_idx: Int32,
    cluster_variant: Int32,
    is_mixed_cga: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
) -> Pointer:
    """Convert a raw GMEM descriptor slot to the type accepted by TMA copy."""
    raw_pointer = raw_discrete_weight_descriptor_pointer(
        descriptor_workspace,
        tensor_role,
        expert_idx,
        cluster_variant,
        is_mixed_cga,
    )
    generic_llvm_pointer = llvm.addrspacecast(
        llvm.PointerType.get(AddressSpace.generic),
        raw_pointer.llvm_ptr,
        loc=loc,
        ip=ip,
    )
    generic_pointer = cute.make_ptr(
        raw_pointer.dtype,
        generic_llvm_pointer,
        AddressSpace.generic,
        assumed_align=raw_pointer.alignment,
        loc=loc,
        ip=ip,
    )
    descriptor_pointer_type = cute_ir.PtrType.get(
        cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        generic_pointer.memspace,
        generic_pointer.alignment,
    )
    return cute_ir.recast_iter(descriptor_pointer_type, generic_pointer.value)


@cute.jit
def _rewrite_tensor_shape(tensor: cute.Tensor, new_shape: Tuple) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.make_layout(new_shape, stride=tensor.stride))


@dataclasses.dataclass(frozen=True)
class GluMxFp8Fc12SchedExtension:
    """Kernel-owned work-tile preparation and GMEM view adapter (non-swap MXFP8)."""

    work_tile_type: ClassVar[type] = NonSwapAbFc12WorkTileInfo

    sf_vec_size: int
    fc1_done_counter_pointer: Pointer
    fc2_spin_threshold: Int32
    fc1_ready_counter_pointer: Optional[Pointer] = None
    cluster_m: int = 1
    done_slot_m: int = 1
    fc2_ready_is_mask: bool = False
    weight_storage_mode: WeightStorageMode = "contiguous"
    weight_descriptor_workspace: Optional[Pointer] = None
    is_mixed_cga: bool = False

    def __post_init__(self) -> None:
        if self.sf_vec_size <= 0:
            raise ValueError(f"sf_vec_size must be positive, got {self.sf_vec_size}.")
        if self.cluster_m <= 0:
            raise ValueError(f"cluster_m must be positive, got {self.cluster_m}.")
        if self.done_slot_m <= 0 or self.cluster_m % self.done_slot_m != 0:
            raise ValueError(
                f"done_slot_m must be positive and divide cluster_m={self.cluster_m}, "
                f"got {self.done_slot_m}."
            )
        if self.weight_storage_mode not in ("contiguous", "discrete"):
            raise ValueError(
                "weight_storage_mode must be 'contiguous' or 'discrete', got "
                f"{self.weight_storage_mode!r}."
            )
        if self.weight_storage_mode == "discrete" and self.weight_descriptor_workspace is None:
            raise ValueError("Discrete weights require a weight descriptor workspace.")
        if not isinstance(self.is_mixed_cga, bool):
            raise ValueError(f"is_mixed_cga must be a bool, got {self.is_mixed_cga!r}.")
        object.__setattr__(self, "fc2_spin_threshold", Int32(self.fc2_spin_threshold))

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.fc1_done_counter_pointer))
        values.extend(extract_mlir_values(self.fc2_spin_threshold))
        if self.fc1_ready_counter_pointer is not None:
            values.extend(extract_mlir_values(self.fc1_ready_counter_pointer))
        if self.weight_descriptor_workspace is not None:
            values.extend(extract_mlir_values(self.weight_descriptor_workspace))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GluMxFp8Fc12SchedExtension":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field, values[value_index : value_index + field_value_count]
            )
            value_index += field_value_count
            return result

        fc1_done_counter_pointer = rebuild(self.fc1_done_counter_pointer)
        fc2_spin_threshold = rebuild(self.fc2_spin_threshold)
        fc1_ready_counter_pointer = (
            rebuild(self.fc1_ready_counter_pointer)
            if self.fc1_ready_counter_pointer is not None
            else None
        )
        weight_descriptor_workspace = (
            rebuild(self.weight_descriptor_workspace)
            if self.weight_descriptor_workspace is not None
            else None
        )
        if value_index != len(values):
            raise ValueError(
                f"GluMxFp8Fc12SchedExtension MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=fc1_ready_counter_pointer,
            cluster_m=self.cluster_m,
            done_slot_m=self.done_slot_m,
            fc2_ready_is_mask=self.fc2_ready_is_mask,
            weight_storage_mode=self.weight_storage_mode,
            weight_descriptor_workspace=weight_descriptor_workspace,
            is_mixed_cga=self.is_mixed_cga,
        )

    @cute.jit
    def _counter_slot(self, work_tile: NonSwapAbFc12WorkTileInfo) -> Int32:
        # Cluster-granular token-block slot: dispatch_pull increments one counter per
        # cluster-level token block, so M-direction tiles fold by cluster_m.
        return work_tile.cumulative_token_block_count + work_tile.tile_m_idx // Int32(
            self.cluster_m
        )

    @cute.jit
    def _done_counter_slot(self, work_tile: NonSwapAbFc12WorkTileInfo) -> Int32:
        # fc1_done folds M by done_slot_m, which is coarser than cluster_m only on the
        # standalone path. Mirrors the epilogue producer and the kernel's fc2 consumer.
        return work_tile.cumulative_token_block_count * Int32(
            self.cluster_m // self.done_slot_m
        ) + work_tile.tile_m_idx // Int32(self.done_slot_m)

    @cute.jit
    def prepare_work_tile(self, work_tile: NonSwapAbFc12WorkTileInfo) -> NonSwapAbFc12WorkTileInfo:
        """Pack kernel readiness observations into the published tile flags."""
        phase_and_flags = work_tile.phase_and_flags
        if work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            is_fc2 = work_tile.phase == Int32(BlockPhase.Linear2)

            if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
                if is_fc1:
                    counter_pointer = self.fc1_ready_counter_pointer + self._counter_slot(work_tile)
                    peek_flag = Int32(0)
                    if spin_peek(
                        counter_pointer,
                        lambda value: value >= work_tile.valid_tokens_in_cluster_tile,
                    ):
                        peek_flag = Int32(peek_ready_bit)
                    phase_and_flags = work_tile.phase_and_flags | peek_flag

            if is_fc2:
                counter_pointer = self.fc1_done_counter_pointer + self._done_counter_slot(work_tile)
                peek_flag = Int32(0)
                if cutlass.const_expr(self.fc2_ready_is_mask):
                    ready = spin_peek(
                        counter_pointer,
                        lambda value: (value & self.fc2_spin_threshold) == self.fc2_spin_threshold,
                    )
                else:
                    ready = spin_peek(
                        counter_pointer,
                        lambda value: value >= self.fc2_spin_threshold,
                    )
                if ready:
                    peek_flag = Int32(peek_ready_bit)
                phase_and_flags = work_tile.phase_and_flags | peek_flag

        return NonSwapAbFc12WorkTileInfo(
            expert_idx=work_tile.expert_idx,
            tile_m_idx=work_tile.tile_m_idx,
            tile_n_idx=work_tile.tile_n_idx,
            cumulative_data_physical_row=work_tile.cumulative_data_physical_row,
            cumulative_sf_physical_row=work_tile.cumulative_sf_physical_row,
            cumulative_token_block_count=work_tile.cumulative_token_block_count,
            valid_tokens_in_cta_cluster_tile=work_tile.valid_tokens_in_cta_cluster_tile,
            phase_and_flags=phase_and_flags,
        )

    @cute.jit
    def wait_for_input(self, work_tile: NonSwapAbFc12WorkTileInfo) -> None:
        """Wait until this FC1 input tile's cluster-level token count has arrived."""
        if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
            counter_pointer = self.fc1_ready_counter_pointer + self._counter_slot(work_tile)
            spin_wait(
                counter_pointer,
                lambda value: value >= work_tile.valid_tokens_in_cluster_tile,
                peek_status=work_tile.peek_ready,
            )

    @cute.jit
    def _discrete_weight_descriptor(
        self,
        tensor_role: TensorRole,
        expert_idx: Int32,
        is_fallback_cluster: Boolean,
    ) -> Pointer:
        cluster_variant = Int32(0)
        if cutlass.const_expr(self.is_mixed_cga):
            if is_fallback_cluster:
                cluster_variant = Int32(1)
        return copy_discrete_weight_descriptor_pointer(
            self.weight_descriptor_workspace,
            tensor_role,
            expert_idx,
            cluster_variant,
            self.is_mixed_cga,
        )

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: TensorRole,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
        is_fallback_cluster: Boolean = False,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """Phase-invariant GMEM slice for the operands."""
        expert_idx = work_tile_info.expert_idx
        data_token_offset = work_tile_info.cumulative_data_physical_row
        sf_token_offset = work_tile_info.cumulative_sf_physical_row

        shape = gmem_tensor_in_moe_view.shape
        stride = gmem_tensor_in_moe_view.stride
        c1 = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        if cutlass.const_expr(tensor_name == "fc1_activation"):
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc1_weight"):
            if cutlass.const_expr(self.weight_storage_mode == "discrete"):
                real = _rewrite_tensor_shape(gmem_tensor_in_moe_view, (shape[0], shape[1], c1))
                descriptor = self._discrete_weight_descriptor(
                    tensor_name, expert_idx, is_fallback_cluster
                )
                return (real, descriptor)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc1_activation_sf"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc1_weight_sf"):
            if cutlass.const_expr(self.weight_storage_mode == "discrete"):
                per_expert_shape = (shape[0], shape[1], c1)
                sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
                real = cute.make_tensor(
                    gmem_tensor_in_moe_view.iterator,
                    cute.make_layout(sf_layout.shape, stride=stride),
                )
                descriptor = self._discrete_weight_descriptor(
                    tensor_name, expert_idx, is_fallback_cluster
                )
                return (real, descriptor)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "c"):
            # Raw fc1 accumulator output (gate+up FP32, pre-SwiGLU): token-indexed.
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "d"):
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "sfd"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "topk"):
            real = cute.domain_offset((data_token_offset,), gmem_tensor_in_moe_view)
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_activation"):
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc2_activation_sf"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_weight"):
            if cutlass.const_expr(self.weight_storage_mode == "discrete"):
                real = _rewrite_tensor_shape(gmem_tensor_in_moe_view, (shape[0], shape[1], c1))
                descriptor = self._discrete_weight_descriptor(
                    tensor_name, expert_idx, is_fallback_cluster
                )
                return (real, descriptor)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc2_weight_sf"):
            if cutlass.const_expr(self.weight_storage_mode == "discrete"):
                per_expert_shape = (shape[0], shape[1], c1)
                sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
                real = cute.make_tensor(
                    gmem_tensor_in_moe_view.iterator,
                    cute.make_layout(sf_layout.shape, stride=stride),
                )
                descriptor = self._discrete_weight_descriptor(
                    tensor_name, expert_idx, is_fallback_cluster
                )
                return (real, descriptor)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")


__all__ = [
    "GluMxFp8Fc12SchedExtension",
    "MixedCgaPipeline",
    "OneShotSchedConsumer",
    "TensorRole",
    "WeightStorageMode",
    "make_mixed_cga_tma_umma_pipeline",
]


class OneShotSchedConsumer:
    """Scheduler consumer that maps blockIdx.z directly to one work tile (FC1 tiles first,
    then FC2); the first ``consume_work()`` returns it, later calls return the DONE sentinel."""

    def __init__(self, num_fc1_tiles: int, num_fc2_tiles: int, valid_tokens_packed: int):
        self._num_fc1_tiles = num_fc1_tiles
        self._num_fc2_tiles = num_fc2_tiles
        self._valid_tokens_packed = valid_tokens_packed
        self._consumed = Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return extract_mlir_values(self._consumed)

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "OneShotSchedConsumer":
        expected_value_count = len(extract_mlir_values(self._consumed))
        if len(values) != expected_value_count:
            raise ValueError(
                f"OneShotSchedConsumer MLIR value count mismatch: expected {expected_value_count}, got {len(values)}."
            )
        result = type(self)(self._num_fc1_tiles, self._num_fc2_tiles, self._valid_tokens_packed)
        result._consumed = new_from_mlir_values(self._consumed, values)
        return result

    @cute.jit
    def consume_work(self) -> NonSwapAbFc12WorkTileInfo:
        _, _, bidz = cute.arch.block_idx()
        z = Int32(bidz)
        expert = Int32(-1)
        tile_n = Int32(0)
        phase = Int32(int(BlockPhase.Linear1))
        if self._consumed == Int32(0):
            if z < Int32(self._num_fc1_tiles):
                expert = Int32(0)
                tile_n = z
            else:
                if z < Int32(self._num_fc1_tiles + self._num_fc2_tiles):
                    expert = Int32(0)
                    tile_n = z - Int32(self._num_fc1_tiles)
                    phase = Int32(int(BlockPhase.Linear2))
        self._consumed = Int32(1)
        return NonSwapAbFc12WorkTileInfo(
            expert,
            Int32(0),
            tile_n,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(self._valid_tokens_packed),
            phase,
        )


# Mixed-CGA pipeline adapter for make_participants().


@dataclasses.dataclass(frozen=True)
class MixedCgaPipeline(PipelineTmaUmmaMixedCga):
    """``PipelineTmaUmmaMixedCga`` with the ``loc`` / ``ip`` keywords that
    ``PipelineProducer`` / ``PipelineConsumer`` pass to pipeline methods."""

    @cute.jit
    def producer_acquire(
        self, state, try_acquire_token=None, *, expected_tx=None, loc=None, ip=None
    ) -> None:
        super().producer_acquire(state, try_acquire_token, expected_tx=expected_tx)

    @cute.jit
    def producer_commit(self, state, *, loc=None, ip=None) -> None:
        super().producer_commit(state)

    @cute.jit
    def consumer_release(self, state, *, loc=None, ip=None) -> None:
        super().consumer_release(state)


def make_mixed_cga_tma_umma_pipeline(**create_kwargs) -> MixedCgaPipeline:
    """Create the pipeline with the CUTLASS factory and rewrap it as ``MixedCgaPipeline``."""
    base = PipelineTmaUmmaMixedCga.create(**create_kwargs)
    return MixedCgaPipeline(
        sync_object_full=base.sync_object_full,
        sync_object_empty=base.sync_object_empty,
        num_stages=base.num_stages,
        producer_mask=base.producer_mask,
        consumer_mask=base.consumer_mask,
        is_mma_leader_cta=base.is_mma_leader_cta,
        cta_group=base.cta_group,
    )
