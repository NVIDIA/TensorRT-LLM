"""Block-scaled SwapAb adapter between FC12 scheduling and kernel tensor views."""

import dataclasses
from typing import ClassVar, List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from .....helpers.dsl_helpers import spin_peek, spin_wait
from ....schedulers.fc12_mapping import BlockPhase, SwapAbFc12WorkTileInfo, peek_ready_bit


TensorRole = Literal["a", "b", "sfa", "sfb", "c", "sfc", "topk"]


@cute.jit
def _rewrite_tensor_shape(tensor: cute.Tensor, new_shape: Tuple) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.make_layout(new_shape, stride=tensor.stride))


@dataclasses.dataclass(frozen=True)
class BlockScaledSwapAbFc12Extension:
    """Kernel-owned work-tile preparation and GMEM view adapter."""

    work_tile_type: ClassVar[type] = SwapAbFc12WorkTileInfo

    sf_vec_size: int
    fc1_done_counter_pointer: Pointer
    fc2_spin_threshold: Int32
    fc1_ready_counter_pointer: Optional[Pointer] = None

    def __post_init__(self) -> None:
        if self.sf_vec_size <= 0:
            raise ValueError(f"sf_vec_size must be positive, got {self.sf_vec_size}.")
        object.__setattr__(self, "fc2_spin_threshold", Int32(self.fc2_spin_threshold))

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.fc1_done_counter_pointer))
        values.extend(extract_mlir_values(self.fc2_spin_threshold))
        if self.fc1_ready_counter_pointer is not None:
            values.extend(extract_mlir_values(self.fc1_ready_counter_pointer))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "BlockScaledSwapAbFc12Extension":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return result

        fc1_done_counter_pointer = rebuild(self.fc1_done_counter_pointer)
        fc2_spin_threshold = rebuild(self.fc2_spin_threshold)
        fc1_ready_counter_pointer = (
            rebuild(self.fc1_ready_counter_pointer) if self.fc1_ready_counter_pointer is not None else None
        )
        if value_index != len(values):
            raise ValueError(
                f"BlockScaledSwapAbFc12Extension MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=fc1_ready_counter_pointer,
        )

    @cute.jit
    def prepare_work_tile(self, work_tile: SwapAbFc12WorkTileInfo) -> SwapAbFc12WorkTileInfo:
        """Pack kernel readiness observations into the published tile flags."""
        phase_and_flags = work_tile.phase_and_flags
        if work_tile.is_valid_tile:
            counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            is_fc2 = work_tile.phase == Int32(BlockPhase.Linear2)

            if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
                if is_fc1:
                    counter_pointer = self.fc1_ready_counter_pointer + counter_slot
                    peek_flag = Int32(0)
                    if spin_peek(counter_pointer, lambda value: value >= work_tile.valid_tokens_in_cta_tile):
                        peek_flag = Int32(peek_ready_bit)
                    phase_and_flags = work_tile.phase_and_flags | peek_flag

            if is_fc2:
                counter_pointer = self.fc1_done_counter_pointer + counter_slot
                peek_flag = Int32(0)
                if spin_peek(counter_pointer, lambda value: value >= self.fc2_spin_threshold):
                    peek_flag = Int32(peek_ready_bit)
                phase_and_flags = work_tile.phase_and_flags | peek_flag

        return SwapAbFc12WorkTileInfo(
            expert_idx=work_tile.expert_idx,
            tile_m_idx=work_tile.tile_m_idx,
            tile_n_idx=work_tile.tile_n_idx,
            cumulative_data_physical_row=(work_tile.cumulative_data_physical_row),
            cumulative_sf_physical_row=(work_tile.cumulative_sf_physical_row),
            cumulative_token_block_count=(work_tile.cumulative_token_block_count),
            valid_tokens_in_cta_tile=(work_tile.valid_tokens_in_cta_tile),
            phase_and_flags=phase_and_flags,
        )

    @cute.jit
    def wait_for_input(self, work_tile: SwapAbFc12WorkTileInfo) -> None:
        """Wait until the current FC1 input tile is ready."""
        if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
            counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
            counter_pointer = self.fc1_ready_counter_pointer + counter_slot
            spin_wait(
                counter_pointer,
                lambda value: value >= work_tile.valid_tokens_in_cta_tile,
                peek_status=work_tile.peek_ready,
            )

    @cute.jit
    def get_gmem_tensor(
        self, tensor_name: TensorRole, tensor: cute.Tensor, work_tile: SwapAbFc12WorkTileInfo
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """Resolve one kernel tensor to its current expert/task-tile view."""
        expert_idx = work_tile.expert_idx
        data_token_offset = work_tile.cumulative_data_physical_row
        sf_token_offset = work_tile.cumulative_sf_physical_row
        shape = tensor.shape
        stride = tensor.stride
        singleton = Int32(1)

        if cutlass.const_expr(tensor_name == "a"):
            result = cute.domain_offset((0, 0, expert_idx), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "b"):
            result = cute.domain_offset((data_token_offset, 0, 0), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "sfa"):
            result = cute.domain_offset((0, 0, expert_idx), tensor)
            per_expert_shape = (shape[0], shape[1], singleton)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            return (cute.make_tensor(result.iterator, cute.make_layout(sf_layout.shape, stride=stride)), None)

        if cutlass.const_expr(tensor_name in ("sfb", "sfc")):
            result = cute.domain_offset((sf_token_offset, 0, 0), tensor)
            per_expert_shape = (shape[0], shape[1], singleton)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            return (cute.make_tensor(result.iterator, cute.make_layout(sf_layout.shape, stride=stride)), None)

        if cutlass.const_expr(tensor_name == "c"):
            result = cute.domain_offset((data_token_offset, 0, 0), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "topk"):
            return (cute.domain_offset((data_token_offset,), tensor), None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")


__all__ = ["BlockScaledSwapAbFc12Extension", "TensorRole"]
