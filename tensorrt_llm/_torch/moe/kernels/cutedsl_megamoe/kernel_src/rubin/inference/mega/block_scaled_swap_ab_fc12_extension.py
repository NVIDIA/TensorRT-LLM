"""Rubin block-scaled SwapAB adapter between FC12 scheduling and kernel tensor views."""

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
    # Which expert holds each slot the scheduler enumerates. Absent when the
    # scheduler walks experts directly, in which case a slot is an expert.
    expert_remap: Optional[cute.Tensor] = None
    # n_split: expert e's weights and scales sit at half * E + e (half = tile_m_idx // half_cta_tiles of its phase).
    n_split: bool = False
    local_expert_count: int = 0
    fc1_half_cta_tiles: int = 0
    fc2_half_cta_tiles: int = 0
    # Shared-expert FC1 done counter (one slot per token block) and the count a shared FC2 tile waits for.
    shared_fc1_done_counter_pointer: Optional[Pointer] = None
    shared_fc2_spin_threshold: Optional[Int32] = None
    # Ring workspace (0: off). The dispatched-token pool and the FC1 output hold ``ring_token_blocks`` token
    # blocks instead of the worst case: pool token block b lives in slot b % ring_token_blocks, a slot is
    # ``ring_data_rows_per_block`` data rows (one token tile) and ``ring_sf_rows_per_block`` scale rows, a whole
    # multiple of the data rows (the kernels validate the ring and its token tile when they are constructed).
    ring_token_blocks: int = 0
    ring_data_rows_per_block: int = 0
    ring_sf_rows_per_block: int = 0

    def __post_init__(self) -> None:
        if self.sf_vec_size <= 0:
            raise ValueError(f"sf_vec_size must be positive, got {self.sf_vec_size}.")
        object.__setattr__(self, "fc2_spin_threshold", Int32(self.fc2_spin_threshold))
        if (self.shared_fc1_done_counter_pointer is None) != (self.shared_fc2_spin_threshold is None):
            raise ValueError("shared_fc1_done_counter_pointer and shared_fc2_spin_threshold must be set together.")
        if self.shared_fc2_spin_threshold is not None:
            object.__setattr__(self, "shared_fc2_spin_threshold", Int32(self.shared_fc2_spin_threshold))

    @property
    def shared_enabled(self) -> bool:
        return self.shared_fc1_done_counter_pointer is not None

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.fc1_done_counter_pointer))
        values.extend(extract_mlir_values(self.fc2_spin_threshold))
        if self.fc1_ready_counter_pointer is not None:
            values.extend(extract_mlir_values(self.fc1_ready_counter_pointer))
        if self.expert_remap is not None:
            values.extend(extract_mlir_values(self.expert_remap))
        if self.shared_fc1_done_counter_pointer is not None:
            values.extend(extract_mlir_values(self.shared_fc1_done_counter_pointer))
            values.extend(extract_mlir_values(self.shared_fc2_spin_threshold))
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
        expert_remap = rebuild(self.expert_remap) if self.expert_remap is not None else None
        shared_fc1_done_counter_pointer = None
        shared_fc2_spin_threshold = None
        if self.shared_fc1_done_counter_pointer is not None:
            shared_fc1_done_counter_pointer = rebuild(self.shared_fc1_done_counter_pointer)
            shared_fc2_spin_threshold = rebuild(self.shared_fc2_spin_threshold)
        if value_index != len(values):
            raise ValueError(
                f"BlockScaledSwapAbFc12Extension MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=fc1_ready_counter_pointer,
            expert_remap=expert_remap,
            n_split=self.n_split,
            local_expert_count=self.local_expert_count,
            fc1_half_cta_tiles=self.fc1_half_cta_tiles,
            fc2_half_cta_tiles=self.fc2_half_cta_tiles,
            shared_fc1_done_counter_pointer=shared_fc1_done_counter_pointer,
            shared_fc2_spin_threshold=shared_fc2_spin_threshold,
            ring_token_blocks=self.ring_token_blocks,
            ring_data_rows_per_block=self.ring_data_rows_per_block,
            ring_sf_rows_per_block=self.ring_sf_rows_per_block,
        )

    @property
    def ring_enabled(self) -> bool:
        return self.ring_token_blocks > 0

    @cute.jit
    def _half_cta_tiles(self, work_tile: SwapAbFc12WorkTileInfo) -> Int32:
        half_cta_tiles = Int32(self.fc2_half_cta_tiles)
        if work_tile.phase == Int32(BlockPhase.Linear1):
            half_cta_tiles = Int32(self.fc1_half_cta_tiles)
        return half_cta_tiles

    @cute.jit
    def weight_half(self, work_tile: SwapAbFc12WorkTileInfo) -> Int32:
        """Die-major half (0 / 1) holding this tile's weight rows; 0 unless n_split, and for a shared tile."""
        half = Int32(0)
        if cutlass.const_expr(self.n_split):
            half = work_tile.tile_m_idx // self._half_cta_tiles(work_tile)
            if cutlass.const_expr(self.shared_enabled):
                # The shared expert's operands are not die-major.
                if work_tile.phase >= Int32(BlockPhase.SharedLinear1):
                    half = Int32(0)
        return half

    @cute.jit
    def local_tile_m(self, work_tile: SwapAbFc12WorkTileInfo) -> Int32:
        """Weight-row CTA tile index inside the die-major half (== tile_m_idx unless n_split, and for a shared tile)."""
        local = work_tile.tile_m_idx
        if cutlass.const_expr(self.n_split):
            local = work_tile.tile_m_idx % self._half_cta_tiles(work_tile)
            if cutlass.const_expr(self.shared_enabled):
                if work_tile.phase >= Int32(BlockPhase.SharedLinear1):
                    local = work_tile.tile_m_idx
        return local

    @cute.jit
    def prepare_work_tile(self, work_tile: SwapAbFc12WorkTileInfo) -> SwapAbFc12WorkTileInfo:
        """Pack kernel readiness observations into the published tile flags.

        Also the one place the scheduler's slot becomes an expert index, when the
        two differ. Translating here rather than at each use is what keeps the
        remap invisible downstream: every consumer of a published tile -- the
        weight and scale-factor views, the per-expert alphas, the per-expert
        scale-factor gate -- addresses an expert. The row offsets the tile carries
        were resolved in slot order and the remap leaves them alone. With a ring
        workspace a routed tile is republished in block form (the last branch).
        """
        phase_and_flags = work_tile.phase_and_flags
        expert_idx = work_tile.expert_idx
        tile_n_idx = work_tile.tile_n_idx
        data_physical_row = work_tile.cumulative_data_physical_row
        sf_physical_row = work_tile.cumulative_sf_physical_row
        token_block_count = work_tile.cumulative_token_block_count
        if work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            is_fc2 = work_tile.phase == Int32(BlockPhase.Linear2)
            # Inside the validity branch so a terminal tile keeps its sentinel; shared tiles are never remapped.
            if cutlass.const_expr(self.expert_remap is not None):
                if cutlass.const_expr(self.shared_enabled):
                    if is_fc1 | is_fc2:
                        expert_idx = self.expert_remap[work_tile.expert_idx]
                else:
                    expert_idx = self.expert_remap[work_tile.expert_idx]
            counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx

            if cutlass.const_expr(self.shared_enabled):
                if work_tile.phase == Int32(BlockPhase.SharedLinear2):
                    counter_pointer = self.shared_fc1_done_counter_pointer + work_tile.tile_n_idx
                    peek_flag = Int32(0)
                    if spin_peek(counter_pointer, lambda value: value >= self.shared_fc2_spin_threshold):
                        peek_flag = Int32(peek_ready_bit)
                    phase_and_flags = work_tile.phase_and_flags | peek_flag

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

            if cutlass.const_expr(self.ring_enabled):
                # A routed tile is published in block form: the token block is the whole tile (tile_n_idx 0),
                # cumulative_token_block_count is the pool token block itself, the data row is the absolute pool
                # row of the block (metadata and scores stay pool-indexed) and the scale row is the row of the
                # block's ring slot. Every consumer that adds tile_n_idx * tile to a base keeps working, and the
                # ring pools are addressed from the scale row alone (see get_gmem_tensor).
                if is_fc1 | is_fc2:
                    data_physical_row = work_tile.cumulative_data_physical_row + work_tile.tile_n_idx * Int32(
                        self.ring_data_rows_per_block
                    )
                    sf_physical_row = (counter_slot % Int32(self.ring_token_blocks)) * Int32(
                        self.ring_sf_rows_per_block
                    )
                    token_block_count = counter_slot
                    tile_n_idx = Int32(0)

        return SwapAbFc12WorkTileInfo(
            expert_idx=expert_idx,
            tile_m_idx=work_tile.tile_m_idx,
            tile_n_idx=tile_n_idx,
            cumulative_data_physical_row=data_physical_row,
            cumulative_sf_physical_row=sf_physical_row,
            cumulative_token_block_count=token_block_count,
            valid_tokens_in_cta_tile=work_tile.valid_tokens_in_cta_tile,
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
        # Row of the token block in the pools read as B and written as C. With a ring workspace a routed tile
        # carries its ring scale row (prepare_work_tile); the data row of the slot follows from it. Shared tiles
        # keep their own rows.
        pool_token_offset = data_token_offset
        if cutlass.const_expr(self.ring_enabled):
            ring_data_row = sf_token_offset // Int32(self.ring_sf_rows_per_block // self.ring_data_rows_per_block)
            if cutlass.const_expr(self.shared_enabled):
                if (work_tile.phase == Int32(BlockPhase.Linear1)) | (work_tile.phase == Int32(BlockPhase.Linear2)):
                    pool_token_offset = ring_data_row
            else:
                pool_token_offset = ring_data_row
        shape = tensor.shape
        stride = tensor.stride
        singleton = Int32(1)
        weight_expert = expert_idx
        if cutlass.const_expr(self.n_split):
            weight_expert = self.weight_half(work_tile) * Int32(self.local_expert_count) + expert_idx

        if cutlass.const_expr(tensor_name == "a"):
            result = cute.domain_offset((0, 0, weight_expert), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "b"):
            result = cute.domain_offset((pool_token_offset, 0, 0), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "sfa"):
            result = cute.domain_offset((0, 0, weight_expert), tensor)
            per_expert_shape = (shape[0], shape[1], singleton)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            return (cute.make_tensor(result.iterator, cute.make_layout(sf_layout.shape, stride=stride)), None)

        if cutlass.const_expr(tensor_name in ("sfb", "sfc")):
            result = cute.domain_offset((sf_token_offset, 0, 0), tensor)
            per_expert_shape = (shape[0], shape[1], singleton)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            return (cute.make_tensor(result.iterator, cute.make_layout(sf_layout.shape, stride=stride)), None)

        if cutlass.const_expr(tensor_name == "c"):
            result = cute.domain_offset((pool_token_offset, 0, 0), tensor)
            return (_rewrite_tensor_shape(result, (shape[0], shape[1], singleton)), None)

        if cutlass.const_expr(tensor_name == "topk"):
            return (cute.domain_offset((data_token_offset,), tensor), None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")


__all__ = ["BlockScaledSwapAbFc12Extension", "TensorRole"]
