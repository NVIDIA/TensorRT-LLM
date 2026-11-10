# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""tcgen05 TMEM transpose used by the Rubin MXFP8 FC12 epilogue.

Accumulator layouts and pipeline synchronization are managed by the callers.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05


class Region:
    """Codegen-time region tag for a 16x32 sub-region within a 32x32 tile."""

    Top = 0
    Bottom = 1


class _TmemTranspose16x32Core:
    """Physical implementation of the 16x32 -> 32x16 TMEM in-place transpose.

    The transpose is a fixed sequence of tcgen05 32-bit element atoms; each
    32-bit slot holds an fp32 value. After all four rounds, each warp lane
    owns one row of 16 columns.

    Per-thread RMEM coordinate convention:

      - ``lane_idx`` -- warp lane id (= thread index within warp), in [0, 32).
      - ``elem_idx`` -- per-thread reg index, in [0, 16).
    """

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor, num_regs: int, offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset, cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),))
        )

    @staticmethod
    def load_subtile_raw_acc(
        tmem_subtile_tensor: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """LDTM the entire 32-lane x 64-col raw acc region of one epi
        subtile into 4 independent (16,) fp32 RMEM tensors.

        Used by the FC1 overlap-acc unroll path to extract all raw acc data
        of the first 2 subtiles up front, so that the acc TMEM can be released
        after the first subtile's 4 LDTMs.

        ``tmem_subtile_tensor`` is the (32 lanes, 64 cols) view onto a
        single epi subtile's acc TMEM region (already offset by
        ``warp_lane_offset + acc_stage_col_offset + subtile_col_offset``).

        Returns a 4-tuple of (16,) fp32 RMEM tensors carrying the FC1 raw
        LDTM distribution:

          [0] gate_lo / first-half top   -- subtile cols 0..31, lanes 0..15
          [1] up_lo   / first-half bot   -- subtile cols 0..31, lanes 16..31
          [2] raw_top / second-half top  -- subtile cols 32..63, lanes 0..15
          [3] raw_bot / second-half bot  -- subtile cols 32..63, lanes 16..31

        4 atom calls of ``Ld16x64bOp(Repetition.x16) Float32`` -- the same
        atom used by the per-subtile entry LDTM.  Each output is in the
        raw-LDTM input distribution consumed by this transpose core.
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), _TmemTranspose16x32Core._io_dtype
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride

        # 4 source 16-lane x 32-col views over the (32, 64) subtile region:
        #   first  half (cols 0..31): top  lanes 0..15  / bot lanes 16..31
        #   second half (cols 32..63): top lanes 0..15  / bot lanes 16..31
        # All offsets are Python ints (compile-time const) so cute can
        # const-fold them and infer the correct (>= 8 B / 2 col) ptr
        # alignment that the LDTM atom requires.  Using ``cutlass.Int32``
        # offsets here would wrap them as SSA values that cute treats as
        # alignment-unknown, tripping the atom's verifier.
        first_top_view = cute.make_tensor(ptr, _TmemTranspose16x32Core._tmem_layout(16, 32))
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off, _TmemTranspose16x32Core._tmem_layout(16, 32)
        )
        second_top_view = cute.make_tensor(ptr + 32, _TmemTranspose16x32Core._tmem_layout(16, 32))
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off, _TmemTranspose16x32Core._tmem_layout(16, 32)
        )

        first_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        first_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)

        cute.copy(
            atom_ld16x64, first_top_view, _TmemTranspose16x32Core._rmem_copy_view(first_top, 16)
        )
        cute.copy(
            atom_ld16x64, first_bot_view, _TmemTranspose16x32Core._rmem_copy_view(first_bot, 16)
        )
        cute.copy(
            atom_ld16x64, second_top_view, _TmemTranspose16x32Core._rmem_copy_view(second_top, 16)
        )
        cute.copy(
            atom_ld16x64, second_bot_view, _TmemTranspose16x32Core._rmem_copy_view(second_bot, 16)
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(self, tmem_ptr, region: int, reg_tensor: Optional[cute.Tensor] = None) -> None:
        # The whole transpose is built from 32-bit element atoms; _io_dtype
        # drives _src_regs / output / every LDTM/STTM atom below, so guard the
        # invariant once here (tautological today, defensive against future
        # dtype edits).
        if cutlass.const_expr(self._io_dtype.width != 32):
            raise TypeError(
                f"{type(self).__name__} requires a 32-bit _io_dtype (the "
                f"transpose uses 32-bit element atoms), got {self._io_dtype} "
                f"(width {self._io_dtype.width})."
            )

        half_lane_off = 16 * self._TmemRowStride
        if region == Region.Top:
            src_ptr = tmem_ptr
            dst_ptr = tmem_ptr
        elif region == Region.Bottom:
            src_ptr = tmem_ptr + half_lane_off
            dst_ptr = tmem_ptr + 16
        else:
            raise ValueError("region must be Region.Top or Region.Bottom")

        self.region = region

        self._tmem_src_full = cute.make_tensor(src_ptr, self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr, self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr, self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(dst_ptr + half_lane_off, self._tmem_layout(16, 16))

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), self._io_dtype
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), self._io_dtype
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), self._io_dtype
        )

        self._src_regs = cute.make_rmem_tensor((16,), self._io_dtype)
        # ``output`` is a bare (16,) RMEM fragment; after all four rounds,
        # lane_idx selects the output row and elem_idx selects its column.
        self.output = cute.make_rmem_tensor((16,), self._io_dtype)

        # skip-R1.Load mode: ``reg_tensor`` must already be in the transpose
        # input distribution produced by ``load_subtile_raw_acc``;
        # we copy it in lieu of the R1 LDTM.
        # Weak entry guard (replaces the removed input contract): the transpose
        # atoms are 32-bit element atoms over exactly 16 regs/lane, so the fed
        # tensor must be a 32-bit element type of size 16.
        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            if cutlass.const_expr(reg_tensor.element_type.width != 32):
                raise TypeError(
                    f"{type(self).__name__} reg_tensor must be a 32-bit element "
                    f"type, got element type "
                    f"{reg_tensor.element_type} (width {reg_tensor.element_type.width})."
                )
            if cutlass.const_expr(cute.size(reg_tensor) != 16):
                raise ValueError(
                    f"{type(self).__name__} reg_tensor must hold exactly 16 elements, got {cute.size(reg_tensor)}."
                )
            for r in range(16):
                self._src_regs[r] = reg_tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r1_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(self._atom_st16x128, self._rmem_copy_view(self.output, 16), self._tmem_src_full)

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r2_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self._src_regs, 16), self._tmem_dst_full)

    # -- R3 ------------------------------------------------------------------

    def r3_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r3_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r3_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

    # -- R4 ------------------------------------------------------------------

    def r4_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r4_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r4_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR4[r]]

    def r4_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

    def from_r1_perm_until_last_store(self) -> cute.Tensor:
        self.r1_perm()
        self.r1_store()
        self.r2_load()
        self.r2_store()
        self.r3_load_top()
        self.r3_load_bot()
        self.r3_perm()
        self.r3_store()
        self.r4_load_top()
        self.r4_load_bot()
        self.r4_perm()
        return self.output


__all__ = ["_TmemTranspose16x32Core"]
