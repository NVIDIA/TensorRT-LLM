# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility


def _install_flash_attn_tile_scheduler_compatibility() -> None:
    """Give FA4's four-axis CuTe tiles their own coordinate storage and MLIR reconstruction."""
    try:
        from flash_attn.cute import tile_scheduler
    except (ImportError, OSError):
        return
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Boolean, extract_mlir_values

    work_tile_info = tile_scheduler.WorkTileInfo
    if "__init__" in vars(work_tile_info):
        return

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean) -> None:
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)
        self._tile_idx_num_values = None

    def __extract_mlir_values__(self) -> list:
        tile_idx_values = extract_mlir_values(self._tile_idx)
        valid_values = extract_mlir_values(self._is_valid_tile)
        self._tile_idx_num_values = len(tile_idx_values)
        return tile_idx_values + valid_values

    @cute.jit
    def tile_idx(self) -> cute.Coord:
        return self._tile_idx

    @cute.jit
    def is_valid_tile(self) -> Boolean:
        return self._is_valid_tile

    work_tile_info.__init__ = __init__
    work_tile_info.__extract_mlir_values__ = __extract_mlir_values__
    work_tile_info.tile_idx = property(tile_idx)
    work_tile_info.is_valid_tile = property(is_valid_tile)


def load_flash_attention_4_forward():
    os.environ.setdefault("FA_DISABLE_2CTA", "1")
    install_cutlass_dsl_compatibility()
    from flash_attn.cute.interface import _flash_attn_fwd

    _install_flash_attn_tile_scheduler_compatibility()
    return _flash_attn_fwd
