# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python SMEM declarations with lifetime-aware overlay placement."""

import dataclasses
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute

from .utils import (
    cosize_from_shape_stride_tuples,
    is_power_of_two,
    round_up,
    row_major_stride,
    validate_static_integer_tuple,
)

SmemRegionKind = Literal["mbarrier", "tensor"]
SwizzleSpec = Tuple[int, int, int]


def _swizzle_alignment(swizzle: Optional[SwizzleSpec]) -> Optional[int]:
    if swizzle is None or swizzle[0] == 0:
        return None
    num_bits, num_base, _ = swizzle
    return 1 << (num_base + num_bits)


@dataclasses.dataclass(frozen=True)
class SmemRegion:
    """One logical SMEM tensor with no assigned byte offset."""

    name: str
    kind: SmemRegionKind
    dtype: Type[cutlass.Numeric]
    shape: Tuple
    stride: Tuple
    swizzle: Optional[SwizzleSpec]
    byte_alignment: int

    @property
    def cosize(self) -> int:
        return int(cosize_from_shape_stride_tuples(self.shape, self.stride))

    @property
    def nbytes(self) -> int:
        return (self.cosize * int(self.dtype.width) + 7) // 8


class SmemLifetime:
    """One mutually exclusive use of an overlay's physical storage."""

    def __init__(
        self,
        workspace: "SmemWorkspace",
        overlay: "SmemOverlay",
        name: str,
    ) -> None:
        self._workspace = workspace
        self._overlay = overlay
        self.name = name
        self._regions: List[SmemRegion] = []

    @property
    def regions(self) -> Tuple[SmemRegion, ...]:
        return tuple(self._regions)

    def register_tensor(
        self,
        name: str,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        *,
        stride: Optional[Tuple] = None,
        swizzle: Optional[SwizzleSpec] = None,
        byte_alignment: Optional[int] = None,
    ) -> SmemRegion:
        region = self._workspace._make_region(
            name=name,
            kind="tensor",
            dtype=dtype,
            shape=shape,
            stride=stride,
            swizzle=swizzle,
            byte_alignment=byte_alignment,
        )
        self._workspace._claim_region_name(region)
        self._regions.append(region)
        return region


class SmemOverlay:
    """One physical allocation shared by mutually exclusive lifetimes."""

    def __init__(self, workspace: "SmemWorkspace", name: str) -> None:
        self._workspace = workspace
        self.name = name
        self._lifetimes: List[SmemLifetime] = []
        self._lifetime_names: set[str] = set()

    @property
    def lifetimes(self) -> Tuple[SmemLifetime, ...]:
        return tuple(self._lifetimes)

    def add_lifetime(self, name: str) -> SmemLifetime:
        if self._workspace.finalized:
            raise RuntimeError("Cannot add an SMEM lifetime after finalize().")
        if not name:
            raise ValueError("An SMEM lifetime needs a non-empty name.")
        if name in self._lifetime_names:
            raise ValueError(f"Duplicate lifetime {name!r} in overlay {self.name!r}.")
        lifetime = SmemLifetime(self._workspace, self, name)
        self._lifetimes.append(lifetime)
        self._lifetime_names.add(name)
        return lifetime


_TopLevelDeclaration = Union[SmemRegion, SmemOverlay]


class SmemWorkspace:
    """Collect static SMEM declarations and finalize one physical placement."""

    def __init__(
        self,
        *,
        base_alignment: int = 1024,
        total_alignment: int = 16,
    ) -> None:
        if not is_power_of_two(base_alignment):
            raise ValueError("base_alignment must be a positive power of two.")
        if not is_power_of_two(total_alignment):
            raise ValueError("total_alignment must be a positive power of two.")
        self.base_alignment = base_alignment
        self.total_alignment = total_alignment
        self._mbarriers: List[SmemRegion] = []
        self._declarations: List[_TopLevelDeclaration] = []
        self._region_by_name: Dict[str, SmemRegion] = {}
        self._overlay_names: set[str] = set()
        self._offset: Dict[str, int] = {}
        self._total_bytes = 0
        self._finalized = False

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "SmemWorkspace":
        return self

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def total_bytes(self) -> int:
        self._require_finalized()
        return self._total_bytes

    def register_mbarrier(
        self,
        name: str,
        count: int,
        *,
        byte_alignment: Optional[int] = None,
    ) -> SmemRegion:
        if count <= 0:
            raise ValueError(f"Mbarrier region {name!r} needs a positive count.")
        region = self._make_region(
            name=name,
            kind="mbarrier",
            dtype=cutlass.Int64,
            shape=(count,),
            stride=(1,),
            swizzle=None,
            byte_alignment=byte_alignment,
        )
        self._claim_region_name(region)
        self._mbarriers.append(region)
        return region

    def register_tensor(
        self,
        name: str,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        *,
        stride: Optional[Tuple] = None,
        swizzle: Optional[SwizzleSpec] = None,
        byte_alignment: Optional[int] = None,
    ) -> SmemRegion:
        region = self._make_region(
            name=name,
            kind="tensor",
            dtype=dtype,
            shape=shape,
            stride=stride,
            swizzle=swizzle,
            byte_alignment=byte_alignment,
        )
        self._claim_region_name(region)
        self._declarations.append(region)
        return region

    def create_overlay(self, name: str) -> SmemOverlay:
        if self._finalized:
            raise RuntimeError("Cannot create an SMEM overlay after finalize().")
        if not name:
            raise ValueError("An SMEM overlay needs a non-empty name.")
        if name in self._overlay_names or name in self._region_by_name:
            raise ValueError(f"Duplicate SMEM overlay {name!r}.")
        overlay = SmemOverlay(self, name)
        self._overlay_names.add(name)
        self._declarations.append(overlay)
        return overlay

    def _make_region(
        self,
        *,
        name: str,
        kind: SmemRegionKind,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        stride: Optional[Tuple],
        swizzle: Optional[SwizzleSpec],
        byte_alignment: Optional[int],
    ) -> SmemRegion:
        if self._finalized:
            raise RuntimeError("Cannot register an SMEM region after finalize().")
        if not name:
            raise ValueError("An SMEM region needs a non-empty name.")
        validate_static_integer_tuple(shape, field_name=f"{name}.shape")
        if stride is None:
            stride = row_major_stride(shape)
        validate_static_integer_tuple(stride, field_name=f"{name}.stride")
        if len(shape) != len(stride):
            raise ValueError(f"SMEM region {name!r} shape and stride ranks differ.")
        if swizzle is not None:
            if len(swizzle) != 3 or not all(isinstance(parameter, int) for parameter in swizzle):
                raise TypeError(f"SMEM region {name!r} swizzle must be three Python ints.")
            if swizzle[0] < 0 or swizzle[1] < 0:
                raise ValueError(f"SMEM region {name!r} swizzle bits/base must be non-negative.")

        natural_alignment = max(1, (int(dtype.width) + 7) // 8)
        explicit_alignment = natural_alignment if byte_alignment is None else byte_alignment
        if not is_power_of_two(explicit_alignment):
            raise ValueError(f"SMEM region {name!r} alignment must be a positive power of two.")
        swizzle_alignment = _swizzle_alignment(swizzle)
        effective_alignment = max(
            explicit_alignment,
            natural_alignment,
            1 if swizzle_alignment is None else swizzle_alignment,
        )
        if effective_alignment > self.base_alignment:
            raise ValueError(
                f"SMEM region {name!r} needs {effective_alignment}B alignment, "
                f"but the workspace base only promises {self.base_alignment}B."
            )
        return SmemRegion(
            name=name,
            kind=kind,
            dtype=dtype,
            shape=shape,
            stride=stride,
            swizzle=swizzle,
            byte_alignment=effective_alignment,
        )

    def _claim_region_name(self, region: SmemRegion) -> None:
        if region.name in self._region_by_name or region.name in self._overlay_names:
            raise ValueError(f"Duplicate SMEM region {region.name!r}.")
        self._region_by_name[region.name] = region

    def estimate_total_bytes(self) -> int:
        """Exact size of what is registered so far, priced by running the real placement.

        ``_build_placement`` is side-effect free -- ``finalize`` is what stores its result -- so the layout can be
        measured without consuming the workspace. Summing region sizes instead would miss the alignment padding
        that only exists once regions are placed, and a budget derived from that undercount overspends the
        workspace: ``finalize`` then rejects a plan the host arithmetic had already accepted.
        """
        return self._build_placement()[1]

    def finalize(self, *, max_bytes: Optional[int] = None) -> None:
        if self._finalized:
            raise RuntimeError("SmemWorkspace.finalize() may only be called once.")
        offsets, total_bytes = self._build_placement()
        if max_bytes is not None and total_bytes > max_bytes:
            raise ValueError(f"SMEM plan needs {total_bytes} bytes, exceeding {max_bytes} bytes.")
        self._offset = offsets
        self._total_bytes = total_bytes
        self._finalized = True

    def _build_placement(self) -> Tuple[Dict[str, int], int]:
        offsets: Dict[str, int] = {}
        cursor = 0
        for mbarrier in self._mbarriers:
            cursor = round_up(cursor, mbarrier.byte_alignment)
            offsets[mbarrier.name] = cursor
            cursor += mbarrier.nbytes
        for declaration in self._declarations:
            if isinstance(declaration, SmemRegion):
                cursor = round_up(cursor, declaration.byte_alignment)
                offsets[declaration.name] = cursor
                cursor += declaration.nbytes
                continue
            relative_offsets, overlay_alignment, overlay_bytes = self._layout_overlay(declaration)
            cursor = round_up(cursor, overlay_alignment)
            for region_name, relative_offset in relative_offsets.items():
                offsets[region_name] = cursor + relative_offset
            cursor += overlay_bytes
        return offsets, int(round_up(cursor, self.total_alignment))

    def _layout_overlay(
        self,
        overlay: SmemOverlay,
    ) -> Tuple[Dict[str, int], int, int]:
        if not overlay.lifetimes:
            raise ValueError(f"SMEM overlay {overlay.name!r} has no lifetimes.")
        relative_offsets: Dict[str, int] = {}
        overlay_alignment = 1
        overlay_bytes = 0
        for lifetime in overlay.lifetimes:
            if not lifetime.regions:
                raise ValueError(f"SMEM lifetime {overlay.name}.{lifetime.name} has no regions.")
            lifetime_cursor = 0
            for region in lifetime.regions:
                overlay_alignment = max(
                    overlay_alignment,
                    region.byte_alignment,
                )
                lifetime_cursor = round_up(
                    lifetime_cursor,
                    region.byte_alignment,
                )
                relative_offsets[region.name] = lifetime_cursor
                lifetime_cursor += region.nbytes
            overlay_bytes = max(overlay_bytes, lifetime_cursor)
        return relative_offsets, overlay_alignment, overlay_bytes

    def regions(self) -> Tuple[SmemRegion, ...]:
        return tuple(self._region_by_name.values())

    def region(self, name: str) -> SmemRegion:
        return self._region_by_name[name]

    def offset(self, name: str) -> int:
        self._require_finalized()
        return self._offset[name]

    def nbytes(self, name: str) -> int:
        return self._region_by_name[name].nbytes

    def byte_alignment(self, name: str) -> int:
        return self._region_by_name[name].byte_alignment

    def storage_class(self) -> type:
        self._require_finalized()
        storage_bytes = max(self._total_bytes, 1)
        base_alignment = self.base_alignment

        @cute.struct
        class SmemStorage:
            buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, storage_bytes],
                base_alignment,
            ]

        return SmemStorage

    @cute.jit
    def ptr(self, name: str, smem_base: cute.Pointer) -> cute.Pointer:
        region = self._region_by_name[name]
        swizzle = None if region.swizzle is None else cute.make_swizzle(*region.swizzle)
        return cute.make_ptr(
            region.dtype,
            smem_base.toint() + self._offset[name],
            smem_base.memspace,
            assumed_align=region.byte_alignment,
            swizzle_=swizzle,
        )

    @cute.jit
    def tensor(self, name: str, smem_base: cute.Pointer) -> cute.Tensor:
        region = self._region_by_name[name]
        layout = cute.make_layout(region.shape, stride=region.stride)
        return cute.make_tensor(self.ptr(name, smem_base), layout)

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("SmemWorkspace must be finalized first.")


__all__ = [
    "SmemLifetime",
    "SmemOverlay",
    "SmemRegion",
    "SmemRegionKind",
    "SmemWorkspace",
    "SwizzleSpec",
]
