# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen seven-plane MegaMoE weight geometry.

Storage placement is owned by :mod:`megamoe_scheduler.sami.arena`; production
uses one external plane-major live bank without internal receive storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CANONICAL_WEIGHT_PLANE_NAMES = (
    "mega_fc1_weight",
    "mega_fc1_weight_sf",
    "mega_fc2_weight",
    "mega_fc2_weight_sf",
    "fc31_alpha",
    "fc2_alpha",
    "fc1_norm_const",
)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class PlaneViewLayout:
    """One per-expert tensor view of a live weight plane.

    Shape and stride omit the leading physical-slot axis. The descriptor is
    Torch-independent, so validation does not resolve optional FP4/FP8 dtypes.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    element_size: int
    pointer_alignment: int

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.stride):
            raise ValueError("plane view shape and stride must have equal rank")
        if any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError("plane view extents must be positive exact integers")
        if any(type(value) is not int or value <= 0 for value in self.stride):
            raise ValueError("plane view strides must be positive exact integers")
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("plane view dtype must be a non-empty string")
        if type(self.element_size) is not int or self.element_size <= 0:
            raise ValueError("plane view element_size must be a positive exact integer")
        if type(self.pointer_alignment) is not int or self.pointer_alignment <= 0:
            raise ValueError("plane view pointer_alignment must be a positive exact integer")

    @property
    def elements_per_slot(self) -> int:
        if not self.shape:
            return 1
        return 1 + sum((extent - 1) * stride for extent, stride in zip(self.shape, self.stride))

    @property
    def nbytes(self) -> int:
        return self.elements_per_slot * self.element_size

    def batched_metadata(
        self, slot_count: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, int, int]:
        if type(slot_count) is not int or slot_count <= 0:
            raise ValueError("slot_count must be a positive exact integer")
        return (
            (slot_count, *self.shape),
            (self.elements_per_slot, *self.stride),
            self.dtype,
            self.element_size,
            self.pointer_alignment,
        )


@dataclass(frozen=True)
class Plane:
    """One canonical MegaMoE weight plane within an expert bundle.

    The first four fields preserve the original public constructor. Layouts
    created by BundleLayout.create additionally carry the storage view used by
    the loader and the typed view consumed by the kernel.
    """

    name: str
    offset: int
    nbytes: int
    alignment: int
    storage: Optional[PlaneViewLayout] = None
    kernel: Optional[PlaneViewLayout] = None

    def __post_init__(self) -> None:
        if (self.storage is None) != (self.kernel is None):
            raise ValueError("plane storage and kernel layouts must be provided together")
        if self.storage is not None:
            if self.storage.nbytes != self.nbytes:
                raise ValueError(
                    f"{self.name} storage view spans {self.storage.nbytes} bytes; "
                    f"expected {self.nbytes}"
                )
            assert self.kernel is not None
            if self.kernel.nbytes != self.nbytes:
                raise ValueError(
                    f"{self.name} kernel view spans {self.kernel.nbytes} bytes; "
                    f"expected {self.nbytes}"
                )

    def storage_metadata(
        self, slot_count: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, int, int]:
        if self.storage is None:
            raise ValueError(f"{self.name} has no canonical storage view")
        return self.storage.batched_metadata(slot_count)

    def kernel_metadata(
        self, slot_count: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, int, int]:
        if self.kernel is None:
            raise ValueError(f"{self.name} has no canonical kernel view")
        return self.kernel.batched_metadata(slot_count)


@dataclass(frozen=True)
class BundleLayout:
    """NVFP4 bytes/expert for the seven plane-major live tensors.

    ``offset`` and ``stride_bytes`` describe a canonical packed serialization
    only; direct-live SAMI uses each plane's ``nbytes`` as its physical-slot
    byte stride and never allocates that serialization.
    """

    hidden: int
    intermediate: int
    planes: tuple[Plane, ...]
    total_bytes: int
    stride_bytes: int
    alignment: int
    expand_intermediate: Optional[int] = None

    @classmethod
    def create(
        cls,
        hidden: int = 7168,
        intermediate: int = 3072,
        *,
        expand_intermediate: Optional[int] = None,
        weight_bits: int = 4,
        sf_vec_size: int = 16,
        sf_bits: int = 8,
        plane_alignment: int = 512,
    ) -> "BundleLayout":
        dimensions = {
            "hidden": hidden,
            "intermediate": intermediate,
            "weight_bits": weight_bits,
            "sf_vec_size": sf_vec_size,
            "sf_bits": sf_bits,
            "plane_alignment": plane_alignment,
        }
        if any(type(value) is not int or value <= 0 for value in dimensions.values()):
            raise ValueError("all bundle dimensions must be positive exact integers")
        if expand_intermediate is None:
            expand_intermediate = 2 * intermediate
        if type(expand_intermediate) is not int or expand_intermediate <= 0:
            raise ValueError("all bundle dimensions must be positive exact integers")
        if (weight_bits, sf_vec_size, sf_bits) != (4, 16, 8):
            raise ValueError(
                "production bundle format is fixed to NVFP4 weights and "
                "FP8 block-16 scale factors"
            )
        if hidden % sf_vec_size or intermediate % sf_vec_size:
            raise ValueError("hidden/intermediate must be block-scale aligned")
        if expand_intermediate != 2 * intermediate:
            raise ValueError(
                "production bundle requires expand_intermediate == 2 * intermediate"
            )

        # These are the exact TEKit transformed-tensor formulas.  The simpler
        # elements/sf_vec_size expression only coincides at some model shapes.
        fc1_sf_elements = _align_up(expand_intermediate, 128) * _align_up(
            hidden // sf_vec_size, 4
        )
        fc2_sf_elements = _align_up(hidden, 128) * _align_up(
            intermediate // sf_vec_size, 4
        )

        def view(
            shape: tuple[int, ...],
            dtype: str,
            element_size: int,
            pointer_alignment: int,
            *,
            stride: Optional[tuple[int, ...]] = None,
        ) -> PlaneViewLayout:
            if stride is None:
                steps = []
                running = 1
                for extent in reversed(shape):
                    steps.append(running)
                    running *= extent
                stride = tuple(reversed(steps))
            return PlaneViewLayout(
                shape=shape,
                stride=stride,
                dtype=dtype,
                element_size=element_size,
                pointer_alignment=pointer_alignment,
            )

        gate_up = expand_intermediate
        specs = (
            (
                "mega_fc1_weight",
                plane_alignment,
                view((gate_up, hidden // 2), "uint8", 1, 16),
                view(
                    (hidden // 2, gate_up),
                    "float4_e2m1fn_x2",
                    1,
                    16,
                    stride=(1, hidden // 2),
                ),
            ),
            (
                "mega_fc1_weight_sf",
                plane_alignment,
                view((fc1_sf_elements,), "uint8", 1, 16),
                view((fc1_sf_elements,), "float8_e4m3fn", 1, 16),
            ),
            (
                "mega_fc2_weight",
                plane_alignment,
                view((hidden, intermediate // 2), "uint8", 1, 16),
                view(
                    (intermediate // 2, hidden),
                    "float4_e2m1fn_x2",
                    1,
                    16,
                    stride=(1, intermediate // 2),
                ),
            ),
            (
                "mega_fc2_weight_sf",
                plane_alignment,
                view((fc2_sf_elements,), "uint8", 1, 16),
                view((fc2_sf_elements,), "float8_e4m3fn", 1, 16),
            ),
            (
                "fc31_alpha",
                16,
                view((), "float32", 4, 4),
                view((), "float32", 4, 4),
            ),
            (
                "fc2_alpha",
                16,
                view((), "float32", 4, 4),
                view((), "float32", 4, 4),
            ),
            (
                "fc1_norm_const",
                16,
                view((), "float32", 4, 4),
                view((), "float32", 4, 4),
            ),
        )

        planes: list[Plane] = []
        cursor = 0
        for name, alignment, storage, kernel in specs:
            nbytes = storage.nbytes
            if kernel.nbytes != nbytes:
                raise ValueError(f"{name} storage and kernel views span different bytes")
            cursor = _align_up(cursor, alignment)
            planes.append(Plane(name, cursor, nbytes, alignment, storage=storage, kernel=kernel))
            cursor += nbytes
        return cls(
            hidden=hidden,
            intermediate=intermediate,
            planes=tuple(planes),
            total_bytes=sum(plane.nbytes for plane in planes),
            stride_bytes=_align_up(cursor, plane_alignment),
            alignment=plane_alignment,
            expand_intermediate=expand_intermediate,
        )

    @property
    def plane_count(self) -> int:
        return len(self.planes)


__all__ = [
    "BundleLayout",
    "CANONICAL_WEIGHT_PLANE_NAMES",
    "Plane",
    "PlaneViewLayout",
]
