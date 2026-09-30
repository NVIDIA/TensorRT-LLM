"""Frozen seven-plane MegaMoE weight geometry.

Storage placement is owned by :mod:`megamoe_scheduler.sami.arena`; production
uses one external plane-major live bank without internal receive storage.
"""

from __future__ import annotations

from dataclasses import dataclass


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Plane:
    """One canonical MegaMoE weight plane within an expert bundle."""

    name: str
    offset: int
    nbytes: int
    alignment: int


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

    @classmethod
    def create(
        cls,
        hidden: int = 7168,
        intermediate: int = 3072,
        *,
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
        if (weight_bits, sf_vec_size, sf_bits) != (4, 16, 8):
            raise ValueError(
                "production bundle format is fixed to NVFP4 weights and "
                "FP8 block-16 scale factors"
            )
        if hidden % sf_vec_size or intermediate % sf_vec_size:
            raise ValueError("hidden/intermediate must be block-scale aligned")

        fc1_elements = 2 * intermediate * hidden
        fc2_elements = hidden * intermediate
        # These are the exact TEKit transformed-tensor formulas.  The simpler
        # elements/sf_vec_size expression only coincides at some model shapes.
        fc1_sf_elements = _align_up(2 * intermediate, 128) * _align_up(
            hidden // sf_vec_size, 4
        )
        fc2_sf_elements = _align_up(hidden, 128) * _align_up(
            intermediate // sf_vec_size, 4
        )
        specs = (
            (
                "mega_fc1_weight",
                fc1_elements * weight_bits // 8,
                plane_alignment,
            ),
            (
                "mega_fc1_weight_sf",
                fc1_sf_elements * sf_bits // 8,
                plane_alignment,
            ),
            (
                "mega_fc2_weight",
                fc2_elements * weight_bits // 8,
                plane_alignment,
            ),
            (
                "mega_fc2_weight_sf",
                fc2_sf_elements * sf_bits // 8,
                plane_alignment,
            ),
            ("fc31_alpha", 4, 16),
            ("fc2_alpha", 4, 16),
            ("fc1_norm_const", 4, 16),
        )

        planes: list[Plane] = []
        cursor = 0
        for name, nbytes, alignment in specs:
            cursor = _align_up(cursor, alignment)
            planes.append(Plane(name, cursor, nbytes, alignment))
            cursor += nbytes
        return cls(
            hidden=hidden,
            intermediate=intermediate,
            planes=tuple(planes),
            total_bytes=sum(plane.nbytes for plane in planes),
            stride_bytes=_align_up(cursor, plane_alignment),
            alignment=plane_alignment,
        )

    @property
    def plane_count(self) -> int:
        return len(self.planes)


__all__ = ["BundleLayout", "Plane"]
