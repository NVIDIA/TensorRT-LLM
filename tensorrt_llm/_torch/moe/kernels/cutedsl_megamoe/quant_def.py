# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Canonical quantization kinds and payload formats."""

import dataclasses
import enum
from typing import ClassVar, Dict, Literal, Optional, Tuple, Type

import cutlass


class QuantKind(str, enum.Enum):
    """One admissible block-scaled quantization mode.

    A member is defined by its (weight, activation) element pair -- under swap-AB the weight is
    the MMA A operand and the activation is the B operand. The selected K-throughput mode remains
    explicit because one quantization kind can use multiple instruction K extents.
    """

    nvfp4 = "nvfp4"
    mxfp4 = "mxfp4"
    mxfp8_e4m3 = "mxfp8_e4m3"
    mxfp8_e5m2 = "mxfp8_e5m2"
    mxfp4_mxfp8 = "mxfp4_mxfp8"

    # Enum's __str__/__format__ for mixed-in types changed across Python 3.10/3.11/3.12, and the
    # kernels fold the kind into their compiled-kernel cache key. Pin both to the member value so
    # the key cannot silently become "QuantKind.nvfp4" on an interpreter upgrade.
    __str__ = str.__str__
    __format__ = str.__format__

    @property
    def weight_dtype(self) -> Type[cutlass.Numeric]:
        return _element_pair[self][0]

    @property
    def activation_dtype(self) -> Type[cutlass.Numeric]:
        """Also the dtype the FC1 epilogue must emit: FC2 consumes it as its activation."""
        return _element_pair[self][1]

    @property
    def sf_vec_size(self) -> int:
        return 16 if self is QuantKind.nvfp4 else 32

    @property
    def sf_dtype(self) -> Type[cutlass.Numeric]:
        # Hardware allows a UE8M0 scale at vec 16 too, but nvfp4 is the only vec-16 mode we build.
        return cutlass.Float8E4M3FN if self.sf_vec_size == 16 else cutlass.Float8E8M0FNU

    @property
    def umma_kind(self) -> str:
        """The ``tcgen05.mma.kind::`` qualifier.

        Mirrors the dispatch in ``blackwell_helpers._make_blockscaled_trivial_tiled_mma_impl``:
        only an fp4 pair reaches the fp4-specific kinds, everything else -- including any mixed
        pair -- falls back to mxf8f6f4. Keeping the two in sync matters because we build the tiled
        MMA through that helper but emit the instruction ourselves.
        """
        both_fp4 = self.weight_dtype is cutlass.Float4E2M1FN and self.activation_dtype is cutlass.Float4E2M1FN
        if not both_fp4:
            return "mxf8f6f4"
        return "mxf4nvf4" if self.sf_vec_size == 16 else "mxf4"

    @property
    def umma_scale_vec_suffix(self) -> str:
        """PTX modifier after ``.block_scale``; mxf8f6f4 takes none (its scale vector is 32)."""
        if self.umma_kind == "mxf8f6f4":
            return ""
        return ".block16" if self.sf_vec_size == 16 else ".block32"

    def instruction_k(self, mma_k_mode: Literal["1x", "2x"]) -> int:
        instruction_k_1x = 32 if self.umma_kind == "mxf8f6f4" else 64
        if mma_k_mode == "1x":
            return instruction_k_1x
        if mma_k_mode == "2x":
            return instruction_k_1x * 2
        raise ValueError(f"Invalid MMA K mode {mma_k_mode!r}; expected '1x' or '2x'.")

    @property
    def weight_format_code(self) -> int:
        """Instruction-descriptor ``a_format_`` under swap-AB."""
        return _instruction_descriptor_format_code(self.umma_kind, self.weight_dtype)

    @property
    def activation_format_code(self) -> int:
        """Instruction-descriptor ``b_format_`` under swap-AB."""
        return _instruction_descriptor_format_code(self.umma_kind, self.activation_dtype)

    @property
    def scale_format_code(self) -> int:
        """Instruction-descriptor ``scale_format_``: 0 = UE4M3, 1 = UE8M0."""
        return 0 if self.sf_dtype is cutlass.Float8E4M3FN else 1

    def needs_unpack_tma(self, architecture: str) -> bool:
        """Whether the narrow operand must reach SMEM in 1-byte containers (U4_UNPACK_U8 TMA).

        Blackwell mixed-width MMA consumes the narrow operand through UNPACK TMA. Rubin consumes
        mixed FP4 directly from its native packed SMEM image.
        """
        normalized_architecture = architecture.lower().replace("_", "")
        if normalized_architecture.startswith("sm"):
            normalized_architecture = normalized_architecture[2:]
        if normalized_architecture not in ("100", "103", "107"):
            raise ValueError(f"Unsupported TCGen05 architecture {architecture!r}.")
        return normalized_architecture in ("100", "103") and self.weight_dtype.width != self.activation_dtype.width

    @property
    def uses_global_scale(self) -> bool:
        """Whether the caller supplies per-expert alpha / norm_const.

        Only nvfp4: an e8m0 scale is a pure power of two and already carries the whole rescale.
        """
        return self is QuantKind.nvfp4


_element_pair: Dict[QuantKind, Tuple[Type[cutlass.Numeric], Type[cutlass.Numeric]]] = {
    QuantKind.nvfp4: (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN),
    QuantKind.mxfp4: (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN),
    QuantKind.mxfp8_e4m3: (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN),
    QuantKind.mxfp8_e5m2: (cutlass.Float8E5M2, cutlass.Float8E5M2),
    QuantKind.mxfp4_mxfp8: (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN),
}


# The three ``a_format_`` / ``b_format_`` bits are read against one of two disjoint enums, and the
# MMA kind picks which (CUTLASS ``UMMA::MXF4Format`` vs ``UMMA::MXF8F6F4Format`` in
# cute/arch/mma_sm100_desc.hpp). E2M1 is 1 under the first and 5 under the second. Getting it wrong
# still compiles and still runs -- it just computes garbage -- so these live in exactly one place.
_mxf4_format_code: Dict[Type[cutlass.Numeric], int] = {cutlass.Float4E2M1FN: 1}
_mxf8f6f4_format_code: Dict[Type[cutlass.Numeric], int] = {
    cutlass.Float8E4M3FN: 0,
    cutlass.Float8E5M2: 1,
    cutlass.Float4E2M1FN: 5,
}


def _instruction_descriptor_format_code(umma_kind: str, dtype: Type[cutlass.Numeric]) -> int:
    codes = _mxf8f6f4_format_code if umma_kind == "mxf8f6f4" else _mxf4_format_code
    try:
        return codes[dtype]
    except KeyError as error:
        raise ValueError(f"{dtype} has no instruction-descriptor format code under kind::{umma_kind}.") from error


@dataclasses.dataclass(frozen=True)
class CombineFormat:
    """Data and scale representation of one cross-rank FC2 payload."""

    _act_by_tag: ClassVar[Dict[str, type]] = {"e2m1": cutlass.Float4E2M1FN, "e4m3": cutlass.Float8E4M3FN}
    _scale_by_tag: ClassVar[Dict[str, type]] = {"bf16": cutlass.BFloat16, "e8m0": cutlass.Float8E8M0FNU}
    _rejection_reason: ClassVar[Dict[str, str]] = {
        "32e5m2xe8m0": (
            "e5m2 costs the same 8.25 bits per element as e4m3 and trades a mantissa bit (6 dB of "
            "SNR) for exponent range the e8m0 block scale already provides. Use 32e4m3xe8m0."
        )
    }

    act_dtype: type
    scale_dtype: Optional[type]
    scale_block: Optional[int]

    def __post_init__(self) -> None:
        allowed_act = {cutlass.BFloat16, *self._act_by_tag.values()}
        allowed_scale = {None, *self._scale_by_tag.values()}
        if self.act_dtype not in allowed_act:
            raise ValueError(f"Unsupported combine data dtype {self.act_dtype}.")
        if self.scale_dtype not in allowed_scale:
            raise ValueError(f"Unsupported combine scale dtype {self.scale_dtype}.")
        if self.scale_dtype is None:
            if self.act_dtype is not cutlass.BFloat16 or self.scale_block is not None:
                raise ValueError("The unquantized format must be bf16 without a scale block.")
            return
        if self.act_dtype is cutlass.BFloat16:
            raise ValueError("A quantized format cannot use bf16 data.")
        if self.scale_dtype is cutlass.BFloat16 and self.scale_block != 16:
            raise ValueError("A bf16 amax scale requires a 16-element block.")
        if self.scale_dtype is cutlass.Float8E8M0FNU and self.scale_block != 32:
            raise ValueError("An e8m0 scale requires a 32-element block.")

    @property
    def is_quantized(self) -> bool:
        return self.scale_dtype is not None

    @property
    def name(self) -> str:
        if not self.is_quantized:
            return "bf16"
        act_tag = next(tag for tag, dtype in self._act_by_tag.items() if dtype is self.act_dtype)
        scale_tag = next(tag for tag, dtype in self._scale_by_tag.items() if dtype is self.scale_dtype)
        return f"{self.scale_block}{act_tag}x{scale_tag}"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, text: str) -> "CombineFormat":
        specs = {
            "bf16": (cutlass.BFloat16, None, None),
            "16e2m1xbf16": (cutlass.Float4E2M1FN, cutlass.BFloat16, 16),
            "32e4m3xe8m0": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
        }
        token = text.strip().lower()
        if token in cls._rejection_reason:
            raise ValueError(f"Combine format {token!r} is deliberately unsupported: {cls._rejection_reason[token]}")
        if token not in specs:
            raise ValueError(f"Invalid combine format {text!r}; expected one of {tuple(specs)}.")
        act_dtype, scale_dtype, scale_block = specs[token]
        return cls(act_dtype, scale_dtype, scale_block)


# Every Blackwell 1x PTX hardware encoding is restated independently of the derivations above, so
# that a typo in a property fails at import rather than at the first wrong numerical result.
# Ordered as (umma_kind, scale_vec_suffix, instruction_k_1x, a_format_, b_format_, scale_format_).
_pinned_hardware_encoding: Dict[QuantKind, Tuple[str, str, int, int, int, int]] = {
    QuantKind.nvfp4: ("mxf4nvf4", ".block16", 64, 1, 1, 0),
    QuantKind.mxfp4: ("mxf4", ".block32", 64, 1, 1, 1),
    QuantKind.mxfp8_e4m3: ("mxf8f6f4", "", 32, 0, 0, 1),
    QuantKind.mxfp8_e5m2: ("mxf8f6f4", "", 32, 1, 1, 1),
    QuantKind.mxfp4_mxfp8: ("mxf8f6f4", "", 32, 5, 0, 1),
}


def _verify_pinned_hardware_encoding() -> None:
    unpinned = sorted(kind.name for kind in QuantKind if kind not in _pinned_hardware_encoding)
    if unpinned:
        raise AssertionError(f"QuantKind members without a pinned hardware encoding: {unpinned}.")
    for kind, expected in _pinned_hardware_encoding.items():
        derived = (
            kind.umma_kind,
            kind.umma_scale_vec_suffix,
            kind.instruction_k("1x"),
            kind.weight_format_code,
            kind.activation_format_code,
            kind.scale_format_code,
        )
        if derived != expected:
            raise AssertionError(f"QuantKind.{kind.name} derives {derived}, pinned encoding is {expected}.")
        expected_instruction_k_2x = expected[2] * 2
        instruction_k_2x = kind.instruction_k("2x")
        if instruction_k_2x != expected_instruction_k_2x:
            raise AssertionError(
                f"QuantKind.{kind.name} derives 2x instruction K {instruction_k_2x}, expected {expected_instruction_k_2x}."
            )


_verify_pinned_hardware_encoding()


__all__ = ["CombineFormat", "QuantKind"]
