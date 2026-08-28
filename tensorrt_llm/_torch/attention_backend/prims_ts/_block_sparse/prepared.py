# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared layout of the route records produced before sparse attention."""

from dataclasses import dataclass

from .._utils import round_up
from .common import _SIGNED_INT32_MAX, _block_sparse_kv_atom_size

_SECTION_ALIGNMENT_WORDS = 4
_PREPARED_ROUTE_IS_FULL_FLAG = 1 << 0
_SUPPORTED_KV_ROUTE_SIZES = (128, 256)
_SUPPORTED_PAGED_KV_PAGE_SIZES = (16, 32, 64, 128)


def _validate_int(value: object, name: str, *, allow_zero: bool) -> int:
    """Validate a host layout extent while rejecting ``bool`` explicitly."""

    requirement = "non-negative" if allow_zero else "positive"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a {requirement} Python integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {requirement}")
    return value


def _validate_i32_address(value: int, name: str) -> int:
    """Reject an allocation extent or section address device Int32 cannot hold."""

    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


def _build_route_workspace_geometry(
    *,
    route_metadata_words: int,
    route_metadata_capacity: int,
    num_rows: int,
) -> tuple[int, int, int]:
    """Return the shared aligned route stride, base, and workspace extent."""

    route_metadata_stride_words = round_up(
        route_metadata_words,
        _SECTION_ALIGNMENT_WORDS,
    )
    _validate_i32_address(
        num_rows + 1,
        "row_route_offsets_length",
    )
    route_metadata_base_word_offset = _validate_i32_address(
        round_up(num_rows, _SECTION_ALIGNMENT_WORDS),
        "route_metadata_base_word_offset",
    )
    workspace_size_words = _validate_i32_address(
        route_metadata_base_word_offset
        + route_metadata_capacity * route_metadata_stride_words,
        "workspace_size_words",
    )
    return (
        route_metadata_stride_words,
        route_metadata_base_word_offset,
        workspace_size_words,
    )


@dataclass(frozen=True)
class _BlockSparseRouteLayout:
    """Immutable word layout for compact, prepared sparse routes.

    A separate immutable tensor holds the plan-generated ``num_rows + 1``
    CSR-style row offsets. Each offset is a route ordinal into the metadata
    section, not a workspace word offset. This layout describes only mutable
    run scratch: ``num_rows`` route counts from word zero, followed by
    ``route_metadata_capacity`` fixed-stride route metadata on a four-word
    (16-byte) boundary. Every plan assigns uniform-capacity row slices so
    caller-owned BSR boundaries may change between runs.

    Each route's metadata stores logical KV-token atom origins, optional
    physical page IDs, one atom-valid-mask word, one route-flags word, and
    optional token-valid words. ``page_size is None`` selects the contiguous
    record; otherwise the paged record adds one page-ID word per logical
    origin. Logical origins remain independent of the K/V storage locator used
    by the attention load path. An invalid logical origin is encoded as
    ``-1``. Bit ``i`` of the atom-valid mask corresponds to logical origin
    ``i``. ``_PREPARED_ROUTE_IS_FULL_FLAG`` (bit 0) states that the route is
    both structurally full and, when token bits are present, token-full.
    """

    # Store semantic inputs plus the three validated allocation values. All
    # remaining offsets and extents are derived from this route geometry.
    # Number of logical KV tokens represented by one prepared route record.
    kv_route_size: int
    # Smallest independently addressable KV fragment represented by metadata.
    atom_size: int
    # Whether each route's metadata carries per-token validity words.
    has_token_bits: bool
    # Flattened (batch, KV head, Q-block row) count.
    num_rows: int
    # route_workspace = [row route counts | padding | route metadata].
    # Aligned Int32-word distance between adjacent routes' metadata.
    route_metadata_stride_words: int
    # Base offset from route_workspace[0] to the first route's metadata.
    route_metadata_base_word_offset: int
    # Total mutable workspace extent in Int32 words.
    workspace_size_words: int
    # Paged-KV token capacity, or None for contiguous K/V storage.
    page_size: int | None = None

    @staticmethod
    def create(
        *,
        kv_route_size: int,
        kv_block_size: int,
        has_token_bits: bool,
        route_metadata_capacity: int,
        num_rows: int,
        page_size: int | None = None,
    ) -> "_BlockSparseRouteLayout":
        """Build aligned workspace-section and per-route metadata geometry."""

        kv_route_size = _validate_int(
            kv_route_size,
            "kv_route_size",
            allow_zero=False,
        )
        if kv_route_size not in _SUPPORTED_KV_ROUTE_SIZES:
            raise ValueError("kv_route_size must be 128 or 256")
        atom_size = _block_sparse_kv_atom_size(kv_block_size)
        if page_size is not None:
            page_size = _validate_int(page_size, "page_size", allow_zero=False)
            if atom_size > page_size:
                raise ValueError("atom_size must not exceed page_size")
            if page_size % atom_size != 0:
                raise ValueError("page_size must be divisible by atom_size")
            if page_size not in _SUPPORTED_PAGED_KV_PAGE_SIZES:
                raise ValueError("page_size must be 16, 32, 64, or 128")
        logical_origins_per_route = kv_route_size // atom_size
        if not isinstance(has_token_bits, bool):
            raise TypeError("has_token_bits must be a bool")
        route_metadata_capacity = _validate_int(
            route_metadata_capacity,
            "route_metadata_capacity",
            allow_zero=True,
        )
        num_rows = _validate_int(num_rows, "num_rows", allow_zero=False)

        token_words_per_route = kv_route_size // 32
        (
            route_metadata_stride_words,
            route_metadata_base_word_offset,
            workspace_size_words,
        ) = _build_route_workspace_geometry(
            route_metadata_words=(
                logical_origins_per_route
                + (logical_origins_per_route if page_size is not None else 0)
                + 2
                + (token_words_per_route if has_token_bits else 0)
            ),
            route_metadata_capacity=route_metadata_capacity,
            num_rows=num_rows,
        )

        return _BlockSparseRouteLayout(
            kv_route_size=kv_route_size,
            atom_size=atom_size,
            has_token_bits=has_token_bits,
            num_rows=num_rows,
            route_metadata_stride_words=route_metadata_stride_words,
            route_metadata_base_word_offset=route_metadata_base_word_offset,
            workspace_size_words=workspace_size_words,
            page_size=page_size,
        )

    @property
    def is_paged(self) -> bool:
        """Whether each route carries physical-page-ID locator words."""

        return self.page_size is not None

    @property
    def paged_page_size(self) -> int:
        """Return the paged token capacity, failing on contiguous layouts."""

        if self.page_size is None:
            raise RuntimeError("paged page size requested from contiguous layout")
        return self.page_size

    @property
    def logical_origins_per_route(self) -> int:
        """Number of logical KV atom origins stored in one route."""

        return self.kv_route_size // self.atom_size

    @property
    def token_words_per_route(self) -> int:
        """Number of 32-token validity words covered by one route."""

        return self.kv_route_size // 32

    @property
    def physical_page_ids_word_offset(self) -> int:
        """First physical-page-ID word in a paged record's locator section."""

        if not self.is_paged:
            raise RuntimeError("page-ID offset requested from contiguous layout")
        return self.logical_origins_per_route

    @property
    def atom_valid_mask_word_offset(self) -> int:
        """Word holding one validity bit for each route KV atom."""

        locator_words = self.logical_origins_per_route if self.is_paged else 0
        return self.logical_origins_per_route + locator_words

    @property
    def route_flags_word_offset(self) -> int:
        """Word holding route-wide flags such as ``ROUTE_IS_FULL``."""

        return self.atom_valid_mask_word_offset + 1

    @property
    def token_words_word_offset(self) -> int | None:
        """First per-token validity word, or ``None`` for unmasked routes."""

        return self.route_flags_word_offset + 1 if self.has_token_bits else None

    @property
    def route_metadata_capacity(self) -> int:
        """Number of routes whose metadata fits in the mutable workspace."""

        return (
            self.workspace_size_words - self.route_metadata_base_word_offset
        ) // self.route_metadata_stride_words


__all__ = [
    "_PREPARED_ROUTE_IS_FULL_FLAG",
    "_BlockSparseRouteLayout",
]
