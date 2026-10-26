# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Python contract for the tile-wise NVFP4 V1 split-plane layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from typing import Any

UINT64_MAX = (1 << 64) - 1
ALIGNMENT = 128
HIDDEN_ELEMENTS = 7168
BF16_TOKEN_BYTES = 14336
# Scale-free saturating E4M3 wire: one byte per hidden element, no SF plane.
FP8_TOKEN_BYTES = 7168
Q_PAYLOAD_BYTES_PER_TOKEN = 3584
SF_PAYLOAD_BYTES_PER_TOKEN = 896
# Packed-interleaved [Q|SF] record geometry (kTileNvfp4Compact). The SF start
# sits at Q_ALIGNED_BYTES inside each record; the record stride is Q + SF.
Q_ALIGNED_BYTES = Q_PAYLOAD_BYTES_PER_TOKEN
PACKED_RECORD_BYTES = Q_PAYLOAD_BYTES_PER_TOKEN + SF_PAYLOAD_BYTES_PER_TOKEN
# Combine-side runtime EP ceiling (kMaxRanksTile in the unified C++ target).
# The dispatch-side shared kMaxRanks=64 is NOT lifted here; the selector
# cross-checks it (EP in (64, 128] is combine-ABI-ready only).
MAX_EP_SIZE = 128


class MoeA2ACombineTileFormat(str, Enum):
    """Independent Python selector; legacy FP8 remains a separate bool."""

    NONE = "none"
    TILE_NVFP4_V1 = "tile_nvfp4_v1"


class QuantFormat(IntEnum):
    # BF16 passthrough (no quantization on the wire).
    NONE = 0
    # Scale-free saturating E4M3 (NOT MXFP8 -- honest naming, decision point 2a).
    FP8_E4M3 = 1
    TILE_NVFP4 = 2


class PayloadLayout(IntEnum):
    TILE_NVFP4_SPLIT_PLANE = 1
    # Packed-interleaved [Q|SF]: one contiguous 4480-B record per token.
    TILE_NVFP4_COMPACT = 2


class ScaleLayout(IntEnum):
    # No separate SF plane (fp8_e4m3 / bf16 wires).
    NONE = 0
    TILE64_BLOCK16_E4M3_PLUS_F32_LE = 1


# ---------------------------------------------------------------------------
# Unified tile-TMA wire recipes.
#
# Canonical strings and their fixed torch-schema ids: the tile-TMA combine ops
# take `int recipe` with 0=nvfp4, 1=fp8_e4m3, 2=bf16,
# 3=nvfp4_packed.  The per-recipe wire descriptors below hold the wire
# constants BY VALUE and mirror the constexpr RecipeDesc table of the
# combine ops field for field.
# ---------------------------------------------------------------------------

TILE_TMA_RECIPES = ("nvfp4", "fp8_e4m3", "bf16", "nvfp4_packed")
# Legacy names accepted at every user-facing surface (env, CLI, ctor).
TILE_TMA_RECIPE_ALIASES = {"tile_nvfp4": "nvfp4", "fp8": "fp8_e4m3"}
# USER-FROZEN torch-schema `int recipe` values (ABI; never reorder).
TILE_TMA_RECIPE_IDS = {"nvfp4": 0, "fp8_e4m3": 1, "bf16": 2, "nvfp4_packed": 3}


def normalize_tile_tma_recipe(recipe: Any) -> str:
    """Normalize a user-facing recipe spelling to its canonical name.

    Canonical set: ``nvfp4 | fp8_e4m3 | bf16 | nvfp4_packed``.  The legacy
    aliases ``tile_nvfp4`` (-> nvfp4) and ``fp8`` (-> fp8_e4m3) are accepted;
    anything else is a hard error (no silent fallback).
    """

    if not isinstance(recipe, str):
        raise TypeError(f"tile-TMA recipe must be a string, got {type(recipe).__name__}")
    cleaned = recipe.strip().lower()
    canonical = TILE_TMA_RECIPE_ALIASES.get(cleaned, cleaned)
    if canonical not in TILE_TMA_RECIPE_IDS:
        raise ValueError(
            f"unknown tile-TMA recipe {recipe!r}; canonical recipes are "
            f"{TILE_TMA_RECIPES} (accepted aliases: "
            f"{sorted(TILE_TMA_RECIPE_ALIASES)})"
        )
    return canonical


@dataclass(frozen=True)
class TileTmaWireDesc:
    """Per-recipe wire descriptor (python mirror of the C++ RecipeDesc).

    ``sf_plane_bytes_per_slot`` is the workspace-bound multiplier for the
    recipe-keyed second plane after the graph-static BF16 SOURCE plane:
    the nvfp4 SF plane (896 B/slot), the packed [Q|SF] plane (4480 B/slot),
    or nothing (fp8_e4m3/bf16 -- 0, so the bound check degenerates to
    ``sf_plane_offset <= workspace row``, the zero-SF branch).
    """

    quant_format: int
    payload_layout: int
    scale_layout: int
    reserved0: int  # 0, or Q_ALIGNED_BYTES(3584) for the packed recipe
    q_payload_bytes_per_token: int
    sf_payload_bytes_per_token: int
    sf_token_stride_bytes: int
    sf_plane_bytes_per_slot: int


_RECIPE_WIRE = {
    # [0] nvfp4: tile-wise NVFP4 split-plane (Q 3584 B + SF 896 B per token).
    "nvfp4": TileTmaWireDesc(
        quant_format=int(QuantFormat.TILE_NVFP4),
        payload_layout=int(PayloadLayout.TILE_NVFP4_SPLIT_PLANE),
        scale_layout=int(ScaleLayout.TILE64_BLOCK16_E4M3_PLUS_F32_LE),
        reserved0=0,
        q_payload_bytes_per_token=Q_PAYLOAD_BYTES_PER_TOKEN,
        sf_payload_bytes_per_token=SF_PAYLOAD_BYTES_PER_TOKEN,
        sf_token_stride_bytes=SF_PAYLOAD_BYTES_PER_TOKEN,
        sf_plane_bytes_per_slot=SF_PAYLOAD_BYTES_PER_TOKEN,
    ),
    # [1] fp8_e4m3: scale-free saturating E4M3 split-plane (no SF plane).
    "fp8_e4m3": TileTmaWireDesc(
        quant_format=int(QuantFormat.FP8_E4M3),
        payload_layout=int(PayloadLayout.TILE_NVFP4_SPLIT_PLANE),
        scale_layout=int(ScaleLayout.NONE),
        reserved0=0,
        q_payload_bytes_per_token=FP8_TOKEN_BYTES,
        sf_payload_bytes_per_token=0,
        sf_token_stride_bytes=0,
        sf_plane_bytes_per_slot=0,
    ),
    # [2] bf16: passthrough split-plane (no quantization, no SF plane).
    "bf16": TileTmaWireDesc(
        quant_format=int(QuantFormat.NONE),
        payload_layout=int(PayloadLayout.TILE_NVFP4_SPLIT_PLANE),
        scale_layout=int(ScaleLayout.NONE),
        reserved0=0,
        q_payload_bytes_per_token=BF16_TOKEN_BYTES,
        sf_payload_bytes_per_token=0,
        sf_token_stride_bytes=0,
        sf_plane_bytes_per_slot=0,
    ),
    # [3] nvfp4_packed: compact [Q|SF] plane; reserved0 carries q_aligned.
    "nvfp4_packed": TileTmaWireDesc(
        quant_format=int(QuantFormat.TILE_NVFP4),
        payload_layout=int(PayloadLayout.TILE_NVFP4_COMPACT),
        scale_layout=int(ScaleLayout.TILE64_BLOCK16_E4M3_PLUS_F32_LE),
        reserved0=Q_ALIGNED_BYTES,
        q_payload_bytes_per_token=Q_PAYLOAD_BYTES_PER_TOKEN,
        sf_payload_bytes_per_token=PACKED_RECORD_BYTES,
        sf_token_stride_bytes=PACKED_RECORD_BYTES,
        sf_plane_bytes_per_slot=PACKED_RECORD_BYTES,
    ),
}

assert tuple(_RECIPE_WIRE) == TILE_TMA_RECIPES


def tile_tma_wire_desc(recipe: str) -> TileTmaWireDesc:
    """Resolve the wire descriptor for a (possibly aliased) recipe string."""

    return _RECIPE_WIRE[normalize_tile_tma_recipe(recipe)]


def _checked_u64(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0 or value > UINT64_MAX:
        raise ValueError(f"{name} is outside uint64")
    return value


def _checked_add(left: int, right: int, name: str) -> int:
    left = _checked_u64(left, f"{name}.left")
    right = _checked_u64(right, f"{name}.right")
    if left > UINT64_MAX - right:
        raise OverflowError(f"{name} overflows uint64")
    return left + right


def _checked_mul(left: int, right: int, name: str) -> int:
    left = _checked_u64(left, f"{name}.left")
    right = _checked_u64(right, f"{name}.right")
    if left and right > UINT64_MAX // left:
        raise OverflowError(f"{name} overflows uint64")
    return left * right


def align128(value: int) -> int:
    return _checked_add(value, ALIGNMENT - 1, "align128") & ~(ALIGNMENT - 1)


@dataclass(frozen=True)
class TileNvfp4V1Layout:
    abi_version: int
    quant_format: int
    format_version: int
    payload_layout: int
    scale_layout: int
    hidden_elements: int
    tile_elements: int
    scale_block_elements: int
    ep_size: int
    runtime_max_tokens_per_rank: int
    graph_static_max_tokens_per_rank: int
    reserved0: int
    q_plane_offset_bytes: int
    sf_plane_offset_bytes: int
    src_token_stride_bytes: int
    q_token_stride_bytes: int
    sf_token_stride_bytes: int
    q_payload_bytes_per_token: int
    sf_payload_bytes_per_token: int
    workspace_size_per_rank_bytes: int

    @classmethod
    def create(
        cls,
        *,
        ep_size: int,
        runtime_max_tokens_per_rank: int,
        graph_static_max_tokens_per_rank: int,
        q_plane_offset_bytes: int,
        workspace_size_per_rank_bytes: int,
        recipe: str = "nvfp4",
        payload_layout: int | None = None,
    ) -> "TileNvfp4V1Layout":
        """Build the recipe-keyed graph-static layout descriptor.

        ``recipe`` selects the wire descriptor (canonical or aliased string;
        default preserves the historical NVFP4 split-plane behavior exactly).
        ``payload_layout`` is an optional cross-check: recipes bind their
        payload layout by value (split-plane vs packed), so a caller that
        carries both must agree with the recipe or fail closed.
        """

        wire = tile_tma_wire_desc(recipe)
        if payload_layout is not None and int(payload_layout) != wire.payload_layout:
            raise ValueError(
                f"payload_layout {int(payload_layout)} contradicts recipe "
                f"{recipe!r} (canonical payload_layout {wire.payload_layout})"
            )
        if ep_size <= 0 or ep_size > MAX_EP_SIZE:
            raise ValueError(f"ep_size must be in (0, {MAX_EP_SIZE}]")
        if runtime_max_tokens_per_rank <= 0:
            raise ValueError("runtime max R must be positive")
        if graph_static_max_tokens_per_rank < runtime_max_tokens_per_rank:
            raise ValueError("graph-static max M must satisfy M >= R")
        q_plane_offset_bytes = _checked_u64(q_plane_offset_bytes, "q_plane_offset_bytes")
        workspace_size_per_rank_bytes = _checked_u64(
            workspace_size_per_rank_bytes, "workspace_size_per_rank_bytes"
        )
        if q_plane_offset_bytes % ALIGNMENT:
            raise ValueError("q_plane_offset_bytes must be 128-byte aligned")
        static_slots = _checked_mul(ep_size, graph_static_max_tokens_per_rank, "E*M")
        q_end = _checked_add(
            q_plane_offset_bytes,
            _checked_mul(static_slots, BF16_TOKEN_BYTES, "Q reservation"),
            "Q end",
        )
        sf_offset = align128(q_end)
        sf_end = _checked_add(
            sf_offset,
            _checked_mul(static_slots, wire.sf_plane_bytes_per_slot, "second-plane reservation"),
            "SF end",
        )
        if sf_end > workspace_size_per_rank_bytes:
            raise ValueError(
                f"SF end {sf_end} exceeds workspace row {workspace_size_per_rank_bytes}"
            )
        return cls(
            abi_version=1,
            quant_format=wire.quant_format,
            format_version=1,
            payload_layout=wire.payload_layout,
            scale_layout=wire.scale_layout,
            hidden_elements=HIDDEN_ELEMENTS,
            tile_elements=64,
            scale_block_elements=16,
            ep_size=ep_size,
            runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            graph_static_max_tokens_per_rank=graph_static_max_tokens_per_rank,
            reserved0=wire.reserved0,
            q_plane_offset_bytes=q_plane_offset_bytes,
            sf_plane_offset_bytes=sf_offset,
            src_token_stride_bytes=BF16_TOKEN_BYTES,
            q_token_stride_bytes=BF16_TOKEN_BYTES,
            sf_token_stride_bytes=wire.sf_token_stride_bytes,
            q_payload_bytes_per_token=wire.q_payload_bytes_per_token,
            sf_payload_bytes_per_token=wire.sf_payload_bytes_per_token,
            workspace_size_per_rank_bytes=workspace_size_per_rank_bytes,
        )

    @property
    def sf_end_bytes(self) -> int:
        slots = self.ep_size * self.graph_static_max_tokens_per_rank
        return self.sf_plane_offset_bytes + slots * self.sf_payload_bytes_per_token

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def graph_key(self, *, kernel_build_id: str) -> tuple[Any, ...]:
        if not kernel_build_id:
            raise ValueError("kernel_build_id is required for a graph-safe key")
        return (
            "moe_a2a_tile_nvfp4_v2",
            kernel_build_id,
            *self.as_dict().values(),
        )


# ---------------------------------------------------------------------------
# Push-combine identity fields (ADDITIVE).
#
# These mirror the frozen constants of the push-combine kernel/op family
# (moeAlltoAllTileNvfp4V2PushFlags.h + ...PushK1K4.h + ...PushOp.cpp).  They are
# NOT part of the physical-workspace allocation key: the push landing region C is
# a pure STORAGE ALIAS onto the packed SF plane
# (cft_region_c_alias_v1), so the bytes are sized by the packed recipe and the
# base NVLinkOneSided allocation key alone.  Instead these belong in the
# CUDA-graph key and the workspace-reuse guard key so a captured graph or a
# reused workspace record can never silently mix a different flag ABI, control
# slab geometry, launch geometry, poll policy, staging stride, or supported
# (ep_size, records_per_token) tuple.
# ---------------------------------------------------------------------------

# raw-u64 flag ABI ids (moeAlltoAllTileNvfp4V2PushFlags.h kPushFlag*).
PUSH_FLAG_ABI_VERSION = "raw_u64_epoch_v1"
PUSH_FLAG_LAYOUT_ID = "compact_row_v2"
# ready[kPushReadyCapacity] slots per consumer row and the 272-B control slab.
PUSH_READY_CAPACITY = 32
PUSH_CONTROL_SLAB_BYTES = 272
# The first (lowest) generation any opened push epoch may carry; 0 is the
# reserved INVALID_GENERATION sentinel and is never a live step.
PUSH_FIRST_GENERATION = 1
# K1Geometry baseline row (threads, cta_cap, target_sm_count, s_alloc, s_active).
# cta_cap and target_sm_count are resolved at launch (0 here), so the graph-safe
# identity pins only the frozen fields (threads must be 224 or HOST_REJECT).
PUSH_K1_BASELINE_GEOMETRY = (224, 0, 0, 2, 2)
PUSH_K1_BASELINE_THREADS = 224
# K3 consumer single-gate poll policy id (policy_id == 0 baseline).
PUSH_K3_POLL_POLICY_ID = 0
# K4 selected-shared-stride option: default packed stride (4480) or the padded
# stride (4496).  Both decode the same 4480 payload bytes.
PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES = 4480
PUSH_K4_SELECTED_SHARED_STRIDE_AB_BYTES = 4496
# Supported (ep_size, records_per_token) tuples.
PUSH_SUPPORTED_PACKED_TUPLE_MANIFEST = (
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 8),
    (32, 8),
)
# The one legal push recipe: packed-interleaved [Q|SF] (recipe id 3).
PUSH_RECIPE = "nvfp4_packed"


def push_identity_fields(
    *,
    k4_selected_shared_stride_bytes: int = PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES,
    k3_poll_policy_id: int = PUSH_K3_POLL_POLICY_ID,
) -> tuple:
    """Return the frozen push-combine identity tuple (graph/workspace keys only).

    The result is a pure hashable tuple of the push flag ABI ids, the control
    slab geometry, the K1 launch geometry, the K3 poll policy, the K4 staging
    stride, and the supported packed (ep_size, records_per_token) manifest.  It
    is deliberately NOT fed into the physical-workspace allocation key (see the
    module note above); callers append it to the CUDA-graph key and the
    workspace-reuse guard key.
    """

    if k4_selected_shared_stride_bytes not in (
        PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES,
        PUSH_K4_SELECTED_SHARED_STRIDE_AB_BYTES,
    ):
        raise ValueError(
            "k4_selected_shared_stride_bytes must be "
            f"{PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES} (default) or "
            f"{PUSH_K4_SELECTED_SHARED_STRIDE_AB_BYTES} (A/B); got "
            f"{k4_selected_shared_stride_bytes}"
        )
    if int(k3_poll_policy_id) != PUSH_K3_POLL_POLICY_ID:
        raise ValueError(
            f"k3_poll_policy_id must be {PUSH_K3_POLL_POLICY_ID} (the only "
            f"published push gate policy); got {k3_poll_policy_id}"
        )
    return (
        "push_combine_identity_v2",
        PUSH_FLAG_ABI_VERSION,
        PUSH_FLAG_LAYOUT_ID,
        int(PUSH_READY_CAPACITY),
        int(PUSH_CONTROL_SLAB_BYTES),
        tuple(int(field) for field in PUSH_K1_BASELINE_GEOMETRY),
        int(k3_poll_policy_id),
        int(k4_selected_shared_stride_bytes),
        tuple((int(ep), int(rpt)) for ep, rpt in PUSH_SUPPORTED_PACKED_TUPLE_MANIFEST),
    )


def push_control_plane_bytes(ep_size: int) -> int:
    """Logical bytes for ``ep_size`` densely stored 272-B control rows.

    The live MNNVL path allocates one 272-B segment per rank and uses the tensor's
    fabric stride for peer hops; this helper is only aggregate sizing arithmetic.
    """

    if ep_size <= 0 or ep_size > PUSH_READY_CAPACITY:
        raise ValueError(
            f"push control plane ep_size must be in (0, {PUSH_READY_CAPACITY}]; got {ep_size}"
        )
    return align128(_checked_mul(ep_size, PUSH_CONTROL_SLAB_BYTES, "control plane"))


def workspace_extra_bytes(
    ep_size: int, graph_static_max_tokens_per_rank: int, recipe: str = "nvfp4"
) -> int:
    """Recipe-keyed second-plane reservation after the BF16 SOURCE plane.

    nvfp4 keeps the historical split SF plane (E*M*896, then 128-B aligned);
    nvfp4_packed reserves the packed [Q|SF] plane (E*M*4480); fp8_e4m3/bf16
    reserve nothing (0).
    """

    wire = tile_tma_wire_desc(recipe)
    return align128(
        _checked_mul(
            _checked_mul(ep_size, graph_static_max_tokens_per_rank, "E*M"),
            wire.sf_plane_bytes_per_slot,
            "second-plane reservation",
        )
    )


def packed_workspace_extra_bytes(ep_size: int, graph_static_max_tokens_per_rank: int) -> int:
    """The packed [Q|SF] plane is E*M*4480 bytes (then 128-B aligned).

    Unlike the split-plane SF plane (896 B/slot), the packed plane holds the ONE
    contiguous 4480-B [Q|SF] record per slot; it is indexed at runtime by
    epRank*R + dstIdx (a compact prefix of the graph-static E*M reservation).
    """

    return align128(
        _checked_mul(
            _checked_mul(ep_size, graph_static_max_tokens_per_rank, "E*M"),
            PACKED_RECORD_BYTES,
            "packed reservation",
        )
    )


def graph_static_q_offset(
    *,
    payload_data_offset_bytes: int,
    ep_size: int,
    graph_static_max_tokens_per_rank: int,
    payload_bytes_per_token: tuple[int, ...],
) -> int:
    """Place V2 Q after graph-static E*M reservations for dispatch payloads."""

    current = _checked_u64(payload_data_offset_bytes, "payload_data_offset_bytes")
    if current % ALIGNMENT:
        raise ValueError("payload_data_offset_bytes must be 128-byte aligned")
    if not payload_bytes_per_token:
        raise ValueError("at least one dispatch payload is required")
    static_slots = _checked_mul(ep_size, graph_static_max_tokens_per_rank, "E*M")
    for index, bytes_per_token in enumerate(payload_bytes_per_token):
        if bytes_per_token <= 0:
            raise ValueError(f"payload {index} bytes_per_token must be positive")
        current = align128(
            _checked_add(
                current,
                _checked_mul(
                    static_slots,
                    bytes_per_token,
                    f"dispatch payload {index} reservation",
                ),
                f"dispatch payload {index} end",
            )
        )
    return current
