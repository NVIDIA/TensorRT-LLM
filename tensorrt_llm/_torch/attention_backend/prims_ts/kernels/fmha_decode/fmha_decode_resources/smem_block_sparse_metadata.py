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

"""Shared-memory metadata resources for prepared K/V and Softmax work.

Each KV instruction owns two independent resources. During HEAD, the load task
loads a prepared route, stores it for the matching K/V pair, issues K, then
stages a copy for Softmax. During LOOP, V consumes the previous K/V metadata
before the load task overwrites it with the next route and issues K. Softmax
waits for its staged copy, moves the seven-slot task payload to registers, and
releases the stage before masking.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...._block_sparse.prepared import (
    _PREPARED_ROUTE_IS_FULL_FLAG,
    _PREPARED_ROUTE_IS_PROXY_FLAG,
    _BlockSparseRouteLayout,
)
from ...placeholder_helpers import _placeholder_smem_array
from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    _TASK_CACHE_SEQ_LEN_KV,
    _TASK_CACHE_WARP_IDX,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
    _keeps_col_base,
    _sparse_task_cache_route_begin,
    _sparse_task_cache_route_count,
    _warp_broadcast_i32,
)


# Keeps staging uses the low four bits for structural KV64 validity. Bit 4
# carries the conservative prepared summary that token masking can be skipped;
# structural, tail, and causal masking remain independent.
_SOFTMAX_TOKEN_MASK_IS_FULL_FLAG = 1 << 4
# Keeps reserves bit 5 for the prepared route kind. The low four structural
# validity bits and bit 4 keep their existing meaning.
_SOFTMAX_ROUTE_IS_PROXY_FLAG = 1 << 5

# SWAP origins are at least eight-token aligned, so their low two bits are free
# while the route is in Softmax's private staging payload. Reusing them avoids
# adding a word to every pipeline stage for prepared FULL/PROXY route flags.
_SWAPS_PACKED_ROUTE_FLAGS_CLEAR_MASK = ~(
    _PREPARED_ROUTE_IS_FULL_FLAG | _PREPARED_ROUTE_IS_PROXY_FLAG
)


@cute.jit
def _paged_sparse_kv_load_coordinate(
    logical_origin: Int32,
    physical_page_id: Int32,
    atom_is_valid: cutlass.Boolean,
    page_size: Constexpr[int],
) -> tuple[Int32, Int32]:
    """Return the in-page token and physical-page TMA coordinates.

    Invalid atoms still issue their normal TMA transaction so the pipeline
    barrier observes a fixed transaction count.  Mapping them to the first
    token just beyond page zero makes the token coordinate OOB while keeping
    the page coordinate itself valid.
    """

    token_in_page = Int32(page_size)
    page_id = Int32(0)
    if atom_is_valid:
        token_in_page = logical_origin % Int32(page_size)
        page_id = physical_page_id
    return token_in_page, page_id


def _swaps_forwards_packed_route_full(cfg: FmhaDecodeConfig) -> bool:
    """Whether SWAP forwards prepare's route-full bit in a staged origin.

    Q8/B8 would otherwise need four origin checks, so it forwards the prepared
    summary. Larger Q tiles keep straight-line per-score predicates: forwarding
    the summary lengthens their hot path more than the skipped checks save.
    """

    return (
        cfg.tile_size_q == 8
        and cfg.kv_block_size == 8
        and not cfg.uses_prepared_score_keep_words
        and not cfg.uses_uniform_causal_mask
        and not cfg.uses_per_row_causal_mask
    )


def _kv_retained_route_words(
    route_layout: _BlockSparseRouteLayout,
    *,
    retain_proxy_kind: bool = False,
) -> int:
    """Return the aligned SMEM words retained from K issue through V.

    Contiguous routes retain their load-origin payload. Paged routes retain
    parallel logical-origin and physical-page-ID arrays so every atom has an
    independent storage locator. A two-origin contiguous route additionally
    keeps its atom-valid mask. Proxy-capable exact/proxy routes reserve the
    final aligned word for an explicit source kind.
    """

    payload_words = route_layout.logical_origins_per_route
    if route_layout.is_paged:
        payload_words *= 2
    elif route_layout.logical_origins_per_route == 2:
        payload_words += 1
    if retain_proxy_kind:
        payload_words += 1
    return ((payload_words + 3) // 4) * 4


@dataclass(frozen=True)
class _BlockSparseSoftmaxStagingLayout:
    """Layout of the staged cross-warp Softmax metadata payload.

    Keeps retains all route origins, a flags word, alignment padding, and the
    optional K32 token words. KV256 consumers then select the four words owned
    by their spatial half. SWAP stores execution-ordered origins followed by
    optional logical K32 token words, one for each consumer warp. Selected
    prepared route flags travel in the otherwise-zero low bits of each warp's
    first aligned origin.
    """

    # Logical-origin scalars staged for one complete KV route.
    num_origin_words: int
    # SWAP origins consumed by one Softmax warp; Keeps retains all origins.
    origins_per_warp: int
    route_flags_word_offset: int | None
    token_words_word_offset: int | None
    stage_stride_words: int
    total_words: int

    @property
    def size_bytes(self) -> int:
        """Return the 16-byte-aligned staged allocation size."""

        return self.total_words * 4

    @staticmethod
    def create(
        *,
        use_keeps_mma_ab: bool,
        route_layout: _BlockSparseRouteLayout,
        num_stages: int,
    ) -> "_BlockSparseSoftmaxStagingLayout":
        """Build a stage-count-dependent layout for Softmax metadata."""

        kv_route_size = route_layout.kv_route_size
        atom_size = route_layout.atom_size
        has_token_bits = route_layout.has_token_bits
        if use_keeps_mma_ab:
            num_origin_words = kv_route_size // atom_size
            assert num_origin_words <= 4, (
                "Keeps softmax staging supports at most four route origins"
            )
            origins_per_warp = num_origin_words
            route_flags_word_offset = num_origin_words
            aligned_payload_words = ((route_flags_word_offset + 1 + 3) // 4) * 4
            token_words_word_offset = aligned_payload_words if has_token_bits else None
            token_words = kv_route_size // 32 if has_token_bits else 0
            stage_stride_words = ((aligned_payload_words + token_words + 3) // 4) * 4
        else:
            softmax_atom_size = min(atom_size, 32)
            num_origin_words = kv_route_size // softmax_atom_size
            origins_per_warp = 32 // softmax_atom_size
            route_flags_word_offset = None
            token_words_word_offset = num_origin_words if has_token_bits else None
            stage_stride_words = num_origin_words + (
                kv_route_size // 32 if has_token_bits else 0
            )
        return _BlockSparseSoftmaxStagingLayout(
            num_origin_words=num_origin_words,
            origins_per_warp=origins_per_warp,
            route_flags_word_offset=route_flags_word_offset,
            token_words_word_offset=token_words_word_offset,
            stage_stride_words=stage_stride_words,
            total_words=num_stages * stage_stride_words,
        )


@dataclass(kw_only=True)
class SmemBlockSparseKvMetadataResource(DecodeGenResourceBase):
    """Pipeline-free route metadata retained from one K issue through V.

    ``route_metadata`` points at the first prepared GMEM record. Resolution
    returns logical/source-domain origins to masking consumers. The private
    SMEM copy keeps contiguous load origins or paged ``(logical origin,
    physical page ID)`` pairs through the matching V issue. Invalid atoms are
    retained as safe storage-specific OOB coordinates. The proxy-capable
    contiguous specialization interprets origins in the selected source domain
    (summary tokens for proxy routes, K/V tokens for exact routes) and
    retains the prepared route kind in a separate aligned word. Exact-only
    specializations keep their original allocation.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "resolved_record_word_slot",
            Int32,
            Int32(0),
            "Lane-owned record word; locator lanes carry route origins.",
        ),
        ("resolved_origin1_slot", Int32, Int32(0), "Second logical origin."),
        (
            "resolved_atom_validity_slot",
            Int32,
            Int32(0),
            "Fine lane validity or coarse two-fragment validity mask.",
        ),
        (
            "route_record_word_offset_slot",
            Int32,
            Int32(-1),
            "Metadata-relative record offset, or -1 for a dummy route.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    route_metadata: cute.Pointer | None = None
    route_layout: Constexpr[_BlockSparseRouteLayout | None] = None
    tma_oob_origin: Int32 = None
    _retained_route_words: Constexpr[int] = 0
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_words: cutlass.Array = None
    resolved_record_word_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    resolved_origin1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    resolved_atom_validity_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    route_record_word_offset_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self) -> None:
        """Derive the retained K/V payload from the prepared route layout."""

        assert self.route_layout is not None
        assert self.route_layout.is_paged == self.cfg.use_paged_kv
        if self.cfg.use_block_sparse_proxy_routes:
            assert not self.route_layout.is_paged
            assert self.route_layout.uses_one_warp_transport
        self._retained_route_words = _kv_retained_route_words(
            self.route_layout,
            retain_proxy_kind=self.cfg.use_block_sparse_proxy_routes,
        )
        super().__post_init__()

    def _init_placeholder_state(self) -> None:
        """Create shape-correct K/V metadata SMEM for task-graph tracing."""

        self._smem_words = _placeholder_smem_array(Int32, self._retained_route_words)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one aligned instruction-local K/V metadata slot."""

        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self._retained_route_words * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """K/V route metadata uses SMEM only."""

        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the K/V metadata allocation on the load warp."""

        if cutlass.const_expr(context is not None and context.smem_base is not None):
            self._smem_words = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=Int32,
                shape=(self._retained_route_words,),
                addrspace=3,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Bind producer-local K/V metadata before the first resolution."""

        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _prepared_route_logical_origin(
        self,
        route_record_word_offset: Int32,
        atom_idx: Int32,
    ) -> Int32:
        """Load one logical KV-token origin from a prepared GMEM record."""

        assert self.route_metadata is not None
        return Int32(self.route_metadata[route_record_word_offset + atom_idx])

    @cute.jit
    def _prepared_route_physical_page_id_if_valid(
        self,
        route_record_word_offset: Int32,
        atom_idx: Int32,
    ) -> Int32:
        """Load a page ID only when the prepared-record offset is valid."""

        assert self.route_metadata is not None
        assert self.route_layout.is_paged
        physical_page_id = Int32(-1)
        if route_record_word_offset >= Int32(0):
            physical_page_id = Int32(
                self.route_metadata[
                    route_record_word_offset
                    + Int32(self.route_layout.physical_page_ids_word_offset)
                    + atom_idx
                ]
            )
        return physical_page_id

    @consumer_work(
        returns=(
            resolved_record_word_slot,
            resolved_origin1_slot,
            resolved_atom_validity_slot,
            route_record_word_offset_slot,
        )
    )
    @cute.jit
    def resolve_route(
        self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]
    ) -> tuple[Int32, Int32, Int32, Int32]:
        """Load this resource instance's real or dummy prepared KV route."""

        assert self.route_metadata is not None
        task_cache = _decode_gen_task_cache(stage_info)
        row_route_begin = _sparse_task_cache_route_begin(task_cache)
        route_count = _sparse_task_cache_route_count(task_cache)
        # HEAD publishes one route per instruction. LOOP starts after those
        # two publications, hence the one-based loop offset below. Keeping the
        # constexpr branch local lets the task scheduler specialize each work
        # clone together with its HEAD/LOOP section.
        if cutlass.const_expr(section == FmhaStage.Head):
            route_idx = Int32(self.inst_id)
        else:
            route_idx = (stage_info.loop_offset + Int32(1)) * Int32(
                self.cfg.num_insts_kv
            ) + Int32(self.inst_id)
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        route_record_word_offset = Int32(-1)
        if route_idx < route_count:
            route_record_word_offset = (row_route_begin + route_idx) * Int32(
                self.route_layout.route_metadata_stride_words
            )
        route_record_word_offset = cute.arch.make_warp_uniform(route_record_word_offset)

        num_logical_origins = self.route_layout.logical_origins_per_route
        uses_two_fragment_route = num_logical_origins == 2
        # Keep the validity load adjacent to the lane-distributed origins.
        # For the maximal 32-origin layout, lane 31 can retain its origin and
        # load the independent validity scalar before the warp broadcasts it.
        valid_mask_lane = min(num_logical_origins, 31)
        logical_origin = Int32(-1)
        atom_valid_mask = Int32(0)
        route_record_is_valid = route_record_word_offset >= Int32(0)

        if cutlass.const_expr(self.route_layout.uses_one_warp_transport):
            assert self.route_layout.token_words_word_offset is not None
            meaningful_words = (
                self.route_layout.token_words_word_offset
                + self.route_layout.token_words_per_route
            )
            resolved_record_word = Int32(0)
            if lane_idx < Int32(num_logical_origins):
                resolved_record_word = Int32(-1)
            if route_record_is_valid and lane_idx < Int32(meaningful_words):
                resolved_record_word = Int32(
                    self.route_metadata[route_record_word_offset + lane_idx]
                )
            atom_valid_mask = _warp_broadcast_i32(
                resolved_record_word,
                self.route_layout.atom_valid_mask_word_offset,
            )
            if cutlass.const_expr(uses_two_fragment_route):
                return (
                    resolved_record_word,
                    _warp_broadcast_i32(resolved_record_word, 1),
                    atom_valid_mask,
                    route_record_word_offset,
                )
            atom_is_valid = cutlass.Boolean(
                lane_idx < Int32(num_logical_origins)
                and (atom_valid_mask & (Int32(1) << lane_idx)) != Int32(0)
            )
            return (
                resolved_record_word,
                Int32(0),
                Int32(atom_is_valid),
                route_record_word_offset,
            )

        if route_record_is_valid:
            if lane_idx < Int32(num_logical_origins):
                logical_origin = self._prepared_route_logical_origin(
                    route_record_word_offset,
                    lane_idx,
                )
            if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
                if lane_idx == Int32(valid_mask_lane):
                    atom_valid_mask = Int32(
                        self.route_metadata[
                            route_record_word_offset
                            + Int32(self.route_layout.atom_valid_mask_word_offset)
                        ]
                    )

        if cutlass.const_expr(uses_two_fragment_route):
            origin0 = _warp_broadcast_i32(logical_origin, 0)
            origin1 = _warp_broadcast_i32(logical_origin, 1)
            if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
                # The validity word shares the prepared record's cache line
                # with fields consumed shortly afterward by Softmax.
                atom_valid_mask = _warp_broadcast_i32(atom_valid_mask, valid_mask_lane)
            else:
                # Unmasked metadata needs no later fields. Recover validity
                # from the invalid-origin sentinel and avoid the dead load.
                origin_is_valid = cutlass.Boolean(
                    lane_idx < Int32(2) and logical_origin >= Int32(0)
                )
                atom_valid_mask = Int32(cute.arch.vote_ballot_sync(origin_is_valid))
            return origin0, origin1, atom_valid_mask, route_record_word_offset

        # Wider routes stay lane-distributed: each active lane carries only
        # its origin and validity through the existing three-scalar K/V ABI.
        valid = cutlass.Boolean(False)
        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            atom_valid_mask = _warp_broadcast_i32(atom_valid_mask, valid_mask_lane)
            if lane_idx < Int32(num_logical_origins):
                valid = (atom_valid_mask & (Int32(1) << lane_idx)) != Int32(0)
        else:
            if lane_idx < Int32(num_logical_origins):
                valid = logical_origin >= Int32(0)
        return logical_origin, Int32(0), Int32(valid), route_record_word_offset

    @producer_work
    @cute.jit
    def store_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
        resolved_atom_validity: Int32,
        route_record_word_offset: Int32,
    ) -> None:
        """Retain storage coordinates for the matching K/V load pair."""

        del stage_info
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        num_origins = self.route_layout.logical_origins_per_route
        if cutlass.const_expr(self.route_layout.is_paged):
            if lane_idx < Int32(num_origins):
                logical_origin = Int32(resolved_record_word)
                atom_is_valid = resolved_atom_validity != Int32(0)
                if cutlass.const_expr(num_origins == 2):
                    if lane_idx == Int32(0):
                        logical_origin = resolved_record_word
                    else:
                        logical_origin = resolved_origin1
                    atom_is_valid = (
                        resolved_atom_validity & (Int32(1) << lane_idx)
                    ) != Int32(0)
                physical_page_id = Int32(-1)
                if atom_is_valid:
                    physical_page_id = self._prepared_route_physical_page_id_if_valid(
                        route_record_word_offset,
                        lane_idx,
                    )
                else:
                    logical_origin = Int32(-1)
                self._smem_words[lane_idx] = logical_origin
                self._smem_words[
                    Int32(self.route_layout.physical_page_ids_word_offset) + lane_idx
                ] = physical_page_id
        elif cutlass.const_expr(num_origins == 2):
            if lane_idx == Int32(0):
                self._smem_words[Int32(0)] = resolved_record_word
                self._smem_words[Int32(1)] = resolved_origin1
                self._smem_words[
                    Int32(self.route_layout.atom_valid_mask_word_offset)
                ] = resolved_atom_validity
        else:
            if lane_idx < Int32(num_origins):
                load_origin = Int32(resolved_record_word)
                if resolved_atom_validity == Int32(0):
                    # Fine-route K and V both consume this retained value.
                    # Materialize their TensorMap OOB coordinate once here
                    # instead of rechecking the invalid-origin sentinel for
                    # every atom copy in both producer passes.
                    load_origin = Int32(self.tma_oob_origin)
                self._smem_words[lane_idx] = load_origin
        if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
            if lane_idx == Int32(self.route_layout.route_flags_word_offset):
                prepared_route_flags = Int32(resolved_record_word)
                self._smem_words[Int32(self._retained_route_words - 1)] = Int32(
                    prepared_route_flags
                ) & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG)
        # K consumes this slot immediately, while V consumes it at the start
        # of the next cadence.  Both execute in this warp, so a warp fence is
        # sufficient; no cross-warp mbarrier belongs here.
        cute.arch.sync_warp()

    @cute.jit
    def route_tma_coordinate(
        self,
        atom_idx: Int32,
        logical_b_idx: Int32,
    ) -> tuple[Int32, Int32]:
        """Load one retained sparse atom and return its TensorMap coordinates."""

        logical_origin = Int32(self._smem_words[atom_idx])
        if cutlass.const_expr(not self.route_layout.is_paged):
            # Invalid contiguous atoms were normalized to the TensorMap OOB
            # token coordinate when the route was retained, so their storage
            # coordinate remains the live batch index.
            return logical_origin, logical_b_idx

        physical_page_id = Int32(
            self._smem_words[
                Int32(self.route_layout.physical_page_ids_word_offset) + atom_idx
            ]
        )
        atom_is_valid = cutlass.Boolean(
            logical_origin >= Int32(0) and physical_page_id >= Int32(0)
        )
        return _paged_sparse_kv_load_coordinate(
            logical_origin,
            physical_page_id,
            atom_is_valid,
            self.route_layout.paged_page_size,
        )

    @cute.jit
    def route_atom_valid_mask(self) -> Int32:
        """Load the retained route's two-fragment validity mask."""

        assert not self.route_layout.is_paged
        assert self.route_layout.logical_origins_per_route == 2
        return Int32(
            self._smem_words[Int32(self.route_layout.atom_valid_mask_word_offset)]
        )

    @cute.jit
    def route_is_proxy(self) -> cutlass.Boolean:
        """Return the retained prepared route kind for the current K/V pair."""

        if cutlass.const_expr(not self.cfg.use_block_sparse_proxy_routes):
            return cutlass.Boolean(False)
        return cutlass.Boolean(
            (
                Int32(self._smem_words[Int32(self._retained_route_words - 1)])
                & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG)
            )
            != Int32(0)
        )


@dataclass(kw_only=True)
class SmemBlockSparseSoftmaxMetadataResource(DecodeGenResourceBase):
    """Staged route and token metadata consumed by one Softmax group.

    ``inst_id`` identifies which of the two Softmax pipelines owns this
    resource. Route resolution belongs to the paired K/V resource, so the
    producer passes the resolved payload explicitly instead of recomputing it.
    For Keeps, every route token word moves through SMEM without a
    data-dependent branch; each consumer receives at most four words through
    the stable task-local ABI. Runtime route flags carry the conservative FULL
    summary and, for proxy-capable builds, the route source kind.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("softmax_origin0_slot", Int32, Int32(0), "Loaded first logical origin."),
        ("softmax_origin1_slot", Int32, Int32(0), "Loaded second logical origin."),
        (
            "softmax_route_flags_slot",
            Int32,
            Int32(0),
            "Keeps route flags or SWAP's third logical origin.",
        ),
        (
            "softmax_token_word0_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Keeps token word 0 or SWAP's fourth origin as unsigned bits.",
        ),
        (
            "softmax_token_word1_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Keeps token word 1 or SWAP's packed logical K32 token word.",
        ),
        (
            "softmax_token_word2_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded third Keeps token word or SWAP's packed route flags.",
        ),
        (
            "softmax_token_word3_slot",
            Uint32,
            Uint32(0xFFFFFFFF),
            "Loaded fourth Keeps token-validity word; unused by SWAP.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    route_metadata: cute.Pointer | None = None
    staging_layout: Constexpr[_BlockSparseSoftmaxStagingLayout | None] = None
    route_layout: Constexpr[_BlockSparseRouteLayout | None] = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_words: cutlass.Array = None
    softmax_origin0_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_origin1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_route_flags_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word0_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word1_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word2_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    softmax_token_word3_slot: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self) -> None:
        """Derive the staged metadata layout."""

        assert self.route_layout is not None
        assert self.route_layout.is_paged == self.cfg.use_paged_kv
        if self.cfg.use_block_sparse_proxy_routes:
            assert self.route_layout.uses_one_warp_transport
        self.staging_layout = _BlockSparseSoftmaxStagingLayout.create(
            use_keeps_mma_ab=self.cfg.use_keeps_mma_ab,
            route_layout=self.route_layout,
            num_stages=self.pipeline_config.num_stages,
        )
        super().__post_init__()

    def _init_placeholder_state(self) -> None:
        """Create shape-correct staged SMEM for task-graph tracing."""

        self._smem_words = _placeholder_smem_array(
            Int32, self.staging_layout.total_words
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one metadata payload per configured pipeline stage."""

        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.staging_layout.size_bytes,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Softmax route metadata uses SMEM and registers only."""

        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the staged allocation on producer and consumer tasks."""

        if cutlass.const_expr(context is not None and context.smem_base is not None):
            self._smem_words = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=Int32,
                shape=(self.staging_layout.total_words,),
                addrspace=3,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Bind producer-side metadata storage before the first route."""

        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        """Bind consumer-side metadata storage before the first wait."""

        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _producer_stage_base(self, stage_info: StageInfo) -> Int32:
        """Return the producer stage selected by the task scheduler."""

        return stage_info.stage_idx * Int32(self.staging_layout.stage_stride_words)

    @cute.jit
    def _consumer_stage_base(self) -> Int32:
        """Return the consumer stage selected by the latest wait."""

        return self.consumer_work_stage * Int32(self.staging_layout.stage_stride_words)

    @cute.jit
    def _store_route_swaps(
        self,
        stage_info: StageInfo,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
        resolved_atom_validity: Int32,
        route_record_word_offset: Int32,
    ) -> None:
        """Stage SWAP origins and optional logical-K32 token metadata.

        Selected prepared route flags use the free low bits of each warp's
        first aligned origin.
        """

        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        stage_base = self._producer_stage_base(stage_info)
        task_cache = _decode_gen_task_cache(stage_info)
        seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])
        uses_one_warp_transport = self.route_layout.uses_one_warp_transport
        if cutlass.const_expr(
            not uses_one_warp_transport
            and (
                _swaps_forwards_packed_route_full(self.cfg)
                or self.cfg.uses_prepared_score_keep_words
            )
        ):
            route_record_is_valid = route_record_word_offset >= Int32(0)

        packed_route_flags = Int32(0)
        if cutlass.const_expr(
            _swaps_forwards_packed_route_full(self.cfg)
            or self.cfg.use_block_sparse_proxy_routes
        ):
            if cutlass.const_expr(uses_one_warp_transport):
                packed_route_flags = _warp_broadcast_i32(
                    resolved_record_word,
                    self.route_layout.route_flags_word_offset,
                )
            else:
                if lane_idx == Int32(0) and route_record_is_valid:
                    packed_route_flags = Int32(
                        self.route_metadata[
                            route_record_word_offset
                            + Int32(self.route_layout.route_flags_word_offset)
                        ]
                    ) & Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
                packed_route_flags = _warp_broadcast_i32(packed_route_flags, 0)

        softmax_origin = Int32(-1)
        if cutlass.const_expr(self.cfg.kv_block_size < 64):
            if lane_idx < Int32(self.staging_layout.num_origin_words):
                softmax_origin = Int32(resolved_record_word)
                if resolved_atom_validity == Int32(0):
                    softmax_origin = Int32(-1)
                if cutlass.const_expr(
                    _swaps_forwards_packed_route_full(self.cfg)
                    or self.cfg.use_block_sparse_proxy_routes
                ):
                    # Every SWAP atom is at least B8 aligned. Replicate the
                    # route flags in each K32 slice's first origin so the
                    # established seven-slot Softmax ABI also carries source
                    # kind without growing the staged payload.
                    if lane_idx % Int32(self.staging_layout.origins_per_warp) == Int32(
                        0
                    ):
                        softmax_origin = (
                            softmax_origin & Int32(_SWAPS_PACKED_ROUTE_FLAGS_CLEAR_MASK)
                        ) | packed_route_flags
                self._smem_words[stage_base + lane_idx] = softmax_origin
        else:
            # SWAP with a coarse KV atom expands the two resolved KV64
            # fragments into the four logical K32 origins consumed by its
            # four softmax warps.
            coarse_origin0 = Int32(resolved_record_word)
            if cutlass.const_expr(uses_one_warp_transport):
                coarse_origin0 = _warp_broadcast_i32(resolved_record_word, 0)
            if lane_idx < Int32(4):
                fragment_idx = lane_idx >> Int32(1)
                softmax_origin = coarse_origin0
                valid = (resolved_atom_validity & Int32(1)) != Int32(0)
                if fragment_idx == Int32(1):
                    softmax_origin = Int32(resolved_origin1)
                    valid = (resolved_atom_validity & Int32(2)) != Int32(0)
                softmax_origin = softmax_origin + (lane_idx & Int32(1)) * Int32(32)
                if not valid or softmax_origin >= seq_len_kv:
                    softmax_origin = Int32(-1)
                if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
                    # Coarse SWAP expands KV64 atoms to K32-aligned origins;
                    # their low bits carry the same typed-route flags as the
                    # fine-route representation above.
                    softmax_origin = (
                        softmax_origin & Int32(_SWAPS_PACKED_ROUTE_FLAGS_CLEAR_MASK)
                    ) | packed_route_flags
                self._smem_words[stage_base + lane_idx] = softmax_origin

        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            assert self.route_metadata is not None
            assert self.route_layout.token_words_word_offset is not None
            assert self.staging_layout.token_words_word_offset is not None
            if cutlass.const_expr(uses_one_warp_transport):
                token_begin = Int32(self.route_layout.token_words_word_offset)
                token_end = token_begin + Int32(self.route_layout.token_words_per_route)
                if lane_idx >= token_begin and lane_idx < token_end:
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset)
                        + lane_idx
                        - token_begin
                    ] = Int32(resolved_record_word)
            elif lane_idx < Int32(self.route_layout.token_words_per_route):
                logical_word = Uint32(0)
                if route_record_is_valid:
                    logical_word = Uint32(
                        self.route_metadata[
                            route_record_word_offset
                            + Int32(self.route_layout.token_words_word_offset)
                            + lane_idx
                        ]
                    )
                self._smem_words[
                    stage_base
                    + Int32(self.staging_layout.token_words_word_offset)
                    + lane_idx
                ] = Int32(logical_word)
        cute.arch.sync_warp()

    @cute.jit
    def _store_route_keeps(
        self,
        stage_info: StageInfo,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
        resolved_atom_validity: Int32,
        route_record_word_offset: Int32,
    ) -> None:
        """Stage a Keeps route and its already prepared token metadata.

        Origins and token words are lane-distributed, so KV128 and KV256 use
        the same producer shape. The consumer later selects the two KV64
        origins and K32 words owned by its KV256 spatial half.
        """

        assert self.staging_layout.route_flags_word_offset is not None
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        num_origins = self.route_layout.logical_origins_per_route
        uses_one_warp_transport = self.route_layout.uses_one_warp_transport
        if cutlass.const_expr(
            not uses_one_warp_transport and self.cfg.uses_prepared_score_keep_words
        ):
            route_record_is_valid = route_record_word_offset >= Int32(0)
        route_flags = Int32(resolved_atom_validity)
        if cutlass.const_expr(num_origins > 2):
            route_flags = Int32(
                cute.arch.vote_ballot_sync(
                    lane_idx < Int32(num_origins) and resolved_atom_validity != Int32(0)
                )
            )

        token_word = Uint32(0)
        route_token_mask_is_full = cutlass.Boolean(False)
        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            assert self.route_layout.token_words_word_offset is not None
            assert self.route_metadata is not None
            if cutlass.const_expr(not uses_one_warp_transport):
                gmem_route_flags = Int32(0)
                if lane_idx == Int32(0) and route_record_is_valid:
                    gmem_route_flags = Int32(
                        self.route_metadata[
                            route_record_word_offset
                            + Int32(self.route_layout.route_flags_word_offset)
                        ]
                    )
                gmem_route_flags = _warp_broadcast_i32(gmem_route_flags, 0)
                # Prepared bit 0 summarizes the whole route. Staged low bits
                # already hold fragment validity, so remap it above them.
                route_token_mask_is_full = cutlass.Boolean(
                    (gmem_route_flags & Int32(_PREPARED_ROUTE_IS_FULL_FLAG)) != Int32(0)
                )
                if (
                    lane_idx < Int32(self.route_layout.token_words_per_route)
                    and route_record_is_valid
                ):
                    token_word = Uint32(
                        self.route_metadata[
                            route_record_word_offset
                            + Int32(self.route_layout.token_words_word_offset)
                            + lane_idx
                        ]
                    )

        stage_base = self._producer_stage_base(stage_info)
        if cutlass.const_expr(num_origins == 2):
            if lane_idx == Int32(0):
                self._smem_words[stage_base] = Int32(resolved_record_word)
                self._smem_words[stage_base + Int32(1)] = Int32(resolved_origin1)
        elif lane_idx < Int32(num_origins):
            self._smem_words[stage_base + lane_idx] = Int32(resolved_record_word)

        if cutlass.const_expr(uses_one_warp_transport):
            if lane_idx == Int32(self.route_layout.route_flags_word_offset):
                prepared_route_flags = Int32(resolved_record_word)
                route_flags = route_flags | (
                    Int32(
                        (prepared_route_flags & Int32(_PREPARED_ROUTE_IS_FULL_FLAG))
                        != Int32(0)
                    )
                    * Int32(_SOFTMAX_TOKEN_MASK_IS_FULL_FLAG)
                )
                if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
                    route_flags = route_flags | (
                        Int32(
                            (
                                prepared_route_flags
                                & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG)
                            )
                            != Int32(0)
                        )
                        * Int32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)
                    )
                self._smem_words[
                    stage_base + Int32(self.staging_layout.route_flags_word_offset)
                ] = route_flags
        elif lane_idx == Int32(0):
            if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
                route_flags = route_flags | (
                    Int32(route_token_mask_is_full)
                    * Int32(_SOFTMAX_TOKEN_MASK_IS_FULL_FLAG)
                )
            self._smem_words[
                stage_base + Int32(self.staging_layout.route_flags_word_offset)
            ] = route_flags

        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            assert self.staging_layout.token_words_word_offset is not None
            if cutlass.const_expr(uses_one_warp_transport):
                token_begin = Int32(self.route_layout.token_words_word_offset)
                token_end = token_begin + Int32(self.route_layout.token_words_per_route)
                if lane_idx >= token_begin and lane_idx < token_end:
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset)
                        + lane_idx
                        - token_begin
                    ] = Int32(resolved_record_word)
            elif lane_idx < Int32(self.route_layout.token_words_per_route):
                self._smem_words[
                    stage_base
                    + Int32(self.staging_layout.token_words_word_offset)
                    + lane_idx
                ] = Int32(token_word)
        cute.arch.sync_warp()

    @producer_work
    @cute.jit
    def store_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
        resolved_atom_validity: Int32,
        route_record_word_offset: Int32,
    ) -> None:
        """Store one resolved route and its optional token words in a stage."""

        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            self._store_route_keeps(
                stage_info,
                resolved_record_word,
                resolved_origin1,
                resolved_atom_validity,
                route_record_word_offset,
            )
        else:
            self._store_route_swaps(
                stage_info,
                resolved_record_word,
                resolved_origin1,
                resolved_atom_validity,
                route_record_word_offset,
            )

    @cute.jit
    def _load_route_swaps_values(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Uint32, Uint32, Uint32]:
        """Load one SWAP warp's logical KV origins and optional token mask.

        ``origin0..3`` are logical KV-token atom bases assigned to
        this Softmax warp's logical K32 slice; unused or invalid origins are
        negative. To preserve the shared seven-slot task ABI, origin 2/3
        subsequently travel through the shared route-flags/token-word-0 slots.
        Token-word 1 carries the logical K32 mask, token-word 2 carries packed
        route flags, and token-word 3 is unused.
        """

        stage_base = self._consumer_stage_base()
        local_warp_idx = Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_IDX])
        warp_origin_base = stage_base + local_warp_idx * Int32(
            self.staging_layout.origins_per_warp
        )
        origin0 = Int32(self._smem_words[warp_origin_base])
        origin1 = Int32(-1)
        origin2 = Int32(-1)
        origin3 = Int32(-1)
        if cutlass.const_expr(self.staging_layout.origins_per_warp >= 2):
            origin1 = Int32(self._smem_words[warp_origin_base + Int32(1)])
        if cutlass.const_expr(self.staging_layout.origins_per_warp >= 3):
            origin2 = Int32(self._smem_words[warp_origin_base + Int32(2)])
        if cutlass.const_expr(self.staging_layout.origins_per_warp >= 4):
            origin3 = Int32(self._smem_words[warp_origin_base + Int32(3)])

        token_word = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            assert self.staging_layout.token_words_word_offset is not None
            token_word = Uint32(
                self._smem_words[
                    stage_base
                    + Int32(self.staging_layout.token_words_word_offset)
                    + local_warp_idx
                ]
            )
        route_flags = Uint32(0)
        if cutlass.const_expr(
            _swaps_forwards_packed_route_full(self.cfg)
            or self.cfg.use_block_sparse_proxy_routes
        ):
            packed_route_flags = origin0 & Int32(
                _PREPARED_ROUTE_IS_FULL_FLAG | _PREPARED_ROUTE_IS_PROXY_FLAG
            )
            route_flags = Uint32(
                packed_route_flags & Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
            )
            if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
                route_flags = route_flags | Uint32(
                    Int32(
                        (packed_route_flags & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG))
                        != Int32(0)
                    )
                    * Int32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)
                )
            origin0 = origin0 & Int32(_SWAPS_PACKED_ROUTE_FLAGS_CLEAR_MASK)
        return (
            origin0,
            origin1,
            origin2,
            origin3.bitcast(Uint32),
            token_word,
            route_flags,
        )

    @consumer_work(
        returns=(
            softmax_origin0_slot,
            softmax_origin1_slot,
            softmax_route_flags_slot,
            softmax_token_word0_slot,
            softmax_token_word1_slot,
            softmax_token_word2_slot,
            softmax_token_word3_slot,
        )
    )
    @cute.jit
    def load_route(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Uint32, Uint32, Uint32, Uint32]:
        """Copy the waited stage to task-local registers before release."""

        if cutlass.const_expr(not self.cfg.use_keeps_mma_ab):
            # Reuse the original seven-slot task ABI: Task7 interprets the
            # middle fields as origin2, origin3 bits, and the logical K32 mask.
            origin0, origin1, origin2, origin3_bits, token_word, route_flags = (
                self._load_route_swaps_values(stage_info)
            )
            return (
                origin0,
                origin1,
                origin2,
                origin3_bits,
                token_word,
                route_flags,
                Uint32(0xFFFFFFFF),
            )

        assert self.staging_layout.route_flags_word_offset is not None
        stage_base = self._consumer_stage_base()
        stored_route_flags = Int32(
            self._smem_words[
                stage_base + Int32(self.staging_layout.route_flags_word_offset)
            ]
        )
        origin0_idx = Int32(0)
        origin1_idx = Int32(1)
        route_flags = stored_route_flags
        if cutlass.const_expr(self.route_layout.kv_route_size == 256):
            # KV256 threads [0, 64) and [64, 128) own alternating KV64
            # origins: (0, 2) and (1, 3), respectively. Remap their validity
            # bits into the existing two-origin task-local ABI.
            assert self.route_layout.atom_size == 64
            warp_grp_thread_idx = Int32(
                _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            )
            spatial = warp_grp_thread_idx >> Int32(6)
            origin0_idx = spatial
            origin1_idx = spatial + Int32(2)
            valid0 = (stored_route_flags >> origin0_idx) & Int32(1)
            valid1 = (stored_route_flags >> origin1_idx) & Int32(1)
            route_flags = valid0 | (valid1 << Int32(1))
            if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
                route_flags = route_flags | (
                    stored_route_flags & Int32(_SOFTMAX_TOKEN_MASK_IS_FULL_FLAG)
                )
            if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
                route_flags = route_flags | (
                    stored_route_flags & Int32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)
                )
        origin0 = Int32(self._smem_words[stage_base + origin0_idx])
        origin1 = Int32(self._smem_words[stage_base + origin1_idx])
        token_word0 = Uint32(0xFFFFFFFF)
        token_word1 = Uint32(0xFFFFFFFF)
        token_word2 = Uint32(0xFFFFFFFF)
        token_word3 = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.cfg.uses_prepared_score_keep_words):
            assert self.staging_layout.token_words_word_offset is not None
            if cutlass.const_expr(self.route_layout.kv_route_size == 256):
                token_base = Int32(self.staging_layout.token_words_word_offset)
                word0_idx = origin0_idx * Int32(2)
                word1_idx = origin1_idx * Int32(2)
                token_word0 = Uint32(
                    self._smem_words[stage_base + token_base + word0_idx]
                )
                token_word1 = Uint32(
                    self._smem_words[stage_base + token_base + word0_idx + Int32(1)]
                )
                token_word2 = Uint32(
                    self._smem_words[stage_base + token_base + word1_idx]
                )
                token_word3 = Uint32(
                    self._smem_words[stage_base + token_base + word1_idx + Int32(1)]
                )
            elif cutlass.const_expr(self.cfg.tile_size_q == 64):
                lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
                local_word_base = _keeps_col_base(
                    self.cfg,
                    lane_idx,
                    self.cfg.num_s_regs_per_thread,
                ) >> Int32(5)
                token_word0 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset)
                        + local_word_base
                    ]
                )
                token_word1 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset)
                        + local_word_base
                        + Int32(1)
                    ]
                )
            else:
                token_word0 = Uint32(
                    self._smem_words[
                        stage_base + Int32(self.staging_layout.token_words_word_offset)
                    ]
                )
                token_word1 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset + 1)
                    ]
                )
                token_word2 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset + 2)
                    ]
                )
                token_word3 = Uint32(
                    self._smem_words[
                        stage_base
                        + Int32(self.staging_layout.token_words_word_offset + 3)
                    ]
                )
        return (
            origin0,
            origin1,
            route_flags,
            token_word0,
            token_word1,
            token_word2,
            token_word3,
        )


__all__ = [
    "SmemBlockSparseKvMetadataResource",
    "SmemBlockSparseSoftmaxMetadataResource",
]
