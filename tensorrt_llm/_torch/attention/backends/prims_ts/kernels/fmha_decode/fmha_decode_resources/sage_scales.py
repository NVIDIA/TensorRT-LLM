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

"""Sage attention scale addressing for the decode softmax and epilogue.

Every scale reaches the kernel from the buffer the Sage prepare writes
before attention (``_SageScaleLayout``), in the order the kernel reads it:
``sfQ`` as one word per row of each Q tile, ``sfK`` as one atom of words per
64 K tokens, ``sfV`` and ``v_mean`` as one row per sequence. The prepare
resolves the flat layout (``flat_scale_slot`` of
:mod:`flashinfer.attention.prims_ts.sage`, ``load_flat_scale``) and gives Q
rows past the valid row count and K tokens past the sequence end the last
valid scale, so masked scores keep a finite scale and exponentiate to zero.
Each softmax lane loads its row's ``sfQ`` word once per work tile
(``load_q_scale``).

:class:`SageKScalesResource` is one softmax instance's ring of ``sfK``
tiles, which the load warp produces after each K tile with ``cp.async``
copies under the ring's ``AsyncLoad`` commit: the tile's atoms, a dense
tile's or a block-sparse route's, from the prepared scales
(``_copy_atom_scales``). After the softmax task waits a stage,
``open_stage`` publishes the stage's shared-memory offset as a task-local
value that the schedule routes into the max pass and the P pass; both read
the stage in place through ``sage_k_view`` and ``load_sage_k_fragment``, one
K32 fragment at a time with loads of at most 16 bytes, so attention never
addresses ``k_scale`` itself and no other resource holds the ring.

:class:`SageVScalesResource` stages the work tile's per-channel V scales
(and means) in SMEM once per tile, while no output is pending, so the
epilogue scales column ``c`` by ``sfV[c]`` and adds ``v_mean[c]`` without
waiting on global loads.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
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
    _PREPARED_ROUTE_IS_PROXY_FLAG,
    _BlockSparseRouteLayout,
)
from ....sage import _FLOAT4_WORDS, _SAGE_ATOM_TOKENS, _SageKSection, flat_scale_slot
from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import KV_KIND_K
from .helpers_common import (
    DecodeGenResourceBase,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _decode_gen_task_cache,
    _keeps_spatial_half,
    _logical_head_batch,
    _warp_broadcast_i32,
    fmul2,
)
from .helpers_kv_tile_idx import (
    _local_kv_tile_idx_for_section,
    resolve_keeps_tile_context,
)

Constexpr = cutlass.Constexpr


@cute.jit
def _load_scale(scale_addr: Int64, index: Int32) -> Float32:
    """Return one FP32 scale loaded from a global memory base address."""
    value_ptr = cutlass.inttoptr(
        scale_addr + Int64(index) * Int64(4),
        mem_space=1,
        dtype=Float32,
    )
    return Float32(value_ptr.load(count=1, alignment=4)[0])


@cute.jit
def load_flat_scale(
    scale_addr: Int64,
    head_stride: Int32,
    *,
    head_idx: Int32,
    batch_idx: Int32,
    seq_len: Int32,
    token_idx: Int32,
    log2_block: Int32,
) -> Float32:
    """Return one head's scale of one token from a flat-layout scale array.

    A token past ``seq_len`` takes the last valid slot.
    """
    token_idx = cute.math.min(token_idx, seq_len - Int32(1))
    slot = flat_scale_slot(batch_idx, token_idx, seq_len, log2_block)
    return _load_scale(scale_addr, head_idx * head_stride + slot)


@cute.jit
def sage_sequence_words(
    cfg: Constexpr[FmhaDecodeConfig],
    *,
    batch_idx: Int32,
    kv_head_idx: Int32,
    num_heads_kv: Int32,
) -> Int32:
    """Return the first word of sequence ``b * Hkv + h`` in the prepared scales."""
    layout = cfg.sage_scale_layout(cfg.static_seq_len_kv)
    sequence_idx = batch_idx * num_heads_kv + kv_head_idx
    return sequence_idx * Int32(layout.sequence_words)


@cute.jit
def load_q_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    scales_ptr: cute.Pointer,
    *,
    sequence_words: Int32,
    q_group_idx: Int32,
    row_idx: Int32,
) -> Float32:
    """Return ``sfQ`` of row ``row_idx`` of Q tile ``q_group_idx`` from the prepared scales.

    ``sequence_words`` is the first word of the row's sequence
    (``sage_sequence_words``). The word already holds the scale of the row's
    token and Q head, the last valid row's for a row past the tile's valid
    rows (``_SageQSection``).
    """
    q_rows = cfg.sage_scale_layout(cfg.static_seq_len_kv).q_rows
    word = (
        sequence_words
        + Int32(q_rows.word_offset)
        + q_group_idx * Int32(q_rows.tile_size_q)
        + row_idx
    )
    return _load_scale(scales_ptr.toint(), word)


def sage_k_section(cfg: FmhaDecodeConfig, proxy: bool = False) -> _SageKSection:
    """Return the prepared K section one route kind reads.

    A proxy route reads the summaries, every other tile the K tokens. The
    section's words per atom are the softmax's scale groups of the atom's
    two fragments, so an unpadded atom's words are contiguous per fragment.
    """
    sections = cfg.sage_scale_layout(cfg.static_seq_len_kv).k_sections
    section = sections[-1] if proxy else sections[0]
    assert section.words_per_atom == (
        cfg.keeps_fragments_per_atom * cfg.sage_k_groups_per_fragment(proxy)
    )
    return section


def sage_k_stage_words(cfg: FmhaDecodeConfig) -> int:
    """Return the ``sfK`` words of one ring stage: a KV tile's atoms of the larger section.

    A route kind's atoms lie contiguously in the stage in the order of
    ``FmhaDecodeConfig.keeps_route_atom_owner``: atom ``a``, held by spatial
    half ``h`` at position ``p``, is stage atom ``h * atoms_per_half + p``,
    so each half's words are contiguous.
    """
    atom_words = max(sage_k_section(cfg, proxy).atom_words for proxy in (False, True))
    return cfg.tile_size_kv // cfg.keeps_atom_tokens * atom_words


@cute.jit
def _copy_atom_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    scales_ptr: cute.Pointer,
    stage_words: cute.Pointer,
    section_words: Int32,
    atom_words: Constexpr[int],
    atom_index: Constexpr,
    num_atoms: Constexpr[int],
) -> None:
    """Copy ``num_atoms`` atoms of one K section into a ring stage.

    Lane ``l`` copies float4 ``l``, ``l + 32``, ... of the atoms with
    ``cp.async``, so the TMA unit stays free for K and V. Each float4 moves
    from word ``section_words + atom_index(atom) * atom_words`` of the
    prepared scales plus its offset in the atom, ``section_words`` being the
    section's first word in the tile's sequence, to the stage word of the
    atom's spatial half and position (``sage_k_stage_words``) plus the same
    offset.
    """
    atom_float4s = atom_words // _FLOAT4_WORDS
    num_float4s = num_atoms * atom_float4s
    lane_idx = cute.arch.thread_idx()[0] & Int32(31)
    for round_idx in cutlass.range_constexpr((num_float4s + 31) // 32):
        float4 = lane_idx + Int32(round_idx * 32)
        atom = float4 // Int32(atom_float4s)
        word_in_atom = (float4 % Int32(atom_float4s)) * Int32(_FLOAT4_WORDS)
        half, position = cfg.keeps_route_atom_owner(atom)
        stage_atom = half * Int32(num_atoms // cfg.keeps_spatial_halves) + position
        # Every lane resolves its atom's index: a route's lookup shuffles.
        source_word = section_words + atom_index(atom) * Int32(atom_words)
        if float4 < Int32(num_float4s):
            prims.cp_async_shared_global(
                stage_words + stage_atom * Int32(atom_words) + word_in_atom,
                scales_ptr + source_word + word_in_atom,
                _FLOAT4_WORDS * 4,
                "cg",
            )


@cute.jit
def _route_atom_origin(
    route_layout: Constexpr[_BlockSparseRouteLayout],
    record_word: Int32,
    origin1: Int32,
    atom: Int32,
) -> Int32:
    """Return the token origin of route atom ``atom`` from the resolved record.

    ``SmemBlockSparseKvMetadataResource.resolve_route`` hands the load warp
    its record in one of three forms:

    * a two-atom route without one-warp transport returns both origins as
      warp-uniform scalars, ``record_word`` for atom 0 and ``origin1`` for
      atom 1;
    * a two-atom route with one-warp transport returns lane-distributed
      origins in ``record_word``, lane ``a`` holding atom ``a``'s, and atom
      1's again in ``origin1``;
    * a wider route returns lane-distributed origins in ``record_word`` and
      zero in ``origin1``.

    A lane-distributed origin is shuffled from its lane, one SHFL per float4
    round, also where lane ``a`` copies atom ``a`` and holds it already. An
    invalid atom has origin ``-1``.
    """
    if cutlass.const_expr(
        route_layout.logical_origins_per_route == 2
        and not route_layout.uses_one_warp_transport
    ):
        origin = Int32(record_word)
        if atom != Int32(0):
            origin = Int32(origin1)
        return origin
    return cute.arch.shuffle_sync(record_word, atom)


@cute.jit
def _route_atom_index(
    route_layout: Constexpr[_BlockSparseRouteLayout],
    record_word: Int32,
    origin1: Int32,
    atom: Int32,
) -> Int32:
    """Return the section atom of route atom ``atom``: the 64-token atom of its origin.

    An invalid atom takes atom 0, whose finite scales its masked scores
    never use.
    """
    origin = _route_atom_origin(route_layout, record_word, origin1, atom)
    return cute.math.max(origin, Int32(0)) // Int32(_SAGE_ATOM_TOKENS)


@cute.jit
def copy_route_k_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    route_layout: Constexpr[_BlockSparseRouteLayout],
    scales_ptr: cute.Pointer,
    *,
    stage_words: cute.Pointer,
    sequence_words: Int32,
    record_word: Int32,
    origin1: Int32,
) -> None:
    """Copy a block-sparse route's ``sfK`` atoms from the prepared scales into a ring stage.

    ``sequence_words`` is the first word of the route's sequence. Each route
    atom reads the atom of its origin (``_route_atom_index``) from the
    section its route kind reads: the K tokens, or the summaries of a proxy
    route. A mixed plan's sections differ in atom size, so each kind has its
    own copy; otherwise one copy serves both kinds from the kind's section.
    """
    k_tokens = sage_k_section(cfg)
    summaries = sage_k_section(cfg, proxy=True)
    num_atoms = route_layout.logical_origins_per_route
    is_proxy_route = cutlass.Boolean(False)
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        route_flags = _warp_broadcast_i32(
            record_word, route_layout.route_flags_word_offset
        )
        is_proxy_route = (route_flags & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG)) != Int32(
            0
        )
    atom_index = lambda atom: _route_atom_index(
        route_layout, record_word, origin1, atom
    )
    copy = lambda section, section_words: _copy_atom_scales(
        cfg,
        scales_ptr,
        stage_words,
        section_words,
        section.atom_words,
        atom_index,
        num_atoms,
    )
    if cutlass.const_expr(cfg.sage_mixed_k_geometry):
        if is_proxy_route:
            copy(summaries, sequence_words + Int32(summaries.word_offset))
        else:
            copy(k_tokens, sequence_words + Int32(k_tokens.word_offset))
    else:
        section_words = sequence_words + Int32(k_tokens.word_offset)
        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            if is_proxy_route:
                section_words = sequence_words + Int32(summaries.word_offset)
        copy(k_tokens, section_words)


@cute.jit
def scale_pairs_in_place(
    values: cutlass.Array, factor: Float32, count: Constexpr[int]
) -> None:
    """Multiply ``count`` values in place by ``factor``, two at a time."""
    assert count % 2 == 0
    for pair_base in cutlass.range_constexpr(0, count, 2):
        values[pair_base], values[pair_base + 1] = fmul2(
            (factor, factor),
            (Float32(values[pair_base]), Float32(values[pair_base + 1])),
        )


def staged_v_channel_scale_entries(cfg: FmhaDecodeConfig) -> int:
    """Return the number of SMEM floats holding one KV head's V scales and means.

    The scales occupy ``[0, headdim)``; with ``sage_v_mean`` the means follow
    at ``[headdim, 2 * headdim)``.
    """
    return cfg.headdim * (2 if cfg.sage_v_mean else 1)


@cute.jit
def stage_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    scales_ptr: cute.Pointer,
    staged: cutlass.Array,
    *,
    sequence_words: Int32,
    thread_idx: Int32,
    num_threads: Constexpr[int],
) -> None:
    """Copy one sequence's per-channel V scales and means into SMEM.

    The sequence's V rows of the prepared scales (``_SageVSection``) have the
    staged form already: the scales, then the means. Each of the
    ``num_threads`` callers moves ``headdim / num_threads`` channels of each
    row. Callers order these writes before the epilogue reads, and the next
    tile's writes after the last read.
    """
    assert cfg.headdim % num_threads == 0
    v_rows = cfg.sage_scale_layout(cfg.static_seq_len_kv).v_rows
    scales_addr = scales_ptr.toint()
    rows_base = sequence_words + Int32(v_rows.word_offset)
    for base in cutlass.range_constexpr(0, cfg.headdim, num_threads):
        channel = Int32(base) + thread_idx
        staged[channel] = _load_scale(scales_addr, rows_base + channel)
        if cutlass.const_expr(cfg.sage_v_mean):
            staged[channel + Int32(cfg.headdim)] = _load_scale(
                scales_addr, rows_base + Int32(v_rows.mean_word_offset) + channel
            )


@cute.jit
def load_staged_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    staged: cutlass.Array,
    *,
    first_col: Int32,
    count: Constexpr[int],
) -> tuple[cutlass.Array, cutlass.Array]:
    """Return ``count`` staged V scales and means from ``first_col``.

    Without ``sage_v_mean`` the returned mean array is zero so callers can
    apply one fused multiply-add.
    """
    assert count % 4 == 0
    scales = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    means = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    for chunk in cutlass.range_constexpr(0, count, 4):
        chunk_col = first_col + Int32(chunk)
        scale_vec = (staged.data_ptr() + chunk_col).load(count=4, alignment=16)
        for elem in cutlass.range_constexpr(4):
            scales[chunk + elem] = Float32(scale_vec[elem])
        if cutlass.const_expr(cfg.sage_v_mean):
            mean_vec = (staged.data_ptr() + chunk_col + Int32(cfg.headdim)).load(
                count=4, alignment=16
            )
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(mean_vec[elem])
        else:
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(0.0)
    return scales, means


def sage_k_half_words(cfg: FmhaDecodeConfig, proxy: bool = False) -> int:
    """Return one route kind's stage words of one spatial half's atoms."""
    tile_atoms = cfg.tile_size_kv // cfg.keeps_atom_tokens
    atoms_per_half = tile_atoms // cfg.keeps_spatial_halves
    return atoms_per_half * sage_k_section(cfg, proxy).atom_words


@cute.jit
def sage_k_view(
    cfg: Constexpr[FmhaDecodeConfig],
    context: ResourceContext,
    stage_bytes: Int32,
    warp_grp_thread_idx: Int32,
    factor: Float32 | None,
    proxy_kind: Constexpr[bool] = False,
):
    """Return a pass's view of a waited ring stage: the lane's half pointer and ``factor``.

    ``stage_bytes`` is the stage's offset in the CTA's shared memory, as
    ``SageKScalesResource.open_stage`` publishes it; ``factor``, if any, is
    multiplied into every word ``load_sage_k_fragment`` returns.
    """
    stage_words = cutlass.Array(
        context.smem_base.data_ptr() + stage_bytes,
        dtype=Float32,
        shape=(sage_k_stage_words(cfg),),
        addrspace=3,
    ).data_ptr()
    half = _keeps_spatial_half(cfg, warp_grp_thread_idx)
    return stage_words + half * Int32(sage_k_half_words(cfg, proxy_kind)), factor


@cute.jit
def load_sage_k_fragment(
    cfg: Constexpr[FmhaDecodeConfig],
    view,
    fragment_idx: Int32,
    proxy_kind: Constexpr[bool] = False,
) -> cutlass.Array:
    """Return ``factor * sfK`` of one fragment's ``groups`` words of ``view``.

    Fragment ``f`` reads words ``(f % 2) * groups`` onward of atom ``f // 2``
    of the half; an unpadded atom's words are contiguous, so that is word ``f
    * groups`` of the half. The loads, at most 16 bytes each, issue ahead of
    the fragment's score wait in the callers.
    """
    groups = cfg.sage_k_groups_per_fragment(proxy_kind)
    values = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
    half_ptr, factor = view
    fragments_per_atom = cfg.keeps_fragments_per_atom
    section = sage_k_section(cfg, proxy_kind)
    if cutlass.const_expr(section.atom_words == section.words_per_atom):
        words_ptr = half_ptr + fragment_idx * Int32(groups)
    else:
        atom_position = fragment_idx // Int32(fragments_per_atom)
        words_ptr = (
            half_ptr
            + atom_position * Int32(section.atom_words)
            + fragment_idx % Int32(fragments_per_atom) * Int32(groups)
        )
    width = min(groups, 4)
    for word in cutlass.range_constexpr(0, groups, width):
        loaded = (words_ptr + Int32(word)).load(count=width, alignment=width * 4)
        for elem in cutlass.range_constexpr(width):
            values[word + elem] = Float32(loaded[elem])
    if cutlass.const_expr(factor is not None):
        if cutlass.const_expr(groups == 1):
            values[0] = Float32(values[0]) * factor
        else:
            scale_pairs_in_place(values, factor, groups)
    return values


@dataclass(kw_only=True)
class SageKScalesResource(DecodeGenResourceBase):
    """One softmax instance's ring of ``sfK`` tiles, produced by the load warp.

    Each stage holds one KV tile's words in the ``sage_k_stage_words`` layout.
    The load warp fills it after the tile's K copy with ``cp.async`` copies of
    the tile's atoms from the prepared scales, a block-sparse route's
    (``copy_route``) or a dense tile's (``copy_tile``), and the pipeline's
    ``AsyncLoad`` commit completes the stage once they land. The softmax waits
    a stage, takes its offset from ``open_stage``, reads it in place through
    ``sage_k_view`` and ``load_sage_k_fragment`` in the max pass and the P
    pass, and releases it after the last pass that reads the words.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "stage_bytes",
            Int32,
            Int32(0),
            "Shared-memory offset of the waited ``sfK`` ring stage.",
        ),
    )
    stage_bytes: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    route_layout: Constexpr[_BlockSparseRouteLayout | None] = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    seq_len_q: Int32 = None
    q_group_idx: Int32 | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    num_heads_kv: Int32 | None = None
    sage_scales_ptr: cute.Pointer | None = None
    _alloc: Constexpr[SmemAllocation | None] = None

    def __post_init__(self) -> None:
        assert self.pipeline_config is not None
        assert (self.route_layout is not None) == self.cfg.use_block_sparse
        super().__post_init__()

    @property
    def tile_words(self) -> int:
        """Return the ``sfK`` words of one tile in a ring stage."""
        return sage_k_stage_words(self.cfg)

    @property
    def tile_atoms(self) -> int:
        """Return the atoms of one KV tile."""
        return self.cfg.tile_size_kv // self.cfg.keeps_atom_tokens

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the ring."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.pipeline_config.num_stages * self.tile_words * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """The words live in SMEM only."""
        return []

    # -- producer steps ----------------------------------------------------

    @cute.jit
    def _stage_words(self, stage_info: StageInfo) -> cute.Pointer:
        """Return the acquired stage's first word as an FP32 SMEM pointer."""
        return self._ring(stage_info.context).data_ptr() + Int32(
            stage_info.stage_idx
        ) * Int32(self.tile_words)

    @cute.jit
    def _sequence_words(self, stage_info: StageInfo) -> Int32:
        """Return the first word of the work tile's sequence in the prepared scales."""
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        return sage_sequence_words(
            self.cfg,
            batch_idx=batch_idx,
            kv_head_idx=kv_head_idx,
            num_heads_kv=self.num_heads_kv,
        )

    @producer_work
    @cute.jit
    def copy_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
    ) -> None:
        """Copy a block-sparse route's words into the acquired ring stage.

        The load warp issues the ``cp.async`` copies only; the pipeline's
        ``AsyncLoad`` commit completes the stage once they land.
        """
        assert self.route_layout is not None
        copy_route_k_scales(
            self.cfg,
            self.route_layout,
            self.sage_scales_ptr,
            stage_words=self._stage_words(stage_info),
            sequence_words=self._sequence_words(stage_info),
            record_word=resolved_record_word,
            origin1=resolved_origin1,
        )

    @producer_work
    @cute.jit
    def copy_tile(
        self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]
    ) -> None:
        """Copy a dense tile's atoms from the prepared scales into the acquired ring stage.

        The tile is the instance's K tile of ``section``, resolved as the
        softmax resolves the tile it reads. Atom ``a`` of the tile reads the
        K-token atom of its first token, ``tile_offset_k + a * 64``, clamped
        to the sequence's last atom: a tile or atom past the sequence end has
        masked scores. The pipeline's ``AsyncLoad`` commit completes the
        stage once the copies land.
        """
        assert self.route_layout is None
        cfg = self.cfg
        (
            _seq_len_kv,
            _q_group_idx,
            _element_mask_end_idx,
            tile_offset_k,
            _window_start_idx,
            _is_valid_effective_tile,
            _is_masked_final_wave,
            _tile_is_unmasked,
            _rows_are_active,
        ) = resolve_keeps_tile_context(
            cfg,
            stage_info,
            inst_id=self.inst_id,
            seqlens_kv=self.seqlens_kv,
            max_seq_len_kv=self.max_seq_len_kv,
            seq_len_q=self.seq_len_q,
            q_group_idx=self.q_group_idx,
            local_tile_idx=_local_kv_tile_idx_for_section(
                cfg, stage_info, self.inst_id, KV_KIND_K, section
            ),
        )
        # A dense plan prepares the K-token section alone.
        k_tokens = sage_k_section(cfg)
        assert cfg.sage_scale_layout(cfg.static_seq_len_kv).summaries is None
        tile_atom = tile_offset_k // Int32(_SAGE_ATOM_TOKENS)
        last_atom = Int32(k_tokens.atoms - 1)
        _copy_atom_scales(
            cfg,
            self.sage_scales_ptr,
            self._stage_words(stage_info),
            self._sequence_words(stage_info) + Int32(k_tokens.word_offset),
            k_tokens.atom_words,
            lambda atom: cute.math.min(tile_atom + atom, last_atom),
            self.tile_atoms,
        )

    # -- storage helpers ---------------------------------------------------

    @cute.jit
    def _ring(self, context: ResourceContext) -> cutlass.Array:
        """Return the ring as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(self.pipeline_config.num_stages * self.tile_words,),
            addrspace=3,
        )

    # -- consumer steps ----------------------------------------------------

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=stage_bytes)
    @cute.jit
    def open_stage(self, stage_info: StageInfo) -> Int32:
        """Publish the waited stage's shared-memory offset for the passes that read it."""
        return Int32(self._alloc.offset) + Int32(self.consumer_work_stage) * Int32(
            self.tile_words * 4
        )


@dataclass(kw_only=True)
class SageVScalesResource(DecodeGenResourceBase):
    """The staged ``sfV`` and ``v_mean`` of the work tile's sequence.

    The correction warps copy the sequence's V rows of the prepared scales
    into the block at work-tile start; the pipeline is a one-stage pipeline
    of those warps.
    """

    cfg: Constexpr[FmhaDecodeConfig] = None
    sage_scales_ptr: cute.Pointer | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    num_heads_kv: Int32 | None = None
    _alloc: Constexpr[SmemAllocation | None] = None

    @property
    def entries(self) -> int:
        """Return the FP32 entries of the block: the scales, then the means."""
        return staged_v_channel_scale_entries(self.cfg)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the one-stage block of scales and means."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.entries * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """The scales live in SMEM only."""
        return []

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def stage_tile(self, stage_info: StageInfo) -> None:
        """Copy the work tile's V scales and means into the acquired block."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        stage_v_channel_scales(
            cfg,
            self.sage_scales_ptr,
            self.staged(stage_info.context),
            sequence_words=sage_sequence_words(
                cfg,
                batch_idx=batch_idx,
                kv_head_idx=kv_head_idx,
                num_heads_kv=self.num_heads_kv,
            ),
            thread_idx=task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX],
            num_threads=cfg.correction_barrier_threads,
        )

    @cute.jit
    def staged(self, context: ResourceContext) -> cutlass.Array:
        """Return the block as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(self.entries,),
            addrspace=3,
        )

    @cute.jit
    def channel_scales(
        self, staged: cutlass.Array, *, first_col: Int32, count: Constexpr[int]
    ) -> tuple[cutlass.Array, cutlass.Array]:
        """Return ``count`` scales and means from ``first_col`` of the waited block.

        Without ``sage_v_mean`` the means are zero so callers apply one fused
        multiply-add.
        """
        return load_staged_v_channel_scales(
            self.cfg, staged, first_col=first_col, count=count
        )
