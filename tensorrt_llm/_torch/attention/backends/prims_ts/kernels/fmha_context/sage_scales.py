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

"""Sage attention scale addressing for the context softmax and epilogue.

The Sage prepare writes one FP32 buffer per plan before attention, laid out
as ``_SageScaleLayout`` describes; a context Q tile holds 128 tokens of one Q
head. A softmax lane reads its ``sfQ`` word once per work tile with
``sage.index.load_sage_q_scale``, which the decode kernel shares. The helpers
here turn work-tile coordinates into words of that buffer and move the other
words the kernel reads:

* ``load_sage_k_tile`` reads one K tile's ``sfK`` atoms from the tile's
  address (``sage_k_section_address`` gives the sequence's first atom) for
  the softmax lanes; ``sage_k_transport`` says whether the plan's K block
  size takes that global-memory path or the load warp's shared-memory ring
  (``SmemSageKScalesResource``), and ``sage_score_scaling`` how the softmax
  applies the words;
  ``prefetch_sage_k_tile`` warms L1 with the next tile's atoms one tile ahead
  of direct loads;
* ``load_sage_v_channels`` reads the sequence's ``sfV`` / ``v_mean`` words per
  output column in the epilogue (``sage_v_section_address`` gives the
  sequence's first ``sfV`` word).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from ...sage.index import sage_section_word
from ...sage.layout import _FLOAT4_WORDS, _SAGE_ATOM_TOKENS, _SageScaleLayout
from ...sage.modes import SageKTransport, SageScoreScaling

Constexpr = cutlass.Constexpr

# Score columns that share one ``sfK`` word when the softmax scales blocks
# (``SageScoreScaling.BLOCK``), the width of one LDTM.STAT piece.
SAGE_SCORE_BLOCK_COLS = 16
# Bytes of one L1 line, the granularity of ``prefetch_sage_k_tile``.
_L1_LINE_BYTES = 128
# Most ``sfK`` words of one K tile the softmax lanes load from global memory
# and carry across the S wait as a task-local vector
# (``SageKTransport.GMEM``); more words go through the load warp's ring.
_GMEM_MAX_TILE_WORDS = 32


def validate_context_sage_layout(layout: _SageScaleLayout, kv_tile_n: int) -> None:
    """Reject a layout the context softmax cannot read tile by tile.

    The K-token atoms must be padded to whole K tiles, since the softmax
    addresses each tile's atoms without clamping.
    """
    if layout.k_tokens.tile_atoms * _SAGE_ATOM_TOKENS != kv_tile_n:
        raise ValueError(
            "the Sage K-token atoms must be padded to the kernel's K tiles"
        )
    if layout.q_rows.heads_interleaved:
        raise ValueError("the context kernel reads one Q head per 128 Q rows")


def sage_score_scaling(layout: _SageScaleLayout) -> SageScoreScaling:
    """Return how the context softmax applies the layout's ``sfK`` words.

    K blocks of 16 tokens or more give one word per 16-column score block:
    the scores stay raw and each block's maximum and exp2 multiplier carry
    ``sfQ * sfK``. Finer blocks need one multiplier per word of fewer
    columns, so the max pass multiplies the scores in place and the P pass
    runs the per-tensor path with ``sfQ`` in its exp2 scale.
    """
    if layout.k_tokens.tokens_per_word >= SAGE_SCORE_BLOCK_COLS:
        return SageScoreScaling.BLOCK
    return SageScoreScaling.DEQUANTIZED


def sage_k_transport(layout: _SageScaleLayout) -> SageKTransport:
    """Return how the context softmax obtains a K tile's ``sfK`` words.

    The lanes load a tile of at most ``_GMEM_MAX_TILE_WORDS`` words from
    global memory before the S wait; a tile with more words (one-token
    blocks) is too many to carry or to broadcast-load per lane, so the load
    warp stages it in the ``sfK`` ring (``SmemSageKScalesResource``).
    """
    if sage_k_tile_words(layout) <= _GMEM_MAX_TILE_WORDS:
        return SageKTransport.GMEM
    return SageKTransport.SMEM


def sage_k_tile_words(layout: _SageScaleLayout) -> int:
    """Return the ``sfK`` words of one K tile: its atoms, each padded to float4s."""
    return layout.k_tokens.tile_atoms * layout.k_tokens.atom_words


def sage_k_tile_bytes(layout: _SageScaleLayout) -> int:
    """Return the bytes of one K tile's ``sfK`` atoms."""
    return sage_k_tile_words(layout) * 4


def sage_k_block_word(layout: _SageScaleLayout, block_idx: int) -> int:
    """Return the tile word holding ``sfK`` of score block ``block_idx``.

    Block ``b`` starts at token ``16 b`` of the tile: word
    ``token_in_atom // tokens_per_word`` of atom ``token // 64``.
    """
    k_tokens = layout.k_tokens
    atom, token_in_atom = divmod(block_idx * SAGE_SCORE_BLOCK_COLS, _SAGE_ATOM_TOKENS)
    return atom * k_tokens.atom_words + token_in_atom // k_tokens.tokens_per_word


@cute.jit
def _section_address(
    scales: cute.Pointer, sequence_words: Int32, section: Constexpr
) -> Int64:
    """Return the address of the first word of a sequence's section."""
    return scales.toint() + Int64(sage_section_word(sequence_words, section)) * Int64(4)


@cute.jit
def sage_k_section_address(
    layout: Constexpr[_SageScaleLayout],
    scales: cute.Pointer,
    *,
    sequence_words: Int32,
) -> Int64:
    """Return the address of a sequence's first ``sfK`` atom.

    K tile ``t`` starts ``t * sage_k_tile_bytes`` after it. The section is
    padded to whole tiles, so every tile's atoms lie inside it; a tile past
    the sequence end reads the pad atoms, which repeat the last scale.
    """
    return _section_address(scales, sequence_words, layout.k_tokens)


@cute.jit
def sage_v_section_address(
    layout: Constexpr[_SageScaleLayout],
    scales: cute.Pointer,
    *,
    sequence_words: Int32,
) -> Int64:
    """Return the address of a sequence's first ``sfV`` word (``_SageVSection``)."""
    return _section_address(scales, sequence_words, layout.v_rows)


@cute.jit
def _load_float4_at(address: Int64):
    """Return the four words at a 16-byte aligned global address."""
    value_ptr = cutlass.inttoptr(address, mem_space=1, dtype=Float32)
    return value_ptr.load(count=_FLOAT4_WORDS, alignment=16)


@cute.jit
def load_sage_k_tile(
    layout: Constexpr[_SageScaleLayout], tile_address: Int64
) -> cutlass.Array:
    """Return the ``sfK`` words of the K tile at ``tile_address``.

    One float4 load per four words; every lane of a warp reads the same
    words, so the loads are warp broadcasts that hit L1 once
    ``prefetch_sage_k_tile`` has warmed the tile's lines.
    """
    tile_words = sage_k_tile_words(layout)
    words = cutlass.Array(Float32, tile_words, space=cutlass.AddressSpace.rmem)
    for word_base in cutlass.range_constexpr(0, tile_words, _FLOAT4_WORDS):
        loaded = _load_float4_at(tile_address + Int64(word_base * 4))
        for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
            words[word_base + elem] = Float32(loaded[elem])
    return words


@cute.jit
def prefetch_sage_k_tile(
    layout: Constexpr[_SageScaleLayout], tile_address: Int64
) -> None:
    """Bring the K tile at ``tile_address`` into L1 for a later ``load_sage_k_tile``.

    The tile after a sequence's last one lies in the sequence's Q rows, so
    the prefetch of a loop's final iteration stays inside the buffer.
    """
    for offset in cutlass.range_constexpr(0, sage_k_tile_bytes(layout), _L1_LINE_BYTES):
        # A tile is a whole number of 32-byte units, so it never straddles a
        # line boundary that this stride misses.
        cute.arch.inline_ptx(
            "prefetch.global.L1 [{$r0}];",
            read_only_args=[tile_address + Int64(offset)],
        )


@cute.jit
def load_sage_v_channels(
    layout: Constexpr[_SageScaleLayout],
    channel_address: Int64,
    *,
    offset: Constexpr[int],
    count: Constexpr[int],
) -> tuple[cutlass.Vector, cutlass.Vector | None]:
    """Return ``count`` ``sfV`` and ``v_mean`` values of consecutive channels.

    ``channel_address`` is the ``sfV`` word of the first channel the caller
    handles (``sage_v_section_address`` plus its channel); ``offset`` is this
    call's constant channel offset from it, so the loads take immediate
    offsets from one address. One float4 per load; the correction warps
    read the same words, so the loads are warp broadcasts. The means are
    ``None`` for a recipe without a V mean.
    """
    assert count % _FLOAT4_WORDS == 0
    mean_bytes = layout.v_rows.mean_word_offset * 4
    scale_values: tuple = ()
    mean_values: tuple = ()
    for channel in cutlass.range_constexpr(offset, offset + count, _FLOAT4_WORDS):
        scale_vec = _load_float4_at(channel_address + Int64(channel * 4))
        for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
            scale_values += (Float32(scale_vec[elem]),)
        if cutlass.const_expr(layout.v_rows.has_mean):
            mean_vec = _load_float4_at(
                channel_address + Int64(mean_bytes + channel * 4)
            )
            for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
                mean_values += (Float32(mean_vec[elem]),)
    scales = cutlass.Vector.from_elements(scale_values, Float32)
    if cutlass.const_expr(layout.v_rows.has_mean):
        return scales, cutlass.Vector.from_elements(mean_values, Float32)
    return scales, None
