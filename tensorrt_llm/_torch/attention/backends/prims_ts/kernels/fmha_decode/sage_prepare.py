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

"""Prepare the Sage scales of a dense or block-sparse plan ahead of attention.

The plan's scales are gathered into one buffer in the order the attention
kernel reads them (``_SageScaleLayout``): per sequence (batch entry and KV
head) the ``sfK`` atoms of the K tokens and of a proxy plan's block
summaries, the ``sfQ`` rows of the Q tiles, then the ``sfV`` row and, with a
V mean, the ``v_mean`` row. This kernel reads the flat ``q_scale``,
``k_scale`` and ``k_summary_scale`` tensors and the ``[B, Hkv, D]``
``v_scale`` and ``v_mean`` tensors; one thread writes one float4 of the
buffer.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv

from flashinfer.utils import ceil_div

from ...sage import (
    _FLOAT4_WORDS,
    _SAGE_ATOM_TOKENS,
    _SageKSection,
    _SageScaleLayout,
    log2_block_size,
)
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_resources.helpers_common import (
    _q_row_token_and_local_head,
    _q_tile_valid_rows_for_seq,
)
from .fmha_decode_resources.sage_scales import load_flat_scale

_THREADS_PER_CTA = 128


@dataclass
class _KSectionSource:
    """One K section's atom geometry and source scales, as a float4 writer reads them.

    The dataclass is not frozen because the writer selects one of the two
    sections inside a traced branch, where the DSL rebuilds a frozen
    dataclass as a proxy.
    """

    # Float4s of one atom.
    atom_float4s: cutlass.Int32
    # Tokens from one word's token to the next word's.
    tokens_per_word: cutlass.Int32
    # Words of an atom that hold scales; the rest are zero padding.
    words_per_atom: cutlass.Int32
    # log2 of the source tensor's block size.
    log2_block: cutlass.Int32
    # Base address and per-head slot count of the flat source tensor.
    scale_addr: cutlass.Int64
    head_stride: cutlass.Int32
    # Tokens of the section: K tokens, or block summaries.
    length: cutlass.Int32

    @staticmethod
    def create(section: _SageKSection, scales: cute.Tensor) -> "_KSectionSource":
        return _KSectionSource(
            atom_float4s=cutlass.Int32(section.atom_float4s),
            tokens_per_word=cutlass.Int32(section.tokens_per_word),
            words_per_atom=cutlass.Int32(section.words_per_atom),
            log2_block=cutlass.Int32(log2_block_size(section.source_block_size)),
            scale_addr=scales.iterator.toint(),
            head_stride=cutlass.Int32(scales.shape[1]),
            length=cutlass.Int32(section.length),
        )


@cute.jit
def _store_float4(
    scales: cute.Tensor, float4_idx: cutlass.Int32, words: cutlass.Array
) -> None:
    """Store four words as float4 ``float4_idx`` of the prepared scales."""

    float4_ptr = cutlass.inttoptr(
        scales.iterator.toint()
        + cutlass.Int64(float4_idx) * cutlass.Int64(_FLOAT4_WORDS * 4),
        mem_space=1,
        dtype=cutlass.Float32,
    )
    float4_ptr.store(
        words.data_ptr().load(count=_FLOAT4_WORDS, alignment=4),
        alignment=_FLOAT4_WORDS * 4,
    )


@cute.jit
def _write_v_float4(
    v_scale: cute.Tensor,
    v_mean: cute.Tensor | None,
    scales: cute.Tensor,
    float4_idx: cutlass.Int32,
    layout: cutlass.Constexpr[_SageScaleLayout],
    *,
    sequence_idx: cutlass.Int32,
    float4_in_section: cutlass.Int32,
) -> None:
    """Write float4 ``float4_idx`` of the buffer: four channels of one V row.

    ``float4_in_section`` places the float4 in the sequence's V rows: the
    ``sfV`` row, copied from the sequence's row of ``v_scale``, or the
    ``v_mean`` row behind it, copied from ``v_mean``.
    """

    v_rows = layout.v_rows
    row_float4s = cutlass.Int32(v_rows.headdim // _FLOAT4_WORDS)
    source_addr = v_scale.iterator.toint()
    float4_in_row = float4_in_section
    if cutlass.const_expr(v_rows.has_mean):
        if float4_in_section >= row_float4s:
            source_addr = v_mean.iterator.toint()
            float4_in_row = float4_in_section - row_float4s
    source_ptr = cutlass.inttoptr(
        source_addr
        + cutlass.Int64(sequence_idx * row_float4s + float4_in_row)
        * cutlass.Int64(_FLOAT4_WORDS * 4),
        mem_space=1,
        dtype=cutlass.Float32,
    )
    words = cutlass.Array(
        cutlass.Float32, _FLOAT4_WORDS, space=cutlass.AddressSpace.rmem
    )
    loaded = source_ptr.load(count=_FLOAT4_WORDS, alignment=_FLOAT4_WORDS * 4)
    for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
        words[elem] = cutlass.Float32(loaded[elem])
    _store_float4(scales, float4_idx, words)


@cute.jit
def _write_k_float4(
    k_scale: cute.Tensor,
    k_summary_scale: cute.Tensor | None,
    scales: cute.Tensor,
    float4_idx: cutlass.Int32,
    layout: cutlass.Constexpr[_SageScaleLayout],
    *,
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    float4_in_sequence: cutlass.Int32,
) -> None:
    """Write float4 ``float4_idx`` of the buffer: four words of one K atom.

    ``float4_in_sequence`` places the float4 in the sequence's K sections:
    the K tokens from ``k_scale``, or the block summaries from
    ``k_summary_scale`` behind them. Word ``w`` of an atom is the section's
    scale of the atom's token ``w * tokens_per_word``; a token past the
    section's length takes the last scale, and padding words are zero.
    """

    section = _KSectionSource.create(layout.k_tokens, k_scale)
    float4_in_section = float4_in_sequence
    if cutlass.const_expr(layout.summaries is not None):
        summaries_first_float4 = layout.summaries.word_offset // _FLOAT4_WORDS
        if float4_in_sequence >= cutlass.Int32(summaries_first_float4):
            section = _KSectionSource.create(layout.summaries, k_summary_scale)
            float4_in_section = float4_in_sequence - cutlass.Int32(
                summaries_first_float4
            )
    atom_idx = float4_in_section // section.atom_float4s
    first_word = (float4_in_section - atom_idx * section.atom_float4s) * cutlass.Int32(
        _FLOAT4_WORDS
    )

    words = cutlass.Array(
        cutlass.Float32, _FLOAT4_WORDS, space=cutlass.AddressSpace.rmem
    )
    for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
        word = first_word + cutlass.Int32(elem)
        words[elem] = cutlass.Float32(0.0)
        # Only a two-word atom pads, and it is a single float4.
        if word < section.words_per_atom:
            words[elem] = load_flat_scale(
                section.scale_addr,
                section.head_stride,
                head_idx=head_idx,
                batch_idx=batch_idx,
                seq_len=section.length,
                token_idx=atom_idx * cutlass.Int32(_SAGE_ATOM_TOKENS)
                + word * section.tokens_per_word,
                log2_block=section.log2_block,
            )
    _store_float4(scales, float4_idx, words)


@cute.jit
def _write_q_float4(
    q_scale: cute.Tensor,
    scales: cute.Tensor,
    float4_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    layout: cutlass.Constexpr[_SageScaleLayout],
    *,
    batch_idx: cutlass.Int32,
    kv_head_idx: cutlass.Int32,
    float4_in_section: cutlass.Int32,
) -> None:
    """Write float4 ``float4_idx`` of the buffer: four rows of one Q tile.

    ``float4_in_section`` places the float4 in the sequence's Q rows. Row
    ``r`` of the tile takes the Q scale of the token and Q head the softmax
    maps the row to (``_q_row_token_and_local_head``); a row past the
    tile's valid rows (``_q_tile_valid_rows_for_seq``) takes the last valid
    row's.
    """

    q_rows = layout.q_rows
    tile_float4s = cutlass.Int32(q_rows.tile_size_q // _FLOAT4_WORDS)
    tile_idx = float4_in_section // tile_float4s
    first_row = (float4_in_section - tile_idx * tile_float4s) * cutlass.Int32(
        _FLOAT4_WORDS
    )
    # The row helpers take the plan's heads per KV head as the packed row
    # count and its fixed Q length; Sage plans have no variable Q lengths.
    heads_q_per_kv = cutlass.Int32(q_rows.heads_q_per_kv)
    seq_len_q = cutlass.Int32(q_rows.seq_len_q)
    last_row = _q_tile_valid_rows_for_seq(
        cfg, heads_q_per_kv, tile_idx, seq_len_q
    ) - cutlass.Int32(1)

    words = cutlass.Array(
        cutlass.Float32, _FLOAT4_WORDS, space=cutlass.AddressSpace.rmem
    )
    for elem in cutlass.range_constexpr(_FLOAT4_WORDS):
        row = cute.math.min(first_row + cutlass.Int32(elem), last_row)
        q_token_idx, local_head_idx = _q_row_token_and_local_head(
            cfg, heads_q_per_kv, tile_idx, row
        )
        words[elem] = load_flat_scale(
            q_scale.iterator.toint(),
            cutlass.Int32(q_scale.shape[1]),
            head_idx=kv_head_idx * heads_q_per_kv + local_head_idx,
            batch_idx=batch_idx,
            seq_len=seq_len_q,
            token_idx=q_token_idx,
            log2_block=cutlass.Int32(log2_block_size(q_rows.q_block_size)),
        )
    _store_float4(scales, float4_idx, words)


class _PrepareSageScales:
    """Write a Sage plan's prepared scales from its flat scale tensors.

    The buffer covers ``batch_size * num_kv_heads`` sequences
    (``_SageScaleLayout``); the grid gives each of its float4s one thread. The kernel is an ordinary launch on the plan's
    stream. A dense plan's attention launch follows it in stream order; a
    block-sparse plan's route prepare follows it as a programmatic dependent
    launch that acquires this grid before exiting, and attention follows the
    route prepare in stream order.
    """

    def __init__(
        self,
        *,
        cfg: FmhaDecodeConfig,
        layout: _SageScaleLayout,
        batch_size: int,
        num_kv_heads: int,
    ) -> None:
        self.cfg = cfg
        self.layout = layout
        self.num_kv_heads = num_kv_heads
        num_sequences = batch_size * num_kv_heads
        self.sequence_float4s = layout.sequence_words // _FLOAT4_WORDS
        self.num_float4s = layout.total_words(num_sequences) // _FLOAT4_WORDS

    @cute.jit
    def __call__(
        self,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        k_summary_scale: cute.Tensor | None,
        v_scale: cute.Tensor,
        v_mean: cute.Tensor | None,
        scales: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.kernel(q_scale, k_scale, k_summary_scale, v_scale, v_mean, scales).launch(
            grid=[ceil_div(self.num_float4s, _THREADS_PER_CTA), 1, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        k_summary_scale: cute.Tensor | None,
        v_scale: cute.Tensor,
        v_mean: cute.Tensor | None,
        scales: cute.Tensor,
    ) -> None:
        """Write this thread's float4 of a K section, the Q rows or the V rows, if any."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        float4_idx = block_idx * cutlass.Int32(_THREADS_PER_CTA) + thread_idx
        if float4_idx < cutlass.Int32(self.num_float4s):
            sequence_idx = float4_idx // cutlass.Int32(self.sequence_float4s)
            float4_in_sequence = float4_idx - sequence_idx * cutlass.Int32(
                self.sequence_float4s
            )
            batch_idx = sequence_idx // cutlass.Int32(self.num_kv_heads)
            head_idx = sequence_idx - batch_idx * cutlass.Int32(self.num_kv_heads)
            q_rows_first_float4 = cutlass.Int32(
                self.layout.q_rows.word_offset // _FLOAT4_WORDS
            )
            v_rows_first_float4 = cutlass.Int32(
                self.layout.v_rows.word_offset // _FLOAT4_WORDS
            )
            if float4_in_sequence >= v_rows_first_float4:
                _write_v_float4(
                    v_scale,
                    v_mean,
                    scales,
                    float4_idx,
                    self.layout,
                    sequence_idx=sequence_idx,
                    float4_in_section=float4_in_sequence - v_rows_first_float4,
                )
            elif float4_in_sequence < q_rows_first_float4:
                _write_k_float4(
                    k_scale,
                    k_summary_scale,
                    scales,
                    float4_idx,
                    self.layout,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    float4_in_sequence=float4_in_sequence,
                )
            else:
                _write_q_float4(
                    q_scale,
                    scales,
                    float4_idx,
                    self.cfg,
                    self.layout,
                    batch_idx=batch_idx,
                    kv_head_idx=head_idx,
                    float4_in_section=float4_in_sequence - q_rows_first_float4,
                )


__all__ = ["_PrepareSageScales"]
