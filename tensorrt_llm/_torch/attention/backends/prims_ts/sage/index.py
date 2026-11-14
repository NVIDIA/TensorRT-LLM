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

"""Device-side index math and loads over the Sage scale buffers.

The helpers turn coordinates into words of the flat scale tensors
(``config.sage_flat_scale_slot``) and of the prepared buffer
(``layout._SageScaleLayout``): a sequence's first word, a section's first
word and a Q tile row's word. They also load single FP32 scales. They hold
no resource or pipeline state; each attention kernel moves the words it
reads with its own resources.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .config import sage_flat_scale_slot
from .layout import _SageKSection, _SageQSection, _SageScaleLayout, _SageVSection


@cute.jit
def sage_sequence_words(
    layout: cutlass.Constexpr[_SageScaleLayout],
    *,
    batch_idx: Int32,
    kv_head_idx: Int32,
    num_kv_heads: Int32,
) -> Int32:
    """Return the first word of sequence ``b * Hkv + h`` in the prepared scales."""
    sequence_idx = batch_idx * num_kv_heads + kv_head_idx
    return sequence_idx * Int32(layout.sequence_words)


@cute.jit
def sage_section_word(
    sequence_words: Int32,
    section: cutlass.Constexpr[_SageKSection | _SageQSection | _SageVSection],
) -> Int32:
    """Return the first word of one section of a sequence's prepared scales."""
    return sequence_words + Int32(section.word_offset)


@cute.jit
def sage_q_word(
    layout: cutlass.Constexpr[_SageScaleLayout],
    *,
    sequence_words: Int32,
    q_tile_idx: Int32,
    row_idx: Int32,
) -> Int32:
    """Return the ``sfQ`` word of row ``row_idx`` of Q tile ``q_tile_idx``.

    ``q_tile_idx`` numbers the tiles as ``_SageQSection`` lays them out: the
    decode kernel's grouped Q CTA, or the context kernel's
    ``tile_index(local_head, token_tile)``.
    """
    q_rows = layout.q_rows
    return (
        sage_section_word(sequence_words, q_rows)
        + q_tile_idx * Int32(q_rows.rows_per_tile)
        + row_idx
    )


@cute.jit
def load_sage_q_scale(
    layout: cutlass.Constexpr[_SageScaleLayout],
    scales: cute.Pointer,
    *,
    sequence_words: Int32,
    q_tile_idx: Int32,
    row_idx: Int32,
) -> Float32:
    """Return ``sfQ`` of row ``row_idx`` of Q tile ``q_tile_idx``.

    ``sequence_words`` is the first word of the row's sequence
    (``sage_sequence_words``). The word already holds the scale of the row's
    token and Q head, the last token's for a row past the sequence's end
    (``_SageQSection``). A softmax lane reads it once per work tile.
    """
    word = sage_q_word(
        layout, sequence_words=sequence_words, q_tile_idx=q_tile_idx, row_idx=row_idx
    )
    return Float32(scales[word])


@cute.jit
def _load_sage_scale(scale_addr: Int64, index: Int32) -> Float32:
    """Return one FP32 scale loaded from a global memory base address.

    One 32-bit load per call; nothing widens it. The callers read single
    words: the prepare gathers each prepared word from its own flat source
    word, and the decode kernel stages a sequence's ``sfV`` row one channel
    per lane, so a warp's loads coalesce. ``sfK`` atoms move in 16-byte units
    instead.
    """
    value_ptr = cutlass.inttoptr(
        scale_addr + Int64(index) * Int64(4),
        mem_space=1,
        dtype=Float32,
    )
    return Float32(value_ptr.load(count=1, alignment=4)[0])


@cute.jit
def load_sage_flat_scale(
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
    slot = sage_flat_scale_slot(batch_idx, token_idx, seq_len, log2_block)
    return _load_sage_scale(scale_addr, head_idx * head_stride + slot)
