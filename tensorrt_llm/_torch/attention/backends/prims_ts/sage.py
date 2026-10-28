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

"""Sage attention inputs for the PrimTS decode kernels and the layout of their prepared scales.

Sage attention runs ``QK^T`` on 8-bit Q/K with one dequantization scale per
token block and ``PV`` on E4M3 P/V with one scale per V channel. Q/K scales use
the trtllm-gen flat layout of TensorRT-LLM's ``sageQuant``: per head, sequence
``b`` starts at slot ``b * S // blk + b`` and token ``t`` uses slot
``t // blk`` inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch

from flashinfer.utils import ceil_div, round_up

SAGE_K_BLOCK_SIZES = (1, 4, 16, 32, 64, 128, 256)
# Sage scale slots of the contiguous attention adapter, in ABI order.
SAGE_ADAPTER_SLOTS = ("q_scale", "k_scale", "k_summary_scale", "v_scale", "v_mean")

_T = TypeVar("_T")


@dataclass(frozen=True)
class SageAttentionConfig:
    """Compile-time Sage attention recipe of one plan.

    ``q_block_size`` tokens share one Q scale and must be a power of two no
    larger than the planned Q tile. ``k_block_size`` tokens share one K scale;
    ``k_summary_block_size`` is the K block size of a proxy plan's summary
    scales; construction sets it to ``k_block_size`` when it is omitted.
    Both must be one of ``SAGE_K_BLOCK_SIZES``. ``v_mean`` says whether every
    run supplies a per-channel V mean that is added back to the output. The
    defaults are TensorRT-LLM's production recipe.
    """

    q_block_size: int = 1
    k_block_size: int = 16
    v_mean: bool = False
    k_summary_block_size: int | None = None

    def __post_init__(self) -> None:
        if self.k_summary_block_size is None:
            object.__setattr__(self, "k_summary_block_size", self.k_block_size)


@dataclass(frozen=True)
class SageAttentionParams:
    """Per-block Q/K scales and per-channel V scales of one run.

    All tensors are FP32 with the block sizes of the plan's
    :class:`SageAttentionConfig`: ``q_scale`` is
    ``[Hq, flat_scale_numel(B, Sq, q_block_size)]``, ``k_scale`` is
    ``[Hkv, flat_scale_numel(B, Skv, k_block_size)]``, and ``v_scale`` and
    ``v_mean`` are ``[B, Hkv, D]``, one per-channel row per batch entry and
    KV head. ``k_summary_scale`` holds the scales of the
    K summaries of a proxy plan,
    ``[Hkv, flat_scale_numel(B, num_kv_blocks, k_summary_block_size)]``.
    ``v_mean`` and ``k_summary_scale`` are present exactly when the plan uses
    them. Scales must be positive and finite; the kernel does not check them.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_summary_scale: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None


def flat_scale_numel(batch_size: int, seq_len: int, block_size: int) -> int:
    """Return the per-head slot count of the flat scale layout."""
    return ceil_div(batch_size * seq_len, block_size) + batch_size - 1


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def log2_block_size(block_size: int) -> int:
    """Return ``log2`` of a power-of-two scale block size."""
    if not is_power_of_two(block_size):
        raise ValueError(f"block size must be a power of two, got {block_size}")
    return block_size.bit_length() - 1


def sage_adapter_slots(values: Mapping[str, _T | None]) -> tuple[_T | None, ...]:
    """Arrange Sage values named by scale into the adapter slots.

    An absent name leaves its slot ``None``, which the adapter binds as a null
    pointer.
    """
    return tuple(values.get(name) for name in SAGE_ADAPTER_SLOTS)


def flat_scale_slot(batch_idx, token_idx, seq_len, log2_block: int):
    """Return the per-head slot of one token in the flat scale layout.

    Sequence ``b`` starts at ``b * S // blk + b``; the extra ``b`` keeps its
    last block apart from the next sequence's first block when ``blk`` does
    not divide ``S``. The arithmetic serves host and device ``Int32`` values.
    """
    return ((batch_idx * seq_len) >> log2_block) + batch_idx + (token_idx >> log2_block)


def _validate_scale_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32, got {tensor.dtype}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def sage_scale_shapes(
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    summary_seq_len: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of every scale tensor a plan consumes, by field name.

    ``summary_seq_len`` is the KV block count of a proxy plan and ``None``
    otherwise.
    """
    shapes = {
        "q_scale": (
            num_qo_heads,
            flat_scale_numel(batch_size, seq_len_q, config.q_block_size),
        ),
        "k_scale": (
            num_kv_heads,
            flat_scale_numel(batch_size, seq_len_kv, config.k_block_size),
        ),
        "v_scale": (batch_size, num_kv_heads, head_dim),
    }
    if summary_seq_len is not None:
        shapes["k_summary_scale"] = (
            num_kv_heads,
            flat_scale_numel(batch_size, summary_seq_len, config.k_summary_block_size),
        )
    if config.v_mean:
        shapes["v_mean"] = (batch_size, num_kv_heads, head_dim)
    return shapes


def validate_sage_params(
    params: SageAttentionParams,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    device: torch.device,
) -> None:
    """Validate the scale tensors of one run against the plan's expected shapes.

    ``expected_shapes`` is the plan's ``sage_scale_shapes`` result; an
    optional tensor is required exactly when its name is present.
    """
    if not isinstance(params, SageAttentionParams):
        raise TypeError("sage must be a SageAttentionParams instance")
    for name, consumer in (
        ("v_mean", "a plan configured with v_mean=True"),
        ("k_summary_scale", "block-sparse proxy routes"),
    ):
        if (getattr(params, name) is None) == (name in expected_shapes):
            raise ValueError(f"{name} is required by {consumer} and rejected otherwise")
    for name, expected_shape in expected_shapes.items():
        _validate_scale_tensor(
            getattr(params, name), name, expected_shape=expected_shape, device=device
        )


_FLOAT4_WORDS = 4
# A Sage K-scale atom is one Keeps layout atom of 64 K tokens, read one word
# per 32-token score fragment at least, so it holds at least two words. The
# prepared scales are written and copied in float4 units, the width of one
# ``cp.async``.
_SAGE_ATOM_TOKENS = 64
_SAGE_ATOM_MIN_WORDS = 2


def _validate_i32_address(value: int, name: str) -> int:
    """Reject a buffer extent device Int32 addressing cannot hold."""

    if value > 2**31 - 1:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


@dataclass(frozen=True)
class _SageKSection:
    """The ``sfK`` words of one route kind: one atom per 64 tokens.

    An atom covers 64 tokens (two K32 score fragments) and holds
    ``words_per_atom = max(2, 64 / block_size)`` scales, word ``w`` being the
    scale of the atom's token ``w * tokens_per_word``, padded with zeros to
    whole float4s. Scales come from the flat source tensor of one scale per
    ``source_block_size`` tokens; a coarser source repeats its value, and a
    token past ``length`` takes the last scale.
    """

    length: int
    block_size: int
    source_block_size: int
    word_offset: int

    @property
    def words_per_atom(self) -> int:
        """Return the words of one atom that hold scales: one per block, at least two."""

        return max(_SAGE_ATOM_MIN_WORDS, _SAGE_ATOM_TOKENS // self.block_size)

    @property
    def tokens_per_word(self) -> int:
        """Return the tokens from one word's token to the next word's."""

        return _SAGE_ATOM_TOKENS // self.words_per_atom

    @property
    def atom_words(self) -> int:
        """Return the words of one atom: ``words_per_atom`` padded to whole float4s."""

        return round_up(self.words_per_atom, _FLOAT4_WORDS)

    @property
    def atom_float4s(self) -> int:
        """Return the float4s of one atom."""

        return self.atom_words // _FLOAT4_WORDS

    @property
    def atoms(self) -> int:
        """Return the section's 64-token atoms, the last one possibly partial."""

        return ceil_div(self.length, _SAGE_ATOM_TOKENS)

    @property
    def words(self) -> int:
        """Return the words of the section's atoms."""

        return self.atoms * self.atom_words


@dataclass(frozen=True)
class _SageQSection:
    """The ``sfQ`` words of a sequence's Q tiles: one word per tile row.

    Tile ``t`` holds ``tile_size_q`` words from word ``word_offset + t *
    tile_size_q``, in the order the softmax lanes read them: row ``r`` is
    token ``t * tokens_per_tile + r // heads_q_per_kv`` of local Q head ``r %
    heads_q_per_kv``, and its word is that head's scale of the token's
    ``q_block_size`` block. Rows past the sequence's last token repeat the
    last valid word, so a masked row keeps a finite positive scale.
    """

    seq_len_q: int
    q_block_size: int
    heads_q_per_kv: int
    tile_size_q: int
    word_offset: int

    def __post_init__(self) -> None:
        if self.tile_size_q % _FLOAT4_WORDS or self.heads_q_per_kv > self.tile_size_q:
            raise ValueError("sfQ tiles hold whole float4s of at least one token")

    @property
    def tokens_per_tile(self) -> int:
        """Return the complete Q tokens of one tile."""

        return self.tile_size_q // self.heads_q_per_kv

    @property
    def tiles(self) -> int:
        """Return the Q tiles of one sequence, the last one possibly partial."""

        return ceil_div(self.seq_len_q, self.tokens_per_tile)

    def valid_rows(self, tile_idx: int) -> int:
        """Return the rows of tile ``tile_idx`` that hold a token of the sequence."""

        remaining = self.seq_len_q - tile_idx * self.tokens_per_tile
        return min(remaining, self.tokens_per_tile) * self.heads_q_per_kv

    @property
    def words(self) -> int:
        """Return the words of the section: ``tile_size_q`` per tile."""

        return self.tiles * self.tile_size_q


@dataclass(frozen=True)
class _SageVSection:
    """The ``sfV`` row and, with a V mean, the ``v_mean`` row of one sequence.

    Row ``r`` holds channel ``c`` at word ``word_offset + r * headdim + c``
    of the sequence, copied from the sequence's rows of the ``[B, Hkv, D]``
    tensors as they are.
    """

    headdim: int
    has_mean: bool
    word_offset: int

    def __post_init__(self) -> None:
        if self.headdim % _FLOAT4_WORDS:
            raise ValueError("sfV rows hold whole float4s")

    @property
    def rows(self) -> int:
        """Return the rows of one head: the scales, then the means."""

        return 2 if self.has_mean else 1

    @property
    def mean_word_offset(self) -> int:
        """Return the first word of the mean row within the section."""

        return self.headdim

    @property
    def words(self) -> int:
        """Return the words of the section's rows."""

        return self.rows * self.headdim


@dataclass(frozen=True)
class _SageScaleLayout:
    """Word layout of the FP32 scale buffer a Sage plan prepares before attention.

    The buffer is one block of ``sequence_words`` words per sequence ``s = b
    * Hkv + h`` (KV head ``h`` of batch entry ``b``), each holding four
    sections::

        | k_tokens | summaries (proxy plans only) | q_rows | v_rows |

    so any scale of a sequence is ``s * sequence_words + section.word_offset
    + offset within the section``.

    ``k_tokens`` and ``summaries`` are ``_SageKSection`` atoms of the block
    size each route kind reads (``FmhaDecodeConfig.sage_k_scale_block_size``):
    16-token K blocks give four words per atom (tokens 0, 16, 32, 48),
    one-token scales 64, blocks of 64 tokens or more the two-word floor (one
    per fragment) padded to a float4. ``q_rows`` is a ``_SageQSection`` and
    ``v_rows`` a ``_SageVSection``. Every section starts on a float4 and K
    atoms are whole float4s because the attention load warp copies them with
    16-byte ``cp.async``.
    """

    seq_len_kv: int
    k_block_size: int
    # Summary sequence length of a proxy plan, else zero.
    num_summaries: int
    k_summary_block_size: int
    k_tokens_block_size: int
    summaries_block_size: int
    seq_len_q: int
    q_block_size: int
    heads_q_per_kv: int
    tile_size_q: int
    headdim: int
    v_mean: bool

    @property
    def k_tokens(self) -> _SageKSection:
        """Return the K-token section, which starts a sequence's words."""

        return _SageKSection(
            length=self.seq_len_kv,
            block_size=self.k_tokens_block_size,
            source_block_size=self.k_block_size,
            word_offset=0,
        )

    @property
    def summaries(self) -> _SageKSection | None:
        """Return a proxy plan's summary section, which follows the K tokens."""

        if self.num_summaries == 0:
            return None
        return _SageKSection(
            length=self.num_summaries,
            block_size=self.summaries_block_size,
            source_block_size=self.k_summary_block_size,
            word_offset=self.k_tokens.words,
        )

    @property
    def k_sections(self) -> tuple[_SageKSection, ...]:
        """Return the K sections in order, paired with ``k_scale`` then ``k_summary_scale``."""

        summaries = self.summaries
        return (self.k_tokens,) if summaries is None else (self.k_tokens, summaries)

    @property
    def q_rows(self) -> _SageQSection:
        """Return the Q section, which follows the K sections."""

        return _SageQSection(
            seq_len_q=self.seq_len_q,
            q_block_size=self.q_block_size,
            heads_q_per_kv=self.heads_q_per_kv,
            tile_size_q=self.tile_size_q,
            word_offset=sum(section.words for section in self.k_sections),
        )

    @property
    def v_rows(self) -> _SageVSection:
        """Return the V section, which follows the Q rows."""

        q_rows = self.q_rows
        return _SageVSection(
            headdim=self.headdim,
            has_mean=self.v_mean,
            word_offset=q_rows.word_offset + q_rows.words,
        )

    @property
    def sequence_words(self) -> int:
        """Return the words of one sequence: K sections, Q rows, then V rows."""

        v_rows = self.v_rows
        return v_rows.word_offset + v_rows.words

    def total_words(self, num_sequences: int) -> int:
        """Return the words of the buffer for ``batch * Hkv`` sequences."""

        return _validate_i32_address(
            num_sequences * self.sequence_words, "sage_scales_words"
        )


__all__ = [
    "SAGE_ADAPTER_SLOTS",
    "SAGE_K_BLOCK_SIZES",
    "SageAttentionConfig",
    "SageAttentionParams",
    "flat_scale_numel",
    "flat_scale_slot",
    "is_power_of_two",
    "log2_block_size",
    "sage_adapter_slots",
    "sage_scale_shapes",
    "validate_sage_params",
]
