# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How many leading rows of a forward are real, for kernels that skip padding rows.

The requests that pad a batch to a captured CUDA graph's size and the placeholder
request of an idle attention-DP rank follow the real requests, so the real rows
lead every token-indexed tensor of a forward. Before each forward the engine writes
one 32-bit word in device memory, once a kernel that reads it has allocated it: bits
0-15 hold the real requests R, bits 16-31 the requests P of the forward. A kernel over
T rows keeps :func:`live_row_count` of them: none when P is 0, all when R >= P or P
does not divide T, else the leading T / P * R. Only the SM100/SM103 MegaMoE CuteDSL
kernel reads the word; EPLB statistics still count the rows it drops.

That rule holds for a launch over all rows of the forward. A MoE layer that splits a
forward into chunks launches over parts of it, so those launches read a word that keeps
every row (:func:`part_of_the_forward`). Under Helix CP with attention DP the attention
output is reduce-scattered across the CP group and each CP rank's MoE runs on its own
slice of the forward's rows, so no launch covers the forward: the engine's word is not
registered there (:func:`engine_word`) and every launch keeps every row. Setting
``TRTLLM_MEGAMOE_KEEP_PADDING_ROWS=1`` does the same everywhere: a debugging switch that turns
the drop off without a rebuild.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Sequence

import torch
from cuda.bindings import driver

from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.live_row_layout import (
    REQUEST_BITS,
    REQUEST_MASK,
)
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

# The most requests a word can describe.
MAX_REQUESTS = REQUEST_MASK
NO_ROWS = 0
ALL_ROWS = (1 << REQUEST_BITS) | 1
# Key of an engine's word in the per-forward model extra attrs.
LIVE_ROW_WORD_ATTR = "live_row_word"
# Debugging only: "1" registers no engine word, so the kernels keep every row.
KEEP_PADDING_ROWS_ENV = "TRTLLM_MEGAMOE_KEEP_PADDING_ROWS"


def encode(real_requests: int, requests: int) -> int:
    """The word of a forward over ``requests`` requests, the first ``real_requests`` of them real."""
    if not 0 <= real_requests <= requests <= MAX_REQUESTS:
        raise ValueError(f"no live-row word for {real_requests} real of {requests} requests")
    return (requests << REQUEST_BITS) | real_requests


def live_row_count(word: int, rows: int) -> int:
    """The leading rows of a launch over ``rows`` rows that ``word`` keeps (host reference).

    The kernels compute the same rule (``live_row_count`` in
    ``kernels/cutedsl_megamoe/helpers/live_rows.py``).
    """
    real, requests = word & REQUEST_MASK, (word >> REQUEST_BITS) & REQUEST_MASK
    if requests == 0:
        return 0
    return rows if real >= requests or rows % requests else rows // requests * real


def forward_word(requests: Sequence, *, num_contexts: int, seq_lens: Optional[torch.Tensor]) -> int:
    """The word of a forward over ``requests`` in row order, ``is_dummy`` marking padding.

    ``num_contexts`` and ``seq_lens`` are the forward's attention metadata: its
    context requests and the rows of each of its sequences. The padding after the last
    real request is dropped only when the forward holds generation requests alone and
    every sequence owns as many rows; a forward of padding only drops all of its rows,
    and any other forward keeps every row.
    """
    real = len(requests)
    while real > 0 and requests[real - 1].is_dummy:
        real -= 1
    if real == len(requests):
        return ALL_ROWS
    if real == 0:
        return NO_ROWS
    uniform = (
        num_contexts == 0
        and seq_lens is not None
        and seq_lens.numel() > 0
        and bool((seq_lens == seq_lens[0]).all())
    )
    return encode(real, len(requests)) if uniform else ALL_ROWS


class LiveRowWord:
    """An engine's word: allocated by the first kernel that reads it, then written in stream order.

    A write is a stream memory operation on the current stream, issued only when
    the value changes. A driver without stream memory operations leaves the word
    at ALL_ROWS, so every row is kept.
    """

    def __init__(self) -> None:
        self._value = ALL_ROWS
        self._tensor: Optional[torch.Tensor] = None
        self._stream_writes_supported = True

    @property
    def read_by_kernels(self) -> bool:
        """Whether a kernel has allocated the word and its writes reach the device.

        Until then a write would only be recorded, and the forward that allocates the word
        keeps every row, so the engine need not compute the word.
        """
        return self._tensor is not None and self._stream_writes_supported

    def tensor(self, device: torch.device) -> torch.Tensor:
        """The word's int32[1] device tensor on ``device``, allocated on the first call.

        The first call must run outside CUDA graph capture: it allocates the word holding the
        last value written, synchronizes the current stream so kernels on any stream read it,
        and allocates ``all_rows_word`` of the device too. Later calls return the same tensor
        and require the same device (a device without an index means the word's).
        """
        if self._tensor is not None:
            requested = torch.device(device)
            if requested.type != self._tensor.device.type or requested.index not in (
                None,
                self._tensor.device.index,
            ):
                raise ValueError(
                    f"the live-row word lives on {self._tensor.device}, not on {device}"
                )
            return self._tensor
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Run a forward eagerly before capturing it: it allocates the live-row word."
            )
        self._tensor = torch.full((1,), ALL_ROWS, dtype=torch.int32, device=device)
        value, self._value = self._value, ALL_ROWS
        self.write(value)
        # Kernels on any stream read it from here on.
        torch.cuda.current_stream(device).synchronize()
        # The word of launches over part of a forward (part_of_the_forward), allocated with the
        # engine's word, outside graph capture.
        all_rows_word(device)
        return self._tensor

    def write(self, value: int) -> None:
        """Set the word to ``value`` for the kernels launched next on the current stream.

        A stream memory write, issued only when the value changes; before a kernel has
        allocated the word it only records the value. Without stream memory operations
        the word stays at ALL_ROWS.
        """
        if value == self._value or not self._stream_writes_supported:
            return
        if self._tensor is not None:
            (error,) = driver.cuStreamWriteValue32(
                torch.cuda.current_stream(self._tensor.device).cuda_stream,
                self._tensor.data_ptr(),
                value,
                driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            if error == driver.CUresult.CUDA_ERROR_NOT_SUPPORTED:
                logger.warning_once(
                    f"{self._tensor.device} does not support stream memory operations "
                    "(cuStreamWriteValue32): the MegaMoE kernels compute the rows of padding "
                    "requests like real rows.",
                    key="live_row_word_no_stream_writes",
                )
                self._stream_writes_supported = False
                return
            if error != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuStreamWriteValue32 of the live-row word failed: {error!r}")
        self._value = value


def forward_rows_split_across_cp_ranks(mapping: Mapping) -> bool:
    """Whether each CP rank's token-indexed modules (MoE) run on a slice of the forward's rows.

    The condition under which the attention output projection reduce-scatters across the
    CP group (``_helix_cp_output_projection`` in ``attention/attention.py``). Other CP types
    keep the forward's rows on every CP rank: Ulysses folds CP into attention TP, and the
    MoE token counts of attention DP are then not divided by the CP size.
    """
    return mapping.has_cp_helix() and mapping.enable_attention_dp


def engine_word(model: torch.nn.Module, mapping: Mapping) -> LiveRowWord:
    """A new engine's word, registered in ``model.extra_attrs`` for the kernels to read.

    Where the forward's rows are split across CP ranks, or with ``KEEP_PADDING_ROWS_ENV`` set
    to 1, the word is not registered, so every launch reads ``all_rows_word`` and the engine's
    writes never reach a kernel.
    """
    word = LiveRowWord()
    if os.environ.get(KEEP_PADDING_ROWS_ENV, "0") == "1":
        logger.warning_once(
            f"{KEEP_PADDING_ROWS_ENV}=1: the MegaMoE kernels compute the rows of padding requests "
            "like real rows (a debugging switch).",
            key="live_row_word_keep_padding_rows",
        )
    elif not forward_rows_split_across_cp_ranks(mapping):
        model.extra_attrs[LIVE_ROW_WORD_ATTR] = word
    return word


_ALL_ROWS_WORDS: Dict[Optional[int], LiveRowWord] = {}
_PART = threading.local()


@contextlib.contextmanager
def part_of_the_forward() -> Iterator[None]:
    """Kernels launched inside cover only part of the forward's rows (one chunk of a MoE layer):
    the forward's word describes all of its rows, so they read ``all_rows_word`` instead."""
    depth = getattr(_PART, "depth", 0)
    _PART.depth = depth + 1
    try:
        yield
    finally:
        _PART.depth = depth


def all_rows_word(device: torch.device) -> torch.Tensor:
    """A word of ``device`` that is never written: it keeps every row."""
    index = torch.device(device).index
    word = _ALL_ROWS_WORDS.get(index)
    if word is None:
        word = _ALL_ROWS_WORDS[index] = LiveRowWord()
    return word.tensor(device)


def current_word(device: torch.device) -> torch.Tensor:
    """The word a kernel launched now on ``device`` reads: the running engine forward's for a launch
    over all of its rows, else ``all_rows_word``."""
    attrs = get_model_extra_attrs()
    word = attrs.get(LIVE_ROW_WORD_ATTR) if attrs else None
    if word is None or getattr(_PART, "depth", 0):
        return all_rows_word(device)
    return word.tensor(device)
