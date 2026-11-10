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
"""Greedy draft tokens of a BF16 (or FP16) vocabulary-sharded LM head without materializing its logits.

``BF16DraftHead.sample`` runs ``trtllm::lm_head_argmax``: one kernel streams this rank's BF16 / FP16 shard
of the LM head, folds every logit into a per-row argmax key in registers and agrees on the global
argmax with the other TP ranks through the TP key exchange's protocol on one of its channels
(``ll_exchange.TPKeyExchange.channel_protocol_args``: same peer buffers, no extra peer memory). The
tokens equal those of the GEMM + shard argmax + key exchange path: each logit is rounded to the
dtype the GEMM would store before the comparison, and equal maxima go to the lowest global id.

The kernel needs SM 90+ and takes up to 8 rows (fewer for very large hidden sizes); the hidden size
must be a multiple of 256. ``build_bf16_draft_head`` returns None elsewhere and callers keep the
GEMM path. ``TLLM_SPEC_FUSED_DRAFT_HEAD=0`` turns the head off.
"""

from __future__ import annotations

import functools
import os
from typing import List, Optional

import torch

from tensorrt_llm.logger import logger

# Rows one launch can take (the kernel's kMaxRows); the device and hidden size may allow fewer.
MAX_ROWS = 8
# int64 words of the per-head scratch (the kernel's kScratchWords: 256 CTAs x 8 rows + a counter).
SCRATCH_WORDS = 256 * MAX_ROWS + 1
# Key-exchange channel of the draft-side greedy exchanges (shared with TPKeyExchange.argmax_logits
# and the MXFP8 draft head, which run on the same stream in the same order on every rank).
KEY_EXCHANGE_CHANNEL = 0
FUSED_DRAFT_HEAD_ENV_VAR = "TLLM_SPEC_FUSED_DRAFT_HEAD"
# Smallest weight shard the fused head takes. Below it the kernel's fixed cost (its last-CTA reduction
# and exchange) outweighs what it saves over the GEMM + argmax-exchange launches (measured on B200).
MIN_SHARD_BYTES = 96 << 20


def fused_draft_head_enabled() -> bool:
    """Whether the stock one-model speculative workers may use ``BF16DraftHead``
    (``TLLM_SPEC_FUSED_DRAFT_HEAD``, default on)."""
    return os.environ.get(FUSED_DRAFT_HEAD_ENV_VAR, "1").strip() != "0"


@functools.lru_cache(maxsize=None)
def max_rows(hidden_size: int, device_index: int) -> int:
    """Rows one ``trtllm::lm_head_argmax`` launch takes for ``hidden_size`` on the device; 0 when the
    op is not built, the device is older than SM 90 or the hidden size is not supported."""
    if not hasattr(torch.ops.trtllm, "lm_head_argmax_max_rows"):
        return 0
    with torch.cuda.device(device_index):
        return int(torch.ops.trtllm.lm_head_argmax_max_rows(hidden_size))


def is_plain_mtp_shared_head(shared_head: torch.nn.Module) -> bool:
    """Whether ``shared_head(x, lm_head, attn_metadata, True)`` is just this rank's vocab shard of
    ``lm_head(x)`` outside attention DP (``DeepseekV3MTPHead`` and subclasses that keep its forward;
    heads that normalize or transform ``x`` do not qualify)."""
    from ..models.modeling_deepseekv3 import DeepseekV3MTPHead

    return type(shared_head).forward is DeepseekV3MTPHead.forward


class BF16DraftHead:
    """Fused greedy head over a BF16 (or FP16) vocab-sharded ``LMHead`` (same interface as
    ``MXFP8DraftHead``). Reads the head's current weight on every call, so weight reloads apply.

    Args:
        lm_head: the vocab-sharded BF16 / FP16 ``LMHead`` (``vocab_parallel.is_vocab_sharded_lm_head``) or an
            unsharded one.
        key_exchange: the TP group's ``ll_exchange.TPKeyExchange`` (None for a single rank).
        rows: the most rows one call takes (``max_rows``).
    """

    def __init__(self, lm_head, key_exchange, rows: int) -> None:
        weight = lm_head.weight
        self.lm_head = lm_head
        self.hidden_size = weight.shape[1]
        self.max_rows = rows
        tp_size = lm_head.tp_size
        self.rank = lm_head.tp_rank if tp_size > 1 else 0
        self.vocab_offset = self.rank * lm_head.out_features
        # Compare what the GEMM path would store: logits in the weight dtype unless the head
        # writes fp32.
        self.round_logits = getattr(lm_head, "output_dtype", None) in (None, weight.dtype)
        self.scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device=weight.device)
        self.exchange_epoch: Optional[torch.Tensor] = None
        self.peer_buffers: List[int] = []
        self.exchange_words = 0
        if tp_size > 1:
            self.exchange_epoch, self.peer_buffers, self.exchange_words = (
                key_exchange.channel_protocol_args(KEY_EXCHANGE_CHANNEL)
            )
        logger.info(
            f"BF16 fused draft head ({weight.dtype}): shard {lm_head.out_features} x {self.hidden_size} "
            f"(offset {self.vocab_offset}, tp {tp_size}), up to {rows} rows per launch"
        )

    def supports(self, rows: int) -> bool:
        return 1 <= rows <= self.max_rows

    def sample(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy token ids ``[M]`` int32 (global vocabulary of the head), identical on every TP
        rank, of ``x`` ``[M, hidden]`` in the head's dtype; written into ``out`` (contiguous int32 ``[M]``) when
        given. Every TP rank must call it with the same ``M``."""
        if x.stride(-1) != 1 or x.data_ptr() % 16 or (x.stride(0) * x.element_size()) % 16:
            x = x.contiguous()
        if out is None:
            out = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
        torch.ops.trtllm.lm_head_argmax(
            x,
            self.lm_head.weight,
            self.vocab_offset,
            self.round_logits,
            self.scratch,
            self.exchange_epoch,
            self.peer_buffers,
            self.exchange_words,
            self.rank,
            out,
        )
        return out


def build_bf16_draft_head(lm_head, key_exchange) -> Optional[BF16DraftHead]:
    """``BF16DraftHead`` for ``lm_head`` when the op, the device and the head allow it; otherwise
    None (logged), and the caller keeps the GEMM path. Every TP rank reaches the same decision
    (configuration and device type only); nothing here is collective.

    Args:
        lm_head: the draft step's LM head: a vocab-sharded BF16 / FP16 ``LMHead`` under TP.
        key_exchange: the TP group's ``ll_exchange.TPKeyExchange``; required when TP > 1.
    """
    if not fused_draft_head_enabled():
        return None
    weight = getattr(lm_head, "weight", None)
    reason = None
    if weight is None or weight.dim() != 2 or not weight.is_cuda:
        reason = "the LM head has no 2D CUDA weight"
    elif getattr(lm_head, "has_any_quant", False) or weight.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        reason = "the LM head is not BF16 or FP16"
    elif getattr(lm_head, "bias", None) is not None or weight.stride(-1) != 1:
        reason = "the LM head has a bias or a strided weight"
    elif lm_head.tp_size > 1 and (key_exchange is None or key_exchange.nranks != lm_head.tp_size):
        reason = "no TP key exchange for the LM head's group"
    elif weight.numel() * weight.element_size() < MIN_SHARD_BYTES:
        reason = (
            f"the shard ({weight.numel() * weight.element_size() >> 20} MiB) is below "
            f"{MIN_SHARD_BYTES >> 20} MiB, where the GEMM path is faster"
        )
    elif max_rows(weight.shape[1], weight.device.index) == 0:
        reason = (
            f"trtllm::lm_head_argmax unavailable for hidden size {weight.shape[1]} on this "
            "device (needs SM 90+, hidden size a multiple of 256)"
        )
    if reason is not None:
        logger.info(f"BF16 fused draft head not used: {reason}")
        return None
    return BF16DraftHead(lm_head, key_exchange, max_rows(weight.shape[1], weight.device.index))


__all__ = [
    "BF16DraftHead",
    "FUSED_DRAFT_HEAD_ENV_VAR",
    "MIN_SHARD_BYTES",
    "build_bf16_draft_head",
    "fused_draft_head_enabled",
    "is_plain_mtp_shared_head",
    "max_rows",
]
