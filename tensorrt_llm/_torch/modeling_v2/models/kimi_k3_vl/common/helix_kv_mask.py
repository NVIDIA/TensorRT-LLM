# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-token mask of the tokens a helix CP rank holds no KV for.

:func:`_helix_zero_kv_mask` is Modeling V1's (``tensorrt_llm/_torch/attention/attention.py`` at
21b86d335db76818167f583d14d5dd2dc2b6a0d9) token for token; only the imports differ.
``test_k3_common.py`` checks it against its source.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

__all__ = ["_helix_zero_kv_mask"]


def _helix_zero_kv_mask(
    attn_metadata: AttentionMetadata,
    num_tokens: int,
    *,
    seq_start: int = 0,
    num_seqs: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return a per-token bool mask marking tokens with zero local KV on this CP rank.

    These tokens belong to sequences for which this rank owns no KV blocks. Since
    kv_lens_cuda is per-sequence, it is expanded to per-token using the
    seq_lens_cuda query lengths, which stays correct when a sequence spans
    multiple tokens (e.g. speculative decoding). The buffers are static and
    output_size is passed to repeat_interleave, so the mask is CUDA-graph safe.
    Returns None when a length buffer is unavailable.

    Args:
        attn_metadata: Attention metadata holding the length buffers.
        num_tokens: Number of tokens covered by partial_o; used as the static
            output_size of the per-token expansion.
        seq_start: Index of the first sequence covered by partial_o. Defaults to
            0 for generation-only callers; MLA passes num_contexts to select the
            generation slice.
        num_seqs: Number of sequences covered by partial_o. Defaults to all
            sequences after seq_start.
    """
    kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
    seq_lens = getattr(attn_metadata, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    if num_seqs is None:
        num_seqs = attn_metadata.num_seqs - seq_start
    seq_slice = slice(seq_start, seq_start + num_seqs)
    per_seq_mask = kv_lens[seq_slice] == 0
    return torch.repeat_interleave(per_seq_mask, seq_lens[seq_slice], output_size=num_tokens)
