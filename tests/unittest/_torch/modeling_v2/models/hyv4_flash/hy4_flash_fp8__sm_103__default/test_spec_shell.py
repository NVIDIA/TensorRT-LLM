# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 worker's sampling walk and block length bookkeeping, on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from _target import requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.spec_worker import (  # noqa: E501
    Hyv4FlashDFlashWorker,
)
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

pytestmark = requires_checkpoint


def _worker() -> Hyv4FlashDFlashWorker:
    config = DFlashDecodingConfig(
        decoding_type="DFlash", max_draft_len=BLOCK_WIDTH - 1, mask_token_id=MASK_TOKEN_ID
    )
    return Hyv4FlashDFlashWorker(config, SimpleNamespace(rank=0))


def test_the_walk_pairs_each_column_with_the_token_before_it():
    """Step `s` corrects column `s + 1`'s hidden with column `s`'s token, starting from the anchor."""
    worker = _worker()
    num_gens, num_drafts = 2, 5
    vocab = 8
    # Each row's hidden carries its own column number; a step draws that number.
    rows = torch.zeros(num_gens * num_drafts, vocab)
    rows[:, 0] = torch.arange(1, num_drafts + 1).repeat(num_gens)
    pairs = []

    class _Numbered:
        def hidden_correction(self, hidden, previous):
            pairs.append((hidden[:, 0].tolist(), previous.tolist()))
            return hidden, None

        def block_logits(self, corrected, attn_metadata):
            return torch.nn.functional.one_hot(corrected[:, 0].long(), vocab).float()

    worker.greedy_sample_draft_with_tp_gather = lambda logits, spec_metadata: logits.argmax(dim=-1)
    tokens, got_vocab = worker._sample_block_draft_tokens(
        _Numbered(),
        torch.zeros(num_gens, num_drafts, vocab),
        rows,
        {
            "first_prev_tokens": torch.tensor([100, 200]),
            "block_placement": SimpleNamespace(attn_metadata=None),
        },
        SimpleNamespace(wants_advanced_draft_sampling=False),
        num_contexts=0,
        batch_size=num_gens,
    )
    assert got_vocab == vocab
    assert tokens.tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert pairs == [
        ([1, 1], [100, 200]),
        ([2, 2], [1, 1]),
        ([3, 3], [2, 2]),
        ([4, 4], [3, 3]),
        ([5, 5], [4, 4]),
    ]


def test_the_worker_writes_the_blocks_end_into_every_length():
    """Generation rows get `anchors + width` (aliased or split runtime view), and no rewind follows."""
    worker = _worker()
    for aliased in (True, False):
        meta = SimpleNamespace(kv_lens_cuda=torch.tensor([10, 20, 30, 40], dtype=torch.int32))
        meta.kv_lens_cuda_runtime = meta.kv_lens_cuda if aliased else meta.kv_lens_cuda.clone()
        meta._seq_lens = torch.zeros(4, dtype=torch.int32)
        meta._seq_lens_cuda = torch.zeros(4, dtype=torch.int32)
        worker._prepare_kv_for_draft_forward(meta, torch.zeros(2, dtype=torch.int32), 2, 4)
        worker._write_block_kv_lens(meta, 2, 4, torch.tensor([100, 200]), BLOCK_WIDTH)
        assert worker._kv_rewind_amount is None
        assert meta.kv_lens_cuda.tolist() == [10, 20, 106, 206]
        assert meta.kv_lens_cuda_runtime.tolist() == [10, 20, 106, 206]
        assert meta._seq_lens.tolist() == [0, 0, 6, 6]
        assert meta._seq_lens_cuda.tolist() == [0, 0, 6, 6]
