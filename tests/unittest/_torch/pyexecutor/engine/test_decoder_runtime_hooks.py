# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner's short/long decode graph split, asked of the sparse attention config."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
    MiniMaxM3SparseAttentionConfig,
)

pytestmark = pytest.mark.cpu_only


def _runner(
    sparse_attention_config: BaseSparseAttentionConfig | None, max_draft_len: int = 0
) -> DecoderRunner:
    # Only the attributes the threshold reads.
    runner = object.__new__(DecoderRunner)
    runner._sparse_attention_config = sparse_attention_config
    runner._config = SimpleNamespace(max_draft_len=max_draft_len)
    runner._get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    return runner


def test_no_threshold_without_a_split():
    assert _runner(None)._short_long_seq_len_threshold() is None
    # The MiniMax-M3 Triton reference implementation stages nothing different for short contexts.
    assert _runner(MiniMaxM3SparseAttentionConfig())._short_long_seq_len_threshold() is None
    # A sequence-length-aware config that does not ask for the split.
    config = DeepSeekSparseAttentionConfig(index_topk=512, skip_indexer_for_short_seqs=False)
    assert _runner(config)._short_long_seq_len_threshold() is None


def test_a_sequence_length_aware_config_that_asks_for_the_split_sets_the_threshold():
    config = DeepSeekSparseAttentionConfig(index_topk=512, skip_indexer_for_short_seqs=True)

    # Its own threshold, with no headroom.
    assert _runner(config, max_draft_len=3)._short_long_seq_len_threshold() == 512


@pytest.mark.parametrize("max_draft_len", [0, 3])
def test_msa_threshold_keeps_one_step_of_headroom_in_the_topk_span(max_draft_len):
    """The short graph family runs the identity selection: one generation step's tokens below the top-k span, the most
    the graph key's host length estimate can lag the attended length under the overlap scheduler."""
    config = MiniMaxM3SparseAttentionConfig(
        implementation="msa", sparse_topk_blocks=16, sparse_block_size=128
    )

    runner = _runner(config, max_draft_len)
    assert runner._short_long_seq_len_threshold() == 16 * 128 - (max_draft_len + 1)
