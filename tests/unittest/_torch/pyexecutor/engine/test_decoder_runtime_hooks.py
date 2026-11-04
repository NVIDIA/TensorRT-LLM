# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner's short/long decode graph split, asked of the sparse config and the attention metadata class."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
    MiniMaxM3SparseAttentionConfig,
)

pytestmark = pytest.mark.cpu_only


def _runner(
    sparse_attention_config: BaseSparseAttentionConfig | None,
    metadata_cls: type,
    max_draft_len: int = 0,
) -> DecoderRunner:
    # Only the attributes the threshold reads.
    runner = object.__new__(DecoderRunner)
    runner._sparse_attention_config = sparse_attention_config
    runner._config = SimpleNamespace(
        attention_backend=SimpleNamespace(Metadata=metadata_cls), max_draft_len=max_draft_len
    )
    runner._get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    return runner


def test_no_threshold_without_a_split():
    assert _runner(None, TrtllmAttentionMetadata)._short_long_seq_len_threshold() is None
    # A MiniMax-M3 config on a backend whose metadata stages nothing different for short contexts (the Triton
    # reference implementation uses plain TRTLLM metadata here).
    assert (
        _runner(
            MiniMaxM3SparseAttentionConfig(), TrtllmAttentionMetadata
        )._short_long_seq_len_threshold()
        is None
    )
    # Metadata without the classmethod (another attention backend).
    assert _runner(None, object)._short_long_seq_len_threshold() is None


def test_a_sparse_config_that_asks_for_the_split_sets_the_threshold():
    config = DeepSeekSparseAttentionConfig(index_topk=512, skip_indexer_for_short_seqs=True)

    # The sparse config wins over the metadata class.
    assert (
        _runner(config, MiniMaxM3MsaSparseAttentionMetadata)._short_long_seq_len_threshold() == 512
    )


def test_msa_metadata_splits_at_the_topk_span():
    config = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128)

    assert (
        _runner(config, MiniMaxM3MsaSparseAttentionMetadata)._short_long_seq_len_threshold()
        == 16 * 128 - 1
    )
    # Without a MiniMax-M3 config the MSA metadata has no span to split at.
    assert (
        _runner(None, MiniMaxM3MsaSparseAttentionMetadata)._short_long_seq_len_threshold() is None
    )


@pytest.mark.parametrize("max_draft_len", [0, 3])
def test_msa_threshold_keeps_one_step_of_headroom_in_the_topk_span(max_draft_len):
    """The short graph family runs the identity selection: one generation step's tokens below the top-k span, the most
    the graph key's host length estimate can lag the attended length under the overlap scheduler."""
    config = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128)

    runner = _runner(config, MiniMaxM3MsaSparseAttentionMetadata, max_draft_len)
    assert runner._short_long_seq_len_threshold() == 16 * 128 - (max_draft_len + 1)
