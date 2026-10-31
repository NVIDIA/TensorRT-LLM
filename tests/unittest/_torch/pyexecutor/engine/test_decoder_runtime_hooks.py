# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner and DecoderMetadata wiring of the step-loop hooks a model declares."""

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import DecoderMetadata
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    MsaRuntimePolicy,
    RuntimeHooks,
)
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
    MiniMaxM3SparseAttentionConfig,
)

pytestmark = pytest.mark.cpu_only

_SHORT_IDENTITY = RuntimeHooks(msa=MsaRuntimePolicy(short_sequence_identity=True))


def _runner(
    sparse_attention_config: BaseSparseAttentionConfig | None, hooks: RuntimeHooks
) -> DecoderRunner:
    # Only the two attributes the threshold reads.
    runner = object.__new__(DecoderRunner)
    runner._sparse_attention_config = sparse_attention_config
    runner.runtime_hooks = hooks
    return runner


def _metadata(hooks: RuntimeHooks, attn_metadata: object) -> DecoderMetadata:
    # Only the two attributes the MSA arming reads.
    metadata = object.__new__(DecoderMetadata)
    metadata.runtime_hooks = hooks
    metadata.attn_metadata = attn_metadata
    return metadata


def test_no_threshold_without_a_split_or_a_short_identity_policy():
    assert _runner(None, DEFAULT_RUNTIME_HOOKS)._short_long_seq_len_threshold() is None
    assert (
        _runner(
            MiniMaxM3SparseAttentionConfig(), DEFAULT_RUNTIME_HOOKS
        )._short_long_seq_len_threshold()
        is None
    )


def test_a_sparse_config_that_asks_for_the_split_sets_the_threshold():
    config = DeepSeekSparseAttentionConfig(index_topk=512, skip_indexer_for_short_seqs=True)

    # The sparse config wins over the hooks.
    assert _runner(config, _SHORT_IDENTITY)._short_long_seq_len_threshold() == 512


def test_short_identity_threshold_is_the_topk_span():
    config = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128)

    assert _runner(config, _SHORT_IDENTITY)._short_long_seq_len_threshold() == 16 * 128


@pytest.mark.parametrize(
    "config",
    [None, DeepSeekSparseAttentionConfig(index_topk=512, skip_indexer_for_short_seqs=False)],
    ids=["dense", "dsa_without_split"],
)
def test_short_identity_requires_the_minimax_m3_sparse_config(config):
    with pytest.raises(ValueError, match="requires a MiniMax-M3 sparse attention config"):
        _runner(config, _SHORT_IDENTITY)._short_long_seq_len_threshold()


def test_a_default_msa_policy_arms_nothing():
    # A non-MSA metadata and no cache manager are fine when nothing is armed.
    metadata = _metadata(DEFAULT_RUNTIME_HOOKS, object.__new__(TrtllmAttentionMetadata))

    metadata._arm_msa_policy(None)


def test_an_msa_policy_requires_the_msa_backend_and_cache():
    metadata = _metadata(
        RuntimeHooks(msa=MsaRuntimePolicy(reuse_pool_views=True)),
        object.__new__(TrtllmAttentionMetadata),
    )

    with pytest.raises(ValueError, match="requires the MiniMax-M3 MSA sparse attention backend"):
        metadata._arm_msa_policy(object.__new__(MiniMaxM3KVCacheManagerV2))


def test_an_msa_policy_reaches_the_metadata_and_the_cache():
    policy = MsaRuntimePolicy(
        short_sequence_identity=True,
        skip_unchanged_block_offsets=True,
        reuse_pool_views=False,
    )
    metadata = _metadata(
        RuntimeHooks(msa=policy), object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    )
    manager = object.__new__(MiniMaxM3KVCacheManagerV2)

    metadata._arm_msa_policy(manager)

    assert metadata.attn_metadata.msa_policy is policy
    assert manager.skip_unchanged_block_offsets is True
    assert manager.reuse_pool_views is False
