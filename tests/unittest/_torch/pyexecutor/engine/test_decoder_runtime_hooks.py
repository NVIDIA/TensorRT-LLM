# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner and DecoderMetadata wiring of the step-loop hooks a model declares."""

from types import SimpleNamespace

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
    sparse_attention_config: BaseSparseAttentionConfig | None,
    hooks: RuntimeHooks,
    max_draft_len: int = 0,
) -> DecoderRunner:
    # Only the attributes the threshold reads.
    runner = object.__new__(DecoderRunner)
    runner._sparse_attention_config = sparse_attention_config
    runner.runtime_hooks = hooks
    runner._config = SimpleNamespace(max_draft_len=max_draft_len)
    runner._get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
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


@pytest.mark.parametrize("max_draft_len", [0, 3])
def test_short_identity_threshold_keeps_one_step_of_headroom_in_the_topk_span(max_draft_len):
    """The short graph family runs the identity selection: one generation step's tokens below the top-k span, the most
    the graph key's host length estimate can lag the attended length under the overlap scheduler."""
    config = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128)

    threshold = _runner(config, _SHORT_IDENTITY, max_draft_len)._short_long_seq_len_threshold()
    assert threshold == 16 * 128 - (max_draft_len + 1)


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
    # The MSA skip does not arm the base class's draft block-offset guard.
    assert manager.skip_unchanged_draft_block_offsets is False


# -- MsaRuntimePolicy.kv_update_tail ----------------------------------------------------------


def _tail_metadata(tail_update, *, num_tokens=3, tail_start=3, buffer_rows=8):
    """Only what MiniMaxM3MsaSparseAttentionMetadata._kv_update_with_tail reads."""
    import torch

    return SimpleNamespace(
        msa_policy=MsaRuntimePolicy(kv_update_tail=tail_update),
        _msa_fields_ready=True,
        _msa_kv_lens_dynamic=True,
        num_tokens=num_tokens,
        num_seqs=1,
        _msa_tail_start=tail_start,
        msa_out_cache_loc=torch.zeros(buffer_rows, dtype=torch.int64),
    )


def _tail_op(*args):
    return True


def test_the_tail_update_is_taken_only_with_a_tail_to_fill():
    take = MiniMaxM3MsaSparseAttentionMetadata._kv_update_with_tail

    assert take(_tail_metadata(_tail_op)) is _tail_op
    assert take(_tail_metadata(None)) is None
    # No tail past the live rows, or a tail start inside them.
    assert take(_tail_metadata(_tail_op, tail_start=8)) is None
    assert take(_tail_metadata(_tail_op, num_tokens=4, tail_start=3)) is None
    assert take(_tail_metadata(_tail_op, tail_start=None)) is None
    static = _tail_metadata(_tail_op)
    static._msa_kv_lens_dynamic = False
    assert take(static) is None


# -- RuntimeHooks.spec_slot_ids_changed_only ---------------------------------------------------


@pytest.mark.parametrize("changed_only", [False, True])
def test_slot_ids_are_uploaded_again_only_when_they_change(monkeypatch, changed_only):
    import torch

    from tensorrt_llm._torch.speculative import interface

    copies = []
    meta = SimpleNamespace(slot_ids_changed_only=changed_only, _batch_slot_ids_last=[None, None])
    buffer = torch.zeros(4, dtype=torch.long)

    class _Buffer:
        """batch_slot_ids, recording every copy into it."""

        def __getitem__(self, index):
            view = buffer[index]
            original = view.copy_

            def copy_(src, non_blocking=False):
                copies.append(src.tolist())
                return original(src)

            view.copy_ = copy_
            return view

    meta.batch_slot_ids = _Buffer()
    monkeypatch.setattr(interface, "prefer_pinned", lambda: False)
    upload = interface.SpecMetadata._upload_batch_slot_ids

    upload(meta, [2, 0])
    upload(meta, [2, 0])
    upload(meta, [1, 3])

    expected = [[2, 0], [1, 3]] if changed_only else [[2, 0], [2, 0], [1, 3]]
    assert copies == expected
    assert buffer[:2].tolist() == [1, 3]
