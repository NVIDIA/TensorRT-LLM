# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched engine steps of the Beam attention contract: request spans, rejections and rows.

The native runtime orders a step's requests contexts first, then generations; each request
has its own cached prefix and KV length. ``check_batch_metadata`` must return each request's
``(past, new tokens)`` from the prepared metadata and reject a step whose rows or requests
disagree; ``check_metadata`` keeps the one-request contract; the full model selects the last
row of each request for the sampler. Needs one CUDA device (a small KVCacheManagerV2); no
model files.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.beam import _attention
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (  # noqa: E501
    BeamFullModelNvfp4Sm103Tp4,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

KV_HEADS, HEAD_DIM, TOKENS_PER_BLOCK, MAX_SEQ_LEN = 2, 128, 32, 4096
#: (cached tokens, new tokens) per request: two contexts (fresh, after a cached prefix) and
#: two generation requests at different depths.
CONTEXTS = ((0, 40), (64, 20))
GENERATIONS = ((100, 1), (7, 1))


@contextlib.contextmanager
def prepared_step(contexts, generations, *, cache_reuse=True):
    """A one-layer BF16 KVCacheManagerV2 and prepared metadata for one engine step."""
    requests = list(contexts) + list(generations)
    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=64 * TOKENS_PER_BLOCK * len(requests), enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=len(requests),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    try:
        request_ids = list(range(1, len(requests) + 1))
        mgr.add_dummy_requests(request_ids, token_nums=[past + n for past, n in requests])
        md = TrtllmAttentionMetadata(
            max_num_requests=len(requests),
            max_num_tokens=8192,
            kv_cache_manager=mgr,
            num_contexts=len(contexts),
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[past for past, _ in requests]
            ),
            seq_lens=torch.tensor([n for _, n in requests], dtype=torch.int),
            request_ids=request_ids,
            # The engine's context length of a context request is its new tokens (this chunk).
            prompt_lens=[n for _, n in contexts] + [past + n for past, n in generations],
            runtime_features=AttentionRuntimeFeatures(cache_reuse=cache_reuse),
        )
        md.prepare()
        yield md
    finally:
        mgr.shutdown()


def _spans(md, num_tokens):
    return _attention.check_batch_metadata(
        md, num_tokens, layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
    )


def test_mixed_step_spans():
    requests = list(CONTEXTS) + list(GENERATIONS)
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        assert _spans(md, sum(n for _, n in requests)) == requests


def test_generation_only_step_spans():
    with prepared_step((), GENERATIONS) as md:
        assert _spans(md, len(GENERATIONS)) == list(GENERATIONS)


def test_one_request_contract_is_kept():
    with prepared_step(CONTEXTS[1:], ()) as md:
        past = _attention.check_metadata(
            md, CONTEXTS[1][1], layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
        )
        assert past == CONTEXTS[1][0]
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="one request per call"):
            _attention.check_metadata(md, 62, layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM)


@pytest.mark.parametrize("num_tokens", [61, 63, 1])
def test_rows_must_be_the_requests_new_tokens(num_tokens):
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="do not fill"):
            _spans(md, num_tokens)


def test_cached_prefix_needs_paged_context_fmha():
    with prepared_step(CONTEXTS, GENERATIONS, cache_reuse=False) as md:
        with pytest.raises(ValueError, match="paged-context FMHA"):
            _spans(md, 62)


def test_generation_request_appends_one_token():
    with prepared_step(CONTEXTS, ((100, 2),)) as md:
        with pytest.raises(ValueError, match="exactly one token"):
            _spans(md, 62)


def test_unheld_layer_is_rejected():
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="does not hold logical layer 3"):
            _attention.check_batch_metadata(
                md, 62, layer_idx=3, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
            )


@pytest.mark.parametrize(
    "spans,expected",
    [
        ([(0, 40)], [39]),
        ([(0, 40), (64, 20), (100, 1), (7, 1)], [39, 59, 60, 61]),
        ([(5, 1), (9, 1), (2, 1)], [0, 1, 2]),
    ],
)
def test_default_rows_are_each_requests_last_row(spans, expected):
    select = BeamFullModelNvfp4Sm103Tp4._select_rows
    assert select(None, None, spans) == expected
    assert select(None, "all", spans) == list(range(sum(n for _, n in spans)))
