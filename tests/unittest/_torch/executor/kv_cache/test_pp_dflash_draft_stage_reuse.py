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
"""Prefix reuse on the pipeline stage that holds the DFlash draft layers, on the C++ manager.

Draft layers with the target's sliding window share its life cycle, so on that
stage every sliding-window page also carries the draft KV. A ``pool_ratio``
written for the target layers then leaves the stage fewer sliding-window pages
than a stage without draft layers at the same quota. A stored prompt is
reusable only while the pages of its trailing window are resident; once they
are evicted the match falls back past every endpoint whose window is gone.

The tests replay one pass of distinct prompts that share a template, then ask
for each prompt again:

* a stage without draft layers matches every prompt in full;
* the draft stage at the target's ``pool_ratio`` evicts the oldest windows, and
  those exact repeats match no deeper than the template;
* the draft stage at ``draft_inclusive_pool_ratio`` matches every prompt in full;
* an extension of a stored prompt matches the whole stored prompt on the draft
  stage, at either ratio.

They need a GPU and the compiled KV cache manager.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU and the C++ KV cache manager"
)

TOKENS_PER_BLOCK = 4
WINDOW = 2 * TOKENS_PER_BLOCK
MAX_SEQ_LEN = 1 << 20
QUOTA = 32 << 20
# Byte shares in life-cycle registration order: sliding window first, then full.
POOL_RATIO = [0.25, 0.75]
TEMPLATE_BLOCKS = 2
UNIQUE_BLOCKS = 6
PROMPT_TOKENS = (TEMPLATE_BLOCKS + UNIQUE_BLOCKS) * TOKENS_PER_BLOCK


@pytest.fixture(scope="module")
def kv():
    """The runtime and layer-spec names, imported only where the bindings exist."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor._util import draft_inclusive_pool_ratio
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        KVCacheLayerOwner,
        KVCacheLayerSpec,
        Role,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        GPU_LEVEL,
        AttentionLayerConfig,
        BufferConfig,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        TokenId,
    )

    # Named one by one: an import used only through ``locals()`` looks unused to
    # the linter, which would delete it.
    return SimpleNamespace(
        draft_inclusive_pool_ratio=draft_inclusive_pool_ratio,
        KVCacheLayerOwner=KVCacheLayerOwner,
        KVCacheLayerSpec=KVCacheLayerSpec,
        Role=Role,
        DataType=DataType,
        GPU_LEVEL=GPU_LEVEL,
        AttentionLayerConfig=AttentionLayerConfig,
        BufferConfig=BufferConfig,
        GpuCacheTierConfig=GpuCacheTierConfig,
        KVCacheManager=KVCacheManager,
        KVCacheManagerConfig=KVCacheManagerConfig,
        LayerId=LayerId,
        TokenId=TokenId,
    )


def _spec(kv, layer_id, owner, window):
    return kv.KVCacheLayerSpec(
        global_layer_id=layer_id,
        owner=owner,
        dtype=kv.DataType.BF16,
        num_kv_heads=1,
        head_dim=128,
        buffer_roles=(kv.Role.KEY, kv.Role.VALUE),
        sliding_window_size=window,
    )


def _target_stage(kv):
    target = kv.KVCacheLayerOwner.TARGET
    return (
        _spec(kv, 0, target, WINDOW),
        _spec(kv, 1, target, None),
        _spec(kv, 2, target, WINDOW),
    )


def _draft_stage(kv):
    """The target stage plus two draft layers with its window: sliding bytes double."""
    draft = kv.KVCacheLayerOwner.DRAFT
    return _target_stage(kv) + (_spec(kv, 3, draft, WINDOW), _spec(kv, 4, draft, WINDOW))


def _sliding_page_bytes(specs):
    return sum(
        spec.bytes_per_token() * TOKENS_PER_BLOCK
        for spec in specs
        if spec.sliding_window_size is not None
    )


def _manager(kv, specs, pool_ratio):
    layers = [
        kv.AttentionLayerConfig(
            layer_id=kv.LayerId(local_id),
            buffers=[
                kv.BufferConfig(role=role, size=spec.bytes_per_token(role) * TOKENS_PER_BLOCK)
                for role in spec.buffer_roles
            ],
            sliding_window_size=spec.sliding_window_size,
            num_sink_tokens=None,
        )
        for local_id, spec in enumerate(specs)
    ]
    return kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kv.GpuCacheTierConfig(quota=QUOTA)],
            layers=layers,
            initial_pool_ratio=pool_ratio,
            commit_min_snapshot=True,
        )
    )


def _sliding_pool(kv, manager, specs):
    page_bytes = _sliding_page_bytes(specs)
    pools = [
        stats
        for stats in manager.get_storage_statistics(kv.GPU_LEVEL)
        if sum(stats.slot_sizes) == page_bytes
    ]
    assert len(pools) == 1, "the sliding-window life cycle must have a pool group of its own"
    return pools[0]


def _prompt(kv, index):
    template = list(range(TEMPLATE_BLOCKS * TOKENS_PER_BLOCK))
    start = 1_000_000 + index * UNIQUE_BLOCKS * TOKENS_PER_BLOCK
    unique = list(range(start, start + UNIQUE_BLOCKS * TOKENS_PER_BLOCK))
    return [kv.TokenId(token) for token in template + unique]


def _commit(manager, stream, tokens):
    """One context: reuse what the tree holds, write and commit the rest, close."""
    kv_cache = manager.create_kv_cache(None, tokens)
    assert kv_cache.resume(stream.cuda_stream)
    assert kv_cache.resize(len(tokens))
    uncommitted = tokens[kv_cache.num_committed_tokens :]
    if uncommitted:
        kv_cache.commit(uncommitted)
    kv_cache.stop_committing()
    stream.synchronize()
    kv_cache.close()


def _replay(kv, specs, pool_ratio, num_prompts):
    """Commit ``num_prompts`` prompts in order, then probe each one again in the same order."""
    manager = _manager(kv, specs, pool_ratio)
    try:
        stream = torch.cuda.Stream()
        prompts = [_prompt(kv, index) for index in range(num_prompts)]
        for prompt in prompts:
            _commit(manager, stream, prompt)
        return [manager.probe_reuse(None, prompt) for prompt in prompts]
    finally:
        manager.shutdown()


def _capacity(kv, specs, pool_ratio):
    """Sliding-window pages of a fresh manager, and the pages one stored prompt keeps."""
    manager = _manager(kv, specs, pool_ratio)
    try:
        stream = torch.cuda.Stream()
        total = _sliding_pool(kv, manager, specs).total
        used = []
        for index in range(3):
            _commit(manager, stream, _prompt(kv, index))
            pool = _sliding_pool(kv, manager, specs)
            used.append(pool.total - pool.free)
        return total, used[2] - used[1]
    finally:
        manager.shutdown()


def test_draft_stage_exact_repeat_misses_at_the_target_ratio_and_hits_at_the_rescaled_one(kv):
    target, draft = _target_stage(kv), _draft_stage(kv)
    rescaled = kv.draft_inclusive_pool_ratio(POOL_RATIO, draft, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
    assert rescaled is not None and rescaled[0] > POOL_RATIO[0]
    assert kv.draft_inclusive_pool_ratio(POOL_RATIO, target, MAX_SEQ_LEN, TOKENS_PER_BLOCK) is None

    target_pages, per_prompt = _capacity(kv, target, POOL_RATIO)
    draft_pages, draft_per_prompt = _capacity(kv, draft, POOL_RATIO)
    rescaled_pages, _ = _capacity(kv, draft, rescaled)
    assert per_prompt == draft_per_prompt > 0
    # One pass that overflows the draft stage at the target ratio and fits everywhere else.
    num_prompts = draft_pages // per_prompt + draft_pages // (4 * per_prompt)
    working_set = num_prompts * per_prompt
    assert draft_pages < working_set
    assert working_set + UNIQUE_BLOCKS + TEMPLATE_BLOCKS < min(target_pages, rescaled_pages), (
        "the quota no longer separates the three layouts",
        target_pages,
        draft_pages,
        rescaled_pages,
        per_prompt,
    )

    target_matches = _replay(kv, target, POOL_RATIO, num_prompts)
    draft_matches = _replay(kv, draft, POOL_RATIO, num_prompts)
    rescaled_matches = _replay(kv, draft, rescaled, num_prompts)

    assert target_matches == [PROMPT_TOKENS] * num_prompts
    assert rescaled_matches == [PROMPT_TOKENS] * num_prompts
    # The draft stage lost the oldest windows; those repeats fall back to the template or less.
    assert draft_matches[0] <= TEMPLATE_BLOCKS * TOKENS_PER_BLOCK
    assert draft_matches[-1] == PROMPT_TOKENS
    assert all(
        match == PROMPT_TOKENS or match <= TEMPLATE_BLOCKS * TOKENS_PER_BLOCK
        for match in draft_matches
    )


@pytest.mark.parametrize("rescale", [False, True], ids=["target-ratio", "rescaled-ratio"])
def test_draft_stage_extension_matches_the_stored_prompt(kv, rescale):
    draft = _draft_stage(kv)
    pool_ratio = (
        kv.draft_inclusive_pool_ratio(POOL_RATIO, draft, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
        if rescale
        else POOL_RATIO
    )
    manager = _manager(kv, draft, pool_ratio)
    try:
        stream = torch.cuda.Stream()
        stored = _prompt(kv, 0)
        _commit(manager, stream, stored)
        extension = stored + [kv.TokenId(2_000_000 + i) for i in range(2 * TOKENS_PER_BLOCK)]
        assert manager.probe_reuse(None, extension) == len(stored)
        _commit(manager, stream, extension)
        assert manager.probe_reuse(None, extension) == len(extension)
    finally:
        manager.shutdown()
