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
"""DeepSeek-V4.1 encoder SWA bounded replay: the runtime half.

The prefix cache keeps main KV and index K only; a hit of ``reused`` tokens (a
multiple of the compress ratio) schedules the last ``min(window, reused)`` cached
tokens as replay rows at the head of the first chunk, charged to the token budget,
and the attention metadata tells the target which rows they are.
"""

import os
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    cache_manager as v4_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    _v41_encoder_replay_refusal,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY,
    DEEPSEEK_V41_ENCODER_REPLAY_ENV,
    get_deepseek_v41_encoder_replay,
    get_deepseek_v41_layout,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    DeepSeekV4SparseAttentionConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

from .test_deepseek_v41_layout import (
    HEAD_DIM,
    INDEX_HEAD_DIM,
    TOKENS_PER_BLOCK,
    WINDOW,
    T,
    kv_storage_env,
    make_v41_config,
    reference_page_indices,
)

ENGRAM_HISTORY = 3  # engram_max_ngram_size 4


@contextmanager
def replay_env(value: Optional[str]) -> Iterator[None]:
    saved = os.environ.get(DEEPSEEK_V41_ENCODER_REPLAY_ENV)
    try:
        if value is None:
            os.environ.pop(DEEPSEEK_V41_ENCODER_REPLAY_ENV, None)
        else:
            os.environ[DEEPSEEK_V41_ENCODER_REPLAY_ENV] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(DEEPSEEK_V41_ENCODER_REPLAY_ENV, None)
        else:
            os.environ[DEEPSEEK_V41_ENCODER_REPLAY_ENV] = saved


def make_manager(
    replay: bool,
    *,
    block_reuse: bool = True,
    max_batch_size: int = 8,
    max_seq_len: int = 4096,
    max_num_tokens: int = 8192,
    mapping: Optional[Mapping] = None,
    switch: Optional[str] = "",
    **extra,
) -> tuple:
    """A V4.1-layout manager whose replay mode resolves to ``replay``. ``switch`` is the
    environment value it is built under: "" = "1" / "0" from ``replay``, None = unset."""
    config = make_v41_config()
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW,
        compress_ratios=[1] * len(config.compress_ratios),
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=block_reuse,
        max_tokens=max_seq_len * max_batch_size,
        event_buffer_max_size=0,
    )
    env = ("1" if replay else "0") if switch == "" else switch
    with replay_env(env), kv_storage_env("bf16"):
        manager = DeepseekV4CacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=config.num_hidden_layers,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_seq_len,
            mapping=mapping or Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=sparse_config,
            pretrained_config=config,
            **extra,
        )
    assert manager.v41_encoder_replay == replay
    return manager, sparse_config


def make_request(request_id: int, tokens: Sequence[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def prompt(seed: int, length: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 120000, (length,), generator=g).tolist()


def run_context(manager: DeepseekV4CacheManager, request: LlmRequest) -> None:
    """Schedule the whole context, run it, and commit it to the prefix cache."""
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_chunk_size)
    request.context_current_position = request.prompt_len
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    manager.update_context_resources(batch)


def pool_stats(manager: DeepseekV4CacheManager, layer: int, attn_type):
    impl = manager.impl
    lc = impl.get_layer_group_id(manager._layer_attn_to_layer_id[layer, attn_type])
    pool_group = impl.get_life_cycle_pool_group_indices()[lc]
    return impl.get_storage_statistics()[pool_group]


# ---------------------------------------------------------------------------
# Switch and layout (host only).
# ---------------------------------------------------------------------------


def test_switch_and_layout():
    with replay_env(None):
        assert get_deepseek_v41_encoder_replay() is DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY
    for value, expected in (
        ("1", True),
        ("true", True),
        ("ON", True),
        ("0", False),
        ("off", False),
    ):
        with replay_env(value):
            assert get_deepseek_v41_encoder_replay() is expected
    with replay_env("maybe"), pytest.raises(ValueError, match=DEEPSEEK_V41_ENCODER_REPLAY_ENV):
        get_deepseek_v41_encoder_replay()

    for replay in (False, True):
        with replay_env("1" if replay else "0"):
            layout = get_deepseek_v41_layout(make_v41_config())
        assert layout.encoder_replay is replay
        assert layout.replay_window == WINDOW
        assert layout.reuse_alignment == 2  # compress ratios {1, 2}
        for draft in (0, 5):
            assert layout.window(T.SWA, draft) == WINDOW + draft
            assert layout.window(T.COMPRESSOR_KV, draft) == 2 + draft
            # A replay step starts `window` positions before the reuse boundary the
            # manager keeps live, and its first rows hash the 3 positions before it.
            assert layout.window(T.ENGRAM_TOKEN, draft) == (
                ENGRAM_HISTORY + 1 + (WINDOW if replay else 0) + draft
            )
            assert layout.window(T.COMPRESS, draft) is None


@skip_pre_blackwell
@pytest.mark.parametrize(
    "case",
    ["disagg", "small_token_budget", "joint_draft_reuse"],
)
def test_unsupported_combinations_fail_fast(case):
    kwargs = {}
    match = DEEPSEEK_V41_ENCODER_REPLAY_ENV
    if case == "disagg":
        kwargs["is_disagg"] = True
        match = "disaggregated"
    elif case == "small_token_budget":
        kwargs["max_num_tokens"] = WINDOW
        match = "max_num_tokens"
    else:
        kwargs["joint_kv_cache_reuse"] = True
        match = "draft KV pool"
    with pytest.raises(ValueError, match=match):
        make_manager(True, **kwargs)
    if case == "small_token_budget":
        # Refused only because of replay.
        manager, _ = make_manager(False, **kwargs)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "case", ["tp", "attention_dp", "disagg", "connector", "small_token_budget"]
)
def test_unset_switch_resolves_per_configuration(case, capfd, monkeypatch):
    """Ruling R21: unset, replay is on exactly where it is supported (aggregated, no
    connector; attention TP or DP: under attention DP each rank's manager replays the
    hits of its own requests) and off elsewhere with one line naming why; an explicit
    1 on an unsupported configuration raises and an explicit 0 is always off."""
    kwargs = {}
    reason = None
    if case == "attention_dp":
        kwargs["mapping"] = Mapping(world_size=1, rank=0, tp_size=1, enable_attention_dp=True)
    elif case == "disagg":
        kwargs["is_disagg"] = True
        reason = "disaggregated"
    elif case == "connector":
        kwargs["kv_connector_manager"] = object()
        reason = "KV cache connector"
        layout = get_deepseek_v41_layout(make_v41_config())
        refusal = _v41_encoder_replay_refusal(layout, Mapping(), 8192, kwargs)
        assert refusal is not None and reason in refusal[1], refusal
        # The explicit switch raises before any pool exists; a manager with a stand-in
        # connector is not built.
        with pytest.raises(ValueError, match=reason):
            make_manager(True, switch="1", **kwargs)
        return
    elif case == "small_token_budget":
        kwargs["max_num_tokens"] = WINDOW
        reason = "max_num_tokens"
    supported = case in ("tp", "attention_dp")
    assert DEEPSEEK_V41_DEFAULT_ENCODER_REPLAY is True
    # The line is once per process: start from none printed, whatever ran before.
    monkeypatch.setattr(v4_cache_manager, "_V41_ENCODER_REPLAY_ANNOUNCED", set())
    capfd.readouterr()
    manager, _ = make_manager(supported, switch=None, **kwargs)
    manager.shutdown()
    out = capfd.readouterr().out
    if supported:
        assert DEEPSEEK_V41_ENCODER_REPLAY_ENV not in out, out
    else:
        lines = [line for line in out.splitlines() if DEEPSEEK_V41_ENCODER_REPLAY_ENV in line]
        assert len(lines) == 1 and "resolves off" in lines[0] and reason in lines[0], out
        # Once per process: a second manager (the engine builds an estimation one
        # first) does not repeat it.
        manager, _ = make_manager(False, switch=None, **kwargs)
        manager.shutdown()
        assert DEEPSEEK_V41_ENCODER_REPLAY_ENV not in capfd.readouterr().out
        with pytest.raises(ValueError, match=reason):
            make_manager(True, switch="1", **kwargs)
    manager, _ = make_manager(False, switch="0", **kwargs)
    manager.shutdown()


# ---------------------------------------------------------------------------
# The prefix cache and the reuse boundary (GPU).
# ---------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize("replay", [False, True], ids=["replay_off", "replay_on"])
def test_prefix_cache_keeps_global_kv_only(replay: bool):
    """With replay on, a finished request leaves main KV and index K in the prefix
    cache and no window page (SWA, compressor state, Engram history)."""
    manager, _ = make_manager(replay)
    try:
        request = make_request(0, prompt(0, 1000))
        run_context(manager, request)
        manager.free_resources(request)
        cached = {
            role: pool_stats(manager, layer, role).evictable
            for layer, role in (
                (0, T.SWA),
                (2, T.COMPRESSOR_KV),
                (1, T.ENGRAM_TOKEN),
                (2, T.COMPRESS),
                (20, T.COMPRESS),
            )
        }
        assert cached[T.COMPRESS] > 0, cached
        for role in (T.SWA, T.COMPRESSOR_KV, T.ENGRAM_TOKEN):
            assert (cached[role] == 0) == replay, (role, cached)
    finally:
        manager.shutdown()


# (shared prefix length with the cached request, expected reuse boundary)
HIT_CASES = {
    "mid_block_even": (700, 700),
    "mid_block_odd": (701, 700),
    "block_aligned": (896, 896),
    "below_window": (101, 100),
    "whole_prompt": (1000, 1000),
}


@skip_pre_blackwell
@pytest.mark.parametrize("name", list(HIT_CASES))
def test_hit_schedules_replay_rows(name: str):
    shared, reused = HIT_CASES[name]
    manager, _ = make_manager(True)
    requests = []
    try:
        first = make_request(0, prompt(1, 1000))
        requests.append(first)
        run_context(manager, first)  # kept alive: its window pages stay private
        tokens = first.get_tokens(0)[:shared] + prompt(2, 300)
        second = make_request(1, tokens)
        requests.append(second)
        assert manager.prepare_context(second)

        num_replay = min(WINDOW, reused)
        start = reused - num_replay
        kv_cache = manager.kv_cache_map[second.py_request_id]
        assert kv_cache.num_committed_tokens == reused  # the cache's claim, even
        assert reused % 2 == 0
        # The cursor sits at the replay start: the replay rows head the first chunk
        # and count as context tokens of it (the scheduler's budget).
        assert second.context_current_position == start
        assert second.prepopulated_prompt_len == start
        assert second.is_first_context_chunk
        assert second.context_remaining_length == len(tokens) - start
        plan = manager.get_v41_replay_plan(second.py_request_id, start)
        assert plan is not None
        assert (plan.start, plan.num_tokens, plan.reused) == (start, num_replay, reused)
        assert list(plan.history_token_ids) == tokens[max(0, start - ENGRAM_HISTORY) : start]
        assert manager.get_v41_replay_plan(second.py_request_id, start + 1) is None

        # A chunk that cannot hold every replay row is refused.
        assert not manager.resize_context(second, num_replay - 1)
        assert manager.resize_context(second, second.context_chunk_size)

        request_ids = [first.py_request_id, second.py_request_id]
        # Window pages of the replay rows (and the Engram history before them) are
        # allocated and private: never the cached request's pages.
        window_starts = (
            (0, T.SWA, start),
            (39, T.SWA, start),
            (1, T.ENGRAM_TOKEN, max(0, start - 3)),
        )
        for layer, role, lo in window_starts:
            pages = reference_page_indices(manager, request_ids, 2, layer, role, pages_mode(role))
            for ordinal in range(lo // TOKENS_PER_BLOCK, (reused - 1) // TOKENS_PER_BLOCK + 1):
                assert int(pages[1, ordinal]) >= 0, (role, ordinal)
                assert int(pages[1, ordinal]) != int(pages[0, ordinal]), (role, ordinal)
        # Main KV and index K of the fully cached blocks are the cached request's.
        for owner in (2, 20):
            for role in (T.COMPRESS, T.INDEXER_COMPRESS):
                pages = reference_page_indices(
                    manager, request_ids, 2, owner, role, pages_mode(role)
                )
                for ordinal in range(reused // TOKENS_PER_BLOCK):
                    assert int(pages[1, ordinal]) == int(pages[0, ordinal]), (owner, role, ordinal)
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


def pages_mode(role):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

    if role in (T.COMPRESS, T.INDEXER_COMPRESS):
        return PageIndexMode.SHARED
    return PageIndexMode.PER_LAYER


@skip_pre_blackwell
def test_replay_off_keeps_the_claim():
    """Replay off: a hit settles the cursor at the claim and schedules no replay rows."""
    manager, _ = make_manager(False)
    requests = []
    try:
        first = make_request(0, prompt(1, 1000))
        requests.append(first)
        run_context(manager, first)
        second = make_request(1, first.get_tokens(0) + prompt(2, 50))
        requests.append(second)
        assert manager.prepare_context(second)
        claimed = manager.kv_cache_map[second.py_request_id].num_committed_tokens
        assert second.context_current_position == claimed
        assert manager.get_v41_replay_plan(second.py_request_id, claimed) is None
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("chunked", [False, True], ids=["whole_context", "chunked"])
def test_replay_rows_are_charged_to_the_token_budget(chunked: bool):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy

    manager, _ = make_manager(True)
    requests = []
    try:
        first = make_request(0, prompt(1, 1000))
        requests.append(first)
        run_context(manager, first)
        manager.free_resources(first)
        requests.remove(first)
        tokens = first.get_tokens(0)[:700] + prompt(2, 300)  # 1000 tokens, 700 cached
        start = 700 - WINDOW
        needed = len(tokens) - start  # suffix 300 + replay 128

        def schedule(budget: int, request_id: int):
            request = make_request(request_id, tokens)
            requests.append(request)
            ctx_chunk_config = (
                (ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TOKENS_PER_BLOCK)
                if chunked
                else None
            )
            scheduler = KVCacheV2Scheduler(
                max_batch_size=8,
                max_num_tokens=budget,
                kv_cache_manager=manager,
                scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
                ctx_chunk_config=ctx_chunk_config,
            )
            return request, scheduler.schedule_request([request], set())

        if chunked:
            # The first chunk starts at the replay start and holds the replay rows.
            request, out = schedule(2 * TOKENS_PER_BLOCK, 10)
            assert [r.py_request_id for r in out.context_requests] == [10]
            assert request.context_current_position == start
            assert request.context_chunk_size == 2 * TOKENS_PER_BLOCK
            assert manager.get_v41_replay_plan(10, start) is not None
        else:
            request, out = schedule(needed, 10)
            assert [r.py_request_id for r in out.context_requests] == [10]
            assert request.context_current_position == start
            assert request.context_chunk_size == needed
            # One token short: the replay rows are charged, so the context no
            # longer fits, and the failed admission drops the claim and the plan.
            request, out = schedule(needed - 1, 11)
            assert out.context_requests == []
            assert request.context_current_position == 0
            assert 11 not in manager.kv_cache_map
            assert manager.get_v41_replay_plan(11, start) is None
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


# ---------------------------------------------------------------------------
# What the target sees (GPU).
# ---------------------------------------------------------------------------


def build_metadata(manager, sparse_config, requests, num_contexts, cached, new_lens):
    metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(new_lens, dtype=torch.int),
        request_ids=[r.py_request_id for r in requests],
        max_num_requests=manager.max_batch_size,
        num_contexts=num_contexts,
        prompt_lens=[c + n for c, n in zip(cached, new_lens)],
        max_num_tokens=8192,
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    torch.cuda.synchronize()
    return metadata


@skip_pre_blackwell
@pytest.mark.parametrize("replay", [False, True], ids=["replay_off", "replay_on"])
def test_metadata_exposes_replay_rows(replay: bool):
    """A batch of [hit context, cold context, generation]: the hit's first rows are
    replay rows; the compressed stream and its write slots start at the reuse
    boundary; the replay rows' window starts at the replay start."""
    manager, sparse_config = make_manager(replay)
    requests = []
    try:
        first = make_request(0, prompt(1, 1000))
        run_context(manager, first)
        manager.free_resources(first)

        hit = make_request(1, first.get_tokens(0)[:701] + prompt(2, 299))  # 1000 tokens
        cold = make_request(2, prompt(3, 257))
        gen = make_request(3, prompt(4, 130))
        for request in (hit, cold, gen):
            requests.append(request)
            assert manager.prepare_context(request)
            assert manager.resize_context(request, request.context_chunk_size)
        claimed = manager.kv_cache_map[hit.py_request_id].num_committed_tokens
        reused = 700 if replay else claimed
        num_replay = WINDOW if replay else 0
        start = reused - num_replay
        assert hit.context_current_position == start
        cached = [start, 0, 129]
        new_lens = [1000 - start, 257, 1]
        metadata = build_metadata(manager, sparse_config, requests, 2, cached, new_lens)
        num_tokens = sum(new_lens)
        rows_of = [range(0, new_lens[0]), range(new_lens[0], new_lens[0] + 257), [num_tokens - 1]]

        assert metadata.v41_num_replay_tokens_cpu[:3].tolist() == [num_replay, 0, 0]
        assert metadata.v41_num_replay_tokens_cuda[:3].tolist() == [num_replay, 0, 0]
        assert metadata.v41_num_replay_tokens_total == num_replay
        is_replay = metadata.v41_is_replay_token_cuda[:num_tokens].tolist()
        assert is_replay == [t < num_replay for t in range(num_tokens)]
        assert metadata.v41_num_compute_tokens == num_tokens - num_replay
        compute = metadata.v41_compute_token_index_cuda[: num_tokens - num_replay].tolist()
        assert compute == list(range(num_replay, num_tokens))
        compute_start = [start + num_replay, 0, 129]
        assert metadata.v41_compute_start_pos_cuda[:3].tolist() == compute_start
        compute_lens = [new_lens[0] - num_replay, 257, 1]
        assert metadata.v41_compute_cu_seq_lens_cuda[:4].tolist() == [
            0,
            compute_lens[0],
            compute_lens[0] + 257,
            compute_lens[0] + 258,
        ]

        # Compressed stream: new entries from the compute start only, so no cached
        # main-KV / index-K entry of the replay rows is written.
        for r in (1, 2):
            kv_lens = [c + n for c, n in zip(cached, new_lens)]
            new_comp = [k // r - s // r for k, s in zip(kv_lens, compute_start)]
            assert metadata.new_comp_kv_lens_cuda[r][:3].tolist() == new_comp
            assert metadata.past_kv_lens_cuda[r][:3].tolist() == [s // r for s in compute_start]
            assert metadata.num_total_compressed_tokens[r] == sum(new_comp)
            positions = metadata.compressed_position_ids_cuda[r][: new_comp[0]].tolist()
            assert positions == [(compute_start[0] // r + k) * r for k in range(new_comp[0])]
            slots = metadata.v41_compress_slot_mapping[r][: new_comp[0]].cpu()
            table = metadata.compress_block_tables[r][0].cpu()
            per_page = TOKENS_PER_BLOCK // r
            for k, slot in zip(range(compute_start[0] // r, kv_lens[0] // r), slots.tolist()):
                assert slot == int(table[k // per_page]) * per_page + k % per_page

        # Window of each query: truncated at the replay start for the hit's rows.
        swa = metadata.swa_local_indices_cuda[:num_tokens].cpu()
        floors = [start if replay else 0, 0, 0]
        assert metadata.v41_swa_floor_cuda[:3].tolist() == floors
        # The context attention's seq_lens: kv length minus the floor.
        kv_lens = [c + n for c, n in zip(cached, new_lens)]
        seq_lens = metadata.v41_swa_seq_lens_cuda[:2].tolist()
        assert seq_lens == [kv_lens[0] - floors[0], kv_lens[1]]
        for b, rows in enumerate(rows_of):
            for j, t in enumerate(rows):
                p = cached[b] + j
                want = list(range(max(floors[b], p - WINDOW + 1), p + 1))
                row = swa[t].tolist()
                assert row[: len(want)] == want, (b, p)
                assert all(i == -1 for i in row[len(want) :]), (b, p)
                if b < 2:
                    # The kernel's window count covers exactly the window: no -1.
                    assert min(WINDOW, seq_lens[b] - new_lens[b] + j + 1) == len(want), (b, p)
        # Every row writes its window entry, replay rows included.
        assert int((metadata.v41_swa_slot_mapping[0, :num_tokens] < 0).sum()) == 0

        # The Engram history before the replay start, as engram_hash input rows.
        if replay:
            ids = hit.get_tokens(0)[start - ENGRAM_HISTORY : start]
            assert metadata.v41_replay_history_len == ENGRAM_HISTORY
            assert metadata.v41_replay_history_token_ids_cuda[:3].tolist() == ids
            assert metadata.v41_replay_history_req_idx_cuda[:3].tolist() == [0, 0, 0]
            assert metadata.v41_replay_history_cu_seq_lens_cuda[:3].tolist() == [0, 3, 3]
            assert metadata.v41_replay_history_start_pos_cuda[:2].tolist() == [start - 3, 0]
            table = metadata.engram_token_block_table[0].cpu()
            for p in range(start - ENGRAM_HISTORY, reused):
                assert int(table[p // TOKENS_PER_BLOCK]) >= 0, p
        else:
            assert metadata.v41_replay_history_len == 0

        # A later chunk of the same request carries no replay rows.
        later = build_metadata(
            manager, sparse_config, requests[:1], 1, [start + 256], [1000 - start - 256]
        )
        assert later.v41_num_replay_tokens_total == 0
        assert later.v41_compute_start_pos_cuda[:1].tolist() == [start + 256]
        assert later.v41_swa_seq_lens_cuda[:1].tolist() == [1000]
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()
