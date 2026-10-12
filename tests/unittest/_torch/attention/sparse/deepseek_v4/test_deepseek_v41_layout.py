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
"""DeepSeek-V4.1 layout mode of DeepseekV4CacheManager and DeepseekV4TrtllmAttentionMetadata."""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, List, Optional, Sequence, Tuple

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DEFAULT_KV_STORAGE,
    DEEPSEEK_V41_KV_STORAGE_BF16,
    DEEPSEEK_V41_KV_STORAGE_ENV,
    DEEPSEEK_V41_KV_STORAGE_NATIVE,
    DeepseekV41Layout,
    deepseek_v41_native_entry_split,
    deepseek_v41_sparse_attention_config,
    deepseek_v41_token_bytes,
    get_deepseek_v41_kv_storage,
    get_deepseek_v41_layout,
    is_deepseek_v41_config,
    use_deepseek_v41_layout_mode,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    CSA2SparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, AttnLifeCycle, PageIndexMode

T = DeepseekV4AttentionType
TOKENS_PER_BLOCK = 128
WINDOW = 128
INDEX_TOPK = 512
HEAD_DIM = 512
INDEX_HEAD_DIM = 128

# The checkpoint's text_config fields the layout reads (config.json of
# DeepSeek-V4.1-Flash-NVFP4), 40 backbone layers plus 3 DSpark stages.
V41_TEXT_FIELDS = dict(
    model_type="deepseek_v41_text",
    num_hidden_layers=40,
    num_nextn_predict_layers=3,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    engram_layer_ids=[1, 14],
    engram_max_ngram_size=4,
    sliding_window=WINDOW,
    head_dim=HEAD_DIM,
    index_head_dim=INDEX_HEAD_DIM,
    index_topk=INDEX_TOPK,
    kv_lora_rank=448,
    qk_rope_head_dim=64,
)


def make_v41_config(**overrides) -> SimpleNamespace:
    """A composite config as ModelConfig sees it: text fields mirrored to the top level."""
    text = dict(V41_TEXT_FIELDS, **overrides)
    top = {k: v for k, v in text.items() if k != "model_type"}
    return SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **top,
    )


@contextmanager
def kv_storage_env(kv_storage: Optional[str]) -> Iterator[None]:
    """Set TRTLLM_DSV41_KV_STORAGE for the block (None: unset); the previous value is restored."""
    saved = os.environ.get(DEEPSEEK_V41_KV_STORAGE_ENV)
    try:
        if kv_storage is None:
            os.environ.pop(DEEPSEEK_V41_KV_STORAGE_ENV, None)
        else:
            os.environ[DEEPSEEK_V41_KV_STORAGE_ENV] = kv_storage
        yield
    finally:
        if saved is None:
            os.environ.pop(DEEPSEEK_V41_KV_STORAGE_ENV, None)
        else:
            os.environ[DEEPSEEK_V41_KV_STORAGE_ENV] = saved


# Bytes per entry: SWA / main KV in bf16 containers or native, index K MXFP4 in both.
ENTRY_BYTES = {
    DEEPSEEK_V41_KV_STORAGE_BF16: {"SWA": 1024, "COMPRESS": 1024, "INDEXER_COMPRESS": 68},
    DEEPSEEK_V41_KV_STORAGE_NATIVE: {"SWA": 528, "COMPRESS": 288, "INDEXER_COMPRESS": 68},
}
# Persistent (global) KV bytes per token, R16: 3 * (C + 68) / 2 + (C + 68).
PERSISTENT_BYTES_PER_TOKEN = {
    DEEPSEEK_V41_KV_STORAGE_BF16: 2730,
    DEEPSEEK_V41_KV_STORAGE_NATIVE: 890,
}


# ---------------------------------------------------------------------------
# Reference: the checkpoint's inference/model.py index rules.
# ---------------------------------------------------------------------------


def ref_get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """model.py:409-426 (without the lru_cache)."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def ref_window_positions(position: int) -> List[int]:
    """Window positions a query at ``position`` attends, per the reference.

    A decode step (one query at start_pos = position) sees the ring; its slots
    are mapped back to the positions they hold. A query inside a prompt sees
    row ``position`` of the causal prefill rule (start_pos == 0); chunked
    prefill and verify rows are queries inside the prompt at their absolute
    positions.
    """
    if position == 0:
        row = ref_get_window_topk_idxs(WINDOW, 1, 1, 0)[0, 0].tolist()
        return sorted(i for i in row if i >= 0)
    ring = ref_get_window_topk_idxs(WINDOW, 1, 1, position)[0, 0].tolist()
    decode = sorted(position - ((position - s) % WINDOW) for s in ring if s >= 0)
    prefill_row = ref_get_window_topk_idxs(WINDOW, 1, position + 1, 0)[0, position].tolist()
    prefill = sorted(i for i in prefill_row if i >= 0)
    assert decode == prefill, (position, decode, prefill)
    return prefill


def ref_valid_compressed(position: int, end_pos: int, ratio: int) -> int:
    """Valid entries of the reference indexer top-k row (model.py:563-580).

    ``compress_lens`` = (i + 1) // ratio in prefill, end_pos // ratio in
    decode; ``topk = min(index_topk, end_pos // ratio)``; entries at or past
    compress_lens become -1, and sort after the valid ones.
    """
    topk = min(INDEX_TOPK, end_pos // ratio)
    return min(topk, (position + 1) // ratio)


# ---------------------------------------------------------------------------
# Layout (host only).
# ---------------------------------------------------------------------------


def test_layout_detection_and_role_table():
    config = make_v41_config()
    assert is_deepseek_v41_config(config)
    assert is_deepseek_v41_config(config.text_config)
    assert not is_deepseek_v41_config(
        SimpleNamespace(architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4")
    )
    assert get_deepseek_v41_layout(SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])) is None

    layout = get_deepseek_v41_layout(config)
    assert layout.num_layers == 40
    assert layout.compressed_ratios == (1, 2)
    assert layout.layer_ratios == (0, 1, 2)
    # plan.md section 2 "Layer roles".
    for layer in (0, 1):
        assert layout.kv_source(layer) is None and layout.index_source(layer) is None
        assert layout.attention_types(layer) == (
            (T.SWA, T.ENGRAM_TOKEN) if layer == 1 else (T.SWA,)
        )
    for layer in range(2, 20):
        assert layout.kv_source(layer) == max(s for s in (2, 8, 14) if s <= layer)
        assert layout.index_source(layer) == layout.kv_source(layer)
    for layer in range(20, 40):
        assert layout.kv_source(layer) == 20
        assert layout.index_source(layer) == max(s for s in (20, 24, 28, 32, 36) if s <= layer)
    owners = {
        attn_type: [layer for layer in range(40) if layout.has_attention(layer, attn_type)]
        for attn_type in T
    }
    assert owners[T.SWA] == list(range(40))
    assert owners[T.COMPRESS] == [2, 8, 14, 20]
    assert owners[T.INDEXER_COMPRESS] == [2, 8, 14, 20]
    assert owners[T.COMPRESSOR_KV] == [2, 8, 14]
    assert owners[T.COMPRESSOR_SCORE] == [2, 8, 14]
    assert owners[T.ENGRAM_TOKEN] == [1]
    assert owners[T.INDEXER_COMPRESSOR_KV] == []
    assert owners[T.INDEXER_COMPRESSOR_SCORE] == []
    assert layout.owner(14, T.ENGRAM_TOKEN) == 1
    assert layout.owner(27, T.COMPRESS) == 20
    assert layout.owner(27, T.COMPRESSOR_KV) is None
    # Windows: 128 + D, 2 + D, 4 + D (history 3 plus the step's own position;
    # plus the 128-token replay span under encoder SWA bounded replay);
    # persistent roles have none.
    assert layout.window(T.SWA, 5) == 133
    assert layout.window(T.COMPRESSOR_KV, 5) == 7
    assert layout.window(T.ENGRAM_TOKEN, 5) == 9 + (WINDOW if layout.encoder_replay else 0)
    assert layout.window(T.COMPRESS, 5) is None
    # R16: 3 * (C + 68) / 2 + (C + 68), C = 1024 (bf16 containers) or 288 (native).
    for storage, expected in PERSISTENT_BYTES_PER_TOKEN.items():
        pinned = DeepseekV41Layout.from_config(config, kv_storage=storage)
        assert pinned.kv_storage == storage
        assert pinned.persistent_bytes_per_token(range(40)) == expected, storage


@pytest.mark.parametrize("max_draft_len", [0, 5])
def test_windows_keep_every_position_the_step_reads(max_draft_len: int):
    """Each windowed role's window, read through KVCacheManagerV2's live-window rule,
    keeps every position a step starting at history ``c`` reads.

    KVCacheManagerV2 treats ``[c + 1 - window, c]`` as live (the window counts the
    in-flight token at position ``c``); blocks before it are stale and are neither
    kept for the request nor mapped on a prefix hit. What the V4.1 forward reads
    before position ``c``:

    - SWA: ``c - 127 .. c - 1`` (the attention window of token ``c``);
    - compressor pending pair: ``c - 1`` when ``c`` is odd (ratio 2);
    - Engram token history: ``c - 3 .. c - 1`` (the 4-gram ending at ``c``).

    Measured before the Engram window counted the in-flight token: a prefix hit at
    ``c = 4610`` (``c % 128 == 2``) left the page of ``c - 3`` unmapped.
    """
    layout = get_deepseek_v41_layout(make_v41_config())
    reads = {
        T.SWA: lambda c: range(max(0, c - (layout.window_size - 1)), c),
        T.COMPRESSOR_KV: lambda c: range(c - 1, c) if c % 2 else range(0),
        T.COMPRESSOR_SCORE: lambda c: range(c - 1, c) if c % 2 else range(0),
        T.ENGRAM_TOKEN: lambda c: range(max(0, c - layout.engram_history_len), c),
    }
    for attn_type, needed in reads.items():
        lc = AttnLifeCycle.make(layout.window(attn_type, max_draft_len), 0, TOKENS_PER_BLOCK)
        for c in range(1, 4 * TOKENS_PER_BLOCK + 3):
            stale = lc.get_stale_range(c, TOKENS_PER_BLOCK)
            for p in needed(c):
                assert not (stale.beg <= p // TOKENS_PER_BLOCK < stale.end), (attn_type, c, p)


def test_kv_storage_switch():
    with kv_storage_env(None):
        assert get_deepseek_v41_kv_storage() == DEEPSEEK_V41_DEFAULT_KV_STORAGE
        assert get_deepseek_v41_layout(make_v41_config()).kv_storage == (
            DEEPSEEK_V41_DEFAULT_KV_STORAGE
        )
    for value, expected in (
        ("bf16", DEEPSEEK_V41_KV_STORAGE_BF16),
        (" Native ", DEEPSEEK_V41_KV_STORAGE_NATIVE),
        ("", DEEPSEEK_V41_DEFAULT_KV_STORAGE),
    ):
        with kv_storage_env(value):
            assert get_deepseek_v41_kv_storage() == expected, value
            assert get_deepseek_v41_layout(make_v41_config()).kv_storage == expected, value
    with kv_storage_env("fp8"), pytest.raises(ValueError, match="is not one of"):
        get_deepseek_v41_layout(make_v41_config())

    # Native entries: codes, then block scales.
    assert deepseek_v41_native_entry_split(T.SWA, HEAD_DIM) == (512, 16)
    assert deepseek_v41_native_entry_split(T.COMPRESS, HEAD_DIM) == (256, 32)
    with pytest.raises(ValueError):
        deepseek_v41_native_entry_split(T.INDEXER_COMPRESS, HEAD_DIM)
    for storage, table in ENTRY_BYTES.items():
        for name, expected in table.items():
            assert (
                deepseek_v41_token_bytes(T[name], HEAD_DIM, INDEX_HEAD_DIM, storage) == expected
            ), (storage, name)
        # The windowed state roles do not depend on the storage mode.
        assert deepseek_v41_token_bytes(T.COMPRESSOR_KV, HEAD_DIM, INDEX_HEAD_DIM, storage) == 2048
        assert deepseek_v41_token_bytes(T.ENGRAM_TOKEN, HEAD_DIM, INDEX_HEAD_DIM, storage) == 4


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(compress_ratios=[0, 0, 4] + [1] * 40), "must be in"),
        (dict(kv_source_layer_ids=[8, 14, 20]), "no KV-source layer"),
        (dict(index_source_layer_ids=[2, 8, 14, 24]), "index"),
        (dict(kv_source_layer_ids=[1, 8, 14, 20]), "ratio 0"),
    ],
)
def test_layout_rejects_inconsistent_roles(overrides, message):
    with pytest.raises(ValueError, match=message):
        get_deepseek_v41_layout(make_v41_config(**overrides))


def test_v4_role_predicates_unchanged():
    assert T.ENGRAM_TOKEN not in DEEPSEEK_V4_SLIDING_ATTENTION
    assert len(DEEPSEEK_V4_SLIDING_ATTENTION) == 5
    for ratio in (1, 4, 128):
        assert not compress_ratio_has_attention(ratio, T.ENGRAM_TOKEN)
    # Disaggregated synthetic layer ids keep DeepSeek-V4's encoding width (7 roles);
    # only the DeepSeek-V4.1 layout mode counts ENGRAM_TOKEN.
    v4_manager = DeepseekV4CacheManager.__new__(DeepseekV4CacheManager)
    assert v4_manager.v41_layout is None
    assert v4_manager.disagg_num_attn_types == 7
    v41_manager = DeepseekV4CacheManager.__new__(DeepseekV4CacheManager)
    v41_manager.v41_layout = get_deepseek_v41_layout(make_v41_config())
    assert v41_manager.disagg_num_attn_types == T.ENGRAM_TOKEN.value + 1


def test_layout_mode_sparse_config():
    """The deepseek_v4 sparse config of the layout mode, and its installation on a
    ModelConfig that from_pretrained gave the CSA2 config."""
    from tensorrt_llm._torch.model_config import ModelConfig

    config = make_v41_config()
    sparse = deepseek_v41_sparse_attention_config(config)
    assert sparse.algorithm == "deepseek_v4"
    # A placeholder: the manager and the metadata read the layout from the checkpoint.
    assert sparse.compress_ratios == [1] * len(V41_TEXT_FIELDS["compress_ratios"])
    assert sparse.window_size == WINDOW
    assert (sparse.index_head_dim, sparse.index_topk) == (INDEX_HEAD_DIM, INDEX_TOPK)
    assert sparse.indexer_k_dtype == "fp4"
    with pytest.raises(ValueError, match="compress_ratios"):
        deepseek_v41_sparse_attention_config(make_v41_config(compress_ratios=[]))

    model_config = ModelConfig(pretrained_config=config)
    assert isinstance(model_config.sparse_attention_config, CSA2SparseAttentionConfig)
    model_config._frozen = True
    use_deepseek_v41_layout_mode(model_config)
    assert model_config._frozen
    installed = model_config.sparse_attention_config
    assert isinstance(installed, DeepSeekV4SparseAttentionConfig)
    assert installed == sparse
    use_deepseek_v41_layout_mode(model_config)  # idempotent
    assert model_config.sparse_attention_config is installed

    # A CSA2 config the user tuned would be dropped silently: refused instead.
    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=CSA2SparseAttentionConfig(use_fp8_staging=False),
    )
    with pytest.raises(ValueError, match="use_fp8_staging"):
        use_deepseek_v41_layout_mode(model_config)
    # Not a DeepSeek-V4.1 checkpoint.
    with pytest.raises(ValueError, match="DeepSeek-V4.1"):
        use_deepseek_v41_layout_mode(
            SimpleNamespace(
                pretrained_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
                sparse_attention_config=None,
            )
        )


@pytest.mark.parametrize("pool_ratio", [None, [1.0]])
def test_v41_cache_config_keeps_the_long_decode_constraint(pool_ratio):
    """While the pool split is derived, the layout mode's pools must hold one request at
    max_seq_len beside minimal decode requests filling the resident batch, as DeepSeek-V4's
    do. Without it, a 1,048,576-token, 256-request manager got 9,400 SWA pages instead of
    10,920 and could not hold the 119th request of a step."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BatchDesc,
        GpuCacheTierConfig,
        KVCacheDesc,
        KVCacheManagerConfig,
    )

    with kv_storage_env(DEEPSEEK_V41_KV_STORAGE_NATIVE):
        layout = get_deepseek_v41_layout(make_v41_config())
    manager = object.__new__(DeepseekV4CacheManager)
    manager.v41_layout = layout
    manager.pp_layers = list(range(layout.num_layers))
    manager._max_draft_len = 0
    manager.head_dim = HEAD_DIM
    manager.index_head_dim = INDEX_HEAD_DIM
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.compressed_block_sizes = [
        TOKENS_PER_BLOCK // ratio if ratio else TOKENS_PER_BLOCK for ratio in layout.compress_ratios
    ]
    manager.max_seq_len = 1 << 20
    manager.max_batch_size = 256
    manager.is_estimating_kv_cache = False
    manager.max_cuda_graph_batch_size = None
    manager.max_attention_window_vec = [None] * layout.num_layers
    manager.max_draft_len = 0
    manager.num_extra_kv_tokens = 0

    context_constraint = BatchDesc([KVCacheDesc(capacity=16384, history_length=0)])
    base_config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
        layers=[],
        constraints=[context_constraint] if pool_ratio is None else [],
        initial_pool_ratio=pool_ratio,
    )
    config = manager._build_cache_config(base_config)

    assert len(config.layers) == manager._num_manager_layers
    if pool_ratio is None:
        assert config.constraints == [
            context_constraint,
            BatchDesc(
                [KVCacheDesc(capacity=1 << 20, history_length=(1 << 20) - 1)]
                + [KVCacheDesc(capacity=1, history_length=0)] * 255
            ),
        ]
        assert base_config.constraints == [context_constraint]
    else:
        assert config.initial_pool_ratio == pool_ratio
        assert config.constraints == []


def test_pipeline_stage_split():
    """A stage runs on its own pools only; for this checkpoint that leaves boundaries at 1 and 20."""
    layout = get_deepseek_v41_layout(make_v41_config())
    assert layout.pipeline_boundaries() == (1, 20)
    # PP2's even split and PP3 as 1 | 19 | 20 are valid.
    for stage in (range(0, 20), range(20, 40), range(0, 1), range(1, 20)):
        assert layout.stage_dependency_error(stage) is None, stage
    cases = {
        range(10, 20): "layer 10 reads its KV source layer 8",
        range(21, 40): "layer 21 reads its KV source layer 20",
        range(30, 40): "layer 30 reads its KV source layer 20",
        range(14, 40): "Engram layer 14 reads the token history of layer 1",
    }
    for stage, message in cases.items():
        err = layout.stage_dependency_error(stage)
        assert err is not None and err.startswith(message), (stage, err)
    # A re-index layer needs the candidate source's mask (a synthetic role table
    # where layer 24 has a KV source of its own but still applies layer 20's mask).
    other = get_deepseek_v41_layout(make_v41_config(kv_source_layer_ids=[2, 8, 14, 20, 24]))
    err = other.stage_dependency_error(range(24, 40))
    assert err is not None and "candidate-block mask of layer 20" in err, err


# ---------------------------------------------------------------------------
# Cache manager and metadata (GPU).
# ---------------------------------------------------------------------------


def create_v41_manager(
    max_batch_size: int = 8,
    max_seq_len: int = 4096,
    spec_config: Optional[object] = None,
    enable_swa_scratch_reuse: bool = True,
    config: Optional[SimpleNamespace] = None,
    kv_storage: Optional[str] = DEEPSEEK_V41_KV_STORAGE_BF16,
    mapping: Optional[Mapping] = None,
) -> Tuple[DeepseekV4CacheManager, DeepSeekV4SparseAttentionConfig]:
    """A V4.1 manager; ``kv_storage`` pins TRTLLM_DSV41_KV_STORAGE while it is built (None: unset)."""
    with kv_storage_env(kv_storage):
        return _create_v41_manager(
            max_batch_size, max_seq_len, spec_config, enable_swa_scratch_reuse, config, mapping
        )


def _create_v41_manager(
    max_batch_size: int,
    max_seq_len: int,
    spec_config: Optional[object],
    enable_swa_scratch_reuse: bool,
    config: Optional[SimpleNamespace],
    mapping: Optional[Mapping] = None,
) -> Tuple[DeepseekV4CacheManager, DeepSeekV4SparseAttentionConfig]:
    config = make_v41_config() if config is None else config
    # What ModelConfig synthesizes for V4.1: a placeholder all-1 ratio list.
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW,
        compress_ratios=[1] * len(config.compress_ratios),
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=max_seq_len * max_batch_size,
        event_buffer_max_size=0,
        enable_swa_scratch_reuse=enable_swa_scratch_reuse,
    )
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
        mapping=mapping
        if mapping is not None
        else Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=8192,
        sparse_attn_config=sparse_config,
        pretrained_config=config,
        spec_config=spec_config,
    )
    return manager, sparse_config


def make_request(request_id: int, prompt_len: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=64,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def allocate(manager: DeepseekV4CacheManager, lengths: Sequence[int]) -> List[LlmRequest]:
    requests = []
    for request_id, length in enumerate(lengths):
        request = make_request(request_id, length)
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        requests.append(request)
    return requests


def reference_page_indices(
    manager: DeepseekV4CacheManager,
    request_ids: List[int],
    num_contexts: int,
    layer_idx: int,
    attn_type: DeepseekV4AttentionType,
    page_index_mode: PageIndexMode,
) -> torch.Tensor:
    """Page table rows through the V2 manager's own Python page-index converter."""
    layer_id = manager._layer_attn_to_layer_id[layer_idx, attn_type]
    pool_id = manager.layer_to_pool_mapping_dict[layer_id]
    converter = manager.impl.get_page_index_converter(layer_id, attn_type.role)
    copy_idx = manager.index_mapper.get_copy_index(request_ids, num_contexts, 1)
    expected = torch.full(
        (len(request_ids), manager.max_blocks_per_seq), BAD_PAGE_INDEX, dtype=torch.int32
    )
    for row, request_id in enumerate(request_ids):
        base = manager.host_kv_cache_block_offsets[pool_id, int(copy_idx[row]), 0].tolist()
        scratch = None
        if row < num_contexts and page_index_mode == PageIndexMode.PER_LAYER:
            scratch = manager.kv_cache_map[request_id].get_scratch_desc(pool_id)
        converted = converter(base, page_index_mode, scratch)
        expected[row, : len(converted)] = torch.tensor(converted, dtype=torch.int32)
    return expected


@skip_pre_blackwell
def test_manager_allocates_owner_pools_only():
    manager, _ = create_v41_manager()
    try:
        layout = manager.v41_layout
        assert layout is not None
        keys = set(manager._layer_attn_to_layer_id)
        expected = {
            (layer, attn_type)
            for layer in range(40)
            for attn_type in T
            if layout.has_attention(layer, attn_type)
        }
        assert keys == expected
        # Readers own no compressed pool; the V4 per-ratio predicates are not used.
        assert (3, T.COMPRESS) not in keys and (24, T.INDEXER_COMPRESS) not in keys
        assert not any(t in (T.INDEXER_COMPRESSOR_KV, T.INDEXER_COMPRESSOR_SCORE) for _, t in keys)
        assert manager.get_disagg_ignored_roles() == frozenset()
        assert manager.num_sliding_attention_columns == T.ENGRAM_TOKEN.value + 1

        # Page geometry and element types of every role.
        geometry = {
            (0, T.SWA): (TOKENS_PER_BLOCK, HEAD_DIM, torch.bfloat16),
            (39, T.SWA): (TOKENS_PER_BLOCK, HEAD_DIM, torch.bfloat16),
            (2, T.COMPRESS): (64, HEAD_DIM, torch.bfloat16),
            (8, T.COMPRESS): (64, HEAD_DIM, torch.bfloat16),
            (20, T.COMPRESS): (128, HEAD_DIM, torch.bfloat16),
            (2, T.INDEXER_COMPRESS): (64, 68, torch.uint8),
            (20, T.INDEXER_COMPRESS): (128, 68, torch.uint8),
            (14, T.COMPRESSOR_KV): (TOKENS_PER_BLOCK, HEAD_DIM, torch.float32),
            (14, T.COMPRESSOR_SCORE): (TOKENS_PER_BLOCK, HEAD_DIM, torch.float32),
            (1, T.ENGRAM_TOKEN): (TOKENS_PER_BLOCK, 1, torch.int32),
        }
        for (layer, attn_type), (entries, width, dtype) in geometry.items():
            buffer = manager.get_buffers(layer, attn_type)
            assert buffer.shape[1:] == (entries, width), (layer, attn_type, buffer.shape)
            assert buffer.dtype == dtype, (layer, attn_type, buffer.dtype)
        assert manager.get_indexer_k_cache_buffers(20).shape[1:] == (128, 1, 68)
        assert manager.compressed_block_sizes[2] == 64
        assert manager.compressed_block_sizes[33] == 128

        # Distinct owners of one ratio have distinct pool bases.
        bases = {manager.get_buffers(owner, T.COMPRESS).data_ptr() for owner in (2, 8, 14, 20)}
        assert len(bases) == 4
        assert manager.compress_pool_ptrs[2] == manager.get_buffers(2, T.COMPRESS).data_ptr()
        assert manager.compress_pool_ptrs[1] == manager.get_buffers(20, T.COMPRESS).data_ptr()

        # Windowed vs persistent lifecycles.
        windows = {
            layer_config.layer_id: layer_config.sliding_window_size
            for layer_config in manager.kv_cache_manager_py_config.layers
        }
        for (layer, attn_type), layer_id in manager._layer_attn_to_layer_id.items():
            assert windows[layer_id] == layout.window(attn_type, 0), (layer, attn_type)

        # Core global KV: main KV bf16 + native MXFP4 index K on the four owners.
        assert manager.get_cache_bytes_per_token() == 2730
        pool_bytes_per_token = 0
        for owner in (2, 8, 14, 20):
            for attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
                buffer = manager.get_buffers(owner, attn_type)
                page_bytes = buffer.shape[1] * buffer.shape[2] * buffer.element_size()
                pool_bytes_per_token += page_bytes // TOKENS_PER_BLOCK
        assert pool_bytes_per_token == 2730
        model_config = SimpleNamespace(
            pretrained_config=make_v41_config(),
            sparse_attention_config=None,
            spec_config=None,
        )
        with kv_storage_env(DEEPSEEK_V41_KV_STORAGE_BF16):
            per_token, per_batch = DeepseekV4CacheManager.get_cache_size_per_token(
                model_config, Mapping(), tokens_per_block=TOKENS_PER_BLOCK, max_batch_size=2
            )
        assert per_token == 2730
        # Windows per request, each up to two pages (a window straddles a page
        # boundary): SWA on 40 layers, the compressor pair on 3, Engram on 1
        # (three pages under encoder replay, whose Engram window spans the replay).
        engram_pages = 3 if get_deepseek_v41_layout(make_v41_config()).encoder_replay else 2
        assert per_batch == 2 * TOKENS_PER_BLOCK * (
            2 * (40 * 1024 + 3 * 2 * 2048) + engram_pages * 4
        )
    finally:
        manager.shutdown()


@skip_pre_blackwell
def test_manager_rejects_unsupported_v41_settings():
    with pytest.raises(ValueError, match="is not one of"):
        create_v41_manager(kv_storage="fp8")
    with pytest.raises(ValueError, match="must equal"):
        create_v41_manager(config=make_v41_config(sliding_window=64))


@skip_pre_blackwell
def test_manager_native_pools():
    """TRTLLM_DSV41_KV_STORAGE=native: uint8 SWA / main-KV entries, 890 B/token."""
    manager, _ = create_v41_manager(kv_storage=DEEPSEEK_V41_KV_STORAGE_NATIVE)
    try:
        layout = manager.v41_layout
        assert layout.kv_storage == DEEPSEEK_V41_KV_STORAGE_NATIVE
        # The same owner-only roles as bf16 storage; only the entry format changes.
        keys = set(manager._layer_attn_to_layer_id)
        assert keys == {
            (layer, attn_type)
            for layer in range(40)
            for attn_type in T
            if layout.has_attention(layer, attn_type)
        }

        geometry = {
            **{(layer, T.SWA): (TOKENS_PER_BLOCK, 528, torch.uint8) for layer in range(40)},
            (2, T.COMPRESS): (64, 288, torch.uint8),
            (8, T.COMPRESS): (64, 288, torch.uint8),
            (14, T.COMPRESS): (64, 288, torch.uint8),
            (20, T.COMPRESS): (128, 288, torch.uint8),
            (2, T.INDEXER_COMPRESS): (64, 68, torch.uint8),
            (20, T.INDEXER_COMPRESS): (128, 68, torch.uint8),
            (14, T.COMPRESSOR_KV): (TOKENS_PER_BLOCK, HEAD_DIM, torch.float32),
            (14, T.COMPRESSOR_SCORE): (TOKENS_PER_BLOCK, HEAD_DIM, torch.float32),
            (1, T.ENGRAM_TOKEN): (TOKENS_PER_BLOCK, 1, torch.int32),
        }
        for (layer, attn_type), (entries, width, dtype) in geometry.items():
            buffer = manager.get_buffers(layer, attn_type)
            assert buffer.shape[1:] == (entries, width), (layer, attn_type, buffer.shape)
            assert buffer.dtype == dtype, (layer, attn_type, buffer.dtype)
            assert buffer.is_contiguous(), (layer, attn_type)
        assert manager.get_indexer_k_cache_buffers(20).shape[1:] == (128, 1, 68)
        # Every SWA view starts at the shared SWA base, as in bf16 storage.
        assert {manager.get_buffers(layer, T.SWA).data_ptr() for layer in range(40)} == {
            manager.swa_pool_ptr
        }

        # (codes, scales) views: codes first, then scales, within each entry.
        for layer, attn_type, codes_width, scales_width in (
            (0, T.SWA, 512, 16),
            (39, T.SWA, 512, 16),
            (8, T.COMPRESS, 256, 32),
            (20, T.COMPRESS, 256, 32),
        ):
            buffer = manager.get_buffers(layer, attn_type)
            codes, scales = manager.get_v41_native_kv_views(layer, attn_type)
            entry = codes_width + scales_width
            assert codes.dtype == scales.dtype == torch.uint8
            assert codes.shape == buffer.shape[:2] + (codes_width,)
            assert scales.shape == buffer.shape[:2] + (scales_width,)
            assert codes.stride() == scales.stride() == (buffer.shape[1] * entry, entry, 1)
            assert codes.data_ptr() == buffer.data_ptr()
            assert scales.data_ptr() == buffer.data_ptr() + codes_width
        for layer, attn_type in ((2, T.INDEXER_COMPRESS), (14, T.COMPRESSOR_KV)):
            with pytest.raises(RuntimeError, match="not a DeepSeek-V4.1 native pool"):
                manager.get_v41_native_kv_views(layer, attn_type)

        # Distinct owners of one ratio have distinct pool bases.
        bases = {manager.get_buffers(owner, T.COMPRESS).data_ptr() for owner in (2, 8, 14, 20)}
        assert len(bases) == 4
        assert manager.compress_pool_ptrs[2] == manager.get_buffers(2, T.COMPRESS).data_ptr()
        assert manager.compress_pool_ptrs[1] == manager.get_buffers(20, T.COMPRESS).data_ptr()

        # Windows are unchanged by the storage mode.
        windows = {
            layer_config.layer_id: layer_config.sliding_window_size
            for layer_config in manager.kv_cache_manager_py_config.layers
        }
        for (layer, attn_type), layer_id in manager._layer_attn_to_layer_id.items():
            assert windows[layer_id] == layout.window(attn_type, 0), (layer, attn_type)

        # 890 B/token three ways: the manager's figure, the pools' page geometry,
        # and the engine's static sizing hook.
        assert manager.get_cache_bytes_per_token() == 890
        pool_bytes_per_token = 0
        for owner in (2, 8, 14, 20):
            for attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
                buffer = manager.get_buffers(owner, attn_type)
                page_bytes = buffer.shape[1] * buffer.shape[2] * buffer.element_size()
                pool_bytes_per_token += page_bytes // TOKENS_PER_BLOCK
        assert pool_bytes_per_token == 890
        model_config = SimpleNamespace(
            pretrained_config=make_v41_config(),
            sparse_attention_config=None,
            spec_config=None,
        )
        with kv_storage_env(DEEPSEEK_V41_KV_STORAGE_NATIVE):
            per_token, per_batch = DeepseekV4CacheManager.get_cache_size_per_token(
                model_config, Mapping(), tokens_per_block=TOKENS_PER_BLOCK, max_batch_size=2
            )
        assert per_token == 890
        # SWA windows at 528 B/entry; compressor pair and Engram unchanged.
        engram_pages = 3 if get_deepseek_v41_layout(make_v41_config()).encoder_replay else 2
        assert per_batch == 2 * TOKENS_PER_BLOCK * (
            2 * (40 * 528 + 3 * 2 * 2048) + engram_pages * 4
        )

        # Pool sizing follows the native pools: for the same max_tokens the
        # quota is smaller and the persistent pools still hold max_tokens.
        bf16_manager, _ = create_v41_manager(kv_storage=DEEPSEEK_V41_KV_STORAGE_BF16)
        try:
            assert bf16_manager.get_cache_bytes_per_token() == 2730
            bf16_quota = int(bf16_manager.impl.get_quota(0))
        finally:
            bf16_manager.shutdown()
        assert int(manager.impl.get_quota(0)) < bf16_quota
        max_tokens = 4096 * 8  # create_v41_manager: max_seq_len * max_batch_size
        for owner in (2, 20):
            buffer = manager.get_buffers(owner, T.COMPRESS)
            assert buffer.shape[0] * buffer.shape[1] * layout.compress_ratios[owner] >= max_tokens
    finally:
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "kv_storage", [DEEPSEEK_V41_KV_STORAGE_BF16, DEEPSEEK_V41_KV_STORAGE_NATIVE]
)
@pytest.mark.parametrize("scratch_reuse", [False, True])
def test_manager_page_tables_per_ratio(scratch_reuse: bool, kv_storage: str):
    # max_batch_size sizes the windowed pools for 8 resident requests; three
    # concurrent fresh contexts do not fit V2's pool split at max_batch_size=3
    # with SWA scratch reuse (one context plus two decodes is its typical step).
    manager, _ = create_v41_manager(enable_swa_scratch_reuse=scratch_reuse, kv_storage=kv_storage)
    requests = []
    try:
        prompt_len = 2 * TOKENS_PER_BLOCK + 1
        requests = allocate(manager, [prompt_len] * 3)
        request_ids = [r.py_request_id for r in requests]
        num_contexts = 3

        # Per-ratio COMPRESS and INDEXER_COMPRESS tables, shared by that ratio's owners.
        for ratio, owners in ((2, (2, 8, 14)), (1, (20,))):
            for attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
                actual = torch.empty(
                    len(request_ids), manager.max_blocks_per_seq, dtype=torch.int32
                )
                if attn_type == T.COMPRESS:
                    manager.copy_batch_compress_block_tables(
                        actual,
                        request_ids,
                        compress_ratio=ratio,
                        beam_width=1,
                        num_contexts=num_contexts,
                        num_seqs=len(request_ids),
                    )
                else:
                    manager.copy_batch_indexer_compress_block_tables(
                        actual,
                        request_ids,
                        beam_width=1,
                        num_contexts=num_contexts,
                        num_seqs=len(request_ids),
                        compress_ratio=ratio,
                    )
                for owner in owners:
                    expected = reference_page_indices(
                        manager, request_ids, num_contexts, owner, attn_type, PageIndexMode.SHARED
                    )
                    torch.testing.assert_close(actual, expected)
                assert (actual[:, :3] != BAD_PAGE_INDEX).all()

        # Sliding tables: SWA on every layer, compressor pair, Engram history.
        shape = (
            manager.num_local_layers,
            manager.num_sliding_attention_columns,
            len(request_ids),
            manager.max_blocks_per_seq,
        )
        actual = torch.empty(shape, dtype=torch.int32, device="cuda")
        with torch.cuda.stream(manager._stream):
            manager.compute_sliding_block_tables(request_ids, num_contexts=num_contexts)
            manager.copy_batch_sliding_block_tables(
                actual, request_ids, num_contexts=num_contexts, num_seqs=len(request_ids)
            )
        manager._stream.synchronize()
        actual = actual.cpu()
        for (layer, attn_type), _ in manager._layer_attn_to_layer_id.items():
            if attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
                continue
            expected = reference_page_indices(
                manager, request_ids, num_contexts, layer, attn_type, PageIndexMode.PER_LAYER
            )
            torch.testing.assert_close(
                actual[manager.layer_offsets[layer], attn_type.value], expected
            )
        # Columns of roles V4.1 never allocates stay empty.
        for attn_type in (T.INDEXER_COMPRESSOR_KV, T.INDEXER_COMPRESSOR_SCORE):
            assert (actual[:, attn_type.value] == BAD_PAGE_INDEX).all()

        # Owners of one ratio share page ids but not storage.
        table = torch.empty(len(request_ids), manager.max_blocks_per_seq, dtype=torch.int32)
        manager.copy_batch_compress_block_tables(
            table, request_ids, compress_ratio=2, beam_width=1, num_contexts=3, num_seqs=3
        )
        pages = table[:, :3].flatten().tolist()
        for value, owner in enumerate((2, 8, 14), start=1):
            manager.get_buffers(owner, T.COMPRESS)[pages] = float(value)
        for value, owner in enumerate((2, 8, 14), start=1):
            assert (manager.get_buffers(owner, T.COMPRESS)[pages] == float(value)).all()
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_manager_pp2_stage_pools(pp_rank: int):
    """PP2's even split: each stage holds its own layers' pools, and the page table of
    a ratio with no KV source on the stage is empty rather than an error."""
    mapping = Mapping(world_size=2, rank=pp_rank, tp_size=1, pp_size=2, gpus_per_node=2)
    manager, sparse_config = create_v41_manager(mapping=mapping)
    requests = []
    try:
        layout = manager.v41_layout
        layers = list(range(20)) if pp_rank == 0 else list(range(20, 40))
        assert list(manager.pp_layers) == layers
        expected = {
            (layer, attn_type)
            for layer in layers
            for attn_type in T
            if layout.has_attention(layer, attn_type)
        }
        assert set(manager._layer_attn_to_layer_id) == expected
        owners = {2: (2, 8, 14)} if pp_rank == 0 else {1: (20,)}
        absent = 1 if pp_rank == 0 else 2
        assert set(manager.compress_pool_ptrs) == set(owners)
        # Persistent bytes per token of the stage: 3 ratio-2 owners, or 1 ratio-1 owner.
        assert manager.get_cache_bytes_per_token() == (3 * 1092 // 2 if pp_rank == 0 else 1092)

        prompt_len = 2 * TOKENS_PER_BLOCK + 1
        requests = allocate(manager, [prompt_len] * 3)
        request_ids = [r.py_request_id for r in requests]
        for ratio in (1, 2):
            for attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
                actual = torch.empty(
                    len(request_ids), manager.max_blocks_per_seq, dtype=torch.int32
                )
                if attn_type == T.COMPRESS:
                    manager.copy_batch_compress_block_tables(
                        actual,
                        request_ids,
                        compress_ratio=ratio,
                        beam_width=1,
                        num_contexts=3,
                        num_seqs=3,
                    )
                else:
                    manager.copy_batch_indexer_compress_block_tables(
                        actual,
                        request_ids,
                        beam_width=1,
                        num_contexts=3,
                        num_seqs=3,
                        compress_ratio=ratio,
                    )
                if ratio == absent:
                    assert (actual == BAD_PAGE_INDEX).all(), (ratio, attn_type)
                    continue
                for owner in owners[ratio]:
                    ref = reference_page_indices(
                        manager, request_ids, 3, owner, attn_type, PageIndexMode.SHARED
                    )
                    torch.testing.assert_close(actual, ref)
        # The metadata builds on the stage (the absent ratio's slots stay -1).
        metadata = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor([prompt_len] * 3, dtype=torch.int),
            request_ids=request_ids,
            max_num_requests=manager.max_batch_size,
            num_contexts=3,
            prompt_lens=[prompt_len] * 3,
            max_num_tokens=8192,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0, 0]),
            mapping=mapping,
            sparse_attention_config=sparse_config,
        )
        metadata.prepare()
        torch.cuda.synchronize()
        total = metadata.num_total_compressed_tokens[absent]
        assert total > 0
        assert (metadata.v41_compress_slot_mapping[absent][:total] == -1).all()
        present = 2 if pp_rank == 0 else 1
        total = metadata.num_total_compressed_tokens[present]
        assert (metadata.v41_compress_slot_mapping[present][:total] >= 0).all()
        assert (metadata.engram_token_block_table is None) == (pp_rank == 1)
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
def test_manager_rejects_a_stage_split_across_a_kv_source():
    mapping = Mapping(
        world_size=2, rank=1, tp_size=1, pp_size=2, gpus_per_node=2, pp_partition=[10, 30]
    )
    with pytest.raises(
        ValueError, match="layer 10 reads its KV source layer 8.*layers \\[1, 20\\]"
    ):
        create_v41_manager(mapping=mapping)


def collect_block_tables(
    manager: DeepseekV4CacheManager, request_ids: List[int], num_contexts: int
) -> dict:
    """Per-ratio COMPRESS / INDEXER_COMPRESS tables and all sliding tables of one batch."""
    num_seqs = len(request_ids)
    tables = {}
    for ratio in (1, 2):
        table = torch.empty(num_seqs, manager.max_blocks_per_seq, dtype=torch.int32)
        manager.copy_batch_compress_block_tables(
            table, request_ids, ratio, beam_width=1, num_contexts=num_contexts, num_seqs=num_seqs
        )
        tables[T.COMPRESS, ratio] = table.clone()
        manager.copy_batch_indexer_compress_block_tables(
            table,
            request_ids,
            beam_width=1,
            num_contexts=num_contexts,
            num_seqs=num_seqs,
            compress_ratio=ratio,
        )
        tables[T.INDEXER_COMPRESS, ratio] = table.clone()
    sliding = torch.empty(
        manager.num_local_layers,
        manager.num_sliding_attention_columns,
        num_seqs,
        manager.max_blocks_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    with torch.cuda.stream(manager._stream):
        manager.compute_sliding_block_tables(request_ids, num_contexts=num_contexts)
        manager.copy_batch_sliding_block_tables(
            sliding, request_ids, num_contexts=num_contexts, num_seqs=num_seqs
        )
    manager._stream.synchronize()
    tables["sliding"] = sliding.cpu()
    return tables


@skip_pre_blackwell
def test_block_tables_independent_of_kv_storage():
    """Native storage changes entry bytes only: the same batch gets the same page ids."""
    tables = {}
    for kv_storage in (DEEPSEEK_V41_KV_STORAGE_BF16, DEEPSEEK_V41_KV_STORAGE_NATIVE):
        manager, _ = create_v41_manager(kv_storage=kv_storage)
        requests = []
        try:
            requests = allocate(manager, [257, 700, 130])
            tables[kv_storage] = collect_block_tables(
                manager, [r.py_request_id for r in requests], num_contexts=3
            )
        finally:
            for request in requests:
                manager.free_resources(request)
            manager.shutdown()
    bf16, native = tables[DEEPSEEK_V41_KV_STORAGE_BF16], tables[DEEPSEEK_V41_KV_STORAGE_NATIVE]
    assert bf16.keys() == native.keys()
    for key in bf16:
        torch.testing.assert_close(native[key], bf16[key], rtol=0, atol=0, msg=str(key))
    # 130 tokens at ratio 2 are 65 entries: two 64-entry pages for every request.
    assert (native[T.COMPRESS, 2][:, :2] != BAD_PAGE_INDEX).all()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "kv_storage", [DEEPSEEK_V41_KV_STORAGE_BF16, DEEPSEEK_V41_KV_STORAGE_NATIVE]
)
def test_disagg_page_table_enumerates_v41_roles(kv_storage: str):
    manager, _ = create_v41_manager(max_batch_size=1, max_seq_len=512, kv_storage=kv_storage)
    try:
        page_table = build_page_table_from_manager(manager)
        roles = set()
        # role -> per-page byte sizes the transfer views carry for it.
        view_page_bytes = {}
        for layer_group in page_table.layer_groups:
            for pool_view in layer_group.pool_views:
                roles.update(str(role) for role in pool_view.pool_role)
                sizes = {int(entry[2]) for entry in pool_view.buffer_entries}
                assert len(sizes) == 1, (pool_view.pool_role, sizes)
                for role in pool_view.pool_role:
                    view_page_bytes.setdefault(str(role), set()).update(sizes)
        for attn_type in (
            T.SWA,
            T.COMPRESS,
            T.INDEXER_COMPRESS,
            T.COMPRESSOR_KV,
            T.COMPRESSOR_SCORE,
            T.ENGRAM_TOKEN,
        ):
            assert str(attn_type.role) in roles, (attn_type, roles)
        # The role-generic page-table walk sees the pools' own page sizes.
        entry = ENTRY_BYTES[kv_storage]
        assert view_page_bytes == {
            str(T.SWA.role): {TOKENS_PER_BLOCK * entry["SWA"]},
            str(T.COMPRESS.role): {64 * entry["COMPRESS"], 128 * entry["COMPRESS"]},
            str(T.INDEXER_COMPRESS.role): {64 * 68, 128 * 68},
            str(T.COMPRESSOR_KV.role): {TOKENS_PER_BLOCK * 2048},
            str(T.COMPRESSOR_SCORE.role): {TOKENS_PER_BLOCK * 2048},
            str(T.ENGRAM_TOKEN.role): {TOKENS_PER_BLOCK * 4},
        }
    finally:
        manager.shutdown()


# Each scenario: (prompt_len used for allocation, cached tokens, new tokens) per
# request, context requests first.
PREFILL = [(1, 0, 1), (130, 0, 130), (700, 0, 700)]
CHUNKED = [(1000, 512, 300), (1500, 1024, 476), (257, 0, 257)]
DECODE = [(2, 1, 1), (128, 127, 1), (129, 128, 1), (2049, 2048, 1)]
VERIFY = [(40, 34, 6), (300, 294, 6), (1100, 1094, 6)]


def build_metadata(
    manager, sparse_config, requests, num_contexts, cached, new_lens, max_draft_len: int = 0
):
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
    if max_draft_len > 0:
        # What the engine does before prepare() when drafting.
        metadata.update_spec_dec_param(
            batch_size=len(requests),
            is_spec_decoding_enabled=False,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_draft_len,
        )
    metadata.prepare()
    torch.cuda.synchronize()
    return metadata


@skip_pre_blackwell
@pytest.mark.parametrize(
    "scenario, contexts",
    [(PREFILL, True), (CHUNKED, True), (DECODE, False), (VERIFY, False)],
    ids=["prefill", "chunked_prefill", "decode", "dspark_verify"],
)
def test_metadata_indices_match_reference(scenario, contexts):
    manager, sparse_config = create_v41_manager()
    requests = []
    try:
        requests = allocate(manager, [alloc for alloc, _, _ in scenario])
        cached = [c for _, c, _ in scenario]
        new_lens = [n for _, _, n in scenario]
        num_contexts = len(scenario) if contexts else 0
        metadata = build_metadata(manager, sparse_config, requests, num_contexts, cached, new_lens)
        assert metadata.is_deepseek_v41
        assert metadata.compress_ratio_set == {1, 2}
        assert set(metadata.sparse_mla_topk_lens) == {0, 1, 2}

        num_tokens = sum(new_lens)
        swa = metadata.swa_local_indices_cuda[:num_tokens].cpu()
        topk_lens = {r: metadata.sparse_mla_topk_lens[r][:num_tokens].cpu() for r in (0, 1, 2)}
        visible = {r: metadata.compressed_visible_lens_cuda[r][:num_tokens].cpu() for r in (1, 2)}
        req_idx = metadata.req_idx_per_token[:num_tokens].cpu()

        token = 0
        for b, (c, n) in enumerate(zip(cached, new_lens)):
            end_pos = c + n
            for j in range(n):
                p = c + j
                assert int(req_idx[token]) == b
                row = swa[token].tolist()
                want = ref_window_positions(p)
                # Valid window rows packed at the front in position order, -1 after.
                assert row[: len(want)] == want, (p, row[:8])
                assert all(i == -1 for i in row[len(want) :]), p
                assert int(topk_lens[0][token]) == WINDOW
                for r in (1, 2):
                    assert int(visible[r][token]) == (p + 1) // r
                    assert int(topk_lens[r][token]) == WINDOW + ref_valid_compressed(p, end_pos, r)
                token += 1

        # Per-ratio compressed lengths, new entries and group-first positions.
        for r in (1, 2):
            kv_lens = [c + n for c, n in zip(cached, new_lens)]
            comp = metadata.compressed_kv_lens_cuda[r][: len(scenario)].cpu().tolist()
            new_comp = metadata.new_comp_kv_lens_cuda[r][: len(scenario)].cpu().tolist()
            assert comp == [k // r for k in kv_lens]
            assert new_comp == [k // r - c // r for k, c in zip(kv_lens, cached)]
            cu = metadata.cu_new_comp_kv_cuda[r][: len(scenario) + 1].cpu().tolist()
            assert cu == [0] + list(torch.tensor(new_comp).cumsum(0).tolist())
            positions = metadata.compressed_position_ids_cuda[r].cpu().tolist()
            for b, c in enumerate(cached):
                for k in range(new_comp[b]):
                    # RoPE position of a compressed entry: its group's first token.
                    assert positions[cu[b] + k] == (c // r + k) * r
            if contexts:
                assert metadata.max_ctx_compressed_tokens[r] == max(new_comp)

        # Page tables per ratio, and the Engram history table.
        request_ids = [r.py_request_id for r in requests]
        for r, owner in ((2, 2), (1, 20)):
            expected = reference_page_indices(
                manager, request_ids, num_contexts, owner, T.COMPRESS, PageIndexMode.SHARED
            )
            torch.testing.assert_close(
                metadata.compress_block_tables[r][: len(scenario)].cpu(), expected
            )
            expected = reference_page_indices(
                manager, request_ids, num_contexts, owner, T.INDEXER_COMPRESS, PageIndexMode.SHARED
            )
            torch.testing.assert_close(
                metadata.indexer_compress_block_tables[r][: len(scenario)].cpu(), expected
            )
        expected = reference_page_indices(
            manager, request_ids, num_contexts, 1, T.ENGRAM_TOKEN, PageIndexMode.PER_LAYER
        )
        torch.testing.assert_close(
            metadata.engram_token_block_table[: len(scenario)].cpu(), expected
        )
        for r in (1, 2):
            assert metadata.max_compressed_kv_lens[r] == metadata.max_seq_len // r
            assert metadata.max_compressed_kv_lens[r] >= int(visible[r].max())

        # Pool pointers resolve reader layers to their owners (no KeyError).
        for layer in range(40):
            owner = manager.v41_layout.kv_source(layer)
            if owner is None:
                assert layer not in metadata.compressed_buffer_ptrs
            else:
                assert (
                    metadata.compressed_buffer_ptrs[layer]
                    == manager.get_buffers(owner, T.COMPRESS).data_ptr()
                )
            assert metadata.swa_buffer_ptrs[layer] == manager.swa_pool_ptr
        assert metadata.sparse_mla_base_ptrs[0] == manager.swa_pool_ptr
        assert metadata.sparse_mla_base_ptrs[2] == manager.compress_pool_ptrs[2]
        assert metadata.sparse_mla_base_ptrs[1] == manager.compress_pool_ptrs[1]
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


# ---------------------------------------------------------------------------
# Address book: write slots and prefill index-K inputs (GPU).
# ---------------------------------------------------------------------------

INDEX_K_DATA_BYTES = INDEX_HEAD_DIM // 2
INDEX_K_SCALE_BYTES = INDEX_HEAD_DIM // 32
INDEX_K_ENTRY_BYTES = INDEX_K_DATA_BYTES + INDEX_K_SCALE_BYTES
# The index-K / main-KV owner whose pool a ratio's pages address.
RATIO_OWNER = {2: 2, 1: 20}

# (scenario, number of context requests); contexts first. Prefill mixes odd and
# even lengths and spans up to six pages; chunked prefill has odd cached counts
# (a ratio-2 group straddles the chunk boundary); the decode batch ends near
# max_seq_len so its reserved ratio-2 slots derive positions at max_seq_len.
ADDRESS_BOOK_SCENARIOS = {
    "prefill": ([(1, 0, 1), (130, 0, 130), (257, 0, 257), (700, 0, 700), (512, 0, 512)], 5),
    "chunked_prefill": (
        [(1000, 513, 300), (1500, 1024, 476), (257, 0, 257), (900, 255, 301)],
        4,
    ),
    "decode": ([(2, 1, 1), (128, 127, 1), (129, 128, 1), (4095, 4094, 1)], 0),
    "mixed": ([(300, 0, 300), (901, 513, 388), (129, 128, 1), (2049, 2048, 1), (64, 63, 1)], 2),
    "dspark_verify": (VERIFY, 0),
}


def ref_page_slot(table: List[List[int]], b: int, entry: int, per_page: int) -> Tuple[int, int]:
    return table[b][entry // per_page], entry % per_page


def ref_index_k_offsets(page: int, slot: int, per_page: int) -> Tuple[int, int]:
    """Page-blocked MXFP4 index K: per page ``per_page`` x 64 code bytes, then x 4 scale bytes."""
    base = page * per_page * INDEX_K_ENTRY_BYTES
    return (
        base + slot * INDEX_K_DATA_BYTES,
        base + per_page * INDEX_K_DATA_BYTES + slot * INDEX_K_SCALE_BYTES,
    )


def ref_compressed_rows(cached, new_lens, num_contexts, ratio) -> List[Optional[Tuple[int, int]]]:
    """(request, entry) of every packed compressed row; None for a reserved slot.

    Context entries, then generation entries, in request order; then the
    generation requests' unused reserved slots, ceil(next_n / ratio) each.
    """
    rows = [
        (b, k)
        for b, (c, n) in enumerate(zip(cached, new_lens))
        for k in range(c // ratio, (c + n) // ratio)
    ]
    num_ctx_rows = sum(1 for b, _ in rows if b < num_contexts)
    gen_lens = new_lens[num_contexts:]
    reserved = len(gen_lens) * -(-gen_lens[0] // ratio) if gen_lens else 0
    total = num_ctx_rows + reserved
    assert len(rows) <= total
    return rows + [None] * (total - len(rows))


def read_address_book(metadata, num_tokens: int) -> dict:
    book = {"swa": metadata.v41_swa_slot_mapping[:, :num_tokens].cpu()}
    for r in (1, 2):
        total = metadata.num_total_compressed_tokens[r]
        book[("positions", r)] = metadata.compressed_position_ids_cuda[r][:total].cpu()
        book[("compress", r)] = metadata.v41_compress_slot_mapping[r][:total].cpu()
        book[("data", r)] = metadata.v41_index_k_slot_data[r][:total].cpu()
        book[("scale", r)] = metadata.v41_index_k_slot_scale[r][:total].cpu()
    return book


def check_address_book(metadata, manager, cached, new_lens, num_contexts) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
        dsv4_local_to_global_indices,
    )

    tpb = manager.tokens_per_block
    num_tokens = sum(new_lens)
    tokens = [(b, c + j) for b, (c, n) in enumerate(zip(cached, new_lens)) for j in range(n)]
    book = read_address_book(metadata, num_tokens)

    # (A) SWA write rows, every local layer.
    swa = book["swa"]
    assert metadata.v41_swa_slot_mapping.dtype == torch.int64
    assert metadata.v41_swa_slot_mapping.shape == (manager.num_local_layers, 8192)
    sliding = metadata.sliding_block_tables[:, T.SWA.value, : len(cached)].cpu().tolist()
    swa_rows = manager.get_buffers(0, T.SWA).shape[0] * tpb
    for local_layer in range(manager.num_local_layers):
        expected = []
        for b, p in tokens:
            page, slot = ref_page_slot(sliding[local_layer], b, p, tpb)
            expected.append(page * tpb + slot if page >= 0 else -1)
        got = swa[local_layer].tolist()
        assert got == expected, local_layer
        # Every token of the step has its own row in the pool.
        assert min(got) >= 0 and max(got) < swa_rows
        assert len(set(got)) == num_tokens
    # The attention reader maps each token's own window position to the same row.
    req_id = metadata.req_idx_per_token[:num_tokens]
    own_column = torch.tensor([min(p, WINDOW - 1) for _, p in tokens], device="cuda")
    for layer in range(40):
        local_layer = manager.layer_offsets[layer]
        read_rows = dsv4_local_to_global_indices(
            req_id,
            metadata.sliding_block_tables[local_layer, T.SWA.value],
            metadata.swa_local_indices_cuda[:num_tokens],
            tpb,
            0,
            None,
            None,
            0,
            0,
            False,
        )
        own = read_rows.gather(1, own_column.unsqueeze(1)).squeeze(1).cpu().to(torch.int64)
        torch.testing.assert_close(own, swa[local_layer], rtol=0, atol=0)

    # (B) Compressed stream write rows and index-K offsets, per ratio.
    for r in (1, 2):
        per_page = tpb // r
        rows = ref_compressed_rows(cached, new_lens, num_contexts, r)
        assert metadata.num_total_compressed_tokens[r] == len(rows)
        comp_table = metadata.compress_block_tables[r][: len(cached)].cpu().tolist()
        index_table = metadata.indexer_compress_block_tables[r][: len(cached)].cpu().tolist()
        mask = metadata.compressed_mask_cuda[r][: len(rows)].cpu().tolist()
        positions = book[("positions", r)].tolist()
        slots = book[("compress", r)].tolist()
        data = book[("data", r)].tolist()
        scale = book[("scale", r)].tolist()
        for buffers in (
            metadata.v41_compress_slot_mapping,
            metadata.v41_index_k_slot_data,
            metadata.v41_index_k_slot_scale,
        ):
            assert buffers[r].dtype == torch.int64 and buffers[r].shape == (8192,)
        for i, row in enumerate(rows):
            if row is None:
                assert not mask[i], (r, i)
                assert (slots[i], data[i], scale[i], positions[i]) == (-1, -1, -1, 0), (r, i)
                continue
            b, k = row
            assert mask[i] and positions[i] == k * r, (r, i)
            page, slot = ref_page_slot(comp_table, b, k, per_page)
            assert page >= 0 and slots[i] == page * per_page + slot, (r, i)
            page, slot = ref_page_slot(index_table, b, k, per_page)
            assert page >= 0 and (data[i], scale[i]) == ref_index_k_offsets(page, slot, per_page)
        # Positions feed an unchecked RoPE table lookup.
        assert all(0 <= p < metadata.max_seq_len for p in positions), r
        valid = [i for i, row in enumerate(rows) if row is not None]
        valid_slots = [slots[i] for i in valid]
        assert len(set(valid_slots)) == len(valid)
        owner = RATIO_OWNER[r]
        assert (
            max(valid_slots, default=0) < manager.get_buffers(owner, T.COMPRESS).shape[0] * per_page
        )
        index_bytes = manager.get_indexer_k_cache_buffers(owner).numel()
        assert max((data[i] for i in valid), default=0) + INDEX_K_DATA_BYTES <= index_bytes
        assert max((scale[i] for i in valid), default=0) + INDEX_K_SCALE_BYTES <= index_bytes
        if valid:
            # The compressed-segment reader maps (request, entry) to the same row.
            read_rows = dsv4_local_to_global_indices(
                torch.tensor([rows[i][0] for i in valid], dtype=torch.int32, device="cuda"),
                metadata.sliding_block_tables[0, T.SWA.value],
                torch.full((len(valid), 1), -1, dtype=torch.int32, device="cuda"),
                tpb,
                0,
                metadata.compress_block_tables[r],
                torch.tensor([[rows[i][1]] for i in valid], dtype=torch.int32, device="cuda"),
                per_page,
                0,
                False,
            )
            assert read_rows[:, 1].cpu().tolist() == valid_slots

    # (C) Prefill index-K gather offsets and per-token row ranges.
    if num_contexts == 0:
        assert metadata.v41_max_ctx_q_len == 0
        for r in (1, 2):
            assert metadata.v41_ctx_index_k_len[r] == 0
            assert metadata.v41_ctx_index_k_gather_data[r] is None
            assert metadata.v41_ctx_index_k_gather_scale[r] is None
            assert metadata.v41_ctx_index_ks[r] is None
            assert metadata.v41_ctx_index_ke[r] is None
        return
    assert metadata.v41_max_ctx_q_len == max(new_lens[:num_contexts])
    ctx_tokens = [(b, p) for b, p in tokens if b < num_contexts]
    for r in (1, 2):
        per_page = tpb // r
        index_table = metadata.indexer_compress_block_tables[r][:num_contexts].cpu().tolist()
        entries = [
            (b, k) for b in range(num_contexts) for k in range((cached[b] + new_lens[b]) // r)
        ]
        comp_lens = [(cached[b] + new_lens[b]) // r for b in range(num_contexts)]
        first_row = [sum(comp_lens[:b]) for b in range(num_contexts)]
        assert metadata.v41_ctx_index_k_len[r] == len(entries)
        gather_data = metadata.v41_ctx_index_k_gather_data[r]
        gather_scale = metadata.v41_ctx_index_k_gather_scale[r]
        assert gather_data.dtype == gather_scale.dtype == torch.int64
        assert gather_data.shape == gather_scale.shape == (len(entries),)
        expected = []
        for b, k in entries:
            page, slot = ref_page_slot(index_table, b, k, per_page)
            assert page >= 0
            expected.append(ref_index_k_offsets(page, slot, per_page))
        got = list(zip(gather_data.cpu().tolist(), gather_scale.cpu().tolist()))
        assert got == expected, r
        # The entries this step writes are gathered from where they are written.
        rows = ref_compressed_rows(cached, new_lens, num_contexts, r)
        write = dict(zip(rows, zip(book[("data", r)].tolist(), book[("scale", r)].tolist())))
        for j, (b, k) in enumerate(entries):
            if (b, k) in write:
                assert write[(b, k)] == got[j]
        ks = metadata.v41_ctx_index_ks[r]
        ke = metadata.v41_ctx_index_ke[r]
        assert ks.dtype == ke.dtype == torch.int32
        assert ks.shape == ke.shape == (len(ctx_tokens),)
        assert ks.cpu().tolist() == [first_row[b] for b, _ in ctx_tokens]
        assert ke.cpu().tolist() == [first_row[b] + (p + 1) // r for b, p in ctx_tokens]


def poison_address_book(metadata) -> None:
    metadata.v41_swa_slot_mapping.fill_(-7)
    for r in (1, 2):
        metadata.v41_compress_slot_mapping[r].fill_(-7)
        metadata.v41_index_k_slot_data[r].fill_(-7)
        metadata.v41_index_k_slot_scale[r].fill_(-7)


def assert_same_book(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=str(key))


def replay_address_book_in_cuda_graph(metadata, positions: List[int]) -> None:
    """The address-book rebuild captures, and replay reads the live page tables."""
    num_tokens = len(positions)
    token_positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
    before = read_address_book(metadata, num_tokens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        metadata._prepare_v41_address_book(token_positions)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata._prepare_v41_address_book(token_positions)
    poison_address_book(metadata)
    graph.replay()
    torch.cuda.synchronize()
    assert_same_book(read_address_book(metadata, num_tokens), before)

    # Move every allocated page one page up; replay follows the tables.
    tables = [
        metadata.sliding_block_tables,
        *metadata.compress_block_tables.values(),
        *metadata.indexer_compress_block_tables.values(),
    ]
    saved = [table.clone() for table in tables]
    try:
        for table in tables:
            table.add_((table >= 0).to(table.dtype))
        graph.replay()
        torch.cuda.synchronize()
        moved = read_address_book(metadata, num_tokens)
    finally:
        for table, original in zip(tables, saved):
            table.copy_(original)
    step = {"swa": TOKENS_PER_BLOCK}
    for r in (1, 2):
        per_page = TOKENS_PER_BLOCK // r
        step[("positions", r)] = 0
        step[("compress", r)] = per_page
        step[("data", r)] = per_page * INDEX_K_ENTRY_BYTES
        step[("scale", r)] = per_page * INDEX_K_ENTRY_BYTES
    expected = {
        key: value if step[key] == 0 else torch.where(value >= 0, value + step[key], value)
        for key, value in before.items()
    }
    assert_same_book(moved, expected)


@skip_pre_blackwell
@pytest.mark.parametrize("name", list(ADDRESS_BOOK_SCENARIOS))
def test_metadata_address_book_matches_reference(name):
    scenario, num_contexts = ADDRESS_BOOK_SCENARIOS[name]
    manager, sparse_config = create_v41_manager()
    requests = []
    try:
        requests = allocate(manager, [alloc for alloc, _, _ in scenario])
        cached = [c for _, c, _ in scenario]
        new_lens = [n for _, _, n in scenario]
        max_draft_len = new_lens[-1] - 1 if num_contexts < len(scenario) else 0
        metadata = build_metadata(
            manager, sparse_config, requests, num_contexts, cached, new_lens, max_draft_len
        )
        check_address_book(metadata, manager, cached, new_lens, num_contexts)
        num_tokens = sum(new_lens)
        positions = [c + j for c, n in zip(cached, new_lens) for j in range(n)]

        if num_contexts > 0:
            # The prefill index-K inputs are sized from host lengths: no sync.
            gathered = {r: metadata.v41_ctx_index_k_gather_data[r].clone() for r in (1, 2)}
            token_positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                metadata._prepare_v41_ctx_index_k(token_positions)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            for r in (1, 2):
                assert torch.equal(metadata.v41_ctx_index_k_gather_data[r], gathered[r])

        # on_update_kv_lens() (inside a captured step) rebuilds the address
        # book on device: poison it, and the same values come back.
        before = read_address_book(metadata, num_tokens)
        ctx_fields = {
            field: dict(getattr(metadata, field))
            for field in (
                "v41_ctx_index_k_len",
                "v41_ctx_index_k_gather_data",
                "v41_ctx_index_k_gather_scale",
                "v41_ctx_index_ks",
                "v41_ctx_index_ke",
            )
        }
        poison_address_book(metadata)
        metadata.on_update_kv_lens()
        torch.cuda.synchronize()
        assert_same_book(read_address_book(metadata, num_tokens), before)
        for field, values in ctx_fields.items():
            assert getattr(metadata, field) == values, field
        check_address_book(metadata, manager, cached, new_lens, num_contexts)

        if num_contexts == 0:
            replay_address_book_in_cuda_graph(metadata, positions)
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


# ---------------------------------------------------------------------------
# Native storage: native writers -> manager pools -> dequantizing readers (GPU).
# ---------------------------------------------------------------------------


def multiscale_rows(rows: int, block: int) -> torch.Tensor:
    """bf16 [rows, 512], one magnitude per quantization block, row 0's first block all zero."""
    x = torch.randn(rows, HEAD_DIM // block, block)
    x = x * torch.exp2(torch.randint(-12, 9, (rows, HEAD_DIM // block, 1)).float())
    x = x.reshape(rows, HEAD_DIM).to(torch.bfloat16)
    if rows > 0:
        x[0, :block] = 0
    return x.cuda()


def assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor, msg: str) -> None:
    assert actual.shape == expected.shape, (msg, actual.shape, expected.shape)
    mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    assert mismatches == 0, (msg, mismatches)


@skip_pre_blackwell
@pytest.mark.parametrize("name", ["prefill", "mixed", "dspark_verify"])
def test_native_pools_round_trip_with_native_kernels(name):
    """Native writers fill the manager's pools at the address book's rows; the
    dequantizing readers return the bf16-container value set bit for bit."""
    from tensorrt_llm._torch.custom_ops.mxfp8_dequantize import mxfp8_dequantize
    from tensorrt_llm._torch.custom_ops.mxfp8_fake_quantize import mxfp8_fake_quantize
    from tensorrt_llm._torch.custom_ops.nvfp4_fake_quantize import nvfp4_fake_quantize
    from tensorrt_llm._torch.custom_ops.nvfp4_kv_to_bf16 import (
        nvfp4_kv_to_bf16 as nvfp4_kv_dequantize,
    )

    scenario, num_contexts = ADDRESS_BOOK_SCENARIOS[name]
    manager, sparse_config = create_v41_manager(kv_storage=DEEPSEEK_V41_KV_STORAGE_NATIVE)
    requests = []
    try:
        requests = allocate(manager, [alloc for alloc, _, _ in scenario])
        cached = [c for _, c, _ in scenario]
        new_lens = [n for _, _, n in scenario]
        max_draft_len = new_lens[-1] - 1 if num_contexts < len(scenario) else 0
        metadata = build_metadata(
            manager, sparse_config, requests, num_contexts, cached, new_lens, max_draft_len
        )
        num_tokens = sum(new_lens)
        torch.manual_seed(0)

        # SWA: window KV of several layers into the one shared SWA pool, E4M3 x
        # UE8M0/32. Layer by layer, as the forward runs: with SWA scratch reuse a
        # context's out-of-window rows are scratch pages shared by all layers.
        full_swa = max(
            (manager.get_buffers(layer, T.SWA) for layer in range(40)), key=lambda b: b.shape[0]
        )
        swa_before = full_swa.clone()
        written = []
        for layer in (0, 1, 20, 39):
            codes, scales = manager.get_v41_native_kv_views(layer, T.SWA)
            x = multiscale_rows(num_tokens, 32)
            ref = torch.empty_like(x)
            mxfp8_fake_quantize(x, ref)  # the bf16-container value set
            slots = metadata.v41_swa_slot_mapping[manager.layer_offsets[layer]][:num_tokens]
            assert int(slots.min()) >= 0 and len(set(slots.tolist())) == num_tokens
            mxfp8_fake_quantize(x, codes, slots, scales)
            written.append(slots)
            # Compact gather of the step's rows.
            compact = torch.full_like(ref, float("nan"))
            mxfp8_dequantize(codes, scales, compact, src_slots=slots)
            assert_bits_equal(compact, ref, f"SWA gather, layer {layer}")
            # Shadow pool with the native pool's geometry: rows keep their numbering.
            shadow = torch.full(codes.shape, float("nan"), dtype=torch.bfloat16, device="cuda")
            mxfp8_dequantize(codes, scales, shadow, src_slots=slots, dst_slots=slots)
            assert_bits_equal(shadow.view(-1, HEAD_DIM)[slots], ref, f"SWA shadow, layer {layer}")
            del shadow
        # Nothing outside the step's rows changed.
        untouched = torch.ones(
            full_swa.shape[0] * TOKENS_PER_BLOCK, dtype=torch.bool, device="cuda"
        )
        untouched[torch.cat(written)] = False
        assert torch.equal(full_swa.view(-1, 528)[untouched], swa_before.view(-1, 528)[untouched])
        del swa_before

        # Main KV: every owner of a ratio at the same rows, E2M1 x E4M3/16, no global scale.
        global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        for r, owners in ((2, (2, 8, 14)), (1, (20,))):
            per_page = TOKENS_PER_BLOCK // r
            n = metadata.num_total_compressed_tokens[r]
            slots = metadata.v41_compress_slot_mapping[r][:n]
            valid = slots >= 0
            valid_slots = slots[valid]
            packed_rows = ref_compressed_rows(cached, new_lens, num_contexts, r)
            assert len(packed_rows) == n
            before = {
                owner: (
                    manager.get_buffers(owner, T.COMPRESS).clone(),
                    manager.get_indexer_k_cache_buffers(owner).clone(),
                )
                for owner in owners
            }
            refs = {}
            for owner in owners:
                codes, scales = manager.get_v41_native_kv_views(owner, T.COMPRESS)
                latent = multiscale_rows(n, 16)
                ref = torch.empty_like(latent)
                nvfp4_fake_quantize(latent, ref)  # the bf16-container value set
                nvfp4_fake_quantize(latent, codes, slots, scales)
                refs[owner] = ref
            torch.cuda.synchronize()
            bases = {owner: manager.get_buffers(owner, T.COMPRESS).data_ptr() for owner in owners}
            for owner in owners:
                buffer = manager.get_buffers(owner, T.COMPRESS)
                codes, scales = manager.get_v41_native_kv_views(owner, T.COMPRESS)
                ref = refs[owner]
                # Only the step's entries changed. The owners of one ratio share
                # a coalesced page slot (their bases are whole pages apart), so
                # an owner's view also spans its neighbours' entries: every
                # changed row must be one some owner wrote.
                old_buffer, old_index_k = before[owner]
                changed = (buffer.view(-1, 288) != old_buffer.view(-1, 288)).any(dim=1)
                changed = set(torch.nonzero(changed).flatten().tolist())
                written = set()
                for other, base in bases.items():
                    offset = base - bases[owner]
                    assert offset % (per_page * 288) == 0, (owner, other, offset)
                    written.update(s + offset // 288 for s in valid_slots.tolist())
                assert changed <= written, sorted(changed - written)[:8]
                assert len(changed) > 0 or valid_slots.numel() == 0
                # The index K beside the main KV is untouched.
                assert torch.equal(manager.get_indexer_k_cache_buffers(owner), old_index_k)
                if valid_slots.numel() > 0:
                    # Row gather (decode / verify top-k form): one entry per page.
                    rows = buffer.view(-1, 288)
                    out = torch.full(
                        (valid_slots.numel(), 1, 1, HEAD_DIM),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    nvfp4_kv_dequantize(
                        rows[:, :256][:, None, None],
                        rows[:, 256:][:, None, None],
                        valid_slots.to(torch.int32).view(-1, 1).contiguous(),
                        torch.ones(valid_slots.numel(), dtype=torch.int32, device="cuda"),
                        global_scale,
                        out,
                    )
                    assert_bits_equal(
                        out[:, 0, 0], ref[valid], f"COMPRESS gather r={r} owner {owner}"
                    )
                # Whole context through the per-ratio page table, for requests
                # whose every entry this step wrote.
                fresh = [b for b in range(num_contexts) if cached[b] == 0 and new_lens[b] >= r]
                if fresh:
                    lens = [new_lens[b] // r for b in fresh]
                    table = metadata.compress_block_tables[r][torch.tensor(fresh, device="cuda")]
                    out = torch.full(
                        (len(fresh), max(lens), 1, HEAD_DIM),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    nvfp4_kv_dequantize(
                        codes.unsqueeze(2),
                        scales.unsqueeze(2),
                        table.to(torch.int32).contiguous(),
                        torch.tensor(lens, dtype=torch.int32, device="cuda"),
                        global_scale,
                        out,
                    )
                    row_of = {row: i for i, row in enumerate(packed_rows) if row is not None}
                    for i, b in enumerate(fresh):
                        expected = ref[[row_of[(b, k)] for k in range(lens[i])]]
                        assert_bits_equal(
                            out[i, : lens[i], 0],
                            expected,
                            f"COMPRESS context r={r} owner {owner} b={b}",
                        )
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()
