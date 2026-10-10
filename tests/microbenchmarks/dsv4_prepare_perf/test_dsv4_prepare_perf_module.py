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
"""Module-level perf sanity for the DeepSeek-V4 block-table host path.

Scope: the block-table preparation that ``attn_metadata.prepare()`` performs on
the generation path — ``compute_sliding_block_tables`` plus the sliding,
per-compression-ratio and indexer table copies. This is host code; it is
measured with dispatch counts, not wall-clock. See ``prepare_perf_harness`` for
why, and ``tests/microbenchmarks/attention_perf`` for the discrete/continuous
split this follows.

What each test guards
---------------------
``test_mapping_walks_once_per_step``
    The request->slot mapping is used by three independent table builders. It is
    deterministic in ``(request_ids, num_contexts, beam_width)`` and returns a
    view of one shared pinned buffer, so it must be walked once per step, not
    once per consumer. Zero threshold: an integer count.

``test_dispatch_count_is_batch_invariant``
    Growing the batch must not grow the per-request Python work. The bounded
    quantity is the *slope*: dispatches must not scale with ``num_seqs``. This
    catches a per-request loop reintroduced anywhere in the sequence, which is
    the regression shape that hurts most at the batch sizes decode runs at.

``test_dispatch_total_matches_golden``
    Zero-threshold ``==`` against a blessed per-(case, torch version) golden, so
    any change in the dispatch structure has to be looked at and re-blessed
    rather than drifting silently. Bootstraps by skipping when no golden exists.

Run:
    pytest tests/microbenchmarks/dsv4_prepare_perf -m discrete -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from prepare_perf_harness import (
    DispatchCounter,
    PrepareSignals,
    count_mapping_walks,
    sliding_block_tables_shape,
)
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

_HERE = Path(__file__).parent
_GOLDEN_PATH = _HERE / "golden_prepare.json"

# Shapes: one compressed + one sparse + one dense layer is enough to exercise all
# three table builders (sliding, per-ratio compress, indexer compress) while
# keeping allocation small enough for a unit-test GPU.
_COMPRESS_RATIOS = [1, 4, 128]
_TOKENS_PER_BLOCK = 128
_MAX_SEQ_LEN = 2048
_HEAD_DIM = 512
_INDEX_HEAD_DIM = 128
_WINDOW_SIZE = 128
_VOCAB_SIZE = 129280


def _make_cache_manager(max_batch_size: int) -> DeepseekV4CacheManager:
    """Build a real cache manager, no model.

    Mirrors the helper in
    ``tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py``.
    """
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=_INDEX_HEAD_DIM,
        window_size=_WINDOW_SIZE,
        compress_ratios=_COMPRESS_RATIOS,
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=_MAX_SEQ_LEN * max_batch_size,
        event_buffer_max_size=0,
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    return DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(_COMPRESS_RATIOS),
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MAX_SEQ_LEN,
        max_batch_size=max_batch_size,
        max_input_len=_MAX_SEQ_LEN,
        mapping=mapping,
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=_VOCAB_SIZE,
        max_num_tokens=max_batch_size * (_MAX_SEQ_LEN + 1),
        sparse_attn_config=sparse_attn_config,
    )


def _prepare_block_tables(cache_manager, request_ids: List[int], num_seqs: int) -> None:
    """One generation-step block-table preparation, in production order.

    `compute_sliding_block_tables` runs first (it seeds the per-step state the
    other builders reuse), then the three table copies.
    """
    sliding = torch.empty(
        sliding_block_tables_shape(cache_manager, num_seqs),
        dtype=torch.int32,
        device="cuda",
    )
    compress_dst = torch.empty(
        num_seqs, cache_manager.max_blocks_per_seq, dtype=torch.int32, device="cpu"
    )
    indexer_dst = torch.empty(
        num_seqs, cache_manager.max_blocks_per_seq, dtype=torch.int32, device="cpu"
    )

    with torch.cuda.stream(cache_manager._stream):
        cache_manager.compute_sliding_block_tables(request_ids, num_contexts=0)
        cache_manager.copy_batch_sliding_block_tables(
            sliding, request_ids, num_contexts=0, num_seqs=num_seqs
        )
        for ratio in cache_manager._compress_ratios:
            if ratio <= 1:
                continue
            cache_manager.copy_batch_compress_block_tables(
                compress_dst,
                request_ids,
                compress_ratio=ratio,
                beam_width=1,
                num_contexts=0,
                num_seqs=num_seqs,
            )
        cache_manager.copy_batch_indexer_compress_block_tables(
            indexer_dst, request_ids, beam_width=1, num_contexts=0, num_seqs=num_seqs
        )
    cache_manager._stream.synchronize()


def _collect(num_seqs: int) -> Tuple[PrepareSignals, DeepseekV4CacheManager]:
    """Measure one prepare for a decode-shaped batch of `num_seqs` requests."""
    mgr = _make_cache_manager(max_batch_size=max(num_seqs, 2))
    request_ids = list(range(num_seqs))
    mgr.add_dummy_requests(
        request_ids=request_ids,
        token_nums=[_TOKENS_PER_BLOCK + 1] * num_seqs,
        is_gen=True,
    )

    # Warm up once: the first call allocates the per-step staging buffers, and we
    # want the steady-state count, not the one-off allocations.
    _prepare_block_tables(mgr, request_ids, num_seqs)

    n_ratios = sum(1 for r in mgr._compress_ratios if r > 1)
    with count_mapping_walks(mgr) as walks:
        with DispatchCounter() as counter:
            _prepare_block_tables(mgr, request_ids, num_seqs)

    sig = PrepareSignals(
        case_id="dsv4_prepare_block_tables",
        num_seqs=num_seqs,
        num_ratios=n_ratios,
        dispatch_total=counter.total,
        mapping_walks=walks.calls if walks.observed else None,
        per_op=dict(counter.by_op),
    )
    return sig, mgr


# --------------------------------------------------------------------------- #
# golden handling — bless explicitly, never auto-write (attention_perf §10.4)
# --------------------------------------------------------------------------- #


def _golden_key() -> str:
    """Key the golden on the torch minor version, not on the GPU.

    Dispatch counts depend on the torch version's decompositions and are
    independent of the device, which is why this differs from ``attention_perf``
    (that harness keys its gpu_time goldens on ``get_device_name``).
    """
    return "torch" + ".".join(torch.__version__.split(".")[:2])


def _load_golden() -> dict:
    if not _GOLDEN_PATH.exists():
        return {}
    with open(_GOLDEN_PATH) as fh:
        return json.load(fh)


def _golden_for(case_id: str, num_seqs: int):
    entry = _load_golden().get(case_id, {}).get(_golden_key(), {})
    return entry.get(str(num_seqs))


def _bootstrap_or_skip(case_id: str, num_seqs: int, observed):
    golden = _golden_for(case_id, num_seqs)
    if golden is None:
        pytest.skip(
            f"[bootstrap] no golden for {case_id} num_seqs={num_seqs} on "
            f"{_golden_key()}. observed={observed!r}. Confirm the value is stable "
            f"across runs, then add it to {_GOLDEN_PATH.name}."
        )
    return golden


# --------------------------------------------------------------------------- #
# DISCRETE tests — zero-threshold structural asserts, pre-merge gate
# --------------------------------------------------------------------------- #


@skip_pre_blackwell
@pytest.mark.discrete
def test_mapping_walks_once_per_step():
    """The shared request->slot mapping must be computed once per step.

    Three builders consume it with identical arguments. Walking it per consumer
    costs O(num_seqs) extra dispatches each time and was measured at ~0.9 ms per
    generation iteration at batch 64 on GB300.
    """
    sig, _ = _collect(num_seqs=8)
    if sig.mapping_walks is None:
        pytest.skip("get_copy_index not interceptable in this build")
    assert sig.mapping_walks == 1, (
        f"request->slot mapping walked {sig.mapping_walks}x in one step; expected 1. "
        f"Each extra walk re-derives an identical result. {sig.describe()}"
    )


@skip_pre_blackwell
@pytest.mark.discrete
def test_dispatch_count_is_batch_invariant():
    """Per-request Python/ATen work must not scale with the batch.

    Compares two batch sizes rather than asserting an absolute number, so the
    test states the property (no per-request loop) instead of pinning an
    implementation detail. A tolerance is allowed for genuinely per-request
    device work; the failure this catches is a *slope*, not a constant.
    """
    small, large = 4, 32
    sig_small, _ = _collect(num_seqs=small)
    sig_large, _ = _collect(num_seqs=large)

    growth = sig_large.dispatch_total - sig_small.dispatch_total
    # 3 dispatches per extra request is generous: a reintroduced per-request loop
    # over 5 sliding types x 3 layers would add hundreds.
    budget = 3 * (large - small)
    assert growth <= budget, (
        f"dispatch count scales with batch: {sig_small.dispatch_total} at "
        f"num_seqs={small} -> {sig_large.dispatch_total} at num_seqs={large} "
        f"(+{growth}, budget +{budget}). A per-request host loop was likely "
        f"reintroduced. top ops small={sorted(sig_small.per_op.items(), key=lambda kv: -kv[1])[:5]} "
        f"large={sorted(sig_large.per_op.items(), key=lambda kv: -kv[1])[:5]}"
    )


@skip_pre_blackwell
@pytest.mark.discrete
@pytest.mark.parametrize("num_seqs", [4, 32])
def test_dispatch_total_matches_golden(num_seqs):
    """Zero-threshold gate on the exact dispatch count.

    Any change here — better or worse — should be seen and re-blessed rather than
    drifting. On a host with no golden the test skips and prints the observed
    value for blessing.
    """
    sig, _ = _collect(num_seqs=num_seqs)
    golden = _bootstrap_or_skip(sig.case_id, num_seqs, sig.dispatch_total)
    assert sig.dispatch_total == golden, (
        f"dispatch count changed: observed={sig.dispatch_total} golden={golden} "
        f"at num_seqs={num_seqs}. If this is an intended improvement, re-bless "
        f"the golden; if not, a host-side regression was introduced. "
        f"top ops={sorted(sig.per_op.items(), key=lambda kv: -kv[1])[:8]}"
    )
