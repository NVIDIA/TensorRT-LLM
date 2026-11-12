# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the async H2D staging race in MixedPrecisionKVStaging.update().

update() fills this batch's page tables and lengths on the host, then issues an asynchronous
H2D copy (``copy_(non_blocking=True)``). If either source is a per-call temporary that goes out of
scope when update() returns, its memory can be recycled or overwritten before the in-flight DMA
retires -- silently tearing this batch's KV page indices and lengths on the device.

The test makes the race deterministic without relying on allocator timing: it queues the copies
behind an in-flight GPU workload so they stay pending, overwrites every CPU int32 source allocated
during the update() call while those copies are still outstanding, then checks the device buffers
against a clean baseline captured from a plain (fully drained) update. A persistent host staging
buffer (allocated once in __init__, kept alive for the process) survives the overwrite; a per-call
temporary does not -- so this asserts exactly the property the fix adds.

A duck-typed manager is used instead of KVCacheManagerV2 so the test drives the real update() H2D
path under test without coupling to KvCacheConfig's mixed-precision schema or GPU pool allocation;
the race is a property of staging.update() alone, not of how the mixed-precision pools are sized.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_staging import MixedPrecisionKVStaging

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device for the async H2D path"
)

_CLOBBER = 777777


class _StagingManager:
    """Minimal stand-in exposing only what MixedPrecisionKVStaging consumes from its manager."""

    tokens_per_block = 128
    mixed_precision_max_seq_len = 512  # -> max_pages = ceil(512 / 128) = 4
    kv_factor = 2
    layer_offsets = {0: 0}
    mixed_precision_high_layers = {0}
    dtype = torch.bfloat16
    mixed_precision_low_dtype = torch.float32

    def __init__(self, rows):
        self.kv_cache_map = {i: SimpleNamespace(capacity=rows * 128) for i in range(rows)}

    def get_buffers(self, layer_idx, layout, mixed_precision_pool=None):
        # Pool tensors; only .shape/.dtype matter to MixedPrecisionKVStaging.__init__.
        return torch.zeros((self.kv_factor, 2, 4), dtype=torch.bfloat16)

    def mixed_precision_page_table_key(self, layer_idx):
        return ((layer_idx,),)  # one shared page-table owner for the whole batch

    def fill_mixed_precision_page_tables(self, request_ids, owner, out_np):
        # Deterministic per-(request, role) values, stable across calls, so a clobber to _CLOBBER is detectable.
        n = len(request_ids)
        for i in range(n):
            for role in range(out_np.shape[0]):
                out_np[role, i, :] = 1000 * (i + 1) + 7 * role


def _make_staging(rows=4):
    return MixedPrecisionKVStaging(
        _StagingManager(rows),
        get_memory_buffers(),
        rows=rows,
        layout="NHD",
        pool_layout="NHD",
        tokens_per_block=128,
    )


def _queue_gated_update(view, request_ids, cached, seq_lens):
    """Run update() with its H2D copies queued behind an in-flight matmul; capture the CPU sources.

    Returns (captured_sources, gate_stream, gate_event); on return the copies are still pending.
    """
    gate_stream, gate_event = torch.cuda.Stream(), torch.cuda.Event()
    with torch.cuda.stream(gate_stream):
        a = torch.ones((8192, 8192), dtype=torch.float16, device="cuda")
        for _ in range(24):
            a = a @ a
        gate_event.record(gate_stream)

    # The copies run on the current stream; hold it behind the pending workload so they stay queued.
    torch.cuda.current_stream().wait_event(gate_event)

    captured = []
    full, tensor = torch.full, torch.tensor

    def is_source(t):
        return isinstance(t, torch.Tensor) and t.is_cpu and t.dtype == torch.int32

    def wrap(orig):
        def fn(*args, **kwargs):
            result = orig(*args, **kwargs)
            if is_source(result):
                captured.append(result)
            return result

        return fn

    torch.full, torch.tensor = wrap(full), wrap(tensor)
    try:
        view.update(request_ids, cached, seq_lens, capture_graph=False)
    finally:
        torch.full, torch.tensor = full, tensor

    assert not gate_event.query(), "gate already retired -- the copies were never in flight"
    return captured, gate_stream, gate_event


def test_update_keeps_async_h2d_sources_alive():
    rows = 4
    view = _make_staging(rows)
    request_ids = list(range(rows))
    cached = [128, 128, 256, 0]
    seq_lens = [128] * rows

    # Baseline: a plain update plus full drain gives the correct device page tables and lengths.
    view.update(request_ids, cached, seq_lens, capture_graph=False)
    torch.cuda.synchronize()
    base_tables = {layer: view.page_tables[layer].clone() for layer in view.pools}
    base_lengths = view.lengths.clone()

    # Raced: hold the copies pending behind an in-flight workload and clobber every CPU source.
    captured, gate_stream, gate_event = _queue_gated_update(view, request_ids, cached, seq_lens)
    for source in captured:
        source.fill_(_CLOBBER)
    gate_stream.synchronize()
    torch.cuda.current_stream().synchronize()

    # A persistent staging buffer survives the overwrite; a per-call temporary does not.
    assert all(torch.equal(view.page_tables[layer], base_tables[layer]) for layer in view.pools), (
        "page tables changed while their H2D copy was still pending"
    )
    assert torch.equal(view.lengths, base_lengths), (
        "lengths changed while their H2D copy was still pending"
    )
