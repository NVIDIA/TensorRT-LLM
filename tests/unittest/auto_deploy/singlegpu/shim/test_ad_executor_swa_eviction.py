# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the shim's SWA front-eviction metadata path.

Drives a real two-window CachedSequenceInterface + KVCacheManager (so the
historical `get_cache_indices` payload reflects real C++ page-table state),
then monkeypatches `get_num_front_blocks_removed` to simulate eviction at
controlled counts.  The shim helper `_compute_window_local_view` must
produce a window-coherent local-coord view: live page slice, window-capped
seq_len_with_cache, derived last_page_len, and a coherent spillover slot.

`get_num_front_blocks_removed` is monkeypatched (not driven by real decode)
because firing real SWA eviction from Python without a model requires
either spinning up a full ADEngine + model or adding a test-only C++
binding to bump the counter directly — both are out of scope.  The patch
covers exactly what the C++ side would otherwise do: bump a counter
without mutating mCacheBlockIds, so the historical page list still
contains the evicted-but-stale-entries-at-the-front pattern that the
shim is supposed to handle.
"""

import pytest
import torch
from _model_test_utils import default_max_num_tokens

from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import KVPagedResourceHandler
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import _compute_window_local_view
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# Geometry shared across cases: SWA window of 64 tokens with tokens_per_block=32
# gives 2 pages per window — small enough to exercise eviction boundaries
# without ballooning runtime.
SWA_WINDOW = 64
FULL_WINDOW = 128
TOKENS_PER_BLOCK = 32
PAGES_PER_SWA_WINDOW = SWA_WINDOW // TOKENS_PER_BLOCK  # 2


@pytest.fixture
def two_window_interface():
    """A real CachedSequenceInterface hosting a dual-pool KVCacheManager.

    Layer 0 lives in an SWA window (W=64); layer 1 lives in a full window.
    This is the Gemma-4 geometry — exactly the path Phase 2 fixes.
    """
    interface = CachedSequenceInterface(
        max_seq_len=FULL_WINDOW,
        max_batch_size=2,
        max_num_tokens=default_max_num_tokens(FULL_WINDOW, 2),
        device="cuda",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            max_tokens=1024,
            free_gpu_memory_fraction=0.0,
        ),
    )
    interface.add_resource(
        "kv_swa", KVPagedResourceHandler(4, 32, dtype=torch.float16, sliding_window=SWA_WINDOW)
    )
    interface.add_resource("kv_full", KVPagedResourceHandler(4, 32, dtype=torch.float16))
    interface.initialize_resources()
    return interface


def _add_request(manager, request_id: int, token_num: int):
    """Add a dummy request that allocates pages for `token_num` tokens."""
    out = manager.add_dummy_requests([request_id], token_nums=[token_num])
    assert out is not None and len(out) == 1, "manager did not allocate the request"
    return out[0]


def test_helper_no_eviction_passes_through(two_window_interface):
    """Sanity guard: front_removed=0 reproduces the pre-Phase-2 view.

    Asserts `all_indices[:num_active]`, full window-capped
    seq_len_with_cache, and the next slot as the spillover page.
    """
    manager = two_window_interface.kv_cache_manager
    req = _add_request(manager, request_id=42, token_num=SWA_WINDOW)

    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)
    assert manager.get_num_front_blocks_removed(req.py_request_id, window_size=SWA_WINDOW) == 0

    active_indices, extra_page, swc, lpl = _compute_window_local_view(
        all_indices,
        front_removed=0,
        end_compute_i=SWA_WINDOW,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    assert active_indices == list(all_indices[:PAGES_PER_SWA_WINDOW])
    assert swc == SWA_WINDOW
    assert lpl == TOKENS_PER_BLOCK
    # No spillover slot when num_active fully consumes the historical list.
    if len(all_indices) > PAGES_PER_SWA_WINDOW:
        assert extra_page == all_indices[PAGES_PER_SWA_WINDOW]
    else:
        assert extra_page == -1


@pytest.mark.parametrize("front_removed", [1, PAGES_PER_SWA_WINDOW])
def test_helper_with_simulated_eviction_slices_correctly(
    two_window_interface, monkeypatch, front_removed
):
    """Eviction-aware slicing and window-capping.

    With front_removed > 0, the slice must start at front_removed and
    seq_len_with_cache must stay within the window even when end_compute_i
    represents many evicted-token-worths of progress.

    Simulates a request that has cumulatively seen
    ``front_removed * tokens_per_block + SWA_WINDOW`` tokens, i.e.
    ``front_removed`` pages have already been front-evicted on the SWA pool.
    """
    manager = two_window_interface.kv_cache_manager
    # Allocate enough pages to span (evicted + live + a spillover slot) so
    # get_cache_indices returns a realistic-length list.
    total_tokens = front_removed * TOKENS_PER_BLOCK + SWA_WINDOW + 1
    req = _add_request(manager, request_id=43, token_num=total_tokens)

    # Bump the eviction counter the way detachFrontBlock would (the C++ side
    # bumps `mNumFrontBlocksRemovedPerWindow[ws]` without mutating
    # mCacheBlockIds, so get_cache_indices STILL returns the full list and
    # the shim must slice [front_removed:] off it).
    real_query = manager.get_num_front_blocks_removed

    def fake_front_removed(request_id, window_size=None):
        if request_id == req.py_request_id and window_size == SWA_WINDOW:
            return front_removed
        return real_query(request_id, window_size=window_size)

    monkeypatch.setattr(manager, "get_num_front_blocks_removed", fake_front_removed)

    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)
    front = manager.get_num_front_blocks_removed(req.py_request_id, window_size=SWA_WINDOW)
    assert front == front_removed

    # end_compute_i is the global position; the helper must cap it at the window.
    end_compute_i = total_tokens
    active_indices, extra_page, swc, lpl = _compute_window_local_view(
        all_indices,
        front_removed=front,
        end_compute_i=end_compute_i,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # 1. Live slice starts at front_removed, exactly the pages a Phase-2-aware
    #    shim hands the kernel.
    expected_slice = list(all_indices[front_removed : front_removed + PAGES_PER_SWA_WINDOW])
    assert active_indices == expected_slice, (
        f"slice mismatch: front_removed={front_removed}, "
        f"all_indices={list(all_indices)}, active={active_indices}"
    )

    # 2. seq_len_with_cache is window-capped: this is the contract the kernel
    #    relies on (total_kv_len - q_len ≤ W in local coords).
    assert swc == SWA_WINDOW
    assert swc <= SWA_WINDOW, f"swc={swc} exceeded window={SWA_WINDOW}"

    # 3. last_page_len consistent with seq_len_with_cache.
    expected_lpl = (swc - 1) % TOKENS_PER_BLOCK + 1 if swc > 0 else 0
    assert lpl == expected_lpl

    # 4. The page table the kernel sees holds enough capacity for the
    #    cached portion (the integration invariant called out in the
    #    backlog: len(active) * PAGE_SIZE >= seq_len_with_cache - q_len).
    q_len = 0  # Pure cache-coverage check, ignore new tokens here.
    assert len(active_indices) * TOKENS_PER_BLOCK >= swc - q_len


def test_helper_caps_seq_len_with_cache_below_window(two_window_interface, monkeypatch):
    """Below-window end_compute_i reports actual progress, not the window size.

    When end_compute_i is BELOW the window (early prefill / short request),
    seq_len_with_cache reports the actual progress, not the window size — the
    helper must not over-pad.
    """
    manager = two_window_interface.kv_cache_manager
    req = _add_request(manager, request_id=44, token_num=SWA_WINDOW)

    monkeypatch.setattr(manager, "get_num_front_blocks_removed", lambda req_id, window_size=None: 0)
    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)

    short_end = TOKENS_PER_BLOCK + 5  # < SWA_WINDOW
    _, _, swc, lpl = _compute_window_local_view(
        all_indices,
        front_removed=0,
        end_compute_i=short_end,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    assert swc == short_end
    assert lpl == ((short_end - 1) % TOKENS_PER_BLOCK + 1)


def test_helper_does_not_consume_evicted_extra_slot(two_window_interface, monkeypatch):
    """Spillover slot must live AFTER the live window, never inside evicted range.

    Pre-Phase-2 code that referenced `all_indices[num_active]` would land
    inside the evicted region whenever front_removed > 0 — this test guards
    against that.
    """
    manager = two_window_interface.kv_cache_manager
    front_removed = 2
    total_tokens = front_removed * TOKENS_PER_BLOCK + SWA_WINDOW + 1
    req = _add_request(manager, request_id=45, token_num=total_tokens)
    monkeypatch.setattr(
        manager,
        "get_num_front_blocks_removed",
        lambda req_id, window_size=None: front_removed,
    )

    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)
    _, extra_page, _, _ = _compute_window_local_view(
        all_indices,
        front_removed=front_removed,
        end_compute_i=total_tokens,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    expected_extra_idx = front_removed + PAGES_PER_SWA_WINDOW
    if len(all_indices) > expected_extra_idx:
        assert extra_page == all_indices[expected_extra_idx]
        # Crucially, the spillover slot is NOT in the front-evicted range.
        assert expected_extra_idx >= front_removed
    else:
        assert extra_page == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
