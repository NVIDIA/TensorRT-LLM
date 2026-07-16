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
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    AttentionType,
    KVPagedResourceHandler,
)
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import (
    _compute_cyclic_full_view,
    _compute_window_local_view,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# Geometry shared across cases: SWA window of 64 tokens with tokens_per_block=32
# gives 2 pages per window — small enough to exercise eviction boundaries
# without ballooning runtime.  FULL_WINDOW doubles as the fixture's
# ``max_seq_len`` and therefore sets the SWA pool's per-sequence cap
# (``maxTokenNum = max(windowSize, maxSequenceLength)`` in kvCacheManager.cpp).
# It must exceed the longest scenario the tests construct (192-token prefill
# in ``test_helper_pre_eviction_long_prefill_passes_full_pages``) or the C++
# side clamps allocation and the helper sees fewer live pages than expected.
SWA_WINDOW = 64
FULL_WINDOW = 256
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
        "kv_swa",
        KVPagedResourceHandler(
            4, 32, dtype=torch.float16, attention_type=AttentionType.mha, sliding_window=SWA_WINDOW
        ),
    )
    interface.add_resource(
        "kv_full",
        KVPagedResourceHandler(4, 32, dtype=torch.float16, attention_type=AttentionType.mha),
    )
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
    """Eviction-aware slicing in window-local coordinates.

    With ``front_removed > 0`` the slice must start at ``front_removed`` and
    ``seq_len_with_cache`` must equal the live (window-local) cache length:
    ``end_compute_i - front_removed * tokens_per_block``.  The C++ eviction
    loop guarantees this stays within ``window + tokens_per_block``, so it
    may exceed ``SWA_WINDOW`` by up to one spillover page — the helper does
    not artificially clamp at ``SWA_WINDOW`` because doing so would discard
    a live page's worth of data that the kernel still has to read.

    Simulates a request that has cumulatively seen
    ``front_removed * tokens_per_block + SWA_WINDOW + 1`` tokens, i.e.
    ``front_removed`` pages have been front-evicted and one spillover token
    sits in the page after the window.
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

    end_compute_i = total_tokens
    active_indices, extra_page, swc, lpl = _compute_window_local_view(
        all_indices,
        front_removed=front,
        end_compute_i=end_compute_i,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # Window-local cache length matches the C++ accounting.
    expected_swc = end_compute_i - front_removed * TOKENS_PER_BLOCK
    assert swc == expected_swc, (
        f"swc={swc} != expected {expected_swc} "
        f"(end_compute_i={end_compute_i}, front_removed={front_removed})"
    )
    # C++ eviction invariant: live cache ≤ window + one spillover page.
    assert swc <= SWA_WINDOW + TOKENS_PER_BLOCK

    # Live slice starts at front_removed and covers every page needed to
    # address swc tokens — including the spillover page when present.
    expected_num_active = (expected_swc + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    expected_slice = list(all_indices[front_removed : front_removed + expected_num_active])
    assert active_indices == expected_slice, (
        f"slice mismatch: front_removed={front_removed}, "
        f"all_indices={list(all_indices)}, active={active_indices}"
    )

    # last_page_len matches the modular arithmetic on the local cache length.
    expected_lpl = (swc - 1) % TOKENS_PER_BLOCK + 1 if swc > 0 else 0
    assert lpl == expected_lpl

    # Page table capacity covers the live cache length.
    assert len(active_indices) * TOKENS_PER_BLOCK >= swc


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


def test_helper_pre_eviction_long_prefill_passes_full_pages(two_window_interface, monkeypatch):
    """Long prefill that has not yet been front-evicted — hand FULL live pages.

    Captures the MMLU-on-Gemma3n regime: addSequenceBatch admits a prefill
    longer than the SWA window and allocates blocks linearly for the full
    prompt (kvCacheManager.cpp::addSequenceBatch comment "For SWA, blocks
    are allocated linearly for the full prompt; out-of-window blocks are
    only detached during generation in adjustBlocksIfNeeded").  At the
    first forward pass ``front_removed == 0`` and the kernel needs every
    allocated page plus ``seq_len_with_cache == end_compute_i`` so the
    sliding-window mask is applied inside the kernel over the full prefill,
    matching the contract validated by
    ``test_long_prefill_sw_matches_sdpa``.

    Pre-fix the helper unconditionally clamped ``active_token_count`` to
    ``group_window`` even when no eviction had fired, which truncated the
    page list to the first ``window/page_size`` pages and silently
    corrupted long prefills.
    """
    manager = two_window_interface.kv_cache_manager
    # End at 3x the window so the prefill clearly exceeds W on the SWA pool
    # but front-eviction has not run yet.
    prefill_len = SWA_WINDOW * 3  # 192 tokens
    req = _add_request(manager, request_id=46, token_num=prefill_len)
    monkeypatch.setattr(manager, "get_num_front_blocks_removed", lambda req_id, window_size=None: 0)

    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)
    expected_pages = (prefill_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    assert len(all_indices) >= expected_pages, (
        f"manager must over-allocate pages for SWA prefill: "
        f"got {len(all_indices)}, expected >= {expected_pages}"
    )

    active_indices, extra_page, swc, lpl = _compute_window_local_view(
        all_indices,
        front_removed=0,
        end_compute_i=prefill_len,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # Hand the kernel every live page (not just the window's worth).
    assert active_indices == list(all_indices[:expected_pages]), (
        f"expected full prefill page list, got {active_indices} vs all_indices={list(all_indices)}"
    )
    # Window-local cache length equals the unclamped prefill length — the
    # kernel will mask down to the window itself.
    assert swc == prefill_len, f"swc={swc} should equal prefill_len={prefill_len}"
    assert lpl == (prefill_len - 1) % TOKENS_PER_BLOCK + 1
    # Page table capacity covers the full prefill (the kernel writes 1
    # K/V token per query token).
    assert len(active_indices) * TOKENS_PER_BLOCK >= swc
    # extra_page only populated if the manager pre-allocated past the prefill.
    if len(all_indices) > expected_pages:
        assert extra_page == all_indices[expected_pages]
    else:
        assert extra_page == -1


def test_helper_does_not_consume_evicted_extra_slot(two_window_interface, monkeypatch):
    """extra_page must live AFTER the live region, never inside the evicted range.

    Pre-Phase-2 code that referenced ``all_indices[num_active]`` would land
    inside the evicted region whenever front_removed > 0 — this test guards
    against that.  Under the live-page-derived view the live region already
    covers every allocated page up to and including the spillover, so
    ``extra_page == -1`` in the typical post-eviction case; we additionally
    check that whenever extra_page IS populated, the index sits past the
    evicted prefix.
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
    active_indices, extra_page, swc, _ = _compute_window_local_view(
        all_indices,
        front_removed=front_removed,
        end_compute_i=total_tokens,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # Live region exhausts every page the C++ side handed us, so the
    # spillover slot beyond it is -1 (no allocated next page).
    num_active = len(active_indices)
    expected_extra_idx = front_removed + num_active
    if len(all_indices) > expected_extra_idx:
        assert extra_page == all_indices[expected_extra_idx]
        # Crucially, the extra slot is NOT in the front-evicted range.
        assert expected_extra_idx >= front_removed
    else:
        assert extra_page == -1


# ---------------------------------------------------------------------------
# Multi-pool gate: trtllm (and any backend) may host >1 KV pool
# ---------------------------------------------------------------------------


def _build_two_pool_interface(requires_uniform_kv_caches: bool):
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
        requires_uniform_kv_caches=requires_uniform_kv_caches,
    )
    interface.add_resource(
        "kv_swa",
        KVPagedResourceHandler(
            4, 32, dtype=torch.float16, attention_type=AttentionType.mha, sliding_window=SWA_WINDOW
        ),
    )
    interface.add_resource(
        "kv_full",
        KVPagedResourceHandler(4, 32, dtype=torch.float16, attention_type=AttentionType.mha),
    )
    return interface


def test_two_distinct_windows_allowed_by_default():
    """Default (requires_uniform_kv_caches=False, the trtllm setting) hosts two pools."""
    interface = _build_two_pool_interface(requires_uniform_kv_caches=False)
    interface.initialize_resources()  # must not raise
    # SWA pool + full-attention pool == two distinct windows.
    windows = sorted(
        {SWA_WINDOW, FULL_WINDOW}
        & {pc.window_size for pc in interface._identify_managed_kv_resources()[1]}
    )
    assert windows == [SWA_WINDOW, FULL_WINDOW]


def test_uniform_kv_caches_still_enforced_when_requested():
    """The uniformity mechanism is intact: opting in still rejects >1 pool.

    (No backend opts in today; trtllm now defaults to False -- this guards the
    mechanism so a future single-pool backend can still rely on it.)
    """
    interface = _build_two_pool_interface(requires_uniform_kv_caches=True)
    with pytest.raises(RuntimeError, match="not uniform"):
        interface.initialize_resources()


# ---------------------------------------------------------------------------
# Cyclic-SWA view (trtllm): full block table + global KV length, no slicing
# ---------------------------------------------------------------------------


def test_cyclic_view_passes_full_table_and_global_length(two_window_interface):
    """Trtllm path: hand the kernel the FULL block table and the GLOBAL length.

    The trtllm kernel masks the sliding window internally via cyclic indexing,
    so -- unlike the host-sliced triton/flashinfer path -- the executor must NOT
    front-slice and must report the un-window-capped KV length.
    """
    manager = two_window_interface.kv_cache_manager
    # A prefill that exceeds the SWA window so window-local slicing WOULD differ.
    prefill_len = SWA_WINDOW * 3  # 192 tokens
    req = _add_request(manager, request_id=50, token_num=prefill_len)
    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)

    active_indices, extra_page, swc, lpl = _compute_cyclic_full_view(
        all_indices,
        end_compute_i=prefill_len,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # Full table verbatim (no front-slice, no window cap).
    assert active_indices == list(all_indices)
    # Global (un-capped) KV length -- matches host_past_key_value_lengths.
    assert swc == prefill_len
    assert lpl == (prefill_len - 1) % TOKENS_PER_BLOCK + 1
    # No deferred-page insertion in cyclic mode.
    assert extra_page == -1


def test_cyclic_view_differs_from_window_local_when_evicted(two_window_interface, monkeypatch):
    """Cyclic view ignores front-eviction; window-local view slices it off.

    Guards that the two staging paths genuinely diverge once the window has
    been exceeded (so a backend mix-up would be caught).
    """
    manager = two_window_interface.kv_cache_manager
    front_removed = 2
    total_tokens = front_removed * TOKENS_PER_BLOCK + SWA_WINDOW + 1
    req = _add_request(manager, request_id=51, token_num=total_tokens)
    monkeypatch.setattr(
        manager, "get_num_front_blocks_removed", lambda req_id, window_size=None: front_removed
    )
    all_indices = manager.get_cache_indices(req, window_size=SWA_WINDOW)

    cyc_indices, _, cyc_swc, _ = _compute_cyclic_full_view(
        all_indices, end_compute_i=total_tokens, tokens_per_block=TOKENS_PER_BLOCK
    )
    win_indices, _, win_swc, _ = _compute_window_local_view(
        all_indices,
        front_removed=front_removed,
        end_compute_i=total_tokens,
        group_window=SWA_WINDOW,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # Cyclic keeps the full list + global length; window-local slices + caps.
    assert cyc_indices == list(all_indices)
    assert cyc_swc == total_tokens
    # Window-local view drops the stale front pages and starts at front_removed.
    assert win_indices == list(all_indices[front_removed : front_removed + len(win_indices)])
    assert len(win_indices) < len(cyc_indices)
    assert win_swc == total_tokens - front_removed * TOKENS_PER_BLOCK
    assert cyc_swc != win_swc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
