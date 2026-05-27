# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``bypass_processor_output_validation``.

The context manager mutates module-level attributes on transformers modules,
so concurrent entries from worker threads (preprocessing is dispatched via
``asyncio.to_thread``) used to stack wrappers on top of each other. Under
high video concurrency that produced hundreds of nested ``_filtered_validate``
frames and ultimately a ``RecursionError`` (HTTP 400 to the client). It also
corrupted the restore: if Thread A exited first it would restore the *real*
function, then Thread B's ``finally`` would re-patch with a stale wrapper.

These tests pin down both behaviors.
"""

import threading
import time

import pytest

# Skip the whole module gracefully if transformers isn't importable in the
# test environment — the bypass logic only matters when the binders exist.
transformers = pytest.importorskip("transformers")

from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
    _PROCESSOR_OUTPUT_KEYS,
    bypass_processor_output_validation,
)


def _binders():
    import transformers.processing_utils as _pu
    import transformers.video_processing_utils as _vpu

    candidates = [_pu, _vpu]
    for name in ("transformers.image_processing_utils", "transformers.image_processing_utils_fast"):
        try:
            candidates.append(__import__(name, fromlist=[""]))
        except ImportError:
            pass
    return [b for b in candidates if hasattr(b, "validate_typed_dict")]


def test_originals_restored_after_exit():
    binders = _binders()
    assert binders, "expected at least one transformers binder"
    originals = {b: b.validate_typed_dict for b in binders}

    with bypass_processor_output_validation():
        for b in binders:
            assert b.validate_typed_dict is not originals[b]

    for b in binders:
        assert b.validate_typed_dict is originals[b]


def test_filter_strips_output_keys():
    binders = _binders()
    seen = {}

    real = binders[0].validate_typed_dict

    # Wrap the real validator just to capture what it actually receives.
    def spy(schema, data):
        seen["data"] = data
        # Don't actually validate — we only care that the filter ran.
        return None

    for b in binders:
        b.validate_typed_dict = spy
    try:
        with bypass_processor_output_validation():
            wrapped = binders[0].validate_typed_dict
            payload = {
                "image_grid_thw": "leaked",
                "video_grid_thw": "leaked",
                "pixel_values": "leaked",
                "legit_key": "keep",
            }
            wrapped(object(), payload)
    finally:
        for b in binders:
            b.validate_typed_dict = real

    assert "data" in seen
    received = seen["data"]
    for k in _PROCESSOR_OUTPUT_KEYS:
        assert k not in received, f"{k} should have been filtered"
    assert received.get("legit_key") == "keep"


def test_concurrent_entry_does_not_stack_wrappers():
    """The core regression: many threads inside the CM at once must not
    stack wrappers (which previously produced unbounded recursion)."""
    binders = _binders()
    real_originals = {b: b.validate_typed_dict for b in binders}

    num_threads = 64
    barrier = threading.Barrier(num_threads)
    observed_wrappers = [None] * num_threads
    errors = []

    def worker(idx):
        try:
            barrier.wait(timeout=10)
            with bypass_processor_output_validation():
                # All threads observe the *same* wrapper object.
                observed_wrappers[idx] = binders[0].validate_typed_dict
                # Keep everyone inside the CM simultaneously.
                time.sleep(0.05)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors}"
    # The crucial assertion: a single shared wrapper, not a stack of them.
    assert len(set(map(id, observed_wrappers))) == 1
    # And it must not be the real function (i.e. the patch did apply).
    assert observed_wrappers[0] is not real_originals[binders[0]]
    # After everyone exits, originals are restored exactly.
    for b in binders:
        assert b.validate_typed_dict is real_originals[b]


def test_concurrent_entry_does_not_recurse():
    """If wrappers were stacked, calling the patched function would recurse
    through N layers. Assert that the call stack stays shallow regardless of
    how many threads are concurrently inside the CM."""
    binders = _binders()
    real = binders[0].validate_typed_dict

    depths = []
    depth_lock = threading.Lock()

    def measuring_real(schema, data):
        import sys

        # Frame depth from this call upward, only counting frames that came
        # from the bypass wrapper. We just record total depth and check it
        # stays bounded.
        frame = sys._getframe()
        n = 0
        while frame is not None:
            if frame.f_code.co_name == "_filtered_validate":
                n += 1
            frame = frame.f_back
        with depth_lock:
            depths.append(n)
        return None

    for b in binders:
        b.validate_typed_dict = measuring_real
    try:
        num_threads = 32
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait(timeout=10)
            with bypass_processor_output_validation():
                binders[0].validate_typed_dict(object(), {"pixel_values": 1})

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for b in binders:
            b.validate_typed_dict = real

    assert depths, "expected at least one validator call"
    # With the fix, exactly one wrapper frame regardless of concurrency.
    # Without the fix, this would be up to num_threads.
    assert max(depths) == 1, f"wrapper stacking detected: depths={depths}"


def test_sequential_reentry_restores_cleanly():
    """Repeated enter/exit on a single thread must always restore the
    original. (Guards against the counter / saved-originals state going
    out of sync.)"""
    binders = _binders()
    originals = {b: b.validate_typed_dict for b in binders}
    for _ in range(5):
        with bypass_processor_output_validation():
            for b in binders:
                assert b.validate_typed_dict is not originals[b]
        for b in binders:
            assert b.validate_typed_dict is originals[b]
