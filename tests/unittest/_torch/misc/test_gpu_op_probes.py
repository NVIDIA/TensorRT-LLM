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
"""Self-tests for the generic custom-op soundness probes in
``_torch.gpu_op_probes``, exercised against synthetic examples so no model,
kernel, or (for the cache probes) even a GPU is required:

* a toy id-keyed cache, in a deliberately unsound and a sound (weakref-
  pruned) variant, to show the recycled-id and same-object-reuse probes
  detect stale-entry bugs and pass sound implementations;
* a deliberately stream-unsafe toy op that launches its math on the raw
  default stream, to show the delayed-stream parity harness detects missing
  current-stream plumbing while passing a stream-correct op.
"""

import gc
import weakref

import pytest
import torch

from _torch.gpu_op_probes import (alloc_with_recycled_id,
                                  alloc_with_recycled_id_or_skip,
                                  assert_delayed_stream_parity,
                                  run_on_delayed_stream,
                                  same_object_reuse_probe)

# ---------------------------------------------------------------------------
# Toy id-keyed cache. The "op" doubles even tensors and negates odd ones,
# but memoizes the even/odd verdict — a contents-derived property — under
# id(tensor), mimicking drivers that cache per-tensor layout/variant
# decisions keyed by object identity.
# ---------------------------------------------------------------------------


class ToyIdKeyedCacheOp:

    def __init__(self, prune_on_gc: bool):
        self._verdict_cache = {}
        self._prune_on_gc = prune_on_gc

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        key = id(t)
        if key not in self._verdict_cache:
            self._verdict_cache[key] = bool((t.to(torch.int64) % 2 == 0).all())
            if self._prune_on_gc:
                # Sound variant: drop the entry when the keyed tensor is
                # deallocated, before its id can be recycled.
                weakref.finalize(t, self._verdict_cache.pop, key, None)
        return t * 2 if self._verdict_cache[key] else -t


def toy_reference(t: torch.Tensor) -> torch.Tensor:
    return t * 2 if bool((t.to(torch.int64) % 2 == 0).all()) else -t


def _make_even():
    return torch.arange(64, dtype=torch.int32) * 2


def _make_odd():
    return torch.arange(64, dtype=torch.int32) * 2 + 1


def _land_replacement_on_recycled_id(op, or_skip: bool):
    """Cache an 'even' verdict under id(t1), free t1, and return a new odd
    tensor aliasing the old id (or None / skip)."""
    t1 = _make_even()
    assert torch.equal(op(t1), toy_reference(t1))
    old_id = id(t1)
    del t1
    gc.collect()
    if or_skip:
        return alloc_with_recycled_id_or_skip(old_id, _make_odd)
    return alloc_with_recycled_id(old_id, _make_odd)


def test_recycled_id_probe_detects_stale_cache_entry():
    """An id-keyed cache with no dealloc pruning inherits the freed tensor's
    verdict: the recycled-id replacement (odd contents) takes the 'even'
    fast path and produces a wrong result the probe catches."""
    op = ToyIdKeyedCacheOp(prune_on_gc=False)
    t2 = _land_replacement_on_recycled_id(op, or_skip=False)
    if t2 is None:
        pytest.skip("could not obtain a recycled id")
    assert id(t2) in op._verdict_cache  # the stale entry is still there
    assert not torch.equal(op(t2), toy_reference(t2)), (
        "expected the deliberately unsound toy cache to be caught by the "
        "recycled-id probe")


def test_recycled_id_probe_passes_sound_cache():
    """The sound variant prunes at dealloc, so the recycled-id replacement
    gets a fresh verdict and full parity."""
    op = ToyIdKeyedCacheOp(prune_on_gc=True)
    t2 = _land_replacement_on_recycled_id(op, or_skip=True)
    assert id(t2) not in op._verdict_cache  # pruned before the recycle
    assert torch.equal(op(t2), toy_reference(t2))


def test_alloc_with_recycled_id_returns_none_when_unattainable():
    """A live tensor pins its id, so the bounded retry loop must give up."""
    anchor = _make_even()
    got = alloc_with_recycled_id(id(anchor), _make_odd, attempts=8)
    assert got is None
    del anchor


def _toy_assert_equal(actual, expected, label):
    assert torch.equal(actual, expected), label


def test_same_object_reuse_probe_detects_stale_cache_entry():
    """Same object, mutated contents: the unsound cache reuses the call-1
    'even' verdict for the call-2 odd payload."""
    op = ToyIdKeyedCacheOp(prune_on_gc=False)
    key_tensor = _make_even()
    with pytest.raises(AssertionError, match="call 1"):
        same_object_reuse_probe(op, toy_reference, key_tensor,
                                [_make_even(), _make_odd()],
                                assert_close=_toy_assert_equal)


def test_same_object_reuse_probe_passes_contents_keyed_op():
    """An op that derives its verdict from contents on every call (here: the
    reference itself) passes the same-object-reuse probe."""
    same_object_reuse_probe(toy_reference, toy_reference, _make_even(),
                            [_make_even(), _make_odd()],
                            assert_close=_toy_assert_equal)


# ---------------------------------------------------------------------------
# Delayed-stream parity harness, against a toy op whose launcher ignores the
# current stream (the failure class: custom-op launchers hardcoding the
# default stream instead of torch.cuda.current_stream()).
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="requires CUDA")


def stream_correct_op(a, b):
    return a @ b + 1


def stream_unsafe_op(a, b):
    # Deliberately launch on the raw default stream regardless of the
    # caller's current stream — the toy stand-in for a launcher missing its
    # stream argument.
    with torch.cuda.stream(torch.cuda.default_stream()):
        return a @ b + 1


def _matmul_inputs():
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(512, 512, device="cuda", generator=gen)
    b = torch.randn(512, 512, device="cuda", generator=gen)
    return a, b


@cuda_only
def test_delayed_stream_parity_passes_stream_correct_op():
    assert_delayed_stream_parity(stream_correct_op, _matmul_inputs())


@cuda_only
def test_delayed_stream_parity_detects_default_stream_launch():
    """The unsafe op's default-stream matmul runs ahead of the delayed
    stream's pending input writes, reading buffers before they exist —
    the parity check must fail."""
    with pytest.raises(AssertionError):
        assert_delayed_stream_parity(stream_unsafe_op, _matmul_inputs())


@cuda_only
def test_run_on_delayed_stream_preserves_correct_results():
    """The lower-level harness on its own: outputs of a stream-correct op
    match an eager default-stream run."""
    a, b = _matmul_inputs()
    expected = stream_correct_op(a.clone(), b.clone())
    torch.cuda.synchronize()
    actual = run_on_delayed_stream(stream_correct_op, (a, b))
    torch.testing.assert_close(actual, expected)
