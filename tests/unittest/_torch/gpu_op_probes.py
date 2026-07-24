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
"""Reusable probes for two classes of custom-op integration bugs that pass
single-stream, tensor-reusing unit tests:

1. **id-keyed wrapper caches** (``recycle`` helpers). Custom-op drivers often
   memoize per-tensor artifacts — compiled-variant selections, foreign-ABI
   tensor wrappers, layout verdicts — in a dict keyed by ``id(tensor)``.
   CPython recycles an object's address (and therefore its ``id``) as soon as
   the object is freed, so a cache entry that outlives the tensor it was keyed
   by can be silently inherited by an unrelated new tensor that happens to
   land on the recycled address. Unit tests that keep every input alive for
   the duration of the test can never observe this; production runtimes that
   allocate fresh metadata tensors every step hit it routinely. The helpers
   below deterministically engineer the aliasing: cache something under
   ``id(t)``, free ``t``, re-allocate until a new object lands on the same id,
   and check the op against a reference with contents for which the stale
   cache entry would be wrong.

   A closely related failure mode needs no id recycling at all: the *same*
   tensor object is passed twice with mutated contents, and a per-call
   override poisons the id-keyed cache so the second call takes the wrong
   path. ``same_object_reuse_probe`` drives that scenario.

2. **custom-op launchers on non-default streams** (``run_on_delayed_stream``
   / ``assert_delayed_stream_parity``). Executors commonly run the model on a
   dedicated non-blocking ``torch.cuda.Stream``. A custom-op launcher that
   enqueues (some of) its kernels on the *default* stream instead of
   ``torch.cuda.current_stream()`` races with the producers of its inputs and
   the consumers of its outputs — silent, intermittent corruption end-to-end
   while every single-stream unit test passes. The harness makes the race
   deterministic: it holds a fresh non-blocking stream back with a long GPU
   sleep, materializes the inputs *behind* the sleep, invokes the op on that
   stream, and consumes the outputs on that stream immediately. Any kernel
   the op launches on the default stream then reads inputs before they exist
   and/or has its output read before it is written, so a simple parity check
   against a single-stream reference run fails loudly.
"""

import gc
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import pytest
import torch

__all__ = [
    "alloc_with_recycled_id",
    "alloc_with_recycled_id_or_skip",
    "same_object_reuse_probe",
    "run_on_delayed_stream",
    "assert_delayed_stream_parity",
]

# ---------------------------------------------------------------------------
# 1. id-keyed cache soundness probes.
# ---------------------------------------------------------------------------


def alloc_with_recycled_id(target_id: int,
                           make: Callable[[], Any],
                           attempts: int = 512) -> Optional[Any]:
    """Allocate via ``make()`` until an object lands on address ``target_id``.

    Intended usage, given some tensor ``t1`` whose ``id`` was used as a cache
    key::

        old_id = id(t1)
        del t1          # drop the *only* reference
        gc.collect()
        t2 = alloc_with_recycled_id(old_id, lambda: make_tensor(...))

    ``t2`` (if not ``None``) is a brand-new object that aliases the old cache
    key, which is exactly the situation an id-keyed cache must survive.

    Wrong-address candidates are kept alive until the function returns so the
    allocator cannot hand the same wrong slot back on the next attempt; with
    CPython's small-object allocator the freed address is normally returned
    on the first attempt. Returns ``None`` if no candidate lands on
    ``target_id`` within ``attempts`` allocations (e.g. another thread took
    the slot) — callers in tests should treat that as a skip, not a failure
    (see :func:`alloc_with_recycled_id_or_skip`).
    """
    hold = []
    for _ in range(attempts):
        cand = make()
        if id(cand) == target_id:
            return cand
        hold.append(cand)
    return None


def alloc_with_recycled_id_or_skip(target_id: int,
                                   make: Callable[[], Any],
                                   attempts: int = 512) -> Any:
    """Like :func:`alloc_with_recycled_id`, but ``pytest.skip`` when the id
    cannot be recycled instead of returning ``None``.

    Landing on a recycled id is an allocator behavior, not a guarantee; a
    test built on it must skip rather than fail when the setup is
    unattainable."""
    gc.collect()  # make sure the target slot is actually free
    obj = alloc_with_recycled_id(target_id, make, attempts=attempts)
    if obj is None:
        pytest.skip(f"could not land a new object on recycled id "
                    f"{target_id:#x} within {attempts} attempts")
    return obj


def same_object_reuse_probe(
        op: Callable[[torch.Tensor], Any],
        reference: Callable[[torch.Tensor], Any],
        key_tensor: torch.Tensor,
        payloads: Iterable[torch.Tensor],
        assert_close: Callable[[Any, Any, str], None],
) -> None:
    """Drive an op with the *same* tensor object carrying *different*
    contents on successive calls, checking each call against a reference.

    This targets id-keyed caches that memoize a verdict derived from the
    tensor's **contents** (not just its identity/layout): the first call
    populates the cache under ``id(key_tensor)``, the payload refill keeps
    the id unchanged, and a stale hit on the second call takes the wrong
    path. The reference is invoked with a fresh clone each time so it can
    never benefit from (or be poisoned by) object identity.

    Args:
        op: callable under test; receives ``key_tensor`` itself.
        reference: trusted implementation; receives a fresh clone.
        key_tensor: the reused object. Refilled in place from each payload.
        payloads: tensors copyable into ``key_tensor`` (same shape/dtype).
        assert_close: ``assert_close(actual, expected, label)`` — raises on
            mismatch. Kept caller-supplied because tolerance is op-specific.
    """
    for i, payload in enumerate(payloads):
        key_tensor.copy_(payload)
        actual = op(key_tensor)
        expected = reference(key_tensor.clone())
        assert_close(actual, expected, f"same-object reuse, call {i}")


# ---------------------------------------------------------------------------
# 2. Delayed-stream parity harness.
# ---------------------------------------------------------------------------


def _map_tensors(fn, obj):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_map_tensors(fn, x) for x in obj)
    return obj


def run_on_delayed_stream(op: Callable[..., Any],
                          inputs: Sequence[torch.Tensor],
                          delay_cycles: int = 1 << 27) -> Any:
    """Run ``op(*inputs)`` on a fresh non-blocking CUDA stream whose queue is
    held back by a GPU sleep, with the input buffers written *behind* the
    sleep and the outputs consumed on that stream immediately after the call.

    A launcher inside ``op`` that enqueues a kernel on the *default* stream
    instead of the current stream will read inputs before they are written
    and/or have its outputs read before they are written, corrupting the
    result deterministically. A launcher that correctly uses
    ``torch.cuda.current_stream()`` is unaffected — stream order serializes
    everything behind the sleep.

    Note: any lazy compilation inside ``op`` must be warmed up *before*
    calling this (a long host-side JIT inside the delayed region lets the
    sleep expire and hides the race). :func:`assert_delayed_stream_parity`
    handles the warmup for you.

    Args:
        op: callable taking the (re-materialized) inputs positionally.
        inputs: input tensors, already valid on the default stream.
        delay_cycles: argument to ``torch.cuda._sleep``; ``1 << 27`` is tens
            of milliseconds on current hardware — enough to cover the launch
            burst without slowing the suite noticeably.

    Returns:
        The outputs of ``op`` (tensor / nested tuples-lists of tensors),
        after a full device synchronize.
    """
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()  # non-blocking w.r.t. the default stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(delay_cycles)
        # Re-materialize the inputs on the delayed stream: these copies are
        # pending behind the sleep, so a default-stream launch inside `op`
        # reads buffers that have not been written yet.
        staged = [x * 1 if isinstance(x, torch.Tensor) else x for x in inputs]
        out = op(*staged)
        # Consume the outputs on this stream immediately, like a downstream
        # layer would: a stray default-stream kernel writing `out` races
        # with this read.
        out = _map_tensors(lambda t: t * 1, out)
    torch.cuda.synchronize()
    return out


def assert_delayed_stream_parity(
        op: Callable[..., Any],
        inputs: Sequence[torch.Tensor],
        reference: Optional[Callable[..., Any]] = None,
        delay_cycles: int = 1 << 27,
        assert_close: Optional[Callable[[Any, Any, str], None]] = None,
        warmup: bool = True) -> Tuple[Any, Any]:
    """Check that ``op`` produces the same result on a delayed non-default
    stream (see :func:`run_on_delayed_stream`) as ``reference`` does on the
    default stream. This is a deterministic detector for custom-op launchers
    that drop or ignore the current-stream argument.

    Args:
        op: op under test.
        inputs: input tensors (valid on the default stream). ``op`` and
            ``reference`` each receive fresh copies; ``op`` may cache by
            input identity without affecting the comparison.
        reference: trusted implementation run single-stream; defaults to
            ``op`` itself, which is sufficient — the single-stream run of a
            stream-buggy op is still numerically correct.
        delay_cycles: see :func:`run_on_delayed_stream`.
        assert_close: ``assert_close(actual, expected, label)``; defaults to
            ``torch.testing.assert_close`` over the (nested) outputs.
        warmup: run ``op`` once on the default stream first so lazy
            compilation cannot eat the delay window.

    Returns:
        ``(streamed_outputs, reference_outputs)``.
    """
    if reference is None:
        reference = op
    if assert_close is None:

        def assert_close(actual, expected, label):
            torch.testing.assert_close(actual, expected, msg=lambda m:
                                       f"[{label}] {m}")

    ref_inputs = [
        x.clone() if isinstance(x, torch.Tensor) else x for x in inputs
    ]
    expected = reference(*ref_inputs)
    if warmup:
        op(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
    torch.cuda.synchronize()

    actual = run_on_delayed_stream(op, inputs, delay_cycles=delay_cycles)
    assert_close(actual, expected, "delayed-stream parity")
    return actual, expected
