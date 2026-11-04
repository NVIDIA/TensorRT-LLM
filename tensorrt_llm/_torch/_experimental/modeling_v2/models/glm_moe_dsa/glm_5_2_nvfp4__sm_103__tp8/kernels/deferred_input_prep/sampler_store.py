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
"""One-launch store update and host copy for the one-model speculative sampler.

Native ``SpecSampler.sample_async`` issues, after each decode-graph replay, an
H2D copy of the slot list, four ``index_copy_`` kernels that scatter the
worker outputs into the slot-indexed store, and three D2H copies of the store
into fresh pinned host tensors. ``store_and_copy_out`` does the same in one
single-CTA launch: it reads the slot list from pinned memory, scatters the
rows, and then writes the three complete store tensors straight into the
pinned host tensors (zero-copy), so the host tensors hold exactly what the
native D2H copies would have produced.

The kernel reads the pinned slot list and writes the pinned host tensors at
execution time, possibly long after the call returned. Both are therefore
owned by a ring entry until the completion event recorded behind the launch
has completed: the slot list is reused and the host tensors are released only
then (a native D2H copy gets the same protection from the caching host
allocator's stream events).

The writer is used only in the executor-loop domain deferred_input_prep is validated in
(``compat.loop_domain``: the overlap loop of an executor without attention
DP); ``supported`` checks it before every launch, so any other loop keeps the
native path.
"""

from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

from .compat import loop_domain
from .packet import _NEVER_DONE


@triton.jit
def _sampler_store_kernel(
    slots_addr,
    n_req,
    o_new_ptr,
    o_new_s0,
    o_new_s1,
    w_new,
    o_next_new_ptr,
    o_next_new_s0,
    o_next_new_s1,
    w_next_new,
    o_lens_ptr,
    o_lens_s0,
    o_draft_ptr,
    o_draft_s0,
    o_draft_s1,
    w_draft,
    st_new_ptr,
    st_new_s0,
    st_new_s1,
    st_next_new_ptr,
    st_next_new_s0,
    st_next_new_s1,
    st_lens_ptr,
    st_draft_ptr,
    st_draft_s0,
    n_st_new,
    n_st_lens,
    n_st_draft,
    h_new_addr,
    h_lens_addr,
    h_draft_addr,
    BLOCK: tl.constexpr,
):
    slots = slots_addr.to(tl.pointer_type(tl.int64))
    offs = tl.arange(0, BLOCK)
    for j in range(n_req):
        slot = tl.load(slots + j)
        m = offs < w_new
        v = tl.load(o_new_ptr + j * o_new_s0 + offs * o_new_s1, mask=m)
        tl.store(st_new_ptr + offs * st_new_s0 + slot * st_new_s1, v, mask=m)
        m = offs < w_next_new
        v = tl.load(o_next_new_ptr + j * o_next_new_s0 + offs * o_next_new_s1, mask=m)
        tl.store(st_next_new_ptr + offs * st_next_new_s0 + slot * st_next_new_s1, v, mask=m)
        v = tl.load(o_lens_ptr + j * o_lens_s0)
        tl.store(st_lens_ptr + slot, v)
        m = offs < w_draft
        v = tl.load(o_draft_ptr + j * o_draft_s0 + offs * o_draft_s1, mask=m)
        tl.store(st_draft_ptr + slot * st_draft_s0 + offs, v, mask=m)
    tl.debug_barrier()
    h_new = h_new_addr.to(tl.pointer_type(tl.int32))
    h_lens = h_lens_addr.to(tl.pointer_type(tl.int32))
    h_draft = h_draft_addr.to(tl.pointer_type(tl.int32))
    for start in range(0, n_st_new, BLOCK):
        m = start + offs < n_st_new
        tl.store(h_new + start + offs, tl.load(st_new_ptr + start + offs, mask=m), mask=m)
    for start in range(0, n_st_lens, BLOCK):
        m = start + offs < n_st_lens
        tl.store(h_lens + start + offs, tl.load(st_lens_ptr + start + offs, mask=m), mask=m)
    for start in range(0, n_st_draft, BLOCK):
        m = start + offs < n_st_draft
        tl.store(h_draft + start + offs, tl.load(st_draft_ptr + start + offs, mask=m), mask=m)


class SamplerStoreWriter:
    """Holds the pinned slot-list ring for ``store_and_copy_out``."""

    def __init__(self, ring: int = 4, max_requests: int = 1024):
        # Pageable when prefer_pinned() is False; the sampler then never engages the writer.
        self._slots = [
            torch.zeros(max_requests, dtype=torch.int64, pin_memory=prefer_pinned())
            for _ in range(ring)
        ]
        self._events = [None] * ring
        # Zero-copy host targets of each entry's launch, held until its event completes.
        self._keep = [()] * ring
        self._next = 0
        self.num_calls = 0

    def _slot_buffer(self) -> torch.Tensor:
        for _ in range(len(self._slots)):
            i = self._next
            self._next = (self._next + 1) % len(self._slots)
            ev = self._events[i]
            if ev is None or ev.query():
                self._keep[i] = ()
                return i
        self._slots.append(torch.zeros_like(self._slots[0], pin_memory=prefer_pinned()))
        self._events.append(None)
        self._keep.append(())
        return len(self._slots) - 1

    def supported(self, store, o_new, o_next_new, o_lens, o_draft) -> bool:
        """Whether this sampling step may use the writer (checked before any side effect)."""
        return loop_domain() and self.tensors_supported(store, o_new, o_next_new, o_lens, o_draft)

    @staticmethod
    def tensors_supported(store, o_new, o_next_new, o_lens, o_draft) -> bool:
        """Whether the store and output tensors have the layout the kernel handles."""
        tensors = (
            store.new_tokens,
            store.next_new_tokens,
            store.new_tokens_lens,
            store.next_draft_tokens,
            o_new,
            o_next_new,
            o_lens,
            o_draft,
        )
        if any(t.dtype != torch.int32 or not t.is_cuda for t in tensors):
            return False
        if not all(t.is_contiguous() for t in tensors[:4]):
            return False
        if store.new_tokens.shape[-1] != 1 or store.next_new_tokens.shape[-1] != 1:
            return False
        return (
            o_new.shape[1] == store.new_tokens.shape[0]
            and o_next_new.shape[1] == store.next_new_tokens.shape[0]
            and o_draft.shape[1] == store.next_draft_tokens.shape[1]
            and max(o_new.shape[1], o_next_new.shape[1], o_draft.shape[1]) <= 64
        )

    def store_and_copy_out(
        self,
        store,
        slots: Sequence[int],
        o_new,
        o_next_new,
        o_lens,
        o_draft,
        h_new,
        h_lens,
        h_draft,
        record_event: bool = True,
    ) -> None:
        """Scatter outputs into ``store`` rows ``slots`` and copy the store to host.

        ``record_event=False``: the caller records an event right after this
        call (the sampler event) and passes it to ``set_completion_event``.
        """
        i = self._slot_buffer()
        buf = self._slots[i]
        n = len(slots)
        if n > buf.numel():
            raise ValueError("too many sampling requests for the slot buffer")
        buf.numpy()[:n] = slots
        self._keep[i] = (h_new, h_lens, h_draft)
        _sampler_store_kernel[(1,)](
            buf.data_ptr(),
            n,
            o_new,
            o_new.stride(0),
            o_new.stride(1),
            o_new.shape[1],
            o_next_new,
            o_next_new.stride(0),
            o_next_new.stride(1),
            o_next_new.shape[1],
            o_lens,
            o_lens.stride(0),
            o_draft,
            o_draft.stride(0),
            o_draft.stride(1),
            o_draft.shape[1],
            store.new_tokens,
            store.new_tokens.stride(0),
            store.new_tokens.stride(1),
            store.next_new_tokens,
            store.next_new_tokens.stride(0),
            store.next_new_tokens.stride(1),
            store.new_tokens_lens,
            store.next_draft_tokens,
            store.next_draft_tokens.stride(0),
            store.new_tokens.numel(),
            store.new_tokens_lens.numel(),
            store.next_draft_tokens.numel(),
            h_new.data_ptr(),
            h_lens.data_ptr(),
            h_draft.data_ptr(),
            BLOCK=64,
            num_warps=1,
        )
        if record_event:
            ev = torch.cuda.Event()
            ev.record()
            self._events[i] = ev
        else:
            self._events[i] = _NEVER_DONE
            self._last = i
        self.num_calls += 1

    def set_completion_event(self, event: torch.cuda.Event) -> None:
        """Mark the last launch's slot list reusable once ``event`` completes."""
        i = getattr(self, "_last", None)
        if i is not None:
            self._events[i] = event
            self._last = None
