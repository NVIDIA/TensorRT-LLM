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
"""deferred_input_prep staging of one decode input preparation into a single device launch.

Native input preparation between two decode-graph replays issues ~50 small
stream-ordered operations (H2D copies, index/elementwise kernels, a
page-table expansion kernel). Each costs launch/dependency latency on the
critical stream although it moves only bytes. While a ``PrepStager`` is
active, the instrumented call sites hand those writes to the stager instead;
``commit`` then applies all of them with one kernel that reads the host
values straight from pinned memory, and finally runs the few native compute
operations that must observe the staged values (``defer``).

Semantics preserved per staged write:

* H2D copies: the host bytes are read when ``commit`` runs, i.e. after the
  whole host preparation, which is also when the native asynchronous copy
  would have read them (the host runs ahead of the GPU and must not modify a
  pending copy's source, see ``KVCacheManager._stage_block_offsets_for_copy``).
* Device gathers from the previous iteration's sampler store read the same
  device values the native index kernels read: the store is written before
  the commit in stream order and the packet never writes it.
* Order: staged writes are added to the packet in staging (native) order.
  Descriptors of one launch run concurrently, so a write that overlaps an
  earlier one of the same packet, or that does not fit, first flushes the
  packet (one more launch); a write too large for an empty packet is issued
  natively at that point. The final bytes therefore always equal the native
  sequence of writes.
* Deferral: a staged write lands at the launch instead of at its call site.
  This is equivalent only if no other device operation in between touches
  the destination (the window contract below).

The target declares one ``PrepStager`` to the engine as
``RuntimeHooks.step_input_stager`` (``runtime.make_runtime_hooks``). It is
attached to the objects that own the instrumented call sites (their
``step_input_stager`` attribute) only for the duration of one preparation; the
native path runs whenever that attribute is ``None``.

Deferred-write window. ``DecoderRunner._forward_decoder`` opens it with
``begin_if_eligible`` right before ``DecoderInputPreparer.prepare_inputs`` and
closes it with ``end`` right before the decode-graph replay. In between, the
instrumented call sites (on the input preparer, the attention and spec
metadata and the KV cache managers, each marked "Staged write") only record
their writes, and the commit at ``end`` performs them. Contract for code that runs inside the
window:

* No other device operation may read or write a buffer the stager was
  handed. A read sees the bytes from before the staged write; a write is
  overwritten by the commit.
* Route new writes to such a buffer through the staging API (``copy``,
  ``values``, ``gather``, ``zero``, ``clamp_min``, ``block_offsets``), or call
  ``flush`` first, which issues every write staged so far in staging order.
* Exception: the input buffers' ``input_ids_cuda`` and ``position_ids_cuda``.
  ``overlap_inputs`` and ``position_ids`` also stage the decode graph's static
  copies of them (``static_inputs``), and the replay then skips its
  own copy from the input buffers, so a later staged write to either buffer
  does not reach the graph. To change them inside the window, call ``flush``
  and then write them natively; ``flush`` returns both to the replay's copy.
* Deferred native operations (``defer``) run after the commit at ``end``,
  also when ``flush`` was called. Their outputs (the indexer's scheduler
  metadata when ``scheduler_metadata`` declines) are not written before
  ``end``, and they read the values staged up to ``end``, including writes
  staged after the ``defer`` call.
* Nothing checks the contract in a production run. The debug guard
  (``PrepStager(debug_level=1)``, which
  ``TRTLLM_MODELING_V2_VALIDATE=1`` selects; ``guard``
  lists what it covers) raises on an operation that touches a staged buffer
  inside the window. ``TRTLLM_STAIRCASE_GLM_LL_DISABLE=deferred_input_prep``
  keeps every preparation native.
"""

import contextlib
import functools
from typing import Callable, List, Optional, Sequence

import torch

from ...dispatch import DISABLE_ENV
from .guard import DebugWindow
from .packet import PacketFull, PacketOverlap, PrepPacket

BAD_PAGE_INDEX = -1

_PREP_NOTICE = (
    "deferred-write window from DecoderRunner._forward_decoder's begin_if_eligible to "
    "stager.end(): the buffers written through the deferred_input_prep stager are written "
    "later by the commit; do not read or write them in between with other device ops; route "
    "writes through the stager API or call stager.flush() first; for input_ids_cuda and "
    "position_ids_cuda (also staged into the graph's static inputs) call stager.flush() "
    f"and then write natively; disable with {DISABLE_ENV}=deferred_input_prep"
)


def _host_memory_pinned() -> bool:
    """Whether TensorRT-LLM keeps its host staging buffers pinned."""
    from tensorrt_llm._utils import prefer_pinned

    return prefer_pinned()


def _log_prep_decision(reason: Optional[str]) -> None:
    """Log the first preparation decision of each kind (once per process and rank).

    ``deferred_input_prep/prep`` is the engaged decision, logged with the window contract;
    every decline reason gets its own key, so the log shows each kind of
    preparation deferred_input_prep left native.
    """
    from ...dispatch import log_capability

    if reason is None:
        log_capability("deferred_input_prep", "prep", True, _PREP_NOTICE)
    else:
        log_capability("deferred_input_prep", f"prep[{reason}]", False, reason)


def _stager_internal(method):
    """Run a staging method as the stager's own work (not checked by the debug guard)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._internal():
            return method(self, *args, **kwargs)

    return wrapper


class PrepStager:
    """Collects one preparation's device writes; see the module docstring.

    ``debug_level``: 0 (default) runs no checks; 1 turns on the debug guard; 2 the guard
    plus poisoning of every staged destination (``guard``).
    """

    def __init__(self, debug_level: int = 0):
        if debug_level not in (0, 1, 2):
            raise ValueError(f"debug_level must be 0, 1 or 2, got {debug_level!r}")
        self.packet = PrepPacket()
        self.active = False
        # Staged writes in native order: (kind, args).
        self._writes: List[tuple] = []
        self._deferred: List[Callable[[], None]] = []
        self.num_commits = 0
        self.last_num_descriptors = 0
        self.last_reason: Optional[str] = None
        self.decisions: dict = {}
        self.num_native_copies = 0
        self.num_flushes = 0
        self._attached: list = []
        # The replayed decode graph's static inputs while a window is open.
        self._static: Optional[dict] = None
        self._static_input_ids = 0
        self._static_positions = 0
        # Why every preparation of this stager stays native (None: may engage).
        self._decline_reason: Optional[str] = None
        if not _host_memory_pinned():
            # block_offsets stages copies from the pinned host page table; it
            # is pageable when prefer_pinned() is False (Confidential Compute).
            self._decline_reason = "host page tables are not pinned (prefer_pinned)"
        self.debug: Optional[DebugWindow] = DebugWindow(debug_level) if debug_level else None

    # ---- lifecycle ------------------------------------------------------

    def begin(self) -> None:
        if self.active:
            # A previous preparation raised before its launch; its writes are void.
            self.discard()
        self.packet.begin()
        self._writes = []
        self._deferred = []
        self._static_input_ids = 0
        self._static_positions = 0
        self.active = True
        if self.debug is not None:
            self.debug.open()

    def commit(self, record_event: bool = True) -> None:
        """Apply every staged write (one launch), then the deferred native ops.

        ``record_event=False``: the caller records an event right after this
        returns and hands it to ``set_completion_event``.
        """
        if not self.active:
            return
        try:
            if self.debug is not None:
                self.debug.close()
                self.debug.verify()
            for kind, args in self._writes:
                self._apply(kind, args)
            self.last_num_descriptors = self.packet.commit(record_event=record_event)
            deferred = self._deferred
        finally:
            self._writes = []
            self._deferred = []
            self.active = False
        self.num_commits += 1
        if self.debug is not None and self.num_commits % 64 == 1:
            from tensorrt_llm.logger import logger

            logger.info(
                f"[glm_moe_dsa] deferred_input_prep debug level {self.debug.level}: commits={self.num_commits} "
                f"decisions={self.decisions} flushes={self.num_flushes} "
                f"native={self.num_native_copies} guard_checked_ops={self.debug.num_checked_ops}"
            )
        for fn in deferred:
            fn()

    def flush(self) -> None:
        """Issue every write staged so far, in staging order, before the window closes.

        For code inside the deferred-write window that must touch a staged
        buffer with another device operation: once ``flush`` returns, the
        staged values precede that operation in stream order. Later stagings
        go into a new packet; deferred native operations still run at the
        commit. The graph replay copies ``input_ids`` and ``position_ids``
        from the input buffers again (``static_inputs_tokens`` is 0 unless
        later stagings write both static inputs again), so a native write to
        ``input_ids_cuda`` or ``position_ids_cuda`` after ``flush`` reaches
        the graph.
        """
        if not self.active:
            return
        with self._internal():
            if self.debug is not None:
                self.debug.verify()
            writes, self._writes = self._writes, []
            for kind, args in writes:
                self._apply(kind, args)
            if self.packet.num_descriptors:
                self._flush()
        self._static_input_ids = 0
        self._static_positions = 0

    def _flush(self) -> None:
        self.packet.commit(record_event=True)
        self.packet.begin()
        self.num_flushes += 1

    def _apply(self, kind: str, args: tuple) -> None:
        try:
            self._add(kind, args)
            return
        except (PacketFull, PacketOverlap):
            pass
        if self.packet.num_descriptors:
            self._flush()
            try:
                self._add(kind, args)
                return
            except PacketFull:
                pass
        self.num_native_copies += 1
        self._native(kind, args)

    def _add(self, kind: str, args: tuple) -> None:
        pk = self.packet
        if kind == "copy":
            dst, src, clamp = args
            pk.add_copy(dst, src, clamp_min=clamp)
        elif kind == "values":
            pk.add_values(*args)
        elif kind == "gather":
            pk.add_gather(*args)
        elif kind == "zero":
            pk.add_zero(*args)
        elif kind == "clamp":
            pk.add_clamp_min(*args)
        elif kind == "block_row":
            dst_k, dst_v, host_row, scale, kv_offset = args
            pk.add_block_offsets(
                dst_k,
                dst_v,
                host_row.data_ptr(),
                host_row.numel(),
                scale,
                kv_offset,
                BAD_PAGE_INDEX,
            )
        else:
            raise ValueError(f"unknown staged write kind {kind!r}")

    @staticmethod
    def _native(kind: str, args: tuple) -> None:
        """Issue one staged write with native torch operations."""
        if kind == "copy":
            dst, src, clamp = args
            dst.copy_(src, non_blocking=True)
            if clamp is not None:
                dst.clamp_(min=clamp)
        elif kind == "values":
            dst, values = args
            dst.copy_(torch.tensor(values, dtype=dst.dtype), non_blocking=True)
        elif kind == "gather":
            dst, src, indices, addend = args
            flat = torch.as_strided(
                src, (src.untyped_storage().nbytes() // 4,), (1,), src.storage_offset()
            )
            dst.zero_()
            if indices:
                idx = torch.tensor(indices, dtype=torch.long).to(src.device, non_blocking=True)
                dst[: len(indices)].copy_(flat[idx] + addend)
        elif kind == "zero":
            args[0].zero_()
        elif kind == "clamp":
            dst, lo = args
            dst.clamp_(min=lo)
        elif kind == "block_row":
            dst_k, dst_v, host_row, scale, kv_offset = args
            v = host_row.to(dst_k.device, non_blocking=True)
            bad = v == BAD_PAGE_INDEX
            dst_k.copy_(torch.where(bad, 0, v * scale))
            dst_v.copy_(torch.where(bad, 0, v * scale + kv_offset))
        else:
            raise ValueError(f"unknown staged write kind {kind!r}")

    def _stage(self, kind: str, args: tuple, *dsts, overwrite: bool = True) -> None:
        self._writes.append((kind, args))
        if self.debug is not None:
            for dst in dsts:
                self.debug.register(dst, overwrite)

    def _internal(self):
        """Context for the stager's own operations (not checked by the debug guard)."""
        return self.debug.internal() if self.debug is not None else contextlib.nullcontext()

    # ---- engagement (decided before any device work of the preparation) -

    _ATTACH_ATTR = "step_input_stager"

    def begin_if_eligible(
        self,
        preparer,
        requests,
        attn_metadata,
        spec_metadata,
        new_tensors_device,
        *,
        graph_replay: bool,
        static_inputs: Optional[dict],
        is_dummy: bool,
        enable_spec_decode: bool,
        promoted: bool,
    ) -> Optional["PrepStager"]:
        """Begin staging this preparation if it is in the deferred_input_prep support domain.

        Support domain: replay of an already captured decode graph, no dummy or
        promoted-context rows, generation-only batch whose rows all
        carry the previous iteration's device tensors (overlap scheduler),
        linear one-model speculative decoding, no beam search / helix /
        attention-DP / pipeline parallelism / MRoPE / LoRA / guided decoding,
        DSA attention metadata on a KV-cache manager V2, pinned host staging
        memory (``prefer_pinned``). Anything else returns ``None`` (native
        path). The first decision of each kind is logged once per process
        (``deferred_input_prep/prep`` when engaged, with the window contract;
        ``deferred_input_prep/prep[<reason>]`` per decline reason). An engaged call opens the
        deferred-write window (module docstring).
        """
        reason = (
            self._decline_reason
            if self._decline_reason is not None
            else self._ineligible_reason(
                preparer,
                requests,
                attn_metadata,
                spec_metadata,
                new_tensors_device,
                graph_replay,
                is_dummy,
                enable_spec_decode,
                promoted,
            )
        )
        self.last_reason = reason
        key = reason or "engaged"
        seen = self.decisions.get(key, 0)
        self.decisions[key] = seen + 1
        if not seen:
            _log_prep_decision(reason)
        if reason is not None:
            return None
        self.begin()
        self._static = self._usable_static_inputs(static_inputs)
        self._attached = [preparer, attn_metadata, spec_metadata, attn_metadata.kv_cache_manager]
        draft_manager = getattr(attn_metadata, "draft_kv_cache_manager", None)
        if draft_manager is not None:
            self._attached.append(draft_manager)
        for obj in self._attached:
            setattr(obj, self._ATTACH_ATTR, self)
        return self

    @staticmethod
    def _ineligible_reason(
        preparer,
        requests,
        attn_metadata,
        spec_metadata,
        new_tensors_device,
        graph_replay,
        is_dummy,
        enable_spec_decode,
        promoted,
    ) -> Optional[str]:
        cfg = preparer._config
        if not graph_replay or is_dummy or promoted:
            return "not a steady decode-graph replay"
        if not enable_spec_decode or spec_metadata is None or cfg.spec_config is None:
            return "speculative decoding off"
        mode = cfg.spec_config.spec_dec_mode
        if not (mode.use_one_engine() and cfg.spec_config.is_linear_tree):
            return "not linear one-model speculation"
        if cfg.disable_overlap_scheduler or cfg.enable_attention_dp or cfg.use_mrope:
            return "overlap off, attention DP or MRoPE"
        if preparer.use_beam_search or preparer.mapping.has_cp_helix():
            return "beam search or helix"
        if getattr(preparer.mapping, "pp_size", 1) > 1:
            # deferred_input_prep is not validated with the pipeline-parallel executor loop.
            return "pipeline parallelism"
        if preparer.guided_decoder is not None or preparer._lora.cuda_graph_manager is not None:
            return "guided decoding or LoRA graphs"
        if getattr(new_tensors_device, "next_draft_tokens", None) is None:
            return "no previous speculative device tensors"
        if requests.num_context_requests != 0:
            return "context rows"
        for r in requests.generation_requests:
            if r.is_dummy or r.py_batch_idx is None:
                return "dummy row or row without previous tensors"
        if type(attn_metadata).__name__ != "DSAtrtllmAttentionMetadata":
            return "attention metadata is not DSA"
        managers = [
            attn_metadata.kv_cache_manager,
            getattr(attn_metadata, "draft_kv_cache_manager", None),
        ]
        for manager in managers:
            if manager is None:
                continue
            if not hasattr(manager, "index_mapper") or getattr(
                manager, "_use_per_layer_page_tables", True
            ):
                return "KV cache manager is not V2 with a shared page table"
        return None

    def end(self, record_event: bool = True) -> None:
        """Close the deferred-write window: detach and launch the staged preparation."""
        self._detach()
        self.commit(record_event=record_event)

    def set_completion_event(self, event: torch.cuda.Event) -> None:
        """Event recorded by the caller right after ``end(record_event=False)``."""
        self.packet.set_completion_event(event)

    def discard(self) -> None:
        """Detach after a failed preparation; the staged writes are dropped."""
        self._detach()
        if self.debug is not None:
            self.debug.close()
            self.debug.staged = []
        self._writes = []
        self._deferred = []
        self.packet.abort()
        self.active = False

    def _detach(self) -> None:
        for obj in getattr(self, "_attached", ()):
            if getattr(obj, self._ATTACH_ATTR, None) is self:
                setattr(obj, self._ATTACH_ATTR, None)
        self._attached = []
        self._static = None

    # ---- staging API used by the call sites -----------------------------

    def copy(self, dst: torch.Tensor, src: torch.Tensor, clamp_min: Optional[int] = None) -> None:
        """Stage ``dst.copy_(src, non_blocking=True)`` for a host ``src``.

        ``src`` is kept alive and read at commit time.
        """
        with self._internal():
            if src.is_cuda or src.numel() != dst.numel():
                raise ValueError("PrepStager.copy needs a host source of dst's size")
            if src.dtype != dst.dtype:
                # Same conversion as ``copy_``.
                src = src.to(dst.dtype)
            if dst.is_contiguous():
                self._stage("copy", (dst, src, clamp_min), dst)
                return
            if dst.dim() != 2 or dst.stride(1) != 1:
                raise ValueError("PrepStager.copy needs contiguous rows")
            src2 = src.reshape(dst.shape)
            for r in range(dst.shape[0]):
                self._stage("copy", (dst[r], src2[r], clamp_min), dst[r])

    def values(self, dst: torch.Tensor, values: Sequence[int]) -> None:
        """Stage ``dst.copy_(torch.tensor(values))`` for an int32 ``dst``."""
        with self._internal():
            if dst.dtype != torch.int32 or dst.numel() != len(values):
                raise ValueError("values needs an int32 destination of len(values)")
            self._stage("values", (dst, list(values)), dst)

    def gather(
        self, dst: torch.Tensor, src: torch.Tensor, indices: Sequence[int], addend: int = 0
    ) -> None:
        """Stage ``dst[:k] = src_storage[indices] + addend``, ``dst[k:] = 0``."""
        with self._internal():
            self._stage("gather", (dst, src, list(indices), addend), dst)

    def zero(self, dst: torch.Tensor) -> None:
        """Stage ``dst.zero_()``."""
        with self._internal():
            self._stage("zero", (dst,), dst)

    @_stager_internal
    def overlap_inputs(
        self,
        buffers,
        previous_batch_indices: Sequence[int],
        previous_pos_indices: Sequence[int],
        num_tokens: int,
        num_draft_tokens: int,
        tokens_per_gen_step: int,
        num_extend_requests: int,
        new_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
    ) -> None:
        """Stage the overlap-scheduler block of ``DecoderInputPreparer._copy_token_ids_to_device``.

        Native: zero ``previous_pos_id_offsets_cuda`` and
        ``previous_kv_lens_offsets_cuda``; upload the previous slots and
        per-token slots; gather the previous iteration's accepted tokens into
        ``input_ids_cuda``, its draft tokens into ``draft_tokens_cuda``, its
        accepted counts into the position offsets and ``count - T`` into the
        KV-length offsets, at the rows of the requests that carry previous
        tensors.
        """
        T = tokens_per_gen_step
        D = T - 1
        prev = len(previous_batch_indices)
        pos_off = buffers.previous_pos_id_offsets_cuda
        kv_off = buffers.previous_kv_lens_offsets_cuda
        if prev == 0:
            self.zero(pos_off)
            self.zero(kv_off)
            return
        nt_s0, nt_s1 = new_tokens.stride(0), new_tokens.stride(1)
        nd_s0, nd_s1 = next_draft_tokens.stride(0), next_draft_tokens.stride(1)
        nl_s0 = new_tokens_lens.stride(0)
        self.values(buffers.previous_batch_indices_cuda[:prev], previous_batch_indices)
        tok_idx = [t * nt_s0 + slot * nt_s1 for slot in previous_batch_indices for t in range(T)]
        self.gather(buffers.input_ids_cuda[num_tokens : num_tokens + prev * T], new_tokens, tok_idx)
        static = self._static
        if static is not None and num_tokens == 0:
            # Also fill the graph's static input buffer; CUDAGraphRunner.replay
            # then skips its device-to-device copy of input_ids.
            self.gather(static["input_ids"][: prev * T], new_tokens, tok_idx)
            self._static_input_ids = prev * T
        if D > 0:
            draft_idx = [
                slot * nd_s0 + d * nd_s1 for slot in previous_batch_indices for d in range(D)
            ]
            self.gather(
                buffers.draft_tokens_cuda[num_draft_tokens : num_draft_tokens + prev * D],
                next_draft_tokens,
                draft_idx,
            )
        self.values(buffers.previous_pos_indices_cuda[: prev * T], previous_pos_indices)
        lo, hi = (num_extend_requests - prev) * T, num_extend_requests * T
        self._zero_except(pos_off, lo, hi)
        self.gather(pos_off[lo:hi], new_tokens_lens, [i * nl_s0 for i in previous_pos_indices])
        lo, hi = num_extend_requests - prev, num_extend_requests
        self._zero_except(kv_off, lo, hi)
        self.gather(
            kv_off[lo:hi], new_tokens_lens, [s * nl_s0 for s in previous_batch_indices], addend=-T
        )

    @staticmethod
    def _usable_static_inputs(static: Optional[dict]) -> Optional[dict]:
        """The graph's static inputs when they hold token ids and 2-D positions, else None."""
        if not static or "input_ids" not in static or static["position_ids"].dim() != 2:
            return None
        return static

    def position_ids(self, dst: torch.Tensor, host_position_ids: torch.Tensor) -> None:
        """Stage ``dst.copy_(host_position_ids)`` and the graph's static copy of it."""
        with self._internal():
            n = dst.numel()
            self.copy(dst, host_position_ids)
            static = self._static
            if static is not None:
                self.copy(static["position_ids"][0, :n], host_position_ids)
                self._static_positions = n

    def static_inputs_tokens(self) -> int:
        """Tokens of the static graph inputs written by the last launch (0: none)."""
        if self._static_input_ids and self._static_input_ids == self._static_positions:
            return self._static_input_ids
        return 0

    def _zero_except(self, buf: torch.Tensor, lo: int, hi: int) -> None:
        if lo > 0:
            self.zero(buf[:lo])
        if hi < buf.numel():
            self.zero(buf[hi:])

    def clamp_min(self, dst: torch.Tensor, lo: int) -> None:
        """Stage ``dst.clamp_(min=lo)`` (reads the destination's current bytes)."""
        with self._internal():
            if dst.dtype != torch.int32:
                raise ValueError("clamp_min supports int32 only")
            self._stage("clamp", (dst, lo), dst, overwrite=False)

    def block_offsets(
        self,
        host_table: torch.Tensor,
        dst: torch.Tensor,
        copy_index: Sequence[int],
        index_scales: Sequence[int],
        kv_offsets: Sequence[int],
    ) -> None:
        """Stage ``copy_batch_block_offsets_to_device`` (KV cache manager V2).

        ``host_table``: pinned ``[pools, src_seqs, 2, blocks]`` page table;
        ``dst``: ``[pools, dst_seqs, 2, blocks]``. For every pool ``p`` and
        output row ``s``: ``dst[p, s, 0] = scale * host[p, copy_index[s], 0]``
        and ``dst[p, s, 1] = dst[p, s, 0] + kv_offset`` with ``BAD_PAGE_INDEX``
        mapped to 0 in both.
        """
        with self._internal():
            if not host_table.is_pinned() or host_table.dtype != torch.int32:
                raise ValueError("block_offsets needs the pinned int32 host page table")
            pools, _, kv, blocks = host_table.shape
            if kv != 2 or dst.shape[0] != pools or dst.shape[3] != blocks:
                raise ValueError("block_offsets: unexpected page-table layout")
            for p in range(pools):
                for s, src_row in enumerate(copy_index):
                    self._stage(
                        "block_row",
                        (
                            dst[p, s, 0],
                            dst[p, s, 1],
                            host_table[p, src_row, 0],
                            int(index_scales[p]),
                            int(kv_offsets[p]),
                        ),
                        dst[p, s, 0],
                        dst[p, s, 1],
                    )

    def scheduler_metadata(self, metadata) -> bool:
        """Stage ``Indexer.prepare_scheduler_metadata`` from host KV lengths.

        Native: ``kv_lens_cuda_2d[:G, :W] = gen_lens`` (broadcast), then
        ``scheduler_metadata_buffer`` / ``scheduler_metadata_buffer_full_next_n``
        = DeepGEMM paged-MQA schedules for next_n = 1 / W. ``gen_lens`` are the
        values this packet uploads into ``kv_lens_cuda``, known on the host
        (``kv_lens_runtime``), so the three results are staged as copies of the
        host replica (``paged_mqa_schedule``). Returns False (caller defers the
        native op) outside the replicated configuration.
        """
        with self._internal():
            return self._scheduler_metadata(metadata)

    def _scheduler_metadata(self, metadata) -> bool:
        from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _DG_SCHEDULE_BLOCK_KV

        from .paged_mqa_schedule import SPLIT_KV_FOR_BLOCK_KV, paged_mqa_logits_schedule

        split_kv = SPLIT_KV_FOR_BLOCK_KV.get(_DG_SCHEDULE_BLOCK_KV)
        if (
            split_kv is None
            or metadata.use_expanded_buffers_for_mtp
            or metadata._indexer_compress_ratio > 1
        ):
            return False
        num_ctx, num_gen = metadata.num_contexts, metadata.num_generations
        lens = metadata.kv_lens_runtime[num_ctx : num_ctx + num_gen].tolist()
        num_sms = metadata.num_sms
        width = min(metadata.gen_token_stride, metadata.kv_lens_cuda_2d.shape[1])
        sched = metadata.scheduler_metadata_buffer
        if tuple(sched.shape) != (num_sms + 1, 2) or sched.dtype != torch.int32:
            return False
        metadata.gen_indexer_kv_lens_cuda_runtime = metadata.get_indexer_kv_lens(
            metadata.kv_lens_cuda_runtime[num_ctx : num_ctx + num_gen]
        )
        lens_2d = torch.tensor([[x] * width for x in lens], dtype=torch.int32)
        self.copy(metadata.kv_lens_cuda_2d[:num_gen, :width], lens_2d)
        self.copy(sched, torch.from_numpy(paged_mqa_logits_schedule(lens, 1, split_kv, num_sms)))
        if metadata.max_draft_tokens > 0:
            self.copy(
                metadata.scheduler_metadata_buffer_full_next_n,
                torch.from_numpy(paged_mqa_logits_schedule(lens, width, split_kv, num_sms)),
            )
        return True

    def defer(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` (native device work that reads staged values) after the launch."""
        self._deferred.append(fn)
