# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The work between two decode CUDA graphs as at most two launches.

The model declares both launches to the executor as its ``runtime_hooks``
(``make_runtime_hooks``; contract in ``tensorrt_llm/_torch/pyexecutor/runtime_hooks.py``).
Between two decode graphs the built-in executor issues, per step, 13 pinned
host-to-device copies, a pageable one, 12 index / copy kernels, 2 ``mul_`` and a
``sub`` that zero and fill the overlap-scheduler offsets,
``copyBatchBlockOffsetsToDeviceKernel`` (which reads the whole
``max_blocks_per_seq`` row of every pool from pinned memory with one CTA), 4
``index_copy_`` into the sampler store and 3 device-to-host copies of it: about
39 stream operations.

Prepare (``StepInputStaging``, which opens the model's step-input stager on
each step it accepts): while the executor prepares a steady decode step
(graph replay, generation rows only, every row carrying the previous step's
device tensors), the host-to-device
writes and the device-side gathers of the built-in prepare are recorded as
descriptors (``ops/runtime/step_commit.py``) and issued as one launch right
after the prepare. The recorded values are the built-in prepare's own: its
Python runs unchanged and only the device writes are rerouted. The block-offset
copy covers the columns a kernel can read, ``ceil(capacity / tokens_per_block) +
1`` per row, instead of the whole row. When the recorded writes cover the
replay's ``input_ids`` / ``position_ids`` prefix, the same launch also writes the
graph's static input buffers and the replay skips its two device-to-device
copies.

Sample (``SampleStoreWriter``, the model's sampler store writer): the sampler
store's four ``index_copy_`` and the three device-to-host snapshots become one
launch that writes the store rows and the pinned host snapshots directly.

Steps that do not qualify -- prefill or mixed batches, eager steps, graph
capture, dummies, any configuration outside the one ``StepInputStaging.ineligible``
accepts -- run the built-in code untouched. Within a qualifying step, a write
the stager cannot record (unsupported layout, or a staging block without room
for it; see ``StepStaging.room``) is given back to its caller after the writes
recorded so far are launched, and so is every later write of that step, so the
rest of the built-in prepare runs in its original order.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from .ops.runtime import step_commit as _sc

#: Store slot count (one per request: the sampler's ``max_batch_size``) above
#: which the sample-store launch declines. A step records 3 descriptors per
#: slot (the host snapshot) and 4 per sampled slot, at most 3,584 here, well
#: within ``step_commit.MAX_DESCRIPTORS``; a step the staging block cannot
#: hold yet declines and the block grows.
_MAX_STORE_SLOTS = 512
#: Pinned host snapshot sets kept in rotation; a set is reused after this many steps.
_HOST_RING = 4
#: Staged prepares between two debug lines of the counters.
_COUNTS_EVERY = 2000

#: Bytes of one element of the sampler store (int32).
_INT32_BYTES = 4

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _addr(t: torch.Tensor, element_offset: int = 0) -> int:
    return t.data_ptr() + element_offset * t.element_size()


class _PrepareStager:
    """One step's window, the model's ``StepInputStager``: records the prepare-phase device
    writes (see the module docstring).

    Sets itself as ``step_input_stager`` on the runner, the attention metadata
    and the speculative metadata it is given for as long as the window is open,
    so the executor's call sites route their writes to it. After its first
    decline it declines every later write, so the rest of the prepare runs
    natively, in its original order.
    """

    def __init__(
        self,
        staging: _sc.StepStaging,
        counts: Counter,
        runner: Any,
        attn_metadata: Any,
        spec_metadata: Any,
    ):
        self.staging = staging
        self.counts = counts
        self.runner = runner
        self.attn_metadata = attn_metadata
        self.declined: list[str] = []
        self.block_cols: Optional[int] = None
        self.block_seqs: int = 0
        self._static: Optional[tuple] = None
        # Writes into the buffers the graph replay copies into its static inputs:
        # name -> [(descriptor index, element offset, element count)].
        self._watch = {}
        for name in ("input_ids_cuda", "position_ids_cuda"):
            t = getattr(runner, name, None)
            if isinstance(t, torch.Tensor) and t.dtype == torch.int32 and t.is_contiguous():
                self._watch[name] = (t.data_ptr(), t.data_ptr() + t.numel() * 4)
        self.input_writes: dict[str, list[tuple[int, int, int]]] = {
            name: [] for name in self._watch
        }
        self._targets = tuple(t for t in (runner, attn_metadata, spec_metadata) if t is not None)
        for target in self._targets:
            target.step_input_stager = self

    def _close(self) -> None:
        for target in self._targets:
            target.step_input_stager = None

    def _dst(self, addr: int, nbytes: int) -> int:
        for name, (lo, hi) in self._watch.items():
            if lo <= addr < hi:
                self.input_writes[name].append((self.staging.num, (addr - lo) // 4, nbytes // 4))
        return addr

    def _decline(self, reason: str) -> bool:
        """Give a write back to the caller; issue what is recorded so far first so order holds."""
        self.declined.append(reason)
        self.staging.launch()
        return False

    def _after_decline(self) -> bool:
        """Whether a write of this window was declined: every later one is declined as well.

        The declined write's caller runs native device work that may read what
        later writes produce, so those writes must not wait for the launch at
        ``end``. Everything recorded was launched at the first decline.
        """
        return bool(self.declined)

    # StepInputStager -------------------------------------------------------

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        """Record ``dst.copy_(src, non_blocking=True)`` (``StepInputStager.copy``).

        Returns:
            True if the write is recorded; False if it is declined (the caller
            copies), which every write after an earlier decline is.
        """
        if self._after_decline():
            return False
        if (
            not dst.is_cuda
            or src.is_cuda
            or not dst.is_contiguous()
            or dst.numel() != src.numel()
            or dst.element_size() * dst.numel() % 4
        ):
            return self._decline(
                f"copy {tuple(dst.shape)} {dst.dtype} <- {tuple(src.shape)} {src.dtype}"
            )
        if src.dtype != dst.dtype:
            if src.dtype not in _INT_DTYPES or dst.dtype not in _INT_DTYPES:
                return self._decline(f"copy dtype {src.dtype} -> {dst.dtype}")
            src = src.to(dst.dtype)
        payload = src.contiguous().reshape(-1).view(torch.uint8).numpy()
        if not self.staging.room(1, payload.nbytes, 1):
            return self._decline("staging block full")
        self.staging.copy_bytes(self._dst(dst.data_ptr(), payload.nbytes), payload)
        return True

    def overlap_inputs(
        self,
        runner: Any,
        *,
        new_tokens_device: torch.Tensor,
        new_tokens_lens_device: torch.Tensor,
        next_draft_tokens_device: torch.Tensor,
        previous_batch_indices: list[int],
        previous_pos_indices: list[int],
        num_tokens: int,
        num_draft_tokens: int,
        tokens_per_gen_step: int,
        num_extend_requests: int,
    ) -> bool:
        """The overlap-scheduler branch of the built-in prepare, as descriptors.

        Built-in: zero ``previous_pos_id_offsets_cuda`` and
        ``previous_kv_lens_offsets_cuda``; upload the previous slots and
        position indices; gather each previous request's ``T`` tokens and ``T - 1``
        drafts into ``input_ids_cuda`` / ``draft_tokens_cuda``; its accepted
        length into ``T`` position offsets and ``length - T`` into one KV-length
        offset. Here the zeroing covers exactly the words the gathers do not
        write, so one launch needs no order between them.

        Returns:
            True if the branch's writes are recorded; False if they are declined
            (the caller runs the branch), which they are after an earlier decline.
        """
        if self._after_decline():
            return False
        T = tokens_per_gen_step
        W = T - 1
        P = len(previous_batch_indices)
        E = num_extend_requests
        pos = runner.previous_pos_id_offsets_cuda
        kvo = runner.previous_kv_lens_offsets_cuda
        tensors = (
            new_tokens_device,
            new_tokens_lens_device,
            next_draft_tokens_device,
            pos,
            kvo,
            runner.input_ids_cuda,
            runner.draft_tokens_cuda,
        )
        if any((not t.is_cuda) or t.dtype != torch.int32 for t in tensors):
            return self._decline("previous tensors: not all int32 on device")
        if (
            new_tokens_device.dim() != 3
            or new_tokens_device.shape[2] < 1
            or new_tokens_device.shape[0] < T
        ):
            return self._decline(f"previous tensors: new_tokens {tuple(new_tokens_device.shape)}")
        if next_draft_tokens_device.dim() != 2 or (W > 0 and next_draft_tokens_device.shape[1] < W):
            return self._decline(
                f"previous tensors: next_draft {tuple(next_draft_tokens_device.shape)}"
            )
        if (
            new_tokens_lens_device.dim() != 1
            or E < P
            or not pos.is_contiguous()
            or not kvo.is_contiguous()
        ):
            return self._decline("previous tensors: lens / offsets layout")
        # The gathers below index new_tokens_lens by slot; the built-in branch indexes it
        # by previous_pos_indices, which is each previous slot repeated T times.
        if list(previous_pos_indices) != [
            slot for slot in previous_batch_indices for _ in range(T)
        ]:
            return self._decline(
                "previous tensors: position indices are not the slots repeated T times"
            )
        lo, hi = (E - P) * T, E * T
        if hi > pos.numel() or E > kvo.numel():
            return self._decline("previous tensors: offsets out of range")
        if num_tokens + P * T > runner.input_ids_cuda.numel() or (
            W > 0 and num_draft_tokens + P * W > runner.draft_tokens_cuda.numel()
        ):
            return self._decline("previous tensors: inputs out of range")
        # 4 zero fills, then per previous slot 3 or 4 gathers, plus the 2 index uploads.
        descriptors = 4 + (2 + P * (3 + (W > 0)) if P else 0)
        if not self.staging.room(descriptors, P * 4 + P * T * 4, 2 if P else 0):
            return self._decline("staging block full")

        st = self.staging
        dst = self._dst
        # Zero what the built-in `*= 0` zeroes and nothing below rewrites.
        for buf, a, b in (
            (pos, 0, lo),
            (pos, hi, pos.numel()),
            (kvo, 0, E - P),
            (kvo, E, kvo.numel()),
        ):
            if b > a:
                st.fill(dst(_addr(buf, a), (b - a) * 4), 1, b - a, 0)
        if P == 0:
            return True
        idx = torch.tensor(previous_batch_indices, dtype=torch.int32)
        st.copy_bytes(
            dst(_addr(runner.previous_batch_indices_cuda), P * 4), idx.view(torch.uint8).numpy()
        )
        pidx = torch.tensor(previous_pos_indices, dtype=torch.int32)
        st.copy_bytes(
            dst(_addr(runner.previous_pos_indices_cuda), P * T * 4),
            pidx.view(torch.uint8).numpy(),
        )
        nt_s0, nt_s1 = new_tokens_device.stride(0), new_tokens_device.stride(1)
        nd_s0, nd_s1 = next_draft_tokens_device.stride(0), next_draft_tokens_device.stride(1)
        lens_s0 = new_tokens_lens_device.stride(0)
        for j, slot in enumerate(previous_batch_indices):
            st.gather(
                dst(_addr(runner.input_ids_cuda, num_tokens + j * T), T * 4),
                1,
                _addr(new_tokens_device, slot * nt_s1),
                nt_s0,
                T,
            )
            if W > 0:
                st.gather(
                    dst(_addr(runner.draft_tokens_cuda, num_draft_tokens + j * W), W * 4),
                    1,
                    _addr(next_draft_tokens_device, slot * nd_s0),
                    nd_s1,
                    W,
                )
            lens_addr = _addr(new_tokens_lens_device, slot * lens_s0)
            st.gather(dst(_addr(pos, lo + j * T), T * 4), 1, lens_addr, 0, T)
            st.gather(dst(_addr(kvo, E - P + j), 4), 1, lens_addr, 0, 1, add=-T)
        return True

    def block_offsets(
        self,
        manager: Any,
        dst: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> bool:
        """``KVCacheManagerV2.copy_batch_block_offsets`` over the readable columns only.

        A row's kernels read block columns below ``ceil(kv_len / tokens_per_block)``,
        and ``kv_len`` never exceeds the request's allocated capacity, so the
        first ``ceil(capacity / tokens_per_block) + 1`` columns (the batch maximum)
        carry everything read; the built-in copy moves all ``max_blocks_per_seq``.

        Returns:
            True if the copy is recorded; False if it is declined (the caller
            copies), which it is after an earlier decline.
        """
        if self._after_decline():
            return False
        host = getattr(manager, "host_kv_cache_block_offsets", None)
        if (
            host is None
            or getattr(manager, "sparse_metadata_batch", None) is not None
            or getattr(manager, "_use_per_layer_page_tables", True)
            # Shared attention-op tables are not the life cycles' SHARED-path tables.
            or getattr(manager, "_attention_op_pool_groups", None) is not None
            or not dst.is_cuda
            or dst.dtype != torch.int32
            or host.dtype != torch.int32
            or not dst.is_contiguous()
            or not host.is_contiguous()
            or dst.dim() != 4
            or host.dim() != 4
            or dst.shape[0] != host.shape[0]
            or dst.shape[2] != 2
            or dst.shape[3] != host.shape[3]
            or num_seqs > dst.shape[1]
            or beam_width > getattr(manager, "max_copy_beam_width", 0)
        ):
            return self._decline("block offsets: layout or manager mode")
        from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp

        if getattr(manager, "kv_cache_type", None) == CacheTypeCpp.CROSS:
            return self._decline("block offsets: cross KV")
        tokens_per_block = int(manager.tokens_per_block)
        num_blocks = host.shape[3]
        cols = 0
        for rid in request_ids:
            kv_cache = manager.kv_cache_map.get(rid)
            if kv_cache is None:
                return self._decline("block offsets: request without a KV cache")
            cols = max(cols, math.ceil(int(kv_cache.capacity) / tokens_per_block) + 1)
        cols = min(cols, num_blocks)
        copy_idx = manager.index_mapper.get_copy_index(request_ids, num_contexts, beam_width, False)
        rows = copy_idx.tolist()
        if len(rows) != num_seqs:
            return self._decline("block offsets: copy index length")
        if not self.staging.room(host.shape[0] * len(rows)):
            return self._decline("staging block full")
        scales = [int(s) for s in manager._index_scale_ints]
        kv_offsets = manager.kv_offset.tolist()
        host_rows, dst_rows = host.shape[1], dst.shape[1]
        st = self.staging
        for pool in range(host.shape[0]):
            for seq, row in enumerate(rows):
                src = _addr(host, (pool * host_rows + row) * 2 * num_blocks)
                k = self._dst(_addr(dst, (pool * dst_rows + seq) * 2 * num_blocks), cols * 4)
                v = self._dst(_addr(dst, ((pool * dst_rows + seq) * 2 + 1) * num_blocks), cols * 4)
                st.blocks(k, v, src, cols, scales[pool], int(kv_offsets[pool]))
        self.block_cols = cols
        self.block_seqs = num_seqs
        return True

    def end(self, inputs: dict) -> None:
        """Close the window and launch what it recorded (``StepInputStager.end``).

        Args:
            inputs: the prepared forward inputs. When the recorded writes cover
                the replay's ``input_ids`` / ``position_ids`` prefix, the same
                launch also writes the graph's static inputs (``static_inputs``).
        """
        self._close()
        mirrored = self._mirror_static_inputs(inputs)
        if mirrored is None:
            self.counts["static inputs copied by replay"] += 1
        self.staging.launch()
        self._static = mirrored
        if self.declined:
            self.counts["prepare partially declined"] += 1
            if self.counts["prepare partially declined"] <= 5:
                logger.info(f"[musespark_open] prepare staged with fallbacks: {self.declined}")
        self.counts["prepare staged"] += 1
        if self.counts["prepare staged"] % _COUNTS_EVERY == 0:
            logger.debug(f"[musespark_open] step input staging counts {dict(self.counts)}")

    def discard(self) -> None:
        """Close the window after a failed prepare; what it recorded and has not launched is
        dropped (``StepInputStager.discard``)."""
        self._close()
        self._static = None
        self.counts["prepare aborted"] += 1

    def static_inputs(self) -> Optional[tuple]:
        """``(input_ids ptr, position_ids ptr, tokens)`` that ``end`` also wrote into the graph's
        static inputs, or None when the replay must copy them (``StepInputStager.static_inputs``).
        """
        return self._static

    def _mirror_static_inputs(self, inputs: dict) -> Optional[tuple]:
        """Also write the replay's static ``input_ids`` / ``position_ids`` in this launch.

        ``CUDAGraphRunner.replay`` copies ``input_ids_cuda[:n]`` and
        ``position_ids_cuda[:n]`` into the graph's static buffers. When this step's
        recorded descriptors wrote every one of those ``n`` words and nothing else
        wrote them, the same descriptors aimed at the static buffers produce the
        same words, and the replay may skip its two copies. Returns the
        ``(input_ids ptr, position_ids ptr, n)`` the replay must present, or ``None``.
        """
        runner = self.runner
        graph_runner = getattr(runner, "cuda_graph_runner", None)
        static = getattr(graph_runner, "shared_static_tensors", None) or {}
        input_ids, position_ids = inputs.get("input_ids"), inputs.get("position_ids")
        if (
            self.declined
            or getattr(self.attn_metadata, "padded_num_tokens", None) is not None
            or set(self.input_writes) != {"input_ids_cuda", "position_ids_cuda"}
            or not isinstance(input_ids, torch.Tensor)
            or not isinstance(position_ids, torch.Tensor)
            or input_ids.data_ptr() != runner.input_ids_cuda.data_ptr()
            or position_ids.data_ptr() != runner.position_ids_cuda.data_ptr()
            or position_ids.dim() != 2
            or "input_ids" not in static
            or "position_ids" not in static
            or static["position_ids"].dim() != 2
        ):
            return None
        n = input_ids.numel()
        targets = {
            "input_ids_cuda": static["input_ids"],
            "position_ids_cuda": static["position_ids"],
        }
        if any(
            t.numel() < n or not t.is_contiguous() or t.dtype != torch.int32
            for t in targets.values()
        ):
            return None
        desc = self.staging._desc_np
        rows = []
        for name, writes in self.input_writes.items():
            covered = sorted((off, off + cnt) for _, off, cnt in writes)
            end = 0
            for lo, hi in covered:
                if lo > end:
                    break
                end = max(end, hi)
            if end < n or any(off + cnt > n for _, off, cnt in writes):
                return None
            real = getattr(runner, name)
            for index, off, cnt in writes:
                row = desc[index].copy()
                if index >= self.staging.num or row[1] != _addr(real, off) or row[3] != cnt:
                    return None
                if row[0] in (_sc.KIND_GATHER, _sc.KIND_FILL) and row[6] != 1:
                    return None
                row[1] = _addr(targets[name], off)
                rows.append(row)
        if not self.staging.room(len(rows)):
            return None
        for row in rows:
            self.staging._add(*(int(v) for v in row[:7]))
        return (input_ids.data_ptr(), position_ids.data_ptr(), n)


class StepInputStaging:
    """The model's ``StepInputStagerSource``: one ``_PrepareStager`` window per eligible step."""

    def __init__(self, pp_size: int = 1):
        """Allocate the pinned staging block.

        Args:
            pp_size: the model's pipeline-parallel size; any value above 1 is ineligible.
        """
        self.pp_size = pp_size
        self.staging = _sc.StepStaging()
        self.counts: Counter = Counter()

    def ineligible(
        self,
        runner: Any,
        requests: Any,
        attn_metadata: Any,
        spec_metadata: Any,
        new_tensors_device: Any,
        graph_replay: bool,
        is_dummy: bool,
        enable_spec_decode: bool,
        promoted: Any,
    ) -> Optional[str]:
        """``None`` if this step's prepare may be staged, else why not."""
        from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
        from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec

        cfg = runner._config
        checks = (
            (not graph_replay, "no graph replay"),
            (is_dummy, "dummy step"),
            (not enable_spec_decode, "no speculative decoding"),
            (bool(promoted), "promoted context rows"),
            (requests.num_context_requests > 0, "context rows"),
            (
                not isinstance(new_tensors_device, SampleStateTensorsSpec),
                "no previous spec tensors",
            ),
            (getattr(new_tensors_device, "next_draft_tokens", None) is None, "no previous drafts"),
            (
                not isinstance(attn_metadata, TrtllmAttentionMetadata)
                or type(attn_metadata).prepare is not TrtllmAttentionMetadata.prepare,
                "attention metadata type",
            ),
            (
                getattr(attn_metadata, "draft_kv_cache_manager", None) is not None,
                "draft KV cache manager",
            ),
            (runner.guided_decoder is not None, "guided decoding"),
            (cfg.use_mrope, "mrope"),
            (cfg.enable_attention_dp, "attention DP"),
            (cfg.disable_overlap_scheduler, "overlap scheduler off"),
            (runner.use_beam_search, "beam search"),
            (runner.mapping.has_cp_helix(), "helix"),
            (getattr(runner.mapping, "pp_size", 1) > 1 or self.pp_size > 1, "pipeline parallelism"),
            (spec_metadata is None, "no spec metadata"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        return None

    def begin_if_eligible(
        self,
        runner: Any,
        requests: Any,
        attn_metadata: Any,
        spec_metadata: Any,
        new_tensors_device: Any,
        *,
        graph_replay: bool,
        is_dummy: bool,
        enable_spec_decode: bool,
        promoted: bool,
    ) -> Optional[_PrepareStager]:
        """Open this step's window, or keep the step native (``StepInputStagerSource``).

        Args:
            runner: the decoder runner about to prepare the step.
            requests: the step's scheduled requests.
            attn_metadata: the attention metadata the prepare fills.
            spec_metadata: the speculative metadata the prepare fills.
            new_tensors_device: the previous step's sampler outputs on the device.
            graph_replay: whether the step replays a captured CUDA graph.
            is_dummy: whether the step is a dummy step.
            enable_spec_decode: whether the step decodes speculatively.
            promoted: whether context rows were promoted into the graph batch.

        Returns:
            The window, already set as ``step_input_stager`` on the runner and
            both metadata objects; or None when ``ineligible`` names a reason
            (logged once per reason).
        """
        reason = self.ineligible(
            runner,
            requests,
            attn_metadata,
            spec_metadata,
            new_tensors_device,
            graph_replay=graph_replay,
            is_dummy=is_dummy,
            enable_spec_decode=enable_spec_decode,
            promoted=promoted,
        )
        if reason is not None:
            key = f"prepare declined: {reason}"
            if not self.counts[key]:
                # Once per reason, so that a configuration change that declines every
                # step (e.g. a re-classed attention metadata) is visible in the log.
                detail = ""
                if reason == "attention metadata type":
                    meta = type(attn_metadata)
                    detail = f" ({meta.__module__}.{meta.__qualname__})"
                logger.info(
                    f"[musespark_open] step input staging declined for the first time: "
                    f"{reason}{detail}"
                )
            self.counts[key] += 1
            return None
        self.staging.reset()
        return _PrepareStager(self.staging, self.counts, runner, attn_metadata, spec_metadata)


class SampleStoreWriter:
    """The model's ``SamplerStoreWriter``: the store rows and the host snapshots in one launch."""

    def __init__(self, pp_size: int = 1):
        """Allocate the two pinned staging blocks the steps alternate between.

        Args:
            pp_size: the model's pipeline-parallel size; any value above 1 declines,
                since the sampler's host snapshots are then sent asynchronously
                (``isend_object``), which the snapshot ring does not fence.
        """
        self.pp_size = pp_size
        self._staging = [_sc.StepStaging(), _sc.StepStaging()]
        self._turn = 0
        self._host_ring: list[Optional[tuple[torch.Tensor, ...]]] = [None] * _HOST_RING
        self._host_events: list[Optional[torch.cuda.Event]] = [None] * _HOST_RING
        self._host_turn = 0
        self.counts: Counter = Counter()
        #: ``record_store``'s layouts, one per set of store and host-snapshot tensors.
        self._layouts: dict = {}
        #: The store tensors whose dtype, device and layout were last checked, and the result.
        self._checked_stores: tuple = ()
        self._stores_reason: Optional[str] = None
        #: The last slot list found unique and inside the store, and the store's slot count.
        self._checked_slots: Optional[list] = None
        self._checked_num_slots = -1

    def store_and_copy_out(
        self,
        sampler: Any,
        slots: list[int],
        new_tokens: torch.Tensor,
        next_new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        verify_lens: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Write the store rows and the host snapshots in one launch; ``None`` to decline.

        Returns the host snapshots ``(new_tokens, new_tokens_lens, next_draft_tokens)``,
        each shaped like its store tensor, as the built-in ``_copy_to_host`` does.
        """
        store = sampler.store
        stores = (
            store.new_tokens,
            store.next_new_tokens,
            store.new_tokens_lens,
            store.next_draft_tokens,
        )
        outs = (new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens)
        reason = None
        if self.pp_size > 1:
            reason = "pipeline parallelism"
        elif verify_lens is not None:
            reason = "verify_lens"
        elif sampler._async_worker_active():
            reason = "async worker"
        elif not prefer_pinned():
            reason = "no pinned host memory"
        elif any((not t.is_cuda) or t.dtype != torch.int32 for t in outs):
            reason = "dtypes"
        elif (reason := self._check_stores(stores)) is not None:
            pass
        elif not self._slots_fit(slots, store.new_tokens_lens.shape[0]):
            reason = "slots"
        elif (
            new_tokens.dim() != 2
            or new_tokens.shape[1] != store.new_tokens.shape[0]
            or next_new_tokens.dim() != 2
            or next_new_tokens.shape[1] != store.next_new_tokens.shape[0]
            or next_draft_tokens.dim() != 2
            or next_draft_tokens.shape[1] != store.next_draft_tokens.shape[1]
            or new_tokens_lens.dim() != 1
            or any(o.shape[0] != len(slots) for o in outs)
            or store.new_tokens.dim() != 3
            or store.new_tokens.shape[2] != 1
            or store.next_new_tokens.dim() != 3
            or store.next_new_tokens.shape[2] != 1
        ):
            reason = "shapes"
        if reason is not None:
            self.counts[f"sample declined: {reason}"] += 1
            return None

        staging = self._staging[self._turn]
        self._turn ^= 1
        staging.reset()
        # 3 snapshot gathers per store slot, 4 store gathers per sampled slot.
        if not staging.room(3 * store.new_tokens_lens.shape[0] + 4 * len(slots)):
            self.counts["sample declined: staging block full"] += 1
            return None
        turn = self._host_turn
        self._host_turn = (turn + 1) % _HOST_RING
        if self._host_events[turn] is not None:
            self._host_events[turn].synchronize()
        ring = self._host_ring[turn]
        if ring is None or any(
            r.shape != s.shape for r, s in zip(ring, (stores[0], stores[2], stores[3]))
        ):
            # Pinned: the launch writes these host snapshots directly.
            ring = tuple(
                torch.empty_like(t, device="cpu", pin_memory=prefer_pinned())
                for t in (stores[0], stores[2], stores[3])
            )
            self._host_ring[turn] = ring
        self.record_store(staging, stores, stores, outs, ring, slots, self._layouts)
        staging.launch()
        # This turn's event was synchronized above, so it can be recorded again.
        event = self._host_events[turn]
        if event is None:
            event = torch.cuda.Event()
            self._host_events[turn] = event
        event.record()
        self.counts["sample staged"] += 1
        return ring

    def _check_stores(self, stores: tuple) -> Optional[str]:
        """Why the store tensors do not fit the launch, or None; re-checked only when they change.

        The sampler's store tensors live as long as the sampler, and a tensor's
        dtype, device, shape and strides do not change in place.
        """
        if len(stores) != len(self._checked_stores) or any(
            a is not b for a, b in zip(stores, self._checked_stores)
        ):
            if any((not t.is_cuda) or t.dtype != torch.int32 for t in stores):
                reason = "dtypes"
            elif any(not t.is_contiguous() for t in stores):
                reason = "store layout"
            elif stores[2].shape[0] > _MAX_STORE_SLOTS:
                reason = "too many slots"
            else:
                reason = None
            self._checked_stores, self._stores_reason = stores, reason
        return self._stores_reason

    def _slots_fit(self, slots: list, num_slots: int) -> bool:
        """Whether ``slots`` are distinct and inside the store; a repeat of the last good list is."""
        if num_slots == self._checked_num_slots and slots == self._checked_slots:
            return True
        if len(set(slots)) != len(slots) or (slots and (min(slots) < 0 or max(slots) >= num_slots)):
            return False
        self._checked_slots, self._checked_num_slots = list(slots), num_slots
        return True

    @staticmethod
    def record_store(staging, stores, targets, outs, host, slots, layouts=None) -> None:
        """Record the store-row gathers into ``targets`` and the snapshot gathers into ``host``.

        ``stores`` are the sampler's store tensors (the rows not sampled this step
        are snapshotted from them); ``targets`` receive the sampled rows. Every
        tensor is int32. Per sampled slot ``s`` (output row ``j``): four gathers
        of row ``j`` into the targets' slot ``s`` and three into the host
        snapshot's slot ``s``; per other slot, three gathers of the store's slot
        ``s`` into the host snapshot. No two descriptors write the same element
        and none reads what another writes, so their order does not matter.

        Everything but the output rows' addresses depends only on the store,
        target and host tensors, so it is a ``_StoreLayout`` built once per set
        of them (kept in ``layouts`` when given); a step selects its slots' rows
        of it and fills in the output addresses, in a few numpy operations.
        """
        key = (
            tuple(t.data_ptr() for t in stores + targets + host)
            + tuple(stores[0].shape[:2])
            + (stores[1].shape[0], stores[3].shape[1])
        )
        layout = None if layouts is None else layouts.get(key)
        if layout is None:
            layout = _StoreLayout(stores, targets, host)
            if layouts is not None:
                if len(layouts) >= 2 * _HOST_RING:
                    layouts.clear()
                layouts[key] = layout
        layout.record(staging, outs, slots)


class _StoreLayout:
    """The sample-store descriptors of one set of store, target and host tensors.

    ``sampled[s]`` holds slot ``s``'s seven descriptors when it is sampled, with
    the source address and stride left for the step's output row;
    ``others[s]`` its three snapshot descriptors when it is not. The table of
    the last slot list is kept: a step with the same slots in the same order
    (a steady batch) only rewrites the output rows' addresses in it.
    """

    def __init__(self, stores, targets, host):
        new_tokens, next_new_tokens, lens, drafts = stores
        t_new_tokens, t_next_new_tokens, t_lens, t_drafts = targets
        h_new_tokens, h_lens, h_drafts = host
        num_slots = lens.shape[0]
        Tn, Tnn, D = new_tokens.shape[0], next_new_tokens.shape[0], drafts.shape[1]
        s = np.arange(num_slots, dtype=np.int64)

        def at(t: torch.Tensor, width: int = 1) -> np.ndarray:
            return t.data_ptr() + s * (width * _INT32_BYTES)

        def rows(*descriptors) -> np.ndarray:
            """``[num_slots, len(descriptors), DESC_FIELDS]`` from (dst, dst_stride, src,
            src_stride, count) per descriptor; ``src`` may be 0 (filled per step)."""
            table = np.zeros((num_slots, len(descriptors), _sc.DESC_FIELDS), dtype=np.int64)
            table[..., 0] = _sc.KIND_GATHER
            for i, (dst, dst_stride, src, src_stride, count) in enumerate(descriptors):
                table[:, i, 1] = dst
                table[:, i, 2] = src
                table[:, i, 3] = count
                table[:, i, 4] = src_stride
                table[:, i, 6] = dst_stride
            return table

        self.num_slots = num_slots
        sampled = rows(
            (at(t_new_tokens), num_slots, 0, 0, Tn),
            (at(t_next_new_tokens), num_slots, 0, 0, Tnn),
            (at(t_lens), 1, 0, 0, 1),
            (at(t_drafts, D), 1, 0, 0, D),
            (at(h_new_tokens), num_slots, 0, 0, Tn),
            (at(h_lens), 1, 0, 0, 1),
            (at(h_drafts, D), 1, 0, 0, D),
        )
        others = rows(
            (at(h_new_tokens), num_slots, at(new_tokens), num_slots, Tn),
            (at(h_lens), 1, at(lens), 0, 1),
            (at(h_drafts, D), 1, at(drafts, D), 1, D),
        )
        # Descriptors of a zero count (an empty store dimension) record nothing.
        self.sampled_keep = sampled[0, :, 3] > 0
        self.others_keep = others[0, :, 3] > 0
        self.sampled = sampled[:, self.sampled_keep]
        self.others = others[:, self.others_keep]
        self.max_count = max(Tn, Tnn, D, 1)
        self._slots: Optional[list] = None
        self._table: Optional[np.ndarray] = None

    def record(self, staging, outs, slots) -> None:
        """Record this step's descriptors: ``outs`` row ``j`` is slot ``slots[j]``'s."""
        o_new_tokens, o_next_new_tokens, o_lens, o_drafts = outs
        order = [o_new_tokens, o_next_new_tokens, o_lens, o_drafts, o_new_tokens, o_lens, o_drafts]
        order = [o for o, keep in zip(order, self.sampled_keep) if keep]
        n = len(slots)
        if slots != self._slots:
            sampled = np.asarray(slots, dtype=np.int64)
            others = np.ones(self.num_slots, dtype=bool)
            others[sampled] = False
            self._table = np.concatenate(
                (
                    self.sampled[sampled].reshape(-1, _sc.DESC_FIELDS),
                    self.others[others].reshape(-1, _sc.DESC_FIELDS),
                )
            )
            self._slots = list(slots)
        table = self._table
        head = table[: n * len(order)].reshape(n, len(order), _sc.DESC_FIELDS)
        base = np.array([o.data_ptr() for o in order], dtype=np.int64)
        row_bytes = np.array([o.stride(0) * _INT32_BYTES for o in order], dtype=np.int64)
        head[:, :, 2] = base + np.arange(n, dtype=np.int64)[:, None] * row_bytes
        head[:, :, 4] = [o.stride(1) if o.dim() == 2 else 0 for o in order]
        staging.extend(table, self.max_count)


def make_runtime_hooks(model: Any) -> RuntimeHooks:
    """Build the executor hooks the model declares (its ``runtime_hooks``).

    Allocates the step-input stager's pinned staging block, then the sampler
    store writer's two blocks, then compiles the launch kernel, so all of it
    exists before any CUDA graph is captured.

    Args:
        model: the shell; both parts get its ``_mapping.pp_size`` (1 without one).

    Returns:
        ``RuntimeHooks`` declaring both the step-input stager
        (``StepInputStaging``) and the sampler store writer (``SampleStoreWriter``).
    """
    pp_size = int(getattr(getattr(model, "_mapping", None), "pp_size", 1) or 1)
    hooks = RuntimeHooks(
        step_input_stager=StepInputStaging(pp_size),
        sampler_store_writer=SampleStoreWriter(pp_size),
    )
    _sc.warm_up()
    logger.info(
        f"[musespark_open] executor hooks: step input staging and sampler store writer, "
        f"pp_size={pp_size}"
    )
    return hooks
