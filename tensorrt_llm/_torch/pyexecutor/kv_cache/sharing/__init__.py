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
"""Lending a KV cache manager v2's blocks to transfer backends.

This module is the whole API: callers import only its names, the package's other modules are
private, and ``attach_lender`` selects the implementation. The docstrings of these names state every
caller obligation.

Roles: the integrator attaches one lender to a manager; the holder lends, polls, marks arrivals and
releases; the backend moves the bytes; the waiter asks ``Lender.resume_bounds``. Lender,
lease and hold methods run only on the manager's thread, one call at a time: the builder, then the
executor loop, then the shutdown thread. Positions are tokens. Ready views describe stable,
accessible lent memory; leases and region holds delimit backend access. Cache ownership remains
with the manager.
"""

import typing as _typing

from ._types import (
    LayerGroupBlocks,
    Lease,
    LeaseView,
    Lender,
    MemoryRegion,
    RegionHold,
    ResumeBounds,
    StagingOptions,
)

if _typing.TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "LayerGroupBlocks",
    "Lease",
    "LeaseView",
    "Lender",
    "MemoryRegion",
    "RegionHold",
    "ResumeBounds",
    "StagingOptions",
    "attach_lender",
]


def attach_lender(manager: "KVCacheManagerV2", *, scope: bytes, staging: StagingOptions) -> Lender:
    """Attach the manager's one lender, which relays whole blocks through host staging.

    The lender copies blocks between a request's device pages and host slots on the manager's
    execution stream, for backends that can only read and write host memory, and says where the
    request may resume after a fetch. ``Lender`` states what its holder and backends keep. A
    name covers ``scope``, the committed tokens up to its block's end, their multimodal digests and
    the manager's reuse scope, nothing else a request carries, so the decoder self-attention pool of
    an encoder-decoder model is not lent by name.

    Caller obligations:
        - This implementation reserves the request's device pages before waiting for host regions.
          Keep room for running requests in each pool group while fetches are parked: their locked
          pages, other unreclaimable pages, the largest generation request's full resume window and
          its next step must fit within ``max_util_for_resume``. V2 cannot evict or preempt a parked
          fetch. Reserving space for parked fetches alone does not prevent scheduler deadlock.
        - Copies use the manager's execution stream, ordered with its forward work. Any writer of
          recycled pages on another stream must wait for that stream first. The diagnostic fresh-
          page fill (``TRTLLM_KV_FRESH_PAGE_FILL``) already does so and then synchronizes its own
          stream; a fetch that grows pages with this diagnostic enabled can wait on the CPU.
          Pageable memory under confidential computing can also make a call wait for its stream.
        - A failed copy whose completion cannot be established quarantines its slots permanently.
          Capacity can erode until a pending lease cannot fit and blocks those behind it. Release
          leases that wait too long or free their requests. Quarantined slots, unreleased leases
          or region holds, and cleanup errors can keep host memory until process exit.
        - From ``lend_write`` until the request resumes or its cache is dropped, keep it among the
          executor's active requests and unscheduled until the resume state is applied. To resume at
          ``p``, set ``py_served_prefix_end=p``, raise the cache history to ``p`` if necessary and,
          back in context state with a chunk spanning the remaining prompt, call
          ``set_prepopulated_prompt_len(p, tokens_per_block)``. It moves the prepopulated length
          and context position together; moving either alone is not sufficient. A resume at 0
          from a later context position requires dropping the cache and recomputing.
        - With routed-expert capture, fetch only before the first context step, not after a
          recompute pause; otherwise skipped positions can have no recorded routes. A later
          fetch also does not change the response's original ``cached_tokens`` count.
        - Under attention data parallelism without a transceiver, do not park every active
          request on a rank unless the resulting all-rank forward stall is acceptable.
        - SWA scratch reuse must be off for fetches. A window must not abandon an uncomputed block
          still held for prefix reuse; with extra speculative KV tokens, keep later consecutive
          segments shorter than ``window + tokens_per_block - 1`` or compute the remainder.
          Requests requiring context logits, prompt logprobs or additional per-token model
          outputs cannot be fetched, because those outputs are not restored by this implementation.
        - These device-page reservations, execution-stream ordering, diagnostics, slot quarantine
          and executor-specific constraints belong to the current implementation and may change
          when its backing mechanism changes; the ``Lender`` contract does not expose them.
        - Attach on the thread that builds the executor, to the manager the executor serves. A
          manager built only to estimate the KV cache size shuts down first, after which a lender
          attached there lends nothing; that shutdown frees the staging memory, or keeps it until
          the process exits where a lease or staging hold is still open then, a slot was lost to a
          failed copy, an error in a free stopped the lender (``Lease``), or an error arose in the
          lender's own work within that shutdown.
        - Shut the manager down last, with the device's CUDA context used for the attach current
          on the calling thread, after the shutdown steps ``Lender`` lists. A manager
          that never shuts down keeps the staging memory and its own page-index host buffer until
          the process exits, and a shutdown that raises keeps that buffer until a retried shutdown
          returns, or else until the process exits: the runtime's shutdown raises while any cache is
          open, also one whose close raised in a free, unless the runtime is poisoned. Its leases,
          released ones too, and the lender's records also keep the requests' caches, and with them
          the manager's device pools, until the leases and the lender are dropped; drop the last
          reference on a thread that has used the device's CUDA context.
        - For a request whose one-model draft has a joint-reuse pool of its own, attach a lender to
          both managers, the draft pool's under a ``scope`` of its own (laid out alike, both pools
          would name blocks alike), fetch through both, since a fetch fills only this manager's
          blocks, and resume as ``ResumeBounds`` states, keeping what ``Lender`` states for
          such a pair.
        - A caller serving requests the input processor may treat differently keeps them apart with
          per-item ``multi_modal_uuids``, which names and the multimodal encoder cache both cover,
          since a name trusts the multimodal digests the input pipeline computes, as the manager's
          own prefix reuse does. A digest covers an item's bytes, not how the input processor treats
          them, so blocks with different KV bytes can share a name: a NumPy array and a
          ``torch.Tensor`` with the same dtype, shape and bytes share a digest though some input
          processors rescale only one, and a request's ``mm_processor_kwargs`` reach the processor
          but not the digest. A ``cache_salt`` covers names but not that cache, which some models
          keep by default and which keys an item's embedding by its digest and
          ``mm_processor_kwargs``: salted requests can still share an embedding, so a ``cache_salt``
          alone keeps them apart only with that cache off
          (``multimodal_config.encoder_cache_max_bytes=0``).
        - Under ``SaveHiddenStatesDecodingConfig`` a fetch restores only this manager's KV blocks,
          so the hidden states saved for a resumed request pair its whole prompt with the rows of
          the tokens it computed, then stale rows, misaligned as after the manager's own prefix
          reuse: the caller fetches into no such request, or accepts that.
        - An attach that raises for a reason its docstring does not list may keep the staging memory
          it allocated until the process exits.

    Args:
        manager: A ``KVCacheManagerV2`` with a ``mapping`` that commits blocks to its prefix-reuse
            tree: block reuse on, and joint reuse for a draft manager; not built for a one-model
            draft that reads prompt tokens ahead, and without a KV cache connector. Subclasses are
            accepted; one whose ``try_commit_blocks`` keys blocks from other tokens or another reuse
            scope than the base manager's changes names silently, at worst putting wrong bytes under
            a correct name.
        scope: Equal exactly where KV bytes mean the same; at most 65535 bytes. The lender only puts
            it, unread, into every name's namespace, so different scopes never share a name. It must
            cover every configuration that changes a block's bytes beyond what the manager declares:
            the model, its weights and LoRA adapters (the adapter an id maps to), the one-model
            draft whose pool it is (its weights, mode and the target layers it reads), element types
            and quantization, the attention (windows such as ``max_attention_window``, sinks, sparse
            attention), the attention backend's K/V arrangement (head-major or token-major; the base
            manager declares head-major whichever backend writes its pages), and a release that
            changes what KV bytes mean. An adapter enters a name only through the request's
            ``lora_task_id``: instances sharing a ``scope`` map each id to the same adapter, or else
            lend no LoRA request by name or take different scopes. The framework's assembly builds
            it from its configuration, so no backend guesses which settings it needs.
        staging: The staging size, in whole fetches.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping, ``scope`` is not
            ``bytes``, or ``staging`` is not ``StagingOptions``.
        ValueError: A lender is attached already; the manager is context-parallel (Helix included,
            ``mapping.cp_size > 1``) or pipeline-parallel (``mapping.pp_size > 1``), holds recurrent
            state, commits no blocks (block reuse off, or a draft manager without joint reuse),
            since a publish lends only committed blocks, has a KV cache connector or a lossy
            cold-page codec (``quantization_for_cold_page``), or is built for a
            one-model draft that reads prompt tokens ahead; it has a layer group of sparse buffers
            (``is_sparse``), whose read-only pages a cache can lock in host memory, though sparse
            attention whose buffers are not marked sparse is not refused by this; its layer groups,
            layer configs and declarations do not agree; it has no pages to stage; ``scope`` is
            longer than 65535 bytes; ``staging.max_bytes`` is below one fetch. One-model Eagle and
            MTP-Eagle read 1 prompt token ahead, vanilla MTP its draft length, and the check reads
            the manager, not ``is_draft``, so it refuses draft layers sharing the target's manager,
            a joint-reuse draft pool, and the target's manager of such a draft whose layers live
            elsewhere. Drafts reading no token ahead (PARD, DFlash, DSpark) and managers without
            speculative decoding are accepted. A draft whose read-ahead upstream has not established
            counts as reading none: for one-model DraftTarget, staging accepts the target's manager
            and refuses its unpaired draft pool, which commits no blocks, so a fetch fills only the
            target's blocks, as the manager's own prefix reuse does.
        MemoryError: Page-locking the staging memory failed.
    """
    from ._lender import _attach_staging as _attach

    return _attach(manager, scope=scope, staging=staging)
