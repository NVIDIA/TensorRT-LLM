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

This module is the whole API: callers import only its names, every other module in the package
is private, and ``attach_staging`` loads the implementation. The docstrings of these names state
every obligation a caller keeps and every limit of this first version.

Roles: the integrator attaches one lender to a manager; the holder lends, polls, marks arrivals
and releases; the backend moves the bytes; the waiter asks ``StagingLender.readiness``. Lender,
lease and hold methods run only on the manager's thread, one call at a time: the builder, then the
executor loop, then the shutdown thread. The lender takes no locks, binds no owner thread and
starts no thread, so all progress happens inside lender calls. Positions are tokens.
"""

import typing as _typing

from ._types import (
    GroupRun,
    Lease,
    Part,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
)

if _typing.TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "GroupRun",
    "Lease",
    "Part",
    "PartsHold",
    "Readiness",
    "RegionView",
    "StagingLender",
    "StagingOptions",
    "attach_staging",
]


def attach_staging(
    manager: "KVCacheManagerV2", *, scope: bytes, staging: StagingOptions
) -> StagingLender:
    """Attach the manager's one lender, which relays whole blocks through host staging.

    The lender copies blocks between a request's device pages and host slots on the manager's
    execution stream, for backends that can only read and write host memory, and says where the
    request may resume after a fetch. ``StagingLender`` states what its holder and backends keep.

    First-version limits:
        - One lender per manager, for the manager's life.
        - Staging needs block reuse: a publish lends only committed blocks, so a manager that
          commits no blocks (block reuse off, or a draft manager without joint reuse) is refused.
        - A manager built for a one-model draft that reads prompt tokens past a position is refused:
          one-model Eagle and MTP-Eagle read 1 and vanilla MTP its draft length, while a block's
          name covers only the tokens up to the block's end, so two requests sharing those tokens
          could hold different draft bytes under one name. The check reads the manager, not
          ``is_draft``: draft layers sharing the target's manager, a joint-reuse draft pool, and the
          target's manager of such a draft whose layers live elsewhere are all refused. Drafts that
          read no token ahead (PARD, DFlash, DSpark) and managers without speculative decoding are
          accepted. A draft whose read-ahead upstream has not established counts as reading none:
          for one-model DraftTarget, staging accepts the target's manager and refuses its draft
          pool, which stays unpaired and commits no blocks, so a fetch fills only the target's
          blocks, as the manager's own prefix reuse does.
        - Helix and every other context-parallel manager (``mapping.cp_size > 1``) are refused:
          the layout derives only tensor-parallel shards, so their ranks would give different
          pages the same names.
        - Pipeline-parallel managers (``mapping.pp_size > 1``) are refused: each stage's staging
          slots and windows are its own, so one call could raise on one stage and lend on another.
        - A manager with a layer group holding recurrent state is refused.
        - A manager with a layer group of sparse buffers (``is_sparse``), whose read-only pages a
          cache can lock in host memory, is refused. Sparse attention whose buffers are not marked
          sparse is not refused by this.
        - A manager with a KV cache connector is refused: the connector serves a request's prefix
          at its first context chunk, measured from the committed tokens, so it cannot lower a
          history a windowed fetch moved, and its loads run off the manager's stream, unordered
          with staging copies.
        - Names follow the K/V arrangement the manager declares, not the one its attention backend
          writes; ``scope`` covers the difference.
        - A fetch fills only this manager's blocks: with a joint-reuse draft pool the caller
          fetches through the draft pool's lender too and resumes as ``Readiness`` states.
        - A publish leaves out the window blocks the publishing request's own window has passed,
          although the manager's prefix tree may still hold their committed pages, so a fetch whose
          window still keeps such a block finds its row missing (``StagingLender.lend_read``).
        - DeepSeek-V4 keeps every window the draft length wider under any speculative decoding, and
          a fetch asks for the rows of that margin too: at 128 tokens per block and a draft length
          of 2 or more, a publish from a request whose history stands at least the draft length less
          one token past the fetch's end leaves out the margin's row, so the fetch finds it missing.
        - Under the all-reusable block reuse policy a fetch fails at the call where a window leaves
          behind, at the fetch's end, a block whose page holds tokens past the cache's history: the
          block keeps its page until the commit stores it whole, the request never wrote those
          tokens (the rest of a block its local match copied from another request's page, or a page
          grown before the fetch), and the manager has no way to drop one block's page in one layer
          group. With partial reuse on, local matches often end inside a block, so on a model with a
          sliding window such a request computes from its local match rather than fetch past the
          window (``StagingLender.lend_write``).
        - A name covers the tokens the manager commits, their multimodal digests and the reuse
          scope, not other inputs a request carries: staging lends nothing by name for a request
          with multimodal data but no digests or with encoder input, so the decoder self-attention
          pool of an encoder-decoder model is not lent by name.
        - A name trusts the multimodal digests the input pipeline computes, as the manager's own
          prefix reuse does. A digest covers an item's bytes, not how the input processor treats
          them: a NumPy array and a ``torch.Tensor`` with the same dtype, shape and bytes share a
          digest, while some input processors rescale one and not the other, and a request's
          ``mm_processor_kwargs`` reach the processor but not the digest, so blocks with different
          KV bytes can share a name. A caller serving requests that the input processor may treat
          differently keeps them apart with per-item ``multi_modal_uuids``, which names and the
          multimodal encoder cache both cover. A ``cache_salt`` covers names but not that cache,
          which some models keep by default and which keys an item's embedding by its digest and the
          request's ``mm_processor_kwargs``, not the salt: salted requests can still share an
          embedding, so a ``cache_salt`` alone keeps them apart only with that cache off
          (``multimodal_config.encoder_cache_max_bytes=0``).
        - ``scope`` is fixed at the attach, and lending by name stops for good once the manager
          resets its reuse state (``reset_reuse_state``, with every request's cache closed), as an
          in-place weight update does: every later ``lend_read`` and ``lend_write`` that raises no
          ``ValueError`` fails at the call, and a publish granted before finishes.

    Caller obligations:
        - Attach on the thread that builds the executor, to the manager the executor serves. A
          manager built only to estimate the KV cache size shuts down first, and a lender attached
          there stops serving at that shutdown, which frees its staging memory, or keeps it until
          the process exits where a lease or a parts hold is still open then or a slot was lost to a
          failed copy.
        - Shut the manager down last, after the shutdown steps ``StagingLender`` lists. A manager
          that never shuts down keeps the staging memory and its own page-index host buffer until
          the process exits, and a shutdown that raises keeps that buffer until a retried shutdown
          returns, or else until the process exits: the runtime's shutdown raises while any cache is
          open, also one whose close raised in a free, unless the runtime is poisoned. Its leases,
          released ones too, and the lender's records also keep the requests' caches, and with them
          the manager's device pools, until the leases and the lender are dropped; drop the last
          reference on a thread that has used the device's CUDA context.
        - For a request whose one-model draft has a joint-reuse pool of its own, attach a lender to
          each of the two managers, the draft pool's under a ``scope`` of its own (laid out alike,
          the two pools would name their blocks alike), and keep what ``StagingLender`` states for
          such a pair.
        - An attach that raises for a reason its docstring does not list may keep the staging memory
          it allocated until the process exits.

    Args:
        manager: A ``KVCacheManagerV2`` with a ``mapping`` that commits blocks to its prefix-reuse
            tree: block reuse on, and joint reuse for a draft manager; not built for a one-model
            draft that reads prompt tokens ahead, and without a KV cache connector. Subclasses are
            accepted; one whose ``try_commit_blocks`` keys blocks from other tokens or another reuse
            scope than the base manager's changes names silently, at worst putting wrong bytes under
            a correct name.
        scope: Equal exactly where KV bytes mean the same; at most 65535 bytes. The lender does
            not read it, only puts it into every name's namespace, so a different ``scope`` never
            shares a name. It must cover every configuration that changes a block's bytes beyond
            what the manager declares: the model, its weights and LoRA adapters (the adapter an id
            maps to), the one-model draft whose pool it is (its weights, its mode and the target
            layers it reads), element types and quantization, the attention (windows such as
            ``max_attention_window``, sinks, sparse attention), the attention backend's K/V
            arrangement (head-major or token-major; the base manager declares head-major whichever
            backend writes its pages), and a release that changes what KV bytes mean. An adapter
            enters a name only through the request's ``lora_task_id``: instances sharing a
            ``scope`` map each id to the same adapter, or else lend no LoRA request by name or take
            different scopes. The framework's assembly builds it from its configuration, so no
            backend guesses which settings it needs.
        staging: The staging size, in whole fetches.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping, ``scope`` is not
            ``bytes``, or ``staging`` is not ``StagingOptions``.
        ValueError: A lender is attached already; the manager is context- or pipeline-parallel,
            holds recurrent state, commits no blocks (block reuse off, or a draft manager without
            joint reuse), has a KV cache connector, or is built for a one-model draft that reads
            prompt tokens ahead; it has a layer group of sparse buffers, whose read-only pages a
            cache can lock in host memory; its layer groups, layer configs and declarations do not
            agree; it has no pages to stage; ``scope`` is longer than 65535 bytes;
            ``staging.max_bytes`` is below one fetch.
        MemoryError: Page-locking the staging memory failed.
    """
    from ._lender import _attach_staging as _attach

    return _attach(manager, scope=scope, staging=staging)
