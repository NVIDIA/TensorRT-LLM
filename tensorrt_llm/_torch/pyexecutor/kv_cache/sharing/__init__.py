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
"""Public KV-cache lending API; other modules are private. Positions are tokens.
The integrator attaches; orchestration lends; holders poll and return leases; backends move ready
bytes. Calls run serially on the manager thread through shutdown; leases/holds bound access;
managers own caches."""

import typing as _typing

from ._types import (
    ALL_ARRIVED,
    Arrived,
    HeldPages,
    HoldReason,
    Lease,
    LeaseView,
    Lender,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    RegionHold,
    ResumeBounds,
    RowSelection,
    StagingOptions,
)

if _typing.TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "ALL_ARRIVED",
    "Arrived",
    "HeldPages",
    "HoldReason",
    "LentLayerGroupRows",
    "Lease",
    "LeaseView",
    "Lender",
    "MemoryKind",
    "MemoryRegion",
    "RegionHold",
    "ResumeBounds",
    "RowSelection",
    "StagingOptions",
    "attach_lender",
]


def attach_lender(manager: "KVCacheManagerV2", *, scope: bytes, staging: StagingOptions) -> Lender:
    """Attach a lifetime lender; encoder-decoder models, including CROSS caches, are unsupported.

    HOST supports NAMED; DEVICE supports NAMED and ALL. HOST/ALL or HOST without staging raises
    ``ValueError``. Host regions precede device regions; ``Lender.page_layout`` locates each pool.
    Import layout types from ``tensorrt_llm._torch.pyexecutor.kv_cache.page_layout``.

    Names cover ``scope``, committed prefix tokens, multimodal digests and reuse scope, excluding
    other request inputs. Naming requires committed blocks, a lossless cold-page codec, no model
    recurrent state or context parallelism, and no one-model draft reading past a block's end.
    Otherwise only DEVICE/ALL is available. A HOST region establishes named-lending support;
    DEVICE naming follows these conditions. After reuse reset, NAMED returns failed leases while
    ALL remains available. A loading KV connector disables staging, but does not itself forbid
    DEVICE/NAMED. Eagle and MTP-Eagle look ahead one token, vanilla MTP its draft length; PARD,
    DFlash and DSpark read none. Checks include shared draft layers, joint-reuse pools and targets
    whose draft layers live elsewhere. Unknown upstream lookahead counts as zero, including
    DraftTarget: its target names blocks, its unpaired noncommitting draft does not.

    Contract obligations:
        These caller duties are part of the public lending contract; a lending method a later
        optional protocol adds states the duties of the leases it lends.

        - Follow ``Lender``'s thread, progress, registration and shutdown duties. A lease binds to
          the cache its request holds at the lend: do not run the request on a cache with an open
          in-place write or recurrent-state lease that ``lend_read`` or ``lend_write`` lent.
          Freeing such a cache retires it: its leases keep its pages protected until they are
          returned and their rows count nothing, while the request may run on a new cache, the
          pages then held twice (``Lender.held_page_counts``). Reads may overlap later steps;
          access only returned rows, whose memory remains protected until the lease is returned.
        - Attach on every cache-holding rank or none. Each pipeline stage lends only its own
          layers; ``lend_read`` and ``lend_write`` lend them after the stage processed the step.
          Aggregate attach outcomes across all ranks: any failure fails startup, since an attached
          lender cannot be detached. Model-wide exclusions are checked consistently across stages;
          ``max_bytes`` sufficiency is stage-local and can differ. For each distributed transfer,
          require delivery on every stage, return every stage's leases in the same iteration
          using the transfer path's allgather, and intersect bounds. Before ``return_writes``,
          settle the transfer's outcome on every participating rank and finish all backend access
          to its views. An arrival notification alone is insufficient; a HOST return may enqueue
          a copy from staging immediately.
        - Coordinate scheduling across cache-holding ranks. Protect every rank's locked requests
          from eviction and preemption, and run a batch only after every rank has allocated it.
          If any rank cannot allocate, roll back or retry consistently across ranks before forward.
          A pipeline root's scheduling result does not establish admission on the other stages;
          an integrator attaching lenders to a pipeline executor must supply this coordination.
        - Apply settled bounds before scheduling a request fetched through ``lend_write``. An
          empty intersection requires dropping its fetched caches and computing from zero. Fetch
          both participating joint-reuse target/draft pools before advancing their shared cursor;
          P/D receives that lend no draft pool do not intersect its bounds. Separate draft pools
          need a lender and distinct scope.
        - ALL arrivals need every transferred buffer's bytes this request will read. See
          ``Lender.return_writes`` for permitted omissions; return a partially filled receive
          with ``reusable=False``, and the lender keeps its cache out of reuse, as it does one
          missing window margins. NAMED always transfers the whole object.
        - Treat failed leases and empty bounds as request-level outcomes, including permanent local
          states such as reuse reset. Validation ``ValueError``/``TypeError`` from lends and
          returns and misuse-related ``RuntimeError`` from lease methods and returns are caller
          errors. Entry validation in ``return_writes`` changes nothing: correct the entries and
          retry; lending, progress and bounds queries remain permitted. Any error once returning
          entries begins permits cleanup only, even if other entries returned successfully.
          Exception type alone does not establish that validation failed; if unsure, treat it as
          an operation failure. ``RuntimeError`` from ``lend_*`` may also report a manager or CUDA
          operation failure. After an operation failure
          or an unlisted exception, stop lending, progress, lease polling and bounds queries; return
          leases and release holds after access ends, treat parked fetch intervals as empty, drop
          their caches and compute from zero.
        - Do not subtract ``held_page_counts().deduct`` from a V2 manager's admission capacity: a
          ``DEFERRED_CLOSE`` page stays with the cache it belongs to and is already outside the
          free pages, so subtracting it counts it twice. Fetches can hold device pages while
          waiting for host slots. Reserve the entire device allocation of every live cache that
          locks prevent from being preempted, plus ``DEFERRED_CLOSE`` pages, the largest generation
          resume window and its next step, within each pool group's ``max_util_for_resume``.
          ``held_page_counts().unreclaimable`` counts lease rows and deferred-close pages; it omits
          unleased pages of live locked caches, so it is insufficient for this reservation.
          A parked-fetch quota alone is insufficient. Use
          ``held_page_counts`` per pool group only to check that ranks and pipeline stages hold
          alike, and warn where they do not.
        - Use per-item ``multi_modal_uuids`` to distinguish processor-dependent inputs. Digests trust
          input-pipeline bytes, like prefix reuse, and omit differences such as NumPy/Tensor rescaling
          or ``mm_processor_kwargs``. ``cache_salt`` separates names but not encoder embeddings:
          salt alone suffices only with ``multimodal_config.encoder_cache_max_bytes=0``.

    Implementation constraints:
        The V2 recovery recipe and the following reservations, streams, diagnostics, quarantine
        and executor constraints depend on this implementation and are not frozen API obligations.

        - Attach on the construction thread. Remote-cache configuration alone does not attach a
          lender. A lender cannot lend after its manager shuts down.
        - Host staging copies whole named rows on the manager stream; connector writes are unordered
          with these copies. Each device pool is one region, fixed for the lender's life: do not
          rebalance the pools of a manager with a lender attached.
        - Before this worker finishes context, ALL reads end at finished whole blocks and in-place
          reads omit groups whose window is shorter than that context, with SWA scratch reuse on or
          off; read again after context finishes to obtain them. Recurrent groups lend one state row
          at prompt end, between context and generation; their fetch resumes there or nowhere. Under
          Helix, each rank lends pages of ``PageLayout.tokens_per_page`` tokens; starts align to
          ``cp_size * tokens_per_page`` and ``total_input_len_cp`` is the whole prompt length.
        - Locks prevent eviction, preemption, scheduler suspension and deadlock reports; history
          cannot abandon locked pages. Allocation-failure rollback in ``revert_allocate_context``
          and first-block ``resize_context`` skips suspension, leaving locked caches active;
          explicit suspension and admission paths raise ``RuntimeError`` on locks. Overlapping
          commits retry after unlock; freed caches close at their last unlock, and a reset with such
          caches open raises ``RuntimeError``. Manager query hooks propagate original exception
          types. These V2 errors report calling order, not a failed lease or poisoned domain.
        - With a connector, fetch at or beyond the served prefix; connector loads start beyond a
          fetched prefix. Reads reject prefix reservations and wait for pending loads, during which
          windowed NAMED fetches reject history advancement. Lend outside batch preparation: an
          accepted load can be abandoned before forward.
        - Keep fetched requests active but unscheduled. To resume at ``p``, set
          ``py_served_prefix_end=p``, raise cache history to ``p`` if needed, restore context state
          with a chunk covering the remaining prompt, then call
          ``set_prepopulated_prompt_len(p, tokens_per_block)`` to move both cursor fields. This
          needs ``p`` below prompt length (``total_input_len_cp`` under Helix). A full ALL fetch
          into generation-init resumes at prompt end as after a receive. Rewinding to zero requires
          dropping and recomputing.
        - Routed-expert capture allows fetching only before the first context step, never after
          recompute pause; skipped routes are not restored. Later fetches leave ``cached_tokens``
          unchanged. With attention data parallelism and no transceiver, parking every active
          request on one rank can stall all ranks' forward.
        - Disable SWA scratch reuse for fetches. Retain uncomputed prefix-reuse blocks; with extra
          speculative KV tokens, keep later consecutive segments shorter than
          ``window + tokens_per_block - 1`` or compute the remainder. Fetches do not restore context
          logits, prompt logprobs or additional per-token outputs. With
          ``SaveHiddenStatesDecodingConfig``, exclude fetches or accept saved states pairing the
          full prompt with computed then stale rows.
        - Writers recycling pages off the manager stream must wait for it. Diagnostic
          ``TRTLLM_KV_FRESH_PAGE_FILL`` waits and synchronizes its own stream, so growth can block
          the CPU, as confidential-computing pageable copies can. Shutdown waits for recorded
          copies; the first runtime cache close can synchronize the device. Unknown copy completion
          quarantines slots permanently: return leases waiting too long or free their requests.
        - Shut down with the attach device's CUDA context current. Without shutdown, staging and
          the manager's page-index host buffer survive until exit; failed shutdown retains that
          buffer until retry succeeds or exit. Unless poisoned, runtime shutdown raises with any
          cache open, including failed closes. Open locks/device holds retain runtime and pools;
          open leases/holds, lost slots and cleanup errors can retain host memory until exit.
          Even returned leases and lender records retain caches/pools: drop their last references
          on a thread that has used the device's CUDA context. Unlisted attach errors may retain
          allocated staging until exit.

    Args:
        manager: A ``KVCacheManagerV2`` with a mapping. Naming requires block reuse, joint reuse
            for a draft, and the restrictions above. Subclasses must preserve the committed tokens
            and reuse scope used by ``try_commit_blocks`` or names can identify wrong bytes.
        scope: At most 65535 bytes, equal exactly where KV bytes mean the same; different scopes
            never share names. Cover model/weights, LoRA id-to-adapter mapping, one-model draft
            weights/mode/target layers, dtypes/quantization, attention/windows/sinks/sparsity,
            backend K/V arrangement and incompatible releases. The base manager declares head-major
            regardless of writing backend. Equal scopes must map each ``lora_task_id`` to the same
            adapter, or exclude LoRA from names. Framework assembly derives scope; backends do not.
        staging: Host capacity in fetch units (``StagingOptions``).

    Returns:
        The lender installed for the manager's life.

    Raises:
        TypeError: Manager is not V2 or lacks a mapping; scope is not bytes or staging is not
            ``StagingOptions``.
        ValueError: Already attached; CROSS cache; non-Helix context parallelism; model windows or
            recurrent state under Helix; model sparse buffers (``is_sparse``), whose pages can live
            in host memory; a non-REQUIRED attention group without a window; inconsistent groups,
            configs or declarations, or a missing ``Role.ALL`` mapper declaration; no pages to
            stage; ``max_bytes`` below one local fetch; a malformed region, or two of one memory
            kind that overlap; or scope over 65535 bytes. Unmarked sparse-attention buffers are not
            rejected as sparse.
        MemoryError: Staging page-lock allocation failed."""
    from ._lender import _attach_lender as _attach

    return _attach(manager, scope=scope, staging=staging)
