<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native State Audit: Resident Capture and V2 Compact Extension

Source audit date: 2026-11-02. See [State and host contract](STATE_AND_HOST_AUDIT.md) for the admission decision and host binding.

## Scope and Result

Inspected immutable TensorRT-LLM base **B = `4fd6d1cc4521502a90f08df491af17ebac8e8fb1`** and native V2 sleep candidate #21221 **C = `b8b0ef930808b8e02ba4c99912b8e8f44d0c1145`**. Candidate worktree HEAD matches C and its tracked/untracked status was clean during inspection. Baseline reads used `git show B:path`; candidate reads used the clean C worktree. Links below identify those exact objects, not moving branches.

**Result: concrete native capabilities exist, but neither pinned tree supplies an end-to-end process-capture/restore/validate/activate contract.** This is a source finding, not a failed restore experiment. Resident capture does not require candidate V2 sleep. Compact capture does require its ownership, ordering, and reconstruction preconditions, and is not qualified merely because native sleep tests exist.

The selected starting profile is a fixed, complete MPI cohort on one node, no live user requests, no external KV exposure, immutable local model, and isolated serving admission. One GPU and multiple GPUs remain separate qualification cases. "No external transfer" does not remove local HOST/offload copies or MPI/NCCL dependencies.

The experimental standalone host driver belongs to [restore qualification PR #21481](https://github.com/NVIDIA/TensorRT-LLM/pull/21481). It reuses pinned Snapshot executor operations without running the Kubernetes controller. This native audit does not declare that adapter supported or imply upstream agreement.

Follow-up labels identify the WARP design work items: **S2** = restore qualification; **S3** = native lifecycle/admission/validation; **S5** = KV/resource policy and reconstruction; **S6** = communication/access retirement/containment. These are proposed engineering owners, not upstream API commitments. Tests proposed below remain **NOT RUN** unless a separate qualification record states otherwise.

## Findings

### F01. The Executor Has a Useful Scoped Barrier, Not a Capture Transaction

**Existing owner/capability.** `PyExecutor._handle_control_request` at B:5989-6030 drains `active_requests`, `waiting_queue`, and the connector-pending predicate, synchronizes CUDA, retires the overlap batch, then parks the loop. `control_action(drain=True)` at B:6523-6547 exposes that interval as a context manager and releases it in `finally`. The connector predicate at B:2665-2676 covers connector loads and the asynchronous transfer manager; it is not an inventory of every native registration, lender hold, or external provider. [Control barrier][b-control], [context lifetime][b-control-context], [connector predicate][b-connector-pending].

**Gap and policy.** For resident capture, retain model/KV/graph backing and reuse the step-boundary machinery, but add a capture operation whose closed-admission and quiescence state remain valid through the host snapshot boundary. A queue-empty observation made before entering that interval is insufficient. The inspected API does not return a cross-subsystem capture token, freeze all ingress/exposure paths, or bind evidence to host operation/session IDs. Compact policy must perform native release while this protection is held, then re-establish evidence after any operation that can enqueue work.

**Proposed owner/test.** S3 with S2 binding. Add a race test submitting an ordinary request after drain but before capture acknowledgement, and an overlap-input-copy case. It must remain rejected/parked while host capture is pending; injected action failure must not accidentally activate a restored candidate. Inventory output completion/callbacks separately; the shown drain expression alone does not prove them all retired.

### F02. Wakeup Reopens Admission Before Any Separate Restore Validation

**Existing owner/capability.** `ExecutorRequestQueue.begin_sleep_transition` closes admission under `enqueue_lock`; completion publishes PARKED. `begin_wakeup_transition` enters WAKING, but `complete_wakeup_transition` at B:174-184 returns to RUNNING when the last parked tag is restored. `get_memory_status` returns admission state and parked tags only. [Admission transitions][b-admission].

**Gap and policy.** Native wakeup completion is not authority to expose a restored server. Both resident and compact profiles need a separate restore gate that remains closed while rank compatibility, reconstructed resources, private generation, private-work retirement, and final host success are checked. A restored RUNNING flag or `/memory_status` cannot be inherited as activation authority. Ordinary wakeup is not a ready-made `validate_then_activate` operation.

**Proposed owner/test.** S3, coordinated with S2. Test last-tag wakeup under a still-closed restore gate; stale host notification; validation failure; final host cleanup after an early restore notification; coordinator loss; duplicate activation. Private validation needs an explicitly authorized path that cannot reopen general request admission.

### F03. MPI PREPARE/COMMIT Exists, but Its Latch Is Sleep-Operation Scoped

**Existing owner/capability.** `BaseWorker._multi_rank_sleep_wakeup` at B:605-637 performs rank-0 control parking, peer PREPARE, mutation/COMMIT, and bounded acknowledgements. MNNVL-selected operations commit peers before entering subgroup collectives. Pre-mutation errors can abort; post-mutation errors fail-stop the cohort. The peer listener retains the scheduler barrier during PREPARE, synchronizes CUDA, and releases its barrier as the operation ends. [Worker protocol][b-worker], [peer listener][b-listener].

**Gap and policy.** This is a strong starting primitive for a fixed cohort, not an implemented host capture protocol. Its success says that a sleep/wakeup operation completed, not that a persistent capture latch covers every rank, HOST worker, socket, graph and provider until the host consumes it. Resident capture needs a no-release prepare mode; compact capture additionally needs the C resource operations. Neither should fabricate persistent host evidence from a successful MPI ACK.

**Proposed owner/test.** S3/S6 with S2. Test missing PREPARE ACK, stale operation IDs, host capture during held PREPARE, failed COMMIT after one rank mutated, and incomplete restored cohorts. Preserve the existing distinction between recoverable pre-mutation abort and fail-stop after divergence.

### F04. V2 GPU Pool Reservations, Backing and Logical Pages Are Different Resources

**Existing owner/capability.** `GpuSlotPool` owns a `mVirtMem` reservation; resize changes mapped physical size and slot addresses are reservation base plus slot stride (B and C identical at `storage/core.cpp`:332-371). Baseline `PyExecutor.validate_sleep_tags` explicitly rejects V2 `KV_CACHE` sleep at B:10734-10739. C adds native `KvCacheManager::releaseGpuMemory`/`materializeGpuMemory` at C:456-539 and `StorageManager::releaseMemory`/`materializeMemory` at C:1747-1839. [Pool addresses][b-pools], [baseline rejection][b-reset-and-sleep], [candidate manager][c-native-sleep], [candidate storage][c-storage-sleep].

**Concrete behavior.** C suspends surviving caches, retains metadata needed for restoration, saves pool slot capacities, evicts preserved pages toward colder tiers, checks that released pools have no occupied slots, synchronizes CUDA before unmapping, and clears the GPU physical allocator. GPU tiers are released in both modes; HOST tiers are released only in discard mode; DISK tiers are not selected by this release loop. Restoration expands HOST before GPU and supports retry of partially restored capacity.

**Gap and policy.** Resident policy at B retains reservations, backing and reachable metadata; lack of V2 sleep is not a resident-capture blocker. Compact policy at C preserves reservations/metadata but releases selected backing. Same virtual base does not establish the same physical handles, page identities, slot assignments or graph-visible index contents after process restore. Report GPU mapped bytes, HOST committed/usable bytes and reserved virtual bytes separately; no byte count was measured here.

**Proposed owner/test.** S5. Record pool group, tier, medium, reservation base/length, mapped bytes, occupied slots, and live owners before capture and after restoration. Extend native cold-page tests to process restore, retry after partial allocation failure, and graph replay. Reject an unknown pool owner rather than assuming every allocation belongs to the primary KV manager.

### F05. HOST Capacity Can Be Published Before Its Backing Is Usable

**Existing owner/capability.** `HostSlotPool` constructs `HostMem(..., waitForFill=false)` at B:378-389. `numSlots()` uses configured size, while `waitSlotUsable()` waits for the required byte range; `PoolGroupBase::allocate`/`allocateMultiple` wait before returning slots (B:531-560). `HostMem` starts background fill during construction. Growth updates `mSize` before asynchronous fill; shrinking first joins fill, propagates errors, then decommits inside the same reservation. These files are byte-identical at B and C. [HOST pool readiness][b-host-pool], [allocation fence][b-slot-allocation], [HOST fill/grow/shrink][b-hostmem].

**Gap and policy.** A warm server, positive HOST quota, or CUDA synchronization is not proof that all deferred CPU backing/registration work is complete. C's materialization explicitly relies on slot-level waits rather than waiting for the whole HOST tier. Resident capture must either prove fill/registration workers safe for capture and join or freeze them through an owner API, or explicitly exclude a configured HOST tier from the first accepted profile. Compact discard has a synchronous shrink path, but restored growth again has deferred readiness. Do not replace slot waits with a global assumption that materialization made every byte usable.

**Proposed owner/test.** S5 plus S2 host compatibility. Add capture-ready evidence for target size, usable watermark, fill failure and worker lifecycle. Test capture while deferred fill is active, restore with delayed fill, early allocation at the high end of a pool, and fill failure propagation. Existing native address/readiness tests are useful references, not restore proof: [HOST tests][c-host-tests].

### F06. Disk Pools Are Anonymous FD-Owned State; Staging Is Not Removed by Pool Sleep

**Existing owner/capability.** `DiskSlotPool` opens an `O_TMPFILE` descriptor; fallback `mkstemp` immediately unlinks the name (B:431-454). `StorageManager` separately owns device index staging, pinned HOST page staging, and `CopyEngine` (C:442-446, 527-533); those owners are destroyed by `destroy()`, not by C's pool release loop. [Anonymous disk backing][b-disk], [staging owners][c-staging], [release selection][c-storage-sleep].

**Concrete copy behavior.** `CopyEngine` uses a HOST staging hop for GPU/DISK transfer. HOST/DISK copies are CUDA-stream host callbacks around `pread`/`pwrite`; the callback bodies log I/O errors rather than returning a per-copy success object to the caller. Stream/context completion can order these callbacks, but does not itself turn the logged disk errors into positive data-integrity evidence. These copy files are identical at B and C. [Two-hop copy][b-copy-engine], [host callbacks and I/O errors][b-copy-utils].

**Gap and policy.** It is inaccurate to describe existing disk backing as a named file the adapter can simply reopen. Qualify anonymous/deleted-file descriptor preservation and storage contents with the actual host runtime, or exclude DISK in the starting profile. This is not a claim that CRIU cannot support it. Compact KV release leaves disk capacity and independent staging relevant to artifact footprint; it cannot be advertised as releasing all KV-related GPU/HOST bytes. Resident capture must retire the stream callbacks and include every descriptor/staging owner.

**Proposed owner/test.** S5/S2; capture-policy default: DISK excluded until qualified, retained staging measured explicitly. Test unlinked-file survival after source termination, disk read/write failure, callback completion before capture, and staging lifetime across restore. Include integrity checks in any admitted disk profile; successful synchronization alone is insufficient.

### F07. Batch Publication Has Two Different Copy Boundaries

**Existing owner/capability.** Native `Batch` owns host staging rows, upload stream, device staging, mirror/converted indices, sparse lease mappings and member references. B `publish()` asynchronously uploads staging, records `mPublished`, orders mirror/index work on the Batch stream, and records staging-consumed completion. `waitPublished()` waits for the host-upload event, not a blanket completion of every subsequent consumer (B:1478-1547). [Batch publication][b-batch-publish].

**Candidate extension.** C `_suspendForSleep` calls `waitPublished()` **and** synchronizes the Batch stream, preserves row owners/lease permutation, and suspends caches without ordinary removal. `_resumeAfterSleep` restores base-page-index/scratch-row bindings, republishes, waits for upload and synchronizes the stream (C:463-536). [Sleep-aware Batch][c-batch-sleep].

**Gap and policy.** For either profile, host row mutation must wait before the mutation, not merely immediately before the next `publish()`. Resident capture retains rows and graph-visible buffer addresses after all relevant streams are settled. Compact capture must use sleep-aware suspension; ordinary remove/reset is not an equivalent cleanup because it can discard sparse state. The C sequence is available native ordering, not process-restore evidence.

**Proposed owner/test.** S5. Test a pending upload while capture starts, direct cache resize/commit touching staged rows, multiple Batch objects, and restored padded graph replay. Observe host upload and GPU conversion completion separately. Existing candidate native Batch tests: [Python integration][c-integration-tests], [native-backed Batch test][c-batch-tests].

### F08. Sparse Arena Preservation Moves Bytes to HOST, Not Out of the Artifact

**Existing owner/capability.** `SparseCellArena` owns a single reservation containing cell bytes, shared metadata and per-lease metadata. C `releaseMemory(false)` synchronizes CUDA and copies all `mNumBytes` into `mSleepContents` before unmapping GPU backing. Restore remaps the reservation and either copies that saved vector or initializes metadata for discarded contents; the vector is released afterward (C:137-229). [Sparse arena ownership and sleep][c-sparse].

**Gap and policy.** Preserving dirty sparse cells is not the same as flushing them into ordinary cold KV storage or deleting them. CPU/PINNED compact mode can trade GPU artifact bytes for an additional HOST vector outside the ordinary HOST tier quota. Discard mode is only valid when retained internal caches meet discard preconditions. Resident capture retains the arena and current lease/index state. Neither permits arbitrary reset of sparse lease metadata under captured graph references.

**Proposed owner/test.** S5. Account for `mSleepContents` separately from HOST pool bytes; test dirty/unflushed cells, lease permutation, and multi-Batch restoration before replay. Compare total artifact bytes and reconstruction cost, not just released GPU capacity. Existing sleep-cycle sparse coverage is a reference only: [sparse test location][c-sparse-tests].

### F09. Reuse Reset Is Not Dummy-Safe Capture Cleanup and Does Not Erase Every Sidecar

**Existing owner/capability.** B/C native `clearReusableBlocks()` calls `_checkNoLivingKvCaches`, which requires the entire live-cache set to be empty, with no stats-excluded/dummy exemption (B:410-453). B Python `reset_reuse_state` clears native reusable blocks, invokes an optional post-clear callback, clears conversation state, calls lender `_on_reset`, and **flushes** streaming events in `finally` (B:8397-8414). `PyExecutor.reset_prefix_cache` coordinates route-cache clearing and optional draft reset (B:10717-10732). [Native reset precondition][b-native-reset], [wrapper reset][b-reuse-reset], [route/draft reset][b-reset-and-sleep].

**Concrete consequence.** A drained scheduler can still own native graph-padding caches, so ordinary reuse reset is not guaranteed to succeed after warmup. The lender `_on_reset` sets `_reuse_was_reset=True`, stopping subsequent by-name lending; it does not release existing registration holds, drain all granted publishes, or establish a new compatible namespace. [Lender reset][b-lender-reset]. C discard sleep is a different path: it deliberately permits only stats-excluded, zero-committed surviving caches and clears native radix state within its own release ordering. [Candidate discard guard][c-wrapper-sleep], [candidate native ordering][c-native-sleep].

**Gap and policy.** Correct the broad wording "clears sharing metadata and streaming events": the actual behavior is lender invalidation plus event flushing, not blanket deletion/retirement. Resident policy may retain valid warmed reuse under immutable weights, or choose a separately qualified reset policy; it must not blindly invoke reset after graph warmup. Compact policy must use the candidate's surviving-internal-cache semantics and keep route/draft/conversation sidecars consistent if reuse is intentionally discarded.

**Proposed owner/test.** S5/S3; S6 for an attached lender extension. Test prefix reset with an extant padding dummy, no stale route/prefix hit after discard, draft-cache consistency, and flush failure. State whether reuse is retained or discarded in the capture manifest. A reset is a correctness policy, not evidence of reduced backing bytes.

### F10. Graph Padding Is Live Ownership Even When User Request Count Is Zero

**Existing owner/capability.** `CudaGraphRunner.preallocate_padding_dummies` runs after graph capture. `_padding_dummy_managers` enumerates main KV, draft KV, speculative resources and cross-KV when configured; `release_padding_dummy` frees through all owners and drops runner entries for later recreation (B:1370-1455). [Dummy lifecycle][b-graph-dummies]. The V2 wrapper creates dummy requests without prefix reuse/commits and requires caller cleanup across managers on failure/lifetime end (C:6867 onward). [V2 dummy allocation][c-dummies].

**Gap and policy.** "No user request" must not be encoded as "no native KvCache objects." Resident capture should preserve the required dummy/resource ownership and graph allocations. Compact C can keep logical internal caches while reconstructing backing. Freeing only primary-manager dummies leaks ownership in other managers; freeing every dummy is not proof that already captured graph references remain valid. Recreating dummies is permissible only under an explicitly tested policy, not an assumed substitute for replay validation.

**Proposed owner/test.** S5 and model/graph owners. Record dummy IDs, owning managers, graph shape/draft variant, pointer-bearing buffers and stats-exclusion status. Test padded batches after source termination/restoration without recapture, and all configured draft/cross-KV/speculative variants. The initial experiment selects dense text-only, non-speculative Qwen3 to bound this surface. Optional model families remain in the audit and require separate qualification.

### F11. Compact Wakeup Reconstructs Pages and Slot Indices; Stable Pool Base Is Insufficient

**Existing owner/capability.** C `KvCache::releaseMemory` changes released page references into `UnmaterializedPage{cacheLevel}` after requiring SUSPENDED and zero committed tokens. `materializeMemory` allocates fresh slots/pages, initializes colder tiers before hot allocation across caches, and publishes new holders only after allocation/eviction registration succeeds (C:996-1107). [Page reconstruction][c-pages]. The Python manager refreshes non-Batch page-index buffers, guard-page mappings and sparse metadata, then synchronizes its stream (C:8466-8494). [Wrapper reconstruction][c-wrapper-sleep].

**Model-specific example.** `MambaCacheManagerV2.materialize_memory` at C:1345-1366 explicitly compares old and new SSM slots, invokes `_on_state_slots_relocated`, synchronizes and updates the mapping, because padding requests can bypass normal preparation. [SSM remap][c-mamba]. This is direct evidence that native sleep does not promise unchanged slot identity.

**Gap and policy.** Resident capture keeps existing ownership but still needs restored-address validation. Compact capture must retain logical metadata and refresh every derived index/alias before graph replay. Physical page-handle equality is the wrong compact acceptance test; correct logical ownership, valid references and rebuilt indices are required. A blanket V2 sleep claim must not implicitly admit Mamba, MLA, index/value or model-specific auxiliary buffers.

**Proposed owner/test.** S5 with model cache owners. Force slot permutations during sleep/wake, then test padded generation, sparse metadata, guard pages and SSM index refresh without recapture. Record unsupported model combinations as rejected. Candidate integration tests already include a MiniMax M3 index/value rejection, not generic support: [integration guards][c-integration-tests].

### F12. Native and Tagged Auxiliary Resources Have Deliberate Release/Restore Ordering

**Existing owner/capability.** C `PyExecutor._kv_memory_resources` deduplicates every V2 manager in the resource manager. Its `release_memory` runs before tagged auxiliary buffers are released, while `materialize_memory` runs after them. `virtual_memory.release_with_tag(resource=...)` calls the native resource first; `materialize_with_tag` restores tagged buffers, synchronizes CUDA, then calls the native owner. `preserve_contents()` converts metadata in a discard scope into CPU-preserved tagged allocations while retaining the parent tag (C:114-179). [All-manager enumeration][c-executor-sleep], [tagged ordering and metadata][c-vmm].

**Gap and policy.** Compact reconstruction cannot be implemented by calling a primary pool's release method in isolation. The host driver must bind to a native lifecycle that enforces these dependencies and inventories auxiliary copies. Resident policy retains those objects and does not need to execute destructive tagged sleep. Keep weights/transformed tensors and graph state resident for the first latency experiment unless a separate policy is explicitly admitted; clearing KV is not evidence that weights avoid load/transform on restore.

**Proposed owner/test.** S5/S3. Instrument call order for multiple unique managers and duplicate aliases; inject tagged restore failure and confirm native consumers do not run; measure auxiliary HOST backups; check no model reload/transform or graph recapture in process-restore qualification. Pointer/digest/counter evidence for weights remains uncollected here.

### F13. Graph Pointer Checks and MNNVL Hooks Are Partial, Useful Primitives

**Existing owner/capability.** `CudaGraphRunner.allow_capture` defaults to disabled outside explicit capture because shared workspace resizing can invalidate baked addresses. `_snapshot_graph_tensor_ptrs` visits top-level CUDA tensor attributes, excludes temporary attributes, and the corresponding assertion validates those recorded pointers, including temporarily swapped draft attrs (B:787-812, 1078-1125). It is not a recursive inventory of all graph resources or an unconditional restore validator. [Graph checks][b-graph-checks].

`PyExecutor._mnnvl_checkpoint_resources` finds `CheckpointableCommunication` objects only through selected engine modules' `.comm`, deduplicates keys, and `_run_mnnvl_checkpoint_resources` calls process-local prepare/restore while parked (B:4225-4277). [MNNVL hook scope][b-mnnvl].

**Gap and policy.** Retain graph pool/workspace/metadata addresses and replay the actual admitted shapes privately. Do not count eager fallback or silent graph recapture as preserved warm-state success. MPI/NCCL and MNNVL remain configured communication resources in a no-external-KV-transfer profile. MNNVL hooks do not qualify all NCCL communicators, external KV registration, changed topology or process-restored peer connectivity.

**Proposed owner/test.** S3/S5 for graph inventory; S6/S2 for communication restoration. Test unchanged graph capture count, required graph hits, recorded and nested aliases, rank mapping and configured collective operations. Reject unqualified communication/model combinations or add explicit owner evidence; do not assume a hook name is a full capture contract.

### F14. A No-Exposure Profile Must Be Proven by Construction, Not Queue Emptiness

**Existing owner/capability.** `KvCacheTransceiverV2` constructs `TransferWorker` during initialization (B:325-365). The worker creates its transport agent, registers KV/AUX memory, creates sender/receiver and exports descriptors in `_setup_transfer_engine`, before a scheduler request needs to exist (B:4104-4145). [Transceiver construction][b-transceiver-init], [registration setup][b-transfer-setup]. `has_inflight_transfer(req)` and `has_retired_send_session(req)` are request-local checks, not registration/provider inventories. [Per-request checks][b-session-checks].

**Candidate guard.** C rejects V2 KV sleep whenever a transceiver exists or `_kv_connector_moves_kv` is true; the V2 wrapper rejects any attached lender, not just a lender with currently busy leases. These guards cite missing re-registration/lender sleep lifecycles. [Executor exclusions][c-executor-sleep], [lender exclusion][c-wrapper-sleep].

**Gap and policy.** For the selected resident no-exposure profile, a sufficient conservative admission rule is no constructed KV transceiver, no KV-moving connector, no attached externally lending backend and no other plugin-published memory on every rank. Record object/config evidence. Merely observing zero transfers after warming a transfer-capable instance does not establish the selected profile. If registration existed, choose a separate exact transport/backing extension and prove retirement; do not silently treat it as no exposure. Native MPI coordination alone is not an external KV transfer path.

**Proposed owner/test.** S3/S6 capture-policy guard, S5 for the candidate checks. Negative tests: an empty-request transceiver with registrations; a lender attached but idle; requestless incoming work; connector loads completed but retained provider state. The resident/no-exposure work has no blanket prerequisite on implementing cancellation for every external transport.

### F15. PACT Has Per-Attempt and Per-Instance Evidence, but No Resumable Cohort Capture API

Here "PACT" refers to the existing backend/access-retirement/lender contracts inspected below, not a claim that a public class named PACT exists.

**Existing owner/capability.** `Attempt.quiesced()` independently proves physical access has ended; logical outcome and quiescence can occur in either order. `Attempt.cancel()` is advisory and capability-gated. `Backend.shutdown(timeout) -> frozenset` already aggregates **one backend instance's** remaining unquiesced/registered memory; empty means none can still be accessed. Arbitrary submission exceptions can require retaining the lease until that empty result. Multiple instances each own their own registrations, and shutdown must obey the backend's serialized call sequence. [Backend and attempt contract][b-backend]. It would be incorrect to say there is no aggregation API at all.

`RetirementWatchdog` supplies registered ownership/deadline tracking, sticky `request_shutdown`, `has_unclosed_owners`, guarded `require_retired()` and `stop()`. Its boolean snapshot expressly does not close admission or authorize teardown, and its shutdown state is not a temporary pause/resume token. Native `TransferWorker` only creates this watchdog for the configured physical-ownership/fatal-callback path; its `request_shutdown` does nothing to a nonexistent watchdog. [Watchdog API][b-retirement], [worker gating][b-transfer-gating].

**Actual missing contract.** In these inspected APIs there is no reusable operation that atomically closes new exposure, enumerates all backend instances plus lender/provider/local-copy owners, returns typed stable capture evidence across ranks, holds that evidence while the host captures, then restores/re-registers and resumes under a fresh attempt. Terminal backend shutdown is not such a pause/resume API. The gap is cross-owner aggregation and restartable capture lifecycle, not absence of individual access-end evidence.

**Gap and policy.** No-exposure resident capture excludes this transfer-bearing extension and does not wait for universal PACT cancellation. A future transfer-bearing resident or compact profile must choose a precise backend/transport/backing combination, reuse the existing ownership/deadline evidence, and add only the missing adapter/aggregation lifecycle. It must preserve existing deadlines and unresolved roots, not reset a retirement budget when the host retries.

**Proposed owner/test.** S6, consumed by S3/S2. Test two backend instances where one returns nonempty shutdown evidence; outcome complete but `quiesced=False`; unsupported cancellation; unexpected submission exception; late incoming exposure; fatal retained owners; expired deadline during capture; and attempted resume after terminal shutdown. No such tests were run in this audit.

### F16. Lender Leases, Registration Holds and Native Teardown Must Not Be Collapsed

**Existing owner/capability.** `Staging` tracks open leases, independent region holds and quarantined slots at B:439-444; `hold_regions()` at B:457-464 creates a hold. `_on_shutdown` waits only recorded local copies and retains memory when a lease/hold/quarantine/stop reason remains; it logs and retains on errors. A staging lease's `release()` is distinct from `_StagingHold.release()`. `_on_caches_closed` releases the index-buffer keepalive only after successful runtime shutdown. [Lender shutdown][b-lender], [lease and hold APIs][b-lender-release].

**Native teardown limitation.** Native `TransferWorker.shutdown()` is not the formal `Backend.shutdown` interface. After guarded session retirement it sets `_shutdown`, pops registered descriptors, logs `deregister_memory` exceptions, logs agent shutdown exceptions, and clears `_agent` (B:4194-4239). Thus a normal return is not structured affirmative evidence that all registrations were removed, and repeated calls can short-circuit. [Native teardown][b-native-shutdown]. This is a capture-contract limitation; this audit does not claim an observed DMA fault.

**Gap and policy.** For no-exposure resident capture, keep these paths absent. For a future external-transfer profile, a released lease does not imply a released region hold, deregistration, or remote fencing. Do not use the native worker's `None` return as equivalent to the formal backend's empty evidence set. Compact C currently rejects attached lenders, so idle-attached-lender compact support must not be inferred from natural drain.

**Proposed owner/test.** S6 and lender owner, S5 integration. Add evidence-producing teardown or a separately qualified capture lifecycle retaining failed registrations and owners. Inject deregistration failure, agent shutdown failure, an open region hold with no leases, an unrecorded copy and a quarantined slot. Verify fail-closed capture and no premature backing reuse. Tests should preserve existing shutdown-safety behavior rather than bypass it for checkpointing.

### F17. MPI Abort Is Not Isolated by an Executor Communicator

**Existing owner/capability.** B `_fail_unproven_kv_transfer` explicitly retains unresolved resources and delegates to `propagate_hard_kill`; its source warns that the executor communicator is not an isolated failure domain. `propagate_hard_kill` uses communicator `Abort` only when MPI is initialized, not finalized, and thread support is `THREAD_MULTIPLE`; otherwise it self-SIGKILLs and relies on launcher propagation. The source identifies launcher differences, including `srun --kill-on-bad-exit`. Distributed sleep divergence uses the same hard-kill path. [Transfer fatal path][b-fatal], [hard-kill implementation][b-hard-kill], [sleep divergence][b-sleep-fatal].

**External primary-source corroboration.** Open MPI's v5.0.x `MPI_Abort` documentation says its implementation terminates the job, rather than limiting termination to the supplied communicator. This was checked as a documentation/source contract, not against an installed runtime. [Open MPI documentation](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Abort.3.html).

**Gap and policy.** A coordinator outside the checkpointed worker tree can still be inside the same launcher failure domain. The selected resident profile needs the coordinator/host supervisor outside the worker MPI job's effective termination scope. Split communicators do not prove isolation from co-launched executors. This also applies without external KV transfers because sleep divergence/hang handling can use the shared fatal path. Killing a job is not independently proof of remote access fencing or reusable device resources.

**Proposed owner/test.** S6/S2. Pin MPI implementation/version, launcher argv/environment, process tree and job boundaries. Inject a rank failure and verify actual affected processes plus retained/cleaned resources before replacement. Single-rank, full-cohort and co-launched split-world containment are distinct cases. No abort, process kill, MPI launch or allocation was executed here.

## Available Barriers Versus Missing Capture Semantics

| Source primitive | What it actually establishes | What it does not establish |
| --- | --- | --- |
| `control_action(drain=True)` | Scoped scheduler park, queue/connector drain, CUDA sync, overlap retirement | Durable host-bound latch, all producer/provider inventory, restored admission authority |
| Sleep PREPARE/COMMIT | Rank coordination for a selected memory operation; divergence policy | Full process capture/host completion/private validation protocol |
| `Batch.waitPublished()` | Prior staging upload has stopped reading host rows | All subsequent mirror conversion/readers complete by itself |
| C Batch sleep plus stream sync | Retained row/lease ownership with local copy ordering | Host runtime reconstruction of every graph/resource |
| `HostMem.waitUsable(end)` | Specific range is ready or an error is raised | Entire tier ready; arbitrary CPU worker globally capture-safe |
| C storage CUDA sync | Issued codec work complete before unmapping | CPU HOST fill stopped; disk data integrity; remote provider access ended |
| `Attempt.quiesced()` | Evidence-based end of one submission's memory access | Backend registration teardown or multi-instance closure |
| `Backend.shutdown()` empty set | Terminal per-instance absence of accessible/unretired registered memory | Resume, cross-owner/cohort capture latch, native worker equivalence |
| Watchdog `require_retired/stop` | No retained tracked owners before terminal teardown | Untracked transports covered, all registrations gone, reopenable capture pause |
| Native MNNVL hooks | Process-local preparation/restoration of selected communication objects | Generic MPI/NCCL/KV reconnect or supported process restore |

## Capture Policy and Ownership Handoff

| Resource family | Resident no-exposure starting policy | Compact extension at C | Proposed owning item |
| --- | --- | --- | --- |
| Weights/transforms, graph pools/workspaces | Retain; validate addresses/digests and no reload/recapture | Retain unless independently admitted | S3/S5, model/graph owners |
| V2 logical caches, radix/route/conversation metadata | Retain compatible state or explicitly qualify reset; do not reset through live dummies | Discard selected data with candidate internal-cache guards; refresh sidecars | S5 |
| GPU KV backing | Retain; B sleep rejection is irrelevant to this policy | Release selected pools, restore backing/slots/indices in owner order | S5 |
| HOST tier | Retain only with fill/registration capture evidence, or exclude initially | Discard mode shrinks; preserve mode retains cold data; restored fill remains asynchronous | S5/S2 |
| DISK and pinned/device staging | Exclude DISK until anonymous-FD qualification; inventory retained staging | Pool release does not remove all staging/disk resources | S5/S2 |
| Batch/sparse/SSM/dummy ownership | Retain valid graph-visible owner state; constrain optional model features | Preserve required metadata, reconstruct backing and all relocated indices | S5 |
| Admission/scheduler/private validation | Closed through host capture, restored validation and final activation | Same, plus resource mutation ordering | S3/S2 |
| MPI/NCCL/configured MNNVL | Fixed complete cohort; verify restored communication and failure domain | Same; sleep hooks are only partial native assistance | S6/S2 |
| External KV/lender/provider registrations | Prove absence, not just zero traffic | Candidate rejects transceiver, KV-moving connector and attached lender | S6/S3, S5 guards |

## Verification Evidence and Limitations

Executed: immutable `git show` source reads; candidate HEAD/status checks; bounded `rg` symbol/test discovery; direct inspection of the cited owner functions and transitions; `git diff --numstat B C -- <selected paths>` confirming identical pool core, HOST fill, copy engine/utils, backend, retirement, native transfer, lender and graph-runner files where B/C shared behavior is stated. Candidate modifications were inspected at C; a whole-tree B-to-C diff was not treated as the PR's change set because these are independently pinned revisions.

The source-reference check covered **49 immutable references across 30 pinned files**, with no undefined references or out-of-range line anchors. The text is ASCII-only and has no trailing whitespace. These are source and document checks, not runtime qualification.

Existing tests located, **not run**: candidate `test_sleep_warns_when_host_capacity_below_gpu`, `test_sleep_rejects_attached_lender_before_mutation`, `test_wakeup_zeroes_pools_for_memset_and_fp4_mla`, `test_wakeup_keeps_native_batch_bindings`, `test_minimax_m3_rejects_sleep_with_index_value_buffers`; native-backed `test_sleep_preserves_native_batch_state` and sparse sleep-cycle coverage; C++ HOST address/readiness and cold-page discard/rematerialization cases. [Integration tests][c-integration-tests], [Batch tests][c-batch-tests], [sparse tests][c-sparse-tests], [HOST tests][c-host-tests], [cold-page tests][c-cold-tests]. Their presence is evidence of test intent only, not passing results.

This source audit does not report a successful process restore, MPI failure injection or timing/footprint measurement. Every process-restore outcome remains **UNTESTED** until an independently pinned qualification report supplies evidence. External-transfer compact profiles remain **BLOCKED by the specific missing lifecycles** above. Link targets are immutable source locators checked against local Git objects.

## Immutable Source References

[b-control]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L5968-L6037
[b-control-context]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L6523-L6550
[b-connector-pending]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L2665-L2676
[b-admission]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py#L133-L205
[b-worker]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/base_worker.py#L605-L637
[b-listener]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L4140-L4221
[b-pools]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.cpp#L332-L371
[b-reset-and-sleep]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L10717-L10739
[c-native-sleep]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.cpp#L456-L539
[c-storage-sleep]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp#L1747-L1839
[b-host-pool]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.cpp#L378-L424
[b-slot-allocation]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.cpp#L531-L560
[b-hostmem]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.cpp#L369-L500
[b-disk]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.cpp#L431-L454
[c-staging]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp#L442-L553
[b-copy-engine]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/copyEngine.cpp#L83-L158
[b-copy-utils]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/copyUtils.cpp#L53-L160
[b-batch-publish]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/batch.cpp#L1478-L1547
[c-batch-sleep]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/batch.cpp#L463-L536
[c-sparse]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/sparseCache.cpp#L137-L229
[b-native-reset]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.cpp#L410-L453
[b-reuse-reset]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py#L8397-L8414
[b-lender-reset]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_lender.py#L672-L680
[c-wrapper-sleep]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py#L8401-L8494
[b-graph-dummies]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py#L1370-L1455
[c-dummies]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py#L6867-L6985
[c-pages]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp#L996-L1107
[c-mamba]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/mamba_cache_manager_v2.py#L1345-L1366
[c-executor-sleep]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/pyexecutor/py_executor.py#L2522-L2582
[c-vmm]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/virtual_memory.py#L114-L179
[b-graph-checks]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py#L1078-L1125
[b-mnnvl]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L4225-L4277
[b-transceiver-init]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/transceiver.py#L325-L365
[b-transfer-setup]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/native/transfer.py#L4104-L4145
[b-session-checks]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/transceiver.py#L1107-L1114
[b-backend]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/base/backend.py#L328-L449
[b-retirement]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/lifecycle/retirement.py#L270-L371
[b-transfer-gating]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/native/transfer.py#L4020-L4058
[b-lender]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_lender.py#L672-L733
[b-lender-release]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_lender.py#L1510-L1582
[b-native-shutdown]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/native/transfer.py#L4194-L4239
[b-fatal]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/disaggregation/transceiver.py#L98-L146
[b-hard-kill]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/hang_detector.py#L67-L121
[b-sleep-fatal]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/base_worker.py#L872-L893
[c-integration-tests]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tests/unittest/_torch/pyexecutor/kv_cache/test_kvcm2_integration.py#L3638-L3718
[c-batch-tests]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py#L6298
[c-sparse-tests]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py#L4063
[c-host-tests]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tests/unit_tests/batch_manager/kvCacheManagerV2HostMemTest.cpp#L260-L423
[c-cold-tests]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/cpp/tests/unit_tests/batch_manager/kvCacheManagerV2ColdPageTest.cpp#L763-L808
