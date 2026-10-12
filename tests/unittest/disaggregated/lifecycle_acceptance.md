<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV transfer lifecycle acceptance

This test-only qualification suite builds on the lifecycle implementation merged in
[PR #19674](https://github.com/NVIDIA/TensorRT-LLM/pull/19674). It complements the
implementation's focused regressions and adds real GPU/NIXL coverage without changing
production behavior; it does not establish a GPU/RDMA fencing guarantee.
Coverage is grouped into three milestones:

- **A — Ownership:** protect resources until every accessor has settled.
- **B — Bounded retirement:** preserve logical outcomes, settle late completion,
  and fail closed with coordinated containment when quiescence cannot be proven.
- **C — Extensions:** isolate attempts/incarnations, qualify more configurations,
  and support replay-safe coordinated cancellation.

Keep the requirement IDs below stable as implementations land. Pending rows are
acceptance requirements, not promises that those features are already supported.

## Contract and configuration boundary

Logical cancellation, failure, or timeout must not authorize physical reuse.
Every accessor must settle with appropriate completion evidence, or the owning
execution domain must be safely fenced before reuse. A later physical completion
must not revive an already committed logical outcome.

The safety contract is general; implementation and qualification are incremental.
Milestones A and B initially target this **opt-in activation profile**, not all
Python-transceiver configurations:

- Python transceiver over NIXL, using the **C++ NIXL agent binding**, not the C++
  transceiver or pure-Python agent.
- Disaggregated `KVCacheManagerV2`, NVFP4, `SELFKONLY` (FP4-MLA layout).
- Asynchronous monolithic generation-first transfers, attention-DP, PP1/CP1,
  transfer overlap enabled, and one immutable selected CTX ADP cohort.
- No bounce buffer, pipelining, layerwise transfer, synchronous requests, or retry.
- A positive finite `kv_transfer_timeout_ms` and globally unique nonnegative
  integer disaggregated request IDs.
- Both `TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE=1` and
  `TRTLLM_DISAGG_NO_RETRY=1` on every participant, with matching binaries/config.
- An initialized MPI executor world with `MPI_THREAD_MULTIPLE`; non-MPI launchers
  and mixed-version peers are not qualified by the containment implementation.

The code gates are `_validate_fp4_mla_bridge_profile` and
`KvCacheTransceiverV2._validate_bridge_req` in
[transceiver.py](../../../tensorrt_llm/_torch/disaggregation/transceiver.py).
Matching participants and global ID uniqueness remain operator obligations;
the validators do not negotiate or prove them. Selecting the profile is not
evidence that a deployment has passed end-to-end qualification.

Milestone C extends identity, configuration coverage, and coordinated cancellation.
Each additional configuration/adapter needs its own qualification. Neither B nor
the label "C" implies blanket Python coverage; C++ support is a separate extension.

## Runnable cancellation/reuse scenario

[test_transfer_lifecycle_acceptance.py](test_transfer_lifecycle_acceptance.py)
joins real `RxSession`, `TaskHandle`, `KvCacheTransceiverV2` status polling, and
`DisaggTransferCoordinator`. The bridge profile and per-request validators run.
The transport, request registry, executor cleanup effects, and one-slot KV/aux
allocation pools are controlled CPU substitutes. This is **component acceptance**,
not a real `PyExecutor`, KV allocator, distributed cohort, or GPU/NIXL test.

`test_cancelled_receive_retains_destinations_until_all_accessors_settle` covers
four cases: user cancellation or elapsed request timeout, with KV or auxiliary
completion last. Each case requires:

1. Publish a receive with pending KV and auxiliary writers, then cancel it.
   Timeout cases exercise the coordinator's elapsed-time check before and after
   the threshold, then use the same executor-effect cancellation seam. Verify
   the cancellation notification reaches the published CTX endpoint once; later
   polls must not notify that successfully contacted endpoint again.
2. Repeatedly poll actual receive status and attempt replacement allocation;
   neither destination may be released or reused while either writer is pending.
3. Complete both writers; the logical outcome stays cancelled and both
   allocations are released exactly once.
4. Allocate a new request in those slots, replay duplicate completion reports,
   and check unchanged sentinels and release counts. This checks bookkeeping
   after notification replay, not real DMA against reused GPU addresses.

`test_default_receive_cancellation_retains_pending_destinations` repeats the
pending-writer reuse check with the bridge disabled (user and timeout cases).
These are **strict expected failures**, limited to the dedicated premature-reuse
assertion. Other exceptions fail normally, and an unexpected pass fails CI so the
expectation must be revisited. An xfail records an open safety gap, not acceptance.
The default-off baseline is not an additional qualified profile.

The four bridge-enabled cases are ordinary regression tests and must pass. The
two expected failures run the known-bug reproducers without making that specific
failure a pytest job failure; they must not count toward completed acceptance.
Retaining them in CI is a maintainer decision. A fix must remove the corresponding
`xfail` marker and pass the same test normally. `--runxfail` is a diagnostic mode
that deliberately makes the unfixed cases fail, not the normal CI command.

## Roadmap coverage ledger

File aliases below link to the focused suites, rather than duplicating their tests:

- **Acceptance**: [test_transfer_lifecycle_acceptance.py](test_transfer_lifecycle_acceptance.py)
- **GPU qualification**: [test_transfer_lifecycle_gpu.py](test_transfer_lifecycle_gpu.py)
- **Ownership**: [test_transfer_ownership_regressions.py](test_transfer_ownership_regressions.py)
- **Outcome**: [test_task_handle.py](test_task_handle.py)
- **Late settlement**: [test_transfer_late_settlement.py](test_transfer_late_settlement.py)
- **Deadline**: [test_retirement_deadline.py](test_retirement_deadline.py)
- **Containment**: [test_kv_transfer_fail_stop.py](test_kv_transfer_fail_stop.py)
- **Session ACK**: [test_unsubmitted_session.py](test_unsubmitted_session.py)

The implementation status is a dated snapshot. Component evidence below means
an executable assertion exists, not that native CI or deployment qualification
has passed. Track those results separately for each exact revision/configuration.

| ID | Roadmap requirement | Component evidence | Remaining acceptance |
| --- | --- | --- | --- |
| A-PROFILE | Activate only the initial profile | Ownership: `test_fp4_mla_bridge_uses_production_cache_layout`, `test_fp4_mla_bridge_rejects_requests_outside_qualified_protocol` | Activation and matching configuration on every real participant |
| A-ADMISSION | Cancel before publication; retain partial publication | Ownership: `test_pre_cancelled_rx_session_never_publishes_destination`, `test_aborted_publication_cannot_complete_during_failure_unwind`, `test_fp4_mla_bridge_roots_send_and_receive_requests_before_admission` | Actual executor/allocator lifetime and publication-race qualification |
| A-CANCEL-REUSE | Repeated cancellation/status polling cannot release active destinations | Acceptance: `test_cancelled_receive_retains_destinations_until_all_accessors_settle` (4 cases); default-off companion has 2 strict xfails | Real allocator reuse under delayed remote writes and actual executor cancellation/timeout cleanup |
| A-SOURCE | Exact backend completion protects the source | Ownership: `test_sender_physical_operation_rejects_unproven_retirement`, `test_sender_retires_only_after_backend_done`, `test_ambiguous_sender_result_retains_source_and_reports_in_doubt` | Backend conformance and real source-page retention |
| A-COHORT | All writers and independent holds must settle | Ownership: `test_gen_first_no_retry_adp_count_seal_waits_for_one_writer_group`, `test_gen_first_aux_owner_uses_anonymous_no_retry_writer_count`, `test_failed_receive_consensus_waits_for_every_rank_to_drain` | Multi-rank cohort, local CUDA completion, and manager-hold composition |
| B-OUTCOME | Stable event-time logical outcomes | Outcome: `test_receive_outcome_does_not_depend_on_polling_before_late_completion`, `test_receive_cancel_keeps_its_outcome_when_a_writer_later_fails`, `test_sender_cancel_survives_late_kv_and_aux_exceptions` | Same semantics through native executor/backend paths |
| B-LATE-DONE | Exact retained-handle settlement, idempotence, no revival/reopening | Merged [PR #19378](https://github.com/NVIDIA/TensorRT-LLM/pull/19378); Late settlement: `test_late_settlement_requires_positive_retained_status`, `test_concurrent_late_done_retires_once_without_changing_failure`, `test_late_done_does_not_retire_active_sibling` | Qualify retained-handle completion and actual source/destination reuse with the real backend |
| B-DEADLINE | Non-resettable quiescence clock, independent progress, sticky fatal expiry | Merged [PR #19673](https://github.com/NVIDIA/TensorRT-LLM/pull/19673); Deadline: `test_first_drain_trigger_cannot_be_extended`, `test_blocked_backend_cannot_delay_fatal_deadline`, `test_receiver_deadline_retains_kv_aux_and_registration` | Real outstanding GPU/NIXL access through deadline expiry; coordinator request timeout alone does not test the drain clock |
| B-CONTAINMENT | Rank-aligned fail-close and qualified fencing/replacement | Merged foundation [PR #19674](https://github.com/NVIDIA/TensorRT-LLM/pull/19674); Containment: `test_real_mpi_retirement_kills_blocked_world_before_fresh_world_starts`; Session ACK: `test_session_ack_cannot_bypass_existing_unsettled_access`, `test_missing_candidate_still_expires_and_late_ack_cannot_reverse_fatal` | GPU/RDMA access revocation, surviving-peer behavior, qualified fencing before reuse, and replacement recovery; MPI process death is not this proof |
| C-IDENTITY | Attempt/incarnation isolation | Pending; no acceptance test here | Old evidence cannot settle new attempt memory, including retries/restarts |
| C-COVERAGE | Qualify additional configurations individually | Pending; no acceptance test here | Repeat full contract for each named profile; bounce/layerwise probes alone do not qualify those modes |
| C-CANCELLATION | Replay-safe coordinated in-flight cancellation | Pending; no acceptance test here | Exact participant-cancellation scenario, partial completion, B's bounded containment, and C-IDENTITY where replay/retry applies |

B-DEADLINE and B-CONTAINMENT form a paired delivery boundary. Deadline expiry
bounds the decision to fail closed, not actual fencing/restart duration. The
implementation and its CPU/MPI regressions do not close the remaining backend/E2E
qualification. Do not represent those gaps with skipped acceptance tests.

## Running and recording evidence

In a TensorRT-LLM test environment with its normal dependencies installed:

```bash
python -m pytest tests/unittest/disaggregated/test_transfer_lifecycle_acceptance.py -m cpu_only -q -rXx
python -m pytest tests/unittest/disaggregated/test_transfer_lifecycle_acceptance.py -k default_receive --runxfail -q
```

The first command should report 4 passes and 2 expected failures at this baseline.
The second intentionally exposes the two default-path failures. Existing CPU CI
already selects `unittest/disaggregated` via
[l0_cpu.yml](../../integration/test_lists/test-db/l0_cpu.yml); the new file contains
the required `pytest.mark.cpu_only` marker. GPU qualification is a separate test
target; a CPU-stage pass does not execute it.

For each ledger row, record the commit, exact configuration, command/stage, result,
and artifact link separately for component tests, backend conformance, and E2E.
Native CI execution is still required; source-only runs with dependency stubs are
diagnostic evidence, not native CI qualification.

### Recorded native CI evidence

These are exact-revision results, not validation of later harness changes. Both
CPU stages use the listed component substitutes; the GPU stage uses the opt-in
four-B200, CTX2/GEN2, C++ NIXL profile described below.

| Revision / run | Suite and stage | Observed result | Native evidence |
| --- | --- | --- | --- |
| `2886217a8aec42236dffb2d4bd195a1a2d27ba59` / [#62874](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/62874/) | Acceptance, `CPU-Generic-x86-1-cbts` and `CPU-Generic-arm-1-cbts` | 4 passed + 2 strict XFAILs on each architecture | [x86 log][cpu-x86-evidence], [ARM log][cpu-arm-evidence] |
| Same revision / #62874 | Startup and exit-observer helpers, same CPU stages | 20 passed + 1 setup error on each architecture: startup `capsys` conflicted with the autouse `capfd` fixture | [x86 log][cpu-x86-evidence], [ARM log][cpu-arm-evidence] |
| Same revision / #62874 | GPU qualification | Not executed: the CPU failure blocked multi-GPU dispatch; no GPU pass is claimed | [Dispatch log][gpu-dispatch-evidence] |
| `a16bc402a7302d02edb9ec9eb84985b5cd123a14` / [#62774](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/62774/) | GPU qualification, `DGX_B200-4_GPUs-PyTorch-1-cbts` | 6 passed; `timeout_late_aux` failed during four-rank startup, before transfer assertions | [B200 log][gpu-evidence] |

The CPU setup conflict is corrected by using `capfd`; later startup changes need
fresh native execution. The [PR verification section](https://github.com/NVIDIA/TensorRT-LLM/pull/19663)
tracks subsequent runs. Neither the expected failures nor the partial historical
GPU result completes deployment acceptance.

[cpu-x86-evidence]: https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-1/blue/rest/organizations/jenkins/pipelines/LLM/pipelines/main/pipelines/L0_Test-x86_64-Single-GPU/runs/8651/nodes/172/log/?start=0
[cpu-arm-evidence]: https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-1/blue/rest/organizations/jenkins/pipelines/LLM/pipelines/main/pipelines/L0_Test-SBSA-Single-GPU/runs/8411/nodes/112/log/?start=0
[gpu-dispatch-evidence]: https://prod.blsm.nvidia.com/sw-tensorrt-top-1/blue/rest/organizations/jenkins/pipelines/LLM/pipelines/main/pipelines/L0_MergeRequest_PR/runs/62874/nodes/1418/log/?start=0
[gpu-evidence]: https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-3/blue/rest/organizations/jenkins/pipelines/LLM/pipelines/main/pipelines/L0_Test-x86_64-Multi-GPU/runs/2755/nodes/111/log/?start=0

## GPU/NIXL qualification and its evidence boundary

[test_transfer_lifecycle_gpu.py](test_transfer_lifecycle_gpu.py) exercises the
bridge with real GPU allocations, the C++ NIXL agent, and separate MPI CTX/GEN
worlds. Each world has an active attention-DP rank and an idle peer. The tests
use generic `KVCacheManagerV2` NVFP4/`SELFKONLY` allocations with explicit key and
block-scale coverage and real KV/AUX transfers, without loading a model or
introducing production fault-injection hooks. This deliberately excludes the
unfinished dense FP4 MLA manager/serving integration: its high-precision tail,
V-scale/packed-V roles, and replicated-role mapping are not qualified here.
A minimally initialized
`PyExecutor` exercises its real cancellation gate and termination methods through
the production coordinator and resource manager; no scheduler/serving loop or
asynchronous-send manager is initialized.

Before publication, a real native filler cache pins the allocator's measured
spare slots; the nominal token quota is not assumed to equal physical capacity.
The protected slot must stay unavailable until retirement and then be reused
with unchanged sentinels after duplicate reports. Readiness gates establish
native DONE and allocation pressure before cancellation or timeout; the real
120-second request/grace clocks are never reset by the fixture.

The fault control masks completion evidence **after the real backend reports
DONE**. This deterministically tests the software's behavior while completion is
unproven to the owner: retain resources, preserve the logical outcome, retire
idempotently when the exact handle becomes visible as DONE, or fail closed when
the grace period expires. It does not simulate a still-active DMA, nor qualify
abort/release semantics. A fatal test must establish its fault preconditions and
retention assertions before process death; an arbitrary MPI failure is not a pass.
The fatal-world exit check observes both recorded PID/create-time identities
before cleanup, with a bounded wait and state diagnostics. Absent, exited/zombie,
or reused identities mean the original process no longer executes; a live or
unobservable rank fails. This process observation is not GPU/RDMA fencing proof.
The [exit-observer tests](test_transfer_lifecycle_process_exit.py) exercise all
nine scenarios on each rank. The [startup tests](test_transfer_lifecycle_startup.py)
check that both workers and the supervisor share one 120-second startup deadline,
reject readiness observed at or after expiry, and leave relative transfer waits
unchanged; startup failures print bounded worker logs and missing readiness
markers before cleanup.

The seven cases cover normal delivery, cancellation before publication,
cancellation with KV or AUX completing late, timeout with AUX completing late,
and fatal expiry on the source or destination. Fatal cases deliberately give the
other endpoint a longer timeout to observe the affected endpoint's containment
without simultaneously losing its peer. This is fault isolation, not a claim to
qualify arbitrary mixed configurations. The suite is included in the four-GPU
B200 pre-merge list, [l0_dgx_b200.yml](../../integration/test_lists/test-db/l0_dgx_b200.yml).

```bash
python -m pytest tests/unittest/disaggregated/test_transfer_lifecycle_gpu.py -q
```

Record results at three distinct levels:

1. **CPU component acceptance:** deterministic cancellation/reuse checks and
   existing focused ownership, outcome, deadline, and containment regressions.
2. **Real-backend software qualification:** GPU/NIXL data movement, actual
   allocation retention/release, and cross-world MPI containment with controlled
   completion evidence, including production executor cancellation/cleanup
   methods. This is model-free module integration, not a fully initialized
   serving executor or platform-fencing certificate.
3. **Deployment acceptance:** actual executor cleanup, backend/platform proof
   that old accesses cannot reach reused memory (including surviving peers), and
   a successful request on the replacement after that proof. No restart or reuse
   is authorized merely by observing process exit.

Level 3 remains the Milestone B exit gate. A green result at either earlier level
does not close that gate or qualify any additional Milestone C configuration.
