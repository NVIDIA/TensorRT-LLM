# WideEP Fault-Tolerance MVP End-to-End Prototype

**Status:** Scaffolding (stubs only) | **Branch:** `WideEP-FT/mvp-prototype`

This directory contains the **3–5 day throwaway end-to-end prototype** described
in
[`docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md`](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md).

> The prototype's claim is narrow but high-value: *the MVP integration story
> works, the < 10 s recovery target is achievable in principle, and the seam
> contracts are correct.*
>
> The prototype code is **discarded** once the production PRs land. Every
> component here is stubbed to the absolute minimum needed to exercise the
> integration seam between MVP tracks.

## What's here

| Stub | Production PR | What it stubs | File |
|---|---|---|---|
| **1a.1** EPGroupHealth | PR #13302 | **Real implementation, not stubbed.** Cherry-picked into this branch. | `tensorrt_llm/_torch/modules/fused_moe/ep_group_health.py` |
| **1a.2 + 1a.3** kernel mask + binding | PR #13404 | **Deferred** — see [`kernel/README.md`](kernel/README.md) for two integration paths. | (deferred) |
| **1a.4** AlltoAllWatchdog | (PR 1a.4) | Python timer thread, 100 ms poll of host-visible completion-flag table; 5 s timeout; calls `EPGroupHealth.mark_failed` directly. | `stubs/alltoall_watchdog.py` |
| **1a.7** NCCL FT wrapper | (PR 1a.7) | Sets `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`; one Python thread polling `ncclCommGetAsyncError`. | `stubs/nccl_async_error_monitor.py` |
| **1b.1-3** EPLB `reconfigure_mask_only` | (PRs 1b.1-3) | Zeros dead-rank slot in 1–2 layers; called directly from iteration-boundary hook. | `stubs/eplb_slot_remap.py` |
| **1c.3** MPI FT subcomm | (PR 1c.3) | Global Python state + one `Isend`/`Irecv` pair on a dedicated thread; no `MPI_Comm_split`. | `stubs/mpi_ft_subcomm.py` |
| **1c.4** Model engine health-check hook | (PR 1c.4) | `if health.generation != cached: reconfigure_mask_only()` at top of every iteration. | `stubs/iteration_boundary_hook.py` |
| **1d.0** MPI signal-handler replacement | PR #14160 | **Real implementation, not stubbed.** Cherry-picked into this branch. Activated via `TLLM_FAULT_TOLERANCE_MODE=1`. | `cpp/tensorrt_llm/runtime/utils/mpiUtils.cpp` |
| **1d.4** Fault-injection harness | (PR 1d.4) | `os.kill(rank_pid, SIGKILL)` from a Python test driver + per-event JSON timeline. | `scripts/kill_and_survive_driver.py` + `scripts/kill_and_survive_worker.py` |

## What this is NOT

- Not production code. Not reviewable as MVP PRs. Throwaway scaffolding.
- Not a coverage substitute for the real MVP PRs (no telemetry, no
  feature-flag gating, no error handling for degenerate cases).
- Not a substitute for [Audit
  1b](../../docs/design/wide-ep-fault-tolerance/09-risks-and-open-questions.md):
  72-rank scaling tail, NVSwitch fabric manager behavior, IMEX dynamic re-grant,
  and `kMaxRanks` register-pressure are all out of scope here.

## How to run

### Prerequisites

- 4 or 8 NVLink-connected GPUs on a single node (DGX/HGX B200, B300, H100; or
  GB200/GB300 NVL72 tray with IMEX configured per
  [§3 of the prototype plan](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#3-hardware)).
- `mpirun` available.
- `TLLM_FAULT_TOLERANCE_MODE=1` env var set so PR 1d.0's signal-handler
  replacement activates (otherwise the SIGKILL'd rank's signal handler aborts
  the survivors and the test fails immediately).
- `--mca orte_enable_recovery 1` passed to `mpirun` (audit 1a Day 2 finding;
  without it `mpirun` terminates survivors on any abnormal child exit).

### Single-rank kill-and-survive run

```bash
cd /path/to/TensorRT-LLM
export TLLM_FAULT_TOLERANCE_MODE=1
python prototypes/wide_ep_ft_mvp/scripts/kill_and_survive_driver.py \
    --np 4 \
    --victim-rank 2 \
    --kill-after-iter 3 \
    --output prototypes/wide_ep_ft_mvp/results/kill-iter3.json
```

The driver prints a per-event timeline:

```
t_kill                            : 0.000 s   (SIGKILL issued to rank 2)
t_watchdog_fires                  : 1.247 s   (first survivor's watchdog called mark_failed(2))
t_mark_failed_propagated          : 1.281 s   (all survivors' generation reflects the failure)
t_iteration_boundary              : 1.293 s   (next iteration boundary reached)
t_reconfigure_done                : 1.305 s   (reconfigure_mask_only returned on all survivors)
t_first_new_request_completed     : 2.118 s   (first request completed at N-1 ranks)
─────────────────────────────────────────────
< 10 s recovery budget            : ✓ PASS (2.118 s)
```

The full per-event JSON is saved to `--output` for ingestion as the regression
baseline by PR 1d.4's eventual fault-injection harness.

### Seam-stressing variants

Per
[§4.3 of the plan](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#43-sequence),
exercise the same kill at four different points in the iteration; all should
converge to the same final state and timing budget:

```bash
# During dispatch (kernel actively spinning on dead peer's completion_flags)
... --kill-during dispatch

# During combine
... --kill-during combine

# During routing (between dispatch and combine)
... --kill-during routing

# During EPLB worker stride (off-iteration cleanup)
... --kill-during eplb-stride
```

## Exit criteria (from [§7](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#7-exit-criteria))

1. Single-rank kill survives. Wall-clock from kill to first new request
   completed at N-1 is < 10 s.
2. All four seam-stressing kill points converge to the same final state.
3. Per-event timeline JSON is logged for every run.
4. Findings written up alongside `audit-1a-findings.md`.

## Open questions the prototype is scoped to answer

From
[§8 of the plan](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#8-open-questions-for-the-prototype-to-answer):

- Watchdog vs NCCL collective ordering. Does `TORCH_NCCL_ASYNC_ERROR_HANDLING`
  fire before the AlltoAll watchdog, or vice versa?
- Iteration-boundary semantics. Where exactly does the model engine check
  `EPGroupHealth.generation`?
- Three-part 1d.0 fix interaction. Does
  `_exit(N) + --mca orte_enable_recovery 1 + MPI_ERRORS_RETURN` actually keep
  survivors alive under a real MoE workload?
- Detection latency reality check. Is the 5 s default watchdog timeout right?

## After the prototype

Per
[§9 of the plan](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#9-after-the-prototype):

- Discard this entire `prototypes/wide_ep_ft_mvp/` directory.
- Each production PR design ingests the prototype findings; interface contract
  changes go into PR descriptions before the PRs are opened.
- The per-event timeline JSON gets pulled into PR 1d.4's fault-injection
  harness as the reference baseline.
