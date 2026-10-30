<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2: results

This page records what the workflow has been measured to do on real models, how
each number was measured, and what has not been measured yet.

[design.md](design.md) explains the mechanisms these numbers refer to.

## 1. Onboarding speed and reliability

Two from-scratch onboardings of the same checkpoint measure the planned run.
The checkpoint is unreleased and has no in-tree implementation.

| | |
|---|---|
| hardware | GB300 |
| workload | ISL 8192 / OSL 1024 |
| parallel members | `tp1 tp2 tep2 dep2 tp4 tep4 dep4` |
| feature units | core, MTP (depth 1–7), disaggregated serving, multimodal |
| other features | chunked prefill, block reuse, and cache-dtype variants |
| accuracy anchors | measured with the Hugging Face reference |

The two runs used the flow differently:

- **Run 1** used the flow while it was being fixed. The orchestrator was
  restarted on new flow code whenever a defect blocked the run.
- **Run 2** used one pinned flow revision from start to stop.

### 1.1 The same clock, the same window

Both runs are measured from the run's first journal line (`run opened`) over
their first 11.9 hours:

| | run 1 | run 2 |
|---|---|---|
| core boot and accuracy pass at `tp1` | 9 h 38 | **6 h 45** |
| all four units integrated at `tp1` | built width-first; all widths at 18 h 56 | **10 h 51** |
| contract checks passing at 11.9 h | 10 | **84** |
| workers in flight, mean / peak | 2.46 / 6 | **4.97 / 10** (the cap then) |
| GPU busy | 20.7 GPU-h | **73.8 GPU-h** |
| verdicts per check | 2.83 | **1.00** |
| contract FAILs | 7 | **0** |
| orchestrator restarts / flow changes | 6 / 9 | **0 / 0** |

How each row is measured:

- **Workers in flight:** the interval from a turn's claim to its `.done`
  file, sampled every 5 minutes.
- **GPU busy:** summed from the spool receipts' device-seconds.
- **Verdicts per check:** counts how often a check was measured. More than one
  verdict per check means a check ran again on a tree it had already judged.

**Accuracy:** in run 2, the core's accuracy gate passed at all seven members.
MTP acceptance passed at every depth it was gated at.

### 1.2 Where the runs stopped

Neither run reached the correctness milestone (`CORRECTNESS DELIVERED`). The
operator stopped both:

- **Run 1** was stopped after about 46 hours. The delivery tier had not run on
  its last head.
- **Run 2** was stopped at 12 h 42 m, so that its findings could be folded into
  the flow. At that point:
  - 93 of 139 contract checks were passing;
  - the multimodal and disaggregated units' wide members and the delivery tier
    were still running.

### 1.3 What run 2 found, and what now addresses it

| finding | mechanism now in the flow |
|---|---|
| checks measured the integration checkout while integrations moved it | every check measures a read-only snapshot of one commit; legs on different trees make the verdict VOID |
| the disaggregated witness used filler prompts, blind to lost state | prose prompts, plus an ablation leg that must change the output |
| the attention-DP hand-over only ever reached rank 0 | leg B pins one (context, generation) rank pair per prompt |
| block reuse was asked for but never shown to hit | a reuse probe and per-request reuse counts; a unit that asks for reuse fails when nothing was reused |
| the first code task started 2 h 15 m after the run opened | the survey starts at minute 0; queue rows become kernel tasks without a planner turn; the graph critic holds nothing |
| the planner was woken by events it had already answered, and queued behind workers | answered events are dropped, and planner and critic turns never count against the worker cap |
| the tune task held gate GPUs it never used, and measured ground truth on one node | a tune task on its own perf sessions holds no run device, and ground truth is spread over the perf nodes |
| a session requeued by Slurm lost its running jobs | those jobs run again or close out `node_lost` |
| 16 kernel re-certifications late in the run (target ≤ 3) | a phase-0 geometry probe measures the serving geometry (capture lists, KV pool and page view, context chunks, draft depths) before any kernel is certified, and a `verify` whose required cells the entry already covers closes without a smith turn |
| ready work waited hours on serial gates and one regression review per shared change | a member's ready checks share one engine; check-only work is a `check` task with no agent turn; regression reviews batch landed shared changes; kernel tasks on one shared entry land local variants side by side, merged by one promote |

None of these mechanisms has been re-measured on a full run yet; the late
re-certification count in particular is not yet measured.

## 2. Tuning

### 2.1 The unreleased checkpoint (run 1's target)

The table compares 21 headline points (7 members × 3 metrics) against the tune
baseline:

| tree | geomean speed-up | median % of SOL | points ≥ 50% SOL |
|---|---|---|---|
| tune baseline | 1.00× | 26% | 2 / 21 |
| the campaign's best kept tree | 1.84× | 53% | 11 / 21 |
| plus two allreduce changes landed after it | 1.97× | 53% | 13 / 21 |
| plus member-gated variants (measured; not yet landed when the run stopped) | 2.07× | 56% | 14 / 21 |

- **Measurement caveats:**
  - The last two rows were measured on different hosts from the first two.
  - SOL is a hand-reduced estimate.
- **Where the gain came from:** fusions, kernel sourcing, and one-shot and
  MNNVL allreduce.
- **Accuracy:** the variant tree passed the accuracy gate and MTP acceptance
  at all seven members.
- **A cost the keep rule did not see:** TTFT got worse at two high-concurrency
  points. TTFT is not part of the keep rule (see section 4).

### 2.2 Qwen3.8-Flash-Next-NVFP4

Setup:

- 1× GB300, ISL 8192 / OSL 1024, CUDA graphs on, MTP off.
- The target started at 0.83× the in-tree model at concurrency 1, both on
  their default kernels.
- The tune stage then ran for about 28 hours: 30 candidates, 19 kept, 2
  delivered as kernel entries.
- Every keep passed:
  - the target's test suite;
  - the modeling_v2 contract tests;
  - gsm8k (anchor 96.36; kept scores 95.91–96.66).

A concurrency sweep afterwards ran all three variants on each of 3 hosts
(output tok/s/GPU, `max_batch_size` 128):

| concurrency | in-tree model (default kernels) | target before tune | target after tune | after / in-tree |
|---|---|---|---|---|
| 1 | 119.3 | 99.2 | 272.0 | 2.28× |
| 8 | 712.9 | 546.5 | 1,206.2 | 1.69× |
| 32 | 1,628.5 | 1,345.2 | 2,589.4 | 1.59× |
| 128 | 2,675.3 | 2,243.6 | 4,318.4 | 1.61× |

The in-tree model can also switch to the kernel families the tuned target uses
(`moe_config.backend: TRTLLM` and `TRTLLM_LOW_M_GEMM_BACKEND=auto`). With those
switches the tuned target's lead is 1.11× at concurrency 1 and 1.28× at 128.
That remainder is what the target's own fusions and kernels add.

## 3. Engine defects the runs surfaced

Onboarding exercises engine paths that few models reach. The runs found these
defects outside modeling_v2, each fixed in its own pull request:

- the FLA chunked gated delta rule with tied (zero) log-gates;
- 64-bit conv-state slot offsets in `causalConv1d`;
- thop attention with a bf16 output over an NVFP4 KV cache;
- a routing hook through which a target supplies its input processor;
- disaggregated serving relaying multimodal hashes to generation-only requests;
- MoE custom-op fake registrations matching their schemas;
- the KV-cache estimate for single-KV-head layers under TP.

[LESSONS.md](../LESSONS.md) records what earlier onboardings of public
checkpoints got wrong, and the mechanism that now prevents each failure.

## 4. Not yet measured

- **Correctness end to end on this revision.** The next from-scratch run
  targets:
  - correctness within 16 h;
  - at most 3 late re-certifications;
  - the first code task within 30 min.
- **In-process transport on a full run.** Both runs used the external
  transport. The in-process path has its preflight agent probe and turn
  records, but no full run behind it.
- **TTFT in the keep rule.** The tune stage measures TTFT but does not judge
  candidates on it.
- **Tuning the reference model's own switches.** The ladder never tries engine
  switches for the in-tree reference.
