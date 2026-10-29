<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The DeepSeek V4 Flash token-parity fixes make **TopK, MHC, and FMHA honor
`FORCE_DETERMINISTIC=1`**. Repeated resident-cache runs were already producing
different tokens, so the resident/offload comparison needed a repeatable baseline.
The implementation is split into a disaggregated sparse-offload commit for
transport and configuration, followed by a general determinism commit for
TopK, MHC, FMHA, kernel-specific controls, tests, and this report. The validation
below was performed before splitting the commits.

All five cumulative feature stages pass exact resident-versus-CPU-offload token
parity. Including the fixed baseline and profiling probes, the successful runs
cover **98 requests and 15,680 generated tokens**, with **36 exact sequential
pairs and 8 exact concurrent pairs**. These results compare matching feature
configurations; they do not establish identical tokens across every configuration.
Block reuse is exercised on the context worker only.
[Final validation report](../sparse-disagg-e2e-artifacts/sequential-20261017/final-report.json).

Tracing repeated resident runs found identical TopK scores with different index
orders. A 50-repeat kernel check reproduced native ordering differences; tied
scores also changed the selected set. The PyTorch implementation had no ordering
or selection mismatches in those checks. Separately, MHC split-K reductions
changed intermediate mixing values by approximately `1e-7` to `3e-7`, enough to
propagate into different generated tokens.

- **TopK:** Native atomic appends changed index ordering and, with tied scores,
  sometimes changed which indices were selected. Deterministic mode forces the
  PyTorch implementation for both prefill and decode, which produced repeatable
  selections and ordering in the tested configuration.
  [Implementation](tensorrt_llm/_torch/modules/top_k.py).
- **MHC:** Atomic split-K reductions introduced small floating-point differences
  that propagated into token divergence. Deterministic mode forces a fixed,
  single-split tactic, gives deterministic execution a separate tuning-cache
  identity, and overrides cached or explicitly supplied tactics. It uses the
  half-fused MMA implementation where supported, with a single-split FMA fallback.
  [Implementation](tensorrt_llm/_torch/modules/mhc/mhc_cuda.py).
- **FMHA:** Deterministic mode fixes the KV reduction to one partition so eager
  execution and CUDA graph capture select the same reduction layout.
  [Implementation](cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h).

Sparse-offload settings are also preserved during checkpoint loading so the
offload run actually uses the CPU hot tier.
[Configuration fix](tensorrt_llm/_torch/model_config.py).

Validation passed: **43 kernel regression tests** and **6/6 exact resident/offload
token matches**, covering 12 requests and 1,920 generated tokens on the full
43-layer DeepSeek V4 Flash model. The comparison used four B200 GPUs, NVFP4 KV
cache, context and generation TP2/EP2 with attention TP, the NIXL Python
transceiver, `FORCE_DETERMINISTIC=1`, and disabled autotuning. Prompt lengths were
129, 8,193, 127, 128, 2,049, and 8,193 tokens, with 160 output tokens per request.

That initial comparison used eager execution with CUDA graphs, overlap scheduling,
chunked prefill, and block reuse disabled. Attention DP had encountered a CUDA
illegal access during startup memory profiling; the subsequent validation below
covers its current status.
The passing run used the production implementation without diagnostic overrides.
Local evidence is available in the
[root-cause report](../sparse-disagg-e2e-artifacts/parity/root-cause-evidence.json)
and the [validation report](../sparse-disagg-e2e-artifacts/refresh-20261017T173731Z/final-report.json).

Sequential validation used the original combined commit `432a1ba93d` plus the
fixes described below, before splitting the changes into separate commits.
The following features were enabled cumulatively. Each stage must match resident
versus offloaded tokens with the same enabled features before the next feature
is enabled.
Cross-configuration comparisons are also recorded; exact TP/DP token identity is
a separate criterion because their low-precision reduction layouts differ.
The same six prompts, 160 output tokens, full checkpoint, NVFP4 format, and
deterministic kernel choices are retained. Runtime profiles and worker statistics
verify that the enabled features execute.

| Stage | Newly enabled feature | Status |
| --- | --- | --- |
| 1 | CUDA graphs, decode batch sizes 1 and 2 | Passed: 6/6 resident/offload and eager matches |
| 2 | Generation attention DP | Passed: 6/6 sequential and 2/2 concurrent resident/offload matches |
| 3 | Chunked prefill, 2,048-token scheduling budget | Passed: 6/6 sequential and 2/2 concurrent matches; schedules identical |
| 4 | Context-worker block reuse, including the repeated 8,193-token prompt | Passed: 6/6 sequential and 2/2 concurrent matches; reuse verified |
| 5 | Overlap scheduler | Passed: 6/6 sequential and 2/2 concurrent matches; all sequential outputs also match stage 4 |

The initial CUDA-graph resident run matched both 8,193-token requests and the
2,049-token request, but differed for the 127-, 128-, and 129-token prompts.
An eager resident/offload control on the same allocation matched all six prior
reference outputs and all six resident/offload pairs. A focused deterministic
TopK eager/graph comparison passed 22 cases.

Full-model tracing located the first difference at generation layer 2's attention
output. Earlier layer outputs, queries, and TopK selections matched. Profiles
showed that graph capture selected a CGA multi-CTA KV reduction despite
`FORCE_DETERMINISTIC=1`; eager short requests selected a single-CTA reduction.
Running eager execution with multiple CUDA streams still matched the reference.
The FMHA selector now forces `MultiCtasKvMode::Disabled` and
`mMaxNumCtasPerSeqKv = 1` in deterministic mode. This prevents its heuristics from
overriding the single-partition choice when graph capture supplies a longer KV
length than the live request.
[Implementation](cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h).

Both new native regressions failed before the fix and pass after it. They check
kernel selection with 32 and 64 query heads and exact eager/graph output equality
at seven live KV lengths from 127 to 8,193 tokens, with a graph captured at 9,216.
[Native tests](cpp/tests/unit_tests/kernels/fmhaDeterministicTest.cpp).
The rebuilt libraries pass a fresh eager resident/offload comparison: **6/6 exact
token matches**, with repeatable outputs for the repeated long prompt in both
modes. **Stage 1 passes:** all six CUDA-graph outputs match the corresponding eager
outputs in both resident and offload modes, and all six resident/offload pairs
match. Each mode's profiled repeat also matches its earlier long-prompt output.
Profiles show three graph launches on each generation rank, using only the
single-partition FMHA kernel in the captured decode steps.
[Diagnostic evidence](../sparse-disagg-e2e-artifacts/sequential-20261017/cuda-graph-root-cause.json).

The fix also changes eager reduction choices, so five of the six eager outputs
differ from the old build. The 129-token prompt remains identical; the other
short prompts diverge late in generation. Subsequent feature comparisons use this
fixed implementation in both runs.
[Reference changes](../sparse-disagg-e2e-artifacts/sequential-20261017/fmha-reference-change.json).

Stage 2 completes attention-DP startup without the earlier CUDA illegal access,
but all six resident outputs differ from stage 1. The first differing output-token
indices are 28, 18, 24, 24, 32, and 18, respectively. Repeated outputs are stable.
Tracing found exact equality in the first layer's input, quantized queries, and
native attention output. The first difference is its output projection: TP2
rounds each partial GEMM result to BF16 before reducing the two results, while DP
computes the full projection locally. Replaying captured tensors reproduces both
paths exactly: 1,476 of 4,096 output elements differ, with a maximum absolute
difference of 0.03125. Graph and eager DP runs produce identical tokens.
[Projection evidence](../sparse-disagg-e2e-artifacts/sequential-20261017/attention-dp-projection.json).

The current E2E criterion compares resident and offloaded execution within each
feature configuration. The TP/DP numerical difference is retained in the report.
Sequential requests ran on DP rank 0. Two concurrent long requests exercised both
ranks; one matched the sequential output and one differed starting at output
token 27. The sequential outputs and concurrent output pair both repeated exactly
across two independent resident runs. The harness records this sensitivity
separately and requires the pair
of concurrent outputs to match between resident and offload modes, allowing the
two identical requests to exchange rank assignments. **Stage 2 passes:** all six
sequential outputs and both concurrent outputs match exactly between resident and
offload modes. Worker statistics confirm active requests on DP ranks 0 and 1 in
both modes, and profiles confirm graph replay on both ranks.
[Attention-DP validation](../sparse-disagg-e2e-artifacts/sequential-20261017/attention-dp-root-cause.json).

The stage 3 resident run verifies chunked execution: 29 context iterations, at
most 2,048 scheduled context tokens per iteration, and 21 iterations with cached
context KV. Five of the six sequential outputs match stage 2; the 2,049-token
prompt differs at exactly one output token, index 28. All six sequential
resident/offload pairs match exactly. The initial concurrent pair differed, and
worker statistics showed different prefill chunk schedules: the second resident
request started with 1,920 tokens, while the corresponding offload request
started with 2,048. The harness now submits the second request after the first
streaming token. This preserves concurrent decode on both DP ranks while keeping
prefill scheduling serial and comparable. **Stage 3 passes on rerun:** all six
sequential pairs and both concurrent outputs match exactly, with identical
resident/offload chunk schedules: 30 context iterations, at most 2,048 tokens per
iteration, and 21 iterations with cached context KV. Both DP ranks execute real
requests, and graph replay remains verified.
[Scheduling evidence](../sparse-disagg-e2e-artifacts/sequential-20261017/chunked-prefill-probe-schedule.json).

Stage 4's first attempt did not exercise reuse: `FORCE_DETERMINISTIC=1`
overrides the requested cache configuration and disables it. The test correctly
failed its runtime feature check despite stable tokens. The rerun uses the
existing attention, MoE, and all-reduce determinism switches plus new
`FORCE_TOPK_KERNEL_DETERMINISTIC=1` and `FORCE_MHC_KERNEL_DETERMINISTIC=1`
switches, with global `FORCE_DETERMINISTIC=0`. This preserves the deterministic
kernel choices while allowing prefix reuse. The global flag's behavior is
unchanged. Stages 4 and 5 use:

```bash
export FORCE_DETERMINISTIC=0
export FORCE_ATTENTION_KERNEL_DETERMINISTIC=1
export FORCE_MOE_KERNEL_DETERMINISTIC=1
export FORCE_ALL_REDUCE_DETERMINISTIC=1
export FORCE_TOPK_KERNEL_DETERMINISTIC=1
export FORCE_MHC_KERNEL_DETERMINISTIC=1
```

Reuse is enabled on the context worker in both modes; generation
receives prompt KV through disaggregation. Sparse generation's explicit
restriction on local prefix reuse remains in place, so this test does not
establish support for generation-worker reuse of CPU-backed prefixes.
The 24 focused TopK/MHC regressions pass with the global and kernel-specific
controls, including tied-score selection and cached split-K tactic overrides.
**Stage 4 passes:** all six sequential pairs and both concurrent outputs match
exactly between resident and offload. Both runs verify actual reuse: 276 reused
blocks and 12 context iterations, down from 30 without reuse, with identical
chunk schedules. Repeated long prompts reuse 8,192 cached tokens and compute one
prompt token; their outputs and the profiled repeat remain identical. Five
sequential outputs match stage 3; the 127-token prompt differs beginning at output
token 24. This cross-configuration difference remains recorded separately.
[Reuse-control evidence](../sparse-disagg-e2e-artifacts/sequential-20261017/block-reuse-controls.json).

**Stage 5 passes:** all six sequential pairs and both concurrent outputs match
exactly between resident and offload with overlap scheduling enabled. All six
sequential outputs also match stage 4 in each mode; the repeated prompt and
profiled repeat remain identical. Both modes report the overlap scheduler,
active requests on both attention-DP ranks, 276 reused blocks, and identical
context schedules. CUDA traces show three graph launches on each generation rank.
No further production change was needed for overlap in this configuration.
[Overlap validation](../sparse-disagg-e2e-artifacts/sequential-20261017/overlap-scheduler-validation.json).

Run artifacts are stored locally in
`../sparse-disagg-e2e-artifacts/sequential-20261017/` relative to this worktree.

**32K sparse-offload capacity validation, October 17–18, 2026 (UTC).**
The full 43-layer DeepSeek V4 Flash model sustained **1,134 simultaneous decode
requests, 567 per attention-DP rank**, with exactly **32,768 input tokens and
160 output tokens per request** on four B200 GPUs. This is the measured capacity
with 32 GiB of CPU sparse-history memory per generation rank,
`free_gpu_memory_fraction=0.9`, and generation `max_batch_size=1024` per rank.
It is not an unrestricted hardware maximum or a throughput benchmark.

The two offload trials at 1,134 both completed and produced identical output
multisets. Offering 1,136, 1,140, or 1,182 requests still peaked at 1,134 active
decodes; all offered requests eventually returned 160 tokens. The clean 1,136
probe released its first batch after 1,134 receives. The exploratory 1,140 and
1,182 probes required recorded manual barrier adjustments. Startup's aggregate
estimate of 591 requests per rank (1,182 total) exceeded the observed usable
capacity; the larger gated batches encountered attention-DP dummy-allocation
pressure.
[Capacity evidence](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/offload-capacity-summary.json).

The prompts are **synthetic and identical**: a repeated astronomy paragraph and
a fixed question, with BOS included in the 32,768-token count. Context prefix
reuse is enabled; generation reuse is disabled, so each decode request receives
its own KV through disaggregation. Ratio-4 sparse layers have 8,192 compressed
history rows at this input length. Their GPU fetch cache holds 1,056 rows with
TopK 512, `gpu_cache_factor=2`, and 128-token pages. The complete history is
therefore larger than the GPU fetch cache and uses the CPU hot tier. Dense pools
and ratio-128 history remain on the GPU.

The topology is context TP2/EP2 and generation TP2/EP2 with attention DP2. Both
runs use NVFP4 KV, NIXL Python transfer, CUDA graphs, overlap scheduling, and
2,048-token chunked prefill. Greedy sampling requests exactly 160 tokens and
ignores EOS. The kernel-specific determinism flags listed above remain enabled,
with global `FORCE_DETERMINISTIC=0` and autotuning disabled.

Exact parity compares output-token **multisets**, allowing identical prompts to
exchange attention-DP rank assignments. A test-only receive-publication barrier
starts matching decode batches, and context `max_batch_size=1` fixes prefill
shapes. The barrier leaves transfers, cache operations, model kernels, and
sampling unchanged. Each passing trial must also sustain its full requested
decode concurrency for all 159 decode iterations; the first output token comes
from context execution.

| Offered requests | Exact resident/offload matches | Active decode requests |
| --- | --- | --- |
| 1, cold prefix | 1/1 | 1 |
| 1, reused prefix | 1/1 | 1 |
| 512 | 512/512 | 512 |
| 1,024 | 1,024/1,024 | 1,024 |
| 1,134, first trial | 1,134/1,134 | 1,134 |
| 1,134, repeat | 1,134/1,134 | 1,134 |

All six comparisons pass: **3,806 matched request pairs and 608,960 exactly
matched output tokens per mode**. Both maximum-size repeats sustain 567 decodes
per rank for all 159 decode iterations with zero paused requests. The repeated
output multisets also match within each mode.

The GPU-resident reference needs different allocation settings to hold the same
batch: `free_gpu_memory_fraction=0.99`, `max_util_for_resume=1.0`,
`fp8_context_mla_kv_len_cap=33024`, and `avg_seq_len=32770`. These change memory
allocation, while model computation, input/output lengths, graph sizes, and
sampling remain matched. The last setting sizes windowed pools for decode
crossing a page boundary. The default typical sequence ends exactly on a page
boundary and models one window page where this workload can need two. Earlier
resident attempts paused requests despite spare full-history blocks, changing
the decode batch and failing the comparison. Those attempts are preserved.
[Pool-sizing evidence](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/resident-pool-sizing-diagnostic.json).

**Ordinary asynchronous high-concurrency parity remains unresolved.** Initial
runs without the barrier showed token differences as their batch schedules
changed. The controlled results establish resident/offload parity for matching
batches, not invariance to arbitrary request arrival or batching. Warm prefix
reuse also makes this a capacity test with repeated prompts rather than a test
of sustained throughput on distinct 32K documents.
[Asynchronous comparison](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/initial-comparison.json).

The larger test exposed one additional **general disaggregation fix**. In
generation-first serving, receive descriptors could expire after a fixed 120
seconds while prefill remained queued, even with a 600-second transfer timeout.
`Sender` now retains them for `max(120 seconds, configured transfer timeout)`.
The regression checks the default lifetime, a shorter configured timeout, and
retention plus eventual expiry for a longer timeout. Validation passed
**90 tests, with one skipped**, across the unsubmitted-session and bounded-polling
suites. This fix concerns transfer metadata lifetime and is not specific to
sparse KV offload.
[Implementation](tensorrt_llm/_torch/disaggregation/native/transfer.py),
[regression tests](tests/unittest/disaggregated/test_unsubmitted_session.py),
[validation evidence](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/regression-validation.json).

Complete results, configuration files, source hashes, failed attempts, and
iteration metrics are linked from the
[32K final report](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/final-report.json)
and [exact comparison](../sparse-disagg-e2e-artifacts/long-context-20261017T231142Z/verified-comparison.json).
