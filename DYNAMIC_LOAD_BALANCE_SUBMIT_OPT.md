<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamic load balance: configuration and validation

The MegaMoE path uses HALO-Q route planning and in-switch TMA weight copies with
one CPU submitter, an ordinary MAIN stream, and a higher-priority COPY stream.

## Configuration

```yaml
moe_config:
  backend: MEGAMOE_CUTEDSL
  rebalance:
    enabled: true
    helper_slots_per_rank: 4
```

Set `enabled: false` for OFF. Helper slots are fixed for a model instance;
changing their count requires a separately tuned MegaMoE cache. For example,
48 resident experts plus four helpers is M=52/S=4; three helpers is M=51/S=3.
Use a separate per-rank `TLLM_AUTOTUNER_CACHE_PATH` for each configuration.

Rebuild TensorRT-LLM's native operators: the FP4/FP8 quantizers add SM-budget overloads,
so deploying these Python sources onto an older wheel is insufficient.

ON sets `CUDA_SCALE_LAUNCH_QUEUES=4x` before configuration validators and device
probes, and propagates it through MPI/direct/Ray workers. OFF and zero-helper
configurations leave the environment unchanged. This process-wide setting is
not reset by constructing an OFF model after ON in the same process. If PyTorch
already initialized CUDA without 4x, startup rejects the configuration. That
guard cannot detect every external CUDA context; applications that initialize
CUDA elsewhere first must export `CUDA_SCALE_LAUNCH_QUEUES=4x` before startup.

Shared FC1/FC2 use a static eight-SM launch budget by default. To select CLC:

```bash
export TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER=clc_dynamic
```

This process-wide override applies to supported Rubin dense GEMMs. CLC does not
use the static cluster budget or guarantee eight idle SMs. Unset it or select
`static` to restore ON's default. Static and CLC tuning identities are distinct.
Shared-input MXFP8 and routed-input NVFP4 quantization retain `reserved_sms=8`
in either mode: 816 CTAs on a 212-SM device, versus OFF's 848. These are launch
budgets, not hardware partitions; concurrent memory traffic can still add cost.

## Execution and lifetime

1. MAIN produces route IDs. COPY waits for that input event, then HALO-Q receives
   the original contiguous CUDA int32 `[tokens, topk]` tensor via `submit(ids,
   stream_handle)`. There is no route bitwise-OR, tail fill, or staging kernel.
   `record_stream` protects allocator reuse; the producer must not overwrite IDs.
2. COPY records plan-ready after HALO-Q and then enqueues the TMA weight copy.
   MAIN can enqueue shared experts, quantization, and independent input staging
   before waiting for plan-ready immediately before its first route read.
3. MegaMoE consumes physical routes and observes helper-weight READY publication.
   After enqueuing the consumer, MAIN records completion; the generation lease
   orders the next producer after consumption. Events and driver handles are reused.
4. One submitter owns MAIN and COPY; all layers share COPY. There is no green
   context or steady-state host wait for HALO-Q. Cold warmup-to-worker ownership
   transfer drains bound work once and preserves stream and generation state.

ON tactic tuning requires `MEGAMOE_TACTIC_AUTOTUNE=1`. Its default workload has
equal per-rank token totals and within-rank power-law alpha 0.8, increasing by
expert ID. `MEGAMOE_AUTOTUNE_PL_ALPHA` overrides alpha; leave it unset for the
unchanged uniform OFF workload. ON evaluates MixCGA and `expert`/`token_tile`
READY candidates with the actual HALO/TMA producer and helper-slot ABI. This
synthetic tuning input does not replace the inference dataset.

Upstream pins and file hashes are in the [scheduler manifest](tensorrt_llm/_torch/cute_dsl_kernels/megamoe_scheduler_v2/VENDOR_MANIFEST.json)
and [MegaMoE manifest](tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/VENDOR_MANIFEST.json).
The latter records the FC1 claim-exhaustion fix and the TensorRT-LLM main
compatibility patch. Vendor refresh replays both patches and verifies each file
hash; the compatibility patch requires Ruff 0.9.4 formatting before application.

The default combine format remains main's `bf16`. Set
`MEGAMOE_COMBINE_FORMAT` explicitly to select another supported format, and use
the same value across ranks and benchmark configurations.

## Validation

Run the [portable contracts and targeted GPU tests](scripts/verification/dynamic_load_balance/README.md).
CPU contracts cover submission order, ownership, queue propagation, tactic
selection and packaging. They do not establish GPU numerical correctness.
Validate the native quantizer overloads and shared GEMM on the new source-built
wheel, then run EP8 helper-slot accuracy and same-node unprofiled OFF/ON E2E.
The ON accuracy check must use the tactics selected by its own E2E autotune
cache. Earlier tekit checks and measurements below are historical references,
not validation of this main-branch port.

## Pre-rebase E2E reference

One unprofiled run per configuration at tekit `5cf076ea80c0ba49b8293b5207ab1d916e4b1173`:
DeepSeek V4 NVFP4, eight Rubin GPUs, TP8/EP8 with attention DP, max 8,192 tokens,
batch size 128, CUDA graphs disabled, overlap scheduler enabled. Each completed
2,048 formal requests; startup and separate warmup are excluded.

| Configuration | Wall time (s) | Input + output tokens/s | Throughput vs OFF |
| --- | ---: | ---: | ---: |
| OFF | 290.225462 | 275,373.3715 | baseline |
| ON, four helper slots | 277.793245 | 287,697.2907 | +4.4753% |
| ON, three helper slots | 276.927916 | 288,596.2719 | +4.8018% |

These runs used the same nodes, model, dataset, and source-matching quantization
wrapper overlay, not a full rebuilt wheel. They predate the target rebase and
optional CLC selection. One run does not establish variance or a stable ranking
of helper counts. Completion/protocol checks passed on all eight ranks, but do
not prove accuracy. Earlier slots4 cached-winner MLP checks were elementwise
identical in 136 comparisons; they cover one 8,192-token bucket and one layer,
not slots3, every tactic, or full-model quality. Full raw evidence is retained
with the MR's external validation artifacts.
