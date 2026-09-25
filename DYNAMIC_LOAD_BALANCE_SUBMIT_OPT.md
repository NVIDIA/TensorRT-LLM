<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamic load balance: configuration and validation

The MegaMoE path uses HALO-Q route planning by default and in-switch TMA
weight copies with one CPU submitter, an ordinary MAIN stream, and a
higher-priority COPY stream. The same scheduler path can select GAR-N through
legacy mode.

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

The GAR-N/HALO-Q scheduler and in-switch TMA use TensorRT-LLM's standard
native-operator path: CUDA launchers live under
`cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb`, the Torch bindings live
under `cpp/tensorrt_llm/thop/moe/loadBalance`, and both are linked into
`libth_common.so`. The scheduler uses one `trtllm::moe_rebalance_halo_q` op:
`CudaSchedulerConfig.algorithm="legacy"` selects GAR-N, while
`CudaSchedulerConfig.algorithm="halo_q"` selects HALO-Q and remains the default.
Python calls `torch.ops.trtllm` directly; the production
path does not compile these kernels at runtime. The scheduler op has a fake
registration for `torch.compile`. TMA keeps its stateful create/bind/submit/
destroy lifecycle eager and submits to the current CUDA stream. The legacy TMA
loader remains only for the diagnostic host-plan fallback.

ON sets `CUDA_SCALE_LAUNCH_QUEUES=4x` before configuration validators and device
probes, and propagates it through MPI/direct/Ray workers. OFF and zero-helper
configurations leave the environment unchanged. This process-wide setting is
not reset by constructing an OFF model after ON in the same process. If PyTorch
already initialized CUDA without 4x, startup rejects the configuration. That
guard cannot detect every external CUDA context; applications that initialize
CUDA elsewhere first must export `CUDA_SCALE_LAUNCH_QUEUES=4x` before startup.

For eager DeepSeek V4 on Rubin with the MegaMoE CuTe DSL routed backend and
FP8 shared weights, both OFF and ON use the upstream complete
`BlockScaledSwapAbFc12Kernel`. FC1, clamped SwiGLU, and FC2 execute in one
kernel. Checkpoint loading, TP shards, shared-output scaling, and the enclosing
MoE reduction retain their existing contracts. Other backend/precision/graph
configurations keep their existing shared path.

ON leaves eight SMs outside the shared kernel's launch budget: 102 two-CTA
clusters on a 212-SM device, versus OFF's 106. Shared-input MXFP8 and routed-input
NVFP4 quantization retain `reserved_sms=8`: 816 CTAs for sufficient input rows,
versus OFF's 848. These are launch budgets, not hardware partitions.

The shared forward enqueues input quantization, a small same-stream asynchronous
counter memset, and the complete FC12 kernel. The memset clears 272 bytes at
capacity 8192 and is required before every invocation; it introduces no host
wait or global grid barrier. Token count is dynamic; an initialized count-table
slice supplies metadata without a per-forward fill/copy or shape-specific compile.
Loaded FP8 weight values are preserved while gate/up rows and scales are
rearranged for FC12. The fused intermediate uses FP32 FC1/activation arithmetic
and K32 MXFP8 quantization, whereas the previous split path rounded FC1 and
activation through BF16 and quantized at K128. Numerical equivalence must
therefore be validated with explicit tolerances rather than asserted bitwise.

## Execution and lifetime

1. MAIN produces route IDs. COPY waits for that input event, then the selected
   scheduler (HALO-Q by default) receives the original contiguous CUDA int32
   `[tokens, topk]` tensor via `submit(ids,
   stream_handle)`. There is no route bitwise-OR, tail fill, or staging kernel.
   `record_stream` protects allocator reuse; the producer must not overwrite IDs.
2. COPY records plan-ready after the scheduler and then enqueues the TMA
   weight copy.
   MAIN can enqueue shared experts, quantization, and independent input staging
   before waiting for plan-ready immediately before its first route read.
3. MegaMoE consumes physical routes and observes helper-weight READY publication.
   After enqueuing the consumer, MAIN records completion; the generation lease
   orders the next producer after consumption. Events and driver handles are reused.
4. One submitter owns MAIN and COPY; all layers share COPY. There is no green
   context or steady-state host wait for the scheduler. Cold warmup-to-worker
   ownership transfer drains bound work once and preserves stream and generation state.

ON tactic tuning requires `MEGAMOE_TACTIC_AUTOTUNE=1`. Its default workload has
equal per-rank token totals and within-rank power-law alpha 0.8, increasing by
expert ID. `MEGAMOE_AUTOTUNE_PL_ALPHA` overrides alpha; leave it unset for the
unchanged uniform OFF workload. ON evaluates MixCGA and `expert`/`token_tile`
READY candidates with the actual HALO/TMA producer and helper-slot ABI. This
synthetic tuning input does not replace the inference dataset.

Upstream pins and file hashes are in the [scheduler manifest](tensorrt_llm/_torch/cute_dsl_kernels/megamoe_scheduler_v2/VENDOR_MANIFEST.json)
and [MegaMoE manifest](tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/VENDOR_MANIFEST.json).
The scheduler manifest records the TensorRT-LLM native-operator adapter. The
MegaMoE manifest records the FC1 claim-exhaustion fix, TensorRT-LLM main
compatibility, and the complete shared FC12 source. Vendor refresh replays the recorded patches
and verifies each file hash, including the required Ruff 0.9.4 formatting steps.

The default combine format remains main's `bf16`. Set
`MEGAMOE_COMBINE_FORMAT` explicitly to select another supported format, and use
the same value across ranks and benchmark configurations.

## Validation

Run the [portable contracts and targeted GPU tests](scripts/verification/dynamic_load_balance/README.md).
CPU contracts cover submission order, ownership, queue propagation, tactic
selection and packaging. They do not establish GPU numerical correctness.
Validate the native quantizer overloads and real-weight shared FC12 against both
the previous shared implementation and an independent quantized reference.
Check repeated calls, dynamic token shapes, actual launch grids, and the single
FC12 kernel in a GPU trace. Then run EP8 helper-slot accuracy and same-node
unprofiled OFF/ON E2E with the same shared implementation.
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

## Design proposal: SM-partitioned streams (2026-09-23)

**Status: recorded for evaluation; not implemented or GPU-validated.** The
ordinary-stream implementation described above remains the current baseline.

Keep one CPU submitter, but create two Green Context streams backed by disjoint
SM resources: COMPUTE for quantization, shared FC1/FC2 and MegaMoE; high-priority
COPY for the scheduler (HALO-Q by default) and in-switch TMA weight-copy
kernels. The target is `N-8 + 8` SMs, where `N` is queried from the device.
`204 + 8` is the 212-SM example,
not a hard-coded GB200 configuration. Query the actual partitions and verify
cluster compatibility before treating this split as supported. Ordinary stream
priority alone does not partition SMs.

This separates two objectives:

| Objective | Is changing the stream sufficient? |
| --- | --- |
| Restrict execution to a provisioned SM partition | Yes, for kernels launched on the corresponding Green Context stream. |
| Change quantization grid size from 848 to 816 CTAs | No; grid sizing must use the partition's SM count. |

The current FP4/MXFP8 wrappers cache `getMultiProcessorCount()` in thread-local
storage. That helper queries the device, without a stream argument; the launch
code receives the SM count and stream separately. With a cached count of 212,
zero reservation, 512 threads/block and sufficient rows, the grid is still 848
CTAs even if execution is restricted to 204 SMs. This can retain a tail wave.
Changing the stream does not rewrite the cached count or launch geometry.

For resource isolation alone, this approach could remove the added
`reserved_sms` quantizer overloads. If a grid matched to the partition remains
necessary, assess a stream-resource-aware launch calculation separately.
Do not remove the existing overloads before that decision and validation.
Green Context creation can be exposed at the Python integration boundary, but
the installed CUDA/PyTorch versions and stream/context lifetime need validation.

Preserve the existing route/READY dependencies, single submitter and enlarged
launch queues. SM partitioning does not by itself solve host driver stalls or
isolate memory bandwidth. Validate actual SM placement, GEMM cluster occupancy,
quantization grids and correctness before comparing performance. For CUDA
graphs, capture/create nodes with the intended execution context; merely
changing the replay stream is insufficient.

Compare OFF, current ON and partitioned ON on the same hardware and workload:
quantization/shared-expert latency, scheduler/copy overlap, host enqueue cost,
iteration tails and unprofiled E2E throughput. Retune partition-dependent tactics
and keep their caches distinct. No performance improvement is claimed yet.

References: [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)
and [SM resource partitioning API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html).
