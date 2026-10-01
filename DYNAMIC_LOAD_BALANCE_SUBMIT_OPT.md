<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamic load balance: configuration and validation

The MegaMoE path uses HALO-Q route planning by default and in-switch TMA
weight copies with one CPU submitter, an ordinary MAIN stream, and a
higher-priority COPY stream.

## Configuration

```yaml
moe_config:
  backend: MEGAMOE_CUTEDSL
  rebalance:
    enabled: true
    helper_slots_per_rank: 4
```

Set `enabled: false` for OFF. Helper slots are fixed for a model instance;
changing their count requires a separately tuned MegaMoE cache. Use a separate
per-rank `TLLM_AUTOTUNER_CACHE_PATH` for each configuration.

Rebuild TensorRT-LLM's native operators: the FP4/FP8 quantizers add SM-budget overloads,
so deploying these Python sources onto an older wheel is insufficient.

The HALO-Q scheduler and in-switch TMA use TensorRT-LLM's standard
native-operator path: CUDA launchers live under
`cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb`, the Torch bindings live
under `cpp/tensorrt_llm/thop/moe/loadBalance`, and both are linked into
`libth_common.so`. The scheduler uses one `trtllm::moe_rebalance_halo_q` op.
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

For supported eager DeepSeek V4 configurations with the MegaMoE CuTe DSL
routed backend and FP8 shared weights, both OFF and ON use the complete
`BlockScaledSwapAbFc12Kernel`. FC1, clamped SwiGLU, and FC2 execute in one
kernel. Checkpoint loading, TP shards, shared-output scaling, and the enclosing
MoE reduction retain their existing contracts. Other backend/precision/graph
configurations keep their existing shared path.

ON derives the shared-kernel and quantization launch budgets from the runtime
device size and the configured copy-stream reservation. These are launch
budgets, not hardware partitions.

The shared forward enqueues input quantization, a small same-stream asynchronous
counter memset, and the complete FC12 kernel. The reset is required before
every invocation; it introduces no host
wait or global grid barrier. Token count is dynamic; an initialized count-table
slice supplies metadata without a per-forward fill/copy or shape-specific compile.
Loaded FP8 weight values are preserved while gate/up rows and scales are
rearranged for FC12. The fused intermediate uses FP32 FC1/activation arithmetic
and K32 MXFP8 quantization, whereas the previous split path rounded FC1 and
activation through BF16 and quantized at K128. Numerical equivalence must
therefore be validated with explicit tolerances rather than asserted bitwise.

## Execution and lifetime

1. MAIN produces route IDs. COPY waits for that input event, then HALO-Q
   receives the original contiguous CUDA int32
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

ON tactic tuning requires `MEGAMOE_TACTIC_AUTOTUNE=1` and uses a deterministic,
rank-balanced helper-bearing workload. The workload distribution can be
overridden explicitly; OFF retains its uniform workload. ON evaluates the
supported READY candidates with the actual HALO/TMA producer and helper-slot
ABI. This synthetic tuning input does not replace the inference dataset.

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
cache. Keep benchmark results in the controlled validation artifacts rather
than in production source documentation.

## Performance validation

Run OFF and ON on the same allocation with independent autotune caches, identical
model and request inputs, profiling disabled, and no sample removal. Report the
raw per-run durations, aggregate request throughput, correctness scope, source
identity, and runtime identity in the associated validation artifact. Performance
results are hardware- and workload-specific and are intentionally not embedded in
production source documentation.
