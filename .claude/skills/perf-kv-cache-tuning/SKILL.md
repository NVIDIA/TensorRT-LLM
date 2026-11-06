---
name: perf-kv-cache-tuning
description: >-
  Measure, attribute, and tune the performance of TensorRT-LLM KV-cache
  features: KV cache manager V2 changes, block and partial reuse policies, KV
  cache compression methods, cold-page codecs, mixed-precision KV cache, and
  sparse or evicting attention that changes the KV cache. Use whenever a
  KV-cache change touches the KV-cache data path, its attention metadata or
  executor callbacks, or cache migration, and whenever the user asks whether
  a KV-cache feature is fast enough, why it is slower than a baseline, or how
  to tune it. Covers the production-configuration rule, baseline and
  comparison design, metrics, profiling and attribution, hot-path audits,
  memory checks, measurement traps, and the report format.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# KV-Cache Feature Performance

Performance is a required gate for every change on the KV-cache data path,
not a follow-up. A feature is not ready until it has an end-to-end comparison
against its baseline in the production configuration, with real numbers.

Profiler mechanics live in `docs/source/developer-guide/perf-analysis.md` and
`docs/source/developer-guide/perf-benchmarking.md`. Where they are installed,
the repository skills `perf-nsight-systems`, `perf-host-analysis`,
`perf-host-optimization`, `perf-torch-sync-free`, `perf-torch-cuda-graphs`, and
`perf-regression-cookbook` cover profiling and host-overhead work in depth.
This skill decides what to measure, against which baseline, and how to read
the result for KV-cache features. Read the entry skill for the area first:
`trtllm-kv-cache-manager-v2` (`../trtllm-kv-cache-manager-v2/SKILL.md`) for
KV cache manager V2 changes, or `trtllm-kv-cache-compression`
(`../trtllm-kv-cache-compression/SKILL.md`) for KV cache compression
methods.

## Non-Negotiable Rules

1. **Production configuration only.** CUDA graphs on, overlap scheduler on,
   and the defaults a deployment would use (attention backend, chunked
   prefill, KV cache dtype, batch and token limits). Never report numbers taken
   with `cuda_graph_config: null` or `disable_overlap_scheduler: true` unless
   the deployment's own recipe sets them, as disaggregated context workers
   do; then say so in the report. Never report numbers from a single-process
   debug worker, with synchronizing debug variables, or with debug-level
   logging. If a profiler cannot see inside CUDA graphs, fix the profiler
   (`--cuda-graph-trace node`), never the configuration. Start both arms (the
   two configurations being compared) from the checked-in configuration for
   the model, GPU, sequence lengths, and concurrency in
   `examples/configs/database/` (see `lookup.yaml`) when one exists, and add
   only the feature's keys.
2. **Change one variable.** Hold the checkpoint, workload, sampling, parallel
   topology, KV cache memory budget, GPU model, clocks, and build constant.
   Only the setting under test differs. Diff the resolved configurations the
   two runs print; a confounded comparison cannot be attributed to the
   feature.
3. **Report absolute numbers.** Give throughput, time per output token (TPOT)
   or inter-token latency (ITL), time to first token (TTFT), and memory
   with units. A ratio alone is not a result.
4. **Unprofiled comparison first, profiling second.** A kernel microbenchmark
   shows that a kernel is fast. Only an end-to-end comparison shows that the
   feature helps.
5. **Correct and active before fast.** Prove that the feature path ran
   (counters, logs, trace ranges) and that outputs are correct. Timing a path
   that silently did not activate is worse than no number.
6. **Repeat and show variation.** A difference smaller than the run-to-run
   spread is "no measurable difference", not a win or a regression.
7. **One engine per GPU set.** Never co-locate two engines on the same GPUs.
   With the LLM API, `tensor_parallel_size=N` spawns its own workers; wrapping
   the script in `mpirun -n N` starts N full engines.
8. **Light instrumentation for timed runs.** NVIDIA Tools Extension (NVTX)
   ranges are fine. The PyTorch profiler trace, debug NVTX, garbage-collection
   recording, and synchronizing checks change timings; use them only in a
   separate run that inspects structure.
9. **Never perturb the baseline.** A harness, probe, or control call must not
   add work, delay, or synchronization to either arm. Trigger the feature
   through its normal lifecycle, not through injected calls.
10. **Treat implausible results as bugs.** Examples:
    - non-monotonic curves;
    - a larger cache with a lower hit rate;
    - a compressed arm slower everywhere;
    - a lossier arm scoring higher.

    Each points at an unaligned condition or a defect. Find the cause before
    explaining the result.

## Workflow

### 1. Write the performance hypothesis

Before running anything, state the mechanism and the expected size of the
effect.

- **Capacity**: more retained pages, higher reuse hit rate, or higher
  sustainable concurrency. Savings count only when they become capacity:
  evicted or compressed blocks must be physically freed, and the larger batch
  or retained-token count must be shown. Report the largest batch each arm
  can run.
- **Bytes**: fewer bytes offloaded, onboarded, or transferred between workers.
- **Attention work**: fewer tokens read from the KV cache per step.
- **Overhead**: a per-step cost the feature adds, such as staging copies,
  metadata building, compression launches, or host bookkeeping.

The hypothesis decides the metric. A capacity feature that does not raise hit
rate or concurrency at the operating point has no benefit there, whatever its
kernel speed.

Work out where the feature should win before sweeping. A capacity feature
wins only where the baseline cache can no longer hold the reusable working
set and the feature's cache still can. Estimate that region from the real
configuration, the defaults, and the hardware. Wrong orders of magnitude are
not acceptable. Then put the sweep there.

### 2. Choose baselines

| Feature type | Required baselines |
|---|---|
| Any change on the KV-cache data path | Same build with the feature off; main at the merge base |
| KV cache dtype or precision feature | FP8 KV cache and the model dtype; for the mixed-precision KV cache, also the pure NVFP4 KV cache |
| Compression or eviction | Full KV cache with the same KV cache memory budget; state whether the arms hold equal bytes or equal token capacity |
| Reuse-policy change | The previous policy on the same trace, with hit rates reported for both |
| Optimization of an existing path | The unoptimized path at the same batch size and sequence lengths |

### 3. Choose workloads

- Use the input and output lengths and the concurrency the feature targets.
  Sweep concurrency and report a Pareto view (per-GPU throughput against
  per-user speed or latency) rather than one point.
- Reuse features need traces with realistic prefix sharing: multi-turn or
  agentic sessions, shared system prompts. Identical prompts with block reuse
  on give an artificial 100% hit rate unless reuse itself is under test.
- Long-context features need long inputs or long generations. A short-sequence
  benchmark hides costs that scale with sequence length.
- For lossy features, also run with natural end-of-sequence instead of a fixed
  output length. A lossy KV cache can change how long the model generates;
  answers that run several times longer erase a per-token gain end to end.
- Separate warm-up from measurement, and measure steady state. Report
  ramp-up and drain only when they are the subject.
- Include the worst case: for example, every request compressing, offloading,
  or rebuilding in the same step, and the largest page batches real reuse
  moves. Collect per-iteration data; averages hide spikes.
- Size each arm's KV cache memory and sweep its batch size up to that arm's
  own maximum. A capacity feature sized like the baseline cannot show its
  gain.
- Reproduce the baseline frontier yourself. Do not fill gaps with published
  numbers.
- When reproducing a reference experiment, read its scripts, configurations,
  and raw outputs, and tabulate every setting side by side with yours before
  the first run.

### 4. Run the uninstrumented comparison

- Use `trtllm-bench ... throughput --config <yaml> --report_json <file>` for
  offline throughput, or `trtllm-serve` with
  `tensorrt_llm.serve.scripts.benchmark_serving` for online serving.
- Record KV-cache statistics: free and used blocks, reuse hits, reused
  tokens, and migration counts where the feature affects them. Enable
  `enable_iter_perf_stats` and read the per-iteration statistics through
  `LLM.get_stats()` or the serving metrics, in the same run as the timing.
- Use the same GPU model and, ideally, the same node for both arms. Note the
  clocks, power limits, and persistence settings
  (`perf-benchmarking.md`, "Before Benchmarking").

### 5. Attribute

Profile the same workload over a steady-state window, as in `perf-analysis.md`:

```bash
TLLM_PROFILE_START_STOP=<A>-<B> nsys profile -o <name> -f true \
  -t cuda,nvtx -c cudaProfilerApi --cuda-graph-trace node \
  --trace-fork-before-exec=true \
  trtllm-bench --model <model> throughput --dataset <dataset> \
    --backend pytorch --config <yaml>
```

Keep these four numbers apart. They answer different questions.

| Quantity | Meaning |
|---|---|
| Kernel-work sum | Total GPU time of the feature's kernels; overlapped work still counts |
| GPU-busy critical path | Time the feature adds to the GPU timeline after overlap |
| Exposed host time | CPU launch and bookkeeping time the GPU waits on |
| Application wall time | What the request sees |

Name every per-step cost the feature adds, and say whether it runs inside the
CUDA graph, outside it, or on a side stream.

### 6. Change one thing and re-measure

Apply one scoped change, rerun the comparison at the same batch size and
sequence lengths, and check peak memory. Before declaring an improvement, rerun
the accuracy or functional gate. See `references/optimization-patterns.md` for
changes that worked before and the guardrails that come with them.

## KV-Cache Settings That Move Performance

Tune through configuration before changing code. Change one setting per arm,
read the resolved values from the startup log, and keep everything else
identical. These fields are on `kv_cache_config` unless noted.

| Setting | What it changes |
|---|---|
| `free_gpu_memory_fraction` (default 0.9), `max_tokens`, `max_gpu_total_bytes` | The size of the KV cache pool. Check in the startup log which limit binds, and state whether arms are compared at equal bytes or equal token capacity |
| `tokens_per_block` (default 32) | Page size: reuse granularity, partial pages, migration batch size, staging size |
| `dtype` (`fp8`, `nvfp4`, `mixed_nvfp4_*`) | Bytes per token, and which attention kernels run |
| `enable_block_reuse`, `enable_partial_reuse`, `block_reuse_config.policy` | What a later request can reuse, and when blocks are committed |
| `pool_ratio`, `avg_seq_len`, `enable_kv_pool_rebalance` | How memory is split across layer groups, including sliding-window, sink, and mixed-precision high layer groups |
| `host_cache_size` (bytes, default 0), `cold_tier_offload_min_priority` | Host-tier capacity, and which evicted pages are offloaded rather than dropped |
| Top level: `max_batch_size`, `max_num_tokens`, `enable_chunked_prefill`, `cuda_graph_config` | Execution shape. Keep production values; change them only as a separate, labeled arm |

## Hot-Path Audit for KV-Cache Features

Run this audit on every new or changed per-step path. Read the call sites and
their guards, not just the function body. A buffer zeroed "once" in its
constructor is zeroed every step if the caller rebuilds the object every step.

- **Host work per step**: loops over requests, layers, or pages in Python;
  per-step tensor construction on the CPU; repeated dictionary or config
  lookups.
- **Synchronization**: `.item()`, `.tolist()`, `.cpu()`, `torch.cuda.synchronize`,
  and blocking events in the step loop. These break overlap-scheduler
  overlap.
- **Allocation and zeroing**: per-step `torch.empty`/`zeros`, per-step `memset`
  of staging or scratch buffers, buffers whose size depends on the batch.
- **Work sized by the maximum instead of the actual**: inside CUDA graphs, a
  conversion over `max_seq_len` or the full staging span costs the maximum on
  every step. Quantify the bytes per step.
- **Copies**: extra gather and scatter into staging buffers, double writes,
  host round trips.
- **CUDA graph coverage**: anything that forces eager mode, recapture, or
  padding to larger shapes. Check that the intended CUDA graph is still
  replayed.
- **Kernel selection**: a new dtype or layout can route attention to a slower
  kernel or another backend. Check the kernels that actually run, not the
  configured backend name.
- **Capacity side effects**: extra pools, staging buffers, or retained copies
  reduce the KV cache memory budget. Confirm that the profiling run that sizes
  the KV cache memory budget accounts for them, and report the change in
  maximum concurrency or retained tokens. Read peak device memory from the
  driver (`nvidia-smi`), not from framework pool quotas, which can hide
  allocations outside the pools.
- **Migration placement**: encode, decode, and transfer work should overlap
  model execution unless the request is blocked on it. Show which case
  applies.

## Reading Results

- Higher throughput with worse TPOT usually means larger batches. Compare at
  equal latency, which is what the Pareto view is for.
- With reuse involved, a "faster" arm may simply have reused more because of a
  different eviction outcome. Report hit rates and reused tokens for both arms.
- If an arm runs longer only because of warm-up or slow outliers, report it as
  that; do not blame the feature.
- Check that both arms ran the configuration you intended. Startup logs print
  the resolved KV cache dtype, pools, window sizes, and backends.
- Explain counter-intuitive results, such as "disabling X is faster", with an
  attribution and a full matrix. A single point and a mechanism story are not
  enough.

## Report Template

```markdown
## Performance: <feature> vs <baseline>
- Build: <commit>; local changes: <none | diff hash>
- Model, GPU model x count, parallelism, attention backend, KV cache dtype,
  KV cache memory budget
- Workload: <dataset or trace>, input/output lengths, concurrency points,
  sampling, warm-up, duration, repeats
- Configuration difference between the arms (the configurations compared):
  <exact keys and values>
- Activation evidence: <counter or log line showing the feature ran>

| Concurrency | Arm | Output tokens/s per GPU | Time per output token (TPOT) p50/p99 | Time to first token (TTFT) p50/p99 | Reuse hit rate % | Peak memory |
|---|---|---|---|---|---|---|

- Variation: <spread over N runs>
- Attribution: <kernel-work sum / GPU-busy time / exposed host time / wall
  time; per-step costs the feature adds>
- Conclusion: <within noise | X% at concurrency Y>; not measured: <list>
- Artifacts: <one folder with scripts, configurations, raw outputs, traces>
```

Show absolute values side by side, with units. Do not chart only speedups.
State cold or warm cache, the hardware, and the parallel layout.

## References

- `references/optimization-patterns.md`: optimizations that worked for
  KV-cache features and the guardrails each one needs.
- `../trtllm-kv-cache-accuracy-eval/SKILL.md`: the accuracy gate to rerun
  after each optimization.
- `../trtllm-kv-cache-compat-testing/SKILL.md`: the feature combinations whose
  performance must also be checked, such as reuse, CUDA graph, and
  disaggregated serving.
