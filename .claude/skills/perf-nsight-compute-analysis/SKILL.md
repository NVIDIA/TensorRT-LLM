---
name: perf-nsight-compute-analysis
tags: [profiling]
description: >
  Analyze ncu (NVIDIA Nsight Compute) profiling output: SOL% bottleneck
  classification, roofline analysis, occupancy diagnosis, memory hierarchy
  analysis, warp stall analysis, metric interpretation, and programmatic
  .ncu-rep report analysis. NOT for kernel writing or code generation,
  Nsight Systems (nsys), host-side profiling, or system-level profiling.
compatibility: Requires NVIDIA GPU with ncu (Nsight Compute) installed
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Nsight Compute Analysis

NVIDIA Nsight Compute (`ncu`) profiles individual CUDA kernels to determine
why they are slow and what to optimize. It measures GPU throughput as a
percentage of theoretical peak (Speed of Light / SOL%), enabling systematic
bottleneck classification and targeted optimization.

## When to Use

Reach for this skill when you encounter:

- **Triggers**: User wants to profile a CUDA kernel, analyze `ncu` output,
  interpret `.ncu-rep` reports, or optimize GPU kernel performance
- **Symptoms**: Kernel running slower than expected, low GPU utilization,
  need to classify compute-bound vs memory-bound, occupancy issues
- **Keywords**: "ncu", "nsight compute", "SOL%", "speed of light", "kernel
  profiling", "compute-bound", "memory-bound", "latency-bound", "occupancy",
  "roofline", "warp stalls", "cache hit rate", "ncu-rep"

Do NOT use this skill for:
- System-level profiling (use Nsight Systems / `nsys` instead)
- CUDA API tracing or CPU-GPU timeline analysis (use `nsys`)
- GPU monitoring without profiling (use `nvidia-smi`)

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Toolkit | >=11.0 | Includes `ncu` |
| `ncu` binary | One the driver supports | Nsight Compute 2026.3 needs a driver compatible with CUDA 13. Or set `$NCU` env var |
| NVIDIA GPU | Turing+ | Volta needs Nsight Compute 2025.2 or older |

Permissions: `ncu` may require `sudo`, `CAP_SYS_ADMIN`, or `--privileged`
in containers. `ncu -v` only prints the version; missing permission shows up
as `ERR_NVGPUCTRPERM` when the first kernel is profiled.

## Principles

### Data Integrity

This is a data-driven analysis system. **Every number you present must have
an authoritative source.** Follow these rules without exception:

1. **Quote before you interpret.** When presenting metrics from ncu output,
   always show the actual ncu command you ran AND the relevant raw output
   (CSV lines, metric values) before stating any numeric conclusion.
2. **Never fabricate metrics.** If ncu fails, returns unexpected output, or
   you cannot run it, say so explicitly. Do not invent plausible-looking
   numbers. An honest "profiling failed" is better than fabricated data.
3. **Attribute every value.** For each metric you cite (SOL%, duration,
   occupancy, throughput), the reader must be able to trace it back to a
   specific line in the raw ncu output you showed.

### SOL% Mental Model

Speed of Light (SOL%) measures how close a kernel runs to the GPU's theoretical peak:
- **Compute SOL%** (`Compute (SM) Throughput`) = the busiest SM unit's throughput as a % of its peak
- **Memory SOL%** (`Memory Throughput`) = the busiest memory unit's throughput (DRAM, L2, L1, shared memory) as a % of its peak

The higher metric usually reveals the bottleneck type. Use this as the primary classification signal. Both can be high at once: the kernel is then near the hardware limit for its algorithm.

### Classification Thresholds

| Compute % | Memory % | Bottleneck | Next Step |
|-----------|----------|------------|-----------|
| >60 | <40 | **Compute-bound** | ComputeWorkloadAnalysis section |
| <40 | >60 | **Memory-bound** | MemoryWorkloadAnalysis section |
| <40 | <40 | **Latency-bound** | LaunchStats + Occupancy sections |
| 40-60 | 40-60 | **Balanced** | Profile deeper with detailed sections |
| >60 | >60 | **Near hardware limits** | Algorithmic change; check both workload sections |

Additional signals:
- Duration <10us with many launches -> **Launch-overhead bound** (use nsys first)
- Both <40% but occupancy >50% -> **Instruction-bound** (check InstructionStats)

### SOL% Performance Levels

| SOL% | Level | Action |
|------|-------|--------|
| >80% | Excellent | Minor tuning only |
| 60-80% | Good | Targeted optimization |
| 40-60% | Fair | Significant optimization needed |
| <40% | Poor | Major rework needed |

### Section-First Profiling

Always use targeted `--section` flags instead of bulk `--set` collection. Individual sections are faster and more surgical. Only escalate to `--set basic` or `--set detailed` when broad exploration is needed.

### ncu vs nsys

| Tool | Scope | Overhead | Purpose |
|------|-------|----------|---------|
| **nsys** | System-level | 5-10% | Find which kernels to optimize |
| **ncu** | Kernel-level | 10-100x slower | Understand why a kernel is slow |

Use nsys first to identify top kernels by GPU time, then ncu for deep analysis of those specific kernels.

## Workflow

**Choose your path based on the request:**
- **Knowledge query** (what metrics to use, --section vs --set, how to filter kernels):
  Answer directly from Principles, Command Reference, and References below. Do NOT run ncu.
- **Quick diagnosis** (classify bottleneck, check SOL%): Step 1 only. Escalate if user wants more.
- **Specific diagnosis** (bank conflicts, register pressure, occupancy): Quick SOL% check (Step 1),
  then go directly to the relevant section in Step 2.
- **Deep analysis** (detailed report, optimization recommendations): Full Steps 1-5.
  Present the complete structured report with all key metrics (SOL%, duration,
  occupancy) in your final response — do not split the report across messages
  or replace it with a brief summary.

### Step 0: Verify ncu

```bash
ncu -v
# Or: $NCU -v
```

If not found, ensure CUDA toolkit is installed or set `NCU` env var to the binary path.

### Step 1: SOL% Diagnosis

Always start with SpeedOfLight to classify the bottleneck:

```bash
ncu --section SpeedOfLight --csv \
    --kernel-name regex:"KERNEL" \
    --launch-skip 5 --launch-count 3 \
    -- COMMAND
```

Read `Compute (SM) Throughput` and `Memory Throughput` from the output. Classify using the thresholds above.

### Step 2: Escalate with Targeted Sections

Based on Step 1 classification, add sections:

| Classification | Sections to Add |
|----------------|-----------------|
| Compute-bound | `ComputeWorkloadAnalysis` |
| Memory-bound | `MemoryWorkloadAnalysis` |
| Latency-bound | `LaunchStats`, `Occupancy` |
| Warp stalls | `WarpStateStats`, `SchedulerStats` |
| Need instruction breakdown | `InstructionStats` |

Always include `LaunchStats` and `Occupancy` when diagnosing latency-bound kernels. These reveal register pressure, shared memory limits, and block size issues.

Example -- memory-bound deep dive:
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --csv \
    --kernel-name regex:"embedding_lookup" \
    --launch-count 3 \
    -- python script.py
```

Example -- compute-bound deep dive:
```bash
ncu --section SpeedOfLight --section ComputeWorkloadAnalysis --csv \
    --kernel-name regex:"gemm|nvjet" \
    --launch-count 3 -- python script.py
```

Example -- occupancy investigation:
```bash
ncu --section SpeedOfLight --section LaunchStats --section Occupancy --csv \
    --kernel-name regex:"small_kernel" \
    -- python script.py
```

### Step 3: Roofline Analysis (Optional)

For visual understanding of compute vs memory balance. The roofline sections
are charts: on the CLI they print only with `--print-details all`, which lists
the numbers behind the chart (achieved and peak work and traffic). To see the
chart, save a report with `-o` and open it in the Nsight Compute UI.

```bash
ncu --section SpeedOfLight_RooflineChart --print-details all \
    --kernel-name regex:"KERNEL" -- COMMAND
```

For precision-specific hierarchical roofline:

```bash
# FP16 kernels
ncu --section SpeedOfLight_HierarchicalHalfRooflineChart --print-details all \
    --kernel-name regex:"KERNEL" -- COMMAND

# Tensor core kernels
ncu --section SpeedOfLight_HierarchicalTensorRooflineChart --print-details all \
    --kernel-name regex:"KERNEL" -- COMMAND
```

Interpretation: kernel left of ridge point = memory-bound; right = compute-bound;
far below both roofs = latency/occupancy issue. See `references/roofline-analysis.md`.

### Step 4: Interpret and Optimize

1. Identify the dominant bottleneck from SOL% classification
2. Look up detailed analysis and optimization strategies in `references/bottleneck-guide.md`
3. Apply highest-impact optimization first
4. Re-profile to validate improvement and detect bottleneck shifts

### Step 5: Validate

Re-profile the same kernel after optimization:

```bash
ncu --section SpeedOfLight --csv \
    --kernel-name regex:"optimized_kernel" \
    --launch-count 3 \
    -- python optimized_script.py
```

Compare: Did throughput % increase? Did duration decrease? Did the bottleneck type shift?

## Profiling JIT-Compiled Kernels (Triton/cuTile/CuTeDSL)

JIT-compiled kernels trigger autotuning on first invocation. Isolate the actual execution:

1. **Warmup first**: Run the kernel 3-5 times to complete JIT compilation and autotuning, then `torch.cuda.synchronize()`.
2. **Use profiler markers**: Bracket the measured region with `cudaProfilerStart()`/`cudaProfilerStop()`.
3. **Use `--profile-from-start off`** so ncu only captures the marked region:

```python
# Warmup (JIT + autotuning)
for _ in range(5):
    result = kernel(inputs)
torch.cuda.synchronize()

# Profile only steady-state
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):
    result = kernel(inputs)
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

```bash
ncu --profile-from-start off --section SpeedOfLight --csv \
    --kernel-name regex:"target_kernel" \
    --launch-count 3 -- python script.py
```

Alternative: use `--launch-skip N` to skip autotuning launches. See
`references/advanced-profiling.md` for NVTX range and replay mode alternatives.

## Programmatic Report Analysis

Extract metrics from `.ncu-rep` or `.ncu-repz` files using the `ncu_report` Python module
(in `extras/python/` of the Nsight Compute installation):

```python
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")
for rng in ctx:
    for action in rng:
        name = action.name()
        # SpeedOfLight's "Compute (SM) Throughput" and "Memory Throughput"
        compute = action["sm__throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
        memory = action["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
        duration = action["gpu__time_duration.sum"].as_uint64()

        if compute > 60:
            classification = "compute-bound"
        elif memory > 60:
            classification = "memory-bound"
        else:
            classification = "latency-bound"

        print(f"{name}: {classification} (compute={compute:.1f}%, mem={memory:.1f}%, {duration}ns)")
```

See `references/python-report-api.md` for the full API (IContext, IRange, IAction, IMetric classes).

## Output Formats

**CSV output** (for scripting and automated analysis):
```bash
ncu --csv --section SpeedOfLight --kernel-name regex:"KERNEL" -- COMMAND
ncu --csv --page raw --section SpeedOfLight -- COMMAND   # All metrics flat
```

The default CSV has one row per metric. Its columns include `ID` (the
profiled launch), `Kernel Name`, `Section Name`, `Metric Name`, `Metric Unit`
and `Metric Value`; rule results are rows that fill `Rule Name`,
`Rule Description` and `Estimated Speedup` instead. A `Metric Name` can repeat
across sections (`Memory Throughput` is % in SpeedOfLight and byte/s in
MemoryWorkloadAnalysis), so filter on `Section Name` too. `--page raw` gives one
row per launch, a column per metric, and a second row of units. Values are in
base units (`ns`, `hz`, `%`).

While profiling, ncu's `==PROF==` lines and the application's own output share
stdout with the CSV. For clean CSV, write a report and read it back:

**Report files** (for later analysis):
```bash
ncu -o report.ncu-rep --section SpeedOfLight -- COMMAND
ncu --import report.ncu-rep --csv                       # One row per metric
ncu --import report.ncu-rep --csv --page raw            # One row per launch
```

Give `-o` the extension: without one, Nsight Compute 2026.3 and newer write the
zstd-compressed `report.ncu-repz`, and older versions write `report.ncu-rep`.
`--import` and `ncu_report` read both.

**Key metric rows** (`Metric Name` values):

| Metric Name | Section | Meaning |
|-------------|---------|---------|
| `Duration` | SpeedOfLight | Execution time |
| `Compute (SM) Throughput` | SpeedOfLight | % of peak compute |
| `Memory Throughput` | SpeedOfLight | % of peak of the busiest memory unit |
| `SM Frequency` | SpeedOfLight | Clock the profile ran at |
| `Achieved Occupancy` | Occupancy | Active warps / max warps (%) |

**Success indicators:**
- SOL% values present in output -> profiling succeeded
- Duration values reasonable (not 0 or extremely large)
- Multiple launches captured when `--launch-count > 1`

## Examples

### Example: Classify a GEMM Kernel

cuBLAS GEMM kernels on Blackwell are named `nvjet_*`, so match both names:

```bash
ncu --section SpeedOfLight --csv \
    --kernel-name regex:"gemm|nvjet" \
    --launch-skip 5 --launch-count 3 \
    -- python train.py
```

Output for an FP16 4096x4096 matmul on B200 (columns and rows abridged):
```
"ID",...,"Kernel Name",...,"Section Name","Metric Name","Metric Unit","Metric Value",...
"0",...,"nvjet_sm100_hsh_128x256_64x6_2x2f_2cta_h_bz_NNT",...,"GPU Speed Of Light Throughput","Memory Throughput","%","38.38",
"0",...,"nvjet_sm100_hsh_128x256_64x6_2x2f_2cta_h_bz_NNT",...,"GPU Speed Of Light Throughput","Duration","ns","97600",
"0",...,"nvjet_sm100_hsh_128x256_64x6_2x2f_2cta_h_bz_NNT",...,"GPU Speed Of Light Throughput","Compute (SM) Throughput","%","77.11",
```

Interpretation: compute-bound (77.1% compute, 38.4% memory). Next step:
check tensor core usage with `--section ComputeWorkloadAnalysis`.

### Example: Diagnose a Memory-Bound Embedding Kernel

PyTorch's embedding lookup runs `vectorized_gather_kernel`, so match it too:

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --csv \
    --kernel-name regex:"gather|index|embedding" \
    --launch-count 3 -- python train.py
```

Check L1/L2 cache hit rates in the output: low hit rates suggest poor data
locality. For scattered access, add `--section SourceCounters` and look for
excessive global sectors per instruction.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `ncu: command not found` | Not in PATH | `export PATH=$PATH:/usr/local/cuda/bin` or set `$NCU` |
| `ERR_NVGPUCTRPERM` (no permission to access GPU performance counters) | Counters are restricted to admin users | `sudo ncu ...`, `--cap-add=SYS_ADMIN` in containers, or [enable non-admin profiling](https://developer.nvidia.com/ERR_NVGPUCTRPERM) |
| `Profiling failed because a driver resource was unavailable` | Another tool, such as DCGM, is collecting profiling data | Stop that collection while `ncu` runs |
| No kernels captured | Name regex doesn't match | Run without `--kernel-name` first to see actual names |
| Profiling extremely slow | Using `--set full` or many sections | Use `--section SpeedOfLight` only; reduce `--launch-count` |
| Autotuning pollutes results | JIT kernel warmup captured | Use `--profile-from-start off` with profiler markers |
| Metrics show 0% tensor cores | Kernel doesn't use tensor cores | Check with `--section InstructionStats`; verify dimensions align to 8/16 |
| Report file too large | `--set full` with many kernels | Use targeted sections; limit with `--kernel-name` and `--launch-count` |
| Out-of-range metric values | Async GPU activity or short kernels | Profile on isolated GPU; increase workload size |
| `ncu` hangs on a multi-process app | A profiled kernel waits on another rank (NCCL, NVSHMEM) | `--communicator shmem` (one `ncu` over the node) or `--communicator=tcp --lockstep-kernel-launch` (one `ncu` per rank); see `references/advanced-profiling.md` |
| `Failed to save memory for replay` on NCCL kernels | NCCL's NVLS buffers cannot be saved for kernel replay | `NCCL_NVLS_ENABLE=0` for the profiling run, or `--replay-mode application` |

## Finding More Information

### Tier 1: This File (SKILL.md)

You are reading it now. The section-first workflow and error table above cover
the most common profiling tasks. Search this file first.

### Tier 2: references/ Directory

Grep for keywords across `references/` -- headers are grep-friendly:

- `references/cli-reference.md` -- Complete CLI options, filtering, output formats
- `references/metrics-guide.md` -- Hardware model, metric naming, key metrics
- `references/sections-guide.md` -- All `--section` names, when to use each
- `references/bottleneck-guide.md` -- Per-bottleneck root causes and optimization
- `references/memory-analysis.md` -- Memory hierarchy, cache analysis, coalescing
- `references/roofline-analysis.md` -- Roofline charts and interpretation
- `references/advanced-profiling.md` -- Replay modes, MPI, CUDA graphs, PM sampling, customization
- `references/python-report-api.md` -- `ncu_report` Python module API

**How to search:**
1. `Grep` for your keyword across `references/`
2. `Read` only the file that Grep points to

### Tier 3: Official Documentation

If Tiers 1-2 don't answer:
- [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) -- Metrics, hardware model, analysis concepts
- [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html) -- NVIDIA's top-down bottleneck workflow (Nsight Compute 2026.3+)
- [CLI Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) -- Full CLI options
- [Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html) -- `ncu_report` API
- [Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html) -- Section files, rules

WebFetch or WebSearch these URLs for the latest content. Consider distilling
new findings back into `references/`. A distilled file opens with a `**Source:**` paragraph above its first `##` section: a markdown link to the original, `Distilled <YYYY-MM-DD>` and the revision distilled, and what the original still answers.
