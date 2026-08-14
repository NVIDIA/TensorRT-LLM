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
| `ncu` binary | Match CUDA version | Or set `$NCU` env var |
| NVIDIA GPU | Kepler+ | Volta+ recommended |

Permissions: `ncu` may require `sudo`, `CAP_SYS_ADMIN`, or `--privileged`
in containers. Check with `ncu -v` first.

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
- **Compute SOL%** = actual compute throughput / peak compute throughput
- **Memory SOL%** = actual memory throughput / peak memory throughput

A kernel cannot saturate both simultaneously. The higher metric reveals the bottleneck type. Use this as the primary classification signal.

### Classification Thresholds

| Compute % | Memory % | Bottleneck | Next Step |
|-----------|----------|------------|-----------|
| >60 | <40 | **Compute-bound** | ComputeWorkloadAnalysis section |
| <40 | >60 | **Memory-bound** | MemoryWorkloadAnalysis section |
| <40 | <40 | **Latency-bound** | LaunchStats + Occupancy sections |
| 40-60 | 40-60 | **Balanced** | Profile deeper with detailed sections |

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
    --kernel-name regex:"gemm" \
    --launch-count 3 -- python script.py
```

Example -- occupancy investigation:
```bash
ncu --section SpeedOfLight --section LaunchStats --section Occupancy --csv \
    --kernel-name regex:"small_kernel" \
    -- python script.py
```

### Step 3: Roofline Analysis (Optional)

For visual understanding of compute vs memory balance:

```bash
ncu --section SpeedOfLight_RooflineChart \
    --kernel-name regex:"KERNEL" -- COMMAND
```

For precision-specific hierarchical roofline:

```bash
# FP16 kernels
ncu --section SpeedOfLight_HierarchicalHalfRooflineChart \
    --kernel-name regex:"KERNEL" -- COMMAND

# Tensor core kernels
ncu --section SpeedOfLight_HierarchicalTensorRooflineChart \
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

Extract metrics from `.ncu-rep` files using the `ncu_report` Python module
(in `extras/python/` of the Nsight Compute installation):

```python
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")
for rng in ctx:
    for action in rng:
        name = action.name()
        compute = action["sm__throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
        memory = action["dram__throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
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

**Report files** (for later analysis):
```bash
ncu -o report --section SpeedOfLight -- COMMAND
ncu --import report.ncu-rep --csv --page raw            # Export to CSV
```

**Key CSV columns:**

| Column | Meaning |
|--------|---------|
| `Kernel Name` | CUDA kernel function name |
| `Duration` | Execution time (nanoseconds) |
| `Compute (SM) Throughput` | % of peak compute |
| `Memory Throughput` | % of peak memory bandwidth |
| `Achieved Occupancy` | Active warps / max warps (%) |

**Success indicators:**
- SOL% values present in output -> profiling succeeded
- Duration values reasonable (not 0 or extremely large)
- Multiple launches captured when `--launch-count > 1`

## Examples

### Example: Classify a GEMM Kernel

```bash
ncu --section SpeedOfLight --csv \
    --kernel-name regex:"gemm" \
    --launch-skip 5 --launch-count 3 \
    -- python train.py
```

Output:
```
"Kernel Name","Duration","Compute (SM) Throughput","Memory Throughput"
"ampere_fp16_gemm",1250000,78.5,35.2
```

Interpretation: compute-bound (78.5% compute, 35.2% memory). Next step:
check tensor core usage with `--section ComputeWorkloadAnalysis`.

### Example: Diagnose a Memory-Bound Embedding Kernel

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --csv \
    --kernel-name regex:"embedding" \
    --launch-count 3 -- python train.py
```

Check L1/L2 cache hit rates and coalescing efficiency in output. Low hit rates
suggest poor data locality; low coalescing efficiency suggests scattered access.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `ncu: command not found` | Not in PATH | `export PATH=$PATH:/usr/local/cuda/bin` or set `$NCU` |
| `Permission denied` | Needs elevated privileges | `sudo ncu ...` or `--cap-add=SYS_ADMIN` in containers |
| No kernels captured | Name regex doesn't match | Run without `--kernel-name` first to see actual names |
| Profiling extremely slow | Using `--set full` or many sections | Use `--section SpeedOfLight` only; reduce `--launch-count` |
| Autotuning pollutes results | JIT kernel warmup captured | Use `--profile-from-start off` with profiler markers |
| Metrics show 0% tensor cores | Kernel doesn't use tensor cores | Check with `--section InstructionStats`; verify dimensions align to 8/16 |
| Report file too large | `--set full` with many kernels | Use targeted sections; limit with `--kernel-name` and `--launch-count` |
| Out-of-range metric values | Async GPU activity or short kernels | Profile on isolated GPU; increase workload size |
| `ncu` hangs on MPI app | Dependent kernels across ranks | Use `--communicator=tcp --lockstep-kernel-launch` |

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
- [CLI Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) -- Full CLI options
- [Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html) -- `ncu_report` API
- [Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html) -- Section files, rules

WebFetch or WebSearch these URLs for the latest content. Consider distilling
new findings back into `references/`.
