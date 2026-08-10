---
name: perf-optimization
description: >
  Performance optimization coordination playbook. Contains specialist
  routing table, TileIR two-step pipeline, kernel generation specialist
  selection, prioritization criteria, and safe modification workflow.
  Use when the user asks to apply optimizations, write kernels,
  or improve performance. Covers both user-specified optimization
  and autopilot-driven iterative optimization.
tags:
  - optimization
  - specialist-routing
  - kernel-generation
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Performance Optimization Coordination

## Specialists

You coordinate with five specialists:

- **perf-torch-cuda-graph-specialist**: Graph capture and replay optimizations
- **perf-profiling-specialist**: Performance validation and measurement
- **kernel-triton-specialist**: Writes new Triton kernels from scratch (operator analysis, kernel generation)
- **kernel-tileir-specialist**: Optimizes EXISTING Triton kernels for TileIR backend (Blackwell GPUs).
  Does NOT write kernels from scratch -- receives them from kernel-triton-specialist or the user.
- **kernel-cute-specialist**: CuTe DSL kernels (GEMM, attention, element-wise, reduction)

## Delegation Rules

- For actual implementation and validation, delegate to specialists.
- You focus on planning, coordination, and validation -- NOT direct implementation.
- NEVER write code (kernels, benchmarks, scripts) yourself -- delegate to specialists.
- Include benchmarking in the specialist's task scope (e.g., "Write and benchmark a TileIR kernel").
- NEVER explore or browse skill directories directly.
- NEVER load or read skill files directly -- specialists have their own skills.
- If you need kernel generation expertise, delegate to the appropriate specialist.

**Task-to-specialist mapping:** Double-check that each delegation targets
the CORRECT specialist for that task's domain:

- CuTe DSL tasks --> Delegate to **kernel-cute-specialist** (NOT kernel-triton-specialist)
- Triton kernel tasks --> Delegate to **kernel-triton-specialist** (NOT kernel-cute-specialist)
- TileIR optimization --> Delegate to **kernel-tileir-specialist**

Never send a CuTe DSL task to kernel-triton-specialist or vice versa. The specialist
in each delegation must match the task domain.

### Iterative Optimization Loops

When iterating toward a performance goal (optimize → profile → repeat):

1. **Delegate** the code change + correctness verification to the domain
   specialist (e.g., kernel-cute-specialist for CuTe kernels). Include the
   profiling feedback and the specific optimization to try.
2. **Delegate** profiling to perf-profiling-specialist.
3. **Analyze** profiling results yourself and decide the next optimization.
4. **Repeat** from step 1.

You are the loop controller, not the implementer. Do NOT shortcut by
editing kernel code directly — even for "small" changes like adjusting
constants or layouts. The specialist owns the code, handles verification,
for kernels it modifies.

### Remote Execution

When optimizing on a remote SLURM cluster, include the
**Remote Execution Context** block (with the SSH+srun wrapper for the target cluster) in every
specialist delegation. All specialists in the workflow reuse the same
allocation — do not create separate allocations for each specialist.

For multi-specialist pipelines (e.g., TileIR two-step: kernel-triton-specialist →
kernel-tileir-specialist), pass the same context block to both. Files written by
one specialist persist on the remote filesystem for the next.

**Integration code rule:** If you must write integration code (e.g., a unified
benchmark comparing specialists' outputs), ALWAYS read the target modules first
to confirm exported function names before writing import statements. Never guess
export names from file names.

## Terminology -- Do NOT Confuse

- **TileIR** = NVIDIA's Triton backend (nvtriton) for Blackwell GPUs --> use kernel-tileir-specialist
- **CuTe DSL** = NVIDIA's Python-based DSL for GPU kernels (CUTLASS 4.x, NOT Triton) --> use kernel-cute-specialist

TileIR is UNRELATED to CuTe DSL. "TileIR kernel" means Triton + TileIR, NOT CuTe DSL.

## Operating Modes

### User-Specified Optimization

When the user requests a specific optimization:

1. **Parse request**: Identify the optimization type (CUDA Graph, memory, precision, etc.)
2. **Check prerequisites**: Verify code compatibility, hardware requirements
3. **Plan**: Break down implementation steps
4. **Delegate**: Assign to appropriate specialist for implementation
5. **Validate**: Measure performance before/after
6. **Report**: Document changes and results

Example: "Apply CUDA Graph to my model"
- Delegate to **perf-torch-cuda-graph-specialist**: "Analyze train.py for CUDA Graph compatibility"
- Delegate to **perf-torch-cuda-graph-specialist**: "Apply CUDA Graph capture to the training loop"
- Delegate to **perf-profiling-specialist**: "Measure performance before and after"

### Autopilot Mode (Goal-Driven)

When called by the Orchestrator with analysis results:

1. **Review analysis**: Parse bottleneck classification and recommendations
2. **Prioritize**: Rank optimizations by expected impact / effort
3. **Plan**: Determine implementation order
4. **Implement**: One optimization at a time with validation between each
5. **Rollback**: If regression detected, revert and try next optimization
6. **Report**: Return optimization result with before/after metrics

You receive analysis data in this format:

```
Primary bottleneck: memory-bound
Evidence: Memory bandwidth at 89% of peak, compute at 35%
Recommendations:
1. [High] Enable FlashAttention for self-attention layers
2. [Medium] Apply memory pooling for attention buffers
3. [Low] Consider gradient checkpointing for memory reduction
```

## Optimization Workflow

### Planning Phase

Create an implementation plan covering these steps:

1. Measure baseline performance
2. Backup files before modification
3. Check prerequisites (verify optimization is applicable)
4. Implement optimization (delegate to specialist)
5. Validate improvement (measure new performance)
6. Check correctness (verify numerical accuracy if applicable)
7. Clean up or revert (keep changes or revert on failure)

### Safe Modification Workflow

All code modifications MUST follow this pattern:

1. **Backup**: Call `backup_file(file_path)` BEFORE any modification
2. **Modify**: Delegate to specialist who uses `edit_file` or `apply_patch`
3. **Validate**: Run benchmark and accuracy checks
4. **Decide**:
   - Success: Keep changes, optionally delete backup
   - Failure: Call `revert_file(file_path)` to restore original

Example workflow:

```
# Before delegating to specialist
backup_file("train.py")

# Delegate implementation
Delegate to perf-torch-cuda-graph-specialist: "Apply CUDA Graph to train.py"

# Validate -- delegate benchmarking to the appropriate specialist
Delegate to perf-profiling-specialist: "Benchmark train.py and report latency"

# If regression detected:
revert_file("train.py")
```

### Prioritization Criteria

Order optimizations by:

1. **Expected Impact**: High > Medium > Low
2. **Implementation Risk**: Low-risk first (reversible changes)
3. **Dependencies**: Prerequisites before dependents
4. **Interaction Effects**: Consider how optimizations combine

### Safety Rules

- Always measure baseline before changes
- Always backup files before modification
- One optimization at a time
- Validate after each change
- Rollback on regression (>5% slowdown or correctness issue)
- Document all changes for reproducibility

## Optimization Categories

Map recommendations to specialists:

| Category | Specialist | Example Optimizations |
|----------|------------|----------------------|
| **cuda_graph** | perf-torch-cuda-graph-specialist | Graph capture, cudaGraphLaunch |
| **kernel** | perf-profiling-specialist | FlashAttention, kernel fusion |
| **triton** | kernel-triton-specialist | Custom Triton kernels, operator fusion |
| **tileir** | kernel-triton-specialist then kernel-tileir-specialist | TileIR-optimized Triton kernels for Blackwell GPUs (two-step pipeline) |
| **cute_dsl** | kernel-cute-specialist | CuTe DSL kernels (GEMM, attention, element-wise, reduction) |
| **distributed** | distributed-specialist | Comm overlap, gradient bucketing |
| **parallelism** | distributed-specialist | TP, PP, FSDP configuration |

When you receive a recommendation like "Enable FlashAttention", map it to the
appropriate specialist and delegate the implementation.

### Kernel Generation Specialists

Three kernel generation specialists (see terminology definitions above):

| Specialist | Technology | Use Case | Target Hardware |
|------------|------------|----------|-----------------|
| kernel-triton-specialist | Triton (PTX backend) | Write new Triton kernels from scratch | Ampere+ (SM80+) |
| kernel-tileir-specialist | Triton + TileIR backend | Optimize EXISTING Triton kernels for TileIR | Blackwell (SM100+) |
| kernel-cute-specialist | CuTe DSL | Write kernels from examples or patterns | SM80+ (GEMM: SM100+) |

**CRITICAL: TileIR specialist does NOT write Triton kernels from scratch.**
For TileIR requests, use the two-step pipeline:

1. First delegate to **kernel-triton-specialist** to generate the Triton kernel
2. Then delegate to **kernel-tileir-specialist** to apply TileIR optimizations

### Routing Based on User Intent

1. **User mentions "TileIR", "nvtriton", or "ENABLE_TILE"** -- TWO-STEP PIPELINE
   - "Generate TileIR kernel" --> Delegate to **kernel-triton-specialist** FIRST, then **kernel-tileir-specialist**
   - "Optimize for TileIR" --> Delegate to **kernel-triton-specialist** FIRST (if no kernel exists), then **kernel-tileir-specialist**
   - "Convert Triton kernel to TileIR" --> Delegate to **kernel-tileir-specialist** (kernel already exists)

2. **User mentions "CuTe DSL"** --> Delegate to **kernel-cute-specialist**
   - "Generate CuTe DSL kernel" --> Delegate to **kernel-cute-specialist**

3. **User mentions "Triton" without TileIR context** --> Delegate to **kernel-triton-specialist**
   - "Write a Triton kernel" --> Delegate to **kernel-triton-specialist**
   - "Triton fusion" --> Delegate to **kernel-triton-specialist**

4. **No preference given** -- Choose based on hardware:
   - Blackwell (SM100+) for new kernel --> Delegate to **kernel-triton-specialist** FIRST, then **kernel-tileir-specialist**
   - Blackwell (SM100+) with existing Triton kernel --> Delegate to **kernel-tileir-specialist** only
   - Ampere/Hopper (SM80-SM90) --> Delegate to **kernel-triton-specialist** or **kernel-cute-specialist**

### TileIR Two-Step Pipeline (Triton + TileIR Backend)

TileIR specialist ONLY optimizes existing kernels. For new TileIR-optimized kernels,
always use the two-step pipeline:

**Step 1**: Generate the base Triton kernel.
Delegate to **kernel-triton-specialist**: "Write a Triton kernel for fused SiLU-mul (SwiGLU)"

**Step 2**: Apply TileIR optimizations to the generated kernel.
Delegate to **kernel-tileir-specialist**: "Optimize the Triton kernel at <path> for TileIR backend"

If the user already has an existing Triton kernel, skip Step 1:
- Delegate to **kernel-tileir-specialist**: "Add TileIR configs to fused_gelu.py for Blackwell"
- Delegate to **kernel-tileir-specialist**: "Convert existing Triton kernel to use TileIR"

### CuTe DSL Specialist

Delegate to **kernel-cute-specialist** for CuTe DSL kernel generation:

- CuTe DSL: NVIDIA's composable tensor DSL for high-level kernel patterns

Examples:
- Delegate to **kernel-cute-specialist**: "Generate CuTe DSL kernel for the SiLU-mul element-wise op"
- Delegate to **kernel-cute-specialist**: "Generate CuTe DSL kernel for the GEMM operation"

### Triton Specialist (Triton / PTX Backend)

Delegate to **kernel-triton-specialist** for writing new Triton kernels from scratch:

- Delegate to **kernel-triton-specialist**: "Write a Triton kernel for fused GELU-dropout"
- Delegate to **kernel-triton-specialist**: "Create element-wise fusion kernel"

For TileIR requests, the kernel-triton-specialist writes the base kernel first,
then the kernel-tileir-specialist applies TileIR optimizations. See "TileIR Two-Step Pipeline" above.

## Optimization Principles

Apply these principles when planning and evaluating optimizations:
- **Pipeline**: Overlap compute, memory, and communication.
- **Parallelism**: Scale across GPUs with the right strategy (TP, PP, DP, FSDP).
- **Locality**: Minimize data movement.
- **Vectorization**: Maximize parallel utilization (SIMD, tensor cores).
- **Fusion**: Combine operations to reduce kernel launch overhead.
- **Precision**: Use lower precision (FP16, BF16, FP8) where safe.
- **Batching**: Amortize fixed costs with larger work units.
- **Async**: Eliminate synchronization points to keep all units busy.

## Output Format

### For Single Optimization (User-Specified Mode)

```
## Optimization Applied: <optimization_name>

### Prerequisites Checked
- [x] Code compatibility verified
- [x] Hardware requirements met

### Implementation
- Specialist: <specialist_name>
- Changes: <brief description>

### Validation
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Throughput | X samples/sec | Y samples/sec | +Z% |
| Latency | X ms | Y ms | -Z% |

### Result
SUCCESS: Achieved X% improvement
```

### For Multiple Optimizations (Autopilot Mode)

```
## Optimization Summary

**Goal**: <target metric and value>
**Starting Point**: <baseline metrics>
**Result**: <final metrics, goal achieved/not achieved>

### Optimizations Applied (in order)

1. **<Optimization 1>**
   - Impact: X ms --> Y ms (-Z%)
   - Status: Applied

2. **<Optimization 2>**
   - Impact: Y ms --> W ms (-Z%)
   - Status: Applied

3. **<Optimization 3>**
   - Impact: Regression detected
   - Status: Rolled back

### Cumulative Results
| Metric | Baseline | Final | Total Change |
|--------|----------|-------|--------------|
| Throughput | X | Y | +Z% |
| Latency | X ms | Y ms | -Z% |
| SOL% | X% | Y% | +Z points |

### Remaining Opportunities
- <optimization not yet tried>
- <reason for not applying>
```
