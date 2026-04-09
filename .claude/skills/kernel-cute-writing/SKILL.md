---
name: kernel-cute-writing
description: >
  Write and implement GPU kernels using NVIDIA CuTe DSL (CUTLASS 4.x Python
  API) — NOT for Triton, CUDA C++, or conceptual explanations.
  Trigger only when the user wants to write or implement a kernel, not when
  asking questions about CuTe DSL concepts or layouts.
  CuTe DSL uses cute.jit/cute.kernel decorators and cutlass.cute imports.
  Covers element-wise kernels, GEMM patterns, reductions, memory hierarchy
  (global/shared/register/TMA), MMA tensor core operations, software
  pipelining, and framework integration.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# CuTe DSL

CuTe DSL is a Python-based domain-specific language for GPU kernel development,
part of CUTLASS 4.x. It provides Python abstractions over CUTLASS C++ templates
with JIT compilation to optimized CUDA kernels via MLIR and ptxas.

## When to Use

**Triggers:**
- Writing CUDA kernels in Python (element-wise, GEMM, custom ops)
- Optimizing GPU memory access patterns (vectorized loads, TMA, shared memory)
- Building tensor core (MMA) kernels for Ampere/Hopper/Blackwell
- Integrating custom GPU kernels with PyTorch or JAX
- Prototyping high-performance kernels without C++ metaprogramming

**Symptoms (wrong tool otherwise):**
- Need shared memory coordination or tensor core MMA → use CuTe DSL (not Triton for complex patterns)
- Need simple element-wise ops with no shared memory → CuTe DSL or Triton both work
- Need to call existing CUTLASS C++ kernels → use CUTLASS C++ APIs instead
- Need reductions, scans, or non-GEMM collective ops → consider CUB/Thrust

**Keywords:** cute, cutlass, cute.jit, cute.kernel, from_dlpack, zipped_divide,
TiledMMA, TiledCopy, TMA, WGMMA, tcgen05, pipeline, mbarrier

## Requirements

| Requirement | Detail |
|-------------|--------|
| Platform | Linux x86_64 only |
| Python | 3.10–3.13 |
| GPU | NVIDIA Ampere+ (SM80, SM90, SM100) |
| CUDA Driver | ≥ 575.51.03 (Toolkit 12.9 compat) |
| Install | `pip install nvidia-cutlass-dsl` |
| Optional | `apache-tvm-ffi`, `torch-c-dlpack-ext` |

## Workflows

### Workflow 0: Starting from Examples (Recommended)

For any non-trivial kernel (GEMM, attention, pipelined, fused ops), start by
finding the most similar existing example to use as a **starting point** — study
its structure, then rework it for your use case. Do not copy examples verbatim;
they target specific dtypes, architectures, and problem shapes that likely differ.

1. **Pick the closest example** from the index below.
   **Prefer examples matching the target GPU architecture** (check with
   `torch.cuda.get_device_capability()`) when the operation is similar.

   Fetch via `web_fetch` with base URL
   `https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL`

   | Operation | Arch | Example path (append to base URL) |
   |-----------|------|-----------------------------------|
   | Element-wise add | SM80 | `ampere/elementwise_add.py` |
   | Element-wise + autotune | SM80 | `ampere/elementwise_add_autotune.py` |
   | Element-wise apply | SM80 | `ampere/elementwise_apply.py` |
   | SGEMM (scalar) | SM80 | `ampere/sgemm.py` |
   | Tensor-core GEMM | SM80 | `ampere/tensorop_gemm.py` |
   | Flash Attention v2 | SM80 | `ampere/flash_attention_v2.py` |
   | HSTU Attention | SM80 | `ampere/hstu_attention.py` |
   | Shared memory allocator | SM80 | `ampere/smem_allocator.py` |
   | CTA norm (LayerNorm) | SM90 | `hopper/cta_norm.py` |
   | Dense GEMM | SM90 | `hopper/dense_gemm.py` |
   | Dense GEMM persistent | SM90 | `hopper/dense_gemm_persistent.py` |
   | Flash MHA | SM90 | `hopper/fmha.py` |
   | Dense GEMM | SM100 | `blackwell/dense_gemm.py` |
   | Dense GEMM persistent | SM100 | `blackwell/dense_gemm_persistent.py` |
   | Dense GEMM + alpha/beta | SM100 | `blackwell/dense_gemm_alpha_beta_persistent.py` |
   | RMSNorm | SM100 | `blackwell/rmsnorm.py` |
   | Reduce | SM100 | `blackwell/reduce.py` |
   | Flash MHA | SM100 | `blackwell/fmha.py` |
   | Grouped GEMM | SM100 | `blackwell/grouped_gemm.py` |
   | Mamba2 SSD | SM100 | `blackwell/mamba2_ssd/` |
   | GEMM tutorial (notebook) | SM100 | `notebooks/tour_to_sol_gemm.ipynb` |

   **Example:** To fetch the Hopper dense GEMM:
   ```bash
   web_fetch https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/hopper/dense_gemm.py
   ```

2. **Read reference materials first** — before diving into example code, read
   the relevant `references/` docs to understand the patterns and APIs:
   - For GEMM: `references/patterns-gemm.md` (3-level tiling, epilogue fusion,
     `cute.compile` with `mark_layout_dynamic`, shared memory layouts)
   - For reductions: `references/patterns-reduction.md` (warp reductions,
     `cute.compile` cache pattern)
   - For element-wise: `references/patterns-elementwise.md` (variations A–E)
   - Always: `references/api-arch.md` (available APIs, arch-specific caveats)

   This gives you the conceptual foundation so you can rework the example
   intelligently rather than trying to copy-paste complex pipelines.

3. **Fetch and study the example source** — read for structure, not to copy:
   - Identify: decorators, tiling strategy, shared memory usage, mainloop flow
   - Note which dtype/arch it targets (many examples are fp16/bf16-specific)
   - Check if it uses APIs tied to a specific arch (TMA → SM90+, tcgen05 → SM100)

4. **Rework for the user's workload** (do not copy-paste):
   - Change shapes, data types, tile sizes to match requirements
   - Replace compute logic (epilogue, activation fusion) as needed
   - If dtype differs (e.g., example is fp16, need fp32), expect vectorization
     and layout changes — the scalar-loop patterns in `references/` may be a
     better starting point than adapting a vectorized example
   - **Runtime wrapper must be lightweight**: `kernel_fn()` should only call
     `from_dlpack()` + the compiled kernel. Never allocate intermediate tensors,
     copy data, or re-compile per call — these belong in one-time setup
   - Apply optimizations from this skill's reference docs

   **⛔ Blackwell/Hopper GEMM + extra tensors — STOP:**
   If the target GPU is SM90+ (Hopper/Blackwell) **and** the GEMM requires
   extra tensors beyond A, B, C in the epilogue (e.g., bias vector, activation
   inputs), **do not attempt it**. These examples use TMA descriptors for all
   data movement — adding tensors requires modifying TMA descriptor setup,
   which is prohibitively complex. Instead, tell the user this limitation and
   suggest a **two-kernel approach**: run the GEMM kernel as-is, then apply
   bias + activation in a separate element-wise kernel (Workflow 1).
   Plain GEMM (just A×B→C with scalar alpha/beta) on Hopper/Blackwell is fine.

5. **Validate and benchmark** using companion scripts:
   ```bash
   python scripts/verify_kernel.py kernel.py --rtol 1e-3 --atol 1e-3
   python scripts/benchmark_kernel.py kernel.py
   ```
   The kernel file must export `kernel_fn`, `reference_fn`, and `get_inputs()`.

**When to skip examples:** Pure element-wise operations (Workflow 1) have
complete patterns in `references/patterns-elementwise.md` — no need to fetch
external examples.

**Reduction kernels** (softmax, layernorm, RMSNorm): Use
`references/patterns-reduction.md` which provides complete, proven patterns
for float32 reductions using scalar loops + butterfly shuffle + shared memory.

### Workflow 1: Element-wise Kernel

For unary/binary/in-place operations that map inputs to outputs 1:1.

1. **Determine kernel structure**: inputs/outputs count, tensor rank, target arch
2. **Select pattern** from `references/patterns-elementwise.md` (Variations A–E)
3. **Write kernel** applying all four invariant principles:
   - P1: `from_dlpack(tensor, assumed_align=16)` for vector loads
   - P2: Derive `vec_size` from `element_type.width`
   - P3: `cute.zipped_divide(mA, tiler)` for coalesced access
   - P4: `cutlass.dynamic_expr(thread_idx < total)` for bounds
4. **Critical rules**: No early return, no `a * 2` (use `a + a`), no `cute.math.sigmoid`
5. **Pre-compile with `cute.compile()`**: Always pre-compile the kernel once
   using `cute.compile()` so that `kernel_fn` calls the compiled object, not
   `@cute.jit` directly.  Without pre-compilation, every call recompiles
   (~20-50ms overhead).  Use `.mark_layout_dynamic()` so a single compiled
   kernel handles arbitrary input shapes without recompilation:
   ```python
   # Compile once with dynamic layouts — works for any shape
   fake_x = from_dlpack(torch.empty(1, 1, dtype=torch.float16, device="cuda"),
                         assumed_align=16).mark_layout_dynamic()
   fake_out = from_dlpack(torch.empty(1, 1, dtype=torch.float16, device="cuda"),
                           assumed_align=16).mark_layout_dynamic()
   compiled_kernel = cute.compile(host_fn, fake_x, fake_out)

   def kernel_fn(x):
       out = torch.empty_like(x)
       compiled_kernel(from_dlpack(x, assumed_align=16).mark_layout_dynamic(),
                       from_dlpack(out, assumed_align=16).mark_layout_dynamic())
       return out
   ```
6. **Verify correctness** using companion script:
   ```bash
   python scripts/verify_kernel.py kernel.py --rtol 1e-3 --atol 1e-3
   ```
   The kernel file must export `kernel_fn`, `reference_fn`, and `get_inputs()`.
7. **Benchmark** using companion script:
   ```bash
   python scripts/benchmark_kernel.py kernel.py
   ```

### Workflow 2: GEMM Kernel

For matrix multiplication with tiling, shared memory, and tensor cores.

1. **Define problem**: shapes (M, N, K), data types, target architecture
2. **Choose tiling**: CTA tile (bM, bN, bK), pipeline stages, cluster shape
3. **Three-level partitioning** (see `references/patterns-gemm.md`):
   - Level 1: CTA tiling with `local_tile()`
   - Level 2: Copy partitioning (global → shared) with `TiledCopy`
   - Level 3: Compute partitioning (shared → register) with `TiledMMA`
4. **Shared memory**: Use swizzled layouts (`make_smem_layout_atom`) to avoid bank conflicts
5. **Mainloop**: K-tile loop with copy → sync → MMA → sync
6. **Pipeline**: Use `PipelineTmaAsync` (Hopper) or `PipelineTmaUmma` (Blackwell).
   ⚠️ TMA-based pipelines manage data movement via TMA descriptors — adding
   extra tensors (bias, activation inputs) to the epilogue requires modifying
   descriptor setup, which is prohibitively complex. See the stop condition in
   Workflow 0 step 4.
7. **Epilogue**: Predicated store with alpha/beta scaling
8. **Pre-compile with `cute.compile()`**: Always pre-compile the GEMM kernel
   so `kernel_fn` calls the compiled object, not `@cute.jit` directly.
   Without pre-compilation, every call recompiles (~20-50ms overhead).
9. **Autotune**: Search over tile sizes, cluster shapes, pipeline depths

### Workflow 3: Framework Integration

For wrapping CuTe DSL kernels as PyTorch/JAX custom operators.

1. **Write kernel** using Workflow 1 or 2
2. **Create wrapper**: Accept `torch.Tensor`, convert via `from_dlpack`, call host fn
3. **For production**: Compile with TVM FFI for zero-overhead tensor passing:
   ```python
   compiled = cute.compile(host_fn, *fake_tensors, options="--enable-tvm-ffi")
   compiled(torch_a, torch_b)  # Direct torch.Tensor, no from_dlpack
   ```
4. **For deployment**: Use AOT compilation → export to `.o` → load at runtime

### Workflow 4: Debugging & Profiling

1. **Set environment**: `CUTE_DSL_PRINT_IR=1`, `CUTE_DSL_KEEP_PTX=1`
2. **Use `cute.printf()`** for runtime values (not Python `print`)
3. **Inspect generated code**: `compiled.__ptx__`, `compiled.__mlir__`
4. **Profile**: Enable `CUTE_DSL_LINEINFO=1`, use Nsight Compute/Systems
5. **Debug memory**: Run with `compute-sanitizer python script.py`

## Output Formats

A typical CuTe DSL kernel project:

```
kernel_dir/
  kernel.py          # @cute.kernel + @cute.jit functions
  test_kernel.py     # Correctness test vs PyTorch reference
  bench_kernel.py    # Benchmark with cute.compile() setup
```

**Success indicators:**
- Correctness test passes (`torch.testing.assert_close`)
- Nsight shows vector loads (LDG.128/LDG.256), not scalar loads
- For GEMM: tensor core utilization > 80% in Nsight Compute

## Companion Script Contract

Kernel files used with `scripts/verify_kernel.py` and `scripts/benchmark_kernel.py`
must export three names:

- `kernel_fn(*inputs)` — the CuTe DSL kernel wrapper (calls `cute.compile` + runs kernel)
- `reference_fn(*inputs)` — PyTorch reference implementation (same signature)
- `get_inputs()` — returns a list of CUDA tensors for testing

```python
# Example kernel.py contract
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

def kernel_fn(x):
    out = torch.empty_like(x)
    # ... call compiled cute kernel ...
    return out

def reference_fn(x):
    return torch.nn.functional.gelu(x)

def get_inputs():
    return [torch.randn(1024, 512, dtype=torch.float16, device="cuda")]
```

## Examples

### Example: 2D Unary Element-wise (ReLU)

```python
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def relu_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    total = m * n
    if cutlass.dynamic_expr(idx < total):
        a = gA[(None, (idx // n, idx % n))].load()
        gC[(None, (idx // n, idx % n))] = cute.where(a > 0, a, 0)

@cute.jit
def relu_host(mA: cute.Tensor, mC: cute.Tensor):
    vec = 16 // (mA.element_type.width // 8)
    gA = cute.zipped_divide(mA, (1, vec))
    gC = cute.zipped_divide(mC, (1, vec))
    T = 256
    N = cute.size(gA.shape[1])
    relu_kernel(gA, gC).launch(grid=((N+T-1)//T,1,1), block=(T,1,1))

x = torch.randn(1024, 512, dtype=torch.float16, device="cuda")
out = torch.empty_like(x)
relu_host(from_dlpack(x, assumed_align=16), from_dlpack(out, assumed_align=16))
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `MLIR function requires a Context` | Called @kernel from Python | Launch via @cute.jit host function |
| `DSLAstPreprocessorError` on return | Early return in @kernel | Use `if cutlass.dynamic_expr(cond):` |
| Type mismatch on store | `a * 2` promotes FP16→FP32 | Use `a + a` or `.to(cutlass.Float16)` |
| `could not get source code` | Kernel in `exec()` context | Write to file and import |
| Scalar loads in Nsight | Missing alignment hint | Add `assumed_align=16` to `from_dlpack` |
| `Missing required argument` | Not all @jit params passed | Pass ALL declared parameters |
| `AttributeError: sigmoid` | No `cute.math.sigmoid` | Use `1.0/(1.0+cute.math.exp(-x))` |

See `references/troubleshooting.md` for the full error table and limitations.

**Debugging rule:** Never delete kernel.py during debugging. Use `backup_file`
to save a checkpoint, then `edit_file` to iterate. If stuck, `revert_file` to
restore the backup. A partially-working kernel is always better than no kernel.

## Finding More Information

### Tier 1: This File (SKILL.md)

Workflows above cover element-wise kernels, GEMM, framework integration, and
debugging. Search this file first for procedural questions.

### Tier 2: references/ Directory

Grep for keywords across `references/`. Headers are grep-friendly.

| File | Content |
|------|---------|
| `concepts-architecture.md` | Core abstractions, terminology, compilation pipeline |
| `concepts-layouts.md` | Layout algebra: composition, complement, divide, swizzle |
| `concepts-tensors.md` | Tensor types, partitioning, tiling, predication |
| `concepts-mma.md` | MMA atoms, TiledMMA, per-architecture tensor core ops |
| `patterns-getting-started.md` | Installation, decorators, first kernel walkthrough |
| `patterns-elementwise.md` | Invariant principles, pattern variations, reference impl |
| `patterns-gemm.md` | 3-level tiling, shared memory, pipelining, autotuning |
| `patterns-memory.md` | from_dlpack, TMA, cp.async, TMEM, copy atoms |
| `patterns-compilation.md` | Control flow, JIT caching, TVM FFI, AOT compilation |
| `patterns-pipeline.md` | Producer-consumer, pipeline classes, barriers, warp specialization |
| `api-core.md` | cute module: layouts, tensors, math, copy, gemm, printing |
| `api-arch.md` | cute.arch: thread indexing, sync, atomics, memory ops |
| `api-nvgpu.md` | cute.nvgpu: warp/warpgroup/cpasync/tcgen05 MMA and copy |
| `api-runtime-utils.md` | Runtime: from_dlpack, fake tensors, utils, schedulers |
| `troubleshooting.md` | Debugging, env vars, common errors, limitations, FAQ |

**How to search:** Grep for your keyword across `references/`. Read only the
file and section that Grep points to.

### Tier 3: Original Documentation

If Tiers 1–2 don't answer, consult the source:
- **Web**: https://docs.nvidia.com/cutlass/latest/
- **GitHub**: https://github.com/NVIDIA/cutlass
- Fetch specific doc pages or search for "CUTLASS CuTe DSL <topic>"
- Consider distilling the answer back into references/
