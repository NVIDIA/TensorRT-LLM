---
name: kernel-tileir-optimization
description: >
  Optimize existing Triton kernels for NVIDIA TileIR backend on Blackwell GPUs (sm_100+).
  Adds TileIR-specific autotune configs: occupancy, num_ctas, TMA descriptors. Covers
  kernel classification (dot-related, norm-like, elementwise, reduction), type-specific
  transformations, and PTX-vs-TileIR benchmarking. Triggered by: "optimize for TileIR",
  "add TileIR configs", "Blackwell optimization", "TMA descriptors", "2CTA mode",
  "occupancy tuning". Kernels use standard `import triton`; TileIR activates via
  ENABLE_TILE=1 when nvtriton is installed.
compatibility: >
  Requires Blackwell GPU (sm_100+) for TileIR execution. Supports development on any GPU.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Triton TileIR Optimization

Optimize EXISTING Triton kernels for NVIDIA's TileIR backend on Blackwell GPUs.
This skill does NOT write kernels from scratch -- that is the Triton Specialist's job.

## Principles

### TileIR vs PTX Backend

TileIR is NVIDIA's compiler backend for Triton that generates optimized CUDA code
using CGA-level (Cooperative Grid Array) tile representations. Critical differences:

| Parameter | PTX Backend | TileIR Backend |
|-----------|-------------|----------------|
| `num_warps` | Strict directive | **Ignored** (compiler decides) |
| `num_stages` | Strict directive | Cost hint (compiler optimizes) |
| `occupancy` | Not available | **Critical** tuning param (1-32) |
| `num_ctas` | Limited | 2CTA mode for Blackwell |
| Block sizes | Smaller often better | Larger often better |
| TMA | Not available | Required for dot kernels |

**Key implication**: Do not tune `num_warps` for TileIR -- focus on `occupancy` instead.

### Triton Package Landscape

Three packages share `import triton`:

| Package | Source | Use Case |
|---------|--------|----------|
| `pytorch-triton` | PyTorch wheel | `torch.compile`, standard kernels |
| `triton` | OpenAI PyPI | Official Triton from triton-lang.org |
| nvtriton | [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR) | TileIR backend for Blackwell |

Only one triton package should be installed at a time. "Converting to TileIR" means
adding TileIR-specific configs, NOT changing imports. TileIR activates via `ENABLE_TILE=1`.

### When TileIR Applies

TileIR targets Blackwell (sm_100+). Without nvtriton or Blackwell hardware, the
specialist still adds TileIR-optimized configs that standard triton safely ignores,
enabling future deployment.

**Expected speedups** (with nvtriton on Blackwell):

| Kernel Type | Speedup | Key Lever |
|-------------|---------|-----------|
| Dot-Related (GEMM, Attention) | 1.2-2.0x | TMA + 2CTA |
| Norm-Like (LayerNorm, Softmax) | 2.0-5.0x | High occupancy |
| Element-Wise (ReLU, Add, Exp) | 1.5-3.0x | Occupancy + num_stages |
| Reduction (Sum, Mean, Max) | 1.8-4.0x | High occupancy |

## Workflow

Five-phase workflow: compatibility, classify, transform, validate, benchmark.

### Phase 1: Compatibility Test (ENABLE_TILE=0)

Verify the kernel works in PTX mode before applying TileIR optimizations.

```bash
python scripts/tileir_check.py
```

Then use the **kernel-triton-writing** skill's `verify_kernel.py` to verify with `ENABLE_TILE=0`:

```bash
python scripts/verify_kernel.py --kernel path/to/kernel.py --reference 'torch reference' --shapes '{"x": [32, 512, 4096]}' --dtypes '{"x": "bfloat16"}'
```

### Phase 2: Classify Kernel

Determine kernel type to select the optimization strategy.

```bash
python scripts/classify_kernel.py --file kernel.py
```

Classification decision tree:

```
Contains tl.dot()?
  YES --> dot-related: TMA + 2CTA + occupancy + larger blocks
  NO  --> Has reduction + normalization?
            YES --> norm-like: high occupancy (2, 4) + num_warps (4, 8)
            NO  --> Point-wise only?
                      YES --> element-wise: occupancy (1-16) + num_stages (2-4)
                      NO  --> reduction: high occupancy + num_warps
```

### Phase 3: Apply Transformations

Classify and apply optimizations in one step:

```bash
python scripts/classify_kernel.py --file kernel.py --apply-optimizations
```

Output JSON includes `optimized_code` and `changes_applied` fields.

**Type-specific transformations:**

**Dot-related** (highest priority):
1. Convert `tl.load`/`tl.store` to TMA descriptors (MANDATORY). See `references/tma-conversion.md`.
2. Add 2CTA configs (`num_ctas=2`) with SM oversubscription guard in pre-hook.
3. Add occupancy (1, 2, 4) and extended num_stages (4, 6).
4. Use larger block sizes (256x256, 256x128).

**Norm-like** (LayerNorm, Softmax, RMSNorm):
- Add occupancy (2, 4), num_warps (4, 8). No TMA needed.

**Element-wise** (ReLU, GELU, Add, Mul, Exp):
- Add occupancy (1, 2, 4, 16), num_stages (2, 3, 4). Include extreme configs for small inputs.

**Reduction** (Sum, Mean, Max):
- Same strategy as norm-like: high occupancy (2, 4), num_warps (4, 8).

Gate TileIR-specific configs for sm_100+:

```python
import torch

def get_configs_with_gating(pre_hook=None):
    configs = get_baseline_configs()
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        configs.extend(get_tileir_specific_configs(pre_hook))
    return configs
```

See `references/config-templates.md` for complete config templates per kernel type.

### Phase 4: TileIR Validation (ENABLE_TILE=1)

Use the **kernel-triton-writing** skill's `verify_kernel.py` to verify the optimized kernel with TileIR backend:

```bash
python scripts/verify_kernel.py --kernel path/to/optimized_kernel.py --reference 'torch reference' --shapes '{"x": [32, 512, 4096]}' --dtypes '{"x": "bfloat16"}'
```

Set `ENABLE_TILE=1` before running. Check: numerical correctness, no compilation errors,
TMA/2CTA patterns compile successfully.

### Phase 5: Benchmark

Use `triton.testing.do_bench()` (as documented in the **perf-workload-profiling** skill) to compare PTX (`ENABLE_TILE=0`) vs TileIR (`ENABLE_TILE=1`).

Benchmark across multiple input sizes (128, 1024, 8192) -- performance varies by size.

## Scripts

### tileir_check.py

Check TileIR availability (nvtriton, ENABLE_TILE, Blackwell GPU):

```bash
python scripts/tileir_check.py
```

Returns JSON: `nvtriton_installed`, `tileir_active`, `blackwell_gpu`, `gpu_capability`, `recommendation`.

### classify_kernel.py

Classify kernel type and optionally apply TileIR optimizations:

```bash
# Classify only
python scripts/classify_kernel.py --file kernel.py

# Classify + apply optimizations
python scripts/classify_kernel.py --file kernel.py --apply-optimizations

# From inline code
python scripts/classify_kernel.py --code '<kernel_code>'
```

Returns JSON: `classification`, `confidence`, `indicators`, `recommendations`.
With `--apply-optimizations`: adds `optimized_code` and `changes_applied`.

## Error Handling

### Common Pitfalls

**TMA descriptor errors** (dot-related kernels):
- Always pass `pre_hook=tma_set_block_size_hook` to config generation -- without it,
  TMA descriptors keep dummy block sizes, causing runtime errors or wrong results.
- For GEMM: pass `b.T.contiguous()` in wrapper and use `tl.dot(a, b.T, accumulator)`
  in kernel. Transposition mismatch produces incorrect results silently.

**2CTA oversubscription**:
- Adjust SM count in pre-hook when using `num_ctas=2`:
  ```python
  if "NUM_SMS" in nargs and "NUM_CTAS" in nargs:
      nargs["NUM_SMS"] = nargs["NUM_SMS"] // nargs["NUM_CTAS"]
  ```

**Config function signatures**:
- ALL config helper functions MUST accept `pre_hook=None`, even if unused.
  Without it: `TypeError: get_autotune_configs() takes 0 positional arguments`.

**Hardware gating**:
- Gate TileIR configs with `torch.cuda.get_device_capability()[0] >= 10`.
  TMA/2CTA on pre-Blackwell GPUs causes runtime crashes.

**API availability**:
- Use `1.0 / (1.0 + tl.exp(-x))` instead of `tl.sigmoid(x)` -- not available in
  all Triton versions including some nvtriton builds.

**Performance tuning**:
- Do not over-tune `num_warps` -- TileIR ignores it. Focus on `occupancy`.
- Use larger block sizes (256x256, 256x128) for TileIR, not PTX-tuned small blocks.
- Benchmark across small/medium/large inputs; one-size configs underperform.
- For exp/log heavy kernels, enable approximate math:
  ```bash
  export TILEIR_ENABLE_APPROX=1
  export TILEIR_ENABLE_FTZ=1
  ```

### When to Abort

Stop and report if:

1. **No triton installed** -- cannot proceed.
2. **Compatibility test fails** -- kernel has syntax/runtime errors before optimization.
3. **TileIR validation fails** -- optimized kernel produces wrong results.
4. **No speedup** -- TileIR version is slower than PTX baseline (with nvtriton).
5. **Not Blackwell GPU** -- still add configs for future deployment, but skip
   ENABLE_TILE testing and benchmarking.

### Output Format

After optimization, return:

```
## TileIR Optimization: kernel_name

### Classification
- Kernel type: [dot-related | norm-like | element-wise | reduction]
- Strategy: [TMA + 2CTA | High occupancy | Occupancy + num_stages]

### Compatibility Check (ENABLE_TILE=0)
[PASSED | FAILED] — Max difference: X.Xe-Y

### Transformations Applied
- [List of transformations]

### TileIR Validation (ENABLE_TILE=1)
[PASSED | FAILED] — Max difference: X.Xe-Y

### Benchmark Comparison
| Backend | Time (ms) | Speedup |
|---------|-----------|---------|
| PTX (ENABLE_TILE=0) | X.XXX | 1.0x |
| TileIR (ENABLE_TILE=1) | X.XXX | Y.Yx |

### Output
File: kernel_name_tileir.py
```
