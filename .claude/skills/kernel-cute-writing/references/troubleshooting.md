<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Troubleshooting & Limitations

## Debugging Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUTE_DSL_LOG_TO_CONSOLE=1` | Enable console logging | Debug output |
| `CUTE_DSL_LOG_TO_FILE=file.txt` | Log to file | Persistent logs |
| `CUTE_DSL_LOG_LEVEL=10` | Verbosity (0=off, 10=debug, 50=critical) | Detail control |
| `CUTE_DSL_DUMP_DIR=path` | Directory for dumped files | IR/PTX output |
| `CUTE_DSL_PRINT_IR=1` | Display MLIR IR | Code inspection |
| `CUTE_DSL_KEEP_IR=1` | Preserve IR files | Post-mortem analysis |
| `CUTE_DSL_KEEP_PTX=1` | Save PTX assembly | Performance tuning |
| `CUTE_DSL_KEEP_CUBIN=1` | Save binary cubin | SASS analysis |
| `CUTE_DSL_LINEINFO=1` | Python-to-PTX/SASS correlation | Profiler mapping |

## Runtime Debugging

### Printing

```python
# Compile-time only (Python print) — shows static values
print(f"layout: {tensor.layout}")

# Runtime GPU printf — executes on device, adds PTX overhead
cute.printf("thread %d: val = %f\n", thread_idx, value)
```

### Inspecting Generated Code

```python
compiled = cute.compile(kernel_fn, *args)
print(compiled.__ptx__)    # PTX assembly
print(compiled.__mlir__)   # MLIR IR
with open("kernel.cubin", "wb") as f:
    f.write(compiled.__cubin__)
```

SASS disassembly:
```bash
nvdisasm kernel.cubin > kernel.sass
```

### Custom Kernel Names for Profiling

```python
@cute.kernel
def my_kernel(gA, gC):
    ...

@cute.jit
def host():
    my_kernel.set_name_prefix("my_custom_prefix")
    my_kernel(gA, gC).launch(...)
```

### Compute Sanitizer

```bash
compute-sanitizer python script.py
```

### Handling Unresponsive Kernels

1. Press `Ctrl+Z` to suspend
2. `kill -9 $(jobs -p | tail -1)`

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: An MLIR function requires a Context` | Called `@cute.kernel` directly from Python | Always call through `@cute.jit` host function |
| `OSError: could not get source code` | Kernel code in `exec()` context | Write kernel to file and import as module |
| `DSLRuntimeError: Missing required argument` | Not all `@cute.jit` params passed | Pass ALL declared parameters including type args |
| `DSLAstPreprocessorError` on `return` | Early return in `@cute.kernel` | Use `if cutlass.dynamic_expr(cond):` instead |
| Type mismatch on store | Scalar multiply promoted FP16→FP32 | Use `a + a` instead of `a * 2`, or `.to(dtype)` |
| `AttributeError: cute.math.sigmoid` | No sigmoid in `cute.math` | Implement: `1.0 / (1.0 + cute.math.exp(-x))` |
| Scalar loads instead of vector loads | Missing `assumed_align` | Add `from_dlpack(t, assumed_align=16)` |
| Shape mismatch with compiled function | Static layout with wrong shape | Use `mark_layout_dynamic()` or recompile |
| `functools.lru_cache` errors | MLIR objects are context-sensitive | Use `cute.compile` with custom cache dict |
| JIT recompilation every call | No caching, or `no_cache=True` | Use `cute.compile()` and cache the executor |

## Known Limitations

### Programming Model

- **No early return**: No `return`, `break`, `continue` in dynamic control flow
- **No exception handling**: No try/except in JIT code
- **No single-stepping**: pdb cannot step through JIT code
- **Static typing**: Variable types cannot change within control flow
- **No dynamic indexing**: Cannot index lists with runtime values
- **Function returns**: Only `Constexpr` values can be returned
- **32-bit layouts only**: Shape/stride limited to 32-bit integers
- **No `_` variable reads**: Underscore cannot be read after assignment

### Data Types

- Python lists/dicts/tuples: Compile-time only, cannot modify at runtime
- `Layout` types: Cannot be passed as native Python function arguments
- OOP: Limited when objects contain dynamic values

### Platform

- Linux x86_64 only (no Windows, no ARM)
- Python 3.10–3.13
- No convolution support (GEMM only)
- No preferred cluster support

### Performance

- DLPack conversion overhead: ~2-3 μs per tensor
- Layout algebra requires JIT (cannot use in native Python)
- Implicit JIT caching may not cover all cases — use `cute.compile()` for
  deterministic behavior

## Performance Tips

1. **Use `cute.compile()`**: Pre-compile kernels, cache executors
2. **Use `assumed_align=16` (or 32)**: Enable vector loads/stores
3. **Use TVM FFI**: Eliminate DLPack overhead in production
4. **Cache JIT results**: Set `CUTE_DSL_CACHE_DIR` for persistent cache
5. **Lock GPU clocks**: `nvidia-smi -lgc <freq>` for reproducible benchmarks
6. **Use `mark_layout_dynamic()`**: Single compilation for varying shapes
7. **Profile with Nsight**: Enable `CUTE_DSL_LINEINFO=1` for source correlation
8. **Use `use_32bit_stride=True`**: Reduce register pressure for small tensors

## FAQ Highlights

**Q: Should I port C++ CUTLASS kernels to CuTe DSL?**
Almost certainly not, unless JIT compile time is critical and C++ compile
times are a blocker. C++ APIs continue receiving support.

**Q: Is CuTe DSL replacing CUTLASS C++?**
No. CuTe DSL complements C++. CUTLASS 2.x/3.x APIs continue to be maintained.

**Q: What should newcomers learn?**
Start with Python DSL — significantly lower learning curve. Knowledge
transfers to C++ since the programming models are isomorphic.

**Q: Does CuTe DSL compile to PTX or SASS?**
Compiles to PTX first, then uses ptxas (from CUDA toolkit) to produce SASS.

**Q: Can I use OOP and call functions from functions?**
Yes. Class hierarchies and function composition work normally for organizing
pipeline and scheduler code.

**Q: What about portability during beta?**
No portability guarantees during beta. Breaking changes will be announced
and documented in CHANGELOG entries.
