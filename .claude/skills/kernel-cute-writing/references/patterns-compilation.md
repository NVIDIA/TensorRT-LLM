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

# Compilation & Deployment Patterns

## Code Generation Modes

CuTe DSL has two code generation techniques, combined via the `preprocessor`
flag on `@cute.jit` / `@cute.kernel`:

### Preprocessor Mode (Default, `preprocessor=True`)

Hybrid: AST rewrite handles control flow, then tracing handles arithmetic.

1. AST rewrite converts `for`/`while`/`if` into structured IR
2. Tracer records tensor operations via proxy objects
3. All branches and loops preserved (no correctness issues)

Use for: Most kernels. Captures complete program semantics.

### Tracing Mode (`preprocessor=False`)

Pure tracing: execution with proxy arguments records operations.

- Fastest compilation
- Only captures executed branches (unexecuted branches disappear)
- Loops collapse to observed iteration counts
- Suitable only for straight-line arithmetic

## Static vs Dynamic Control Flow

### For Loops

| Construct | Behavior |
|-----------|----------|
| `for i in range(n)` | IR loop (dynamic, runtime) |
| `for i in cutlass.range(n)` | IR loop + optional unroll/pipeline |
| `for i in cutlass.range_constexpr(n)` | Python-unrolled (compile-time) |

```python
# Runtime loop
for i in range(N):
    cute.printf("%d\n", i)

# Unrolled at compile time (n must be Constexpr)
for i in cutlass.range_constexpr(n):
    process(data[i])

# Software pipelining
for i in cutlass.range(bound, prefetch_stages=stages):
    cute.copy(atom, gmem[i], buffer[i % total_stages])
    use(buffer[i % total_stages])
```

### Conditionals

```python
# Dynamic (generates IR branch, runtime evaluation)
if dynamic_var == 10:
    cute.printf("Dynamic\n")

# Compile-time (evaluated by Python, dead branch eliminated)
if cutlass.const_expr(const_var):
    cute.printf("Compile-time\n")

# Kernel specialization pattern
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    if cutlass.const_expr(do_relu):
        # ReLU code emitted only when True
```

### Limitations of Dynamic Control Flow

- No `break`, `continue`, or early `return`
- Variables originating in control-flow bodies unavailable outside scope
- Variable types cannot change within control flow
- No exception handling

## JIT Compilation with cute.compile

Pre-compile to eliminate JIT overhead on subsequent calls:

```python
# Basic compilation
compiled_fn = cute.compile(host_fn, arg1, arg2)
compiled_fn(arg1, arg2)  # No JIT overhead

# With Constexpr — baked into compilation
compiled_fn = cute.compile(host_fn, tensor, True)  # True is Constexpr
```

The compiled `JitExecutor` maintains:
- Host function pointer with MLIR execution engine
- Optional CUDA modules
- Argument specifications (excluding Constexpr values)

## JIT Caching

### Implicit (Automatic)

Default in-memory caching. Cache key combines hashes of:
- MLIR bytecode
- Python source files
- Shared libraries
- Environment variables

File persistence: `/tmp/{user}/cutlass_python_cache/`

```bash
# Configuration
export CUTE_DSL_CACHE_DIR=/path/to/persistent/cache
export CUTE_DSL_DISABLE_FILE_CACHING=True  # Disable file cache
```

### Custom Caching

```python
kernel_cache = {}

def get_or_compile(config, *args):
    key = f"{config.dtype}x{config.tile}x{config.stages}"
    if key not in kernel_cache:
        kernel_cache[key] = cute.compile(kernel_fn, *args)
    return kernel_cache[key]
```

## TVM FFI Compilation

Enables direct `torch.Tensor` passing without DLPack conversion overhead.

### Setup
```bash
pip install apache-tvm-ffi
pip install torch-c-dlpack-ext  # Optional, improves performance
```

### Compilation with Fake Tensors

```python
# Create symbolic tensors for compilation
n = cute.sym_int()
a_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
b_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

# Compile with TVM FFI
compiled = cute.compile(kernel_fn, a_fake, b_fake,
                        options="--enable-tvm-ffi")

# Call with real torch tensors directly
a = torch.randn(1024, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
compiled(a, b)  # No from_dlpack needed
```

### Environment Stream (Recommended)

```python
stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
compiled = cute.compile(fn, a_fake, b_fake, stream,
                        options="--enable-tvm-ffi")
compiled(a, b)  # Uses current CUDA stream automatically
```

### Enable via Environment

```bash
export CUTE_DSL_ENABLE_TVM_FFI=1
```

## AOT (Ahead-of-Time) Compilation

Compile once, eliminate JIT overhead in production, enable cross-compilation.

### Export

```python
compiled = cute.compile(kernel_fn, *args)
compiled.export_to_c(
    file_path="./artifacts",
    file_name="my_kernel",
    function_prefix="my_kernel",
)
# Generates: my_kernel.h, my_kernel.o
```

### Load in Python

```python
module = cute.runtime.load_module("./artifacts/my_kernel.o")
module.my_kernel(tensor_a, stream)
```

### Load in C++ (Static Linking)

```cpp
#include "my_kernel.h"
my_kernel_Kernel_Module_t module;
my_kernel_Kernel_Module_Load(&module);
cute_dsl_my_kernel_wrapper(&module, &tensor, stream);
my_kernel_Kernel_Module_Unload(&module);
```

### Load in C++ (Dynamic)

```cpp
#include "CuteDSLRuntime.h"
CuteDSLRT_Module_t *module = nullptr;
CuteDSLRT_Module_Load(&module, "./libmy_kernel.so");
CuteDSLRT_Function_t *func = nullptr;
CuteDSLRT_Module_Get_Function(&func, module, "my_kernel");
void* args[] = {&tensor, &stream};
CuteDSLRT_Function_Run(func, args, 2);
```

### Module Export for Shared Libraries

```python
compiled.export_to_c("./kernel.o", function_name="kernel")
runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
# Link kernel.o + runtime_libs → shared library
module = cute.runtime.load_module("./kernel.so", enable_tvm_ffi=True)
```

## Framework Integration Patterns

### PyTorch Custom Operator

```python
@cute.jit
def my_op(mA, mC):
    # kernel implementation
    ...

def torch_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    my_op(from_dlpack(x, assumed_align=16),
          from_dlpack(out, assumed_align=16))
    return out
```

### JAX Integration

JAX tensors support DLPack and can be passed to CuTe DSL functions.
JAX 0.8.1+ with CUDA support recommended.
