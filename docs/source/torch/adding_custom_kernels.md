# Integrating Custom Kernels in TensorRT-LLM

This guide walks through adding or modifying GPU kernels and exposing them through the TensorRT-LLM PyTorch path. It covers source/JIT integration flows: **CUDA C++** kernels, **CuTe DSL** kernels, and **cuTile** (`cuda.tile`) kernels. The **trtllm-gen** path (with pre-built CUBIN integration) is **not** covered here.

## Table of Contents

1. [Overview](#1-overview)
2. [What defining a new op involves](#2-what-defining-a-new-op-involves)
3. [CUDA custom kernel path](#3-cuda-custom-kernel-path)
4. [CuTe DSL / cuTile JIT kernel path](#4-cute-dsl--cutile-jit-kernel-path)
5. [Example walkthrough: `indexer_k_cache_scatter_op`](#5-example-walkthrough-indexer_k_cache_scatter_op)
6. [Testing and validation](#6-testing-and-validation)
7. [Common mistakes](#7-common-mistakes)

---

## 1. Overview

A **custom kernel** is any GPU kernel that TensorRT-LLM dispatches to from the PyTorch backend. In practice, the backend never calls a kernel directly: the kernel is wrapped by a **PyTorch custom op** (a `torch.ops.trtllm.<name>` call) and the model code calls that op like any other PyTorch operator.

This guide focuses on source-level integration where the kernel ships as either:

| Flavor | Source language | When it is built | Where it lives |
| :---- | :---- | :---- | :---- |
| CUDA C++ | `.cu` / `.h` | At wheel build time (CMake → `nvcc`) | [`cpp/tensorrt_llm/kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels) |
| [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) (Python) | Python using `cutlass.cute` / `@cute.kernel` | JIT, on first call (cached in-process) | [`tensorrt_llm/_torch/cute_dsl_kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/cute_dsl_kernels) |
| [cuTile](https://docs.nvidia.com/cuda/cutile-python/) (Python) | Python using `cuda.tile` / `@ct.kernel` | JIT, on first call (cached in-process) | [`tensorrt_llm/_torch/cuda_tile_kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/cuda_tile_kernels) |

---

## 2. What defining a new op involves

Adding a new `torch.ops.trtllm.<name>` means producing four things:

- **The kernel** — the kernel source code itself, in CUDA C++ or a Python DSL (CuTe / cuTile).  
- **The binding** — a thin C++ or Python wrapper that registers the kernel as a Torch op, validates inputs, and launches it on the right stream.  
- **The integration** — `torch.ops.trtllm.<name>(...)` from the existing forward path.  
- **The tests** — a unit test that compares the op output against a PyTorch reference.

Sections 3 and 4 cover the kernel and binding; section 6 covers tests; section 5 walks through a complete example. The integration path into the module should follow one principle — **minimal, in-place change**:

Identify the boundary your op sits at within the `nn.Module`, then wire it into the corresponding call site with a fallback. If an existing op operates at a similar boundary, follow its integration path. Either way, keep the change minimal.

---

## 3. CUDA custom kernel path

This is the path to take when you want a CUDA C++ kernel that is compiled into the TRT-LLM wheel. **Before you start, scan the existing kernels.** Look under [`cpp/tensorrt_llm/kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels) for anything close to what you need.

### 3.1 Kernel source

Place the kernel under [`cpp/tensorrt_llm/kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels). The convention is one header (`MyKernel.h`) and one source file (`myKernel.cu`):

```cpp
// cpp/tensorrt_llm/kernels/MyKernel.h
#pragma once
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN
namespace kernels {

void invokeMyKernel(/* typed pointers, dims, strides */ cudaStream_t stream = 0);

}
TRTLLM_NAMESPACE_END
```

```cpp
// cpp/tensorrt_llm/kernels/myKernel.cu
#include "MyKernel.h"
#include "tensorrt_llm/common/assert.h"

TRTLLM_NAMESPACE_BEGIN
namespace kernels {

namespace {
__global__ void myKernelImpl(/* ... */) { /* ... */ }
}

void invokeMyKernel(/* ... */ cudaStream_t stream) {
    // shape/dtype assertions via TLLM_CHECK_WITH_INFO
    dim3 block(/* ... */);
    dim3 grid(/* ... */);
    myKernelImpl<<<grid, block, 0, stream>>>(/* ... */);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

}
TRTLLM_NAMESPACE_END
```

The top-level kernels [`CMakeLists.txt`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/CMakeLists.txt) already globs `*.cu` and `*.cpp` in this directory, so a new file is picked up automatically. (Subdirectories with their own `add_subdirectory(...)` are excluded from the glob — if you create one, register it in the parent `CMakeLists.txt` like the existing entries.)

### 3.2 Torch op binding

Bindings live under [`cpp/tensorrt_llm/thop/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/thop). A binding is a single `.cpp` file that:

1. Wraps the kernel launcher in a function that takes `torch::Tensor` arguments.  
2. Validates shapes, dtypes, devices, and contiguity.  
3. Registers the op into the `trtllm` `torch.library` namespace and provides a CUDA implementation.

```cpp
// cpp/tensorrt_llm/thop/MyKernelOp.cpp
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/kernels/MyKernel.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN
namespace torch_ext {

void my_kernel_op(th::Tensor const& x, th::Tensor& out /*, ... */) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    // ... more validation ...

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tk::invokeMyKernel(/* ... */, stream);
}

} // namespace torch_ext
TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m) {
    // Schema string. Use Tensor(a!) for in-place / mutated outputs.
    m.def("my_kernel_op(Tensor x, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m) {
    m.impl("my_kernel_op", &tensorrt_llm::torch_ext::my_kernel_op);
}
```

Then add the new file to the `th_common` library in [`cpp/tensorrt_llm/thop/CMakeLists.txt`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/CMakeLists.txt) (it is **not** globbed; new entries are listed explicitly).

After rebuilding the wheel, the op is callable from Python as:

```py
torch.ops.trtllm.my_kernel_op(x, out)
```

### 3.3 Calling the op from Python

From the PyTorch backend (e.g., a module forward, an attention backend, a metadata helper), call the op directly:

```py
torch.ops.trtllm.my_kernel_op(x, out)
```

If your op returns tensors and you want it to be `torch.compile`/FakeTensor friendly, also register a *fake/meta* implementation so shape inference works. For C++-defined ops, fakes are typically registered together in [`tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py) inside the `_register_fake()` function, using `@torch.library.register_fake("trtllm::my_kernel_op")`.

### 3.4 Tests

Add a unit test under [`tests/unittest/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests/unittest). For an op that touches an attention backend the natural location is `tests/unittest/_torch/...`. Write a Python reference, run both, and compare with `torch.testing.assert_close` (or a stricter exact comparison if the kernel is deterministic and integer-typed).

---

## 4. CuTe DSL / cuTile JIT kernel path

CuTe DSL and cuTile kernels are written in Python and JIT-compiled at runtime. They never appear in the C++ build — they are imported from `_torch/` and launched directly.

### 4.1 Where they live

| Flavor | Directory | Availability flag |
| :---- | :---- | :---- |
| CuTe DSL | [`tensorrt_llm/_torch/cute_dsl_kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/cute_dsl_kernels) (Blackwell variants under `blackwell/`) | `IS_CUTLASS_DSL_AVAILABLE` in [`cute_dsl_utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cute_dsl_utils.py) |
| cuTile | [`tensorrt_llm/_torch/cuda_tile_kernels/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/cuda_tile_kernels) | `IS_CUDA_TILE_AVAILABLE` in [`cuda_tile_utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cuda_tile_utils.py) |

### 4.2 Kernel skeleton

**CuTe DSL.** The kernel is a class (or a free function) decorated with `@cute.jit` for host code and `@cute.kernel` for the device entry point. Compilation is cached by the call site and reused across launches. See [`cute_dsl_kernels/argmax.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cute_dsl_kernels/argmax.py) for a self-contained reduction example, and [`cute_dsl_kernels/blackwell/dense_gemm_persistent.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_gemm_persistent.py) for a GEMM with TMA + persistent scheduling.

```py
# tensorrt_llm/_torch/cute_dsl_kernels/my_kernel.py
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    class MyKernel:
        @cute.jit
        def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
            self.kernel(mX, mO).launch(grid=[...], block=[...], stream=stream)

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mO: cute.Tensor):
            # device-side code
            ...

    _compile_cache = {}

    def my_kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        key = (x.dtype, x.shape[-1])
        if key not in _compile_cache:
            _compile_cache[key] = cute.compile(
                MyKernel(), from_dlpack(x), from_dlpack(out),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        _compile_cache[key](
            from_dlpack(x), from_dlpack(out),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out
```

**cuTile.** The kernel is a Python function decorated with `@ct.kernel` that takes tensors plus `ct.Constant[...]` compile-time parameters, and is launched via `ct.launch(stream, grid, kernel, args)`. See [`cuda_tile_kernels/rms_norm.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cuda_tile_kernels/rms_norm.py) and the corresponding Torch op wrapper in [`custom_ops/cuda_tile_custom_ops.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/cuda_tile_custom_ops.py).

```py
# tensorrt_llm/_torch/cuda_tile_kernels/my_kernel.py
from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    import cuda.tile as ct

    @ct.kernel
    def my_kernel(x, y, TILE: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(TILE,))
        ct.store(y, index=(i,), tile=xi * 2.0)
```

### 4.3 Wrapping the JIT kernel as a Torch custom op

For both flavors, register a Python custom op so the rest of the codebase calls it through `torch.ops.trtllm.<name>(...)` exactly like a C++ op. Define the op in a guarded block under [`tensorrt_llm/_torch/custom_ops/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/custom_ops) and re-export it from the package [`__init__.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/__init__.py).

```py
# tensorrt_llm/_torch/custom_ops/cuda_tile_custom_ops.py (excerpt)
from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    import cuda.tile as ct
    from ..cuda_tile_kernels import rms_norm_kernel  # the JIT kernel above

    @torch.library.custom_op("trtllm::cuda_tile_rms_norm", mutates_args=())
    def cuda_tile_rms_norm(x, weight, eps, ...):
        y = torch.empty_like(x)
        ct.launch(torch.cuda.current_stream(), (x.shape[0],),
                  rms_norm_kernel, (x, weight, y, ...))
        return y

    @cuda_tile_rms_norm.register_fake
    def _(x, weight, eps, ...):
        return torch.empty_like(x.contiguous())
```

The same pattern (`@torch.library.custom_op("trtllm::...")`, `register_fake`, conditional import gated on the availability flag) is used throughout [`cute_dsl_custom_ops.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py).

### 4.4 Runtime caveats

- **Compute capability checks.** Kernels under `cute_dsl_kernels/blackwell/` and the entire `cuda_tile_kernels/` tree assume sm_100+. Always probe with `tensorrt_llm._utils.get_sm_version()` (see `_should_use_torch_fallback` in `argmax.py`) and route to a fallback otherwise.  
- **DLPack/CUDA Graphs.** When exporting tensors via DLPack, use the stream override that mimics the `CUDAGraphCompatibleWrapper` in `argmax.py` so the capture replays cleanly.  
- **JIT compile cache.** Always cache compiled kernels by the keys that affect codegen (dtype, last-dim size, hardware variant). `argmax.py` and the `cute_dsl_custom_ops.py` runners are good references.

---

## 5. Example walkthrough: `indexer_k_cache_scatter_op`

`indexer_k_cache_scatter_op` scatters DeepSeek FP8 indexer K-cache entries — per-token quantized keys plus per-block scales — into a paged K-cache in a single CUDA kernel launch.

- **The kernel.** [`cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h) declares `invokeIndexerKCacheScatter(...)`; [`cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu) defines the `__global__` kernel and the host launcher. Picked up automatically by the `*.cu` glob in `cpp/tensorrt_llm/kernels/CMakeLists.txt`.  
- **The binding.** [`cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp) validates inputs, fetches the stream, and calls the launcher; registers the op via `TORCH_LIBRARY_FRAGMENT(trtllm, m)` and `TORCH_LIBRARY_IMPL(trtllm, CUDA, m)`. A one-line addition to [`cpp/tensorrt_llm/thop/CMakeLists.txt`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/CMakeLists.txt) wires it into `th_common` (this directory is not globbed).  
- **The integration.** In [`tensorrt_llm/_torch/attention_backend/sparse/dsa.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/attention_backend/sparse/dsa.py), `_update_k_cache` calls `torch.ops.trtllm.indexer_k_cache_scatter_op(...)` in place of the prior Python scatter loop.  
- **The tests.** [`tests/unittest/_torch/attention/sparse/test_dsa_indexer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/attention/sparse/test_dsa_indexer.py) — `test_indexer_k_cache_scatter_custom_op` runs the CUDA op against a PyTorch reference on two layers of the same cache pool and asserts the outputs match.

The full call chain is three lines:

```py
# tensorrt_llm/_torch/attention_backend/sparse/dsa.py
torch.ops.trtllm.indexer_k_cache_scatter_op(
    k_fp8, k_scale, k_cache,
    metadata.slot_mapping_fp8, metadata.slot_mapping_scale,
    num_tokens,
)
```

```cpp
// cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp
tk::invokeIndexerKCacheScatter(/* unpacked tensors and strides */, stream);
```

```cpp
// cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu
indexerKCacheScatterUnifiedKernel<<<grid, block, 0, stream>>>(/* ... */);
```

---

## 6. Testing and validation

A custom kernel without tests is a regression waiting to happen. At minimum:

1. **Unit test.** Add a test under `tests/unittest/_torch/...` that:  
   - constructs typical inputs on CUDA;  
   - runs your op (`torch.ops.trtllm.<name>`);  
   - runs a Python reference (use plain PyTorch — `torch.scatter`, `torch.softmax`, etc.);  
   - compares with `torch.testing.assert_close` (and an exact comparison if the kernel is integer/byte-deterministic).  
2. **Shape, dtype, device coverage.** Parametrize at least:  
   - typical and edge shapes (small N, large N, non-power-of-2 sizes);  
   - every supported dtype (especially mixed-precision pairs like `bfloat16` weights × `float32` accumulator);  
   - non-contiguous inputs if the kernel claims to support them; otherwise assert and document the contiguity requirement.  
3. **Numerical correctness.** For reductions, fused activations, or attention variants, validate against a deliberately simple, slow PyTorch reference. Avoid comparing to another optimized kernel — that hides bugs that correlate between the two.  
4. **Performance sanity check (optional but recommended).** A small benchmark in the test file (or in `examples/`) that prints wall-time for a representative shape gives reviewers a concrete improvement number.  
5. **Hardware availability.** Decorate tests that require Hopper or newer with the appropriate skip (e.g., `@skip_pre_hopper` / `@pytest.mark.skipif(not has_<dep>(), ...)`), as is done in `test_dsa_indexer.py`.

Set `LLM_MODELS_ROOT` for any test that loads weights — most kernel-only unit tests do **not** need it, but full-model integration tests do (see [`AGENTS.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/AGENTS.md) and [`docs/source/developer-guide/ci-overview.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/ci-overview.md)).

---

## 7. Common mistakes

- **Forgetting op registration.** The op must be visible on `torch.ops.trtllm.<name>` *and* have its module imported on Python startup. For C++ ops, `TORCH_LIBRARY_FRAGMENT` + `TORCH_LIBRARY_IMPL` is enough; for Python ops, make sure the wrapper module is imported via [`custom_ops/__init__.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/__init__.py).  
- **Assuming contiguity.** The PyTorch tensor you receive may be a non-trivial view (e.g., a slice of a paged KV cache). Either handle strides in the kernel (as `indexerKCacheScatter.cu` does) or call `.contiguous()` in Python before the op.  
- **Missing the thop CMake entry.** `cpp/tensorrt_llm/thop/` source files are listed explicitly in `CMakeLists.txt`. Forgetting to add yours leads to "undefined reference to" or "no implementation for op" errors that are easy to misdiagnose.  
- **Schema / signature mismatch.** The Python call site, the C++ schema string in `m.def(...)`, and the C++ implementation must agree on argument count, argument names, optionality, and return type. A `Tensor(a!)` in the schema is required if you mutate that tensor in place.  
- **Missing dtype / device / contiguity checks.** A kernel that does `reinterpret_cast<uint32_t*>(...)` is implicitly assuming element size and alignment. Always assert these in the binding (`TORCH_CHECK(t.element_size() == ..., ...)`, `TORCH_CHECK(t.is_contiguous(), ...)`).  
- **Skipping the fake/meta registration.** Ops that don't have a fake registered will break under `torch.compile` and FakeTensor-based shape inference. Register one whenever the op returns a tensor.  
- **JIT compile-cache key bugs.** For CuTe DSL / cuTile, an under-specified cache key (e.g., omitting dtype) produces correct results until someone changes the dtype of the call site. Include every parameter that influences codegen.
