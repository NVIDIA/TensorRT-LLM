<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modeling V2 native code

This guide covers C++/CUDA code owned by a Modeling V2 family or target. For
model development in general, see the
[Modeling V2 guide](../../../tensorrt_llm/_torch/_experimental/modeling_v2/README.md).

Native code lives next to the Python code that uses it, under
`tensorrt_llm/_torch/_experimental/modeling_v2/models/`:

* `<family>/csrc/` holds the family's `CMakeLists.txt` and code shared by its
  targets.
* `<family>/<target>/csrc/` holds one target's own code.

The build finds every `models/<family>/csrc/CMakeLists.txt` and builds one
library per family, `libth_modeling_v2_<family>.so`, installed next to
`libth_common.so`. Nothing in it is linked into the other TensorRT-LLM
libraries. [`_native.py`](../../../tensorrt_llm/_torch/_experimental/modeling_v2/_native.py)
loads the library only when routing selects one of the family's targets.
Adding a family with native code does not require changes under `cpp/`.

## Shared infrastructure

This directory holds only what every family shares:

| Path | Role |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | Finds the family build files and builds their libraries. |
| [modeling_v2.cmake](../../cmake/modules/modeling_v2.cmake) | Defines `add_modeling_v2_family`, `add_modeling_v2_shared` and `add_modeling_v2_target`. |
| [common/namespace.h](common/namespace.h) | Gives each target, and each family's shared code, its own C++ namespace. |
| [common/torchLibrary.h](common/torchLibrary.h) | Registers Torch ops in the namespace of the code being compiled. |
| [common/familyLibrary.version](common/familyLibrary.version) | Keeps all symbols local, so two family libraries can define the same C++ names. |

## Adding native code to a target

Paths below are relative to `tensorrt_llm/_torch/_experimental/modeling_v2/`.
The steps use a target `foo_nvfp4__sm_103__tp4` in family `foo`. GPT-OSS is a
small working example: see its
[build file](../../../tensorrt_llm/_torch/_experimental/modeling_v2/models/gpt_oss/csrc/CMakeLists.txt)
and its target's `csrc/` directory.

1. Declare the family and the target in `models/foo/csrc/CMakeLists.txt`:

   ```cmake
   add_modeling_v2_family(foo)
   add_modeling_v2_target(foo foo_nvfp4__sm_103__tp4
                          ARCHS 103a SOURCES *.cu *.cpp)
   ```

   `SOURCES` paths and globs are relative to the target's `csrc/` directory.

2. Write the sources in `models/foo/foo_nvfp4__sm_103__tp4/csrc/`:

   * `.cu` files hold kernels. They are compiled for `ARCHS` and are not linked
     against PyTorch; keep PyTorch headers out of them.
   * `.cpp` files hold the Torch op bindings.
   * Include your own headers by their path from `models/`, for example
     `#include "foo/foo_nvfp4__sm_103__tp4/csrc/kernel.h"`. Include the
     shared headers as `tensorrt_llm/modeling_v2/common/namespace.h` and
     `tensorrt_llm/modeling_v2/common/torchLibrary.h`.
   * Wrap your C++ code in `TRTLLM_MV2_NAMESPACE_BEGIN` and
     `TRTLLM_MV2_NAMESPACE_END`. Use `TRTLLM_MV2_NS` to refer to that namespace
     from outside it.
   * Register ops with `TRTLLM_MV2_LIBRARY_FRAGMENT(m)` and
     `TRTLLM_MV2_LIBRARY_IMPL(<dispatch key>, m)`. Do not use `TORCH_LIBRARY`
     directly; the build generates the `TORCH_LIBRARY` block for each
     namespace, and a second one fails when the library loads.

3. In the target's `__init__.py`, load the ops and call them through the
   returned namespace:

   ```python
   from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

   native_ops = load_target_ops(__name__)  # torch.ops.mv2_foo_foo_nvfp4_sm_103_tp4
   ```

4. Register fake implementations in the target's `fake_ops.py`, using
   `namespace_of(__name__)` for the namespace. The loader imports this file
   after it finds the target in the family library.

5. Add tests for the ops, their PyTorch references and their fakes.

## Sharing native code within a family

Put code used by several targets in `models/<family>/csrc/` and declare it with
`add_modeling_v2_shared(<family> ARCHS ... SOURCES ...)`. Call its ops through
`load_family_ops(__name__)` from any module of the family, and register its
fakes in `models/<family>/fake_ops.py`.

A target that uses only the shared code still needs an
`add_modeling_v2_target(...)` call, without `SOURCES`, placed after
`add_modeling_v2_shared`. Its `__init__.py` still calls
`load_target_ops(__name__)`: that call checks that this build includes the
target. Kimi K3 (`models/kimi_k3_vl/`) is an example.

## Build options

`add_modeling_v2_shared` and `add_modeling_v2_target` take these options:

| Option | Values |
|---|---|
| `ARCHS` (required) | `NN`, `NNa` or `NNf`, kept only if the build requests a matching architecture. For example, `103a` builds SM103-specific code when the build includes SM103. `100f` builds family-portable code when the build includes any SM 10.x architecture from SM100 up; use it for `sm_10x` targets. |
| `DEPS` | Any of `cublas`, `cublaslt`, `cuda_driver`, `internal_cutlass`, `nccl`, `trtllm_gen_fmha_interface`. |
| `REQUIRES` | Any of `MULTI_DEVICE`, `NVSHMEM`. |

A target is not built if none of its `ARCHS` is requested, if a `REQUIRES`
feature is disabled, or if `TRTLLM_MODELING_V2_TARGETS` excludes it. CMake
prints the reason as `modeling_v2: <family>/<target> not built: ...`.

## Loading errors

If the family library is missing, or the build did not include the target,
loading raises `NoTarget`. Under `modeling_v2="auto"` the engine then uses
Modeling V1.

If the library is installed but cannot be loaded, or a `fake_ops.py` registers
an op the library does not define, loading raises `NativeLoadError`. This
error is raised under `auto` too, so a broken installation does not silently
fall back.

## Building

To build only some Modeling V2 targets, pass a filter to
`scripts/build_wheel.py`, for example `-D "TRTLLM_MODELING_V2_TARGETS=foo/*"`.
The pattern matches `<family>/<target>`; `all` (the default) and `none` are
also accepted. `build_wheel.py` splits `-D` values at `;`, so pass a single
pattern through it. The rest of TensorRT-LLM still
builds. CMake caches the value in the build directory, so pass
`-D "TRTLLM_MODELING_V2_TARGETS=all"` to build every target again.

To rebuild one family's library after a change:

```bash
cmake --build <build_dir> --target th_modeling_v2_foo
```

`<build_dir>` is `cpp/build` for a Release build, or `cpp/build_<build type>`
otherwise. The package picks up the rebuilt library directly only if
`build_wheel.py` installed it as a symlink (`--skip_building_wheel
--linking_install_binary`). By default the library in `tensorrt_llm/libs/` is
a hard link to the build output; relinking replaces the build output with a
new file and leaves the installed copy unchanged. In that case, rerun
`build_wheel.py` to install the new library.
