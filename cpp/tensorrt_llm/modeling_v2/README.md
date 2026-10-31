<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modeling V2 native code

Use the family-library mechanism for new model-owned C++/CUDA code. Shared
native sources and the build declaration live in
`tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/csrc/`;
a target's own sources live in `models/<family>/<target>/csrc/` under the
same Modeling V2 package.

The build discovers each family's `csrc/CMakeLists.txt` and produces
`libth_modeling_v2_<family>.so`, installed beside `libth_common.so`. Target
packages load their operations through
[_native.py](../../../tensorrt_llm/_torch/_experimental/modeling_v2/_native.py).
See the [Modeling V2 guide](../../../tensorrt_llm/_torch/_experimental/modeling_v2/README.md)
for model development and ownership conventions.

## Shared infrastructure

This directory contains the common native build infrastructure. A new family
is discovered through its own build declaration.

| Path | Role |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | Discovers family build declarations and assembles their libraries. |
| [modeling_v2.cmake](../../cmake/modules/modeling_v2.cmake) | Defines family, shared-source and target build helpers. |
| [common/namespace.h](common/namespace.h) | Isolates family-shared and target-specific C++ namespaces. |
| [common/torchLibrary.h](common/torchLibrary.h) | Registers Torch operations in each unit's namespace. |
| [common/familyLibrary.version](common/familyLibrary.version) | Hides native symbols to avoid collisions between family libraries. |

## Adding native operations

Paths below are relative to
`tensorrt_llm/_torch/_experimental/modeling_v2/`, unless a repository path is given.
The [GPT-OSS build declaration](../../../tensorrt_llm/_torch/_experimental/modeling_v2/models/gpt_oss/csrc/CMakeLists.txt)
is a concrete example.

1. Declare the family and target in `models/<family>/csrc/CMakeLists.txt`:

   ```cmake
   add_modeling_v2_family(foo)
   add_modeling_v2_target(foo foo_nvfp4__sm_103__tp4
                         ARCHS 103a SOURCES *.cu *.cpp)
   ```

2. Put those sources in the target's `csrc/`. Use `TRTLLM_MV2_NAMESPACE_BEGIN` /
   `TRTLLM_MV2_NAMESPACE_END` for its C++ namespace, and
   `TRTLLM_MV2_LIBRARY_FRAGMENT` / `TRTLLM_MV2_LIBRARY_IMPL` for Torch registration.
3. In the target's `__init__.py`, call `native_ops = load_target_ops(__name__)`
   from `_native`. Use the returned namespace to call the operations.
4. Register fake implementations in the target's `fake_ops.py`, using
   `namespace_of(__name__)` to name the namespace. The native loader imports
   these registrations after loading the library.
5. Add tests for the operations, their references and their fake implementations.

For native code shared by a family's targets, use `add_modeling_v2_shared(...)`,
`load_family_ops(__name__)` and the family's `fake_ops.py`. A target that only uses
that shared native code still needs an `add_modeling_v2_target(...)` declaration,
without `SOURCES`.

Declare the supported CUDA `ARCHS` and any `DEPS` or `REQUIRES` in CMake. The build
selects compatible targets. A missing family library or an excluded target raises
`NoTarget`; an installed library that cannot load, or incompatible fake
registrations, raises `NativeLoadError` rather than falling back.

To build only one family's native code, pass
`-D "TRTLLM_MODELING_V2_TARGETS=foo/*"` to `scripts/build_wheel.py`. For an
incremental rebuild, use `cmake --build cpp/build --target th_modeling_v2_foo`
with the build directory for your checkout.
