<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VisualGen native ops

Keep a C++/CUDA implementation beside the Python op that owns its tensor contract. The [VisualGen CMake entry](CMakeLists.txt) discovers `csrc/CMakeLists.txt` under `tensorrt_llm/_torch/visual_gen/modeling_v2/`. Scope names are relative to that directory.

| Source scope | Installed library on Linux | Torch namespace |
| --- | --- | --- |
| `catalog/<category>/csrc/` | `libth_visual_gen_catalog.so` | `vg_catalog` |
| `models/<family>/csrc/` | `libth_visual_gen_<family>.so` | `vg_<family>` |
| `models/<family>/<target>/csrc/` | The same family library | `vg_<family>_<target>` |

Catalog entries use unique op names across categories. Repeated underscores in a Torch namespace collapse to one. The family and target can define the same op name in their respective namespaces. Every `models/<family>/<subpackage>/csrc/` is a target scope, so native code shared across the family goes in `models/<family>/csrc/`. A scope needs a build declaration when it owns native sources.

## Add an op

1. Declare its architectures and sources in the owning scope. For example, [packed SwiGLU](../../../tensorrt_llm/_torch/visual_gen/modeling_v2/catalog/activation/csrc/CMakeLists.txt) declares `add_visual_gen_native(ARCHS 100f SOURCES swigluBlockFp8.cu swigluBlockFp8Op.cpp)`. Only scopes matching the build's requested CUDA architectures enter the library.
2. Put the device kernel and launcher in a `.cu` file with a Torch-independent header. Put tensor validation, output allocation and dispatcher registrations in a `.cpp` file. Wrap the C++ code of both files in `TRTLLM_VG_NAMESPACE_BEGIN` and `TRTLLM_VG_NAMESPACE_END`, which give each scope its own C++ namespace within a shared library. Use `TRTLLM_VG_LIBRARY_FRAGMENT` and `TRTLLM_VG_LIBRARY_IMPL` to register in the scope's namespace. Supply a Meta or fake implementation for tracing with the real output shapes, dtypes and strides. [SwiGLU binding](../../../tensorrt_llm/_torch/visual_gen/modeling_v2/catalog/activation/csrc/swigluBlockFp8Op.cpp)
3. Resolve the op from a private Python module with `load_native_op(scope, name)`. For the activation catalog the scope is `catalog/activation`. Import that module when the native path is used. This keeps V1 and other V2 paths usable without loading that library. For a Python fake, register it after loading the op schema. [Native loader](../../../tensorrt_llm/_torch/visual_gen/modeling_v2/_native.py) · [Activation binding](../../../tensorrt_llm/_torch/visual_gen/modeling_v2/catalog/activation/_native.py)
4. Build the `visual_gen_libs` target or the owning library (for example, `th_visual_gen_catalog`). The wheel build installs the selected libraries and `visual_gen_native.json` in `tensorrt_llm/libs/`. After a manual relink, refresh the installed copy used by Python. A scope omitted from the build raises `NativeScopeUnavailable`; a listed library that fails to load is an installation error.
