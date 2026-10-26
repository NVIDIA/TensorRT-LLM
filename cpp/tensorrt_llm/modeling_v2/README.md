<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# modeling_v2 family libraries: what they share

Each modeling_v2 model family keeps its C++/CUDA with its Python code, in
`tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/csrc/` and
`<family>/<target>/csrc/`, and builds into its own torch-op library,
`libth_modeling_v2_<family>.so`. The guide for authors is
`tensorrt_llm/_torch/_experimental/modeling_v2/NATIVE.md`.

This directory holds only what every family library shares; adding a model
changes nothing here:

| Path | Role |
|---|---|
| `CMakeLists.txt` | Finds every `models/<family>/csrc/CMakeLists.txt` and builds the family libraries (`cpp/cmake/modules/modeling_v2.cmake`). |
| `common/namespace.h`, `common/torchLibrary.h` | The per-unit C++ and torch namespace macros native code registers through. |
| `common/familyLibrary.version` | The version script: a family library exports no symbol. |
| `contractSurface.txt` | The stock headers native code may include. Adding one is reviewed by the header's stock owners. |
