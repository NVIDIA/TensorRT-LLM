<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# AGENTS.md — C++ tests

Scope: this directory and its descendants. Supplements the repository-root
`AGENTS.md`. The build and execution rules also apply to the
[integration wrappers](../../tests/integration/defs/cpp/) that run these tests.

Follow the shared [test guidance](../../tests/AGENTS.md) for test value, CI
placement, and building helpers before the test stage. See [`README.md`](README.md)
for local build and run commands.

## Build and execution

### Register a gtest

Use `add_tllm_gtest` with explicit `SOURCES`, `LINK_LIBRARIES`, `OWNER`, and
scheduling `LABELS`. Pass `FULL_STACK` only when the test needs the full
TensorRT-LLM library; component tests should link only the libraries they need.
Include `gpu` in `LABELS` when the test requires a GPU.

### Choose a build target

- `google-tests`: builds all enabled native tests.
- `cpp-tests-<directory>` (e.g., `cpp-tests-common`, `cpp-tests-executor-bounce`):
  mirrors recursive CTest discovery, including legacy and full-stack
  executables. Use the directory target when the next step runs CTest on that
  directory.
- `<owner>-tests` (e.g., `common-tests`): builds tests registered to that
  component owner; not interchangeable with directory targets.

### CI bundle and source-build fallback

- On Linux x86_64 (`CONFIG_LINUX_X86_64_VANILLA`), the build stage compiles
  `google-tests` and publishes the `cpp_test_bundle` CMake component as a
  separate `cpp-tests-<main-tar-name>` archive (build reuse requires both the
  main and test archives). Only `-CPP-` stages using that build unpack the test
  archive into `src/cpp/build` and run CTest via `tests/integration/defs/cpp/`
  (`parallel_run_ctest`). Because CTest discovers cases on the test node, the
  build node needs no GPU.
- The three existing fixtures in
  [`conftest.py`](../../tests/integration/defs/cpp/conftest.py)
  (`build_google_tests`, `build_kv_cache_compression_tests`, and
  `build_fmha_head104_tests`) are supported exceptions to the shared rule
  against test-stage builds. Each calls `prepare_test_bundle(build_type)` first
  (see [`find_test_bundle`](../../tests/integration/defs/cpp/cpp_common.py) for
  the exact Linux `Release` bundle conditions) and otherwise falls back to
  `configure_only=True` plus building only its bounded target(s). Preserve
  bundle reuse and these bounded fallback targets; extend build-stage packaging
  instead of adding new test-node build paths.
