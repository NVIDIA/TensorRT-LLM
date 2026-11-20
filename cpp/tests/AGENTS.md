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
`AGENTS.md`. The build and execution guidance also applies to the
[integration fixtures](../../tests/integration/defs/cpp/) that run these tests.

Follow the shared [test guidance](../../tests/AGENTS.md) for regression value,
coverage, CI placement and building helpers before the test stage.

## Build and execution

### Register a gtest

Use `add_tllm_gtest` with explicit `SOURCES`, `LINK_LIBRARIES`, `OWNER`, and
scheduling `LABELS`. Use `FULL_STACK` when the test needs the full TensorRT-LLM
library. For a component test, link only the libraries it needs. Include `gpu`
in `LABELS` for tests that need a GPU.

### Choose a build target

- `google-tests` builds all enabled native tests.
- `cpp-tests-<directory>` mirrors recursive CTest discovery, including legacy
  and full-stack executables. Examples include `cpp-tests-common` and
  `cpp-tests-executor-bounce`.
- An OWNER target such as `common-tests` builds the tests registered to that
  component. Use the directory target when the next step runs a CTest directory.

### Package the CI bundle

`CONFIG_LINUX_X86_64_VANILLA` builds `google-tests` and publishes the
`cpp_test_bundle` CMake component as a separate `cpp-tests-<main-tar-name>`
artifact. Build reuse requires both the main and test archives. The test
binaries retain symbols and omit BOLT's extra relocation records. The two FMHA
tests share a test-only library packaged beside their executables. Other
runtime libraries come from the matching installed wheel.

### Run the CPP stage

Only `-CPP-` stages using that build configuration fetch and unpack the test
archive into `src/cpp/build`, including stages that use fat images. CPP stages
use the standard test environment and collect `tests/integration/defs/cpp`.
CTest discovers cases on the test node, so discovery needs no GPU on the build
node. Bundle reuse preserves CTest selection and retries. CI controls CTest
concurrency through `parallel_run_ctest`.

### Existing source-build fallback

The existing `build_google_tests`, `build_kv_cache_compression_tests`, and
`build_fmha_head104_tests` fixtures in
[`conftest.py`](../../tests/integration/defs/cpp/conftest.py) are supported
exceptions to the shared rule against adding test-stage helper compilation.
Each calls `prepare_test_bundle(build_type)` first and returns when it succeeds.

[`find_test_bundle`](../../tests/integration/defs/cpp/cpp_common.py) selects a
bundle only on Linux with the `Release` build type, no explicit
`TRTLLM_BUILD_ROOT`, and a `CTestTestfile.cmake.in` in the resolved build directory.
Otherwise, the fixtures retain their source-build path, including in CI.

The fallback configures with `configure_only=True`, then builds selected native
targets: `selected_build_targets` for `build_google_tests`,
`nvfp4ColdPageKernelsTest` for the compression fixture, and `fmhaHead104Test` for
the head-104 fixture. Preserve these bounded builds and bundle reuse. New helper
requirements should extend build-stage packaging rather than introduce another
test-node build path.

For the RNN formatter test, the Linux Release source-build fixture reuses the
installed CI library. CI source and wheel use the same commit and image per
architecture. Link or load errors fail the tests. Direct CMake builds use
source libraries unless explicitly configured otherwise.
