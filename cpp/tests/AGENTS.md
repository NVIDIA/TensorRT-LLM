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

New C++ unit tests should use `add_tllm_gtest` with explicit `SOURCES`,
`LINK_LIBRARIES`, `OWNER`, and scheduling `LABELS`. Request `FULL_STACK` explicitly
for integration tests and include `gpu` in `LABELS` for tests that need a GPU.
CI controls test concurrency through `parallel_run_ctest`. `google-tests`
continues to build all C++ tests; component aggregate targets build migrated tests.
C++ source-build fallbacks build the collected tests with incremental native targets.
`cpp-tests-<directory>` targets (for example `cpp-tests-common` or
`cpp-tests-executor-bounce`) mirror recursive CTest directory coverage, including
legacy and full-stack executables. OWNER aggregates are component subsets and
are not interchangeable with these directory targets.
The Linux Release source-build fixture reuses the installed CI library for the
RNN formatter test. CI source and wheel use the same commit and image per
architecture. Link/load errors fail the tests. Direct CMake builds use source
libraries unless explicitly configured otherwise.
`CONFIG_LINUX_X86_64_VANILLA` also builds `google-tests`
and publishes the `cpp_test_bundle` CMake component as a separate
`cpp-tests-<main-tar-name>` artifact. The main package does not include the test
programs. Only `-CPP-` stages using this build configuration fetch and unpack
the companion into `src/cpp/build`, including fat-image consumers. Build reuse
requires both artifacts. Linux Release integration fixtures use this bundle
directly, retaining CTest selection and retries. The two FMHA tests share a
test-only library packaged next to their executables; other runtime libraries
come from the matching installed wheel. Test artifacts keep symbols but omit
BOLT's extra relocation records.
CPP stages collect only `tests/integration/defs/cpp` and use the standard test
environment setup. CTest discovers cases on the test node, so the build stage
does not need a GPU.

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
