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

# C++ Tests

Native test sources live in `unit_tests/`. The pytest wrappers in
[`tests/integration/defs/cpp`](../../tests/integration/defs/cpp/) select CTest
groups or native executables and prepare resources required by those tests.
Read the [C++ test guidance](AGENTS.md) before changing targets or build behavior.

## Run through pytest

Use a supported build environment with the matching TensorRT-LLM wheel and test
dependencies installed. Run these commands from the repository root to inspect
the available wrapper cases:

```bash
pytest tests/integration/defs/cpp/test_unit_tests.py --collect-only
pytest tests/integration/defs/cpp/test_multi_gpu.py --collect-only
```

The build fixtures first try the prebuilt C++ test bundle. If no applicable
bundle is available, they configure and build selected native targets. See
[CI bundle and source-build fallback](AGENTS.md#ci-bundle-and-source-build-fallback)
for bundle rules, supported fallback fixtures, and target selection.
The multi-GPU wrappers also use this bundle-first fixture.

Choose a collected case for your hardware and required GPU count. For example,
on a suitable SM90 system:

```bash
pytest 'tests/integration/defs/cpp/test_unit_tests.py::test_unit_tests[runtime-90]'
pytest 'tests/integration/defs/cpp/test_multi_gpu.py::test_mpi_utils[90]'
```

The MPI example uses four ranks. Runtime and batch-manager wrapper groups also
prepare LoRA test weights through the `lora_setup` fixture. Running native
executables directly does not perform that preparation.

## Build from source and run directly

From the repository root, build and install the wheel, then build the tests.
Replace `90` with the CUDA architecture needed by your tests:

```bash
CPP_BUILD_DIR=cpp/build
python3 scripts/build_wheel.py -a "90" --build_dir "${CPP_BUILD_DIR}"
pip install -r requirements-dev.txt
pip install build/tensorrt_llm*.whl
cmake --build "${CPP_BUILD_DIR}" --target google-tests --parallel "$(nproc)"
```

`google-tests` builds all enabled native tests. To build a directory subset, use
its recursive target, for example `cpp-tests-common`. Component targets such as
`common-tests` cover migrated tests for that owner and are not interchangeable
with directory targets.

Run a built executable or its matching CTest directory:

```bash
./${CPP_BUILD_DIR}/tests/unit_tests/common/optionalRefTest
ctest --test-dir "${CPP_BUILD_DIR}/tests/unit_tests/common" --output-on-failure --output-junit "cpp-test-report.xml"
```

After building all enabled tests and preparing their required resources, run
the full CTest tree with:

```bash
ctest --test-dir "${CPP_BUILD_DIR}" --output-on-failure --output-junit "cpp-test-report.xml"
```

Avoid scanning a CTest directory whose executables were only partially built.
For a single-target build, run that executable directly, as the compression and
FMHA head-104 wrappers do.
