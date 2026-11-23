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

# AGENTS.md — tests

Scope: this directory and everything under it. Supplements the repo-root `AGENTS.md`.

## Test value and CI placement

Before adding a test or parameter value, answer three questions:

- Which regression do its assertions catch?
- What coverage does it add beyond existing tests?
- Where should it run — pre-merge CI, post-merge CI, QA, or local-only — and is
  that frequency worth its setup and runtime cost?

Summarize the placement and rationale in the PR description. One sentence is
enough for a small addition (such as another dtype or boundary case), and
related cases can share one rationale. For a new CI entry, a larger parameter
matrix, or an expensive GPU stage, explain the added coverage and recurring
cost.

- **Do not register every new test in pre-merge CI.** Keep pre-merge CI focused
  on fast, reliable regression checks. Use post-merge CI or QA for heavier or
  broader matrices, and keep one-time investigation checks local unless they
  guard a lasting regression. Prune redundant cases while keeping distinct
  failure modes.
- **Make CI registration intentional.** For list-driven CI, verify that an entry
  under `tests/integration/test_lists/test-db/` actually selects the case under
  the stage's options, markers, and waivers — a parent-directory entry alone
  does not prove a case runs. Native C++ gtests run through registered wrappers
  and CTest/binary selection rather than per-case YAML entries. Leaving a test
  out of CI lists is fine for QA-only or local-only tests, as long as it is
  deliberate and confirmed with the author/reviewer when unclear.
- **Pick the list by hardware requirement:**
  - CPU-only unit tests → `l0_cpu.yml` with `pytest.mark.cpu_only` (CPU stages
    select `cpu_only`; standard GPU stages select `not cpu_only`, except sanity
    stages).
  - Single-GPU tests → the target GPU list (e.g., `l0_b200.yml`).
  - Multi-GPU tests → a matching multi-GPU list (e.g., `l0_dgx_b200.yml`,
    `l0_gb300_multi_gpus.yml`).

## Build test helpers before the test stage

Build new CI test helpers and native executables in the build stage and consume
matching artifacts in the test stage. Do not add ad hoc compiler, CMake, or
extension builds during test discovery, setup, or execution.

- Preserve the existing documented source-build fallbacks (see
  [CI bundle and source-build fallback](../cpp/tests/AGENTS.md#ci-bundle-and-source-build-fallback))
  and explicit local source-build workflows. Outside those exceptions, do not
  bypass available matching artifacts on test nodes.
- Keep intentional runtime compilation when it is part of the behavior under
  test (for example, a JIT kernel), and account for its cost when choosing the
  CI tier.

## Scoped guides

- Python unit tests: [unit-test guidance](unittest/AGENTS.md)
- CI test lists, batching, and waives: [test-list guidance](integration/test_lists/AGENTS.md)
  (detailed matching and batching reference in [test-list README](integration/test_lists/README.md))
- C++ tests and wrappers: [C++ test guidance](../cpp/tests/AGENTS.md)
