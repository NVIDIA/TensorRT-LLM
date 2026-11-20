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

## Test value and execution placement

Use these questions when deciding whether to add a test or parameter value.

- Which regression do its assertions catch?
- What coverage does it add beyond existing tests?
- Should it run in pre-merge CI, post-merge CI, QA, or only as a one-time local
  check, and why is that frequency worth its setup and execution cost?

Summarize the rationale in the PR. One sentence can cover a small addition,
such as another dtype, and cases with the same rationale can be described
together. For a new CI entry, a larger parameter matrix or an expensive GPU
stage, explain the extra coverage and why it warrants the recurring cost.

Keep focused, reliable regression coverage in pre-merge CI; justify expensive
or broad coverage there instead of registering every new test automatically.
Use post-merge or QA for coverage suited to those schedules. Keep one-time
investigation checks out of recurring CI unless they protect a lasting
regression. Prune redundant cases while preserving meaningful failure modes.

For Python unit-test case design and parameter matrices, follow
the [unit-test guidance](unittest/AGENTS.md). For C++ targets, build artifacts and
execution, follow the [C++ test guidance](../cpp/tests/AGENTS.md).
Before changing CI registrations or waivers, read the
[test-list guidance](integration/test_lists/AGENTS.md).

## Build test helpers before the test stage

Build new CI test helpers and executables in the build stage and consume
matching artifacts in the test stage. Do not add ad hoc compiler, CMake, or
extension builds during test discovery, setup, or execution merely to prepare a
helper. Extend the existing build targets and artifact flow for new requirements.

Existing documented source-build fallbacks are supported exceptions, including
CI configurations without a usable prebuilt bundle. For the C++ fixtures and
their exact fallback conditions, read
[Existing source-build fallback](../cpp/tests/AGENTS.md#existing-source-build-fallback).
Preserve those paths and explicit local/source-build workflows. Outside these
exceptions, do not bypass available matching artifacts to rebuild ordinary
helpers on each test node.

Keep intentional runtime compilation when it is part of the behavior under
test (for example, a JIT kernel), and account for its cost when choosing the CI
layer.

## Whether a new test runs in CI

This is a reminder for whoever adds a test, not a hard gate. Not every test
belongs in recurring CI; QA-only and local developer tests are valid choices.

For the list-driven CI pipeline, ensure the intended case is selected by an
entry under `tests/integration/test_lists/test-db/`. Python unit tests may be
covered by an existing file or parent-directory bridge entry. Native C++ cases
are reached through registered integration wrappers and their CTest or binary
selection; they do not each need a YAML entry.

Check the effective pre-merge/post-merge stage, options, marker filters and
waivers. A parent entry alone does not prove a case runs. A case that no entry
selects will not execute in this pipeline and cannot protect it from regressions.
Record whether a new test belongs in CI, QA or local testing, and confirm the
intended placement with the author/reviewer rather than leaving it accidental.

Choose the list by the execution requirements: CPU-only unit tests go in
`l0_cpu.yml` and must carry `pytest.mark.cpu_only`; the CPU stages select that
marker. Standard GPU stages select `not cpu_only` (sanity stages are exceptions).
Single-GPU tests use the target platform's list (e.g. `l0_b200.yml`); multi-GPU
tests use a suitable multi-GPU list (e.g. `l0_dgx_b200.yml`,
`l0_gb300_multi_gpus.yml`). If the intended list or selection is unclear, ask the
author/reviewer. Follow the [test-list guidance](integration/test_lists/AGENTS.md)
when adding or changing registrations.
