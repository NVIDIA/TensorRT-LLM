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

Before adding a test or parameterized variant, record in the PR:

- Which regression do its assertions catch?
- What coverage does it add beyond existing tests?
- Should it run in pre-merge CI, post-merge CI, QA, or only as a one-time local
  check, and why is that frequency worth its setup and execution cost?

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

This is a reminder for whoever adds a test, not a hard gate. Not every test runs
in CI — a test is a CI test, a QA test, or a local developer test:

- If a newly added test needs to be protected by CI, it must be included in at
  least one CI test list under `tests/integration/test_lists/test-db/` (e.g.
  `l0_cpu.yml`, `l0_b200.yml`, `l0_dgx_b200.yml`). A CI test that no list
  references — directly or through a listed parent directory — is never
  collected: it reports nothing, so a regression in it merges unnoticed, a
  silent false green.
- A test that is not included in any CI test list is treated as intended for
  local developer testing or QA rather than CI.

So when you add a test, confirm with the author/reviewer whether it should run
in CI and, if so, which platform's test list(s) should carry it. Leaving a new
test out of every CI list is fine for QA-only or local-only tests — but make it
a deliberate, confirmed decision rather than a forgotten step.

Choose the list by where a CI test runs: CPU-only tests go in `l0_cpu.yml`;
single-GPU tests in the per-GPU list for their target (e.g. `l0_b200.yml`);
multi-GPU tests in a multi-GPU list (e.g. `l0_dgx_b200.yml`,
`l0_gb300_multi_gpus.yml`). If you are unsure which list(s) a test belongs in,
ask the author/reviewer rather than leaving it unlisted.
