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

# AGENTS.md — unit tests

Scope: this directory and its descendants. Supplements `../AGENTS.md`.

## Coverage and execution cost

Before adding a unit test or parameterized variant, record in the PR:

- Which regression do its assertions catch?
- What coverage does it add beyond existing tests?
- Should it run in pre-merge CI, post-merge CI, QA, or only as a one-time local
  check, and why is that frequency worth its setup and execution cost?

Keep focused, reliable regression coverage in pre-merge CI; justify expensive
or broad coverage there instead of registering every new test automatically.
Use post-merge or QA for coverage suited to those schedules. Keep one-time
investigation checks out of recurring CI unless they protect a lasting
regression. Prune redundant cases while preserving meaningful failure modes.

- Prefer strengthening an existing test when that protects the behavior without
  another setup or process. Preserve distinct boundary, error, backend and
  lifecycle coverage.
- Compare executed inputs, assertions, fixtures, markers and the implementation
  under test before deduplicating. Similar names or bodies can exercise different
  implementations; different names can execute the same check.
- Parameterize combinations that affect the exercised branch. A representative
  value can cover an inactive dimension, while active dimensions retain their
  meaningful values and boundaries. If a product change makes that dimension
  relevant, expand the matrix with that change. State deliberately relinquished
  coverage in the PR; a passing smaller suite alone is not equivalence evidence.
- For pressure and concurrency tests, explain what the repetition count, payload
  size and concurrency protect. Assert the required completeness, ordering and
  error behavior; per-item assertions alone can pass on an empty stream. Keep
  real communication and process behavior when those are the subject of the test.
- Account for import/setup cost, subprocesses and shared state as well as test-body
  time. Use readiness/completion signals for concurrent work, with a separate
  protective timeout. Scope environment and event-loop changes and clean up owned
  threads, processes and connections.
- For test-time claims, compare the same selected work with representative CPU,
  memory, architecture and runtime settings. Report cumulative execution work and
  the slowest parallel shard separately; case-count reduction does not establish
  wall-time or flake-rate improvement.

Before changing registrations or process boundaries, follow the
[test-list instructions](../integration/test_lists/AGENTS.md). An existing parent
entry may already select a new test; confirm actual selection before adding an
entry.

## Build test helpers before the test stage

For unit tests in CI that need native helpers or executables, build them in the
build stage and consume matching artifacts in the test stage. Do not introduce
ad hoc compiler, CMake, or extension builds in pytest collection, fixtures, or
test bodies merely to prepare a helper. Reuse the existing build targets and
artifact flow; for C++ tests, see the repository-root
[C++ test build guidance](../../AGENTS.md#ci--testing).

Keep intentional runtime compilation when it is part of the behavior under
test (for example, a JIT kernel), and account for its cost when choosing the CI
layer. Preserve explicit local/source-build workflows; these are not a reason
to rebuild ordinary test helpers on every CI test node.
