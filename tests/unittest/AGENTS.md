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

Apply the shared [test value and CI placement rules](../AGENTS.md#test-value-and-ci-placement).
When writing or pruning Python unit tests:

- **Strengthen before adding**: extend an existing test when it covers the
  regression without extra setup or processes. Keep distinct boundary, error,
  backend, and lifecycle cases.
- **Verify before deduplicating**: compare executed inputs, assertions,
  fixtures, markers, and target implementations — similar names can hit
  different code paths, and different names can test the same one.
- **Keep matrices intentional**: parameterize dimensions that change the
  exercised branch; use one representative value for inactive dimensions, and
  expand them if product code later branches on them. Note any coverage you
  deliberately drop in the PR (a smaller passing suite alone does not prove
  equivalence).
- **Keep tests reliable and assertions meaningful**: use readiness/completion
  signals with a separate protective timeout instead of sleep-based timing;
  scope env and event-loop changes and clean up owned threads, processes, and
  connections. For stress/concurrency tests, assert completeness, ordering, and
  error behavior so an empty stream cannot pass silently, and keep real
  communication when communication is under test.
- **When claiming runtime or cost improvements**: compare the same selected
  tests under matching hardware and runtime settings, and report total execution
  work and slowest shard separately (fewer cases alone does not prove faster
  wall time or fewer flakes).

Before changing CI registrations or process boundaries, see the
[test-list guidance](../integration/test_lists/AGENTS.md#test-registration-and-batching).
An existing parent entry may already select a new test; verify actual selection
before adding an entry.
