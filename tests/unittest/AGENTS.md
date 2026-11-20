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

- For a new case or parameter value, identify the regression its assertions
  detect and the gap in existing tests. Prefer strengthening an existing test
  when that protects the behavior without another setup or process. Preserve
  distinct boundary, error, backend and lifecycle coverage.
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
