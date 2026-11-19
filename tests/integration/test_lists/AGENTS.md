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

# AGENTS.md — test lists

Scope: this directory and its descendants, including `waives.txt` and `test-db/`.
Supplements the repository-root and `tests/AGENTS.md` guidance.

Before editing `waives.txt` or changing a waived test-list entry, read the
[waive matching rules](../../../docs/source/developer-guide/ci-overview.md#waiving-tests).

- Copy whole-entry names and options exactly; recheck waives when options change.
  Omit `ISOLATION`, `TIMEOUT (N)`, and YAML comments. Use `SKIP (reason)` and
  fragment-free tracking URLs: `#` starts a comment anywhere.
- Inner unittest matching supports case/function/class/file, not directories.
  A directory waive only skips its exact outer entry.
- Choose [platform scope](../../../docs/source/developer-guide/ci-overview.md#platform-scope):
  unprefixed for all platforms, `full:<platform>/` for actual stage prefixes
  (`full:CPU/` for CPU), or `full:sm<version>/` for visible device 0's detected SM.
  Follow the guide's alias, multiple-platform, and string-replacement limits.
- Use case/function waives for isolated failures or temporary whole-entry waives
  for widespread failures. Track fixes and restore coverage once verified.
- Update this file and the CI overview together when changing waive processing.

## CPU test registration and batching

- Reuse a file or parent-directory entry when its actual marker/options already
  select the intended case. Adding a function entry alongside that parent repeats
  execution. Prefer one file entry for compatible functions from the same file
  when all of its CPU cases are intended to run.
- Keep intentional partial selectors and required `ISOLATION` boundaries. Before
  consolidating entries, compare real pytest collections under the stage's marker,
  exclusions and waivers: check node IDs, multiplicity and skip/xfail markers, not
  just total counts. Explain any deliberate case removal separately.
- Run the proposed batch to check fixture and process-state interactions, including
  event loops and environment variables. Retain separate processes where the tests
  require them; a matching collection does not prove execution compatibility.
- Check CBTS through the generated filtered test-db, including file/function
  changes and any domain rules that depend on explicit registrations. Confirm the
  intended stages, retained tests and split counts. Document any coarser selection
  granularity and its cost; a successful path lookup alone is not sufficient.
- Renaming entries or regrouping parameters can change report, waiver and duration
  keys. Recheck references and safe coverage-history fallback. Use measured batch
  times when evaluating shard balance; summing old per-function startup costs does
  not estimate the new batch correctly. Preserve the existing waiver rules above.

Follow the [unit-test coverage guidance](../../unittest/AGENTS.md) when adding or
reducing cases. These rules preserve useful coverage and efficient execution,
without fixing a maximum test count or preventing justified new regression tests.
