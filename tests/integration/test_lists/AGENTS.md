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

## Waive list

- **Keep registrations for tests you intend to fix.** Add one
  `<test> SKIP (reason)` line to `waives.txt` with a tracking bug (no `#`
  fragment — `#` starts a comment anywhere on the line) or brief explanation.
  For outer test-db entries, copy the exact selector and options (order, spaces,
  quotes) and omit `ISOLATION`, `TIMEOUT (N)`, and YAML comments.
- **Choose the right scope.** Unprefixed skips all platforms; `full:<platform>/`
  targets a stage prefix before its first `-` (case-sensitive, e.g.,
  `full:DGX_H100/`, `full:A100X/`, `full:CPU/`, not `H100` or `A100`);
  `full:sm<N>/` checks device 0's detected SM. Use case/function or
  platform-specific waives for isolated failures; temporary broader waives are
  fine for widespread breakages if you note what unaffected coverage they
  disable and narrow/remove them once fixed. Verify the skip reason in the run.
- **Sorting and duplicates.** Keep `waives.txt` sorted by whole line
  (`file-contents-sorter` reorders it in pre-commit; re-stage if modified).
  `waive list check` rejects duplicate selectors even when `SKIP` reasons differ.
- **Retiring tests.** When dropping a low-value or redundant test, remove its
  YAML registrations and obsolete waives and explain the decision in the PR
  instead of waiving it indefinitely.

For PR/TOT waive merging, outer vs. inner `unittest/` matching, platform/SM
edge cases, and implementation pointers, read the
[waive list reference](README.md#waive-list-reference).

## Test registration and batching

Applies to CPU and GPU `unittest/` bridge entries; ordinary integration tests
and native C++ wrappers keep their own case selectors and launch requirements.

- **Avoid duplicate execution.** Reuse an existing file or parent-directory
  entry when its markers and options already select the case. Prefer one file
  entry for compatible functions when all cases selected for that stage should
  run.
- **Preserve execution boundaries.** Keep intentional partial selectors,
  multi-rank launchers, and required `ISOLATION` boundaries. `ISOLATION`
  separates only the outer test-list entry; it does not isolate inner unit tests
  or disable inner pytest workers.
- **When consolidating, renaming, or re-batching entries** (not needed for
  routine single-entry additions): verify real pytest collection (node IDs,
  multiplicity, skip/xfail), check worker lookup in `agg_unit_mem_df.csv`
  (unmatched entries default to serial), update affected waivers and CBTS rules,
  and validate execution compatibility (including GPU memory and retained CUDA
  or process-group state on GPU stages). See the
  [batching and re-grouping reference](README.md#batching-and-re-grouping-reference)
  for the full validation checklist.
