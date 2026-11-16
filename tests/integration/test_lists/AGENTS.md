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
