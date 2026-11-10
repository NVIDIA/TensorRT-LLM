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

- Whole-entry waives match exactly, including command-line arguments such as `-k`.
  Omit `ISOLATION`, `TIMEOUT (N)`, and YAML comments from waive names.
  Use tracking URLs without fragments; a literal `#` starts a comment anywhere.
  Recheck the corresponding waive when changing those arguments.
- Inner unittest waives support case, function, class, and file matching, but
  not directory matching. A directory waive does not recursively skip its tests.
- Choose platform scope explicitly: no `full:` prefix applies across platforms;
  `full:<platform>/` targets a CI test-prefix namespace; `full:sm<version>/`
  depends on the current pytest process detecting that SM on visible CUDA device 0.
  Use the actual stage prefix: `H100` is not an alias for `DGX_H100` or `H100_PCIe`.
  Repeated platform/SM strings in a name are also rewritten; verify the resulting match.
  For standard CPU stages use `full:CPU/`. List multiple platforms on separate lines.
- Use the narrowest matching scope and include a bug link or brief reason.
