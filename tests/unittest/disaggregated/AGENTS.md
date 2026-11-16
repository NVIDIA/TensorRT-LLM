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

# AGENTS.md — disaggregated unit tests

Scope: this directory and its descendants. Supplements `../../AGENTS.md`.

- Judge a test by the additional regression it detects, not by whether its
  parameters differ. Map slow GPU scenarios to the existing CPU state-machine
  tests before expanding their Cartesian product.
- Keep GPU tests for native transfer, real allocator ownership and process
  containment. Cover backend-free deadline and status permutations in CPU tests;
  retain representative GPU paths for distinct storage layouts and both transfer
  directions. Add a cross-combination when it exercises a distinct interaction.
- Advance a watchdog's own injectable clock only after the native-transfer and
  ownership gates have been observed. Let the real background watchdog deliver
  the timeout or fatal callback. Keep readiness, process-exit and cleanup waits
  on the real clock, and preserve the assertions before and after containment.
- State which combinations are removed and which retained tests protect their
  mechanisms. Passing a smaller suite alone does not establish that coverage is
  adequate. Do not equate mocked resource state with real native ownership.
