<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI test lists

Omit explicit `TIMEOUT (60)` markers. CI's command-line default is 60 minutes:
`getPytestBaseCommandLine()` in `jenkins/L0_Test.groovy` passes `--timeout=3600` to pytest.
Test-level timeout decorators take precedence over this default; removing a list
marker can expose an existing class-level timeout.

Pre-commit enforces this with `python3 scripts/check_ci_test_timeouts.py` from the
repository root. The check only scans this `test-db/` directory; QA lists are excluded.
It also verifies that `pytestTestTimeout` in `getPytestBaseCommandLine()` equals
`_DEFAULT_CI_TIMEOUT_MINUTES * 60` seconds, and fails if the definition cannot be
uniquely identified as a quoted integer literal. Update both definitions together.
