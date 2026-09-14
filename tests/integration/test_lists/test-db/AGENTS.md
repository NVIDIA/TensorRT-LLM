<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI test lists

Omit explicit `TIMEOUT (60)` markers. CI's command-line default is 60 minutes:
`getPytestBaseCommandLine()` in `jenkins/L0_Test.groovy` passes `--timeout=3600` to pytest.

For numeric timeout settings on regular integration tests, precedence is highest
to lowest in this order. All values are in **seconds**, except list `TIMEOUT`
values, which are in **minutes**:

1. Function/method decorator: `@pytest.mark.timeout(seconds)`.
2. Parameter marker: `pytest.param(..., marks=pytest.mark.timeout(seconds))`.
3. Test-list marker: `TIMEOUT (minutes)`.
4. Class decorator: `@pytest.mark.timeout(seconds)`.
5. Module marker: `pytestmark = pytest.mark.timeout(seconds)`.
6. Command-line option: `--timeout=seconds` (CI supplies `--timeout=3600`).
7. Environment variable: `PYTEST_TIMEOUT`.
8. Pytest configuration: `timeout` in `pytest.ini` or equivalent.

`pytest-timeout` uses the first timeout marker on the nearest pytest node, falling
back to the command-line/environment/configuration default when no marker supplies
a timeout. Function and parameter markers are attached to the test item first;
`modify_by_test_list()` in `tests/integration/defs/test_list_parser.py` appends the
list marker to that same item. Consequently, an existing function or parameter
timeout wins over the list marker, while the list marker wins over class and
module markers. For `unittest/...` list entries, the wrapper instead receives the
list timeout as a parameter marker through `pytest_generate_tests()`.

For example, with a class timeout of `14400` seconds and no function or parameter
timeout, `TIMEOUT (60)` makes the effective timeout 60 minutes. Removing that list
marker makes the class's 240-minute timeout effective, even with `--timeout=3600`.
If the function also has `@pytest.mark.timeout(7200)`, its 120-minute timeout wins
both before and after removal. Without any timeout markers, the CI default of
60 minutes applies. The command-line default therefore does not cap marker values.

Pre-commit rejects explicit `TIMEOUT (60)` markers with
`python3 scripts/check_ci_test_timeouts.py` from the repository root.
The check only scans this `test-db/` directory; QA lists are excluded.
It also verifies that `pytestTestTimeout` in `getPytestBaseCommandLine()` equals
`_DEFAULT_CI_TIMEOUT_MINUTES * 60` seconds, and fails if the definition cannot be
uniquely identified as a quoted integer literal. Update both definitions together.
