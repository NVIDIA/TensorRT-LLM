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

# Static Type Checking

TensorRT-LLM uses Pyrefly to check Python code under `tensorrt_llm/` and `tests/`.
The advisory `type-check` pre-commit hook runs Pyrefly on supplied files and
checks registered mock interfaces. Read its diagnostics and report before
committing.

## Prepare a local environment once

Run these commands from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv-precommit
source .venv-precommit/bin/activate
python3 -m pip install pre-commit pyrefly==1.4.0 'tomli>=2; python_version < "3.11"'
python3 -m pre_commit install
```

Keep the environment activated when committing or running hooks. The wrapper
uses `python3` and `pyrefly` from `PATH`; installing packages with `python3 -m pip`
keeps them in the same environment. Python 3.10 needs `tomli` to read the
configuration. Python 3.11 and newer include `tomllib`.

This lightweight setup does not install all project dependencies. For a complete
development environment, follow the [installation guide](../installation/installation-guide.md)
and install the project's development requirements in that environment:

```bash
python3 -m pip install -r requirements-dev.txt
```

Use the Pyrefly version pinned in [requirements-dev.txt](source:requirements-dev.txt).
An unresolved dependency may reflect the environment rather than a bad import.

## Check changed files before committing

After the one-time setup above, follow these steps from the repository root for
each change:

1. Activate the environment, review your changes, and stage only the intended
   files. Include new source files explicitly; avoid staging generated reports.

   ```bash
   source .venv-precommit/bin/activate
   git diff
   git add path/to/file1 path/to/file2
   git diff --cached --stat
   ```

2. Run all applicable pre-commit hooks on the staged files:

   ```bash
   pre-commit run
   ```

3. If hooks modify files, review those edits, restage the affected files, and
   rerun the checks:

   ```bash
   git diff
   git add path/to/modified_file
   pre-commit run
   ```

4. Review the hook diagnostics, including advisory findings from hooks that
   return success. Read [the type-check report](#read-the-report) when the
   current run prints its path. If the hook was skipped and you need a fresh
   report, run `bash scripts/run_pyrefly.sh --full-project`. Fix failures and
   rerun the checks.

`pre-commit run` selects staged files. `pre-commit run --all-files` selects all
tracked files. Pre-commit skips the `type-check` hook when none of the selected
files match its file pattern.

The hook scans registered mocks for product, test, checker, and configuration
inputs. A direct full-project run also includes the mock scan. Pass
`--no-mock-interfaces` to the wrapper for a Pyrefly-only run.

## Other check modes

```bash
# Only the type-check hook on staged files.
pre-commit run type-check

# Check selected files, including unstaged changes.
pre-commit run type-check --files path/to/file1.py path/to/file2.py

# Supply every tracked file to the type-check hook.
pre-commit run type-check --all-files

# All applicable hooks on every tracked file.
pre-commit run -a

# Check every line in selected files rather than filtering by the Git diff.
bash scripts/run_pyrefly.sh --no-diff-filter path/to/file.py

# Scan all configured product and test files, without diff-line filtering.
bash scripts/run_pyrefly.sh --full-project

# Fail a file-list check on remaining diagnostics or incomplete checks.
bash scripts/run_pyrefly.sh --strict --no-diff-filter path/to/file.py
```

Pre-commit passes filenames to the wrapper. Ordinary file-list diagnostics are
filtered to added and modified lines relative to `HEAD` when Git provides a
nonempty set of changed lines; otherwise they are retained at file level.
`--all-files` expands the supplied file list; it does not disable this filtering.
Use `--no-diff-filter` or `--full-project` when reviewing errors outside the diff.
The wrapper does not rebase or create branches/worktrees.

With no arguments, the wrapper skips checking. Checker/config-only inputs run
registered mock checks without an ordinary source-file pass. Full-project checks
always return success, even with `--strict`. For file-list checks, `--strict`
changes the exit policy; it does not enable additional diagnostic categories.
Jenkins and the pre-commit hook remain advisory.

## Read the report

The default output is `pyrefly_final_report.json` in the repository root. Set
`PYREFLY_REPORT_PATH` to choose another destination. File-list checks print all
remaining diagnostics and the report path. Full-project checks save diagnostics
and print only the report's absolute path.

- `result.errors` contains the remaining diagnostics, with paths, locations,
  error names and descriptions. Mock diagnostics have the `mock-interface/`
  prefix and `source = "mock-interface"`.
- `remaining_diagnostic_count` counts the diagnostics after the applicable filters.
- `pyrefly_skipped` is set when a direct run checks only mock interfaces.
- `mock_interfaces.results` contains details only for checks with diagnostics or
  incomplete verification. Successful and cleanly skipped checks are omitted.
  `mock_interfaces.registrations` and `counts` still summarize all checks;
  `analysis_errors` and `check_exit_code` retain check failures.
- A `report_error` means a check did not complete. Resolve that failure before
  treating an empty diagnostic list as a successful check.

Pre-commit can split a large file list into multiple invocations. The terminal
log contains each invocation's output; the report file contains the latest
invocation. Use `--full-project` for a single report of all configured sources.

## Choose a fix

Start with the diagnostic location and the definition being used. Check the
actual function signature, imports, and initialization paths before editing the
caller. Preserve the behavior and assertions that the test is intended to cover.
The configured diagnostic categories are in
[pyproject.toml](source:pyproject.toml), under `[tool.pyrefly.errors]`.

### Errors to fix in code

| Diagnostic | How to choose the fix |
| --- | --- |
| `parse-error` | Correct invalid Python syntax first; subsequent diagnostics may be consequences of the parse failure. |
| `missing-import` | Check spelling and the module's location. Install a declared dependency in the checking environment, correct the import, or expose an existing local test-helper directory as described below. Do not add a search path for a module that does not exist. |
| `unknown-name` | Import or define the referenced symbol. For annotations that must not import a dependency at runtime, use a `TYPE_CHECKING` import. A `global` declaration is appropriate only when referring to an actual module-level binding; it does not initialize a missing value. |
| `unbound-name` | Find a path that reaches the use before assignment. Initialize a meaningful value, handle the missing-value branch, or keep the use within the branch that assigns it. Do not invent defaults or add new exceptions solely to silence the checker. |
| `missing-argument` | Compare the call with the current constructor/function signature and supply the required value. For a factory whose return type is too broad, correct its annotation or narrow the factory before adding arguments that some alternatives do not accept. |
| `unexpected-keyword` | Check for a renamed or removed parameter, a wrong callable, or an inaccurate wrapper signature. For Pydantic aliases, confirm that the model accepts the input; a mapping passed to `model_validate()` can preserve alias-based validation. |
| `bad-argument-count`, `unexpected-positional-argument` | Check argument order, keyword-only parameters, and fixed tuple unpacking. Pass keyword-only arguments by name. Give variable argument tuples an accurate fixed-length annotation when their length is known. |

For example, when a type is used only in an annotation:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .specs import PageAlignedLayout


def validate_layouts(layouts: dict[int, dict[str, "PageAlignedLayout"]]) -> None:
    ...
```

This makes the type visible to static analysis without importing it during
normal execution. If runtime code needs the class, use a normal import instead.
A quoted annotation such as `"PretrainedConfig"` postpones runtime evaluation;
it does not define the name for the checker. Import or define the referenced
type even when the annotation is quoted.

For an optional result, initialize and handle the missing case consistently:

```python
rebalances = None
if enable_rebalance:
    rebalances = arm_rebalance(llm)

if rebalances is not None:
    verify_rebalances(rebalances)
```

Use this pattern only if skipping verification when rebalancing is disabled is
the intended behavior. If every path must produce a result, fix the missing
assignment instead. Do not weaken a test by removing its assertions.

### Errors to ignore at an intentional call site

A negative test may deliberately omit a required argument or pass an unsupported
keyword. Keep that invalid call and add a diagnostic-specific annotation:

```python
with pytest.raises(TypeError):
    # pyrefly: ignore [missing-argument]
    requires_value()
```

The annotation applies to the following statement. An inline
`# pyrefly: ignore [missing-import]` can similarly suppress one optional import.
Use a narrow ignore only after confirming that the code is intentional or that
static analysis cannot represent the supported runtime behavior. Being inside
`pytest.raises` alone is not justification: a typo could raise the expected
exception for the wrong reason.

Use the error name from the current Pyrefly report. Older examples may use mypy's
`name-defined`; Pyrefly reports categories such as `unknown-name` instead.
For an unsupported but valid DSL annotation, suppress only the reported
diagnostic at that annotation rather than changing the kernel's runtime contract.

### Errors to filter for a shared unsupported dependency

Prefer fixing the environment or import first. A shared exclusion is appropriate
for generated bindings, optional third-party dependencies, or dynamic symbols
that cannot be resolved reliably across supported checking environments.
Choose the narrowest relevant setting in `pyproject.toml`:

| Setting | Use |
| --- | --- |
| `[tool.pyrefly].project-excludes` | Exclude an intentionally unsupported source subtree from ordinary checking. This is broader than ignoring a single diagnostic. |
| `[tool.pyrefly].ignore-missing-imports` | Tolerate a missing external module while keeping its types when it is available. This does not suppress arbitrary undefined names. |
| `[tool.pyrefly-exclusions].symbols` | Apply the wrapper's additional symbol filters for supported diagnostic forms. This is custom wrapper configuration, not a native Pyrefly setting. |
| `[tool.pyrefly-tests].search-path` | Expose existing local helper modules to the test pass without changing product import resolution. |

Product checking uses the project root with fallback search disabled. Tests run
with a separately generated configuration, broader helper paths, and fallback
resolution. For existing tests that declare extra runtime import locations, the
wrapper reads literal lists or tuples without executing test code:

```python
__extra_import_path__ = ["../comm"]
```

Existing directories are added to the test search path; ancestor directories use
per-file fallback instead. Do not add broad test roots to the product search path:
they can shadow installed dependencies such as `grpc` or `evaluate`.

File-list checks also filter missing imports of declared but uninstalled test
dependencies. Full-project scans retain those diagnostics. To investigate the
configured path/module and symbol exclusions, run:

```bash
bash scripts/run_pyrefly.sh --no-path-module-filter --no-diff-filter tests/unittest/others/test_mapping.py
```

## Check mocks against production interfaces

Pre-commit, direct, and Jenkins runs scan all registered mock interfaces,
including those in unchanged tests. These diagnostics bypass ordinary diff-line
and path/module filters because a production change can invalidate a test double
elsewhere in the repository.

Use `@mock_target("module:ClassOrFunction")` for a named mock class or function.
Use `verify_instance(value, "module:Class")` after setting up a namespace or an
instance created without its real constructor. The mock check compares member
names, initialization, and callable signatures. For a check limited to direct
member presence and initialization, use `members_only=True`.

The configured Pyrefly diagnostics cover parse errors, missing imports, unknown
and unbound names, incorrect argument counts, missing arguments, and unexpected
keyword or positional arguments. Argument-type checking is disabled, so an
incompatible mock member type can pass both checks.

A missing field or incompatible signature should normally be fixed in the test
double to reflect the real interface. Do not change production interfaces just
to accommodate a mock. `unverified` means that source resolution or
static analysis could not establish the required structure; it is not a successful match. These checks
are static and do not execute the registered classes/functions. Unregistered
mocks are not automatically verified, and compiled `tensorrt_llm.bindings`
interfaces remain excluded.

See the [mock interface helper guide](source:tests/test_common/interface_checks.md)
for registration examples, supported patterns, and analysis limits.

After fixing an error, rerun the selected files with `--no-diff-filter`, inspect
the report, and run pre-commit again. Type checking complements runtime tests;
it does not establish numerical correctness or behavioral equivalence.
