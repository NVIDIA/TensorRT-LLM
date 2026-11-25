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

# Static mock interface checks

Use `test_common.interface_checks` to associate test doubles with their real
interfaces, next to the test code that defines them. Run
`scripts/run_pyrefly.sh` with product/test paths (or `--full-project`) to verify
those associations alongside normal type checking against the
**current source**. There is no central binding list or saved interface baseline.
Production files do not need decorators, imports, or other changes.

The `type-check` pre-commit hook and direct wrapper runs scan all registered
mocks for product, test, checker, and configuration inputs. The Jenkins
full-project scan also checks registered mocks. Their diagnostics bypass
diff-line and path/module filtering so changes to a production interface can
expose incompatible mocks in unchanged tests.
Results are saved under `mock_interfaces` in
`pyrefly_final_report.json`; mock diagnostics also appear in `result.errors`.
File-list checks print diagnostics, while full-project checks print only the
report path. Checks are advisory unless `--strict` is used for a file-list check;
strict checks fail on mismatches or incomplete verification. With no arguments,
the wrapper skips both checks. The standalone `scripts/check_mock_interfaces.py`
command remains available for focused mock checks.

The checker reads Python source with the AST and never imports or executes the
registered modules. Both helpers preserve the supplied object at runtime.

The mock check compares member names and kinds, callable parameter names, order
and kinds, literal defaults, async/generator behavior, and which fields have
been initialized at the registration point. Field value types and callable
annotations are outside this check. Pyrefly still checks product and test
source normally through the same hook.

Registrations targeting compiled `tensorrt_llm.bindings` interfaces are skipped.
For a Python subclass, the checker records the excluded compiled base in
`skipped_checks` and still checks the Python-defined members. An otherwise
clean registration that excludes a compiled interface has status `skipped`.

## Named mock classes and functions

Import the helpers at module scope, including when the decorated mock is local to
one test. Targets use a literal `"module:ClassOrFunction"` string. Import aliases
for the helper are supported.

```python
from test_common.interface_checks import mock_target

# Illustrative target: production.client:Client has this same declaration.
@mock_target("production.client:Client")
class FakeClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, text: str, *, timeout: float = 1.0) -> str:
        return "test response"

# Use the function normally with monkeypatch.setattr(), patch(..., new=...),
# or as a callback. The association belongs to its definition.
@mock_target("production.client:fetch")
def fake_fetch(key: str, *, retry: bool = False) -> bytes:
    return b"test data"
```

Class checks compare the complete statically declared interface, including
inherited members, constructors, private members, properties and class variables.
Function and method checks compare parameter names, order, kinds, literal defaults,
async/generator kind and method binding.
Overload declarations are compared in order. Function bodies may behave
differently; this is an interface check, not a behavioral equivalence test.

Method targets use the signature accessed through the class: ordinary methods
include their receiver, while classmethods and staticmethods expose their bound
call signatures. Preserve `classmethod` or `staticmethod` on the replacement
when the target uses that descriptor. Use
`@mock_target("production.client:Client.send", bound=True)`
for a replacement assigned directly to an instance, where Python does not insert
`self`. The checker resolves the declaring method from source; registering the
entire owning class is not required to compare an individual method.

Inline replacements can retain their original form:

```python
monkeypatch.setattr(
    client_module,
    "fetch",
    mock_target("production.client:fetch")(lambda key: b"test data"),
)
```

The marker returns the same callable. Lambda signatures, including lambdas
wrapped with `staticmethod` or `classmethod`, are checked. A dynamic `Mock` remains
unverified: neither its target nor `spec` proves that the replacement implements
the interface. Prefer a decorated named definition when one already exists.
For mocks created implicitly by `patch`, mark the patch expression:
`mock_target("production.client:fetch")(patch("production.client.fetch"))`.
The context manager or decorator is returned unchanged. Use `autospec=True`
when keeping the mock's call history. The checker verifies that the patch and
registration resolve to the same callable. For `patch.object`, pass a class and
a literal method name. An ordinary autospecced method includes `self` in its
recorded calls.

An additional optional parameter or mock-only public/private field is a mismatch,
even if ordinary structural typing would accept it. Registering a partial fake
audits its differences; it does not declare it an exact replacement or make it
pass. Missing fields remain errors. A target source file or `.pyi` must be
available to the analyzer.

## Instances that bypass initialization

Place a separate `verify_instance` statement after setup, at the point where the
object must be ready for use. Class identity alone does not prove its fields were
initialized.

```python
from production.client import Client
from test_common.interface_checks import verify_instance


def test_client():
    client = object.__new__(Client)
    client.endpoint = "test://endpoint"
    verify_instance(client, "production.client:Client")
    # Test operations follow the checkpoint.
```

`Client.__new__(Client)` is also recognized when the checker can establish
that the class inherits the default allocator. Allocator assignments outside
the class declaration can make that call unverified. Required fields come from
class annotations and direct `self.field` assignments in the class and its
single-inheritance chain. An optional type does
not make initialization optional: `field: str | None` still needs a value.
Existing class defaults are available to an instance. A bare annotation in the
test (`client.endpoint: str`) does not initialize the field.

The checker follows direct assignments, simple aliases, literal `setattr`, field
deletions and `if`/`else` joins. A field written on only one branch is not definitely
initialized after the join. An unresolved mutation helper prevents a
successful verification; it is not assumed to initialize or preserve fields.
This covers objects reachable through nested fields and containers. A helper
can mutate an object captured by a callback, including an object retained by a
default argument after its original name is rebound. A conditional alias can
also leave the affected objects unverified after a mutation.
The checkpoint describes that program point only, not later mutations.
An instance checkpoint can also be used inline, for example
`consume(verify_instance(state, "production.client:Client"))`; the checker
uses the tracked fields at that point, and the helper returns `state` unchanged.
Checkpoints inside deferred lambda bodies remain unverified because their state
at invocation cannot be inferred from the state when the lambda is created.

Use `members_only=True` for a first-layer presence and initialization check:

```python
verify_instance(previous_batch, "production.executor:BatchState", members_only=True)
```

This checks direct member names and whether required members are initialized.
It does not check member-value types, callable signatures, or nested members
inside values such as `previous_batch.sample_state`. Missing members and
unexpected members still fail; `allow_extra=True` separately permits extras.
Plain dataclass fields are supported. A real instance can use class defaults,
but fields with a `default_factory` must be assigned when bypassing `__init__`.
A namespace must supply all fields itself. Reports label this scope
`direct-members`. The default scope also checks callable signatures.

Use a literal `members` tuple when the fixture intentionally implements only
part of the target's interface:

```python
verify_instance(creator, "production.cache:KvCacheCreator", members=("_cache_transceiver_config",))
```

`selected-members` checks that each named member is declared on the production
class and available on the fixture at the checkpoint. Additional fields are
allowed. For a `Mock`, assign the selected fields explicitly before the
checkpoint. Attributes synthesized by the mock do not establish initialization.

## Attribute-only namespaces

For simple data interfaces, register the constructed value:

```python
from types import SimpleNamespace
from test_common.interface_checks import verify_instance


def test_settings():
    settings = SimpleNamespace(limit=8, enabled=True)
    verify_instance(settings, "production.settings:Settings", allow_extra=True)
```

This compares the supplied fields with the target's instance interface. Namespace
objects must supply fields themselves, including fields with defaults on the real
class. `allow_extra=True` permits additional namespace fields while still checking
every required member. Omit it for an exact member-set
check. The flag must be a literal boolean so the checker can read it statically.
Callable namespace members currently need
separate named mock definitions; their equality is not inferred from an arbitrary
callable value.

## Run and retain results

```sh
# Scan all test files for registrations. Run this after production changes too.
python3 scripts/check_mock_interfaces.py

# Focused development check; only the selected files are scanned.
python3 scripts/check_mock_interfaces.py tests/unittest/path/to/test_file.py

# Additional source/stub roots, without importing those packages.
python3 scripts/check_mock_interfaces.py --search-path /path/to/source-or-stubs \
    --report /tmp/mock-interface-report.json

```

The standalone command's default report is `mock_interface_report.json` in the working directory.
Console output identifies the registration, target and individual discrepancies.
JSON records include location, diagnostics, status and whether analysis was
complete. `verification` is `ast-only`. A file without registrations is not
checked for mock consistency. A scan with zero registrations reports that no
interfaces were verified.

| Result | Meaning |
|---|---|
| `matched` | The supported static declarations/state agree. |
| `mismatched` | At least one definite difference was found. |
| `unverified` | Available source or supported analysis cannot establish agreement. |

Exit status is `0` for completed checks with no differences, `1` for definite
mismatches, or `2` for analysis errors or incomplete verification. A mismatch can
also carry incomplete diagnostics; in that case the exit status is `2` and the
JSON retains both findings.

The wrapper runs the static mock scan with the installed `python3`. Discrepancies
and incomplete checks appear in the combined JSON report. The
`mock_interfaces.check_exit_code` field records the mock check status. Use
`scripts/run_pyrefly.sh --strict` with file paths to fail on diagnostics or
incomplete checks.

## Current analysis boundaries

The checker can verify explicit functions and classes and track simple instance
setup. Unregistered mocks are outside its scope. Dynamic mocks, unresolved
bases, generated interfaces, nonliteral defaults, and helper calls that can
mutate tracked instances can leave a registration unverified.

The CPU regression tests are in
`tests/unittest/scripts/test_mock_interface_registrations.py`:

```sh
python3 -m pytest --noconftest -c /dev/null -o 'markers=cpu_only: CPU-only tests' \
    tests/unittest/scripts/test_mock_interface_registrations.py -q
```

## Initial coverage on verify_mocks

The first 15 test files were selected by the number of distinct patterns in
`test_mocks_standins_inventory.html`, with evidence count breaking ties.
Registrations cover selected callbacks, inherited stand-ins, and batch state;
other doubles in those files are not automatically verified.

Run all registered checks with:

```bash
python3 scripts/check_mock_interfaces.py --report mock_interface_report.json
```

Pre-commit runs the same scan in advisory mode when product or test Python
files change. A successful advisory hook does not mean that every registration
matched; inspect `mock_interfaces.counts` and `mock_interfaces.analysis_errors`
in `pyrefly_final_report.json`.
