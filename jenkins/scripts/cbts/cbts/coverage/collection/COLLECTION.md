<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS Coverage Collection

How per-test coverage data is produced, and **what it does not record**.
Consumer side: `../selection/SELECTION.md`. File-by-file roles: `README.md`.

---

## 1. What is recorded

One set per test: **which configured Python functions it entered**.

```
(test, file, qualname)
 │      │      └─ the code object's co_qualname, e.g. "PyExecutor._executor_loop"
 │      └─ repository-relative source path, e.g. examples/llm.py
 └─ "<STAGE>/<pytest nodeid>"
```

No line numbers, no call counts, no call graph.

The default runtime config collects `tensorrt_llm/**/*.py` (from the checkout and the installed
wheel), `examples/**/*.py`, `scripts/**/*.py`, `tests/test_common/**/*.py`, the checkout's root
`conftest.py`, and every `tests/**/conftest.py`.
`[run] source` bounds the physical source trees; the bootstrap resolves `%(repository_root)s` from
its own checkout location. That root maps checkout paths to stable
repository-relative names; `include` selects those names using `fnmatch` patterns (`*` also matches
directory separators). Installed wheel paths still map to `tensorrt_llm/...`, so wheel and checkout
copies merge into the same file identity. Other Python files under `tests/` are excluded.

Collection scope is independent of selection eligibility: the coverage selector still accepts only
core Python residuals. Extra rows accumulate evidence for future support.

## 2. How

`sys.monitoring`'s `PY_START` event (Python 3.12+, tool id 4):

```python
def _on_py_start(self, code, offset):
    fn = code.co_filename
    if self._in_source(fn):
        qual = code.co_qualname
        if qual.rpartition(".")[2] not in _SKIP_QUALNAMES:
            self._data[self._ctx].add((fn, qual))
    return _MON.DISABLE
```

- Each code object fires once on **first entry**, then `DISABLE`s itself: zero cost afterwards.
    - Note: for generator or coroutine, that's the first resume, not the function invocation.
- Each test re-arms with `restart_events()`, so every test gets its own complete function set.
- Cost scales with how many **distinct** functions were entered, not with how often.

The reported qualname is inherently non-unique: Python doesn't enforce every function definition
to have distinct name, and the `@property` idiom even recommends sharing name for getter and setter.
Since we are handling *change*, the line number and code itself are also not reliable identity to
distinguish same-name functions.

### 2.1 Closures

A closure created by a function / class definition, like a decorator's `wrapper`, a registered callback or
the inner function of a cached factory, is recorded normally under its full dotted `co_qualname`
(e.g. `my_decorator.<locals>.wrapper`). Closures created by lambda and generator expression are excluded
because a difficulty in the static analysis when selection tests.

## 3. Crossing processes

`CBTS_COVERAGE_CONFIG` and `PYTHONPATH` are inherited at exec, so `sitecustomize.py` installs the
tracker in every Python process in the tree. No product code is modified.

A process learns what it is from `CBTS_PROCESS_ROLE`, set by whoever launches it and popped by
`sitecustomize.py` so it describes exactly one process:

| Role                                                          | Set by                                     | Context source                                                                               | Activation                                                                                                                                         |
|---------------------------------------------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `outer_pytest`                                                | `jenkins/L0_Test.groovy`                   | `cbts_plugin`'s `pytest_runtest_protocol` switches it directly                               | at interpreter startup                                                                                                                             |
| `inner_pytest`                                                | `tests/integration/defs/test_unittests.py` | inherited `CBTS_TEST_ID`, held for the whole batch                                           | at interpreter startup                                                                                                                             |
| *(unset)* - pool workers, `trtllm-serve`, helper subprocesses | -                                          | inherited `CBTS_TEST_ID` until it subscribes to the context channel, then every announcement | immediate, at interpreter startup - except an `mpi4py.futures` pool worker, deferred to its `_sync_set_data` sync-handshake hook (see `README.md`) |

A subscriber finds the channel through `CBTS_CONTEXT_SOCKET`. Ordinary children inherit it at exec;
MPI pool workers are spawned from an environment the MPI runtime captured before `pytest_configure`
bound the socket, so the patched `MPIPoolExecutor` forwards the address through `mpi4py`'s own `env`
payload, which the worker applies during its sync handshake.

Processes that opt themselves out: `pip` / `setup.py` / `cmake` / `ninja` and everything they spawn,
plus Ray infrastructure processes (`default_worker.py` and friends). These are spawned by pip and
raylet rather than by our own code, so they are recognised from their launch target (`-m` module or
script name).

## 4. Persistence and merge

Every process writes its own compact SQLite (`.cbtscov.<stage>.<host>.X<rand>.pid<N>.sqlite`), on a
5s periodic snapshot plus `atexit`; the pytest process also flushes at `pytest_sessionfinish`. A
channel subscriber additionally flushes synchronously on every context switch, before ACK-ing it
(`ContextSubscriber._pump`) — the periodic snapshot is a backstop for a test still in progress, not
what a confirmed-dead subscriber's already-superseded tests rely on. Leaf, platform-level and final
databases share schema version 5, so one reducer merges every level:

| Table                                            | Contents                                                  |
|--------------------------------------------------|-----------------------------------------------------------|
| `stage`, `case_stage`                            | interned stage names and bare test contexts               |
| `file`, `symbol`                                 | interned product paths and qualnames                      |
| `touch(case_stage_id, symbol_id)`                | which test context entered each symbol                    |
| `process`                                        | stable process identity                                   |
| `test_result`                                    | coordinator outcome observations                          |
| `taint(process_id, case_stage_id, kind, reason)` | coverage the context channel could not vouch for          |
| `process_diagnostic(process_id, key, value)`     | diagnostic only: arbitrary facts recorded about a process |

Currently, the `process_diagnostic` table contains the subscribers' cmdline and inherited pytest case.
They can be used to understand when and why they dropped unexpectedly, is that happens later.
If other requirements arise in the future, other facts can be recorded using their own keys.

The merge (`pystart_report.py`) resolves each input's local IDs by its natural keys and unions the
relations. The final artifact is `cbts_touchmap.sqlite`; the `touch_rows`, `taint_rows`,
`process_diagnostic_rows` and `test_case_meta` views reconstruct its logical query rows. Join
`process_diagnostic_rows` on `taint_rows.process` to pair a taint with whatever diagnostics its
process has — a process with none known simply has no matching rows.

---

## 5. What is not recorded (important)

This section is what forces the concessions on the consumer side.

### 5.1 Anonymous constructs

`_on_py_start` drops `<lambda>` and `<genexpr>` frames (`_SKIP_QUALNAMES`): selection has no
static-analysis path to resolve a changed line to either, so a recorded row would have no consumer; see
`../selection/SELECTION.md` §4.1 for why. This is conservative, not a hard limit.

Note: list/set/dict comprehensions need no entry here either, for an unrelated reason - PEP 709 inlines
them into the enclosing frame on Python 3.12+, so no code object exists for either side to see.

### 5.2 PEP 695 Generics

PEP 695 generic syntax (`def f[T](...)`, `class C[T]:`, `type X[T] = ...`, Python 3.12+) compiles an
implicit type-parameter scope whose qualname is `<generic parameters of f>`. We don't use it in product code
and don't handle it yet.

### 5.3 The outer pytest's own import

Every product process activates `PY_START` and records its own import — immediately for most, or at
its mpi4py sync handshake for an `mpi4py.futures` pool worker specifically, which still precedes any
product code — see `README.md`'s "Process roles" and "The context channel" sections.

The outer pytest's import happens before any test, under the empty context, and the merge filters it
out with `WHERE test != ''`. So the outer pytest's own module-level and class-level import-time code (as opposed to a
subprocess's) earns no rows from any test, by construction — nothing to fix here,
since nothing could ever attribute it to one test over another.

### 5.4 Pytest fixtures and hooks

Function-scoped fixtures and runtest hooks in `conftest.py` are recorded under the current test.
Outer-pytest imports and collection hooks run before a test context exists, so their rows carry the
empty context and do not count as per-test coverage. A shared fixture's setup is recorded under its
first consumer and teardown under its last; cached reuse by another test enters no Python frame.
These rows describe execution, not every test that depends on a shared fixture. Selecting tests for
`conftest.py` changes requires additional dependency and scope handling.

### 5.5 Other blind spots

| Blind spot                                     | Reason                                                                                                                                                     |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C++ / nanobind implementations                 | PY_START only sees Python frames; the C++ side of KV cache manager, scheduler and decoder is invisible                                                     |
| Other test bodies and helpers                  | Under `tests/`, the default include patterns select only `test_common/` and `conftest.py`                                                                  |
| A Ray stage's GPU worker                       | `RayGPUWorker` lives in the opted-out `default_worker.py`                                                                                                  |
| Multi-GPU / multi-node stages                  | phase 1 collects on single-GPU stages only; the channel design is single-producer, so a multi-node stage would need its address published somewhere shared |
| A channel failure inside an inner-pytest batch | `test_unittests_v2` has no `cbts_plugin` (§3's `inner_pytest` row), so nothing records a taint if a subscriber it spawns falls behind                      |

---

## 6. The completeness signal the data carries

`test_case_meta` lets the consumer decide whether a record can be trusted:

```sql
outcome IS NULL OR outcome != 'passed' OR tainted != 0
```

The two terms cover two ways a record can fall short:

- **`outcome`** — a test that did not pass may not have reached the code it normally covers.
- **`tainted`** — the context channel could not vouch for the attribution of some rows, or for their
  completeness. `README.md` documents the kinds, the reasons and their scopes; `taint_rows` gives the
  detail per `(process, test)`.

Together they are the only way to detect lost or misattributed data. None of it is visible in the
rows themselves: the footprint stays large and the usual `py_executor` symbols are still there
whether or not the record is sound.

---

## 7. When collection runs

`L0_MergeRequest.groovy` decides pipeline-level eligibility and `isCbtsStage()` decides each stage:

- official post-merge pipeline only (`ENABLE_CBTS_COVERAGE && JOB_NAME ==~ /.*PostMerge.*/`)
- not a perf stage or a TensorRT / CPP stage
- single-GPU stages only (name carries no `-<N>_GPUs` / `-<N>_Nodes`)
- not listed in `CBTS_EXCLUDE_STAGES`

The per-process files ride back inside `results-<stage>.tar.gz`; architecture checkpoints first
upload `cbts_pystart_report_x86_64.tar.gz` and `cbts_pystart_report_SBSA.tar.gz`, which the selector
requires and hierarchically merges. The later `Test Coverage` stage merges all files and uploads
`${UPLOAD_PATH}/cbts-coverage/cbts_pystart_report.tar.gz` for the full report.
