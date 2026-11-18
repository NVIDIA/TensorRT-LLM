<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS Layer C — Coverage Collection

CI tooling that captures per-test **function/class-level** coverage — which product functions each
test entered — across a stage's whole process tree. CI infrastructure only: nothing ships in the
product wheel, and every file here is a no-op unless `CBTS_COVERAGE_CONFIG` is set.

Capture uses `sys.monitoring` `PY_START` (Python 3.12+). Each function a test enters fires the
callback once, then that code object is disabled until the next test, so overhead scales with the
number of distinct functions entered rather than with lines executed.

`COLLECTION.md` describes the data model and its blind spots; `../selection/SELECTION.md`
covers how the consumer uses the result.

## Files

This package (`cbts.coverage.collection`) holds the library code; the two files Python's `site`
machinery / pytest actually load by bare name live outside the `cbts` package, in
`jenkins/scripts/cbts/cbts_injectors/`, and import this package normally.

| File | Role |
|---|---|
| `pystart.py` | The tracker: a `sys.monitoring` tool (id 4) recording, per test context, the set of product `(file, qualname)` entered, plus outcomes and taints. Writes one compact SQLite per process. |
| `../../../cbts_injectors/sitecustomize.py` | The bootstrap: starts the tracker in every Python process under `CBTS_COVERAGE_CONFIG`, reads the rcfile and the process role, subscribes product processes to the context channel, and saves periodically so coverage survives a non-clean exit. |
| `../../../cbts_injectors/cbts_plugin.py` | The pytest plugin (`-p cbts_plugin`): owns the context channel, announces each test, records outcomes, and flushes the session's record at the end. |
| `channel.py` | The context channel: an `AF_UNIX` broadcast of the current test from the pytest process to its subprocesses, plus the taint records produced when that guarantee fails. |
| `pool.py` | The `MPIPoolExecutor.__init__` patch, forwarding the channel address to spawned pool workers. Kept apart from the plugin so the bootstrap can use it without importing pytest. |
| `compact_db.py` | The compact schema, the leaf writer, and the merge-closed reducer. Strings live once in dimension tables; process identity makes hierarchical merges idempotent. |
| `hooks.py` | Explicit callback registry the bootstrap installs into when active, so peers (the pool patch, the pytest plugin) can call back without importing the guest-injected `sitecustomize` module. |
| `import_watch.py` | The generic meta-path import-completion watcher (run a callback once a named module finishes executing), used by the bootstrap to install the pool patch as `mpi4py.futures` finishes importing, and to patch `_sync_set_data` as `mpi4py.futures._lib` finishes importing. |
| `process_roles.py` | Generic launch-target detection: opts pip/cmake/ninja/Ray-infra subtrees out of instrumentation, and identifies an `mpi4py.futures` pool worker so only that process type defers activation. |
| `../../command/coverage/collection/pystart_report.py` | The CLI (`cbts.command coverage collection pystart-report`): unions compact databases and emits `--out-sqlite` (the selector artifact), `--out-dir` (a split HTML report) and `--out-json` (the full `test -> [file::qualname]` map). |
| `coveragerc.template`, `make_coveragerc.sh` | The runtime rcfile: `[run] source`, `include` and `data_file`, with `@...@` placeholders substituted into `$JOB_WORKSPACE/.coveragerc`. |

The default scope covers library Python from both the installed wheel and checkout, all Python
under `examples/`, `scripts/` and `tests/test_common/`, and the root and `tests/` conftest files.
The Python bootstrap resolves `%(repository_root)s` from its own location in the checkout, so K8s
and SLURM use the checkout visible to their test process. Include patterns match repository-relative
paths with `fnmatch` semantics (`*` also matches `/`). Extending collection does not extend the
coverage selector's core-Python eligibility rules.

Source roots and filenames resolve symlinks before membership checks. Installed
`site-packages/tensorrt_llm/` and `dist-packages/tensorrt_llm/` paths retain the same library
identity even when the virtual environment is inside the checkout. The report skips unrelated
top-level directories when the configured include patterns name literal top-level paths.

## When it runs

`jenkins/L0_MergeRequest.groovy` decides pipeline-level eligibility and propagates it through
`testFilter[cbts_coverage]`: the `ENABLE_CBTS_COVERAGE` kill switch plus the post-merge gate, so
coverage runs on the official post-merge pipeline only.

`isCbtsStage()` in `jenkins/L0_Test.groovy` gates each stage on that flag plus:

- not a perf stage, and not a TensorRT / CPP stage
- single-GPU only — stages named with `-<N>_GPUs` or `-<N>_Nodes` are out of scope for phase 1
- not listed in `CBTS_EXCLUDE_STAGES`

Non-CBTS stages get an empty `.coveragerc` and run uninstrumented.

## Process roles

Every instrumented process is told what it is through `CBTS_PROCESS_ROLE`:

| Value | Set by | Effect |
|---|---|---|
| `outer_pytest` | `jenkins/L0_Test.groovy`, beside the other `CBTS_*` vars | tracker starts at once; the plugin owns context announcements and installs the pool patch |
| `inner_pytest` | `tests/integration/defs/test_unittests.py` | tracker starts at once; holds the inherited `CBTS_TEST_ID` for the whole batch and installs the pool patch itself |
| *(unset)* | — | product process: pool workers, `trtllm-serve`, helper subprocesses |

`sitecustomize.py` **pops** the variable, so a role describes exactly one process — the workers and
servers an outer pytest spawns come up unlabelled, which is what they are. `CBTS_STAGE`, set
alongside it, names the stage the coverage is attributed to.

Build tooling (`pip`, `setup.py`, `cmake`, `ninja`) and Ray infrastructure (`default_worker.py` and
friends) opt themselves and their spawned subtree out. pip and raylet launch these, so there is
nobody to hand them a role; they are recognised from their launch target — the `-m` module or the
script name, parsed out of the interpreter's own options.

A product process subscribes to the context channel, then enables `PY_START` — immediately, for
every process except one: an `mpi4py.futures` pool worker (`MPIPoolExecutor`'s dynamic
`MPI_Comm_spawn`, detected by `process_roles.is_mpi_pool_worker()`). See the next section for why
that one process type defers both steps.

## The context channel

The pytest process owns the current test and broadcasts it to its subprocesses over an `AF_UNIX`
socket. Frames are newline-terminated: `C <seq> <nodeid>` announces a context, `X <seq>` ends the
session, `A <seq>` acknowledges, and `I <process_uid>` identifies a subscriber as it joins.

Registering a subscriber and sending it the current context happen under one lock, so a process
joining while an announcement is in flight sees the old context or the new one, never neither. It is
on the right context before `subscribe()` returns, because the first frame is read there rather than
on the reader thread.

Subscribers acknowledge each announcement, and `announce()` waits for those acknowledgements before
the test body runs. That is what keeps a reused pool worker from spending the start of a test
answering to the one before it.

At session end the producer announces `X`, then waits for every subscriber to take its final save
and disconnect; once they have, the stage's output directory is quiet and the results tar can read
it. A subscriber that misses a deadline has its socket closed, which prompts that final save — a
signal would prevent it.

The channel's address is announced via the `CBTS_CONTEXT_SOCKET` environment variable. Everything
that gets it through ordinary process launch — a fork+exec'd child, or an `mpirun`-launched
`MPICommExecutor` sibling (`mpirun` is itself invoked as a subprocess of the already-running,
already-`pytest_configure`'d outer pytest, e.g. `tests/integration/defs/common.py`'s
`venv_mpi_check_call`, so it inherits the live environment, address included, same as any child) —
subscribes immediately at interpreter startup.

An `MPIPoolExecutor` worker is the one exception: it is spawned via `MPI_Comm_spawn`, which does not
inherit a live environment the way fork+exec does — mpi4py captures an environment snapshot once, at
`MPI_Init` time, likely before `pytest_configure` ever bound the channel. So a worker's own address
instead arrives later, forwarded by mpi4py itself: the coordinator's patched `MPIPoolExecutor.__init__`
(`install_pool_env_patch`) merges it into the `env` kwarg of every pool it builds, and mpi4py carries
that `env` dict inside the `options` payload applied by `_lib._sync_set_data`
(`os.environ.update(data.pop('env'))`) during the worker's MPI sync handshake — which always runs
before `server_exec` (the task loop, the only place product code runs).

Rather than attempt (and always fail) an immediate subscribe for this one process type, the bootstrap
defers its whole activation — subscribe and `PY_START` together — to that handshake: `sitecustomize.py`
wraps `_sync_set_data` itself (`_install_worker_sync_hook`), so it fires right as the address lands and
before any product code has run. Nothing is lost by waiting that long, since nothing would have run
yet anyway.

## Granularity

- **Integration tests**: the outer pytest carries `-p cbts_plugin`, so each test-db entry — one
  pytest item — is its own context.
- **Unit tests** (`test_unittests_v2[entry]`): the inner pytest carries no plugin, so the whole batch
  runs under the one inherited `CBTS_TEST_ID`, which equals the test-db entry. This matches CBTS's
  selection granularity.
- `co_qualname` gives `Class.method`, so results roll up function → class → file. A named closure
  (a decorator's `wrapper`, a registered callback) keeps its full nested qualname and rolls up the
  same way; only a generator expression or lambda frame is skipped, since neither has a positional
  identity to record one by. See `COLLECTION.md` §5.1.

## Taint

A touch row claims a test entered a symbol. That holds only if the recording process was on the
right context at the time, which is what the channel guarantees. Where it cannot, the affected rows
are **tainted**: kept, and marked as not to be trusted. Taint describes the map's fidelity, not
whether a test passed — `outcome` covers that.

Taint and `outcome` are recorded through different mechanisms because they come from different
subsystems: taint is the channel/subscriber system self-reporting an anomaly it detected, while
`outcome` is pytest's own result for the test, recorded directly by the plugin. They are not,
however, independent inputs to the consumer: a test that failed outright is not safe to skip
regardless of how trustworthy its coverage is, exactly like a tainted one, so
`TouchDB.incomplete_capture_tests()` combines both (plus an unsaved-process check) into the one "not
safe to skip" query rather than gating on taint alone.

### Two kinds

| `kind` | Claim | You cannot trust |
|---|---|---|
| `attribution` | The rows recorded under this test may belong to a different one. | what is **there** |
| `incomplete` | The rows recorded under this test are correctly attributed, but some are missing. | what is **absent** |

These are two halves of one event rather than alternatives. When a subscriber stops keeping up while
test A is current and test B starts, B's work keeps landing on A: A is `attribution`-tainted, and B —
with every test after it — is `incomplete`. One failure therefore writes rows of both kinds against
different tests.

Both force a re-run, so the selector treats them alike. The distinction is for whoever reads the
map: an `attribution` row says the coverage recorded here is suspect, an `incomplete` row says it is
sound but partial.

### Who records what

Two recorders, because they see different failures and neither sees the other's.

The **producer** sees subscribers that misbehave — one that stops acknowledging, or never leaves at
session end. It knows which announcement each failure belongs to, so it names individual tests.

A **subscriber** records the one failure the producer structurally cannot see: its own failure to
join. A process that never connected has no socket, no identity and no entry in any table. If it
does not flag itself, nothing does.

### Scope

A taint names `(process_uid, test, kind, reason)`, and the `test` carries the scope. A real nodeid
taints that test alone; the **empty context** taints every test in the stage, for a recorder that
cannot say which tests it covers.

Losing a subscriber spoils both halves at once: the context it was still recording under collects
the replacing test's work, and every test announced from the loss onward never receives its coverage,
since ejecting a subscriber closes its socket and makes it save and stop. The producer names both,
because it announces every test itself — it remembers which subscribers it lost and at which
announcement, then expands that into one row per affected test when the taints are read at session
end. Tests announced before the loss are untouched.

That fan-out is a default, not a certainty: an EOF the producer can trace to a confirmed-dead process
(a ticket file the subscriber held for its lifetime, checked as described in
`ContextServer._confirmed_dead`) narrows to the one test it was on, since a dead process cannot
record anything else, named or not. An EOF that is not confirmed dead falls back to the fan-out — the
subscriber may still be running, wedged on a stale context.

A process that fails to join cannot be that precise. Everything it records lands on the context it
was spawned with, so that context earns an `attribution` row; never having heard an announcement, it
cannot name the tests missing its coverage, so that half is stage-scoped. Where such a process really
did serve exactly one test the stage scope over-reports, forcing re-runs that were not needed — the
cheap error, chosen over the silent one.

`ContextSubscriber.subscribe()` takes an optional `diagnostics` mapping — arbitrary `{key: value}`
facts a subscriber wants remembered about itself, kept if it is later dropped for a reason worth
knowing more about it for (a confirmed death, or an ordinary voluntary leave — see
`ContextServer._promote_diag_locked`), since nothing can be read from it anymore once it has
exited. `sitecustomize.py` currently sends its own command line and, if it was forked from inside a
real pytest's test body (e.g. an inner-pytest batch's own worker), the `PYTEST_CURRENT_TEST` it
inherited — but the mechanism itself carries whatever is sent, with no fixed fact set. These reach
the merged `cbts_touchmap.sqlite` as `process_diagnostic`, queryable through
`process_diagnostic_rows` joined on `taint_rows.process` — see `COLLECTION.md`. Diagnostic only:
the taint's `reason` stays the stable `subscriber_confirmed_dead`
enum value regardless of what, if anything, is known about the process.

### Reasons

| Reason | Kind | Recorded by | Scope |
|---|---|---|---|
| `context_not_acknowledged` | `attribution` | producer | the **superseded** test — a subscriber that misses an announcement keeps recording under the context it still believes current, so that is where the next test's work lands |
| `subscriber_confirmed_dead` | `incomplete` | producer | the **one test it was on** — an EOF traced to a confirmed-dead process (its ticket is unheld); it cannot possibly have recorded anything past that |
| `did_not_finish_before_deadline` | `incomplete` | producer | the **last** test — a subscriber still connected when the drain deadline expired, so its final save may not have happened |
| `unreachable_on_subscribe` | `incomplete` | producer | the **current** test directly — a connection was accepted but the current context could not be delivered to it; the subscriber is also dropped, which taints every later test too, via `subscriber_stopped_recording` |
| `subscriber_stopped_recording` | `incomplete` | producer | **every test from the loss onward** — a subscriber ejected for missing an ACK, dropped for never receiving a context, or lost to an EOF that is *not* confirmed dead (it may still be running, wedged on a stale context) |
| `context_channel_unreachable` | both | the subscriber | `attribution` on its spawn context, `incomplete` stage-scoped |
| `no_context_channel` | both | producer | stage-scoped — `pytest_configure` could not bind, so every process spent the session on its spawn context |
| `py_start_unavailable` | `incomplete` | the process | stage-scoped — `sys.monitoring`'s `PY_START` tool failed to register, so this process records zero touches for its whole life |
| `pool_env_unavailable` | both | the coordinator | stage-scoped — forwarding the channel address into a spawned pool worker's env failed, so that worker has no later retry to find it either |
| `context_switch_failed` | both | the producer | `attribution` on the current test, `incomplete` stage-scoped — the producer's own per-test context switch or announcement to subscribers failed |
| `session_teardown_failed` | both | the producer | stage-scoped — something in session teardown failed outside the narrower paths that already self-taint |

### Storage

Taints live in the leaf database of whichever process recorded them, so they need no separate
artifact and no separate collection path: the ordinary `.cbtscov.*.sqlite` glob picks them up and the
ordinary reducer unions them.

A taint names the process whose rows are in doubt, which is not always the process that wrote the
file — the producer records taints about its subscribers, which is why subscribers send
`I <process_uid>` on connect, the same uid their own leaf database carries. Those uids are inserted
into `process` without implying the process ever ran a test to completion.

## Output

Every instrumented process saves every `CBTS_PERIODIC_SAVE_SECONDS` (default 5s), and the pytest
process flushes the session's whole record — touches, outcomes and taints — at
`pytest_sessionfinish`.

Per-process `.cbtscov.<stage>.<host>.X<rand>.pid<N>.sqlite` files ride back inside the standard
`results-<stage>.tar.gz` under `cbts/`. Travelling inside a compressed tarball keeps their plaintext
product paths away from the publish-artifacts guardword scanner, which byte-matches raw files and
does not recurse into archives.

`L0_MergeRequest.groovy` uploads SQLite-only x86 and SBSA tarballs once their single-GPU stages
finish; the coverage selector requires both and merges their compact databases. The later Test
Coverage stage merges every stage's files through `pystart_report.py` and uploads
`cbts_pystart_report.tar.gz`, holding `cbts_touchmap.sqlite` (the indexed selector artifact, with a
`meta` table carrying the coverage rate) and `cbts_report/` (the split HTML report; open
`cbts_report/index.html` after extracting).

For a coverage-rate report, pass the checkout as `--source-root`. The report reads include patterns
from the same template as the collector, so its denominator covers the same library, examples,
scripts, test_common and conftest scope.

Every database uses schema version 5. `case_stage` stores a bare test ID beside its stage ID, and
logical views reconstruct the selector's `<stage>/<test>` key. Integer IDs are local to a database,
so the reducer resolves stage names, test IDs, file paths, qualnames and process UIDs when merging.
`touch_rows`, `taint_rows` and `test_case_meta` expose the reconstructed logical rows.

## Query the touch DB

Which tests to run for a change, from `cbts_touchmap.sqlite`:

```python
import sqlite3
c = sqlite3.connect("cbts_touchmap.sqlite")
# file-level (phase 1): any test that entered a function in the changed file
c.execute("SELECT DISTINCT test FROM touch_rows WHERE file = ?",
          ("tensorrt_llm/_torch/pyexecutor/py_executor.py",)).fetchall()
# function-level (phase 2): tests that entered a specific function/method
c.execute("SELECT DISTINCT test FROM touch_rows WHERE file = ? AND qualname = ?",
          ("tensorrt_llm/_torch/pyexecutor/py_executor.py", "PyExecutor.forward")).fetchall()
# per-stage: coverage is attributed to the single-GPU stage it came from
c.execute("SELECT DISTINCT test FROM touch_rows WHERE file = ? AND stage = ?",
          ("tensorrt_llm/_torch/pyexecutor/py_executor.py", "RTXPro6000D-PyTorch-1")).fetchall()
# coverage rate + schema version
dict(c.execute("SELECT key, value FROM meta"))
```

A `(test, stage)` is safe to skip only when its coverage is complete: the test passed, and the
channel vouched for it. Force-run anything else:

```python
# rows NOT safe to skip
c.execute("SELECT test, stage FROM test_case_meta WHERE test != '' AND "
          "(outcome IS NULL OR outcome != 'passed' OR tainted != 0)").fetchall()
# every test whose coverage is in doubt, which half is in doubt, and why
c.execute("SELECT process, test, stage, kind, reason FROM taint_rows").fetchall()
```

## Smoke test

```bash
CBTS_DIR="$(pwd)/jenkins/scripts/cbts"
COV_DIR="${CBTS_DIR}/cbts/coverage/collection"
export TRTLLM_WHEEL_PATH=/usr/local/lib/python3.12/dist-packages
export JOB_WORKSPACE=/tmp/cbts_smoke STAGE_NAME=smoke
"${COV_DIR}/make_coveragerc.sh"
export PYTHONPATH="${CBTS_DIR}/cbts_injectors:${CBTS_DIR}:${PYTHONPATH:-}"
export CBTS_COVERAGE_CONFIG="${JOB_WORKSPACE}/.coveragerc"
export CBTS_STAGE="${STAGE_NAME}" CBTS_PROCESS_ROLE=outer_pytest
cd tests/integration/defs
pytest -p cbts_plugin -vs "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4"
cd "${JOB_WORKSPACE}" && PYTHONPATH="${CBTS_DIR}" python3 -m cbts.command coverage collection pystart-report --glob '.cbtscov.smoke*.sqlite'
```
