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
| `pystart.py` | The tracker: a `sys.monitoring` tool (id 4) recording, per test context, the set of product `(file, qualname)` entered, plus outcomes, expected worker counts and taints. Writes one compact SQLite per process. |
| `../../../cbts_injectors/sitecustomize.py` | The bootstrap: starts the tracker in every Python process under `CBTS_COVERAGE_CONFIG`, reads the rcfile and the process role, subscribes product processes to the context channel, and saves periodically so coverage survives a non-clean exit. |
| `../../../cbts_injectors/cbts_plugin.py` | The pytest plugin (`-p cbts_plugin`): owns the context channel, announces each test, records outcomes, and flushes the session's record at the end. |
| `channel.py` | The context channel: an `AF_UNIX` broadcast of the current test from the pytest process to its subprocesses, plus the taint records produced when that guarantee fails. |
| `pool.py` | The `MPIPoolExecutor.__init__` patch, counting each test's pool workers and forwarding the channel address to them. Kept apart from the plugin so the bootstrap can use it without importing pytest. |
| `compact_db.py` | The compact schema, the leaf writer, and the merge-closed reducer. Strings live once in dimension tables; process identity makes hierarchical merges idempotent. |
| `hooks.py` | Explicit callback registry the bootstrap installs into when active, so peers (the pool patch, the pytest plugin) can call back without importing the guest-injected `sitecustomize` module. |
| `import_watch.py` | The generic meta-path import-completion watcher (run a callback once a named module finishes executing), used by the bootstrap to detect the framework's import finishing. |
| `process_roles.py` | Generic launch-target detection: opts pip/cmake/ninja/Ray-infra subtrees out of instrumentation, and identifies an `mpi4py.futures` pool worker so only that process type defers `PY_START` activation. |
| `../../command/coverage/collection/pystart_report.py` | The CLI (`cbts.command coverage collection pystart-report`): unions compact databases and emits `--out-sqlite` (the selector artifact), `--out-dir` (a split HTML report) and `--out-json` (the full `test -> [file::qualname]` map). |
| `coveragerc.template`, `make_coveragerc.sh` | The runtime rcfile: `[run] source` and `data_file`, with `@...@` placeholders substituted into `$JOB_WORKSPACE/.coveragerc`. |

## When it runs

`jenkins/L0_MergeRequest.groovy` decides pipeline-level eligibility and propagates it through
`testFilter[cbts_coverage]`: the `ENABLE_CBTS_COVERAGE` kill switch plus the post-merge gate, so
coverage runs on the official post-merge pipeline only.

`isCbtsStage()` in `jenkins/L0_Test.groovy` gates each stage on that flag plus:

- not a perf stage, and not a TensorRT / CPP / AutoDeploy stage
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

Every product process installs a meta-path finder that wraps the resolved loader so a hook runs the
moment `exec_module` returns, watching for `mpi4py.futures` finishing its import to install the pool
patch before any executor exists (needed regardless of role, since it's usually the *coordinator*,
not a worker, that constructs the pool).

What happens to `PY_START` itself then splits on one check, `process_roles.is_mpi_pool_worker()`
(recognises an `mpi4py.futures` pool worker by its `-m mpi4py.futures.server` spawn signature):

- **An MPI pool worker** additionally keys activation to `tensorrt_llm` finishing its import:
  subscribe to the channel, then enable `PY_START` — in that order, so recording never starts on a
  stale context — with a bounded timer (`CBTS_WORKER_ACTIVATE_MAX_SECONDS`) as backstop for a worker
  that never imports the framework at all. Waiting for that import is deliberate: instrumenting the
  cold-start import would overrun the `wait_shutdown` worker identity barrier
  (`tensorrt_llm/llmapi/mpi_session.py`), a budget shared with the worker's own process spawn. The
  cost is that module and class bodies executed during the deferred import go unrecorded; see
  `COLLECTION.md` §5.2.
- **Any other product process** (`trtllm-serve`, disagg helpers, ...) has no such barrier to protect,
  so it subscribes and enables `PY_START` immediately — its import phase gets recorded too.

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

Workers learn the address through the pool patch. `CBTS_CONTEXT_SOCKET` exists only once
`pytest_configure` has bound the socket, which is after the MPI runtime captured the environment it
spawns workers with, so the patched `MPIPoolExecutor` forwards it through `mpi4py`'s own `env`
payload — applied during the worker's sync handshake, in time for the framework import. Ordinary
children inherit the variable at exec.

## Granularity

- **Integration tests**: the outer pytest carries `-p cbts_plugin`, so each test-db entry — one
  pytest item — is its own context.
- **Unit tests** (`test_unittests_v2[entry]`): the inner pytest carries no plugin, so the whole batch
  runs under the one inherited `CBTS_TEST_ID`, which equals the test-db entry. This matches CBTS's
  selection granularity.
- `co_qualname` gives `Class.method`, so results roll up function → class → file.
- Comprehension, generator and lambda frames are skipped, and so is any frame whose qualname
  contains `<locals>`: a decorator's `wrapper` or a registered callback leaves no row however often
  it runs. The consumer widens such a change to file level; see `COLLECTION.md` §5.1.

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

A process that fails to join cannot be that precise. Everything it records lands on the context it
was spawned with, so that context earns an `attribution` row; never having heard an announcement, it
cannot name the tests missing its coverage, so that half is stage-scoped. Where such a process really
did serve exactly one test the stage scope over-reports, forcing re-runs that were not needed — the
cheap error, chosen over the silent one.

### Reasons

| Reason | Kind | Recorded by | Scope |
|---|---|---|---|
| `context_not_acknowledged` | `attribution` | producer | the **superseded** test — a subscriber that misses an announcement keeps recording under the context it still believes current, so that is where the next test's work lands |
| `subscriber_stopped_recording` | `incomplete` | producer | **every test from the loss onward** — ejecting closes the socket, so the subscriber saves and stops; one dropped at accept never starts |
| `did_not_finish_before_deadline` | `incomplete` | producer | the **last** test — a subscriber still connected when the drain deadline expired, so its final save may not have happened |
| `unreachable_on_subscribe` | `incomplete` | producer | the current test — a connection was accepted but the current context could not be delivered to it |
| `context_channel_unreachable` | both | the subscriber | `attribution` on its spawn context, `incomplete` stage-scoped |
| `no_context_channel` | both | producer | stage-scoped — `pytest_configure` could not bind, so every process spent the session on its spawn context |

### Storage

Taints live in the leaf database of whichever process recorded them, so they need no separate
artifact and no separate collection path: the ordinary `.cbtscov.*.sqlite` glob picks them up and the
ordinary reducer unions them.

A taint names the process whose rows are in doubt, which is not always the process that wrote the
file — the producer records taints about its subscribers, which is why subscribers send
`I <process_uid>` on connect, the same uid their own leaf database carries. Those uids are inserted
into `process` but deliberately **not** into `process_case`: that table is what `saved_procs` counts,
and a tainted process must not be mistaken for one that saved.

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

Every database uses schema version 4. `case_stage` stores a bare test ID beside its stage ID, and
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

A `(test, stage)` is safe to skip only when its coverage is complete: the test passed, every process
it spawned saved, and the channel vouched for all of them. Force-run anything else:

```python
# rows NOT safe to skip
c.execute("SELECT test, stage FROM test_case_meta WHERE test != '' AND "
          "(outcome IS NULL OR outcome != 'passed' OR saved_procs < expected_workers + 1 "
          "OR tainted != 0)").fetchall()
# every test whose coverage is in doubt, which half is in doubt, and why
c.execute("SELECT process, test, stage, kind, reason FROM taint_rows").fetchall()
```

## Smoke test

```bash
CBTS_DIR=jenkins/scripts/cbts
COV_DIR="${CBTS_DIR}/cbts/coverage/collection"
export TRTLLM_WHEEL_PATH=/usr/local/lib/python3.12/dist-packages
export JOB_WORKSPACE=/tmp/cbts_smoke STAGE_NAME=smoke
"${COV_DIR}/make_coveragerc.sh"
export PYTHONPATH="${CBTS_DIR}/cbts_injectors:${CBTS_DIR}:${PYTHONPATH:-}"
export CBTS_COVERAGE_CONFIG="${JOB_WORKSPACE}/.coveragerc"
export CBTS_STAGE="${STAGE_NAME}" CBTS_PROCESS_ROLE=outer_pytest
cd tests/integration/defs
pytest -p cbts_plugin -vs "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4"
cd "${JOB_WORKSPACE}" && PYTHONPATH="${OLDPWD}/${CBTS_DIR}" python3 -m cbts.command coverage collection pystart-report --glob '.cbtscov.smoke*.sqlite'
```
