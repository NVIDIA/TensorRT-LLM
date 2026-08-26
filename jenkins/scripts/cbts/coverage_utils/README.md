# CBTS Layer C — Coverage Utils

CI tooling that captures per-test **function/class-level** coverage (which product functions each
test entered), including subprocesses, on single-GPU L0 stages. CI infrastructure only — nothing
ships in the product wheel, and every file is a no-op unless `CBTS_COVERAGE_CONFIG` is set in the
environment.

Capture uses `sys.monitoring` `PY_START` (Python 3.12+): each function a test enters fires once,
then that code object is disabled until the next test — so overhead scales with functions entered,
not lines executed (far cheaper than line tracing).

`COLLECTION.md` summarises what the data model is and, importantly, what it does not record;
`../coverage_selection/SELECTION.md` covers how the consumer side uses it.

## Files

| File                  | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cbts_pystart.py`     | The tracker: a `sys.monitoring` (tool id 4) `PY_START` tool that records, per test context, the set of product `(file, qualname)` entered. Writes one compact `.cbtscov.<stage>.<suffix>.sqlite` per process; these leave the node only inside compressed tarballs, so the publish-artifacts guardword/secret scanner (which byte-matches product paths in raw files but does not recurse into archives) never sees the paths stored inside.                                                                                                                                                                                                                                                                                                              |
| `sitecustomize.py`    | Starts the tracker in each Python process under `CBTS_COVERAGE_CONFIG` (except dependency build/install tools — `pip`, `setup.py`, `cmake`, … — and Ray infrastructure, which opt out themselves and their spawned subtree). Reads `source` + `data_file` from the rcfile and the process role from `CBTS_PROCESS_ROLE`. Every instrumented process saves periodically so its coverage survives a non-clean exit (`mpi4py.futures` pool workers included — their pool is torn down at test end); product processes additionally subscribe to the context channel. Product processes enable capture only after the framework's first import finishes (bounded by `CBTS_WORKER_ACTIVATE_MAX_SECONDS`), so a `wait_shutdown` identity barrier stays within its timeout. |
| `cbts_plugin.py`      | Pytest plugin (`-p cbts_plugin`): owns the context channel, and per test sets `CBTS_TEST_ID`, switches its own tracker context, announces to every subscriber, and records the test outcome via `sitecustomize`. Only a pytest process loads it, so it imports pytest unconditionally and its hooks are always bound. |
| `cbts_channel.py`     | The context channel: an `AF_UNIX` broadcast from the outer pytest to its subprocesses. Subscribers receive the context current at join, then every announcement, then a final `STOP`. Acknowledgements let the producer wait until every consumer has switched before a test body runs, and the `STOP` drain replaces a stop-file as the signal that all coverage is on disk. |
| `cbts_pool.py`        | Marker path plus the `MPIPoolExecutor.__init__` patch that counts the subprocess workers each test spawns. Separate from the plugin so `sitecustomize` can reach it without importing pytest during interpreter startup. |
| `compact_db.py`       | The compact schema, leaf writer, and merge-closed reducer. Strings live once in dimension tables; `touch` stores `(case_stage_id, symbol_id)`. Process identity makes hierarchical x86/SBSA/final merges idempotent and preserves completeness counts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `pystart_report.py`   | Unions compact `.cbtscov.*.sqlite` inputs into another compact DB and emits any of: `--out-sqlite` (the indexed selector artifact), `--out-dir` (per-file split HTML report), and `--out-json` (full `test -> [file::qualname]` map). With `--source-root` it also computes the file/function coverage rate. The same reducer can merge leaf, platform-level, or final databases.                                                                                                                                                                                                                                                                                                                                                                         |
| `coveragerc.template` | Template for the runtime rcfile; only `[run] source` + `data_file` are used.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `make_coveragerc.sh`  | Substitutes `@...@` placeholders in the template; writes `$JOB_WORKSPACE/.coveragerc`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

## When it runs

`jenkins/L0_MergeRequest.groovy` decides pipeline-level eligibility and propagates it to the runner via `testFilter[cbts_coverage]`:

- `ENABLE_CBTS_COVERAGE` (global kill-switch) AND the post-merge gate — coverage runs only on the official post-merge pipeline

`isCbtsStage()` in `jenkins/L0_Test.groovy` then gates each stage on that propagated flag plus:

- not a perf stage, and not a TensorRT / CPP / AutoDeploy stage
- single-GPU only — stages named with `-<N>_GPUs` or `-<N>_Nodes` (multi-GPU / multi-node) are disabled in phase 1 and enabled incrementally later
- `CBTS_EXCLUDE_STAGES` (per-stage skip)

Non-CBTS stages get an empty `.coveragerc` and run uninstrumented.

## Process roles

Each instrumented process is told what it is through `CBTS_PROCESS_ROLE`, rather than inferring it
from its own or its parent's command line:

| Value | Set by | Effect |
|---|---|---|
| `outer_pytest` | `jenkins/L0_Test.groovy`, next to the other `CBTS_*` vars | tracker starts immediately; the plugin owns context switching and the pool patch |
| `inner_pytest` | `tests/integration/defs/test_unittests.py` | tracker starts immediately; holds the inherited `CBTS_TEST_ID` for the whole batch and installs the pool patch itself |
| *(unset)* | — | product process: subscribes to the context channel; activation and the pool patch both hang off an import hook (see below) |

Three things a product process needs are keyed to imports rather than polled: `sitecustomize.py`
puts a meta-path finder in front of `sys.meta_path` that wraps the resolved loader, so the hook runs
the moment `exec_module` returns. `mpi4py.futures` finishing its import installs the
`MPIPoolExecutor.__init__` patch, before any executor can be constructed, so no pool goes uncounted.
`tensorrt_llm` finishing its import subscribes to the context channel and then enables `PY_START` —
in that order, so recording never starts on a stale context. A bounded timer
(`CBTS_WORKER_ACTIVATE_MAX_SECONDS`) still activates a process that never imports the framework.

The subscribe waits for that import because of how a pool worker learns the channel address:
`CBTS_CONTEXT_SOCKET` only exists once `pytest_configure` has bound the socket, which is after the
MPI runtime captured the environment it spawns workers with. The patched `MPIPoolExecutor` forwards
it through `mpi4py`'s own `env` payload instead, which the worker applies during its sync handshake —
too late for interpreter startup, in time for the framework import. Ordinary children inherit the
variable at exec and could subscribe sooner, but take the same path.

`sitecustomize.py` **pops** the variable, so a role never reaches a child: the pool workers and
servers an outer pytest spawns come up unlabelled, which is what they are. `CBTS_STAGE` names the
stage the coverage is attributed to and is set alongside it.

`CBTS_CONTEXT_SOCKET` is not set by the pipeline: the plugin publishes it after binding.

## Granularity

- **Integration tests**: the outer pytest carries `-p cbts_plugin`, so each test-db entry (one pytest item) is its own context.
- **Unit tests** (`test_unittests_v2[entry]`): the inner pytest carries no plugin, so the whole batch runs under the one inherited `CBTS_TEST_ID` context = the test-db entry. This matches CBTS's selection granularity (entry level).
- `co_qualname` gives `Class.method`, so results roll up to function → class → file. Comprehension / generator / lambda frames are skipped.
- **Closures are skipped too**: any frame whose qualname contains `<locals>` is dropped, so a decorator's `wrapper` or a registered callback leaves no row however often it runs. The consumer side compensates by widening such a change to file level; see `COLLECTION.md` §5.1.
- **A product process's import phase is not captured**: in anything without a pytest role (pool workers, `trtllm-serve`), activation is deferred until `tensorrt_llm` has imported, so module and class bodies get no rows there. See `COLLECTION.md` §5.2.

## Output

- Every instrumented process saves every `CBTS_PERIODIC_SAVE_SECONDS` (default 5s). At session end the plugin announces `STOP` and waits for every subscriber to take its final save and unsubscribe, so the stage output directory is quiet before the results tar reads it. A consumer whose producer died sees EOF and does the same.
- Per-process `.cbtscov.<stage>.<host>.X<rand>.pid<N>.sqlite` files ride back in the standard `results-<stage>.tar.gz` under `cbts/`. Riding inside a compressed tarball keeps their plaintext product paths away from the artifact guardword/secret scanner, which byte-matches raw files but does not recurse into archives.
- `L0_MergeRequest.groovy`'s Test Coverage stage merges all stages' files via `pystart_report.py` and uploads one tarball to `${UPLOAD_PATH}/cbts-coverage/`:
  - `cbts_pystart_report.tar.gz` — contains `cbts_touchmap.sqlite` (indexed touch DB / selector artifact, with a `meta` table holding the coverage rate) and `cbts_report/` (the split HTML report; open `cbts_report/index.html` after extracting). Bundled compressed so the touch DB's plaintext paths never reach the guardword scanner.

Every database uses schema version 3. `case_stage` stores a bare test ID beside its stage ID;
logical views reconstruct the selector's `<stage>/<test>` key. Integer IDs are local to a database,
so the reducer resolves stage names, test IDs, file paths, qualnames, and process UIDs when merging
instead of copying IDs. `touch_rows` and `test_case_meta` expose the reconstructed logical rows for
diagnostics.

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

A `(test, stage)` is only safe to skip when its coverage is complete — the test passed and
every process it spawned saved. Force-run anything else:

```python
# rows NOT safe to skip: non-passed, or a spawned worker/server never saved its coverage
c.execute("SELECT test, stage FROM test_case_meta WHERE test != '' AND "
          "(outcome IS NULL OR outcome != 'passed' OR saved_procs < expected_workers + 1)").fetchall()
```

## Smoke test

```bash
COV_DIR=jenkins/scripts/cbts/coverage_utils
export TRTLLM_WHEEL_PATH=/usr/local/lib/python3.12/dist-packages
export JOB_WORKSPACE=/tmp/cbts_smoke STAGE_NAME=smoke
"${COV_DIR}/make_coveragerc.sh"
export PYTHONPATH="${COV_DIR}:${PYTHONPATH:-}"
export CBTS_COVERAGE_CONFIG="${JOB_WORKSPACE}/.coveragerc"
export CBTS_STAGE="${STAGE_NAME}" CBTS_PROCESS_ROLE=outer_pytest
cd tests/integration/defs
pytest -p cbts_plugin -vs "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4"
cd "${JOB_WORKSPACE}" && python3 "${OLDPWD}/${COV_DIR}/pystart_report.py" --glob '.cbtscov.smoke*.sqlite'
```
