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

| File | Role |
|---|---|
| `cbts_pystart.py` | The tracker: a `sys.monitoring` (tool id 4) `PY_START` tool that records, per test context, the set of product `(file, qualname)` entered. Writes one compact `.cbtscov.<stage>.<suffix>.sqlite` per process; these leave the node only inside compressed tarballs, so the publish-artifacts guardword/secret scanner (which byte-matches product paths in raw files but does not recurse into archives) never sees the paths stored inside. |
| `sitecustomize.py` | Starts the tracker in each Python process under `CBTS_COVERAGE_CONFIG` (except dependency build/install tools — `pip`, `setup.py`, `cmake`, … — which opt out themselves and their spawned subtree). Reads `source` + `data_file` from the rcfile. Every instrumented process saves periodically so its coverage survives a non-clean exit (`mpi4py.futures` pool workers included — their pool is torn down at test end); long-lived non-pytest processes (e.g. `trtllm-serve`) additionally poll a marker file to switch context. `mpi4py.futures` pool workers enable capture only after the product framework's first import settles (bounded by `CBTS_WORKER_ACTIVATE_MAX_SECONDS`), so a `wait_shutdown` identity barrier stays within its timeout. |
| `cbts_plugin.py` | Pytest plugin (`-p cbts_plugin`): per test, writes the marker file, sets `CBTS_TEST_ID`, switches the tracker context, and records the test outcome via `sitecustomize`; patches `MPIPoolExecutor.__init__` to count the subprocess workers each test spawns and to widen their env so they inherit the coverage bootstrap (the caller's env, incl. `env_overrides`, is preserved). |
| `compact_db.py` | The compact schema, leaf writer, and merge-closed reducer. Strings live once in dimension tables; `touch` stores `(case_stage_id, symbol_id)`. Process identity makes hierarchical x86/SBSA/final merges idempotent and preserves completeness counts. |
| `pystart_report.py` | Unions compact `.cbtscov.*.sqlite` inputs into another compact DB and emits any of: `--out-sqlite` (the indexed selector artifact), `--out-dir` (per-file split HTML report), and `--out-json` (full `test -> [file::qualname]` map). With `--source-root` it also computes the file/function coverage rate. The same reducer can merge leaf, platform-level, or final databases. |
| `coveragerc.template` | Template for the runtime rcfile; only `[run] source` + `data_file` are used. |
| `make_coveragerc.sh` | Substitutes `@...@` placeholders in the template; writes `$JOB_WORKSPACE/.coveragerc`. |

## When it runs

`jenkins/L0_MergeRequest.groovy` decides pipeline-level eligibility and propagates it to the runner via `testFilter[cbts_coverage]`:

- `ENABLE_CBTS_COVERAGE` (global kill-switch) AND the post-merge gate — coverage runs only on the official post-merge pipeline

`isCbtsStage()` in `jenkins/L0_Test.groovy` then gates each stage on that propagated flag plus:

- not a perf stage, and not a TensorRT / CPP / AutoDeploy stage
- single-GPU only — stages named with `-<N>_GPUs` or `-<N>_Nodes` (multi-GPU / multi-node) are disabled in phase 1 and enabled incrementally later
- `CBTS_EXCLUDE_STAGES` (per-stage skip)

Non-CBTS stages get an empty `.coveragerc` and run uninstrumented.

## Granularity

- **Integration tests**: the outer pytest carries `-p cbts_plugin`, so each test-db entry (one pytest item) is its own context.
- **Unit tests** (`test_unittests_v2[entry]`): the inner pytest carries no plugin, so the whole batch runs under the one inherited `CBTS_TEST_ID` context = the test-db entry. This matches CBTS's selection granularity (entry level).
- `co_qualname` gives `Class.method`, so results roll up to function → class → file. Comprehension / generator / lambda frames are skipped.
- **Closures are skipped too**: any frame whose qualname contains `<locals>` is dropped, so a decorator's `wrapper` or a registered callback leaves no row however often it runs. The consumer side compensates by widening such a change to file level; see `COLLECTION.md` §5.1.
- **A pool worker's import phase is not captured**: activation is deferred until `tensorrt_llm` has imported, so module and class bodies get no rows there. See `COLLECTION.md` §5.2.

## Output

- Every instrumented process saves every `CBTS_PERIODIC_SAVE_SECONDS` (default 5s). Before result collection the pipeline creates `CBTS_STOP_FILE` (`<stage output dir>/cbts_stop`), which suppresses later saves, then waits for the directory mtime to settle so an in-flight save drains.
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
export CBTS_MARKER_FILE="${JOB_WORKSPACE}/cbts_current_test.txt"
cd tests/integration/defs
pytest -p cbts_plugin -vs "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4"
cd "${JOB_WORKSPACE}" && python3 "${OLDPWD}/${COV_DIR}/pystart_report.py" --glob '.cbtscov.smoke*.sqlite'
```
