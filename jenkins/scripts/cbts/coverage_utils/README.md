# CBTS Layer C — Coverage Utils

CI tooling that captures per-test **function/class-level** coverage (which product functions each
test entered), including subprocesses, on single-GPU L0 stages. CI infrastructure only — nothing
ships in the product wheel, and every file is a no-op unless `CBTS_COVERAGE_CONFIG` is set in the
environment.

Capture uses `sys.monitoring` `PY_START` (Python 3.12+): each function a test enters fires once,
then that code object is disabled until the next test — so overhead scales with functions entered,
not lines executed (far cheaper than line tracing).

## Files

| File | Role |
|---|---|
| `cbts_pystart.py` | The tracker: a `sys.monitoring` (tool id 4) `PY_START` tool that records, per test context, the set of product `(file, qualname)` entered. Writes one `.cbtscov.<stage>.<suffix>.sqlite` per process (binary, so the publish-artifacts guardword/secret scanners skip it — a text `.json` would be flagged). |
| `sitecustomize.py` | Starts the tracker in each Python process under `CBTS_COVERAGE_CONFIG` (except dependency build/install tools — `pip`, `setup.py`, `cmake`, … — which opt out themselves and their spawned subtree). Reads `source` + `data_file` from the rcfile. Long-lived non-pytest processes (e.g. `trtllm-serve`) poll a marker file to switch context; `mpi4py.futures` pool workers use the inherited `CBTS_TEST_ID` context plus the atexit save. |
| `cbts_plugin.py` | Pytest plugin (`-p cbts_plugin`): per test, writes the marker file, sets `CBTS_TEST_ID`, and switches the tracker context via `sitecustomize.switch_test_context`; also patches `mpi_session._start_mpi_pool` so workers inherit the coverage env. |
| `pystart_report.py` | Merges all `.cbtscov.*.sqlite` (union per test; legacy `.json`/`.json.gz` also accepted) and emits any of: `--out-sqlite` (indexed `touch(test, file, qualname)` DB — the selector artifact), `--out-dir` (per-file split HTML report: index + one page per file), `--out-json` (full `test -> [file::qualname]` map). With `--source-root` also computes the file/function coverage rate. Prints a one-line touch-count summary. |
| `coveragerc.template` | Template for the runtime rcfile; only `[run] source` + `data_file` are used. |
| `make_coveragerc.sh` | Substitutes `@...@` placeholders in the template; writes `$JOB_WORKSPACE/.coveragerc`. |

## When it runs

`isCbtsStage()` in `jenkins/L0_Test.groovy` gates each stage on:

- `CBTS_PIPELINE_ELIGIBLE` — set on every pipeline (pre- and post-merge)
- not a perf stage, and not a TensorRT / CPP / AutoDeploy stage
- single-GPU only — stages named with `-<N>_GPUs` or `-<N>_Nodes` (multi-GPU / multi-node) are disabled in phase 1 and enabled incrementally later
- `ENABLE_CBTS_COVERAGE` (global kill-switch) and `CBTS_EXCLUDE_STAGES` (per-stage skip)

Non-CBTS stages get an empty `.coveragerc` and run uninstrumented.

## Granularity

- **Integration tests**: the outer pytest carries `-p cbts_plugin`, so each test-db entry (one pytest item) is its own context.
- **Unit tests** (`test_unittests_v2[entry]`): the inner pytest carries no plugin, so the whole batch runs under the one inherited `CBTS_TEST_ID` context = the test-db entry. This matches CBTS's selection granularity (entry level).
- `co_qualname` gives `Class.method`, so results roll up to function → class → file. Comprehension / generator / lambda frames are skipped.

## Output

- Per-process `.cbtscov.<stage>.<host>.pid<N>.X<rand>.sqlite` files ride back in the standard `results-<stage>.tar.gz` under `cbts/`. Being binary, they are skipped by the artifact guardword/secret scanners (which flag text files carrying product paths).
- `L0_MergeRequest.groovy`'s Test Coverage stage merges all stages' files via `pystart_report.py` and uploads to `${UPLOAD_PATH}/cbts-coverage/`:
  - `cbts_touchmap.sqlite` — indexed touch DB (selector artifact), plus a `meta` table with the coverage rate.
  - `cbts_pystart_report.tar.gz` — the split HTML report (open `cbts_report/index.html` after extracting).

## Query the touch DB

Which tests to run for a change, from `cbts_touchmap.sqlite`:

```python
import sqlite3
c = sqlite3.connect("cbts_touchmap.sqlite")
# file-level (phase 1): any test that entered a function in the changed file
c.execute("SELECT DISTINCT test FROM touch WHERE file = ?",
          ("tensorrt_llm/_torch/pyexecutor/py_executor.py",)).fetchall()
# function-level (phase 2): tests that entered a specific function/method
c.execute("SELECT DISTINCT test FROM touch WHERE file = ? AND qualname = ?",
          ("tensorrt_llm/_torch/pyexecutor/py_executor.py", "PyExecutor.forward")).fetchall()
# coverage rate
dict(c.execute("SELECT key, value FROM meta"))
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
