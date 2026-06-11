# CBTS Layer C — Coverage Utils

CI tooling that captures per-test code coverage (including subprocesses)
on post-merge L0 stages. CI infrastructure only — nothing ships in the
product wheel, and every file is a no-op unless `CBTS_COVERAGE_CONFIG` is
set in the environment.

## Files

| File | Role |
|---|---|
| `sitecustomize.py` | Starts coverage in each Python process under `CBTS_COVERAGE_CONFIG`; workers poll a marker file to switch the test-context. |
| `cbts_plugin.py` | Pytest plugin (`-p cbts_plugin`): per-test `cov.switch_context()`, and patches `mpi_session._start_mpi_pool` so workers inherit the coverage env. |
| `coveragerc.template` | Template for the runtime `.coveragerc`. |
| `make_coveragerc.sh` | Substitutes `@...@` placeholders in the template; writes `$JOB_WORKSPACE/.coveragerc`. |
| `coverage_summary.py` | Prints `covered/ran test cases` to the stage log after pytest (stdlib-only, read-only). |

## When it runs

`isCbtsStage()` in `jenkins/L0_Test.groovy` gates each stage on:

- `CBTS_PIPELINE_ELIGIBLE` — `testFilter[IS_POST_MERGE]` (official PostMerge or `/bot run --post-merge`)
- not a perf stage
- `ENABLE_CBTS_COVERAGE` (global kill-switch) and `CBTS_EXCLUDE_STAGES` (per-stage skip)

Non-CBTS stages get an empty `.coveragerc` and run uninstrumented.

## Output

- Per-process `.coverage.<stage>.<host>.pid<N>.X<rand>` files ride back in the standard `results-<stage>.tar.gz` under `cbts/`.
- `L0_MergeRequest.groovy`'s Test Coverage stage combines all stages' files and uploads the merged DB to `${UPLOAD_PATH}/cbts-coverage/coverage.sqlite`.
- Each stage logs `CBTS coverage [<stage>]: <covered>/<ran> test cases ...`.

## Query the merged DB

Which tests touched a file:

```python
import sqlite3
c = sqlite3.connect("coverage.sqlite")
for (test_id,) in c.execute("""
    SELECT DISTINCT ctx.context
    FROM line_bits lb
    JOIN file f      ON lb.file_id    = f.id
    JOIN context ctx ON lb.context_id = ctx.id
    WHERE f.path LIKE ? AND ctx.context != ''
""", ("%/tensorrt_llm/_torch/pyexecutor/py_executor.py",)):
    print(test_id)
```

Render HTML locally (remap the stored paths to your tree):

```bash
cat > .coveragerc <<EOF
[paths]
source =
    /path/to/your/TensorRT-LLM/tensorrt_llm/
    */tensorrt_llm/
[html]
show_contexts = True
EOF
coverage html -i --show-contexts --data-file=coverage.sqlite -d report
```

## Smoke test

```bash
COV_DIR=jenkins/scripts/cbts/coverage_utils
export TRTLLM_WHEEL_PATH=/usr/local/lib/python3.12/dist-packages
export TRTLLM_SRC_PATH=/path/to/TensorRT-LLM
export JOB_WORKSPACE=/tmp/cbts_smoke STAGE_NAME=smoke
"${COV_DIR}/make_coveragerc.sh"
export PYTHONPATH="${COV_DIR}:${PYTHONPATH:-}"
export CBTS_COVERAGE_CONFIG="${JOB_WORKSPACE}/.coveragerc"
cd tests/integration/defs
pytest -p cbts_plugin -vs "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4"
cd "${JOB_WORKSPACE}" && coverage combine --rcfile="${JOB_WORKSPACE}/.coveragerc"
```
