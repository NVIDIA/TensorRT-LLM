# CBTS Coverage Collection — Current State

How per-test coverage data is produced on this branch, and **what it does not record**.
Consumer side: `../coverage_selection/SELECTION.md`. File-by-file roles: `README.md`.

---

## 1. What is recorded

One set per test: **which product functions it entered**.

```
(test, file, qualname)
 │      │      └─ the code object's co_qualname, e.g. "PyExecutor._executor_loop"
 │      └─ product file, canon'd to the tensorrt_llm/... form
 └─ "<STAGE>/<pytest nodeid>"
```

No line numbers, no call counts, no call graph.

## 2. How

`sys.monitoring`'s `PY_START` event (Python 3.12+, tool id 4):

```python
def _on_py_start(self, code, offset):
    if self._in_source(code.co_filename):
        qual = code.co_qualname
        if "<locals>" not in qual and qual not in _SKIP_QUALNAMES:
            self._data[self._ctx].add((fn, qual))
    return _MON.DISABLE
```

- Each code object fires once on **first entry**, then `DISABLE`s itself: zero cost afterwards.
- Each test re-arms with `restart_events()`, so every test gets its own complete function set.
- Cost scales with how many **distinct** functions were entered, not with how often.

## 3. Crossing processes

Three environment variables (`CBTS_COVERAGE_CONFIG` / `PYTHONPATH` / `CBTS_CONTEXT_SOCKET`) are
inherited by child processes by default, and `sitecustomize.py` installs the tracker in every
Python process at startup. No product code is modified.

A process learns what it is from `CBTS_PROCESS_ROLE`, set by whoever launches it and consumed
by `sitecustomize.py` so it is never inherited:

| Role | Set by | Context source | Activation |
|---|---|---|---|
| `outer_pytest` | `jenkins/L0_Test.groovy` | `cbts_plugin`'s `pytest_runtest_protocol` switches it directly | at interpreter startup |
| `inner_pytest` | `tests/integration/defs/test_unittests.py` | inherited `CBTS_TEST_ID`, held for the whole batch | at interpreter startup |
| *(unset)* — pool workers, `trtllm-serve`, helper subprocesses | — | inherited `CBTS_TEST_ID` until it subscribes to the context channel, then every announcement | **deferred** until `tensorrt_llm` finishes importing, via an import hook |

Processes that opt themselves out: `pip` / `setup.py` / `cmake` / `ninja` and everything they
spawn, plus Ray infrastructure processes (`default_worker.py` and friends). These are spawned by
pip and raylet rather than by our own code, so they are recognised from their launch target
(`-m` module or script name) instead of a role.

## 4. Persistence and merge

Every process writes its own compact SQLite (`.cbtscov.<stage>.<host>.X<rand>.pid<N>.sqlite`), on a
5s periodic snapshot plus `atexit`. Leaf, platform-level, and final databases share schema version
3, so the same reducer can merge every level:

| Table | Contents |
|---|---|
| `stage`, `case_stage` | interned stage names and bare test contexts |
| `file`, `symbol` | interned product paths and qualnames |
| `touch(case_stage_id, symbol_id)` | which test context entered each symbol |
| `process`, `process_case` | stable process identity and the test contexts each process saved |
| `test_result` | coordinator outcome and expected-worker observations |

The merge (`pystart_report.py`) resolves each input's local IDs by its natural keys and unions the
relations. `saved_procs` is derived from distinct process identities, so repeated intermediate
inputs do not inflate completeness. The final artifact is `cbts_touchmap.sqlite`; the
`touch_rows` and `test_case_meta` views reconstruct its logical query rows.

---

## 5. What is not recorded (important)

This section is what forces the concessions on the consumer side.

### 5.1 Closures — not recorded at all, **still unresolved**

```python
if "<locals>" not in qual and ...
```

Any frame whose qualname contains `<locals>` is dropped outright. A decorator's `wrapper`, a
registered callback, the inner function of a cached factory — none leave a trace however often
they run.

Measured scale: `tensorrt_llm` is 646410 lines, of which **14191 (2.2%)** are closure bodies;
7513 of those sit under an enclosing function whose row set trails the whole file's by more than
300 tests.

The consumer side widens a closure change to file level when that is wider, and otherwise runs in
full (`../coverage_selection/SELECTION.md` §4.1); **the collection side itself is unfixed.** A real
fix means having the producer record closure frames (the `<locals>` segments in `co_qualname` can
be kept or folded), which is follow-up work.

### 5.2 Import phase is lost inside product processes

Their `tensorrt_llm` import happens **before deferred activation**, so module bodies
(`<module>`) and class bodies (`ClassName`) get no rows at all in a pool worker or a server.

The outer pytest's import happens before any test, under the empty context, and the merge filters
it out with `WHERE test != ''`.

Net effect: `<module>` and class bodies are only recorded by the tests that spawn subprocesses.
Measured, `llmapi/llm_args.py::<module>` has **509** holders while all **746** known tests import
it from their process — the missing 226 are exactly the accuracy / disagg tests served by MPI pool
workers.

Deferred activation is deliberate: without it, the instrumented cold-start import overruns the
`wait_shutdown` worker identity barrier.

This blind spot is what costs the most on the consumer side: a change landing on a module body, a
class body or a signature / decorator line cannot be bounded from these rows, so the tier declines
and the PR runs in full (`../coverage_selection/SELECTION.md` §4). Recording the workers' import
phase is what would let those changes narrow again.

### 5.3 Other blind spots

| Blind spot | Reason |
|---|---|
| C++ / nanobind implementations | PY_START only sees Python frames; the C++ side of KV cache manager, scheduler and decoder is invisible |
| Comprehensions / genexprs / lambdas | skipped explicitly by `_SKIP_QUALNAMES` |
| Test code itself | `tests/` is outside the source root |
| A Ray stage's GPU worker | `RayGPUWorker` lives in the opted-out `default_worker.py` |
| Multi-GPU / multi-node stages | phase 1 collects on single-GPU stages only |
| The last ≤5s before a worker is SIGKILLed | the periodic snapshot interval |

---

## 6. The completeness signal the data carries

`test_case_meta` lets the consumer decide whether a record can be trusted:

```sql
outcome IS NULL OR outcome != 'passed' OR saved_procs < expected_workers + 1
```

`expected_workers` is counted in the coordinator by the patched `MPIPoolExecutor.__init__`,
`saved_procs` is counted by the merge, and the `+1` is the coordinator itself. A mismatch means
some process's coverage was lost.

This is the only way to detect lost data — the loss is invisible in the data itself (footprint
stays large, `py_executor` rows are still there).

---

## 7. When collection runs

`L0_MergeRequest.groovy` decides pipeline-level eligibility and `isCbtsStage()` decides each
stage:

- official post-merge pipeline only (`ENABLE_CBTS_COVERAGE && JOB_NAME ==~ /.*PostMerge.*/`)
- not a perf stage, not a TensorRT / CPP / AutoDeploy stage
- single-GPU stages only (name carries no `-<N>_GPUs` / `-<N>_Nodes`)
- not listed in `CBTS_EXCLUDE_STAGES`

The per-process files ride back inside `results-<stage>.tar.gz`; the `Test Coverage` stage merges
them all and uploads to `${UPLOAD_PATH}/cbts-coverage/cbts_pystart_report.tar.gz`.
