<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS Coverage Collection

How per-test coverage data is produced, and **what it does not record**.
Consumer side: `../selection/SELECTION.md`. File-by-file roles: `README.md`.

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

`CBTS_COVERAGE_CONFIG` and `PYTHONPATH` are inherited at exec, so `sitecustomize.py` installs the
tracker in every Python process in the tree. No product code is modified.

A process learns what it is from `CBTS_PROCESS_ROLE`, set by whoever launches it and popped by
`sitecustomize.py` so it describes exactly one process:

| Role | Set by | Context source | Activation |
|---|---|---|---|
| `outer_pytest` | `jenkins/L0_Test.groovy` | `cbts_plugin`'s `pytest_runtest_protocol` switches it directly | at interpreter startup |
| `inner_pytest` | `tests/integration/defs/test_unittests.py` | inherited `CBTS_TEST_ID`, held for the whole batch | at interpreter startup |
| *(unset)* — pool workers, `trtllm-serve`, helper subprocesses | — | inherited `CBTS_TEST_ID` until it subscribes to the context channel, then every announcement | **deferred** until `tensorrt_llm` finishes importing, via an import hook |

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
5s periodic snapshot plus `atexit`; the pytest process also flushes at `pytest_sessionfinish`. Leaf,
platform-level and final databases share schema version 4, so one reducer merges every level:

| Table | Contents |
|---|---|
| `stage`, `case_stage` | interned stage names and bare test contexts |
| `file`, `symbol` | interned product paths and qualnames |
| `touch(case_stage_id, symbol_id)` | which test context entered each symbol |
| `process`, `process_case` | stable process identity and the test contexts each process saved |
| `test_result` | coordinator outcome and expected-worker observations |
| `taint(process_id, case_stage_id, kind, reason)` | coverage the context channel could not vouch for |

The merge (`pystart_report.py`) resolves each input's local IDs by its natural keys and unions the
relations. `saved_procs` is derived from distinct process identities, so repeated intermediate inputs
do not inflate completeness. The final artifact is `cbts_touchmap.sqlite`; the `touch_rows`,
`taint_rows` and `test_case_meta` views reconstruct its logical query rows.

---

## 5. What is not recorded (important)

This section is what forces the concessions on the consumer side.

### 5.1 Closures — not recorded at all, **still unresolved**

```python
if "<locals>" not in qual and ...
```

Any frame whose qualname contains `<locals>` is dropped outright. A decorator's `wrapper`, a
registered callback, the inner function of a cached factory — none leave a trace however often they
run.

Measured scale: `tensorrt_llm` is 646410 lines, of which **14191 (2.2%)** are closure bodies; 7513 of
those sit under an enclosing function whose row set trails the whole file's by more than 300 tests.

The consumer side widens a closure change to file level when that is wider, and otherwise runs in
full (`../selection/SELECTION.md` §4.1); **the collection side itself is unfixed.** A fix
means having the producer record closure frames (the `<locals>` segments in `co_qualname` can be kept
or folded), which is follow-up work.

### 5.2 Import phase is lost inside MPI pool workers

Only an actual `mpi4py.futures` pool worker defers activation (`process_roles.is_mpi_pool_worker()`
gates it) — every other product process (`trtllm-serve`, disagg helpers, ...) activates immediately
and records its own import. So a worker's `tensorrt_llm` import happens **before deferred
activation**, and module bodies (`<module>`) and class bodies (`ClassName`) get no rows at all from
it.

The outer pytest's import happens before any test, under the empty context, and the merge filters it
out with `WHERE test != ''`.

Net effect: `<module>` and class bodies are only recorded by the tests that spawn subprocesses.
Measured, `llmapi/llm_args.py::<module>` has **509** holders while all **746** known tests import it
from their process — the missing 226 are exactly the accuracy / disagg tests served by MPI pool
workers.

Deferred activation is deliberate, and scoped to pool workers specifically: the instrumented
cold-start import overruns the `wait_shutdown` worker identity barrier
(`tensorrt_llm/llmapi/mpi_session.py`), whose deadline is shared with process spawn plus this
import — a budget only a pool worker's own spawn-and-barrier sequence is subject to.

This blind spot costs the most on the consumer side: a change landing on a module body, a class body
or a signature / decorator line cannot be bounded from these rows, so the tier declines and the PR
runs in full (`../selection/SELECTION.md` §4). Recording the workers' import phase is what
would let those changes narrow again.

### 5.3 Other blind spots

| Blind spot | Reason |
|---|---|
| C++ / nanobind implementations | PY_START only sees Python frames; the C++ side of KV cache manager, scheduler and decoder is invisible |
| Comprehensions / genexprs / lambdas | skipped explicitly by `_SKIP_QUALNAMES` |
| Test code itself | `tests/` is outside the source root |
| A Ray stage's GPU worker | `RayGPUWorker` lives in the opted-out `default_worker.py` |
| Multi-GPU / multi-node stages | phase 1 collects on single-GPU stages only; the channel design is single-producer, so a multi-node stage would need its address published somewhere shared |
| The last ≤5s before a worker is SIGKILLed | the periodic snapshot interval |
| A channel failure inside an inner-pytest batch | `test_unittests_v2` has no `cbts_plugin` (§3's `inner_pytest` row), so nothing records a taint if a subscriber it spawns falls behind |

---

## 6. The completeness signal the data carries

`test_case_meta` lets the consumer decide whether a record can be trusted:

```sql
outcome IS NULL OR outcome != 'passed' OR saved_procs < expected_workers + 1 OR tainted != 0
```

The three terms cover three ways a record can fall short:

- **`outcome`** — a test that did not pass may not have reached the code it normally covers.
- **`saved_procs` vs `expected_workers + 1`** — `expected_workers` is counted in the coordinator by
  the patched `MPIPoolExecutor.__init__`, `saved_procs` by the merge, and the `+1` is the coordinator
  itself. A shortfall means some process's coverage never reached disk.
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
- not a perf stage, not a TensorRT / CPP / AutoDeploy stage
- single-GPU stages only (name carries no `-<N>_GPUs` / `-<N>_Nodes`)
- not listed in `CBTS_EXCLUDE_STAGES`

The per-process files ride back inside `results-<stage>.tar.gz`; architecture checkpoints first
upload `cbts_pystart_report_x86_64.tar.gz` and `cbts_pystart_report_SBSA.tar.gz`, which the selector
requires and hierarchically merges. The later `Test Coverage` stage merges all files and uploads
`${UPLOAD_PATH}/cbts-coverage/cbts_pystart_report.tar.gz` for the full report.
