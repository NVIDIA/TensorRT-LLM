# CBTS Touch DB — Interface Contract (coverage-based test selection)

Contract between the **coverage producer** (`cbts-coverage-utils` branch:
`jenkins/scripts/cbts/coverage_utils/`) and the **coverage-based selector** developed here.
The selector depends only on this contract, not on the producer's code, so the two branches
can land independently.

Source of truth for every claim below (producer, `cbts-coverage-utils` branch):
- `coverage_utils/pystart_report.py` — builds the merged touch DB.
- `coverage_utils/cbts_pystart.py` — per-process `sys.monitoring` PY_START tracker.

---

## 1. The artifact the selector consumes

- **File:** `cbts_touchmap.sqlite` — the **merged, deduped, indexed** touch DB
  (`pystart_report.py --out-sqlite`, described there as "indexed touch(test,file,qualname) DB for the selector").
- **Packaging / retrieval:** uploaded per post-merge run as
  `…/<JOB_NAME>/<BUILD_NUMBER>/cbts-coverage/cbts_pystart_report.tar.gz`, which contains
  `cbts_touchmap.sqlite` + `cbts_report/`. Extract the `.sqlite`; open **read-only**.
- **Do NOT consume** the per-process `.cbtscov.<stage>.<host>.pid<pid>.X<token>.sqlite` files:
  they carry **raw absolute paths** and are **not deduped**. Only the merged DB is canonicalized
  and indexed. (Per-process schema is `touch(test, file, qualname)` with no constraints.)

---

## 2. Schema (merged `cbts_touchmap.sqlite`)

```sql
CREATE TABLE touch (
    test     TEXT,   -- pytest nodeid that entered the function ('' == import-time / no test context)
    file     TEXT,   -- product-relative path, canonicalized to 'tensorrt_llm/...'
    qualname TEXT,   -- co_qualname of the entered function/method (see §4)
    UNIQUE(test, file, qualname)   -- rows are deduped; no frequency/count is available
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE INDEX ix_file ON touch(file);            -- file -> tests
CREATE INDEX ix_func ON touch(file, qualname);  -- (file, qualname) -> tests
CREATE INDEX ix_test ON touch(test);            -- test -> touched
```

A row `(test, file, qualname)` means: **during an instrumented run, test `test` called the
function `qualname` defined in `file`.** PY_START fires on function *entry* (call), not on
line execution or import.

---

## 3. Path normalization (`file`) — the selector MUST replicate this

`file` in the merged DB is the producer's `canon()` of the absolute `co_filename`:

```python
import re
def canon(path):
    m = re.search(r"(tensorrt_llm/.*)$", path)
    return m.group(1) if m else path
```

So values look like `tensorrt_llm/_torch/pyexecutor/py_executor.py`.

**Join rule:** before querying, canonicalize the changed-file paths the *same* way.
Git-relative paths that already start with `tensorrt_llm/` are already canonical.
A changed path with no `tensorrt_llm/` segment (C++, configs, tests, tools) can **never**
match a `touch.file` — see the fail-safe rule in §7.

---

## 4. `qualname` semantics

`qualname == code.co_qualname`, recorded only when it passes the producer filter
(`cbts_pystart.py`):

- **Excluded:** any name containing `<locals>` (nested/closure functions), and
  `{<genexpr>, <listcomp>, <setcomp>, <lambda>}` (dict/list/set comps + genexprs + lambdas).
- **Included forms:** `foo` (module-level), `Bar.baz` (method), `Outer.Inner.m`
  (method of a nested class), and `<module>` (module-body execution).
  Control-flow blocks (`if`/`try`/`with`) do **not** appear in the name; class scopes prepend
  `ClassName.`; crossing a function scope injects `.<locals>.` (hence excluded).

Granularity guidance:
- **File-level selection** (`WHERE file = ?`) is the robust default.
- **Function-level selection** (`WHERE file = ? AND qualname = ?`) is *best-effort*: closures,
  comprehensions and lambdas are invisible, so a change confined to those maps only at file level.

---

## 5. `meta` table (advisory stats, not required for selection)

Keys currently written by `pystart_report.py`:

| key | meaning |
|-----|---------|
| `tests` | distinct `test` where `test != ''` |
| `files` | distinct `file` where `test != ''` |
| `functions` | distinct `(file, qualname)` where `test != ''` |
| `file_rate_pct`, `func_rate_pct` | coverage rate vs `--source-root` denominator (only if that arg was passed) |
| `total_files`, `total_functions` | denominator sizes (only if `--source-root` was passed) |

All values are **strings**. Treat every key as **optional** (read with a default) — the rate
keys are absent when the report is generated without `--source-root`.
There is **no `schema_version` key yet**; see §8.

---

## 6. Consumer query patterns

```sql
-- Reverse lookup — the core of selection. Always filter test != ''.
SELECT DISTINCT test FROM touch WHERE file = :file AND test != '';                    -- file  -> tests
SELECT DISTINCT test FROM touch WHERE file = :file AND qualname = :q AND test != '';  -- func  -> tests

-- Forward (debug / explain-why):
SELECT file, qualname FROM touch WHERE test = :test;

-- Universe of tests that have coverage data at all:
SELECT DISTINCT test FROM touch WHERE test != '';
```

**Always append `test != ''`.** Rows with `test == ''` are import-time / no-context
attributions (module bodies loaded before any test), not per-test signal.

---

## 7. Selection algorithm contract

Input: set of changed `(file[, qualname])` from `git diff` (+ AST for function granularity).
Output: set of pytest nodeids to run.

```
selected = ∅
for each changed product file f (canonicalized to tensorrt_llm/...):
    selected ∪= { test : (test, f, *) in touch, test != '' }     # file-level, safe default
return selected
```

**Fail-safe (correctness > savings — an undercount silently drops tests → escapes):**
- A changed path **not** under `tensorrt_llm/` (C++/CUDA, YAML, tests, tooling, build) has **no**
  Python coverage → **cannot be decided by this DB** → fall back to "run" (defer to the
  rules-based selector / full set). Never treat "no match" as "skip".
- A changed product file with **zero** `touch` rows → treat as **unknown → run**, not "untested → skip"
  (it may only be instrumented in a stage this DB didn't cover — see §9).
- A renamed/moved function will not match its new `qualname`/`file` → treat rename as "run".

The DB tells you which tests to **keep**; it is not authoritative about which to **drop**.

---

## 8. Versioning & stability

- The schema above is the v1 contract. Consumers should be tolerant: **select named columns**
  (`SELECT test, file, qualname …`), never `SELECT *`; read `meta` keys with defaults.
- **Recommended producer addition (not yet present):** a `meta` row
  `('schema_version', '1')` so the selector can hard-fail on an unknown version instead of
  silently mis-selecting. Track this as a producer-side follow-up on `cbts-coverage-utils`.

---

## 9. Coverage scope & guarantees (read before trusting "no test hit this")

**Guarantee:** if test `T` entered function `F` (`file`, `qualname`) during an instrumented run
and the process's periodic/atexit save succeeded, then `(T, canon(file), qualname)` is present.

**Non-guarantees (all imply fail-safe → run):**
- **Instrumentation is gated** (`L0_Test.groovy::isCbtsStage`, Phase 1): only **single-GPU**,
  **non-Perf / non-TensorRT / non-CPP / non-AutoDeploy**, **post-merge** stages are instrumented.
  Any test outside that set has **no** coverage data here.
- **Call-based, not import-based:** functions imported but never called are absent; module-level
  side-effect code is attributed to `test == ''` (import time), not to a specific test.
- **Closures / comprehensions / lambdas** are not recorded (§4).
- **No C++/CUDA coverage** at all — those changes are out of scope for this DB.
- **Dedup ⇒ no counts:** you cannot rank tests by hit frequency from this DB.
- **Staleness:** the DB reflects the code at collection time; drift (renames, new functions)
  is invisible until recollected.

---

## 10. Local fixture for developing against this contract

Until the producer lands in `main`, build a fixture DB with the exact schema above:

```sql
CREATE TABLE touch (test TEXT, file TEXT, qualname TEXT, UNIQUE(test, file, qualname));
CREATE TABLE meta  (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX ix_file ON touch(file);
CREATE INDEX ix_func ON touch(file, qualname);
CREATE INDEX ix_test ON touch(test);
INSERT OR IGNORE INTO touch VALUES
  ('accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4',
   'tensorrt_llm/_torch/pyexecutor/py_executor.py', 'PyExecutor._forward_step'),
  ('accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4',
   'tensorrt_llm/_torch/pyexecutor/py_executor.py', '<module>');
INSERT OR REPLACE INTO meta VALUES ('tests','1'),('files','1'),('functions','1');
```

Wire the selector to open this read-only and exercise the §6 queries + §7 fail-safe paths.
