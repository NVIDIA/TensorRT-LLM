# CBTS Coverage Selection — Current State

How the touch DB is turned into a decision about which cases to run. Collection side:
`../coverage_utils/COLLECTION.md`.

---

## 1. Where this sits

```
Tier 1  rules      every changed file claimed by a rule → narrow per the rule
   │               some file left unclaimed (residual)
   ▼
Tier 2  coverage   residual is all core Python and all present in the DB → scope=coverage
   │               any condition unmet → decline
   ▼
        full fallback                                                     scope=null
```

Tier 2 only ever looks at the **residual**: the files no Tier 1 rule claimed.

## 2. The qualname concepts

Precision comes down to which qualname a changed line lands on. `qualname_map`'s attribution:

| Changed line sits in | Attributed to | Note |
|---|---|---|
| A function or method body | `Class.method` / `func` | the only precise case |
| A module-level statement | `<module>` | module body, executed once at import |
| A class body (class attribute) | `ClassName` | class body, executed once at import |
| **A signature or decorator line** | **the enclosing scope** | a method's `def` line lands on `ClassName`, not on the method |
| A closure body (`<locals>`) | **the nearest recorded enclosing scope** | changing `inner` in `def outer(): def inner()` lands on `outer` |

The last three are where both the precision and the correctness problems come from: **the
attributed qualname's DB rows do not represent the changed code's executions.**

DB qualnames come from `co_qualname`, so they match what the AST derives (`Class.method`), but
comprehensions, lambdas and closures are skipped during collection, so those names do not exist
in the DB at all.

## 3. The decline gates

`CoverageSelector.decide()` returns `ok=False` — Tier 2 stands down and the run is full — on any
of:

| Gate | Condition |
|---|---|
| Non-core file | `path` is not a `.py` under `tensorrt_llm/` (checked on the **repo path**, not `canon()`) |
| File absent from the DB | `file_has_touch_rows(cf)` is false — new or uninstrumented, so "who touches it" is unknown rather than empty |
| **Import-executed change** | a changed line lands on a module body, a class body, or a signature / decorator line |
| No usable patch | the forge API omitted the diff (binary / rename / oversized), so the changed scope is unknown |
| Unparsable source | the AST walk failed, so lines cannot be mapped to qualnames |
| Closure change with no wider row set | see the next section |

## 4. Why import-executed changes fail closed

Import-phase rows are recorded only by the tests that spawn subprocesses (`../coverage_utils/COLLECTION.md`
§5.2), so a `<module>` or `ClassName` row set is missing the tests served by MPI pool workers. Widening
to the file's row set recovers some of them but is a **recovery heuristic, not an upper bound**: a test
that imports the file and never enters any function in it records nothing there at all, so it is absent
from the file set too. Measured on `llmapi/llm_args.py`, the widening goes from 509 holders to 735 while
about 746 tests import the file — the gap does not close.

Since the missing tests cannot be enumerated from the data, these changes decline outright until the
producer records the pool workers' import phase. Ordinary function-body changes are unaffected and keep
precise narrowing.

### 4.1 Closures

A closure body is not recorded at all, so a changed closure is attributed to its nearest recorded
enclosing scope and put through `_underrecorded_bound(cf, qualname)`:

```
file rows > that qualname's rows  →  use the file rows (the wider bound)
otherwise                        →  decline, run in full
```

`tests_touching_func(f, q)` is by construction a subset of `tests_touching_file(f)`, so a strictly
larger file set contains tests the qualname set does not — tests that recorded some *other* function
in the same file, which is what makes the file exercised beyond import time. When the two are equal
the file is only ever recorded at import time (`__init__.py` and similar, 881 of 1149 files) and no
bound exists.

This bound carries the same incompleteness as the one removed above. It is narrower in blast radius
(4.3% of core-Python commits touch a closure, against 82.6% for import-executed lines) but it is not
sound either: a decorator's `wrapper` is created at import time, so the enclosing scope it attributes
to can itself be import-only.

## 5. Full impact resolution

```
for each residual file:
  ├ no usable diff / unparsable → decline
  ├ diff has no lines           → contributes nothing (comment / blank only)
  └ for each changed qualname:
       ├ import-executed → decline
       ├ closure         → _underrecorded_bound(cf, q)      bound or decline
       ├ has DB rows     → tests_touching_func(cf, q)       precise
       └ no DB rows      → no_data_policy fallback (file by default)
```

Comments and blank lines are stripped upstream by `strip_noop_diff_lines` (`^\s*#` and empty
lines), so a change that edits a function body and adds a module-level comment is not mistaken
for a module-level change. A file whose whole diff survives that stripping as empty changed no
executable line — neither added nor removed, since `-` lines anchor at the following post-image
line — so it contributes no impacted tests at all. Tier 1's rules already read diffs through the
same stripping, so this introduces no assumption the pipeline does not already make.

## 6. Tests that are never skipped

`untrusted_tests()` unions four signals; anything it flags runs regardless:

| Signal | Meaning |
|---|---|
| `test_meta` reports incomplete | `outcome != passed`, or `saved_procs < expected_workers + 1` — the DB's own account |
| Drove inference but has no `py_executor` rows | the coordinator ran but the worker's coverage never arrived |
| Footprint < 30 functions | the record is near-empty |
| On a `-Ray-` stage | the GPU worker is uninstrumented, only driver-side rows exist |

Plus **CPU stages always run** (`ALWAYS_RUN_STAGE_PREFIX`): `main.py` adds them to
`affected_stages`, and `coverage_tier` drops their families from `instrumented` so no block they
serve is ever pruned.

## 7. From test sets to a narrowed test-db

`coverage_tier._build_narrowing` classifies every entry of every block and removes only the `SAFE`
ones:

| Verdict | Meaning |
|---|---|
| `rule_kept` | a Tier 1 rule already asked to keep it |
| `coarse` | the entry carries `-k` and expands to many nodeids, so there is no 1:1 DB key |
| `no_data` | the entry has no rows in the DB |
| `impacted` | it entered changed code |
| `untrusted` | see the previous section |
| **`safe`** | none of the above → **removed** |

Entries are keyed by **stage family**: `A10-PyTorch-2` → `A10-PyTorch`. pytest-split assigns each
entry to exactly one shard and rebalances by duration, so only the family-level union answers "was
this entry ever captured on this stage".

A block is left untouched unless every stage family it serves is instrumented — coverage is
collected on single-GPU post-merge stages only, so blocks belonging to multi-GPU or
Post-Merge-only stages are never pruned.

Single-GPU stages left with nothing go into `coverage_dropped_stages`. Multi-GPU stages are
omitted here and re-added by Groovy under the `MULTI_GPU_FILE_CHANGED` gate.

## 8. Where the DB comes from

Only one producer exists: the `LLM/main/L0_PostMerge` job, which uploads
`<build>/cbts-coverage/cbts_pystart_report.tar.gz`. Every DB therefore describes some revision of
`main` (`artifact.COVERAGE_BRANCH`), and picking one means picking **which revision of `main`**
the selection reasons about.

### 8.1 Resolution

```
Jenkins REST lastBuild                     → newest build number N
for b in N .. N-9:                           (_MAX_PROBE)
   ranged GET the tarball                  → skip b if absent
   GET build_info.txt, parse `commit=`     → sha, or None
   lag(sha)                                → how far main moved past it
rank by (lag known, lag ascending, build descending)
```

Ranking is by **revision, not build number**: a post-merge build can be a re-run of an older
commit, so the highest build number is not necessarily the newest code. The build number is only
the tie-break, and when no candidate's lag can be measured the ranking degenerates to exactly that
tie-break — which is the pre-existing behaviour, not a regression.

### 8.2 Measuring the lag (ranking)

The lag is `ahead_by` from GitHub's compare API on `<sha>...main` — against the **tip of `main`**,
so every candidate is scored on one scale. `ahead_by` covers the full range; only the response's
`commits` array is truncated at 250.

Since every candidate revision is a commit that already merged to `main`, it can only ever be
*behind* the tip: `behind_by` stays 0 and the lag is non-negative. A non-zero `behind_by` would
mean the revision is no longer on `main` at all (history rewritten).

There is no local-git path. The CI checkout is `depth: 1, noTags: true` with a single-SHA refspec
(`trtllm_utils.checkoutSpec`), so no candidate revision is ever in the object store; a git
measurement would also answer against whatever ref it was given, and a merely stale ref returns a
*smaller* number rather than an error.

The API is queried once per distinct revision (`compare_distance` is cached), so probing ten builds
is at most ten calls. It answers unless the revision has not reached the public mirror yet (404) or
the token is missing (403 — the 60/h anonymous quota is shared across NVIDIA's egress IP and is
routinely already spent). Either way `lag` is `null`, the ranking degrades to its build-number
tie-break, and the reason is on stderr.

The token comes from the `github-cred-trtllm-ci` credential — the one `getGithubMRChangedFile`
already uses — bound around the `--print-selection` call in `_cbtsCoverageAudit` and read from
`GITHUB_API_TOKEN`.

This number ranks candidates and reports overall freshness. It is **not** what the gate decides on.

### 8.2b Measuring the drift (gating)

Ranking and gating ask different questions. "Which DB is freshest" is answered against the tip of
`main`; "does this DB still describe the code under test" is not — the code under test is the PR
head, which CI checks out directly (`env.gitlabMergeRequestLastCommit`), so the revision the DB has
to match is the PR's **merge base**. A DB sitting one commit off the tip says nothing useful about
a PR branched three hundred commits back, and the lag scores that case as fresh.

The drift is `merge_base_commit.sha` from `main...<pr head>`, then `ahead_by + behind_by` from
`<db sha>...<merge base>` — two extra calls per run, and only for the candidate ranking already
picked. Both revisions sit on the same linear history, so exactly one term is non-zero and the sum
is their plain distance.

Summing rather than picking a side is deliberate. The one dangerous failure is an edge `(F → T)`
that the code under test really has and the DB never recorded, and **both directions produce it**:

| DB is | relative to the PR base | how the edge goes missing |
|---|---|---|
| older | `ahead` | a caller added since, so `T` now reaches `F` and the DB never saw it |
| newer | `behind` | a call path deleted since, so the DB reflects a graph the PR base still has intact |

The fail-closed bound is symmetric too — a function absent from the DB force-runs either way — and
it only catches whole-function absence, never a row set that is merely too narrow. So there is no
principled basis for weighting one side, and the sum is also the only form that handles a genuinely
diverged base: a PR targeting a release branch, which a `main`-collected DB does not describe at
all, scores as the large number it is instead of slipping through on one small term.

`drift_status` (`ahead` / `behind` / `diverged` / `identical`) rides along in the same response and
is recorded, never weighted — it is there to answer empirically, later, whether one direction
actually correlates with misses.

Any step that cannot be answered leaves the drift null, which the gate reads as `unknown` and
declines: freshness that cannot be shown is not assumed.


### 8.3 What happens with the result

`--prepare DIR` does the whole fetch in one call: select, measure, stream the tarball down
(retried, and streamed rather than buffered — it runs past 200 MB), unpack it, write the selection
JSON beside the sqlite as `cbts_coverage_db.json`, and print `{path, meta}`. Groovy is left with
the two things only it can do — bind the credential and run `coverage_audit.py` over the result —
and any failure anywhere is caught and non-fatal: `coverageDb.path` stays empty, Tier 2 never
runs, and the PR gets a full run.

Those two paths reach `main.py` as `--coverage-db` and `--coverage-db-meta`, so a new selection
field needs no Groovy change. `main.py` records all of it and **gates on the drift**: past
`--coverage-max-drift` (default 30) the tier declines and the PR runs in full, on the grounds that
a DB that far from the PR's base no longer describes who touches what in the code under test. A
drift that could not be measured — including a meta file that is missing or unreadable — is
treated the same way.

All of it lands in the decision and in OpenSearch:

| Decision field | OpenSearch | Note |
|---|---|---|
| `coverage_db_build` | `l_coverage_db_build` | 0 when no DB was consulted |
| `coverage_db_commit` | `s_coverage_db_commit` | |
| `coverage_db_lag` | `l_coverage_db_lag` | ranking / overall freshness; `null` / `-1` when unmeasurable |
| `coverage_db_base_commit` | `s_coverage_db_base_commit` | the PR's merge base |
| `coverage_db_drift` | `l_coverage_db_drift` | **the gated number**; `null` / `-1` when unmeasurable |
| `coverage_db_drift_status` | `s_coverage_db_drift_status` | recorded, never weighted |
| `coverage_freshness` | `s_coverage_freshness` | `ok` / `stale` / `unknown`, empty when no DB |

so the decline rate is queryable per verdict rather than only readable in `s_reason`.

## 9. Decision output

```json
{
  "scope": "coverage",
  "affected_stages": [...],
  "affected_stage_test_counts": {...},
  "affected_stage_split_counts": {...},
  "test_db_dir_override": "cbts_test_db",
  "enable_multi_gpu": true,
  "coverage_dropped_stages": [...],
  "coverage_db_build": 2887,
  "coverage_db_commit": "50edd738...",
  "coverage_db_lag": 11,
  "coverage_db_base_commit": "9f0da65d...",
  "coverage_db_drift": 47,
  "coverage_db_drift_status": "behind",
  "coverage_no_diff_files": 0,
  "reasons": [{"source": "coverage", "impacted": 118, "untrusted": 104, ...}]
}
```

Groovy filters stages by `affected_stages` and renders each stage's test list from
`test_db_dir_override`; when that YAML is absent the stage falls back to the source test-db, i.e.
runs in full.
