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

## 3. The three decline gates

`CoverageSelector.decide()` returns `ok=False` — Tier 2 stands down and the run is full — on any
of:

| Gate | Condition |
|---|---|
| Non-core file | `path` is not a `.py` under `tensorrt_llm/` (checked on the **repo path**, not `canon()`) |
| File absent from the DB | `file_has_touch_rows(cf)` is false — new or uninstrumented, so "who touches it" is unknown rather than empty |
| Under-recorded with no wider bound | see the next section |

## 4. Core: handling an under-recorded qualname

Two kinds of qualname have DB rows that **do not cover the changed code's executions**:

| Kind | Attributed qualname | Collection-side problem |
|---|---|---|
| Import-time (module body / class body / signature line) | itself | only recorded by tests that spawn subprocesses; a pool worker's import precedes activation |
| Closure | the enclosing function | **not recorded at all**; a closure can outlive the call that created it |

Both go through `_underrecorded_bound(cf, qualname)`:

```
file rows > that qualname's rows  →  use the file rows (the wider bound)
otherwise                        →  decline, run in full
```

### 4.1 What the comparison asks

Whether falling back to file level actually recovers the tests that were missed.

`tests_touching_func(f, q)` is by construction a subset of `tests_touching_file(f)`, so a strictly
larger file set means it contains tests the qualname set does not. Those extra tests are the ones
that recorded some *other* function in the same file — evidence that the file is exercised beyond
import time, which is what makes the file set a usable bound. When the two are equal the file is
only ever recorded at import time (`__init__.py` and similar, 881 of 1149 files), nothing is
recovered, and no sound bound exists.

### 4.2 The bound is not complete

A test that imports a file but never calls any function in it records nothing there, so even the
file set misses it. The bound is the widest one available from the data, not a proof; eliminating
the remainder means recording the pool workers' import phase — see
`../coverage_utils/COLLECTION.md` §5.2.

### 4.3 When no diff is available

The forge API omits the patch for binary, renamed and oversized files. The changed qualname is
then unknown, so the worst case is assumed and `<module>` is put through the same test; the count
of such files is reported as `coverage_no_diff_files`.

## 5. Full impact resolution

```
for each residual file:
  ├ no usable diff        → _underrecorded_bound(cf, "<module>")   bound or decline
  ├ diff has no lines     → whole file (the change was comments only)
  └ for each changed qualname:
       ├ import-time or closure → _underrecorded_bound(cf, q)      bound or decline
       ├ has DB rows            → tests_touching_func(cf, q)       precise
       └ no DB rows             → no_data_policy fallback (file by default)
```

Comments and blank lines are stripped upstream by `strip_noop_diff_lines` (`^\s*#` and empty
lines), so a change that edits a function body and adds a module-level comment is not mistaken
for a module-level change.

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

### 8.2 Measuring the lag

The lag is `ahead_by` from GitHub's compare API on `<sha>...main` — always against the **tip of
`main`**, never against the PR's own base commit: every candidate is scored on the same scale, and
a PR whose base predates all the candidates would otherwise score them all identically and collapse
the ranking back to the build number. `ahead_by` covers the full range; only the response's
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

### 8.3 What happens with the result

The tarball is downloaded (retried), the sqlite extracted, and `coverage_audit.py` run over it;
any failure in this whole path is caught and non-fatal — `coverageDb.path` stays empty, Tier 2
never runs, and the PR gets a full run.

The chosen build, its commit and its lag ride into `main.py` and are recorded in the decision
(`coverage_db_build` / `coverage_db_commit` / `coverage_db_lag`; an unmeasurable lag is `null`
here and `-1` in the OpenSearch record).
**These are recorded, not gated**: observed lag varies widely even in healthy operation, so a
threshold needs data first.

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
  "coverage_no_diff_files": 0,
  "reasons": [{"source": "coverage", "impacted": 118, "untrusted": 104, ...}]
}
```

Groovy filters stages by `affected_stages` and renders each stage's test list from
`test_db_dir_override`; when that YAML is absent the stage falls back to the source test-db, i.e.
runs in full.
