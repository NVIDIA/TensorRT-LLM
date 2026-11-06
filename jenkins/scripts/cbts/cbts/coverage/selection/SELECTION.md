<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS Coverage Selection


How the touch DB is turned into a decision about which cases to run. Collection side:
`../collection/COLLECTION.md`.

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

Tier 2 only ever looks at the **residual**: the files no Tier 1 rule claimed. The Git
compatibility gate uses that same residual. Conflicts in files already claimed by Tier 1, such as
`waives.txt`, do not disable Tier 2.

## 2. The qualname concepts

Precision comes down to which qualname a changed line lands on. The shared
`python_change_analysis` module's attribution:

| Changed line sits in | Attributed to | Note |
|---|---|---|
| A function or method body | `Class.method` / `func` | precise |
| A closure body | `outer.<locals>.inner` | also precise: recorded under its own qualname, same as any function |
| A module-level statement | `<module>` | module body, executed once at import |
| A class body (class attribute) | `ClassName` | class body, executed once at import |
| **A signature or decorator line** | **the enclosing scope** | a method's `def` line lands on `ClassName`, not on the method |
| A lambda or genexpr body | **the nearest enclosing named def** | `_collect_scopes` only discovers named `def`/`class` statements; changing a lambda in `def outer(): f = lambda: ...` lands on `outer` |

The last two are where both the precision and the correctness problems come from: **the
attributed qualname's DB rows do not represent the changed code's executions.**

DB qualnames come from `co_qualname`, so they match what the AST derives (`Class.method`,
`outer.<locals>.inner`), but collection drops lambda and genexpr frames since selection has no way
to resolve a line to them (`COLLECTION.md` §5.1), so those names do not exist in the DB at all.

## 3. The decline gates

`CoverageSelector.decide()` returns `ok=False` — Tier 2 stands down and the run is full — on any
of:

| Gate | Condition |
|---|---|
| Non-core file | `path` is not a `.py` under `tensorrt_llm/` (checked on the **repo path**, not `canon()`) |
| File absent from the DB | `file_has_touch_rows(cf)` is false — new or uninstrumented, so "who touches it" is unknown rather than empty |
| **Unbounded import-executed change** | a changed line lands on a class body, signature / decorator, or unsupported module statement; the conservative exceptions are below |
| No usable patch | the forge API omitted the diff (binary / rename / oversized), so the changed scope is unknown |
| Unparsable source | the AST walk failed, so lines cannot be mapped to qualnames |
| Closure change with no wider row set | see the next section |

## 4. Bounding low-risk import-executed changes

Every process, including an `mpi4py.futures` pool worker, records its own import phase (see
`../collection/COLLECTION.md` §5.2); only the outer pytest's own pre-test import is unattributed,
since it runs under the empty context and the merge filters that out with `WHERE test != ''`. So a
`<module>` or `ClassName` row set can still fall short of the full set of tests that import the file:
a test that imports it and never enters any function in it records nothing there at all. Widening to
the file's row set recovers those cases but is a **recovery heuristic, not an upper bound** for the
same reason — a test whose only interaction with the file is the bare import still records nothing,
so it is absent from the file set too.

Consequently, Tier 2 never uses `<module>` or class-body rows as an impact bound. Most import-executed
changes still decline. A small AST allowlist can instead translate these low-risk module statements to
the local functions and methods that consume the changed binding:

- a module binding whose value is a literal container/scalar, including replacement when both
  the old and new statements are safe assignments to the same names;
- an annotated literal when `from __future__ import annotations` postpones annotation evaluation;
- a newly added builtin-module `import` (replacement imports remain unsupported);
- an unambiguous top-level `from ... import ...` binding delta, computed from the complete pre- and
  post-image; old targets must resolve to static declarations and new targets to static top-level
  functions before their target and local-consumer rows are selected;
- imports added inside an import-only `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` block whose
  guard comes directly from `typing` and has not been rebound; these are ignored because the block is
  statically false at runtime;
- a newly added function or method with only unshadowed builtin `classmethod`, `property`, or
  `staticmethod` decorators, literal defaults, and annotations validated as safe name loads or
  supported builtin / `typing` / `collections.abc` forms; plain, unshadowed `torch.Tensor` and
  `torch.dtype` annotation loads are also supported;
- a new top-level dataclass using a unique, unchanged `from dataclasses import dataclass`
  binding (aliases are supported), optionally called with a literal `frozen` boolean. These
  classes must have no bases, metaclass keywords, type parameters, or executable class-body
  statements. Fields require unshadowed builtin type annotations and immutable literal defaults
  in valid order. Methods use the safe declaration checks above with builtin eager annotations;
  duplicate members, special methods, namespace reflection, and unsupported decorators decline.
  The class body and its new methods are bounded together by functions consuming the class
  binding, retaining external reference checks and the normal no-data fallback;
- added parameters on undecorated functions or methods with safe annotations and literal defaults
  when provided, preserving the existing parameters' declarations, relative order, and defaults,
  along with positional-only parameters, variadic parameters, and the type comment;
- a safe return annotation added to an undecorated function or method, with the same restrictions on
  any simultaneous parameter additions;
- a literal module binding deletion, retaining consumers from both the pre- and post-image; unsupported
  deletions adjacent to a declaration still decline rather than being ignored as signature changes.

The AST layer reports bindings, consumers, and direct local caller edges; it does not infer visibility
from underscores or any other spelling convention. Before using those facts, the selector checks the
repository import/reference graph. A binding referenced from another file declines, as does an
externally referenced no-data consumer whose local callers would otherwise be used as its bound.
Every no-data caller reached during recursion is checked too; an external reference there declines
the coverage decision instead of treating local callers or file rows as a complete bound.

The analysis also declines if either image contains an attribute whose name matches a changed module
binding, because the attribute receiver may alias the module and its consumers are not bounded by
direct name loads. Signature additions on decorated functions or methods decline even when the
decorator expression is unchanged: decorators can inspect signatures and register schemas during
import, with consumers that never execute the Python function.

Validated annotation loads of new class bindings do not count as import-time execution consumers;
their function declarations already contribute impact. Other import-time uses of the class, such
as registration calls or module-level construction, still decline. Changes to existing decorated
method signatures remain unsupported even when safe dataclass additions occur in the same file.

The dataclass namespace-discovery check scans the entire post-image, including unchanged functions.
It rejects `dir`, `__import__`, and `import_module` along with the other namespace APIs, whether
loaded directly, accessed as attributes, or imported under an alias. Attribute checks match by name
without resolving the receiver: any `.modules` or `.__dict__` access declines, including an unrelated
`model.modules()` call. This conservative limit also covers aliases of `sys.modules`.

New classes without local consumers still go through external-reference checks. Registered external
unit tests are selected; production-code references, unregistered test references, or references with
unknown locations decline. A class with neither local consumers nor external references can have no
impacted tests.

The analysis also declines if the binding is consumed by import-executed code. Otherwise its consumers
are resolved exactly like changed function bodies. If a consumer is new or absent from the DB, Tier 2
may walk direct, unshadowed local caller edges until every branch reaches a qualname with DB rows. An
unresolved branch, cycle, or use of the function as a value falls back to the configured no-data policy
(the whole file by default).
This is why `_LINUX_EBADHANDLE = 521` can be bounded through `_is_lock_infra_error` and its recorded
callers without treating all import-time execution as safe or assuming that `_` means private.

### 4.1 Closures

A named closure's body is recorded under its own qualname (`COLLECTION.md` §2.1), so both ways a
changed line reaches it resolve to the closure itself, not its enclosing def:

- **A direct line change inside the closure** resolves to that qualname through `qualnames_for_lines`
  and goes through the ordinary `_qualname_impact` path below like any other function:
  `tests_touching_func(f, q)` if it has rows, a local-caller bound or the configured no-data fallback
  (file-level by default) if it does not yet.
- **Binding-consumer resolution** — "does this def use a binding that just changed" — is a separate
  mechanism, `_FunctionNames` in `python_change_analysis.py`. `_recorded_definition_groups` records a
  nested def under its own qualname, so it gets its own fact there too; only a nested def's *signature*
  (decorators, defaults, annotations), which genuinely runs in the outer scope at definition time, rolls
  up into the outer def's own fact (`_include_signature_loads`) — its body, which only runs when the
  nested def itself is later called, gets its own independent `_FunctionNames` fact instead, same as
  any other qualname in `definition_groups`. A closure built by one test and invoked by a different one
  (a cached decorator, a session-scoped fixture) is why this distinction matters: resolving to the outer
  def would only find the constructing test, missing the one that actually depends on the binding.

One case remains conservative by construction: **a lambda or genexpr body** attributes to its nearest
enclosing *named* def, since `_collect_scopes` only discovers named `def`/`class` statements. A lambda or
genexpr can appear anywhere an expression can, nested inside each other and themselves to arbitrary depth,
so there is no static-analysis path to resolve a changed line to it directly; collection follows this by
dropping their frames too (`COLLECTION.md` §5.1). The enclosing def's rows stand in instead, but only
reflect when the enclosing def *itself* ran, not when the lambda inside it did, so the gap is the same
shape as the cached-closure case above: a lambda created by one test and invoked by another gives the
invoking test no rows on the enclosing def, and it can be missed.

## 5. Full impact resolution

```
for each residual file:
  ├ no usable diff / unparsable → decline
  ├ diff has no lines           → contributes nothing (comment / blank only)
  └ for each changed qualname:
       ├ supported module binding/declaration
       │    ├ external reference / import-time consumer → decline
       │    └ local consumers → rows / local caller bound / no-data fallback
       ├ other import-executed change          → decline
       ├ has DB rows     → tests_touching_func(cf, q)       precise (includes closures, §4.1)
       └ no DB rows      → no_data_policy fallback (file by default)
```

Comments and blank lines are stripped upstream by `strip_noop_diff_lines` (`^\s*#` and empty
lines), so a change that edits a function body and adds a module-level comment is not mistaken
for a module-level change. A file whose whole diff survives that stripping as empty changed no
executable line — neither added nor removed, since `-` lines anchor at the following post-image
line — so it contributes no impacted tests at all. Tier 1's rules already read diffs through the
same stripping, so this introduces no assumption the pipeline does not already make.

Verified literal docstring edits have a separate `DocsRule` contribution. For a
mixed file, coverage uses its executable selection diff to compute changed lines
and deletions, while reconstructing the pre-image from the original patch.
Docstring edits therefore do not add function, class, or module impact here;
the remaining code changes retain the same narrowing and decline policies.
The resulting stage set unions with every rule contribution, including
`CPU-Build_Docs`, which is not represented by a test-db block.

## 6. Tests that are never skipped

`untrusted_tests()` unions four signals; anything it flags runs regardless:

| Signal | Meaning |
|---|---|
| `test_case_meta` reports incomplete | `outcome` is missing or not `passed`, or `tainted != 0` — the compact DB's logical completeness view |
| Drove inference but has no `py_executor` rows | the coordinator ran but the worker's coverage never arrived |
| Footprint < 30 functions | the record is near-empty |
| On a `-Ray-` stage | the GPU worker is uninstrumented, only driver-side rows exist |

`tainted` is the collection side's own admission that the context channel could not vouch for a
record (`../collection/README.md`, "Taint"). It comes in two kinds — `attribution`, where rows
recorded under a test may belong to another one, and `incomplete`, where the rows are right but some
are missing. Selection treats them alike: either way the test's row set cannot bound what the test
covers, so it runs. `taint_rows` carries the kind, the reason and the process for anyone reading the
map by hand.

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

Only one producer exists: the `LLM/main/L0_PostMerge` job. After each architecture's single-GPU
stages finish, it uploads `cbts_pystart_report_x86_64.tar.gz` and
`cbts_pystart_report_SBSA.tar.gz`. Every pair therefore describes one revision of `main`
(`artifact.COVERAGE_BRANCH`). The later full-report tarball is not used for test selection.

### 8.1 Resolution

```
run Tier 1                                      → Tier-2 residual paths
GitHub compare main...<PR head>                 → PR base commit
GET Artifactory pin at <PR number>/<PR head>:
   200                                           → reuse its coverage build
   404                                           → resolve the newest build below
   other / malformed                             → decline Tier 2
when no pin:
   Jenkins REST lastBuild                        → newest build number N
   for b in N .. N-49:                            (_MAX_PROBE)
      ranged GET both architecture tarballs      → skip b unless both exist
      GET build_info.txt, parse `commit=`         → sha; skip when absent
      first complete pair with a known sha        → latest coverage DB
   Python writes the build/commit pin locally
   Jenkins uploads that file                      → stable for this PR head
compare <sha>...<PR base>                        → record topology and absolute distance
fetch PR head locally; fetch base and selected DB from the normal CI Git mirror
create a squashed PR commit with `commit-tree`, parented at the PR base
cherry-pick it onto the DB revision
   no unmerged residual path                       → continue
   residual conflict / unavailable revision       → decline Tier 2
```

The pin lives under
`LLM/main/cbts/coverage-db-pins/v1/<PR number>/<PR head>/cbts_db_pin.json`. A new PR head has no
pin and therefore receives the freshest complete pair available at its first CBTS run. Repeated
`/bot run` commands for the same head reuse that build even after newer post-merge DBs appear.
`artifact.py --resolve-pin` owns pin lookup, schema validation, build selection, local pin
creation, and fail-closed reasons. Groovy only supplies PR identity and uses Jenkins'
authenticated artifact uploader when Python reports `pin_upload_required`. Lookup, validation,
and first-write upload fail closed; CBTS never silently substitutes a newer build when a pin
cannot be read or written. Preparation also verifies that the pinned build's `build_info.txt`
still names the commit recorded in the pin, so an overwritten or corrupted build cannot silently
change a repeated run.

Requiring the architecture pair prevents the selector from narrowing only one CPU architecture.
For an unpinned head, builds are probed newest first and the first complete pair with commit
metadata is the only DB considered. The selector does not substitute an older DB merely because
it is closer to the PR base or because the selected DB conflicts with the PR diff.

### 8.2 Measuring the lag (reporting)

After selection, lag is `ahead_by` from GitHub's compare API on `<sha>...main`. It reports overall
freshness against the tip of `main`; it no longer ranks candidates. `ahead_by` covers the full
range; only the response's `commits` array is truncated at 250.

Since every candidate revision is a commit that already merged to `main`, it can only ever be
*behind* the tip: `behind_by` stays 0 and the lag is non-negative. A non-zero `behind_by` would
mean the revision is no longer on `main` at all (history rewritten).

There is no local-git path for measuring lag. The CI checkout is `depth: 1, noTags: true` with a
single-SHA refspec (`trtllm_utils.checkoutSpec`), so the coverage revision is not available there.
The conflict check creates a temporary repository and fetches the checked-out PR head locally.
The base and DB revisions come from the authoritative public
`https://github.com/NVIDIA/TensorRT-LLM.git` repository. This avoids depending on the replication
delay or commit availability of the internal checkout mirror. The temporary repository never
changes the CI checkout or its index.

The compare API answers unless the revision has not reached the public mirror yet (404) or the
token is missing (403 — the 60/h anonymous quota is shared across NVIDIA's egress IP and is
routinely already spent). An unmeasurable candidate-to-base relation is rejected; only the selected
candidate's main-tip lag may remain `null`.

The token comes from the `github-cred-trtllm-ci` credential — the one `getGithubMRChangedFile`
already uses — bound around the artifact selection call in `_cbtsCoverageAudit` and read from
`GITHUB_API_TOKEN`.

This number reports overall freshness. It does not select the DB.

### 8.2b Measuring the drift (reporting)

The PR base is `merge_base_commit.sha` from `main...<PR head>`. The selected DB is compared as
`<db sha>...<PR base>`, and drift is `ahead_by + behind_by`: an absolute distance retained for
telemetry. `ahead`, `behind`, and `identical` describe valid positions on main; diverged and
unknown relations decline Tier 2 because the selector cannot establish a shared main history.

`commit-tree` represents the complete base-to-head PR change as one commit, and `cherry-pick`
tests that commit against the selected DB without serializing through patch format. When the
cherry-pick reports conflicts, only unmerged paths in the Tier-2 residual count; conflicts in
Tier-1-owned files are ignored. This check is independent of whether the DB is older than, equal
to, or newer than the PR base. A residual conflict or an unmeasurable check declines before the
large DB artifacts are downloaded. If it passes, Tier 2 uses the original forge PR payload.


### 8.3 What happens with the result

After Tier 1 computes the residual, `--resolve-pin FILE` returns a reuse/create/decline plan.
Groovy uploads `FILE` only for a create plan, before any large download. `--prepare DIR --build
BUILD --paths-json PATH` then resolves the PR base, validates the pinned pair's residual
compatibility, streams both tarballs down, unpacks their identically named SQLite files
separately, and unions them with `compact_db.merge_databases`.
It writes the selection JSON
beside the merged SQLite as `cbts_coverage_db.json` and prints `{path, meta}`. Groovy binds the
credentials, logs the successful compatibility check, and runs `coverage_audit.py` over the result.
On a residual conflict it returns `{path: null, meta}` so the decline remains observable without
downloading the DB. Any failure is non-fatal: Tier 2 never runs and the PR gets a full run.

The metadata path always reaches `main.py`; the DB path is added only when compatibility is clean.
`main.py` records the drift but does not gate on its size. The residual compatibility check is the
authority for whether a DB collected before or after the PR base can safely be used.

All of it lands in the decision and in OpenSearch:

| Decision field | OpenSearch | Note |
|---|---|---|
| `coverage_db_build` | `l_coverage_db_build` | 0 when no DB was consulted |
| `coverage_db_commit` | `s_coverage_db_commit` | |
| `coverage_db_lag` | `l_coverage_db_lag` | overall freshness only; `null` / `-1` when unmeasurable |
| `coverage_db_base_commit` | `s_coverage_db_base_commit` | the PR's merge base |
| `coverage_db_drift` | `l_coverage_db_drift` | distance from the PR base; `null` / `-1` when unmeasurable |
| `coverage_db_drift_status` | `s_coverage_db_drift_status` | `ahead` / `behind` / `identical` for every selected DB |
| `coverage_freshness` | `s_coverage_freshness` | drift measurement status: `ok` / `unknown`, empty when no DB |
| `coverage_compatibility` | `s_coverage_compatibility` | `clean` / `conflict` / `unknown` / `not_attempted` |
| `coverage_decline_reason` | `s_coverage_decline_reason` | human-readable Tier-2 decline detail |
| `coverage_decline_category` | `s_coverage_decline_category` | aggregation-safe decline category |
| count of `coverage_residual_files` | `l_coverage_residual_files` | Tier-2 opportunity size |

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
  "coverage_db_drift": 7,
  "coverage_db_drift_status": "ahead",
  "coverage_residual_files": ["tensorrt_llm/example.py"],
  "coverage_compatibility": "clean",
  "coverage_decline_reason": "",
  "coverage_decline_category": "",
  "coverage_no_diff_files": 0,
  "reasons": [{"source": "coverage", "impacted": 118, "untrusted": 104, ...}]
}
```

Groovy filters stages by `affected_stages` and renders each stage's test list from
`test_db_dir_override`; when that YAML is absent the stage falls back to the source test-db, i.e.
runs in full.
