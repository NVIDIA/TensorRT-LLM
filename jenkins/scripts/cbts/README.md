# CBTS — Change-Based Testing Selection

CI test-selection tool. Narrows the Jenkins stages and per-stage tests that
run, based on what the PR changed. New rules are added in Python only.

---

## Selection tiers

| Tier | Where | Basis |
|---|---|---|
| **1. Rules** | `cbts/rules/` | Path patterns and diffs. Each rule claims the files it understands and narrows to the blocks they affect. |
| **2. Coverage** | `cbts/coverage/selection/`, `cbts/coverage/tier.py` | Runs only on what Tier 1 left unclaimed (the *residual*), and only when every residual file is core Python present in the touch DB. Maps changed lines to qualnames and removes entries no changed function reaches. Declines to the full run otherwise. See `cbts/coverage/selection/SELECTION.md`. |

The touch DB is produced by the post-merge collection under
`cbts/coverage/collection/`; see `cbts/coverage/collection/COLLECTION.md`
for what it does and does not record.

## Consumption layers

CBTS narrows test cases only; Build always runs.

| Layer | Where | Action |
|---|---|---|
| **2. Stage** | `L0_Test.groovy::launchTestJobs` | Intersect the baseline-eligible stage set with affected stages, then add PackageSanityCheck (kept iff `sanity_required`) and PerfSanity (kept iff `perfsanity_required`). Pure `-Perf-` stages run only when present in `affected_stages` (not force-kept). Empty affectedSet + nothing force-kept → no-op. |
| **2.5. Split-resize** | `L0_Test.groovy::launchTestJobs` (`cbtsResizeSplits`) | Keep only shards `1..k` per narrowed stage, where `k` (duration-sized to ~2h/shard) is `affected_stage_split_counts`. |
| **3. Within-stage tests** | `L0_Test.groovy::renderTestDB` | Point trt-test-db at the narrowed `cbts_test_db/`. Each affected block's `tests:` is restricted to entries in the filter prefix subtree; unaffected blocks are dropped. |

CBTS only subtracts; anything it can't narrow → fallback to the existing
filter chain.

## Rules

Nine rules, registered in `cbts/command/main.py::RULE_CLASSES`:

| Rule | Scope | Files |
|---|---|---|
| `WaivesRule` | `waiveonly` | `tests/integration/test_lists/waives.txt` |
| `TestsDefRule` | `testdefonly` | `tests/**/*` (.py via AST; data files via dir walk-up) |
| `TestListRule` | `testlistonly` | `tests/integration/test_lists/test-db/*.yml` |
| `AutoDeployRule` | `autodeployonly` | `examples/auto_deploy/**`, `tensorrt_llm/_torch/auto_deploy/**` (excl. `.md`; other suffixes incl. images kept as potential test fixtures) |
| `VisualGenRule` | `visualgenonly` | `examples/visual_gen/**`, `scripts/visualgen_eval/**`, `tensorrt_llm/_torch/visual_gen/**`, `tensorrt_llm/media/**`, `tensorrt_llm/visual_gen/**` (excl. `.md`; reference images such as `cat_piano.png` ARE test fixtures and stay claimed; outward-facing files force fallback) |
| `SpecDecRule` | `specdeconly` | `tensorrt_llm/_torch/speculative/**`, `tensorrt_llm/models/{eagle,medusa,redrafter}/**`, `examples/{eagle,medusa,redrafter,draft_target_model,ngram}/**`, `examples/llm-api/llm_speculative_decoding.py` (excl. `.md`; other suffixes incl. images kept as potential test fixtures) |
| `AgentFlowRule` | `agentflowonly` | `agent-flow/**` (excl. `.md`) |
| `OpenEngineRule` | `openengineonly` | `tensorrt_llm/grpc/openengine/**` (excl. `.md`) |
| `OutOfScopeRule` | `noop` | QA / dev test lists, `.test_durations`, `microbenchmarks/`, `**/*.md` (image suffixes intentionally not claimed — image fixtures cannot be distinguished from doc diagrams by location, so image edits fall back to baseline) |

See `cbts/rules/README.md` for per-rule logic.

## Scopes

| Scope | Meaning |
|---|---|
| `waiveonly` | `WaivesRule` fired solo: PR only edits `waives.txt`. |
| `testdefonly` | `TestsDefRule` fired solo: PR only edits files under `tests/**/*`. |
| `testlistonly` | `TestListRule` fired solo: PR only adds entries under `tests/integration/test_lists/test-db/*.yml`. |
| `autodeployonly` | `AutoDeployRule` fired solo: PR only touches AutoDeploy source paths (`examples/auto_deploy/**`, `tensorrt_llm/_torch/auto_deploy/**`; excl. `.md`). Narrows to AD-only blocks (`backend: autodeploy` plus blocks containing `test_llm_api_autodeploy.py` / `_autodeploy-` entries). |
| `visualgenonly` | `VisualGenRule` fired solo: PR only touches VisualGen internal source paths (`examples/visual_gen/**`, `scripts/visualgen_eval/**`, `tensorrt_llm/_torch/visual_gen/**`; excl. `.md`; image fixtures like `cat_piano.png` are claimed). Narrows to blocks containing VG test entries. Outward-facing files under `tensorrt_llm/visual_gen/**` and `tensorrt_llm/media/**` (eagerly imported by `trtllm-serve`) force `null` fallback. |
| `specdeconly` | `SpecDecRule` fired solo: PR only touches speculative-decoding source paths (`tensorrt_llm/_torch/speculative/**`, `tensorrt_llm/models/{eagle,medusa,redrafter}/**`, `examples/{eagle,medusa,redrafter,draft_target_model,ngram}/**`, `examples/llm-api/llm_speculative_decoding.py`; excl. `.md`). Narrows to blocks containing spec-dec test entries (eagle / medusa / redrafter / ngram / draft-target-model / MTP). |
| `agentflowonly` | `AgentFlowRule` fired solo: PR only touches `agent-flow/**` source or test files (excl. `.md`). Runs `CPU-AgentFlow-UnitTest`. |
| `openengineonly` | `OpenEngineRule` fired solo: PR only touches `tensorrt_llm/grpc/openengine/**` source files (excl. `.md`). Narrows to the registered OpenEngine unit test, which lives on the always-run `CPU-Generic-*` stages. |
| `testsonly` | Multiple rules from the testsonly family fired (`waiveonly`, `testdefonly`, `testlistonly`, `autodeployonly`, `visualgenonly`, `specdeconly`, `agentflowonly`, `openengineonly`); their narrows union. |
| `noop` | Rule(s) fired but determined no test stages need to run (QA-only path, removals-only test list, all-miss waives, in-namespace .py with no covering YAML entry, docs-only edits). Layer 2 still applies. |
| `null` (fallback) | A rule cannot decide, scopes don't combine, or there are unhandled files. Groovy defers to baseline filter chain. |

`_combine_scopes` (`cbts/command/main.py`): rules with `scope="noop"` give way to any
actionable rule that also fired. Identical actionable scopes pass through;
testsonly-family mixes combine to `testsonly`; anything else returns
`None`.

A rule can yield `noop` (no impact), a narrow (actionable scope), or
`None` (cannot decide → fallback). Files matched by a rule are claimed;
unclaimed files in `pr.changed_files` → fallback.

## File map

```
jenkins/scripts/cbts/                  outer vendor dir — put on PYTHONPATH for `import cbts.*`
├── README.md              this file
├── cbts/                  the importable package
│   ├── __init__.py
│   ├── blocks.py          YAML index + path/waive lookup + filtered tmp test-db generation + per-stage count
│   ├── rules/
│   │   ├── README.md      per-rule logic
│   │   ├── base.py        Rule ABC + PRInputs + RuleResult
│   │   ├── _helpers.py    diff iteration + lookup-into-block_filters + stages_by_yaml_stem
│   │   ├── waives.py
│   │   ├── tests_def.py
│   │   ├── test_list.py
│   │   ├── auto_deploy.py
│   │   ├── visual_gen.py
│   │   ├── spec_dec.py
│   │   ├── agent_flow.py
│   │   ├── openengine.py
│   │   └── out_of_scope.py
│   ├── coverage/
│   │   ├── tier.py        Tier 2 entry: applies the selector to the test-db YAMLs, classifies every candidate entry
│   │   ├── selection/
│   │   │   ├── SELECTION.md  how a decision is made: qualname concepts, decline gates, narrowing
│   │   │   ├── selector.py   CoverageSelector.decide(): changed lines → qualnames → impacted / skippable per stage family
│   │   │   ├── qualname_map.py  changed lines → co_qualname, plus the import-time and closure classifications
│   │   │   └── touch_db.py   read-only accessor over cbts_touchmap.sqlite + the untrusted-capture signals
│   │   └── collection/    post-merge collection that produces the touch DB (see its README / COLLECTION.md)
│   └── command/           unified CLI, `python -m cbts.command ...`
│       ├── main.py        `main` — CLI entry + Selector + SelectionResult + scope combine + trigger-mode filter
│       ├── dryrun.py       `dryrun` — replay CBTS over historical commits → per-PR summary.txt + filtered YAMLs + INDEX.md (debug only)
│       ├── report_decision.py  `report-decision` — post the decision (hit-stage count, case-level skip rate, fallback) to OpenSearch for CI-health monitoring
│       └── coverage/
│           ├── pilot.py        `coverage pilot` — resolve the PR author for the coverage pilot policy
│           ├── selection/
│           │   ├── artifact.py `coverage selection artifact` — resolve and merge the x86/SBSA post-merge touch DBs
│           │   ├── audit.py    `coverage selection audit` — report a touch DB's format, scale, untrusted rate and HEAD coverage gap
│           │   └── explain.py  `coverage selection explain` — explain one commit's decision case by case (delegates to CoverageSelector)
│           └── collection/
│               ├── pystart_report.py    `coverage collection pystart-report` — union leaf coverage DBs into the touch DB / report
│               └── compact_touch_db.py  `coverage collection compact-touch-db` — one-shot compact-schema sizing spike (dev only)
└── cbts_injectors/        bare-importable guest hooks (NOT part of the `cbts` package)
    ├── sitecustomize.py   loaded automatically by Python's `site` machinery in every instrumented subprocess
    └── cbts_plugin.py     pytest plugin, loaded via `-p cbts_plugin`
```

## Lookup algorithms

`YAMLIndex.find_match_for_waive` (waive ids): walks the pytest tree from
the waive id toward the root; the first level whose YAML has a matching
entry becomes the filter prefix for that block. Prefixes remember their
originating waive id(s) so `write_filtered_test_db` can apply the `-k`
keyword guard.

```
waive id (raw)
   ↓ normalize     strip SKIP/TIMEOUT/full:gpu/comments
   ↓ strip [params] if present
target_lookup     (function/class/file/dir level)
   ↓ try YAML at this level
       hit  → filter prefix = level (with originating waive ids)
       miss → strip one level up and retry
   ↓ all levels miss → recorded as miss; rule decides noop or partial-narrow
```

An entry matches a level when its canonical target (with `SKIP`/`TIMEOUT`/
`full:gpu`/`-k`/`-m`/`[params]` stripped) equals the level AND any `-k`
keyword filter on the entry contains an identifier from the waive id. `-m`
markers always pass (unverifiable from string).

The `-k` keyword guard runs twice: once at lookup, once when writing
`cbts_test_db/` (drops sibling entries whose `-k` doesn't match the waived
test).

`YAMLIndex.find_match_for_path` (testdef anchors): bidirectional pytest-
tree lineage. For an anchor `path::Class::method`, matches YAML entries
that share any common ancestor (sibling methods, sibling classes within
the same file, sibling files within the same dir). Files whose basename
doesn't start with `test_` (conftest.py, __init__.py, helper modules,
data files like `references/*.yaml` or `test_configs/*.yaml`) anchor on
their enclosing dir; if no YAML entry covers that dir, the lookup walks
up one directory at a time to the narrowest YAML-covered ancestor.

`YAMLIndex.git_path_to_yaml_key` (testdef path translation): maps a repo-
relative git path to its YAML namespace form by finding the first path
component that appears at the top of any YAML entry's canonical target.
Returns `None` for paths outside any YAML-referenced tree (top-level
integration conftest, helper modules in dirs no YAML mentions).

## When CBTS activates

CBTS activates on bare `/bot run` and `/bot run --post-merge`. Any
stage-selection flag (`--stage-list`, `--extra-stage`, `--gpu-type`,
`--test-backend`, `--skip-test`, `--add-multi-gpu-test`, `--only-multi-gpu-test`,
`--disable-multi-gpu-test`) makes `getCbtsResult` return null.

`--disable-cbts` is an explicit kill switch: `getCbtsResult` returns null
before any narrowing, so the pipeline runs the full test set. The opt-out is
recorded in OpenSearch with `s_cbts_status=disabled`.

Orthogonal flags (`--reuse-test`, `--disable-reuse-test`, `--debug`,
`--detailed-log`, `--disable-fail-fast`, `--high-priority`) do not affect CBTS.

## Trigger-mode filter

`pr.post_merge` (carried in INPUT_JSON) selects which stages survive:

- `post_merge=False` (default for `/bot run`): drop every stage whose name
  contains `Post-Merge`.
- `post_merge=True` (`/bot run --post-merge`): keep all affected stages —
  pre-merge plus `Post-Merge` (Post-Merge runs on top of pre-merge, matching
  the non-CBTS baseline).

Applied after rules union; rules see all stages so reasons report the
pre-filter narrow. `_log_decision_to_stderr` prints the dropped set for
Jenkins console diagnostics.

## How it's invoked (CI)

`getCbtsResult` calls `cbts.command main` twice on the L0_MergeRequest agent
(via `PYTHONPATH=jenkins/scripts/cbts python3 -m cbts.command main ...`):

1. `main --list-needed-diffs` → file patterns whose diffs Groovy must fetch
   (Ant-style globs).
2. `main cbts_input.json` → decision JSON on stdout. When any block was
   narrowed, writes `${LLM_ROOT}/cbts_test_db/` with the affected YAMLs and
   only their affected blocks (kept entries preserve `TIMEOUT (n)`,
   `ISOLATION`, `-k`, `-m` verbatim).

INPUT_JSON:

```json
{
  "changed_files": ["tests/..."],
  "diffs": {"tests/...": "@@ ..."},
  "post_merge": false
}
```

Decision JSON:

```json
{
  "scope": "testsonly",
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "reasons": [
    "[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages",
    "[testdef] testdef: 1 path(s) → 1 blocks, 2 stages"
  ],
  "test_db_dir_override": "cbts_test_db",
  "affected_stage_test_counts": {"A10-PyTorch-1": 5, "A10-PyTorch-2": 5},
  "affected_stage_split_counts": {"A10-PyTorch-1": 1, "A10-PyTorch-2": 1},
  "sanity_required": false,
  "perfsanity_required": false
}
```

- `scope: null` → fallback; Groovy defers to baseline.
- `scope: "noop"` → rule(s) fired but no narrow contribution; affected
  stages may be empty. Layer 2 still honors `sanity_required` /
  `perfsanity_required`.
- `test_db_dir_override: null` → no Layer 3 narrowing; trt-test-db reads
  the source test-db.
- `affected_stage_test_counts` → per-stage kept-entry count (telemetry).
- `affected_stage_split_counts` → per-stage duration-sized split count (Layer 2.5).

## Cross-job seed for stage agents

`cbts_test_db/` is written on the L0_MergeRequest agent and is not
available to downstream `L0_Test-*` pods. To deliver it per stage:

1. `getCbtsResult` tars `cbts_test_db/` and uploads it to Artifactory,
   recording the path in `result.cbts_test_db_artifact_path` (rides along
   inside `testFilter`).
2. `renderTestDB` on the stage agent downloads and extracts that tarball
   into `${llmSrc}/${test_db_dir_override}`; trt-test-db then renders from
   the narrowed test-db.

If the upload or the download/extraction fails, the override directory is
absent and `renderTestDB` falls back to the source test-db. Layer 2 still
applies. The tarball carries only the narrowed YAMLs, so no PR diff text
travels between jobs.

## Split-resize heuristic (Layer 2.5)

`blocks.compute_stage_split_counts` sizes each narrowed stage to
`k = clamp(ceil(est_seconds / 2h), 1, stage.total_splits)` (cap parsed from
`L0_Test.groovy`). `est_seconds` sums the `.test_durations` cache over the
stage's kept entries (exact node-id, else subtree-sum, else average — over-counts
toward the cap, never under-sizes). `launchTestJobs::cbtsResizeSplits` keeps only
shards `1..k`; pytest-split then balances within them via `least_duration`.

`cbtsResizeSplits` also renames each narrowed stage with a `-cbts` suffix so its
narrowed result is never reused (whole-stage `REUSE_STAGE_LIST` or per-test
`reusePassedTestResults`) by a non-CBTS full run on the same commit. A suffix (not
a prefix) keeps the GPU type as the first `-` token, so positional stage-name
parsers need no change; full sanity / PerfSanity stages keep their original names
and reuse normally.

## Adding a new rule

1. **Create `cbts/rules/my_rule.py`** subclassing `Rule`:

   ```python
   from typing import Optional
   from cbts.blocks import YAMLIndex, Stage
   from .base import PRInputs, Rule, RuleResult

   class MyRule(Rule):
       name = "myrule"
       needs_diff_for = ("tests/**/*.py",)   # Ant globs; tuple per RUF012

       def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]):
           self.yaml_index = yaml_index
           self.stages = stages

       def apply(self, pr: PRInputs) -> Optional[RuleResult]:
           ...
           return RuleResult(
               handled_files={...},
               affected_stages={...},
               scope="myscope",          # or "noop"; None=fallback
               reason="why this fired",
               sanity_relevant=False,
               perfsanity_relevant=False,
               # Optional Layer 3 contribution: per-block prefix →
               # originating waive ids. Selector unions across rules.
               block_filters={
                   (yaml_stem, block_index): {
                       filter_prefix: {originating_waive_id, ...},
                   },
                   ...
               },
           )
   ```

2. Register in `cbts/command/main.py::RULE_CLASSES` (and adjust
   `build_rules()` if the constructor signature differs).

3. No Groovy edits needed.

4. If the new scope name should combine with existing testsonly-family
   scopes, add it to `_TESTSONLY_FAMILY` in `cbts/command/main.py`.

`Selector` unions `affected_stages` and `block_filters`; scopes are
combined via `_combine_scopes`.

## Fallback paths

CBTS defers to the existing filter chain when:

- PostMerge job / `alternativeTRT` set
- `changed_files` is empty
- `cbts.command main` throws or stdout is unparsable
- Python returns `scope: null`
- A file in `pr.changed_files` is unhandled by every rule (e.g. a `.py`
  file outside any YAML namespace)
- A rule's `apply()` returns `RuleResult(scope=None, ...)` (rule fired
  but cannot decide; e.g. testdef blast-radius cap, testlist structural
  YAML edit)
- Combined scope is `None` (incompatible mix)
- Tier 2 declines: a residual file is not core Python, is absent from the
  touch DB, has an import-executed change (module / class body, signature or
  decorator line), has no usable patch, has unparsable source, or has a closure
  change with no wider row set (see `cbts/coverage/selection/SELECTION.md` §3-4)
- No touch DB artifact could be resolved — Tier 2 never runs
- The resolved DB sits more than `--coverage-max-drift` commits from the PR's
  base commit, on either side, or an unmeasurable distance from it — Tier 2
  declines (`coverage_freshness` = `stale` / `unknown`)
- Layer 3 narrowing would empty a block — block keeps original tests
- `cbts_test_db` tarball upload or download/extraction fails — renderTestDB falls back to source
- Narrowed YAML missing/empty on a stage agent — renderTestDB falls back

Every fallback emits an `echo` log line.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors Groovy
`getMakoArgsFromStageName` (~`L0_Test.groovy:2079`) and
`parseTaskConfigFromStageName` (~`:2066`). Update both when adding new
backends / orchestrators / stage-name conventions.
