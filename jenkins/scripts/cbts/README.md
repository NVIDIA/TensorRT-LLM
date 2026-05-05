# CBTS — Change-Based Testing Selection

CI test-selection tool. Narrows the Jenkins stages and per-stage tests that
run, based on what the PR changed. New rules are added in Python only.

---

## Consumption layers

CBTS narrows test cases only; Build always runs.

| Layer | Where | Action |
|---|---|---|
| **2. Stage** | `L0_Test.groovy::launchTestJobs` | Set `parallelJobsFiltered` to affected stages plus PackageSanityCheck (kept iff `sanity_required`) and PerfSanity (kept iff `perfsanity_required`). Pure `-Perf-` stages always excluded. Empty affectedSet + nothing force-kept → no-op. |
| **2.5. Split-collapse** | `L0_Test.groovy::runLLMTestlistOn*` entries | Narrowed test count < 20 → collapse pytest-split to splits=1 (only group 1 runs); else default splits stand. |
| **3. Within-stage tests** | `L0_Test.groovy::renderTestDB` | Point trt-test-db at the narrowed `cbts_test_db/`. Each affected block's `tests:` is restricted to entries in the filter prefix subtree; unaffected blocks are dropped. |

CBTS only subtracts; anything it can't narrow → fallback to the existing
filter chain.

## v0 scope

- Only handles `tests/integration/test_lists/waives.txt` changes (`scope: waiveonly`).
- Anything else → `scope: none` → full run.

## File map

```
jenkins/scripts/cbts/
├── README.md              this file
├── main.py                CLI entry + Selector + SelectionResult
├── blocks.py              YAML index + lookup + filtered tmp test-db generation + per-stage count
└── rules/
    ├── README.md          per-rule logic summary
    ├── base.py            Rule ABC + PRInputs + RuleResult
    └── waives_rule.py     v0's only rule
```

## Lookup algorithm

`YAMLIndex.find_match_for_waive` walks the pytest tree from the waive id
toward the root; the first level whose YAML has a matching entry becomes the
filter prefix for that block. Prefixes remember their originating waive
id(s) so `write_filtered_test_db` can apply the `-k` keyword guard.

```
waive id (raw)
   ↓ normalize     strip SKIP/TIMEOUT/full:gpu/comments
   ↓ strip [params] if present
target_lookup     (function/class/file/dir level)
   ↓ try YAML at this level
       hit  → filter prefix = level (with originating waive ids)
       miss → strip one level up and retry
   ↓ all levels miss → rule emits scope=None
```

An entry matches a level when its canonical target (with `SKIP`/`TIMEOUT`/
`full:gpu`/`-k`/`-m`/`[params]` stripped) equals the level AND any `-k`
keyword filter on the entry contains an identifier from the waive id. `-m`
markers always pass (unverifiable from string).

The `-k` keyword guard runs twice: once at lookup, once when writing
`cbts_test_db/` (drops sibling entries whose `-k` doesn't match the waived
test).

## When CBTS activates

CBTS activates on bare `/bot run` and `/bot run --post-merge`. Any
stage-selection flag (`--stage-list`, `--extra-stage`, `--gpu-type`,
`--test-backend`, `--skip-test`, `--add-multi-gpu-test`, `--only-multi-gpu-test`,
`--disable-multi-gpu-test`) makes `getCbtsResult` return null.

Orthogonal flags (`--reuse-test`, `--disable-reuse-test`, `--debug`,
`--detailed-log`, `--disable-fail-fast`, `--high-priority`) do not affect CBTS.

## How it's invoked (CI)

`getCbtsResult` calls `main.py` twice on the L0_MergeRequest agent:

1. `main.py --list-needed-diffs` → file patterns whose diffs Groovy must fetch
   (Ant-style globs).
2. `main.py cbts_input.json` → decision JSON on stdout. When any block was
   narrowed, writes `${LLM_ROOT}/cbts_test_db/` with the affected YAMLs and
   only their affected blocks (kept entries preserve `TIMEOUT (n)`,
   `ISOLATION`, `-k`, `-m` verbatim).

Decision JSON:

```json
{
  "scope": "waiveonly",
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "reasons": ["[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages"],
  "test_db_dir_override": "cbts_test_db",
  "affected_stage_test_counts": {"A10-PyTorch-1": 5, "A10-PyTorch-2": 5}
}
```

- `scope: null` → no decision; Groovy defers to baseline.
- `test_db_dir_override: null` → no Layer 3 narrowing; trt-test-db reads
  the source test-db.
- `affected_stage_test_counts` → per-stage post-keep-filter test count for
  Layer 2.5 split-collapse.

## Cross-job seed for stage agents

`cbts_test_db/` is written on the L0_MergeRequest agent and is not
available to downstream `L0_Test-*` pods. To regenerate it per stage:

1. `getCbtsResult` stores the input JSON in `result.cbts_input_json`,
   which rides along inside `testFilter`.
2. `renderTestDB` on the stage agent writes it to a temp file and re-runs
   `main.py`. Output is deterministic, so each agent gets the same
   `cbts_test_db/` as L0_MergeRequest produced.

If `cbts_input_json` exceeds 256 KB the piggyback is dropped; Layer 3 falls
back to the source test-db on each stage agent. Layer 2 still applies.

## Split-collapse heuristic (Layer 2.5)

In `_cbtsMaybeCollapseSplits`, when the stage's narrowed count < 20:
- `splitId == 1` → run as splits=1 (single agent runs the full list).
- `splitId > 1` → early return; no agent allocated.

At/above 20, default splits stand. The count is computed by
`blocks.compute_stage_test_counts` using the same keep filter as
`write_filtered_test_db`.

## Adding a new rule

1. **Create `rules/my_rule.py`** subclassing `Rule`:

   ```python
   from typing import Optional
   from blocks import YAMLIndex, Stage
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
               scope="myscope",
               reason="why this fired",
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

2. Register in `main.py` (`RULE_CLASSES` and `build_rules()`).

3. No Groovy edits needed.

`Selector` unions `affected_stages` and `block_filters`; scopes are combined
via `_combine_scopes` (all-agree → that scope; otherwise None).

## Fallback paths

CBTS defers to the existing filter chain when:

- PostMerge job / `alternativeTRT` set
- `changed_files` is empty
- `main.py` throws or stdout is unparsable
- Python returns `scope: null`
- A waive id misses every level in `find_match_for_waive` — rule emits
  `scope: null`
- Layer 3 narrowing would empty a block — block keeps original tests
- `cbts_input_json` exceeds 256 KB — Layer 3 falls back per stage
- Narrowed YAML missing/empty on a stage agent — renderTestDB falls back

Every fallback emits an `echo` log line.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors Groovy
`getMakoArgsFromStageName` (~`L0_Test.groovy:2079`) and
`parseTaskConfigFromStageName` (~`:2066`). Update both when adding new
backends / orchestrators / stage-name conventions.
