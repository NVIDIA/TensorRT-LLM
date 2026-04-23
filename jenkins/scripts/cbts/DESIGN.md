# CBTS — Change-Based Testing Selection for CI

**Status**: Draft for infra + dev review
**v0 Scope**: When a PR only changes `tests/integration/test_lists/waives.txt`, decide which stages + tests to run at block granularity.

---

## 1. Core Idea

PR changes one test in waives.txt → that test lives in some YAML block → the block's `condition` matches only certain stages → **other stages aren't scheduled**; build + test sub-jobs for the arch track (x86 / SBSA) that has no affected stages are **entirely skipped**; inside each scheduled stage, the test list is further filtered by test id.

**Three-layer filtering**:
- **Layer 1 (arch track level)**: affected stages only on x86 → skip the entire SBSA track (SBSA build + all SBSA sub-jobs), and vice versa.
- **Layer 2 (stage level)**: within the same arch, match `block.condition` against `stage.mako` to pick which stages to schedule.
- **Layer 3 (test level)**: inside a running stage, intersect `renderTestDB`'s output with CBTS's test set.

No changes to pytest-split, trt-test-db, or the stage-scheduling core; all new code is glue.

---

## 2. v0 Scope

- **Covers**: PRs that only change waives.txt.
- **Other changes**: any other file → fall through to the existing filter chain (equivalent to full run).
- **Semantics**: add/remove/edit a waive → run the corresponding test and the stages matching its containing block.
- **scope label**: v0 defines one value — `waiveonly` (PR only changed waive-related files). Future rules introduce new scope values (e.g. `testonly`, `modelarch`) when needed; we do not invent an abstract taxonomy upfront.

---

## 3. Code Architecture

### 3.1 File layout

```
jenkins/scripts/cbts/
├── DESIGN.md
├── README.md
├── main.py                    ← Selector + SelectionResult + CLI
├── blocks.py                  ← Stage + Block + YAMLIndex + block_matches_stage
└── rules/
    ├── base.py                ← Rule ABC + PRInputs + RuleResult
    └── waives_rule.py         ← v0 rule
```

**4 Python files** + 2 docs. No `__init__.py` files — directories are used as Python 3.3+ namespace packages, matching the `jenkins/scripts/` convention (sibling directories like `jenkins/scripts/perf/` are structured the same way).

### 3.2 Key contracts

```python
# rules/base.py
@dataclass
class PRInputs:
    changed_files: list[str]
    diffs: dict[str, str]       # Groovy pre-fetches based on needs_diff_for

@dataclass
class RuleResult:
    handled_files: set[str]
    tests: set[str]             # Layer 3: within-stage filter
    affected_stages: set[str]   # Layer 2: stages to schedule
    scope: str                  # rule-declared scope label, v0 only has "waiveonly"
    reason: str

class Rule(ABC):
    name: str
    needs_diff_for: list[str] = []
    @abstractmethod
    def apply(self, pr: PRInputs) -> Optional[RuleResult]: ...
```

Note: `affected_cpu_arch` is not a RuleResult field; the Selector derives it by looking up each affected stage's `cpu_arch` in the stage map. Rules don't set it.

```python
# blocks.py
@dataclass
class Stage:
    name: str
    yaml_stem: str
    cpu_arch: str               # "x86" / "sbsa", inferred from x86TestConfigs vs SBSATestConfigs
    split_id: int
    total_splits: int
    mako: dict[str, str]        # derived from stage name (mirrors getMakoArgsFromStageName)

@dataclass
class Block:
    yaml_stem: str
    block_index: int
    condition: dict             # raw: {ranges, wildcards, terms}
    tests: list[str]

def block_matches_stage(block, stage) -> bool:
    """Generic over YAML field names: adding a new term does not require changing this."""
```

### 3.3 Why no `stages.py`

`Stage` values are derived purely from what Python can parse out of `jenkins/L0_Test.groovy` (stage map entries + stage-name patterns). Keeping `Stage` together with `Block` in `blocks.py` avoids a module for two small dataclasses. `derive_mako_from_stage` mirrors `getMakoArgsFromStageName` on the Groovy side — single source of truth is still the Groovy file; Python only reads it.

---

## 4. Jenkins Integration

### 4.1 Injection point: alongside other testFilter setters

In `L0_MergeRequest.groovy`, find this snippet (excerpt):

```groovy
testFilter[(MULTI_GPU_FILE_CHANGED)]  = getMultiGpuFileChanged(pipeline, testFilter, globalVars)
testFilter[(ONLY_ONE_GROUP_CHANGED)]  = getOnlyOneGroupChanged(pipeline, testFilter, globalVars)
testFilter[(AUTO_TRIGGER_TAG_LIST)]   = getAutoTriggerTagList(pipeline, testFilter, globalVars)
// NEW
testFilter[(CBTS_RESULT)]             = getCbtsResult(pipeline, testFilter, globalVars)
```

`getCbtsResult` returns either `null` (no decision → full run) or `{scope, affected_cpu_arch, affected_stages, affected_tests, reasons}`. **The decision is made once and cached in `testFilter`**; the three downstream layers only read.

### 4.2 Layer 1 — Arch-track skip

Injection point: the `x86_64-Linux` / `SBSA-Linux` track entries in `L0_MergeRequest.groovy::launchStages()` (there is already precedent there: `if (testFilter[(ONLY_ONE_GROUP_CHANGED)] == "Docs") return` skips SBSA for docs-only PRs).

```groovy
"x86_64-Linux": {
    script {
        def cbts = testFilter[(CBTS_RESULT)]
        if (cbts?.scope == "waiveonly" && !("x86" in cbts.affected_cpu_arch)) {
            echo "CBTS waiveonly: no x86 stages affected, skipping x86_64-Linux track"
            return
        }
        // existing Build-x86_64 + Test-x86_64-* logic unchanged
        ...
    }
},
"SBSA-Linux": {
    script {
        if (testFilter[(ONLY_ONE_GROUP_CHANGED)] == "Docs") { return }   // existing
        // NEW: CBTS waiveonly equivalent skip
        def cbts = testFilter[(CBTS_RESULT)]
        if (cbts?.scope == "waiveonly" && !("sbsa" in cbts.affected_cpu_arch)) {
            echo "CBTS waiveonly: no sbsa stages affected, skipping SBSA-Linux track"
            return
        }
        // existing Build-SBSA + Test-SBSA-* logic unchanged
        ...
    }
},
```

**Key points**:
- The condition explicitly matches `scope == "waiveonly"`. Future scopes (`testonly` / `modelarch`) do not trigger track skips by default until their safety is evaluated and an explicit `else if` branch is added.
- Effect: an x86-only waive change → the entire SBSA track disappears from Blue Ocean (including build); and vice versa.

### 4.3 Layer 2 — Stage-scheduling override

Injection point: the filter chain in `L0_Test.groovy` (around lines 3714–3800). **CBTS acts as a short-circuit override appended to the end of the existing chain**; the existing logic is untouched.

```groovy
// Existing filter chain untouched: MULTI_GPU_FILE_CHANGED / AUTO_TRIGGER_TAG_LIST /
// IS_POST_MERGE / ENABLE_SKIP_TEST / GPU_TYPE_LIST / TEST_BACKEND / ...
// Produces parallelJobsFiltered.
...

// NEW: CBTS short-circuit override at the tail
def cbts = testFilter[(CBTS_RESULT)]
if (cbts?.scope == "waiveonly") {
    parallelJobsFiltered = parallelJobs.findAll { key, _ -> key in cbts.affected_stages }
    echo "CBTS waiveonly: limiting to ${cbts.affected_stages.size()} affected stages"
}
```

**Key points**:
- **One `if`, no `else`**: concise, no nested branches.
- **Override semantics**: the existing chain produces `parallelJobsFiltered` first; when waiveonly matches, it is replaced wholesale.
- **All fallback cases naturally don't override**: `cbts == null` / `scope == null` / unknown scope / call failure → condition false → the existing chain's result is preserved.
- **Adding a new scope**: add another parallel `if (cbts?.scope == "testonly") { ... }`; scopes don't interfere.
- **Cost**: on a waiveonly PR the existing filter chain runs once and is then overwritten (pure Groovy set ops; no IO; negligible).

### 4.4 Layer 3 — Within-stage test filter

In the block starting at `L0_Test.groovy:2674`, the CBTS filter is inserted **right before `processShardTestList`**, after all prep (`mergeWaivesTxt` / `reusePassedTestResults`) has completed:

```groovy
def testDBList = renderTestDB(testList, llmSrc, stageName)
mergeWaivesTxt(pipeline, llmSrc, stageName)           // existing: download merged waives.txt
// reusePassedTestResults(...)                         // existing: append previously-passed tests to waives

// NEW: CBTS Layer 3 filter, single-point insertion
def cbts = testFilter[(CBTS_RESULT)]
if (cbts?.scope == "waiveonly") {
    testDBList = filterTestDBList(testDBList, cbts.affected_tests)
}

def preprocessedLists = processShardTestList(llmSrc, testDBList, splitId, splits, perfMode)
```

Same explicit match on `waiveonly`; future scopes must decide independently whether to filter tests at this layer by adding an `else if` branch.

#### Interaction with `mergeWaivesTxt`: verified consistent

The merged waives.txt downloaded by `mergeWaivesTxt` is produced by `jenkins/scripts/mergeWaiveList.py` with the following algorithm:

```
merged = dedupe(PR's waives.txt ∪ TOT's waives.txt) - lines in PR's diff prefixed with `-`
```

This algorithm is **PR-aware**: PR additions are preserved via the union; PR removals are applied via subtraction. So the merged result **fully preserves the PR's intent on waives.txt**.

Three scenarios, all verified:

| Scenario | merged content | runtime | CBTS decision correctness |
|---|---|---|---|
| PR removes a waive (`-test_X`) | no test_X | test_X runs | ✓ stage runs, test is actually verified |
| PR adds a waive (`+test_X`) | contains test_X | test_X skipped | ✓ stage runs, pytest collects the test and skips correctly (verifies the waive mechanism) |
| PR edits a waive line (NVBug link) | new line only | test still skipped | ✓ matches PR intent |

**Conclusion**: CBTS decisions based on the PR diff are naturally aligned with the runtime waive state; **no extra handling for `mergeWaivesTxt` is needed**.

### 4.5 What `getCbtsResult` does

1. PostMerge / alternativeTRT → return null.
2. `getMergeRequestChangedFileList` → `changed_files`, `.unique()`.
3. Ask Python `--list-needed-diffs` for patterns; for each changed file matching a pattern, call `getMergeRequestOneFileChanges` to pull its diff.
4. Write `cbts_input.json` (just `changed_files` + `diffs`), then run `python3 main.py cbts_input.json`.
5. Parse stdout, return a structured result.

Note: the Python side **parses `jenkins/L0_Test.groovy` itself** to derive stage configs (reusing the regex approach already in `scripts/test_to_stage_mapping.py`) and loads YAMLs from `tests/integration/test_lists/test-db/`. Groovy does not need to pack `stage_map` into INPUT_JSON. This keeps the Groovy code surface minimal.

Each stage's `cpu_arch` is inferred by tracking which map-literal (`x86TestConfigs` / `SBSATestConfigs` / ...) the entry lives in. Each stage's `mako` is derived by a Python port of `getMakoArgsFromStageName` (line ~2079) and `parseTaskConfigFromStageName` (line ~2066). Those Python helpers live in `blocks.py` with explicit "keep in sync with groovy source" comments at the top.

### 4.6 Python ↔ Groovy IO contract

Groovy `getCbtsResult` calls Python twice:

1. **Get needs_diff_for patterns** (no args): Python prints the union of all rules' `needs_diff_for` patterns; Groovy uses this to decide which changed files to fetch diffs for.
2. **Make the decision** (JSON file arg): Groovy packs `changed_files` / `diffs` into INPUT_JSON; Python writes the decision to stdout.

**INPUT_JSON** is produced by Groovy `getCbtsResult`, containing only PR data:

```json
{
  "changed_files": ["tests/integration/test_lists/waives.txt", ...],
  "diffs": {
    "tests/integration/test_lists/waives.txt": "@@ -1,3 +1,4 @@\n..."
  }
}
```

Stage configs are parsed by Python directly from `jenkins/L0_Test.groovy` (see 4.5).

**stdout is a JSON blob**:

```json
{
  "scope": "waiveonly",
  "affected_cpu_arch": ["x86"],
  "affected_stages": [
    "DGX_H100-4_GPUs-PyTorch-DeepSeek-1",
    "DGX_H100-4_GPUs-PyTorch-DeepSeek-2"
  ],
  "tests": ["examples/test_deepseek.py::test_xxx"],
  "reasons": ["[waives] waives.txt: +2 / -1 → 2 blocks, 2 stages"]
}
```

When there is no decision, `scope` is `null` (Groovy parses this as "fall back"):

```json
{
  "scope": null,
  "affected_cpu_arch": [],
  "affected_stages": [],
  "tests": [],
  "reasons": ["Unhandled files: [tensorrt_llm/llmapi/llm.py, ...]"]
}
```

Groovy consumes it via `JsonSlurper` in `_cbtsParseSelectionResult`; `scope == null` maps to `testFilter[CBTS_RESULT] = null`.

Exit code 0 = decision succeeded (including the null-scope case); non-zero → Groovy falls back to full run.

### 4.7 Multi-rule combination (future)

v0 has only one rule, so no combining. But `Selector` reserves a `combine_scopes(scopes)` helper:

```python
def combine_scopes(scopes: list[str]) -> str | None:
    # v0: single rule returns "waiveonly" — passthrough.
    if len(set(scopes)) == 1:
        return scopes[0]
    # Multiple differing scopes → conservative None (full run).
    # Fill in a priority table here when priority needs emerge.
    return None
```

---

## 5. Fallback & Safety

Any of the following → **fall through to the existing filter chain** (`testFilter[CBTS_RESULT] = null` or `.scope == null`):

- PostMerge / alternativeTRT job.
- `changed_files` is empty.
- Python call fails / stdout unparsable.
- Python explicitly returns `scope: none` (unhandled files, no rule matched, multi-rule scope conflict).
- Groovy sees an unknown scope (Python upgraded before Groovy caught up).

---

## 6. Extensions (future, not in v0)

| New Rule | New modules | `needs_diff_for` | Likely scope |
|---|---|---|---|
| `test_block_rule` (test-file changes) | none; reuses `blocks.py` | `[]` | `testonly` |
| `case_level_rule` (function-level precision) | `code_analysis/ast_utils.py` + `test_extractor.py` | `["tests/integration/defs/**/*.py"]` | `testonly` |
| `model_arch_rule` (model-arch matching) | `model_arch.py` | `[]` | `modelarch` |

**What adding a rule requires**:
1. Define a new scope value (e.g. `testonly`) and add an `else if` branch at Groovy Layer 1 specifying whether aux stages are skipped for this scope.
2. Write a Rule class on the Python side.
3. If extra PR data is needed, declare it in `needs_diff_for`.

**What adding a rule does NOT change**: the Rule ABC, the `combine_scopes` skeleton, `blocks.py`, the CLI contract, the overall shape of `getCbtsResult`, or the Layer-2 consumption site.

---

## 7. Review Highlights

### For infra
- **Zero changes to pytest-split / trt-test-db / stage rendering.**
- **Reuses existing helpers**: `getMergeRequestChangedFileList`, `getMergeRequestOneFileChanges`, `getMakoArgsFromStageName`, stageList.
- **Injections mirror existing patterns**:
  - Setter: the fourth setter on `testFilter`.
  - Layer 1 consumer: one skip check at each `launchStages` track entry (isomorphic to the existing Docs-only skip).
  - Layer 2 consumer: appended `if` in the `L0_Test.groovy` filter chain tail.
  - Layer 3 consumer: three-line injection at `L0_Test.groovy:2674`.
- **Any failure falls back to full run.**

### For dev
- **v0 behavior is conservative**: only narrows PRs that touch only waives.txt.
- **CI log has a reason line**: you can see why particular stages / tests were selected.
- **Local reproducibility**: pull the `cbts_input.json` CI artifact and run `python3 main.py cbts_input.json` to reproduce the decision.
- **Low bar to add a rule**: write a class with an `apply()` method.

---

## 8. Open Questions

1. **Does infra accept skipping docBuild/sanityCheck under `waiveonly`?** waives.txt is a runtime plain-text list that doesn't flow into the wheel or docs, so skipping should be safe; if the team has a hard rule of "every PR must pass doc build / wheel sanity", the fallback is to keep those two aux stages even under `waiveonly`.
2. **`filterTestDBList` in Groovy or Python?** Leaning Python (more testable).
3. **Self-check mechanism**: Python's `block_matches_stage` could drift from trt-test-db semantics. Deferred to a future revision.
