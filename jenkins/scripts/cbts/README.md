# CBTS — Change-Based Testing Selection

Pre-merge CI test-selection tool. Looks at what the PR changed and narrows the
set of Jenkins stages that actually need to run.

---

## What it does — two layers

Given a PR, CBTS produces a decision that gets consumed at two points in the
Jenkins pipeline:

| Layer | Where consumed | Action |
|---|---|---|
| **1. Arch track** | `L0_MergeRequest.groovy::launchStages` (each track entry) | Skip whole x86 / SBSA track (build + all tests) when no stage on that arch is affected |
| **2. Stage** | `L0_Test.groovy::launchTestJobs` (end of filter chain) | Replace `parallelJobsFiltered` with the CBTS-selected subset |

Anything CBTS can't confidently narrow → **fallback to the existing full filter
chain**. CBTS never adds stages; it only subtracts.

**No within-stage test filtering by design.** Once Layer 2 picks the affected
stages, each stage runs its full rendered testDBList (all blocks matching the
stage's mako). Running the whole block, not just the single changed test id,
is deliberately over-inclusive — if a waive is wrong (depends on another test,
or the node id has a typo that silently matches nothing), running only the
changed test would not surface the problem. The extra per-stage test time is
accepted as the cost of CI robustness.

## v0 scope

- **Only handles** `tests/integration/test_lists/waives.txt` changes
- Any other changed file → CBTS returns `scope: none` → full run
- Scope label for this case: `waiveonly`

## File map

```
jenkins/scripts/cbts/
├── README.md              this file
├── main.py                CLI entry + Selector + SelectionResult
├── blocks.py              YAML loading + stage parsing from groovy + condition matching
└── rules/
    ├── README.md          per-rule logic summary (scope, triggers, matching)
    ├── base.py            Rule ABC + PRInputs + RuleResult
    └── waives_rule.py     v0's only rule
```

## How it's invoked (CI)

`L0_MergeRequest.groovy::getCbtsResult` orchestrates two calls to `main.py`:

1. `python3 main.py --list-needed-diffs` — returns `needs_diff_for` patterns
   so Groovy knows which changed files to fetch diffs for (via the existing
   `getMergeRequestOneFileChanges` API helper).
2. `python3 main.py cbts_input.json` — returns the decision on stdout.

The decision is cached in `testFilter[CBTS_RESULT]` and serialized into the
child job's `testFilter` param alongside the existing filter flags.

Python stdout is a JSON blob:

```json
{
  "scope": "waiveonly",
  "affected_cpu_arch": ["x86"],
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "tests": ["unittest/utils/test_util.py"],
  "reasons": ["[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages"]
}
```

`scope: null` means "no decision, fall back to the existing filter chain".

## Adding a new rule

1. **Create `rules/my_rule.py`** subclassing `Rule`:

   ```python
   from typing import Optional
   from blocks import YAMLIndex, Stage
   from .base import PRInputs, Rule, RuleResult

   class MyRule(Rule):
       name = "myrule"
       needs_diff_for = ["path/or/glob/**/*.py"]  # files whose diffs you need

       def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]):
           self.yaml_index = yaml_index
           self.stages = stages

       def apply(self, pr: PRInputs) -> Optional[RuleResult]:
           # Return None if the rule doesn't apply to this PR.
           # Return a RuleResult otherwise.
           ...
           return RuleResult(
               handled_files={...},      # files you claim
               tests={...},              # changed test ids (logged; not filtered at stage time)
               affected_stages={...},    # Layer 2 stage set
               scope="myscope",          # your scope label
               reason="why this was picked",
           )
   ```

2. **Register in `main.py`**:
   - Add the class to `RULE_CLASSES` (used by `--list-needed-diffs`).
   - Add an instance to `build_rules()` with its dependencies.

3. **Decide Layer 1 / 2 behavior for your scope in Groovy**. Each layer's
   consumer checks `cbts.scope == "waiveonly"` explicitly. For a new scope:
   - Add an `if (cbts.scope == "myscope")` branch in `L0_Test.groovy`
     Layer 2 override (or leave unspecified → behaves as fallback / full run).
   - Similarly decide Layer 1 (arch track skip).
   - If your scope genuinely needs within-stage test filtering, add that
     logic at `L0_Test.groovy:2674` (currently only a comment; `waiveonly`
     deliberately skips this, see the rationale in the first section).
   - **Defaults are conservative**: without explicit branches, new scope
     paths fall through to the existing filter chain, which is safe.

Rule ordering doesn't matter. Rules independently decide whether they apply;
`Selector` combines their `affected_stages` via union and their scopes via
`_combine_scopes` (agreement → that scope; disagreement → `None`).

## Fallback / safety paths

CBTS falls back to the existing filter chain (as if it weren't there) when:

- PostMerge job / `alternativeTRT` set
- `changed_files` is empty
- `main.py` throws / stdout is unparsable
- `scope == none` (Python's explicit "no decision" output)
- Groovy sees an unknown scope value (forward compatibility)

No silent failures: every fallback logs an `echo` line in the CI console.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors the Groovy
`getMakoArgsFromStageName` (in `jenkins/L0_Test.groovy` ~line 2079) and
`parseTaskConfigFromStageName` (~line 2066). When new backends /
orchestrators / stage-name conventions are added on the Groovy side, update
the Python constants here too. The file comments flag this explicitly.
