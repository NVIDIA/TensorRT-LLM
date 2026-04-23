# CBTS — Change-Based Testing Selection

Pre-merge CI test-selection tool. Looks at what the PR changed and narrows the
set of Jenkins stages + tests that actually need to run.

For conceptual design / review notes, see [DESIGN.md](./DESIGN.md). This
README is the operational reference.

---

## What it does — three layers

Given a PR, CBTS produces a decision that gets consumed at three points in the
Jenkins pipeline:

| Layer | Where consumed | Action |
|---|---|---|
| **1. Arch track** | `L0_MergeRequest.groovy::launchStages` (each track entry) | Skip whole x86 / SBSA track (build + all tests) when no stage on that arch is affected |
| **2. Stage** | `L0_Test.groovy::launchTestJobs` (end of filter chain) | Replace `parallelJobsFiltered` with the CBTS-selected subset |
| **3. Test** | `L0_Test.groovy::runLLMTestlistOnPlatformImpl` (after `renderTestDB`) | Intersect rendered `testDBList` with CBTS's `affected_tests` |

Anything CBTS can't confidently narrow → **fallback to the existing full filter
chain**. CBTS never adds stages; it only subtracts.

## v0 scope

- **Only handles** `tests/integration/test_lists/waives.txt` changes
- Any other changed file → CBTS returns `scope: none` → full run
- Scope label for this case: `waiveonly`

## File map

```
jenkins/scripts/cbts/
├── DESIGN.md              design doc (for review)
├── README.md              this file
├── main.py                CLI entry + Selector + SelectionResult
├── blocks.py              YAML loading + stage parsing from groovy + condition matching
└── rules/
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

## Debugging a CBTS decision locally

The input JSON that Groovy sends to Python is uploaded as a CI artifact
(`cbts_input.json` in the pipeline workspace). To reproduce a decision
locally:

```bash
# From the repo root:
python3 jenkins/scripts/cbts/main.py cbts_input.json
```

Or hand-craft a minimal input:

```bash
cat > /tmp/cbts_input.json <<EOF
{
  "changed_files": ["tests/integration/test_lists/waives.txt"],
  "diffs": {
    "tests/integration/test_lists/waives.txt": "@@ -5,6 +5,7 @@\n+unittest/utils/test_util.py SKIP (https://nvbugs/1)\n"
  }
}
EOF

python3 jenkins/scripts/cbts/main.py /tmp/cbts_input.json
```

Output is a JSON blob on stdout (see DESIGN.md §4.6):

```json
{
  "scope": "waiveonly",
  "affected_cpu_arch": ["x86"],
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "tests": ["unittest/utils/test_util.py"],
  "reasons": ["[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages"]
}
```

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
               tests={...},              # Layer 3 test filter
               affected_stages={...},    # Layer 2 stage set
               scope="myscope",          # your scope label
               reason="why this was picked",
           )
   ```

2. **Register in `main.py`**:
   - Add the class to `RULE_CLASSES` (used by `--list-needed-diffs`)
   - Add an instance to `build_rules()` with its dependencies

3. **Decide Layer 1 / 2 / 3 behavior for your scope in Groovy**. Each layer's
   consumer checks `cbts.scope == "waiveonly"` explicitly. For a new scope:
   - Add an `else if (cbts.scope == "myscope")` branch in `L0_Test.groovy`
     Layer 2 override (or leave unspecified → behaves as fallback / full run)
   - Similarly decide Layer 1 (arch track skip) and Layer 3 (test filter)
   - **Defaults are conservative**: without explicit branches, new scope
     paths fall through to the existing filter chain, which is safe

4. **Keep unit-style checks via CLI smoke tests**. See
   `--list-needed-diffs` should include your new pattern; a targeted
   `INPUT_JSON` with a change under your rule's scope should produce the
   expected output.

Rule ordering doesn't matter. Rules independently decide whether they apply;
`Selector` combines their `affected_stages` / `tests` via union and their
scopes via `_combine_scopes` (agreement → that scope; disagreement → `None`).

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
