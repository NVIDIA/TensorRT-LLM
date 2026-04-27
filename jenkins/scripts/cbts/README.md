# CBTS — Change-Based Testing Selection

Pre-merge CI test-selection tool. Looks at what the PR changed and narrows the
set of Jenkins stages that actually need to run. **Adding new rules is
Python-only — Layer 1/2 in Groovy are scope-agnostic and consume the data
directly.**

---

## Two consumption layers

| Layer | Where | Action |
|---|---|---|
| **1. Arch track** | `L0_MergeRequest.groovy::launchStages` | Skip x86 / SBSA track when no stage on that arch is affected |
| **2. Stage** | `L0_Test.groovy::launchTestJobs` (end of filter chain) | Replace `parallelJobsFiltered` with the CBTS-selected subset |

CBTS only **subtracts** stages, never adds. Anything it can't narrow → full
fallback to the existing filter chain.

**No within-stage test filtering by design.** Each picked stage runs its full
testDBList. Filtering down to just the changed test would mask wrong waives
(broken deps, typo'd node ids silently matching nothing) — the extra per-stage
time buys robustness.

## v0 scope

- **Only handles** `tests/integration/test_lists/waives.txt` changes (`scope: waiveonly`).
- Anything else → `scope: none` → full run.
- **v1+ rules can be added in Python alone** (no Groovy edits).

## File map

```
jenkins/scripts/cbts/
├── README.md              this file
├── main.py                CLI entry + Selector + SelectionResult
├── blocks.py              YAML loading + stage parsing + test-id normalization
└── rules/
    ├── README.md          per-rule logic summary
    ├── base.py            Rule ABC + PRInputs + RuleResult
    └── waives_rule.py     v0's only rule
```

## When CBTS activates

CBTS narrows test selection in **two usages only**:

- `/bot run` — full pre-merge with CBTS narrowing.
- `/bot run --post-merge` — post-merge with CBTS narrowing. Layer 2 keeps
  only post-merge hits; no post-merge hit → no-op (no fallback to full
  post-merge baseline).

Any other **stage-selection** flag makes `getCbtsResult` return `null` and
the existing filter chain takes over: `--stage-list`, `--extra-stage`,
`--gpu-type`, `--test-backend`, `--skip-test`, `--add-multi-gpu-test`,
`--only-multi-gpu-test`, `--disable-multi-gpu-test`.

**Orthogonal** flags don't change stage selection and don't affect CBTS:
`--reuse-test`, `--disable-reuse-test`, `--debug`, `--detailed-log`,
`--disable-fail-fast`, `--high-priority`.

## How it's invoked (CI)

`getCbtsResult` calls `main.py` twice:

1. `main.py --list-needed-diffs` → patterns whose diffs Groovy fetches.
   Patterns are **Ant-style globs** (`tests/**/*.py`, `cpp/kernels/**`, exact
   paths), matched via `hudson.util.AntPathMatcher`.
2. `main.py cbts_input.json` → decision JSON on stdout:

```json
{
  "scope": "waiveonly",
  "affected_cpu_arch": ["x86"],
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "tests": ["unittest/utils/test_util.py"],
  "reasons": ["[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages"]
}
```

`scope: null` → no decision, full fallback. Groovy doesn't gate on the scope
value — it's metadata for logs and multi-rule combining only.

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
               tests={...},
               affected_stages={...},
               scope="myscope",
               reason="why this fired",
           )
   ```

2. **Register in `main.py`**: add to `RULE_CLASSES` and `build_rules()`.

3. **No Groovy edits needed.** Exception: if your rule needs to drop
   *individual tests inside* a stage (vs whole stages), add the hook at
   `L0_Test.groovy:2674` — but `waiveonly` deliberately skips this, see
   the "no within-stage filtering" note above.

Rule order is irrelevant. `Selector` unions `affected_stages`; scopes are
combined via `_combine_scopes` (all-agree → that scope; disagreement → `None`).

## Fallback paths

CBTS falls back to the existing filter chain when:

- PostMerge job / `alternativeTRT` set
- `changed_files` is empty
- `main.py` throws / stdout is unparsable
- Python returns `scope: null` ("no decision")
- `affected_stages` is empty (Layer 2 no-op)

Every fallback logs an `echo` line — no silent failures.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors the Groovy
`getMakoArgsFromStageName` (`L0_Test.groovy` ~line 2079) and
`parseTaskConfigFromStageName` (~line 2066). New backends / orchestrators /
stage-name conventions on the Groovy side need a matching Python update —
file comments flag this.
