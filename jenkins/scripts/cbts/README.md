# CBTS — Change-Based Testing Selection

Pre-merge CI test-selection tool. Looks at what the PR changed and narrows the
set of Jenkins stages — and the tests inside each stage — that actually need
to run. **Adding new rules is Python-only — Groovy is scope-agnostic and
consumes the data directly.**

---

## Three consumption layers

| Layer | Where | Action |
|---|---|---|
| **1. Arch track** | `L0_MergeRequest.groovy::launchStages` | Skip x86 / SBSA track when no stage on that arch is affected |
| **2. Stage** | `L0_Test.groovy::launchTestJobs` (end of filter chain) | Replace `parallelJobsFiltered` with the CBTS-selected subset |
| **3. Within-stage tests** | `L0_Test.groovy::renderTestDB` | Point trt-test-db at the CBTS-narrowed tmp test-db (each affected block's `tests:` array filtered to the per-block filter prefix subtree) |

CBTS only **subtracts** stages and tests, never adds. Anything it can't
narrow → full fallback to the existing filter chain.

## v0 scope

- **Only handles** `tests/integration/test_lists/waives.txt` changes (`scope: waiveonly`).
- Anything else → `scope: none` → full run.
- **v1+ rules can be added in Python alone** (no Groovy edits).

## File map

```
jenkins/scripts/cbts/
├── README.md              this file
├── main.py                CLI entry + Selector + SelectionResult
├── blocks.py              YAML index + lookup + filtered tmp test-db generation
└── rules/
    ├── README.md          per-rule logic summary
    ├── base.py            Rule ABC + PRInputs + RuleResult
    └── waives_rule.py     v0's only rule
```

## Lookup algorithm: parent chain with first-match wins

Per waive id, `YAMLIndex.find_match_for_waive` walks the pytest tree from the
waive towards the root. The first level whose YAML has a matching entry wins;
that level becomes the **filter prefix** the block uses for Layer 3.

```
waive id (raw)
   ↓ normalize     strip SKIP/TIMEOUT/full:gpu/comments
   ↓ strip [params] if present
target_lookup     (function-level when waive was parametrized; otherwise
                   class/file/dir level — waive's own granularity)
   ↓ try YAML at this level
       hit  → matched: filter prefix = level
       miss → strip one level up (::method → ::class → /file → /dir → ...)
              and retry
   ↓ all levels miss → fallback: rule emits scope=None, baseline runs
```

An entry "matches" at a level when its **canonical target** (entry with
`SKIP`/`TIMEOUT`/`full:gpu`, pytest options `-k "..."` / `-m "..."`, and
`[params]` all stripped) equals the level **and** any `-k` keyword filter the
entry carries actually contains an identifier present in the waive id.
`-m` markers are unverifiable from a string and always pass (over-include
when in doubt).

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
2. `main.py cbts_input.json` → decision JSON on stdout. If any block was
   narrowed, also writes `${LLM_ROOT}/cbts_test_db/` containing only the
   affected YAMLs with their filtered `tests:` arrays. Each kept entry
   preserves `TIMEOUT (n)`, `ISOLATION`, `-k "..."`, `-m "..."` verbatim
   (YAML-level `# comments` are dropped by PyYAML round-trip but no
   functional info is lost).

Decision JSON:

```json
{
  "scope": "waiveonly",
  "affected_cpu_arch": ["x86"],
  "affected_stages": ["A10-PyTorch-1", "A10-PyTorch-2"],
  "tests": ["unittest/utils/test_util.py"],
  "reasons": ["[waives] waives.txt: +1 / -0 → 1 blocks, 2 stages"],
  "test_db_dir_override": "cbts_test_db"
}
```

- `scope: null` → no decision, full fallback. Groovy doesn't gate on the
  scope value — it's metadata for logs and multi-rule combining only.
- `test_db_dir_override: null` → no Layer 3 narrowing; trt-test-db reads
  the source `tests/integration/test_lists/test-db/` as before.

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
               # Optional Layer 3 contribution: per-block filter prefixes.
               # Selector unions across rules and writes the tmp test-db.
               block_filters={(yaml_stem, block_index): {filter_prefix}, ...},
           )
   ```

2. **Register in `main.py`**: add to `RULE_CLASSES` and `build_rules()`.

3. **No Groovy edits needed.** Layer 1 / 2 / 3 are scope-agnostic and consume
   `affected_cpu_arch` / `affected_stages` / `block_filters` directly.

Rule order is irrelevant. `Selector` unions `affected_stages` and
`block_filters`; scopes are combined via `_combine_scopes` (all-agree → that
scope; disagreement → `None`).

## Fallback paths

CBTS falls back to the existing filter chain when:

- PostMerge job / `alternativeTRT` set
- `changed_files` is empty
- `main.py` throws / stdout is unparsable
- Python returns `scope: null` ("no decision")
- A waive id misses every level up to the root in `find_match_for_waive`
  (likely typo'd or out-of-tree id) — the rule emits `scope: null`
- `affected_stages` is empty (Layer 2 no-op)
- Layer 3 filter would empty a block's `tests:` array — that block keeps
  its original tests instead (per-block safety net)

Every fallback logs an `echo` line — no silent failures.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors the Groovy
`getMakoArgsFromStageName` (`L0_Test.groovy` ~line 2079) and
`parseTaskConfigFromStageName` (~line 2066). New backends / orchestrators /
stage-name conventions on the Groovy side need a matching Python update —
file comments flag this.
