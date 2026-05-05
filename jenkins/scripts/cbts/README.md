# CBTS — Change-Based Testing Selection

Pre-merge CI test-selection tool. Looks at what the PR changed and narrows the
set of Jenkins stages — and the tests inside each stage — that actually need
to run. **Adding new rules is Python-only — Groovy is scope-agnostic and
consumes the data directly.**

---

## Consumption layers

CBTS narrows **test cases only**. Build always runs (`L0_MergeRequest.groovy`
arch-track skipping was removed deliberately) so the wheel exists for sanity
checks and post-merge consumers.

| Layer | Where | Action |
|---|---|---|
| **2. Stage** | `L0_Test.groovy::launchTestJobs` (end of filter chain) | Replace `parallelJobsFiltered` with the CBTS-selected subset. Perf stages are excluded (they have their own trigger model and need full lists); `*-PackageSanityCheck-*` stages are force-kept (their names are runtime-built and invisible to the CBTS Python parser, and they are wheel/image gates that should always run after Build). |
| **2.5. Split-collapse** | `L0_Test.groovy::runLLMTestlistOnSlurm` and `runLLMTestlistOnPlatform` entries | When the affected stage's narrowed test count is < 20, collapse pytest-split's splits to 1 — only group 1 runs everything; groups 2..N skip without allocating a machine. At/above 20 the stage's default splits stand and pytest-split parallelizes normally. |
| **3. Within-stage tests** | `L0_Test.groovy::renderTestDB` | Point trt-test-db at the CBTS-narrowed tmp test-db. Each affected block's `tests:` array is filtered to entries in the per-block filter prefix subtree, **and unaffected blocks are dropped entirely** so a `/bot run --post-merge` can't accidentally activate post-merge blocks the PR never touched. |

(Layer 1 in earlier revisions skipped the entire arch track / Build when no
stage on that arch was affected. Removed: see
`L0_MergeRequest.groovy::launchStages`. CBTS no longer touches Build.)

### Trigger-mode mismatch

If `/bot run` is issued but every CBTS-resolved stage is post-merge (or the
symmetric case with `--post-merge`), Python's trigger-mode filter empties
`affected_stages` while leaving `scope != null`. The Selector safety-net
guarantees `affected_stages` is never empty BEFORE the filter when
`scope != null`, so Layer 2 detects the mismatch unambiguously with
`affectedSet.isEmpty()` and narrows to the PackageSanityCheck stages only
— equivalent to a build-and-sanity-only run, in spirit similar to
`/bot run --stage-list ""`.

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
├── blocks.py              YAML index + lookup + filtered tmp test-db generation + per-stage count
└── rules/
    ├── README.md          per-rule logic summary
    ├── base.py            Rule ABC + PRInputs + RuleResult
    └── waives_rule.py     v0's only rule
```

## Lookup algorithm: parent chain with first-match wins

Per waive id, `YAMLIndex.find_match_for_waive` walks the pytest tree from the
waive towards the root. The first level whose YAML has a matching entry wins;
that level becomes the **filter prefix** the block uses for Layer 3. Each
prefix remembers the originating waive id(s) so `write_filtered_test_db` can
re-apply the `-k` keyword guard when narrowing.

```
waive id (raw)
   ↓ normalize     strip SKIP/TIMEOUT/full:gpu/comments
   ↓ strip [params] if present
target_lookup     (function-level when waive was parametrized; otherwise
                   class/file/dir level — waive's own granularity)
   ↓ try YAML at this level
       hit  → matched: filter prefix = level (recorded with the originating waive id)
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

The `-k` keyword guard is applied **twice** by design:

1. **At lookup** (`find_match_for_waive`) — to decide whether the entry
   contributes to a block's filter prefix.
2. **At write** (`write_filtered_test_db`) — to drop sibling `-k "..."`
   entries that survive prefix-subtree match but whose keyword can't pick
   up the waived test (e.g., waive `func[CUTLASS-fp8-tp4]` keeps
   `-k "CUTLASS"` but not `-k "TRTLLM"`).

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

`getCbtsResult` calls `main.py` twice on the L0_MergeRequest agent:

1. `main.py --list-needed-diffs` → patterns whose diffs Groovy fetches.
   Patterns are **Ant-style globs** (`tests/**/*.py`, `cpp/kernels/**`, exact
   paths), matched via `hudson.util.AntPathMatcher`.
2. `main.py cbts_input.json` → decision JSON on stdout. If any block was
   narrowed, also writes `${LLM_ROOT}/cbts_test_db/` containing only the
   affected YAMLs with only their affected blocks (others dropped). Each
   kept entry preserves `TIMEOUT (n)`, `ISOLATION`, `-k "..."`, `-m "..."`
   verbatim (YAML-level `# comments` are dropped by PyYAML round-trip but
   no functional info is lost).

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

- `scope: null` → no decision, full fallback. Groovy doesn't gate on the
  scope value — it's metadata for logs and multi-rule combining only.
- `test_db_dir_override: null` → no Layer 3 narrowing; trt-test-db reads
  the source `tests/integration/test_lists/test-db/` as before.
- `affected_stage_test_counts` → per-stage post-keep-filter test count.
  Drives Layer 2.5 split-collapse below.

## Cross-job seed for stage agents

The `cbts_test_db/` written above lives on the L0_MergeRequest pipeline pod
and never reaches downstream `L0_Test-*` jobs (separate Kubernetes pods /
SLURM nodes). To make the narrowed test-db available to each stage agent
without a cross-job stash:

1. `getCbtsResult` puts the **input JSON itself** (`changed_files` + diffs)
   into `result.cbts_input_json`, which rides along inside `testFilter` as
   a normal build parameter.
2. `renderTestDB` on the stage agent receives it, writes a temp
   `cbts_input.json` (via `Utils.createTempLocation` → JNLP-writable
   path), and re-runs `python3 jenkins/scripts/cbts/main.py <temp>` so
   the narrowed `cbts_test_db/` materializes locally alongside the
   source. main.py is deterministic, so each agent ends up with a
   byte-identical copy of what L0_MergeRequest produced.
3. trt-test-db then queries `cbts_test_db/` as usual.

**Size cap.** `cbts_input_json` is dropped from the piggyback when its
size exceeds 256 KB (well below ARG_MAX). Layer 2 stage filtering still
applies, but Layer 3 narrowing on each stage agent silently degrades to
"no override" and `renderTestDB` falls back to the source test-db.

## Split-collapse heuristic (Layer 2.5)

When the affected stage's narrowed test count is below the hard-coded
threshold of **20** (in `_cbtsMaybeCollapseSplits`):

- `splitId == 1` → keep, override `splits = 1` so this single agent runs
  the full narrowed list.
- `splitId > 1` → early `return`; no agent allocated.

At/above the threshold, the stage's default splits stand and pytest-split
parallelizes normally.

The per-stage count is computed by `blocks.compute_stage_test_counts`,
which sums kept entries across blocks the stage's mako matches. The same
keep filter as `write_filtered_test_db` is applied so the count matches
what trt-test-db will eventually render.

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
               # Optional Layer 3 contribution: per-block prefix → set of
               # waive ids that resolved to it. Selector unions across
               # rules; write_filtered_test_db uses both the prefix
               # (subtree match) AND the waive ids (-k keyword guard).
               block_filters={
                   (yaml_stem, block_index): {
                       filter_prefix: {originating_waive_id, ...},
                   },
                   ...
               },
           )
   ```

2. **Register in `main.py`**: add to `RULE_CLASSES` and `build_rules()`.

3. **No Groovy edits needed.** Layers 2 / 2.5 / 3 are scope-agnostic and
   consume `affected_stages` / `block_filters` / `affected_stage_test_counts`
   directly.

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
- `cbts_input_json` exceeds the 256 KB piggyback cap — Layer 3 narrowing
  is dropped per stage; renderTestDB falls back to source test-db
- The narrowed YAML for this stage's testContext is missing or empty on
  the stage agent (e.g., main.py regen failed) — renderTestDB falls back
  to source test-db

Every fallback logs an `echo` line — no silent failures.

## Keep-in-sync notes

`blocks.py::derive_mako_from_stage` mirrors the Groovy
`getMakoArgsFromStageName` (`L0_Test.groovy` ~line 2079) and
`parseTaskConfigFromStageName` (~line 2066). New backends / orchestrators /
stage-name conventions on the Groovy side need a matching Python update —
file comments flag this.
