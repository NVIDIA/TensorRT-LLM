# rules/

One rule per file; each inherits from `Rule` in `base.py`. Shared lookup
helpers live in `_helpers.py`. See the top-level [README](../README.md)
for the overall CBTS architecture.

## Current rules

| File | Class | Scope | Triggers on |
|---|---|---|---|
| `waives_rule.py` | `WaivesRule` | `waiveonly` | `tests/integration/test_lists/waives.txt` |
| `tests_def_rule.py` | `TestsDefRule` | `testdefonly` | `tests/**/*` (any file under tests/) |
| `test_list_rule.py` | `TestListRule` | `testlistonly` | `tests/integration/test_lists/test-db/*.yml` |
| `out_of_scope_rule.py` | `OutOfScopeRule` | `noop` | `tests/integration/test_lists/{qa,dev}/**`, `tests/integration/defs/.test_durations*`, `tests/microbenchmarks/**`, `tests/**/*.md` |

## WaivesRule

Reads `+`/`-` lines from the diff, normalizes each to a test id, then for
each id calls `YAMLIndex.find_match_for_waive` to walk the pytest parent
chain (function → class → file → dir) until a YAML entry matches. The
matched level becomes that block's Layer 3 filter prefix; stages whose
`mako` matches the block's `condition` go into `affected_stages`.

Outcomes:

- No actionable diff (whitespace / comment-only) → `scope=noop`.
- All ids unmatchable → `scope=noop`. The waived test isn't in any
  pre-merge YAML, so adding/removing its SKIP doesn't affect what runs.
- Some ids matched → `scope=waiveonly`; unmatchable misses are noted in
  the reason and ignored for narrowing.
- `sanity_relevant=True` when any matched block belongs to
  `l0_sanity_check`.

## TestsDefRule

For each file under `tests/` in the diff:

1. `git_path_to_yaml_key` translates the repo path to a YAML namespace
   key, or returns `None` for paths outside YAML's view (top-level
   integration `conftest.py`, dirs no YAML entry references such as
   `tests/integration/test_input_files/`).
2. `_compute_anchors` parses the diff's post-image line numbers and maps
   them through the file's AST to produce one of:
   - function-level anchors `path::Class::method` (every changed line
     lands in a `test_*` function inside a `Test*` class), or
   - file-level anchor `path` (any line lands at module scope, AST parse
     fails, or the file is unreadable — the latter covers .yaml/.txt/
     .json/etc. data files).
3. `lookup_paths_into_block_filters` calls `find_match_for_path`
   (bidirectional pytest-tree lineage) for each anchor; matches feed
   `block_filters`. For non-`test_*.py` paths, the lookup walks up
   enclosing directories to the narrowest YAML-covered ancestor — so
   `disaggregated/test_configs/foo.yaml` lifts to `disaggregated/`,
   `unittest/api_stability/references/llmapi.yaml` lifts to
   `unittest/api_stability/`.

`accuracy/references/*.yaml` gets a finer-grained refinement: each
top-level YAML key is a HF model name, mapped (via AST scan of
`accuracy/test_*.py` for `MODEL_NAME = "<hf>"` literals) to test
classes. A diff under `meta-llama/Llama-3.1-8B-Instruct:` narrows to
`accuracy/test_*.py::TestLlama3_1_8BInstruct` rather than the whole
`accuracy/` subtree. Models with no matching test class fall back to
the dir-level anchor.

Outcomes:

- Path is out-of-namespace (`git_path_to_yaml_key` returns `None`):
  - basename is `test_*.py`: claimed as noop — pytest doesn't auto-import
    test files into other tests, so an L0-unreferenced standalone test
    file has no impact on what L0 runs.
  - basename is conftest / `__init__` / a helper / data file: unhandled
    — Selector reports it and falls back. Could be implicitly imported
    (top-level conftest, sys-path helpers, test input fixtures).
- Path is in-namespace but no YAML-covered ancestor exists at any
  walk-up level: claimed as no-narrow contribution (`scope=noop` if
  all paths are like this; a miss-note in the reason on partial-narrow
  runs).
- Block-filter coverage ≥ `BLAST_RADIUS_FRACTION` (0.8) of total YAML
  blocks: `scope=None` (rule cannot usefully narrow — fallback).
- `sanity_relevant` / `perfsanity_relevant` follow from the matched
  blocks' YAML stem.

## TestListRule

Per touched `test-db/*.yml`, classifies each `+`/`-` line:

- Comment / blank → ignored.
- Indented `- <pytest_id>` (entry within a block) → recorded as added or
  removed.
- Anything else (top-level `- condition:`, mako edits, structural shifts)
  → `structural` list.

Only **added** entries drive narrowing; removals don't need verification
(the test either still runs elsewhere or is fully retired).

Outcomes:

- Any structural change → `scope=None` (Layer 2 stage matching depends
  on `condition`/mako; fallback).
- No additions (only removals or comment-only edits) → `scope=noop`.
- All added entries unresolvable against the post-PR YAML → `scope=noop`
  (the entries don't appear in any block).
- Some added entries resolved → `scope=testlistonly`; unresolved entries
  are noted in the reason and ignored for narrowing.

## OutOfScopeRule

Pure pattern match. Claims a changed file as `scope=noop` when it lives
in any subtree neither pre-merge nor post-merge L0 consumes:

- `tests/integration/test_lists/qa/` — QA-only test lists, separate
  nightly workflows.
- `tests/integration/test_lists/dev/` — developer-side artifacts, no L0
  consumer.
- `tests/integration/defs/.test_durations` — pytest-split timing cache.
- `tests/microbenchmarks/` — benchmarking scripts, no L0 stage.
- `tests/**/*.md` — Markdown docs.

`OUT_OF_SCOPE_PREFIXES` and `OUT_OF_SCOPE_TESTS_SUFFIXES` in
`out_of_scope_rule.py` list the patterns.

## Helpers (`_helpers.py`)

| Helper | Used by | Purpose |
|---|---|---|
| `iter_diff_changes(diff)` | waives, testlist | Yields `(sign, body)` for each `+`/`-` content line. |
| `iter_diff_post_line_numbers(diff)` | testdef | Yields post-image (`+`) line numbers for AST scope mapping. |
| `lookup_ids_into_block_filters` | waives, testlist | Runs `find_match_for_waive` over a set of test ids; returns block_filters and miss set. |
| `lookup_paths_into_block_filters` | testdef | Runs `find_match_for_path` over a set of anchors; returns block_filters and miss set. |
| `resolve_affected_stages` | all narrowing rules | Maps `block_filters` keys to stage names via `stages_by_yaml_stem`. |
| `stages_by_yaml_stem` | all rules | Builds `{yaml_stem: [Stage, ...]}` index from the parsed Groovy stages. |
