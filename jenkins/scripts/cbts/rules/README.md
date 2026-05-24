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
| `auto_deploy_rule.py` | `AutoDeployRule` | `autodeployonly` | `examples/auto_deploy/**`, `tensorrt_llm/_torch/auto_deploy/**` (each excl. `.md`) |
| `visual_gen_rule.py` | `VisualGenRule` | `visualgenonly` | `examples/visual_gen/**`, `tensorrt_llm/_torch/visual_gen/**`, `tensorrt_llm/visual_gen/**` (each excl. `.md`) |
| `spec_dec_rule.py` | `SpecDecRule` | `specdeconly` | `tensorrt_llm/_torch/speculative/**`, `tensorrt_llm/models/{eagle,medusa,redrafter}/**`, `examples/{eagle,medusa,redrafter,draft_target_model,ngram}/**`, `examples/llm-api/llm_speculative_decoding.py` (each excl. `.md`) |
| `out_of_scope_rule.py` | `OutOfScopeRule` | `noop` | `tests/integration/test_lists/{qa,dev}/**`, `tests/integration/defs/.test_durations*`, `tests/microbenchmarks/**`, `**/*.md` (image suffixes intentionally not claimed — fall back to baseline since fixtures and doc diagrams are indistinguishable by location) |

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

## AutoDeployRule

Path-only rule. Claims source files under `examples/auto_deploy/` and
`tensorrt_llm/_torch/auto_deploy/` (excluding `.md`, which
`OutOfScopeRule` claims as noop). Other suffixes — including images —
are NOT excluded: a binary asset under an AD path could be a test
fixture, so the rule keeps claiming them and forces AD stages to
re-run.

Block selection — entry-based, two cases:
- **Primary**: blocks where `condition.terms.backend == 'autodeploy'`.
  Covers the 9 AD-conditioned blocks across all yamls.
- **Supplementary**: blocks containing entries with
  `test_llm_api_autodeploy.py` in the path or `_autodeploy-` in the
  parametrize id. Covers 3 entries that live in `backend: pytorch`
  blocks (l0_l40s, l0_perf) because Jenkins has no `L40S-AutoDeploy-*`
  / `H100-Perf-AutoDeploy-*` stage to consume a proper AD-conditioned
  block. The two patterns are stable conventions
  (`test_llm_api_autodeploy.py` is the AD accuracy filename;
  `_autodeploy-` is the cross-codebase backend parametrize value).

For each matched block, `block_filters` keeps only the AD entries
(every entry for AD-conditioned blocks; only entries matching the
supplementary patterns for leaker blocks). Non-AD siblings in leaker
blocks stay governed by other rules.

Outcomes:
- No AD source files in the diff → rule returns `None`.
- AD source touched → `scope=autodeployonly`; sanity off
  (AD changes don't affect wheel sanity); perfsanity on iff a
  matched block lives in `l0_perf` or `*perf_sanity*`.
- AD source touched but no AD block found anywhere (defensive) →
  `scope=None` (fallback).

Why narrowing is safe: AD is a beta backend isolated from the main
PyTorch backend. The 7 reverse imports of AD from non-AD code in
`bench/`, `executor/`, `commands/serve.py` are all lazy, guarded by
`if backend == "_autodeploy"`, so AD-only changes don't affect tests
that use the default PyTorch backend.
`scripts/check_auto_deploy_imports.py` enforces AD's outbound import
discipline statically.

## VisualGenRule

Path-only rule. Claims source files under `examples/visual_gen/`,
`tensorrt_llm/_torch/visual_gen/`, and `tensorrt_llm/visual_gen/`
(excluding `.md`, which `OutOfScopeRule` claims as noop). Image
suffixes are intentionally NOT excluded: VG ships reference images
used as test fixtures (e.g. `examples/visual_gen/cat_piano.png` and
`examples/visual_gen/serve/media/woman_skyline_original_720p.jpeg` are
loaded by `tests/unittest/_torch/visual_gen/`), so edits to them must
still force VG stages.

Block selection — entry-pattern based only:
VisualGen has no `condition.terms.backend` of its own; VG entries
live in `backend: pytorch` and `backend: tensorrt` blocks. A block
"belongs to VG" iff any of its `tests:` entries matches one of the
three stable VG path families:

- `unittest/_torch/visual_gen/...` (28 entries)
- `examples/test_visual_gen.py...` (1 entry)
- `visual_gen/test_visual_gen_benchmark.py` (1 entry)

For each matched block, `block_filters` keeps only the VG entries.
Non-VG siblings in the same block stay governed by other rules.

Outward-facing fallback: unlike AutoDeploy, VG is imported eagerly
(top-level `from tensorrt_llm._torch.visual_gen.config import ...`
in `commands/serve.py`, `commands/utils.py`,
`serve/openai_server.py`). The 5 files that define / re-export the
public API symbols (`VisualGenArgs`, `ParallelConfig`, `VisualGen`,
`VisualGenParams`) are listed in `_VG_OUTWARD_FILES`; touching any
of them claims the changed files but emits `scope=None` so Selector
falls back to baseline. This protects trtllm-serve / trtllm-bench
startup paths from VG signature drift slipping through pre-merge.

Outcomes:

- No VG source files in the diff → rule returns `None`.
- VG source touched, all internal → `scope=visualgenonly`; sanity
  off (VG changes don't affect wheel sanity); perfsanity on iff a
  matched block lives in `l0_perf` or `*perf_sanity*`.
- VG source touched, any outward-facing file → `scope=None`
  (fallback).
- VG source touched but no VG block found anywhere (defensive) →
  `scope=None` (fallback).

## SpecDecRule

Path-only rule. Claims source files under `tensorrt_llm/_torch/speculative/`,
`tensorrt_llm/models/{eagle,medusa,redrafter}/`,
`examples/{eagle,medusa,redrafter,draft_target_model,ngram}/`, and the
single file `examples/llm-api/llm_speculative_decoding.py` (excluding
`.md`, which `OutOfScopeRule` claims as noop). Other suffixes —
including images — are NOT excluded: a binary asset under a spec-dec
path could be a test fixture, so the rule keeps claiming them and
forces spec-dec stages to re-run.

Block selection — entry-pattern based only:
Spec-dec has no `condition.terms.backend` of its own; entries live in
`backend: pytorch` and `backend: tensorrt` blocks. A block "belongs to
spec-dec" iff any of its `tests:` entries matches one of the stable
markers in `_SPEC_ENTRY_PATTERNS`:

- Filename / method-name markers: `test_eagle`, `test_medusa`,
  `test_redrafter`, `test_ngram`, `test_draft_target_model`,
  `test_ad_speculative_decoding`, `unittest/_torch/speculative/`,
  `test_spec_decoding_metrics`, `test_llmapi_speculative_decoding`,
  `speculative_decoding_bls`, `test_mtp`.
- MTP parametrize-id markers: `mtp_nextn` (filtered — see below),
  `_mtp` (covers `throughput_mtp`, `*_mtp1`, `*_mtp3` suffixes).

For each matched block, `block_filters` keeps only the spec-dec entries.
Non-spec-dec siblings in the same block stay governed by other rules.

`mtp_nextn=0` carve-out: those parametrizations test MTP-disabled
baseline behavior, not the spec-dec code path. `_entry_is_spec` drops
the bare `mtp_nextn` signal when the entry also contains
`mtp_nextn=0`; another spec-dec marker must match for the entry to
qualify. As of May 2026 all 183 active `mtp_nextn=0` entries lack any
other marker and are therefore dropped.

No outward fallback needed: the only non-spec-dec eager import of
spec-dec types is `tensorrt_llm/commands/build.py` pulling
`SpeculativeDecodingMode` from `tensorrt_llm/models/modeling_utils.py`,
which lives outside `_SPEC_SRC_PREFIXES` and therefore is never claimed
by this rule. PRs that touch `modeling_utils.py` naturally fall back to
baseline.

Outcomes:

- No spec-dec source files in the diff → rule returns `None`.
- Spec-dec source touched → `scope=specdeconly`; sanity off (spec-dec
  changes don't affect wheel sanity); perf-sanity follows the matched
  blocks (True iff any matched block lives in `*_perf_sanity*` yaml).
- Spec-dec source touched but no spec-dec block found anywhere
  (defensive) → `scope=None` (fallback).

## OutOfScopeRule

Pure pattern match. Claims a changed file as `scope=noop` when it lives
in any subtree neither pre-merge nor post-merge L0 consumes:

- `tests/integration/test_lists/qa/` — QA-only test lists, separate
  nightly workflows.
- `tests/integration/test_lists/dev/` — developer-side artifacts, no L0
  consumer.
- `tests/integration/defs/.test_durations` — pytest-split timing cache.
- `tests/microbenchmarks/` — benchmarking scripts, no L0 stage.
- Any `*.md` file (docs anywhere in the repo cannot affect L0 tests).

Image extensions (`.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`) are
intentionally NOT claimed: image files anywhere in the repo can be test
fixtures (e.g. `examples/visual_gen/cat_piano.png` is loaded by
`tests/unittest/_torch/visual_gen/`), and location alone cannot
distinguish a fixture from a doc diagram. Image edits therefore fall
back to baseline unless a more specific rule (AD / VG / spec-dec)
claims them inside its source subtree.

Excludes `*.txt` (`requirements.txt` / `constraints.txt` are
runtime-relevant). `OUT_OF_SCOPE_PREFIXES` and `OUT_OF_SCOPE_SUFFIXES`
in `out_of_scope_rule.py` list the patterns.

## Helpers (`_helpers.py`)

| Helper | Used by | Purpose |
|---|---|---|
| `iter_diff_changes(diff)` | waives, testlist | Yields `(sign, body)` for each `+`/`-` content line. |
| `iter_diff_post_line_numbers(diff)` | testdef | Yields post-image (`+`) line numbers for AST scope mapping. |
| `lookup_ids_into_block_filters` | waives, testlist | Runs `find_match_for_waive` over a set of test ids; returns block_filters and miss set. |
| `lookup_paths_into_block_filters` | testdef | Runs `find_match_for_path` over a set of anchors; returns block_filters and miss set. |
| `resolve_affected_stages` | all narrowing rules | Maps `block_filters` keys to stage names via `stages_by_yaml_stem`. |
| `stages_by_yaml_stem` | all rules | Builds `{yaml_stem: [Stage, ...]}` index from the parsed Groovy stages. |
