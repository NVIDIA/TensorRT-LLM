---
name: sync-qa-tests
description: >-
  Maintains TensorRT-LLM QA lists: sync llm_function_core from test-db (pytorch/autodeploy
  accuracy+disaggregated); regenerate llm_function_core_sanity (P0 cap); align
  llm_perf_multinode.yml with disaggregated multinode perf sanity cases from test-db
  (disagg-/aggr- prefixes). Use for QA lists / test-db / perf multinode alignment.
---

# Sync QA tests (`llm_function_core` + sanity + multinode perf)

## Files

| File | Role |
|------|------|
| `tests/integration/test_lists/qa/llm_function_core.txt` | Primary functional pytest node-id list (release / weekly QA) |
| `tests/integration/test_lists/qa/llm_function_core_sanity.txt` | **P0** subset for faster torch sanity |
| `tests/integration/test_lists/qa/llm_perf_multinode.yml` | QA union of **disaggregated** multinode **perf sanity** `test_e2e[...]` cases (by GPU family) |
| `tests/integration/test_lists/test-db/*.yml` | GPU-matched CI schedules (`condition` + `tests`) |

## Keep `llm_function_core.txt` aligned with test-db

Integration cases referenced under **`backend: pytorch`** or **`backend: autodeploy`** in test-db, for paths starting with **`accuracy/`** or **`disaggregated/`**, should appear in `llm_function_core.txt` so QA coverage matches pre-merge/post-merge GPU DB selections. Other backends in the same YAML files (e.g. `tensorrt`, `cpp`) intentionally use different lists.

From a clone root:

```bash
python3 jenkins/scripts/sync_qa_tests.py sync-core
```

- **Only adds** missing node ids (union with existing); never removes entries. Sorts the full list alphabetically before writing.
- `--dry-run` — print how many would be added and show a sample; no write
- `--repo-root PATH` — non-standard layout

**Do not manually delete entries from `llm_function_core.txt`.** The file may legitimately contain tests beyond the pytorch/autodeploy test-db scope. Removing entries invalidates any corresponding `waives.txt` entries and triggers CI lint failures ("Non-existent test name in l0 or qa list"). `waives.txt` cleanup is handled by the CI pipeline — not by this script.

Implementation: `jenkins/scripts/sync_qa_tests.py` (subcommand `sync-core`).

After updating core, refresh the sanity list if needed:

```bash
python3 jenkins/scripts/sync_qa_tests.py regenerate-sanity
```

## Keep `llm_perf_multinode.yml` aligned with disagg multinode perf sanity (test-db)

`llm_perf_multinode.yml` is the **QA-side union** of disaggregated **perf sanity** multinode tests that appear under GPU DB files named like:

- `tests/integration/test_lists/test-db/l0_*_multi_nodes_perf_sanity*.yml`
- `tests/integration/test_lists/test-db/l0_b200_multi_nodes_perf_sanity*.yml` (same pattern, explicit B200)

When CI adds a **new active** `perf/test_perf_sanity.py::test_e2e[...]` line in those DB files (under **`backend: pytorch`**), add the same node id to `llm_perf_multinode.yml` so release/weekly QA tracks the same disagg perf sanity surface.

### Perf sanity parametrization prefixes (`test_perf_sanity.py`)

Integration lists use the short prefixes **`disagg-`** and **`aggr-`** (e.g. `disagg-gen_only-gb200_...`, `aggr-ctx_only-...`). Do not use the legacy `disagg_upload` / `aggr_upload` spelling in new edits.

### What to pull from test-db

1. **Scope — disagg gen-only perf sanity**  
   Consider **active** (non-`#` commented) list entries of the form:
   `perf/test_perf_sanity.py::test_e2e[disagg-gen_only-...]`  
   Other prefixes inside the bracket (`disagg-e2e-`, `aggr-...`, `aggr-ctx_only-`, etc.) exist in some DB files; **do not** auto-merge non–gen_only disagg lines into `llm_perf_multinode.yml` unless perf/QA owners explicitly want them in this union (today the file is **gen_only** disagg only).

2. **GPU family → YAML section**  
   Route the line into the correct `condition:` block by substring in the parametrization:
   - **`...-b200_...`** → section with `wildcards.gpu: ['b200']` (B200 block).
   - **`...-gb200_...`** → section with `wildcards.gpu: ['gb200']` (GB200 block).  
   If new GPU families appear (e.g. future `*-gb300_*` in DB), add a new `condition` + `tests` block and the same rules.

3. **Timeouts**  
   Preserve the same **` TIMEOUT (NNN)`** suffix as in test-db when present (commonly `120`). Keep list style consistent with existing entries (`  - perf/...`).

4. **Dedup and sort**  
   Within each GPU section, entries should be **unique** and **sorted alphabetically** by full line (as in the current file).

5. **What not to mix**  
   - Do not copy **commented-out** DB lines (`# - perf/...`) unless deliberately promoting them.  
   - Non–multi-node perf YAMLs (e.g. `l0_*_multi_gpus_perf_sanity.yml`) are a **different** schedule shape; only use them for `llm_perf_multinode.yml` if the parametrization is clearly the same multinode disagg perf sanity family and owners agree.

### Manual procedure (until scripted)

1. Search test-db: `rg "perf/test_perf_sanity.py::test_e2e\\[disagg-gen_only" tests/integration/test_lists/test-db/l0_*multi_nodes_perf_sanity*.yml`
2. For each **active** hit, assign **B200 vs GB200** section, append if missing.
3. Re-sort the `tests:` list in that section and re-run any QA list validator your pipeline uses.

## Regenerate `llm_function_core_sanity.txt`

```bash
python3 jenkins/scripts/sync_qa_tests.py regenerate-sanity
```

Implementation: `jenkins/scripts/sync_qa_tests.py` (subcommand `regenerate-sanity`).

- `--dry-run` — print counts and sample lines; no write
- `--repo-root PATH` — if not run from a standard clone layout

The script rewrites `llm_function_core_sanity.txt` with a fixed header comment block.

### Rules encoded in the sanity script

#### 1. Source

Parse `llm_function_core.txt`: non-empty lines that are not whole-line `#` comments; strip inline `#` comments.

#### 2. P0 model filter

Keep a line only if it matches **at least one** of these (case-insensitive where noted):

- **DeepSeek** — substring `deepseek`
- **Kimi** — substring `kimi`
- **GPT-OSS** — `gptoss`, `gpt-oss`, or `gpt_oss`
- **Nemotron** — substring `nemotron`
- **Qwen3** — substring `qwen3` (excludes Qwen2-only ids)
- **Llama 3.1 8B** — `Llama3_1_8B`, `Llama3.1-8B`, `Llama-3.1-8B`, `llama-3.1-8b`, `Meta-Llama-3.1-8B`, `meta-llama/Llama-3.1-8B`
- **Llama 3.3 70B** — `Llama3_3_70B`, `Llama3.3-70B`, `Llama-3.3-70B`

#### 3. Accuracy-only scope

Keep only node ids whose path starts with `accuracy/`. Disaggregated, E2E, and other paths are excluded.

#### 4. Per-method cap + hard total of 200

Group by pytest **method** = `file::TestClass::test_name` (strip `[param]`). Sort parametrizations alphabetically, keep at most **2 per method**, then sort the full list and truncate to **200**.

## Related

- GPU DB YAMLs drive CI; `llm_function_core.txt` is QA-scoped and may contain tests beyond what every GPU file lists (e.g. `test_e2e.py`, `unittest/`, `examples/`). Entries should only be removed when the underlying test case is deleted from the codebase.
- `waives.txt` cleanup (removing entries for deleted tests) is handled by the CI pipeline automatically. Do not remove waive entries or `llm_function_core.txt` entries as part of routine QA list maintenance.
- `llm_perf_multinode.yml` is scoped to the **disagg gen_only** perf sanity union described above; broader perf matrices stay in test-db or other QA YAMLs.
