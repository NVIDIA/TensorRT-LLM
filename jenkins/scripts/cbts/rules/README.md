# rules/

One rule per file; each inherits from `Rule` in `base.py`. See the
top-level [README](../README.md) for the overall CBTS architecture.

## Current rules

| File | Class | Scope | Triggers on | What it picks |
|---|---|---|---|---|
| `waives_rule.py` | `WaivesRule` | `waiveonly` | PR changes `tests/integration/test_lists/waives.txt` | For each added/removed test id, calls `YAMLIndex.find_match_for_waive` to walk the pytest parent chain (function → class → file → dir → ...) until a YAML entry matches. The matched level becomes that block's Layer 3 filter prefix; stages whose `mako` matches the block's `condition` go into `affected_stages`. Any waive that misses every level → `scope=None` (full fallback). |
