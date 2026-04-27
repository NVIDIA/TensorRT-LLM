# rules/

One rule per file; each inherits from `Rule` in `base.py`. See the
top-level [README](../README.md) for the overall CBTS architecture.

## Current rules

| File | Class | Scope | Triggers on | What it picks |
|---|---|---|---|---|
| `waives_rule.py` | `WaivesRule` | `waiveonly` | PR changes `tests/integration/test_lists/waives.txt` | For each added/removed test id (after `blocks.normalize_test_id` strips `SKIP`/`TIMEOUT`/`full:<gpu>/` decorations): look it up in the test-db YAML, pick stages whose `mako` matches the containing block's `condition`. |
