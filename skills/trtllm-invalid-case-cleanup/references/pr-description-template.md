# PR description template

Title: `[None][test] <what was cleaned and why>`

Use a JIRA or NVBug id in place of `None` when one exists. Fill every section;
write "none" rather than deleting a section.

```markdown
## Summary

<Two or three sentences: what kind of invalid cases were found, how many entries
and tests changed, and what the audit range was (for example `origin/main` at
<sha> through HEAD).>

## How the cases were found

- **Audit A (claim vs. behavior):** new tests and parameter values in the range,
  checked against their names, comments, bodies, and the code path they reach,
  including fallback branches.
- **Audit B (skipped on schedule):** source, test, list, and stage changes in
  the range, checked against every existing entry's stage, GPU, and skip
  sources.

## Decisions

| # | Entry or test | Audit | Why it is invalid (evidence) | Change | Coverage afterwards |
|---|---|---|---|---|---|
| 1 | `<node id or list entry>` | A-a / A-b / B | `<what the test claims> vs <what happens>`, `path/to/file.py:<line>` | removed from `l0_x.yml` / fixed / unlisted with NOTE / skip changed to fail | still runs on `<stage>` via `<test>`; or: **lost** |

## Unchanged on purpose

- `<entry>`: <why it looked invalid but is not, or why it could not be decided>.

## Test placement

<One sentence per moved or kept entry: which stage runs it now, and why that
stage and frequency fit the test's cost, as `tests/AGENTS.md` asks.>

## Validation

- [ ] `python scripts/check_test_list.py --validate --check-duplicate-waives`
- [ ] `python scripts/check_test_list.py --l0 --qa --waive`
- [ ] `pytest --collect-only -q` on every touched test file; unlisted ids still collect
- [ ] `pre-commit run --files <changed files>`
- Not run locally: <check and reason, or "none">

## Restoring an unlisted test

For each test kept in the tree with a NOTE: the change needed before it goes
back on a list.
```

Writing rules for the table:

- One row per decision. Rows that share a reason may share one evidence cell.
- Cite `file:line` for the body, branch, marker, or stage that disagrees.
- "Coverage afterwards" names a test and a stage, or says **lost**. A smaller
  suite does not show equal coverage.
- Group commits by reason, and keep the same grouping in the table.
