---
name: trtllm-invalid-case-cleanup
description: >
  Find and clean up invalid TensorRT-LLM tests: newly added cases that do not
  exercise what their name or comments claim (including silent fallbacks to an
  already-covered path), and existing test-db/QA entries that a source, test, or
  list change now makes skip on the platform they are scheduled on. Fixes or
  unlists invalid tests with a NOTE, removes dead list entries, and drafts the
  PR description with the reason for every decision.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Invalid-case cleanup

An invalid case is one of two things:

1. **Claim mismatch.** The test does not do what its node id, parameters,
   docstring, or comments say, or a fallback inside the code under test sends
   it down a path that another test already covers.
2. **Dead entry.** A test-db or QA entry that skips, or never gets selected, on
   the platform it is scheduled to run on.

Both give green CI and no coverage. This skill finds them, changes the code and
lists, and writes the PR.

**Input:** a diff range (default `origin/main...HEAD` plus uncommitted
changes), or specific files, list files, or node ids. **Output:** edits, then a
PR with a per-decision description. If the user asks only for an audit or
report, stop after Step 5 and edit nothing.

Read these before editing; they decide what is safe to drop, merge, or waive:
`tests/AGENTS.md`, `tests/unittest/AGENTS.md`,
`tests/integration/test_lists/AGENTS.md`.

## Rules

- **Evidence over names.** Do not call a test invalid or a case valid from its
  name. Every finding cites file and line of the test body, the code branch, or
  the list condition that disagrees. If you cannot show it, put it under
  "unresolved" and do not edit it.
- **Do not run the suite to find these.** Read code, list conditions, and stage
  definitions. Run only collection and the list checks in Step 7.
- **A smaller suite is not equal coverage.** For each removal state which test
  or stage still covers the feature, or say plainly that coverage is lost.
- `TIMEOUT (N)` and `ISOLATION` are scheduling options, not claims. `#` starts a
  YAML comment anywhere on a list line, so it is never part of a node id.

## Step 1: Build the change set

```bash
BASE=origin/main
git diff --name-only $BASE...HEAD; git diff --name-only   # committed + uncommitted
git diff -U0 $BASE...HEAD -- 'tests/**/*.py' | rg '^\+\s*(def test_|@pytest|@parametrize)'
git diff -U0 $BASE...HEAD -- tests/integration/test_lists | rg '^\+\s*- '
```

Split the files into: **tests** (`tests/`), **lists** (`test_lists/test-db/`,
`qa/`, `waives.txt`), **stages** (`jenkins/L0_Test.groovy`, `conftest.py`
markers), and **source** (`tensorrt_llm/`, `cpp/`, kernels). Audit A takes new
tests and new parameter values. Audit B takes all four groups, and starts from
the entries that already exist.

## Step 2: Audit A — do new tests test what they claim?

For each added test function, added parametrize value, and added list entry:

1. **Collect the claim:** function and class name, every `name=value` token in
   the id (`parametrize_with_ids` in `tests/integration/defs/conftest.py` builds
   ids as `name=value` joined by `-`), docstring, comments on the test and on its
   list entry, list `condition` (`stage`, GPU wildcard, `system_gpu_count`).
2. **Answer (a): does the test cover the claim?** Read the body and every
   decorator. Flag the case when:
   - a `name=value` token is not a parameter of the function;
   - the parameter is accepted but never reaches the object the test builds
     (`LLM(...)`, MoE/attention/MTP/CUDA-graph/compile config), or the body
     overwrites it with a default;
   - nothing asserts the feature ran (a path flag, call count, sharding state,
     nonzero delta). A passing accuracy score alone does not prove the backend,
     graph, or parallelism path ran;
   - the value cannot take effect under the other parameters (for example a
     threshold above a batch cap that is also set by the test);
   - the harness chooses the thing the name claims to configure (for example a
     tester that picks the ctx or gen server itself).
3. **Answer (b): does a fallback change what runs?** Follow the call chain from
   the test into source until the claimed feature either executes or is dropped.
   Check every branch that selects an implementation: backend or strategy
   selectors, config validators that coerce or reset fields, `if not supported:`
   fallbacks, `try/except` that retries another path, env or default overrides,
   and mocks or stubs. Flag the case when:
   - a branch drops or replaces the claimed option for this parameter
     combination, so the claimed code is unreachable;
   - the case reaches only a path that another listed test already reaches.
     Compare executed inputs, assertions, fixtures, markers, and the leaf code;
     similar names are not the same test, and different names can run the same
     check.
4. Record the node id, the claim, the line where the claim is lost, and the
   test that already covers the real path, if any.

[references/invalid-patterns.md](references/invalid-patterns.md) lists the
patterns found in earlier cleanups, with the evidence each needed.

## Step 3: Audit B — do existing entries still run where scheduled?

Work from the changes, then from the entries they touch. A change can silence a
test without touching it. Look for:

- **Source:** a support or capability check now returns false for the GPU (SM
  gate, dtype or backend support table, new `raise`/`return` before the kernel).
- **Test code:** a new or changed `skip_less_device*`, `skip_pre_*`,
  `skip_post_*`, `skip_no_hopper`, `skipif`, `pytest.skip`, `importorskip`,
  `GTEST_SKIP`, a fixture or helper that skips (missing checkpoint, dataset, or
  `--run-*` flag), or a renamed test or changed parametrize id.
- **Lists and stages:** a moved or edited `condition`, a `system_gpu_count`
  that no stage provides, a stage removed or renamed in `L0_Test.groovy`, or an
  entry duplicated by a parent file or directory entry.
- **Waives:** a `waives.txt` line scoped by `full:<platform>/` or `full:sm<N>/`
  that now covers a scheduled entry.

For each affected entry:

1. Map it to its stage: `python scripts/test_to_stage_mapping.py --tests "<id>"`.
   A successful lookup does not prove the case runs.
2. Read the stage in `jenkins/L0_Test.groovy`
   (`"Stage": ["platform", "yaml", split, splits, gpu_count, nodes, ...]`) for
   the real GPU model, GPU count, and whether the stage exists.
3. Compare with every skip source on the test: markers on function, class, and
   module; `conftest.py` skip fixtures; early `pytest.skip` in the body and in
   fixtures; the `cpu_only` selection rule (CPU stages select `cpu_only`, GPU
   stages select `not cpu_only`). Marker skips are decided at run time, so
   `--collect-only` does not show them; read the marker, and run `pytest -rs` on
   matching hardware only if you have it.
4. A waive is a deliberate skip. Report it; it is invalid only when it covers a
   removed entry, duplicates another waive, or has no reason.
5. A `GTEST_SKIP` or `exit 0` skip that is recorded as PASSED is the worst case
   (green with no coverage). Make it fail instead.

## Step 4: Classify and decide

| Finding | Action |
|---|---|
| Entry skips on its stage; same id runs on another live stage | Remove the entry from that list only. |
| Entry skips on its stage; no other stage runs it; a live stage with the right GPU exists | Move it there. Say so in the PR. |
| Same, and no live stage fits | Remove the entry; list the lost coverage in the PR. |
| Entry needs a stage or GPU count that no stage provides | Move to a live stage, else remove. |
| Entry duplicates a parent file or directory entry | Remove the duplicate; parent keeps selecting it. |
| Test skips for an environment gap (missing checkpoint, dataset) | Make it `pytest.fail` with the cause; do not skip. |
| Test needs a flag CI never passes, and no other case covers it | Remove it from the lists. |
| Claim mismatch, with a small unambiguous fix (wire the parameter, add the missing assertion, move a threshold inside the cap) | Fix the test and assert the path ran. |
| Claim mismatch, no clear fix | Remove from every list; keep the function with a NOTE (Step 5). |
| Invalid parametrize values on an otherwise valid test | Remove those values; comment above the decorator. |
| Fallback reaches a path another test covers | Remove the entry or value; name the covering test in the PR. |
| Cannot decide (dynamic ids, generated bodies, needs hardware) | Leave unchanged; list under unresolved. |

Do not delete a test function unless it is an exact duplicate of a named test.
Dropping it from lists already stops the CI cost, and keeping the body keeps the
test id collectable and the reasoning in the code.

## Step 5: Report

```markdown
# Invalid-case audit: <range or files>

| Audit | Node id / entry | Claim or schedule | What actually happens | Where | Action |
|---|---|---|---|---|---|
| A-a | `...::test_x[tp4-flag=True]` | `flag=True` | body never passes `flag` to `LLM(...)` | `tests/...py:120` | fix |
| A-b | `...::test_y[...]` | NVLinkOneSided | `tp4` selects AllGatherReduceScatter, which drops the flag | `tensorrt_llm/...py:88` | remove |
| B | `l0_h100.yml: test_z` | runs on H100 | `skip_pre_blackwell` at `tests/...py:40`; stage is H100 | `jenkins/L0_Test.groovy:310` | remove entry |
```

Add: **Coverage kept** (where each removed case still runs), **Coverage lost**,
and **Unresolved**. Leave valid entries out of the table.

## Step 6: Make the changes

Skip this step in report-only mode.

- **Lists.** Remove the entry from every `test-db/*.yml` and `qa/*.txt` that
  holds it, and remove its `waives.txt` lines. Copy waive names whole and omit
  `ISOLATION`, `TIMEOUT (N)`, and YAML comments when matching. Keep
  `waives.txt` sorted.
- **Unlisted test.** Keep the function and its config so the id still collects.
  Put a NOTE directly above the decorators, in this shape:

  ```python
  # NOTE: this test does not test what its name claims, so it was removed from the
  # test lists. <What the body or code path really does, with the file:line.>
  # Before restoring it, <the change needed>. <Where the real coverage is, if any.>
  ```
- **Removed parametrize values.** Put a one-line comment above the decorator
  naming the dropped values and the reason.
- **Fixed test.** Make the claim observable: assert the path flag, call count,
  or per-step value that proves the feature ran. For an unsupported combination
  that cannot be reached, fail fast with `pytest.fail` instead of letting a
  fallback hide it.
- **Skip to fail.** Replace the environment-driven skip with `pytest.fail` and a
  message that names the missing input.
- One commit per reason group, titled `[None][test] <what>`, signed with
  `git commit -s`. Never use an NVIDIA internal email.

## Step 7: Validate

```bash
python scripts/check_test_list.py --validate --check-duplicate-waives
python scripts/check_test_list.py --l0 --qa --waive     # collection; no wheel needed
pytest --collect-only -q <each touched test file>       # unlisted ids must still collect
pre-commit run --files <changed files>
```

Fix anything these report before opening the PR. If a check cannot run locally,
say which one in the PR.

## Step 8: Open the PR and write the description

Push to your fork (`git remote -v`; usually `fork`) and open the PR against
`NVIDIA/TensorRT-LLM` `main`:

```bash
git push fork HEAD:<branch>
gh pr create --repo NVIDIA/TensorRT-LLM --base main --head <owner>:<branch> \
  --title "[None][test] <summary>" --body-file <description.md>
```

Use [references/pr-description-template.md](references/pr-description-template.md).
`tests/AGENTS.md` and the test-list guide expect the PR to explain every retired
test, every dropped coverage, and the placement of every moved entry.

## Resources

| File | Purpose |
|------|---------|
| `references/invalid-patterns.md` | Patterns found in earlier cleanups, with the evidence for each. |
| `references/pr-description-template.md` | PR description skeleton with one decision row per change. |
| `evals/` | Evaluation prompts and trimmed fixtures used by NVSkills CI. |
