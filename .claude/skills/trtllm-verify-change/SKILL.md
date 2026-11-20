---
name: trtllm-verify-change
description: Declare the tests that prove a TensorRT-LLM change, then run them by stage. The `landing` stage runs while iterating and covers only the tests the edit touches, within about 15 minutes. The `delivery` stage runs before a PR and covers every declared test on the committed tree. Results are judged as CI judges them (a skipped test or an empty selection fails), passes are kept in a per-worktree ledger, and `report --format md` prints the table for the PR description. Use when the user asks to verify a change, run its landing or delivery tests, decide what to run before opening or updating a PR, or attach test evidence to a PR. To build benchmark or evaluation commands, use trtllm-test-specialist.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Verify a change by stage

A change states up front which tests prove it. The tests then run in two stages:

| Stage | When | What runs | A pass stands while |
|---|---|---|---|
| `landing` | After each edit, while iterating | Landing tests whose `paths` the change touches; a test without `paths` always runs | The files the test covers are unchanged |
| `delivery` | Before the PR is opened or updated | Every declared test, in both stages | The tree is unchanged and nothing is uncommitted |

`scripts/verify_change.py` handles the mechanics: selecting tests, checking that each named test
exists, running pytest the way CI runs it, judging the JUnit report, recording results and
reporting them. Your job is the judgment: agreeing with the user on what proves the change,
reading failures, and reporting honestly.

## 1. Agree on the manifest with the user

The manifest lists the acceptance tests and is the user's decision. Propose one, show it, and get
agreement before relying on it.

```yaml
verify:
  tests:
    core:                     # a feature: a name for what these tests prove
      moe_routing:            # the test's name, unique across features
        purpose: the router picks the experts the HF reference picks
        pytest: tests/unittest/_torch/modeling_v2/test_moe_routing.py
        paths: [tensorrt_llm/_torch/modeling_v2/**/moe*.py]
      e2e:
        purpose: GSM8K and a long-context subset on one engine
        pytest: tests/integration/defs/accuracy/<family>/<target>/test_accuracy.py::TestX::test_e2e[{member}]
        members: [tep4, dep4]
        stage: delivery
        timeout: 5400
```

| Key | Required | Meaning |
|---|---|---|
| `purpose` | yes | One line saying what a pass proves. The report shows it. |
| `pytest` | yes | A repository-relative pytest node id, or a list of them. `{member}` and `{feature}` are filled in. |
| `paths` | no | Files, directories or globs (`*`, `**`, `?`). A change to a matching file makes the landing test run and invalidates its pass. Without `paths`, the test always runs at landing, and any change to the tree invalidates its pass. |
| `stage` | no | `landing` (the default) or `delivery`. |
| `members` | no | One case per member, each with `{member}` filled in, for example the parallel configurations a model test is parametrized by. |
| `devices` | no | The devices a test without members needs. `plan` shows the value; the script does not allocate devices. |
| `timeout` | no | The time limit for each case, in seconds (default 3600). On timeout the case's process group is killed and the case fails. |
| `env` | no | Variables added to pytest's environment. |

Choosing tests:

- Prefer tests CI already runs (`tests/integration/test_lists/test-db/`). Then a delivery pass and
  the CI result measure the same thing.
- Write a new test in the tree's existing test infrastructure. For accuracy tests, read
  `tests/integration/defs/accuracy/README.md`. Register the test in the CI test lists in the same
  PR when CI should also run it.
- Keep `landing` under about 15 minutes. If `plan` warns that a landing run exceeds the budget,
  ask the user whether to move tests to `delivery` or to narrow their `paths`.

The manifest lives in `$(git rev-parse --absolute-git-dir)/verify-change/manifest.yaml` by default:
one per worktree, and never committed. Pass `--manifest <file>` to use another file. Only the
file's `verify` block is read.

## 2. Commands

Run the script inside the environment where the tests can run: a TensorRT-LLM build that matches
the checkout, the GPUs the tests need, and `LLM_MODELS_ROOT` for tests that need model weights. The
script does not pick GPUs. Set `CUDA_VISIBLE_DEVICES` to idle ones before running.

```bash
S=.claude/skills/trtllm-verify-change/scripts/verify_change.py
python3 $S plan   [--stage landing|delivery]   # what runs and why, node definitions, estimated time
python3 $S run    [--stage landing|delivery]   # run cases without a standing pass; record results
python3 $S report [--stage landing|delivery] [--format md]   # standing result per case
```

Common options:

- `--only NAME`: restrict to one test (`e2e`) or one case (`core:e2e@tep4`). Repeatable.
- `--all`: at landing, run every landing test, whether or not the change touches it.
- `--base REF`: the ref landing measures the change against (default `origin/main`; the merge
  base with HEAD is used).
- `run --rerun`: run even when a pass stands, for example to check for flakiness.
- `run --fail-fast`: stop at the first failure.

Exit codes are 0 for success, 1 when a test failed or the stage is not satisfied, and 2 when the
manifest is unusable, a node is not defined, or git fails.

## 3. Workflow

1. **Before coding.** Write the manifest with the user. Run `plan --stage delivery` to confirm
   every node exists (exit 2 lists the ones that do not). A test the change itself adds counts as
   undefined until it is written.
2. **While iterating.** After each meaningful edit, run `run`, which uses the landing stage. On a
   failure, read the logs whose path it prints, fix the cause, and run it again. Passes whose
   inputs did not change are reused.
3. **Before the PR.** Commit, then run `run --stage delivery`. Delivery is satisfied only when
   `report --stage delivery` exits 0. Paste the output of `report --stage delivery --format md`
   into the PR description.
4. **After review changes.** Every new commit changes the tree, so delivery runs every case again.
   While debugging, use `--only` to rerun a single case; the report still requires all of them.

## 4. Rules

- Do not edit the manifest, a test it names, a reference value or a threshold to turn a failure
  into a pass. Propose the change to the user and explain why.
- A skipped test is a failure. Do not add skip conditions, narrow a node id or drop a member to
  get a pass.
- Do not say the change is ready, or mark the PR ready for review, until
  `report --stage delivery` exits 0 on the commit you are presenting.
- Report results as they are. Include failed and stale cases, what the logs show, and what you
  did not run.

## 5. Where things are

Everything is stored under `$(git rev-parse --absolute-git-dir)/verify-change/`, which is never
committed:

- `manifest.yaml`: the default manifest.
- `ledger.jsonl`: one record per case run, with HEAD, the content of the tree that was measured,
  the inputs digest, the verdict, the duration and the log directory.
- `runs/<UTC time>-<stage>/`: `<feature>.<test>[@<member>].{stdout,stderr}.log` and
  `.junit.xml` for each case. `--log-dir` puts them elsewhere.

When a result reads "the tree changed during the run", files that are tracked, or untracked but
not ignored, changed while the test ran. Either someone edited the tree, or the test writes into
it. The result is recorded but never reused. Stop editing during runs, or make the test write
outside the tree.
