# AGENTS.md — tests

Scope: this directory and everything under it. Supplements the repo-root `AGENTS.md`.

## Whether a new test runs in CI

This is a reminder for whoever adds a test, not a hard gate. Not every test runs
in CI — a test is a CI test, a QA test, or a local developer test:

- If a newly added test needs to be protected by CI, it must be included in at
  least one CI test list under `tests/integration/test_lists/test-db/` (e.g.
  `l0_cpu.yml`, `l0_b200.yml`, `l0_dgx_b200.yml`). A CI test that no list
  references — directly or through a listed parent directory — is never
  collected: it reports nothing, so a regression in it merges unnoticed, a
  silent false green.
- A test that is not included in any CI test list is treated as intended for
  local developer testing or QA rather than CI.

So when you add a test, confirm with the author/reviewer whether it should run
in CI and, if so, which platform's test list(s) should carry it. Leaving a new
test out of every CI list is fine for QA-only or local-only tests — but make it
a deliberate, confirmed decision rather than a forgotten step.

Choose the list by where a CI test runs: CPU-only tests go in `l0_cpu.yml`;
single-GPU tests in the per-GPU list for their target (e.g. `l0_b200.yml`);
multi-GPU tests in a multi-GPU list (e.g. `l0_dgx_b200.yml`,
`l0_gb300_multi_gpus.yml`). If you are unsure which list(s) a test belongs in,
ask the author/reviewer rather than leaving it unlisted.
