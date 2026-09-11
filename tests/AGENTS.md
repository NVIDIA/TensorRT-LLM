# AGENTS.md — tests

Scope: this directory and everything under it. Supplements the repo-root `AGENTS.md`.

## Every test must be in a CI test list

A new test file — or a new test that needs its own entry — MUST be added to the
appropriate CI test list under `tests/integration/test_lists/test-db/` (e.g.
`l0_cpu.yml`, `l0_b200.yml`, `l0_dgx_b200.yml`). A test that no list references —
directly or through a listed parent directory — is never collected in CI: it reports
nothing, so a regression in it merges unnoticed, a silent false green.

Choose the list by where the test runs: CPU-only tests go in `l0_cpu.yml`; single-GPU
tests in the per-GPU list for their target (e.g. `l0_b200.yml`); multi-GPU tests in a
multi-GPU list (e.g. `l0_dgx_b200.yml`, `l0_gb300_multi_gpus.yml`). If you are unsure
which list(s) a test belongs in, ask the author/reviewer rather than leaving it
unlisted.
