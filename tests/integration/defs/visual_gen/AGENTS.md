# AGENTS.md — VisualGen integration tests

Scope: this directory (`tests/integration/defs/visual_gen/`). Supplements the
repo-root `AGENTS.md`.

## `pytest.skip` policy

A VisualGen test that cannot run because a resource CI is expected to provide is
missing must **fail loudly**, not `pytest.skip` — a silent skip reports green and
hides a non-working test from CI. This covers missing model checkpoints, unbuilt or
unimportable first-party modules, and missing compiled TRT-LLM ops. For example,
resolve checkpoints with `from test_common.llm_data import get_checkpoint` — the same
helper the unit suite uses, which raises `FileNotFoundError` when the model is absent —
and do not fall back to `@cached_in_llm_models_root(..., fail_if_path_is_invalid=False)`,
which green-skips a missing checkpoint for the broader LLM suite.

Skips are only for genuine environment gating a run legitimately can't satisfy:
platform, hardware capability (CUDA / SM / arch), GPU count, and per-test
config/workload preconditions. When a condition couples such a gate with a disallowed
one, keep the gate as a skip and fail only on the resource half.
