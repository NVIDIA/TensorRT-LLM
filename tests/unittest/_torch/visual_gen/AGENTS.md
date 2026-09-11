# AGENTS.md — VisualGen unit tests

Scope: this directory and everything under it (`tests/unittest/_torch/visual_gen/`,
including `multi_gpu/`, `kernels/`, `sparse_attention/`). Supplements the repo-root
`AGENTS.md`.

## `pytest.skip` policy

A VisualGen test that cannot run because a resource CI is expected to provide is
missing must **fail loudly**, not `pytest.skip` — a silent skip reports green and
hides a non-working test from CI. This covers missing model checkpoints, unbuilt or
unimportable first-party modules, and missing compiled TRT-LLM ops. For example,
resolve checkpoints from `LLM_MODELS_ROOT` via
`from utils.llm_data import get_checkpoint`, which raises `FileNotFoundError` when the
model is absent — don't guard the test with `if not os.path.exists(ckpt): pytest.skip(...)`.

Skips are only for genuine environment gating a run legitimately can't satisfy:
platform (`sys.platform`), hardware capability (CUDA / SM / arch), GPU count, and
per-test config/workload preconditions. When a condition couples such a gate with a
disallowed one, keep the gate as a skip and fail only on the resource half.
