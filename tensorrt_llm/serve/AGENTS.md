<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md - tensorrt_llm/serve

Scope: `tensorrt_llm/serve/` and everything under it. Supplements the repo-root `AGENTS.md`
and `tests/AGENTS.md`; `tensorrt_llm/commands/AGENTS.md` points here for the serve CLI.
Where a check enforces a rule, the rule names it; a failing check says what to change.
Name files and symbols here, never line numbers.

## Where code goes

- Model-specific behavior: a hook on `ServingExtension` (`serve/serving_extensions.py`,
  `serve/extensions/`). Do not branch on a model name in a handler.
- Prompt rendering: `serve/render/` only. Handlers do not call `apply_chat_template`.
- Request and response models: a new field needs a row in
  `tests/unittest/api_stability/references/trtllm_serve_api.yaml`.
  Check: `tests/unittest/api_stability/test_serve_api.py`.
- Routes: every server configuration's routes (path, method, route class, and the order of
  routes that can match the same request) are pinned in
  `tests/unittest/api_stability/references/trtllm_serve_routes.yaml`. A route change
  regenerates it and needs the API review committee.
  Check: `tests/unittest/api_stability/test_serve_routes.py`.
- Middleware order matters: `add_middleware` inserts at the front, so the last one added is
  outermost. Do not reorder. Check: `tests/unittest/serve/test_middleware_order.py`.
- Import directions between layers are tests; a PR that creates a package adds its contract.
  Check: `tests/unittest/serve/test_layering.py`.
- VisualGen serve tests live under a path that contains `visual_gen/`
  (`tests/unittest/serve/api/visual_gen/`), or CI selection misses them.

## Moving code

- A move PR only moves code: no reformatting, no logic change.
- In the same PR update the multi-GPU file list (`getMultiGpuFileChanged` in
  `jenkins/L0_MergeRequest.groovy`), `.github/CODEOWNERS` and `legacy-files.txt`. A new file
  takes the formatting group of the file its code comes from.
- Old import paths: a move deletes the old module (or the moved names) without a shim and
  rewrites every import of it in the repository, tests included; module paths written as
  strings (patch targets, subprocess command lines, `_LAZY_ATTRS` in `serve/__init__.py`) are
  fixed by hand. The PR body lists "old path → new path" for the release notes. Only five old
  paths keep a shim, because code outside the repository imports them:
  `tensorrt_llm/llmapi/reasoning_parser.py`, `serve/tool_parser/`,
  `serve/openai_disagg_server.py`, `serve/openai_server.py` and `commands/serve.py`. A shim
  re-exports public names only, lists them in `__all__` (or `_LAZY_NAMES` in
  `commands/serve.py`), warns once with `FutureWarning`, and has its rows in
  `tests/unittest/serve/test_shim_identity.py`.
- `mock.patch("old.module.name")` does not follow moved code. Grep the tests for every moved
  name, including `monkeypatch.setattr(module, "name", ...)` and dotted paths split across lines.

## Tests

- Register every new test file in a CI list in the same PR (`tests/AGENTS.md`). Serve tests
  that need no GPU go in `tests/integration/test_lists/test-db/l0_cpu.yml` and must contain
  `pytest.mark.cpu_only`, or the CPU stage collects nothing from them.
- New serve tests mirror the package: `tests/unittest/serve/` for `tensorrt_llm/serve/`.
- Build servers with the factory and fake generators in `tests/test_common/serve/`
  (`isolated_serve_env`), never `object.__new__(OpenAIServer)` or
  `object.__new__(OpenAIDisaggServer)`. Check: `tests/unittest/serve/test_no_object_new.py`.
- List an app's routes with `effective_routes()` from `tests/test_common/serve/routes.py`, not
  `app.routes`: FastAPI 0.139 keeps included routers as objects without `.path`.
- The harmony tests guard whole files with try-imports. After renaming anything they import,
  run them with `pytest -rs` and check that nothing is skipped as "not importable": a green run
  can mean every test was skipped.

## Commands

- All serve CPU tests (the runner exits when `LLM_MODELS_ROOT` is unset):
  `LLM_MODELS_ROOT=/path/to/models PYTHONPATH=tests python3 -m test_common.serve.cpu_tests`
- Route snapshot: `pytest -m cpu_only tests/unittest/api_stability/test_serve_routes.py tests/unittest/serve/api/visual_gen/test_routes_visual_gen.py`
- Regenerate the route snapshot: `PYTHONPATH=tests python3 -m test_common.serve.routes --write`
- API gates: `pytest tests/unittest/api_stability/test_serve_api.py tests/unittest/api_stability/test_serve_cli.py`
- Extensions and parsers: `pytest tests/unittest/llmapi/apps/test_serving_extensions.py tests/unittest/llmapi/apps/test_tool_parsers.py`
- Disaggregated router and coordinator (CPU): `pytest tests/unittest/disaggregated -m cpu_only -q`
- Which CI stage runs a test: `python scripts/test_to_stage_mapping.py --tests "<test>"`. It
  matches the text of list entries and does not expand directories: a file listed through its
  directory maps only when you pass the directory entry, e.g. `--tests "unittest/serve"` for
  anything under `tests/unittest/serve/`.
- Tests that load model files need `LLM_MODELS_ROOT` (the harmony tests' autouse fixture does).
