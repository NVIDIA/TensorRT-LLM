# Contributing to llm-compiler

## Where to send PRs

The `llmc` standalone package is **generated from the AutoDeploy source tree inside [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)**. The mirror repo on GitHub is **read-only** — it is overwritten on every release by [`create_standalone_package.py`](./create_standalone_package.py).

**Do not open pull requests against the standalone `llmc` repo.** Any changes pushed there will be lost the next time the package is regenerated.

The supported workflow is:

1. **Fork** [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
1. Edit the AutoDeploy source under `tensorrt_llm/_torch/auto_deploy/` (and its tests under `tests/unittest/_torch/auto_deploy/` and `tests/unittest/auto_deploy/`).
1. **Optional:** validate your change against the standalone repo by regenerating it locally and running its test suite (see below).
1. Open a PR against `main` of `NVIDIA/TensorRT-LLM`.

You're welcome to fork the standalone `llmc` repo to experiment, iterate, or build on top — just remember the upstream source of truth is TensorRT-LLM.

## Validating a change with the standalone build

You can run the full standalone test suite locally before sending a PR. From the TensorRT-LLM checkout:

```bash
# 1. Run the import-discipline lint (also enforced as a pre-commit hook)
python scripts/check_auto_deploy_imports.py

# 2. Generate, install, and test the standalone package end-to-end
pytest tests/unittest/auto_deploy/standalone/ -q
```

`tests/unittest/auto_deploy/standalone/test_standalone_package.py` regenerates the `llmc` tree, installs it in an isolated venv, and runs the copied unit tests against the standalone install. This is the same job that gates merges.

To regenerate the standalone tree without running tests:

```bash
python examples/auto_deploy/llmc/create_standalone_package.py \
    --output-dir /path/to/llmc_pkg
```

## PR conventions

PRs against `NVIDIA/TensorRT-LLM` follow the project-wide rules:

- **Title format:** `[JIRA/NVBUG/None][type] description` (e.g. `[None][feat] llmc: add foo transform`).
- **DCO sign-off** is required: commit with `git commit -s`. Don't add AI tools or co-authors to the sign-off line.
- The pre-commit hooks (formatting, lint, the `auto-deploy-import-discipline` hook) run on commit. If they modify files, re-stage and commit again.

For the rest (full coding guidelines, branching policy, CI commands), see [the TensorRT-LLM `CONTRIBUTING.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md) and the [project-level `AGENTS.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/AGENTS.md).

## Import discipline

`llmc` source is the same files as `tensorrt_llm/_torch/auto_deploy/` — copied verbatim, never rewritten. For that to work cleanly:

- Imports **inside** the auto_deploy package must be **relative** (`from ..foo import bar`).
- Imports **to the rest of TensorRT-LLM** must be **absolute** (`from tensorrt_llm.X import Y`) and gated at runtime by `_compat.TRTLLM_AVAILABLE` so they fail gracefully in standalone mode.

Both rules are enforced by `scripts/check_auto_deploy_imports.py`, which runs as a pre-commit hook on every change to `tensorrt_llm/_torch/auto_deploy/`.
