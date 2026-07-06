<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Telemetry Manifest Generation and Premerge Enforcement Design

Status: approved design; implementation has not started.

## Context

`tensorrt_llm/usage/llm_args_golden_manifest.json` is the committed privacy-review surface for every field selected by `build_capture_manifest()`. The repository has a test that compares the committed JSON with the live `TorchLlmArgs` and `TrtLlmArgs` model graph, but that test file is absent from every L0 test-list YAML. Consequently, the repository-defined premerge pipeline does not run the gate.

The current regeneration instructions are an inline `python -c` command. There is no supported generator, check mode, or pre-commit integration. A current-tree diagnostic run reached the assertion and confirmed drift:

- `TorchLlmArgs`: 235 to 258 rows, with 23 additions, no removals, and 4 changed rows.
- `TrtLlmArgs`: 261 to 279 rows, with 18 additions, no removals, and 2 changed rows.

The new rows are bounded numeric, boolean, numeric-list, or categorical values. No free-form string, path, callable, dictionary, or object payload appears in the generated delta.

## Goals

- Provide one deterministic, supported generator for the committed manifest.
- Automatically rewrite the manifest during pre-commit when manifest-defining files change.
- Stop the first commit after a rewrite so the developer reviews and stages the privacy-sensitive diff.
- Run the read-only drift gate in one A10 PyTorch premerge stage.
- Refresh the committed manifest from the current API after reviewing the generated delta.
- Keep `golden_manifest()` as the semantic source of truth and avoid public/runtime API changes.

## Non-goals

- Do not auto-stage or auto-commit the generated JSON.
- Do not let CI rewrite or accept a stale manifest.
- Do not add redundant H100 execution for a deterministic, hardware-independent comparison.
- Do not broaden the hook to every Python change under `tensorrt_llm/`.
- Do not refactor manifest selection, sanitization, or telemetry payload behavior.

## Architecture and data flow

The existing semantic path remains unchanged:

```text
TorchLlmArgs / TrtLlmArgs
        -> build_capture_manifest()
        -> golden_manifest()
        -> dedicated repository generator
        -> committed llm_args_golden_manifest.json
        -> documentation renderer and CI privacy gate
```

Add `scripts/generate_llm_args_golden_manifest.py` as a thin repository-maintenance layer over `golden_manifest()`. It owns path resolution, canonical JSON serialization, writing, checking, and human-readable drift output. Runtime telemetry code continues to own field discovery and manifest semantics.

## Generator interface

The command has two modes:

```bash
# Default: regenerate the committed file.
python3 scripts/generate_llm_args_golden_manifest.py

# Read-only: fail if the committed file differs.
python3 scripts/generate_llm_args_golden_manifest.py --check
```

Canonical output is `json.dumps(golden_manifest(), indent=2, sort_keys=True) + "\n"`. Internal functions accept an explicit `Path` so tests can use `tmp_path`; the public CLI always targets the repository manifest and does not expose an arbitrary output option.

Write mode compares content before writing. If content differs, it writes a sibling temporary file and atomically replaces the manifest. If content is already current, it performs no write. It exits successfully after a rewrite and relies on pre-commit's modified-file detection to stop the commit.

Check mode never writes. It exits zero when current and nonzero when stale, printing a concise unified diff between committed and generated text. Expected file-I/O failures produce a concise error and nonzero exit; unexpected import or generation failures propagate with their traceback. Both cases leave the committed file unchanged.

## Pre-commit integration

Add a local `language: system` hook with `pass_filenames: false`. The hook invokes generator write mode and uses the active TensorRT-LLM development environment. It triggers only when one of these manifest-defining paths changes:

- `scripts/generate_llm_args_golden_manifest.py`
- `tensorrt_llm/llmapi/llm_args.py`
- `tensorrt_llm/models/modeling_utils.py`
- `tensorrt_llm/usage/config.py`
- `tensorrt_llm/usage/llmapi_config.py`
- `tensorrt_llm/usage/llm_args_golden_manifest.json`

If the active environment cannot import the required TensorRT-LLM dependencies, the hook fails rather than silently skipping. Unrelated commits do not run the hook. Indirect manifest drift outside this scoped path set is still caught in a normal full premerge run, and whenever the A10 PyTorch block is selected; explicitly filtered CI runs are outside this guarantee.

The hook never stages files. When it rewrites the manifest, the normal workflow is: review the generated diff, stage it, and commit again.

## Premerge enrollment

Add `unittest/usage/test_llmapi_config_telemetry_docs.py` to the `stage: pre_merge`, `backend: pytorch` block in `tests/integration/test_lists/test-db/l0_a10.yml`.

Refactor `test_build_capture_manifest_matches_committed_golden` to load the generator and exercise its `--check` path in-process. This makes the supported check interface, rather than duplicate comparison logic, the premerge gate. No H100 enrollment is added.

## Manifest refresh and privacy review

Run generator write mode in a prepared current-tree TensorRT-LLM environment. Review the complete JSON diff before staging it. The currently observed delta includes:

- Attention-DP routing controls.
- Multimodal encoder runtime limits.
- KV-cache v2, disk-cache, and transfer polling controls.
- Sparse-attention configuration fields.
- Expanded MoE, NVFP4 GEMM, and sparse-attention categorical domains.

The review must confirm that every added or changed row remains within the existing safe annotation policy. The existing manifest CODEOWNER rule remains the approval mechanism for the committed privacy surface.

## Documentation

Replace the inline regeneration snippet in `tensorrt_llm/usage/schemas/README.md` with the supported generator command and document `--check` for read-only validation. The generated telemetry documentation continues to consume the committed golden; it does not invoke write mode.

## Testing

Generator-focused tests cover:

- Canonical serialization and trailing newline.
- Write-mode creation/update against `tmp_path`.
- Write-mode idempotence when content already matches.
- Check-mode success for current content.
- Check-mode nonzero result and unified diff for stale content.
- Preservation of the prior target when generation or writing fails.

End-to-end validation covers:

- The refreshed committed manifest passes the drift gate.
- The complete `test_llmapi_config_telemetry_docs.py` file passes in a prepared TRT-LLM test environment.
- The scoped pre-commit hook rewrites stale content and becomes clean on the second run.
- `scripts/test_to_stage_mapping.py` maps the gate to A10 PyTorch premerge stages and no H100 stage through this enrollment.
- Relevant formatting and pre-commit checks pass.

## Expected files

- `.pre-commit-config.yaml`
- `scripts/generate_llm_args_golden_manifest.py` (new)
- `tensorrt_llm/usage/llm_args_golden_manifest.json`
- `tensorrt_llm/usage/schemas/README.md`
- `tests/integration/test_lists/test-db/l0_a10.yml`
- `tests/unittest/usage/test_llmapi_config_telemetry_docs.py`

No public API signature, runtime telemetry payload contract, or manifest-selection rule changes are expected.

## Acceptance criteria

- A relevant staged API change automatically regenerates the manifest through pre-commit.
- A rewrite stops the first commit and leaves an inspectable, unstaged JSON diff.
- A second hook run passes once the refreshed manifest is staged/current.
- A stale committed manifest fails A10 premerge with an actionable diff.
- The regenerated current-tree manifest passes the gate and receives privacy CODEOWNER review.
- Unrelated commits do not invoke the heavy generator hook.

## Rejected alternatives

- Putting CLI/file-writing behavior in `tensorrt_llm/usage/llmapi_config.py` would mix repository maintenance into runtime telemetry code.
- Keeping an inline `python -c` pre-commit entry would duplicate serialization behavior and provide no testable tool boundary.
- CI-only generation would remove the committed manifest diff from the privacy-review workflow.
