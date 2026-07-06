# Telemetry Manifest Premerge Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically regenerate the committed LLM-args telemetry manifest during relevant pre-commit runs and enforce the current manifest in A10 premerge CI.

**Architecture:** Keep `golden_manifest()` as the sole semantic source of truth. Add a thin repository generator that atomically writes canonical JSON or checks it read-only, then share that interface between a scoped system pre-commit hook and the existing privacy-gate unit test.

**Tech Stack:** Python 3.10+, Pydantic model metadata, pytest, pre-commit, YAML test-db/Jenkins stage mapping, JSON, Git/DCO.

---

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

**Approved design:** `docs/superpowers/specs/2026-07-06-telemetry-manifest-premerge-design.md`

## Execution prerequisites

- Work on branch `venky/telemetry-manifest-premerge-gate`, based on `upstream/main` commit `7c8dde830bac813e23605d47a1d27c92d5437a92`.
- Run import-dependent commands in a prepared TensorRT-LLM development container/environment. The bare workstation Python currently stops during collection because `transformers` is absent; the test itself needs no model weights and does not require `LLM_MODELS_ROOT`.
- Preserve the project scratch symlink. An untracked `slop/` entry is expected and must not be staged.
- Use `git commit -s`; do not add co-authors or AI attribution.

## File responsibility map

- Create `scripts/generate_llm_args_golden_manifest.py`: canonical rendering, atomic write mode, read-only check mode, unified drift output, CLI exit codes.
- Modify `tests/unittest/usage/test_llmapi_config_telemetry_docs.py`: focused generator tests and the committed-golden privacy gate.
- Modify `tensorrt_llm/usage/llm_args_golden_manifest.json`: generated current-tree privacy surface only.
- Modify `.pre-commit-config.yaml`: scoped `language: system` mutating hook.
- Modify `tensorrt_llm/usage/schemas/README.md`: supported write/check commands.
- Modify `tests/integration/test_lists/test-db/l0_a10.yml`: one A10 PyTorch premerge enrollment.

No runtime API, telemetry selection rule, schema, or payload behavior changes.

### Task 1: Build the deterministic generator with unit tests

**Files:**
- Create: `scripts/generate_llm_args_golden_manifest.py`
- Modify: `tests/unittest/usage/test_llmapi_config_telemetry_docs.py:23-52`
- Test: `tests/unittest/usage/test_llmapi_config_telemetry_docs.py`

- [ ] **Step 1: Add tests for canonical writing, idempotence, stale checks, and atomic-failure preservation**

Add this loader after `_load_generator()`:

```python
def _load_manifest_generator() -> ModuleType:
    module_path = _repo_root() / "scripts/generate_llm_args_golden_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_llm_args_golden_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
```

Add this sample payload and the three tests after `_golden_path()`:

```python
def _sample_manifest() -> dict[str, list[dict[str, object]]]:
    return {
        "TorchLlmArgs": [
            {
                "allowed_values": [],
                "annotation": "<class 'bool'>",
                "converter": "",
                "kind": "value",
                "path": "flag",
            }
        ],
        "TrtLlmArgs": [],
    }


def test_manifest_generator_write_is_canonical_and_idempotent(tmp_path, monkeypatch):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"

    assert generator._write_manifest(manifest_path)
    assert manifest_path.read_text() == json.dumps(
        _sample_manifest(), indent=2, sort_keys=True
    ) + "\n"
    assert not generator._write_manifest(manifest_path)
    assert generator._check_manifest(manifest_path)


def test_manifest_generator_check_reports_unified_diff(tmp_path, monkeypatch, capsys):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"stale": true}\n')

    assert not generator._check_manifest(manifest_path)
    stderr = capsys.readouterr().err
    assert f"--- {manifest_path} (committed)" in stderr
    assert f"+++ {manifest_path} (generated)" in stderr
    assert '-{"stale": true}' in stderr
    assert '+  "TorchLlmArgs": [' in stderr
    assert manifest_path.read_text() == '{"stale": true}\n'


def test_manifest_generator_preserves_target_when_generation_fails(
    tmp_path, monkeypatch
):
    import pytest

    generator = _load_manifest_generator()
    manifest_path = tmp_path / "manifest.json"
    old_content = '{"old": true}\n'
    manifest_path.write_text(old_content)

    def _fail_generation():
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(generator, "golden_manifest", _fail_generation)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        generator._write_manifest(manifest_path)

    assert manifest_path.read_text() == old_content
    assert list(tmp_path.iterdir()) == [manifest_path]


def test_manifest_generator_preserves_target_when_replace_fails(tmp_path, monkeypatch):
    import pytest

    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "golden_manifest", _sample_manifest)
    manifest_path = tmp_path / "manifest.json"
    old_content = '{"old": true}\n'
    manifest_path.write_text(old_content)

    def _fail_replace(*_args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(generator.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        generator._write_manifest(manifest_path)

    assert manifest_path.read_text() == old_content
    assert list(tmp_path.iterdir()) == [manifest_path]


def test_manifest_generator_main_reports_file_io_failure(monkeypatch, capsys):
    generator = _load_manifest_generator()
    monkeypatch.setattr(generator, "_render_manifest", lambda: "{}\n")

    def _fail_write(*_args, **_kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(generator, "_write_manifest", _fail_write)
    assert generator.main([]) == 2
    assert "synthetic write failure" in capsys.readouterr().err


def test_manifest_generator_main_propagates_generation_failure(monkeypatch):
    import pytest

    generator = _load_manifest_generator()

    def _fail_generation():
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(generator, "_render_manifest", _fail_generation)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        generator.main([])
```

- [ ] **Step 2: Run the new tests and verify the RED state**

Run in the prepared environment:

```bash
python3 -m pytest \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py \
  -k 'manifest_generator_' -q -p no:cacheprovider
```

Expected: six failures while `_load_manifest_generator()` tries to execute the missing `scripts/generate_llm_args_golden_manifest.py`.

- [ ] **Step 3: Add the minimal generator implementation**

Create `scripts/generate_llm_args_golden_manifest.py` with this complete content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import difflib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tensorrt_llm.usage.llmapi_config import golden_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST_PATH = _REPO_ROOT / "tensorrt_llm/usage/llm_args_golden_manifest.json"


def _render_manifest() -> str:
    return json.dumps(golden_manifest(), indent=2, sort_keys=True) + "\n"


def _read_manifest(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_manifest(
    path: Path = _DEFAULT_MANIFEST_PATH, *, generated: str | None = None
) -> bool:
    if generated is None:
        generated = _render_manifest()
    if _read_manifest(path) == generated:
        return False

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = 0o644

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            temporary_file.write(generated)
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def _check_manifest(
    path: Path = _DEFAULT_MANIFEST_PATH, *, generated: str | None = None
) -> bool:
    committed = _read_manifest(path)
    if generated is None:
        generated = _render_manifest()
    if committed == generated:
        return True

    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=f"{path} (committed)",
        tofile=f"{path} (generated)",
    )
    sys.stderr.writelines(diff)
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or check the committed LLM-args telemetry manifest."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the committed manifest without modifying it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generated = _render_manifest()
    try:
        if args.check:
            return 0 if _check_manifest(generated=generated) else 1
        changed = _write_manifest(generated=generated)
    except OSError as error:
        print(f"Failed to access {_DEFAULT_MANIFEST_PATH}: {error}", file=sys.stderr)
        return 2

    if changed:
        print(f"Updated {_DEFAULT_MANIFEST_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m pytest \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py \
  -k 'manifest_generator_' -q -p no:cacheprovider
```

Expected: `6 passed`; check mode leaves stale input untouched, both failure-preservation tests leave the original target unchanged, the replace failure leaves no temporary file behind, the CLI converts expected file-I/O failure into exit code 2, and unexpected generation failure propagates.

- [ ] **Step 5: Run formatting and focused pre-commit checks**

Run:

```bash
pre-commit run --files \
  scripts/generate_llm_args_golden_manifest.py \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py
```

Expected: Python formatting/lint and repository checks pass. If a hook reformats either file, inspect the diff, stage the formatting, and rerun until clean.

- [ ] **Step 6: Commit the generator and its tests**

```bash
git add \
  scripts/generate_llm_args_golden_manifest.py \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py
git commit -s -m '[None][feat] add telemetry manifest generator'
```

Expected: commit succeeds with DCO and no unrelated files staged.

### Task 2: Make the real privacy gate use check mode and refresh the golden

**Files:**
- Modify: `tests/unittest/usage/test_llmapi_config_telemetry_docs.py:52-62`
- Modify: `tensorrt_llm/usage/llm_args_golden_manifest.json:1`
- Test: `tests/unittest/usage/test_llmapi_config_telemetry_docs.py::test_build_capture_manifest_matches_committed_golden`

- [ ] **Step 1: Refactor the existing gate to exercise the supported check interface**

Replace the body of `test_build_capture_manifest_matches_committed_golden()` with:

```python
def test_build_capture_manifest_matches_committed_golden():
    """The CI privacy gate (closes TRTLLM-12872).

    Regenerate in-memory and diff against the committed golden; any drift
    ('field X now phones home') must be a deliberate, privacy-reviewed golden
    update committed in the same change.
    """
    generator = _load_manifest_generator()
    assert generator.main(["--check"]) == 0
```

- [ ] **Step 2: Run the gate before regeneration and verify the expected failure**

Run:

```bash
python3 -m pytest \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py::test_build_capture_manifest_matches_committed_golden \
  -q -p no:cacheprovider
```

Expected: FAIL with `assert 1 == 0`; stderr contains a unified diff showing current API rows missing from the committed JSON.

- [ ] **Step 3: Regenerate the committed manifest through the supported writer**

Run:

```bash
python3 scripts/generate_llm_args_golden_manifest.py
```

Expected: stdout is `Updated tensorrt_llm/usage/llm_args_golden_manifest.json`, and only the generated JSON changes.

- [ ] **Step 4: Audit the generated privacy-surface delta**

Run this read-only summary while `HEAD` still contains the old golden:

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

path = "tensorrt_llm/usage/llm_args_golden_manifest.json"
before = json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True))
after = json.loads(Path(path).read_text())

for model_name in sorted(before):
    old_rows = {row["path"]: row for row in before[model_name]}
    new_rows = {row["path"]: row for row in after[model_name]}
    added = sorted(new_rows.keys() - old_rows.keys())
    removed = sorted(old_rows.keys() - new_rows.keys())
    changed = sorted(
        key for key in old_rows.keys() & new_rows.keys() if old_rows[key] != new_rows[key]
    )
    print(
        f"{model_name}: old={len(old_rows)} new={len(new_rows)} "
        f"added={len(added)} removed={len(removed)} changed={len(changed)}"
    )
    assert not removed, removed
    for key in added:
        print("  +", json.dumps(new_rows[key], sort_keys=True))
    for key in changed:
        print("  ~ old", json.dumps(old_rows[key], sort_keys=True))
        print("  ~ new", json.dumps(new_rows[key], sort_keys=True))
PY
```

Expected summary:

```text
TorchLlmArgs: old=235 new=258 added=23 removed=0 changed=4
TrtLlmArgs: old=261 new=279 added=18 removed=0 changed=2
```

Review every printed row. Confirm annotations are bounded numeric, boolean, numeric-list, or categorical values; reject any free-form string, path, dictionary, callable, or object payload instead of blindly accepting it.

- [ ] **Step 5: Verify check mode and the privacy gate now pass**

Run:

```bash
python3 scripts/generate_llm_args_golden_manifest.py --check
python3 -m pytest \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py::test_build_capture_manifest_matches_committed_golden \
  -q -p no:cacheprovider
```

Expected: both commands exit zero; pytest reports `1 passed` and check mode does not modify the JSON.

- [ ] **Step 6: Commit the refreshed privacy contract**

```bash
git add \
  tensorrt_llm/usage/llm_args_golden_manifest.json \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py
git commit -s -m '[None][chore] refresh LLM args telemetry manifest'
```

Expected: the commit contains the generated JSON and the gate refactor only.

### Task 3: Automate regeneration in pre-commit and document the tool

**Files:**
- Modify: `.pre-commit-config.yaml:2853-2863`
- Modify: `tensorrt_llm/usage/schemas/README.md:253-257`
- Test: scoped local hook `generate-llm-args-golden-manifest`

- [ ] **Step 1: Verify the hook does not exist yet**

Run:

```bash
pre-commit run generate-llm-args-golden-manifest --all-files
```

Expected: nonzero exit with `No hook with id 'generate-llm-args-golden-manifest'`.

- [ ] **Step 2: Add the scoped mutating system hook**

Insert this as the first hook under the existing `repo: local` block:

```yaml
    -   id: generate-llm-args-golden-manifest
        name: Generate LLM args telemetry manifest
        entry: python3 scripts/generate_llm_args_golden_manifest.py
        language: system
        files: ^(scripts/generate_llm_args_golden_manifest\.py|tensorrt_llm/(llmapi/llm_args\.py|models/modeling_utils\.py|usage/(config\.py|llmapi_config\.py|llm_args_golden_manifest\.json)))$
        pass_filenames: false
```

Do not set `always_run`; the `files` expression is the agreed environment/cost boundary.

- [ ] **Step 3: Replace the inline README command with the supported interface**

Replace checklist item 6 with:

```markdown
6. **Regenerate the manifest golden** from `build_capture_manifest`:
   `python3 scripts/generate_llm_args_golden_manifest.py`
   To validate without modifying the committed file, run
   `python3 scripts/generate_llm_args_golden_manifest.py --check`.
   Review the golden diff — **it is the privacy review.** A newly captured field
   requires sign-off from the GitHub telemetry/privacy CODEOWNER (`.github/CODEOWNERS`).
```

- [ ] **Step 4: Validate the hook's positive and negative trigger paths**

Run in the prepared environment:

```bash
pre-commit validate-config
pre-commit run generate-llm-args-golden-manifest --all-files
pre-commit run generate-llm-args-golden-manifest --files tensorrt_llm/llmapi/llm_args.py
pre-commit run generate-llm-args-golden-manifest --files tensorrt_llm/usage/schemas/README.md
```

Expected:

- Config validation exits zero.
- `--all-files` and the `llm_args.py` run pass without modifying the current manifest.
- The README-only run reports `(no files to check) Skipped`, proving unrelated docs changes do not invoke the heavy import path.

- [ ] **Step 5: Prove the hook rewrites stale content and passes on its second run**

Temporarily insert one blank line after the manifest's opening brace:

```diff
 {
+
   "TorchLlmArgs": [
```

Then run:

```bash
pre-commit run generate-llm-args-golden-manifest \
  --files tensorrt_llm/usage/llm_args_golden_manifest.json
pre-commit run generate-llm-args-golden-manifest \
  --files tensorrt_llm/usage/llm_args_golden_manifest.json
git diff --exit-code -- tensorrt_llm/usage/llm_args_golden_manifest.json
```

Expected:

- The first hook run reports failure because it rewrites the deliberately stale file.
- The generated content removes the temporary blank line and exactly restores the committed JSON.
- The second hook run passes, and `git diff --exit-code` exits zero.

- [ ] **Step 6: Run checks for the hook and documentation diff**

```bash
pre-commit run --files \
  .pre-commit-config.yaml \
  scripts/generate_llm_args_golden_manifest.py \
  tensorrt_llm/usage/schemas/README.md
git diff --check
```

Expected: all applicable hooks pass and the diff has no whitespace errors.

- [ ] **Step 7: Commit pre-commit automation and documentation**

```bash
git add .pre-commit-config.yaml tensorrt_llm/usage/schemas/README.md
git commit -s -m '[None][infra] automate telemetry manifest generation'
```

Expected: commit hooks run the new generator, leave the manifest unchanged, and accept the commit with DCO.

### Task 4: Enroll the privacy gate in A10 premerge

**Files:**
- Modify: `tests/integration/test_lists/test-db/l0_a10.yml:74-80`
- Test: `scripts/test_to_stage_mapping.py`

- [ ] **Step 1: Capture the missing-stage RED state**

Run:

```bash
python3 scripts/test_to_stage_mapping.py \
  --tests unittest/usage/test_llmapi_config_telemetry_docs.py
```

Expected: no output, confirming the test is still unenrolled.

- [ ] **Step 2: Add the exact test file to the A10 PyTorch premerge block**

Insert the new entry beside the other usage tests:

```yaml
  - unittest/usage/test_collectors.py
  - unittest/usage/test_config.py
  - unittest/usage/test_llmapi_config_telemetry_docs.py
  - unittest/usage/test_opt_out.py
  - unittest/usage/test_reporter.py
  - unittest/usage/test_schema.py
  - unittest/usage/test_transport.py
  - unittest/usage/test_e2e_capture.py
```

Do not add the file to `l0_h100.yml`.

- [ ] **Step 3: Verify stage mapping turns GREEN with A10 only**

Run:

```bash
python3 scripts/test_to_stage_mapping.py \
  --tests unittest/usage/test_llmapi_config_telemetry_docs.py
```

Expected exact output:

```text
A10-PyTorch-1
A10-PyTorch-2
```

No H100 or post-merge stage should appear.

- [ ] **Step 4: Validate the test-list entry against source**

Run:

```bash
pre-commit run validate-test-lists --files tests/integration/test_lists/test-db/l0_a10.yml
python3 scripts/check_test_list.py --validate
```

Expected: both validations exit zero and resolve the new file to an existing pytest module.

- [ ] **Step 5: Commit the premerge enrollment**

```bash
git add tests/integration/test_lists/test-db/l0_a10.yml
git commit -s -m '[None][test] run telemetry manifest gate in premerge'
```

Expected: DCO and test-list validation hooks pass.

### Task 5: Run end-to-end validation and review the final branch

**Files:**
- Verify: all implementation files listed above
- Verify: `.github/CODEOWNERS:306`
- Test: generator, full telemetry-docs test module, pre-commit hook, stage mapper

- [ ] **Step 1: Run the complete telemetry manifest/docs test module**

Run in the prepared TensorRT-LLM environment:

```bash
python3 -m pytest \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py \
  -q -p no:cacheprovider
```

Expected: the full module passes, including generator failure behavior, builder recursion/domain tests, docs rendering, and the committed-golden gate.

- [ ] **Step 2: Re-run the supported check and hook interfaces**

```bash
python3 scripts/generate_llm_args_golden_manifest.py --check
pre-commit run generate-llm-args-golden-manifest --all-files
```

Expected: both exit zero and `git status --short` shows no new manifest modification.

- [ ] **Step 3: Reconfirm CI mapping and privacy ownership**

```bash
python3 scripts/test_to_stage_mapping.py \
  --tests unittest/usage/test_llmapi_config_telemetry_docs.py
rg -n 'llm_args_golden_manifest\.json' .github/CODEOWNERS
```

Expected:

```text
A10-PyTorch-1
A10-PyTorch-2
```

The CODEOWNERS search must show the manifest assigned to `@NVIDIA/trt-llm-oss-compliance`.

- [ ] **Step 4: Run repository checks over every changed implementation file**

```bash
pre-commit run --files \
  .pre-commit-config.yaml \
  scripts/generate_llm_args_golden_manifest.py \
  tensorrt_llm/usage/llm_args_golden_manifest.json \
  tensorrt_llm/usage/schemas/README.md \
  tests/integration/test_lists/test-db/l0_a10.yml \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py
git diff upstream/main...HEAD --check
```

Expected: all applicable hooks pass and the branch diff has no whitespace errors. If a hook changes a tracked file, review and stage that mechanical change, rerun the checks, and commit it with:

```bash
git add \
  .pre-commit-config.yaml \
  scripts/generate_llm_args_golden_manifest.py \
  tensorrt_llm/usage/llm_args_golden_manifest.json \
  tensorrt_llm/usage/schemas/README.md \
  tests/integration/test_lists/test-db/l0_a10.yml \
  tests/unittest/usage/test_llmapi_config_telemetry_docs.py
git commit -s -m '[None][chore] apply telemetry manifest validation fixes'
```

- [ ] **Step 5: Verify final scope and commit history**

```bash
git status --short --branch
git diff --stat upstream/main...HEAD
git log --oneline --decorate upstream/main..HEAD
```

Expected:

- The only untracked entry is the required `slop/` scratch symlink.
- Tracked changes are limited to the design/plan documents and the six implementation files in the file map.
- Every new implementation commit contains a DCO sign-off and one concern.
