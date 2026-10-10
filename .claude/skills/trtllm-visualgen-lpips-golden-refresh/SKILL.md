---
name: trtllm-visualgen-lpips-golden-refresh
description: Refresh TensorRT-LLM VisualGen LPIPS golden images or videos after an NVBug, VisualGen LPIPS CI failure, model/dependency version update, or other intentional output change, including triage, rerunning affected tests, updating per-family golden media zips, keeping JSON sha256/provenance metadata honest, and unwaiving tests only after validated passing runs.
argument-hint: "[nvbug, VisualGen LPIPS failure, dependency update, or model family]"
tags:
  - trtllm
  - visualgen
  - lpips
  - golden
license: Apache-2.0
---

# /trtllm-visualgen-lpips-golden-refresh

Refresh VisualGen LPIPS goldens only when the generated output change is
intentional and reproducible. Invoke this workflow from the incident or change
that motivates the refresh: an NVBug, a VisualGen LPIPS CI failure, a dependency
or model version update, or another expected change in generated VisualGen
artifacts. A pytest node id is useful after triage, but it is not the primary
trigger for the skill.

VisualGen LPIPS goldens are regression baselines. Tests generate new image or
video artifacts, compute the LPIPS score against the checked-in baseline, and
pass only when the score stays below the JSON threshold. A score above the
threshold can be a real regression, but it can also be an expected output drift
from a dependency update, model checkpoint update, runtime change, or intentional
pipeline behavior change. This skill is for proving which case applies and, only
when the drift is accepted, refreshing the baseline.

This workflow is for golden media under:

```text
tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/
```

The goal is not "make LPIPS pass" by overwriting files. The goal is to preserve
an auditable reference artifact for an intended TRT-LLM VisualGen output.

## Ground Rules

- Start from a clean branch in the TensorRT-LLM repo and record the exact commit
  used to generate media before any later rebase.
- Run generation in the same kind of environment the test is meant to cover:
  matching GPU class, container image, locally built wheel when code changed,
  and required runtime tools such as `ffmpeg` for MP4 outputs.
- Refresh only the affected media. Do not regenerate unrelated families just
  because they live in the same zip directory.
- Keep generated artifacts and logs under scratch while working so failures can
  be inspected later.
- Never update `tensorrt_llm_commit` to the current branch tip just because the
  branch was rebased. That field is generation provenance: it should be the
  TRT-LLM commit that actually produced the media bytes.

## Triage the Trigger

Start from the reported NVBug, CI failure, dependency update, or model family.
Identify what changed and why the old baseline may no longer be valid before
touching media:

- For an NVBug or CI failure, capture the failing job, pytest node, generated
  artifact, LPIPS score, threshold, model family, feature flag, and hardware.
- For a dependency or model update, identify the exact package, checkpoint,
  container, or runtime change and whether it is expected to alter generated
  artifacts.
- If there is no plausible intentional cause, treat the LPIPS failure as a
  regression investigation rather than a golden refresh.
- If the output is visually corrupted, wrong resolution, missing audio, or
  produced by a failed run, do not refresh the golden.

After that triage, map the affected case to exact pytest nodes and golden JSON
files.

Useful searches:

```bash
rg -n "lpips_against_golden|feature_accuracy_against_golden" \
  tests/integration/defs/examples/visual_gen
rg -n "visual_gen_lpips|_lpips_golden" \
  tests/integration/defs/examples/visual_gen
```

Each golden JSON names one media file using either `image` or `video`, and
stores the expected SHA-256 in `sha256`. The JSON metadata also records prompt,
shape, steps, seed, feature flags, package versions, container image, and
`tensorrt_llm_commit`.

## Generate Candidates

Run the affected tests in the intended container, not from a partially configured
host shell. Use a persistent temp directory so failure artifacts survive:

```bash
pytest <test node id> --basetemp /path/to/scratch/visualgen_lpips_refresh_tmp
```

When an LPIPS comparison fails, the test utilities preserve generated candidates
under an `lpips_failure_artifacts/` directory. If a test does not preserve the
needed output, rerun it with an explicit output directory or add temporary local
instrumentation outside the committed change; do not commit debug plumbing.

Inspect the generated image or video before accepting it as a golden. For video,
confirm it decodes, has the expected resolution/frame count, and keeps audio if
the workload includes audio:

```bash
ffprobe -v error -show_streams <candidate.mp4>
```

## Update Media and Metadata

The tests extract golden media from per-family zip files named:

```text
visual_gen_lpips_golden_media_<family>.zip
```

Family routing is defined in
`tests/integration/defs/examples/visual_gen/visual_gen_test_utils.py` by
`_golden_media_family()`. Keep the archive member names exactly equal to the
JSON `image` or `video` value, except directory-valued goldens such as
`qwen_image_layered_lpips_golden/`.

Use this mapping unless the source code changes it:

| Media prefix | Zip family |
|---|---|
| `cosmos3_` | `cosmos3` |
| `fastwan_`, `wan21_`, `wan22_` | `wan` |
| `flux1_`, `flux2_` | `flux` |
| `glm_image_` | `glm_image` |
| `hunyuan_` | `hunyuan` |
| `ltx2_` | `ltx2` |
| `qwenimage_`, `qwen_image_layered_` | `qwen_image` |

For each regenerated media file:

1. Replace only that member in the matching extracted family tree.
2. Recreate the family zip with relative paths, no absolute paths, and no `..`
   members.
3. Update the matching JSON `sha256` to the SHA-256 of the new media bytes.
4. Update `tensorrt_llm_commit` only for media that was actually regenerated,
   and set it to the commit used for generation.

Do not update `tensorrt_llm_commit` for media that was only repacked into a new
zip without changing bytes.

Example archive checks:

```bash
python -m zipfile -t tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/visual_gen_lpips_golden_media_<family>.zip
unzip -Z1 tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/visual_gen_lpips_golden_media_<family>.zip
jq empty tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/<golden>.json
```

## Validate

Rerun every affected single-GPU test in the container after the JSON and zip are
updated. If a test still fails, preserve the generated artifact and log path in
the task notes instead of overwriting again blindly.

If the refresh makes waived tests pass, update
`tests/integration/test_lists/waives.txt` in a separate, easy-to-review change.
Keep waives for any tests that still fail or that were not validated.

Before committing, run the repo's normal changed-file pre-commit command. For
TensorRT-LLM branches this is commonly:

```bash
SKIP=type-check,clang-format,ruff-legacy pre-commit run
```

Use `pre-commit run -a` only when the user or repo workflow explicitly asks for
a full-tree check.

## Commit Hygiene

Keep commits reviewable:

- One commit for mechanical zip layout changes, if any.
- One commit for regenerated media plus JSON checksum/provenance updates.
- One commit for waive-list changes, if any.

Commit messages should make clear whether media bytes changed or only archive
layout changed. Sign off commits when contributing to TensorRT-LLM.
