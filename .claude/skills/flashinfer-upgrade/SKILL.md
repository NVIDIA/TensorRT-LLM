---
name: flashinfer-upgrade
description: >-
  Upgrade flashinfer-python version in TensorRT-LLM. Fetches the latest releases
  from GitHub (stable and nightly), compares with the current pinned version,
  lets the user pick a target version, and updates all version references across
  the repo. Use when the user wants to bump or upgrade flashinfer.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# FlashInfer Version Upgrade Skill

Automates upgrading the `flashinfer-python` package version across TensorRT-LLM.

## When to Use

- User asks to upgrade / bump / update flashinfer
- Routine dependency update duty for flashinfer-python

## Prerequisites

### Step 0a: Determine GitHub Username

Query `gh` for the authenticated user's login:
```bash
GITHUB_USERNAME=$(gh api user --jq .login)
echo "$GITHUB_USERNAME"
```

If this fails, `gh` is not authenticated — resolve Step 0c first, then retry.
As a fallback, derive the username from the fork remote:
```bash
GITHUB_USERNAME=$(git remote -v | grep -E 'github\.com/[^/]+/TensorRT-LLM' \
  | head -1 | sed -E 's|.*github\.com[:/]([^/]+)/TensorRT-LLM.*|\1|')
```
If neither works, ask the user via `AskUserQuestion`.

### Step 0b: Verify Fork Remote

Check that a git remote pointing to the user's fork of TensorRT-LLM exists:

```bash
git remote -v | grep -E 'github\.com/${GITHUB_USERNAME}/TensorRT-LLM'
```

If **no fork remote** is found, stop and notify the user:

> No GitHub fork remote detected. A fork of `NVIDIA/TensorRT-LLM` is required
> to push branches and create PRs.
>
> 1. Fork the repo at https://github.com/NVIDIA/TensorRT-LLM/fork
> 2. Add it as a git remote:
>    ```bash
>    git remote add fork https://github.com/<GITHUB_USERNAME>/TensorRT-LLM.git
>    ```
> 3. Re-run this skill.

### Step 0c: Verify `gh` CLI Is Authenticated

This skill uses the GitHub CLI (`gh`) to push branches and open PRs. Confirm it is
installed and authenticated:

```bash
gh auth status
```

Expected: `Logged in to github.com` with at least the `repo` scope. `repo` covers
pushing to the user's fork and opening PRs on `NVIDIA/TensorRT-LLM`, so no
separate fine-grained PATs are needed.

If `gh` reports "not logged in", instruct the user:

> ```bash
> gh auth login
> ```
>
> Choose: GitHub.com → HTTPS → authenticate with a web browser (or paste a PAT
> with `repo` scope).

**Note on `GH_CONFIG_DIR`:** If the user keeps multiple `gh` accounts (e.g. a
personal account and a separate account for `NVIDIA/TensorRT-LLM` work), they may
point `gh` at a non-default config directory. Check `CLAUDE.local.md` /
`AGENTS.md` or the environment for `GH_CONFIG_DIR`; if unclear, ask the user.
When set, prefix every `gh` invocation: `GH_CONFIG_DIR=<path> gh ...`.

Do **not** proceed with the upgrade workflow until `gh auth status` is clean and
the fork remote (Step 0b) is confirmed.

## Workflow

Execute these steps **in order**. Use `AskUserQuestion` for user choices and
`WebFetch` / GitHub API for release data.

### Step 1: Fetch Available Releases from GitHub

Fetch the release list from `https://github.com/flashinfer-ai/flashinfer/releases`.

Use `WebFetch` with the URL `https://github.com/flashinfer-ai/flashinfer/releases`
and extract all release tag names and dates. Collect both stable releases
(e.g., `v0.6.7`) and pre-release / nightly tags (e.g., `v0.7.0.dev20260401`).

Alternatively, use the GitHub API via curl:
```bash
curl -s "https://api.github.com/repos/flashinfer-ai/flashinfer/releases?per_page=30" \
  | python3 -c "
import json, sys
releases = json.load(sys.stdin)
for r in releases:
    tag = r['tag_name']
    pre = ' (pre-release)' if r['prerelease'] else ' (stable)'
    date = r['published_at'][:10]
    print(f'{tag}  {date}{pre}')
"
```

### Step 2: Check Current Version

Read the current pinned version from `requirements.txt`:
```bash
grep flashinfer-python requirements.txt
```
Expected format: `flashinfer-python==X.Y.Z`

### Step 3: Ask User Preferences

Ask the user **two questions** using `AskUserQuestion`:

1. **"Prefer a latest nightly release version?"**
   - Options: "Yes, show nightly/dev releases" | "No, stable releases only (Recommended)"
   - This filters the release list shown in the next question.

2. **"Which flashinfer-python version do you want to upgrade to?"**
   - Present up to 4 versions newer than the current version (filtered by
     the nightly preference above), with the latest as the recommended option.
   - If the current version is already the latest, inform the user and stop.

### Step 4: Update All Version References

After the user selects a target version, update **all** of these files:

| File | What to change |
|------|---------------|
| `requirements.txt` | `flashinfer-python==OLD` → `flashinfer-python==NEW` |
| `security_scanning/pyproject.toml` | `"flashinfer-python (==OLD)"` → `"flashinfer-python (==NEW)"` |
| `security_scanning/poetry.lock` | Update `version = "OLD"` → `version = "NEW"` under `[[package]] name = "flashinfer-python"`, and update the `files` list with new hashes |
| `ATTRIBUTIONS-Python.md` | `## flashinfer-python (OLD)` → `## flashinfer-python (NEW)` |

#### Updating `security_scanning/poetry.lock` hashes

The poetry.lock file contains SHA256 hashes for the wheel and sdist. Fetch them
from PyPI:
```bash
curl -s "https://pypi.org/pypi/flashinfer-python/NEW_VERSION/json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data['urls']:
    print(f'{f[\"filename\"]}  sha256:{f[\"digests\"][\"sha256\"]}')
"
```

Replace the old `files = [...]` block under `[[package]] name = "flashinfer-python"`
with the new filenames and hashes. Also update the `[package.dependencies]` section
if the new version has different dependencies (check PyPI JSON `requires_dist`).

**Important**: After manually editing both `security_scanning/pyproject.toml` and
`security_scanning/poetry.lock`, the lockfile's `metadata.content-hash` becomes stale.
Regenerate it by running:

```bash
cd security_scanning && poetry lock --no-update && cd ..
```

This refreshes the hash without changing any other package versions. If `poetry` is
available, you can alternatively use `poetry add flashinfer-python@NEW_VERSION` in the
`security_scanning/` directory to update both `pyproject.toml` and `poetry.lock`
automatically (including the content-hash).

#### Nightly / dev version special handling

If the user selects a nightly/dev version (e.g., `0.7.0.dev20260401`):
- The PyPI package may not exist — check first with `curl -s "https://pypi.org/pypi/flashinfer-python/VERSION/json"`.
- If not on PyPI, the `security_scanning/poetry.lock` hashes cannot be updated.
  Warn the user and leave a `# TODO: update hashes when published to PyPI` comment.
- The `requirements.txt` can pin to a git install instead:
  `flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer.git@TAG#egg=flashinfer-python`
  Ask the user which approach they prefer (PyPI pin vs git pin).

### Step 5: Verify Version Compatibility

After updating, check if any code has version-gated logic that needs adjusting:

```bash
grep -rn 'flashinfer.*__version__\|flashinfer.*version' \
  tensorrt_llm/ --include="*.py"
```

Known locations with version checks:
- `tensorrt_llm/_torch/speculative/interface.py` — `flashinfer.__version__ >= "0.6.4"`

If the new version is still >= the gated version, no changes needed. Otherwise, flag
to the user.

### Step 6: Summary

Print a summary of all changes made:
- Old version → New version
- Files modified (with line numbers)
- Any warnings (e.g., poetry.lock hashes couldn't be updated for nightly)
- Remind user to run `pip install -r requirements.txt` to test locally
- Remind user to run relevant unit tests:
  ```bash
  pytest tests/unittest/_torch/flashinfer/ -v
  pytest tests/unittest/_torch/attention/test_flashinfer_attention.py -v
  ```

### Step 7: Commit, Push, and Create PR

After all files are updated and verified:

#### 7a. Create a new branch from upstream main

```bash
git stash push -m "flashinfer-upgrade-wip" -- requirements.txt security_scanning/pyproject.toml security_scanning/poetry.lock ATTRIBUTIONS-Python.md
git checkout main
git pull --rebase https://github.com/NVIDIA/TensorRT-LLM.git main
git checkout -b ${GITHUB_USERNAME}/update_flashinfer_${NEW_VERSION}
git stash pop
```

Where `GITHUB_USERNAME` comes from the fork remote (e.g., `yihwang-nv`) and
`NEW_VERSION` is the selected version (e.g., `0.6.7.post3`).

#### 7b. Commit with DCO sign-off

```bash
git add requirements.txt security_scanning/pyproject.toml security_scanning/poetry.lock ATTRIBUTIONS-Python.md
git commit -s -m "[None][chore] Update flashinfer-python from OLD to NEW

Bump flashinfer-python dependency to the latest stable release.
Updated version pins in requirements.txt, security_scanning/pyproject.toml,
security_scanning/poetry.lock, and ATTRIBUTIONS-Python.md."
```

#### 7c. Push the branch to the user's fork

Identify the fork remote (from Step 0b — commonly named `fork`), then push:

```bash
FORK_REMOTE=fork   # adjust if the user named their fork remote differently
BRANCH="${GITHUB_USERNAME}/update_flashinfer_${NEW_VERSION}"
git push -u "${FORK_REMOTE}" "${BRANCH}"
```

If the push is rejected for auth reasons, confirm `gh auth status` shows `repo`
scope — `gh` installs a git credential helper that reuses its token for HTTPS
pushes. Users on a non-default config dir must export `GH_CONFIG_DIR` in the
same shell.

#### 7d. Open the PR on `NVIDIA/TensorRT-LLM`

```bash
gh pr create \
  --repo NVIDIA/TensorRT-LLM \
  --base main \
  --head "${GITHUB_USERNAME}:${BRANCH}" \
  --title "[None][chore] Update flashinfer-python from ${OLD_VERSION} to ${NEW_VERSION}" \
  --body "$(cat <<EOF
## Summary
- Bump flashinfer-python from ${OLD_VERSION} to ${NEW_VERSION} (latest stable)
- Updated version pins in requirements.txt, security_scanning/pyproject.toml, security_scanning/poetry.lock, and ATTRIBUTIONS-Python.md

## Test plan
- [ ] pip install -r requirements.txt installs successfully
- [ ] pytest tests/unittest/_torch/flashinfer/ -v
- [ ] pytest tests/unittest/_torch/attention/test_flashinfer_attention.py -v
- [ ] CI pre-merge passes
EOF
)"
```

`gh pr create` prints the new PR URL on success. Report it back to the user.

## Files Reference

All files that contain flashinfer-python version pins:

| File | Pattern |
|------|---------|
| `requirements.txt` | `flashinfer-python==X.Y.Z` |
| `security_scanning/pyproject.toml` | `"flashinfer-python (==X.Y.Z)"` |
| `security_scanning/poetry.lock` | `name = "flashinfer-python"` block with version + hashes |
| `ATTRIBUTIONS-Python.md` | `## flashinfer-python (X.Y.Z)` |

## Notes

- The `setup.py` has a comment about git+https install URLs — no version pin to update there.
- The `.pre-commit-config.yaml` and `pyproject.toml` reference flashinfer source files, not versions — no changes needed.
- The `flashinfer/` submodule (if present) is separate from the `flashinfer-python` PyPI package.
