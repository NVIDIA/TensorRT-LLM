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

### Step 0a: Verify Fork Remote

Check that a git remote pointing to the user's fork of TensorRT-LLM exists:

```bash
git remote -v | grep -E 'github\.com/[^/]+/TensorRT-LLM'
```

Look for a remote (typically named `fork` or `origin`) with a URL like
`https://github.com/<GITHUB_USERNAME>/TensorRT-LLM.git`. Extract `GITHUB_USERNAME`
from this URL — it is used in later steps for branch pushing and PR creation.

If **no fork remote** is found, stop and notify the user:

> No GitHub fork remote detected. A fork of `NVIDIA/TensorRT-LLM` is required
> to push branches and create PRs.
>
> 1. Fork the repo at https://github.com/NVIDIA/TensorRT-LLM/fork
> 2. Add it as a git remote:
>    ```bash
>    git remote add fork https://github.com/<YOUR_GITHUB_USERNAME>/TensorRT-LLM.git
>    ```
> 3. Re-run this skill.

### Step 0b: Verify GitHub Access Tokens

This skill requires two GitHub personal access tokens to push branches and create PRs.
Check both tokens exist in the environment:

```bash
echo "USE_GH_TOKEN: ${USE_GH_TOKEN:+set (${#USE_GH_TOKEN} chars)}"
echo "NVIDIA_GH_TOKEN: ${NVIDIA_GH_TOKEN:+set (${#NVIDIA_GH_TOKEN} chars)}"
```

| Token | Purpose | Required Permissions |
|-------|---------|---------------------|
| `USE_GH_TOKEN` | Push branches to the user's fork | Scoped to `<GITHUB_USERNAME>/TensorRT-LLM` with **Contents: Read and write** and **Pull requests: Read and write** |
| `NVIDIA_GH_TOKEN` | Create PRs on the upstream repo | Scoped to `NVIDIA/TensorRT-LLM` with **Pull requests: Read and write** |

If either token is **missing or empty**, stop and guide the user through creating them:

> One or more GitHub tokens are not configured.
>
> This skill uses **fine-grained personal access tokens** (recommended over
> classic tokens for security — they are scoped to specific repositories and
> permissions, and expire automatically).
>
> Follow the steps below to create them.
>
> ### How to create `USE_GH_TOKEN` (fork access)
>
> 1. Go to https://github.com/settings/personal-access-tokens/new
>    (Settings -> Developer settings -> Personal access tokens -> **Fine-grained tokens** -> Generate new token)
> 2. **Token name**: enter a descriptive name, e.g. `trtllm-fork-push`
> 3. **Expiration**: choose an expiration (e.g. 90 days)
> 4. **Resource owner**: select your GitHub account
> 5. **Repository access**: select **"Only select repositories"**, then pick
>    `<YOUR_GITHUB_USERNAME>/TensorRT-LLM`
> 6. **Permissions** — expand **"Repository permissions"** and set:
>    - **Contents**: **Read and write** (required to push branches)
>    - **Pull requests**: **Read and write**
> 7. Click **"Generate token"** and copy the token (starts with `github_pat_`)
>
> ### How to create `NVIDIA_GH_TOKEN` (upstream PR access)
>
> 1. Go to https://github.com/settings/personal-access-tokens/new
>    (Settings -> Developer settings -> Personal access tokens -> **Fine-grained tokens** -> Generate new token)
> 2. **Token name**: enter a descriptive name, e.g. `trtllm-upstream-pr`
> 3. **Expiration**: choose an expiration (e.g. 90 days)
> 4. **Resource owner**: select **NVIDIA** (you must be a member of the NVIDIA org)
> 5. **Repository access**: select **"Only select repositories"**, then pick
>    `NVIDIA/TensorRT-LLM`
> 6. **Permissions** — expand **"Repository permissions"** and set:
>    - **Pull requests**: **Read and write** (required to create PRs on the upstream repo)
> 7. Click **"Generate token"** and copy the token (starts with `github_pat_`)
>
> **Note**: Fine-grained tokens may require approval from the organization admin
> if the org has token policies enabled. If your token request is pending, contact
> your org admin to approve it.
>
> ### Export the tokens
>
> Add them to your shell environment before running this skill:
> ```bash
> export USE_GH_TOKEN=github_pat_...
> export NVIDIA_GH_TOKEN=github_pat_...
> ```
>
> To persist across sessions, add the exports to your shell profile (e.g.
> `~/.bashrc`, `~/.zshrc`) or a credentials script you source manually.

Do **not** proceed with the upgrade workflow until the fork remote and both tokens
are confirmed available.

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

#### 7c. Push branch via Git Data API

Direct `git push` with fine-grained PATs often fails. Use the GitHub Git Data API
with `USE_GH_TOKEN` instead:

```python
# 1. Create blobs for each changed file (upload via urllib for large files)
# 2. Create a tree with base_tree = parent tree, overriding the 4 files
# 3. Create a commit referencing the tree and parent
# 4. Create a ref (branch) pointing to the commit
```

API base: `https://api.github.com/repos/{GITHUB_USERNAME}/TensorRT-LLM/git`

For large files (e.g., `ATTRIBUTIONS-Python.md` ~3MB), use Python `urllib` to POST
the blob instead of `curl` to avoid argument-length limits.

#### 7d. Create PR via GitHub API

Use `NVIDIA_GH_TOKEN` to create the PR on upstream:

```python
import json, urllib.request, os

token = os.environ["NVIDIA_GH_TOKEN"]
pr_data = {
    "title": "[None][chore] Update flashinfer-python from OLD to NEW",
    "head": "GITHUB_USERNAME:BRANCH_NAME",
    "base": "main",
    "body": "## Summary\n- Bump flashinfer-python from OLD to NEW (latest stable)\n- Updated version pins in requirements.txt, security_scanning/pyproject.toml, security_scanning/poetry.lock, and ATTRIBUTIONS-Python.md\n\n## Test plan\n- [ ] pip install -r requirements.txt installs successfully\n- [ ] pytest tests/unittest/_torch/flashinfer/ -v\n- [ ] pytest tests/unittest/_torch/attention/test_flashinfer_attention.py -v\n- [ ] CI pre-merge passes\n"
}

payload = json.dumps(pr_data).encode()
req = urllib.request.Request(
    "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls",
    data=payload,
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    },
    method="POST"
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(f"PR created: {result['html_url']}")
```

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
