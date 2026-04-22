---
name: ci-failure-retrieval
description: Retrieve and diagnose CI test failures from TensorRT-LLM pull requests using the GitHub API and Jenkins testReport API. Use when the user asks about CI failures on a PR, wants to see failed test details, or needs stdout/stderr from a CI run.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# CI Failure Retrieval

**Input:** a PR number or a request to check CI failures. **Auth requirement:** requires corporate network access to resolve the Jenkins base URL. **Output:** a summary of failed tests with error details, and optionally full stdout/stderr for specific failures.

## Important: SSL and Authentication

Jenkins requires SSL with certificate verification disabled. Always use `ssl` context bypass in Python or `-sk` flags in curl:
```python
import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
```
The `curl -s` approach often returns HTML login pages; prefer the Python `urllib` approach with SSL bypass.

## Phase 0 — Get the Latest CI Run Info

First, determine the latest CI run commit, build number, and high-level pass/fail counts:

```bash
source ~/utils/github/set_github_token.sh

PR_NUM=<pr_number>

# Get the latest CI bot comment (contains build number and commit)
gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --paginate --jq \
  '[.[] | select(.user.login == "tensorrt-cicd") | select(.body | test("L0_MergeRequest_PR"))] | last | .body'

# Get the PR HEAD commit and its blossom-ci status (high-level pass/fail counts)
HEAD_SHA=$(gh api "repos/NVIDIA/TensorRT-LLM/pulls/${PR_NUM}" --jq '.head.sha')
gh api "repos/NVIDIA/TensorRT-LLM/commits/${HEAD_SHA}/statuses" --jq \
  '[.[] | select(.context == "blossom-ci")] | first | {state, description}'
```

The `description` field shows aggregate counts like `"23969 passed, 1 failed, 8962 skipped"`.

## Phase 1 — Get the Jenkins Build Number

Extract the `L0_MergeRequest_PR` build number from the CI bot comment:
```bash
BUILD_NUM=$(gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --paginate --jq \
  '[.[] | select(.user.login == "tensorrt-cicd") | select(.body | test("L0_MergeRequest_PR"))] | last | .body' \
  | grep -oP 'L0_MergeRequest_PR/\K\d+')
```

## Phase 1.5 — Check Pipeline Stage Failures (before diving into test details)

Many CI failures are **infrastructure-level** (Slurm node issues, pipeline aborts, resource exhaustion) where no test code executes at all. Always check the pipeline stages first:

```python
import json, ssl, urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

JENKINS_BASE = "https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_MergeRequest_PR"
BUILD_NUM = <build_number>

# Get pipeline stage overview
url = f"{JENKINS_BASE}/{BUILD_NUM}/wfapi/describe"
resp = urllib.request.urlopen(urllib.request.Request(url), context=ctx, timeout=30)
data = json.loads(resp.read())

print(f"Pipeline status: {data.get('status')}")
for stage in data.get('stages', []):
    status = stage.get('status', '')
    if status not in ('SUCCESS', 'SKIPPED', 'NOT_EXECUTED'):
        name = stage.get('name', '')
        print(f"  [{status}] {name}")
        if 'error' in stage:
            print(f"    Error: {stage['error']}")
```

## Phase 1.6 — Read Console Log Analysis (Most Valuable for Infrastructure Failures)

The Jenkins console log contains a **CI failure analysis summary** with sections like `## Recommended Actions` and `## Infrastructure Notes`. This is the single most valuable source for understanding infrastructure failures:

```python
url = f"{JENKINS_BASE}/{BUILD_NUM}/consoleText"
resp = urllib.request.urlopen(urllib.request.Request(url), context=ctx, timeout=30)
text = resp.read().decode('utf-8', errors='replace')

# Extract failure-related lines from the end of the log
for line in text[-8000:].split('\n'):
    lo = line.lower()
    if any(kw in lo for kw in ['fail', 'error', 'abort', 'likely cause',
                                'recommended action', 'infrastructure',
                                'no test code', 'stage result']):
        print(line.strip()[:300])
```

Key sections to look for in the console log:
- **`Failing job`** / **`Failed stage`**: which Jenkins sub-job and stage failed
- **`Likely cause`**: automated root cause analysis (Slurm issues, pipeline timeouts, etc.)
- **`No test code was executed`**: confirms infrastructure-only failure (no code fix needed)
- **`Recommended Actions`**: whether to re-trigger CI or investigate code changes

## Phase 2 — Query the Jenkins testReport API for Test Failures

Only proceed here if Phase 1.5/1.6 indicate actual test failures (not infrastructure issues):

```python
url = f"{JENKINS_BASE}/{BUILD_NUM}/testReport/api/json"
resp = urllib.request.urlopen(urllib.request.Request(url), context=ctx, timeout=30)
data = json.loads(resp.read())

print(f'Summary: {data["passCount"]} passed, {data["failCount"]} failed, {data["skipCount"]} skipped')

failed = []
for suite in data.get('suites', []):
    for case in suite.get('cases', []):
        if case.get('status') in ('FAILED', 'REGRESSION'):
            failed.append(case)

if not failed:
    print('No test failures in testReport!')
else:
    print(f'Failed tests ({len(failed)}):')
    for f in failed:
        print(f'  - {f["className"]}.{f["name"]}')
        err = (f.get('errorDetails') or '')[:200]
        if err:
            print(f'    Error: {err}')
```

## Phase 3 — Get Full stdout/stderr for a Specific Test Failure

The `errorStackTrace` can be incomplete when errors originate from subprocesses. Fetch `stdout` and `stderr` for the specific test case to find the real error:
```python
for suite in data.get('suites', []):
    for case in suite.get('cases', []):
        if case.get('status') in ('FAILED', 'REGRESSION'):
            name = f'{case["className"]}.{case["name"]}'
            if '<search_term>' in name:
                print(f'=== {name} ===')
                print('--- Error ---')
                print(case.get('errorDetails', ''))
                print('--- Stack Trace ---')
                print(case.get('errorStackTrace', ''))
                print('--- Stdout (last 3000 chars) ---')
                print((case.get('stdout') or '')[-3000:])
                print('--- Stderr (last 3000 chars) ---')
                print((case.get('stderr') or '')[-3000:])
                break
```

## Available Fields per Failed Test Case (Jenkins testReport API)

- `className`, `name`: test identifier
- `status`: `FAILED` or `REGRESSION`
- `errorDetails`: error message
- `errorStackTrace`: full stack trace (may be incomplete for subprocess errors)
- `stdout`, `stderr`: full test output (can be large, check these when stack trace is insufficient)

## Common Failure Patterns

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| `No test code was executed` + Slurm errors | Infrastructure: Slurm node resource exhaustion | Re-trigger CI |
| `ABORTED` stage + `Downstream job did not succeed` | Cascading failure from fail-fast policy | Fix root cause stage, re-trigger |
| `newosproc` / `errno=11` / `fork/exec` | Kernel process table exhaustion on login node | Wait and re-trigger |
| `testReport: 0 failed` but `blossom-ci: N failed` | Stage-level failures, not test failures | Check Phase 1.5/1.6 |
| `testReport: N failed` with real test names | Actual test code failures | Investigate test errors in Phase 3 |

## Anti-Patterns

- Do not guess Jenkins URLs; always use the known base `https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_MergeRequest_PR`.
- Do not use `curl -s` for Jenkins API; it returns HTML login pages. Use Python `urllib` with SSL bypass.
- Do not jump to testReport (Phase 2) before checking pipeline stages (Phase 1.5) — many failures are infrastructure-only with zero test failures.
- Do not stop at `errorStackTrace` if it mentions generic wrapper failures like `Process exited with status 1`; check `stdout` and `stderr` for the real error.
- Do not fetch all test cases when looking for a specific failure; use the `<search_term>` filter in Phase 3.
