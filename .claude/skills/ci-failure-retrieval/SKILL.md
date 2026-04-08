---
name: ci-failure-retrieval
description: Retrieve and diagnose CI test failures from TensorRT-LLM pull requests using the GitHub API and Jenkins testReport API. Use when the user asks about CI failures on a PR, wants to see failed test details, or needs stdout/stderr from a CI run.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# CI Failure Retrieval

**Input:** a PR number or a request to check CI failures. **Auth requirement:** requires corporate network access to resolve the Jenkins base URL. **Output:** a summary of failed tests with error details, and optionally full stdout/stderr for specific failures.

## Phase 1 — Get the Jenkins Build Number

The CI bot (`tensorrt-cicd`) posts comments with links to the Jenkins build. Extract the `L0_MergeRequest_PR` build number:
```bash
PR_NUM=<pr_number>
BUILD_NUM=$(gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --jq \
  '[.[] | select(.user.login == "tensorrt-cicd") | select(.body | test("L0_MergeRequest_PR"))] | last | .body' \
  | grep -oP 'L0_MergeRequest_PR/\K\d+')
```

## Phase 2 — Query the Jenkins testReport API for Failures

Resolve the Jenkins base URL dynamically from the internal shortcut (requires corporate network):
```bash
JENKINS_BASE="$(curl -skI 'https://nv/trt-llm-cicd' 2>/dev/null | grep -i '^location:' | sed 's/^[Ll]ocation: *//;s/[[:space:]]*$//')job/main/job/L0_MergeRequest_PR"
```

```bash
curl -s "${JENKINS_BASE}/${BUILD_NUM}/testReport/api/json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Summary: {data[\"passCount\"]} passed, {data[\"failCount\"]} failed, {data[\"skipCount\"]} skipped')
failed = []
for suite in data.get('suites', []):
    for case in suite.get('cases', []):
        if case.get('status') in ('FAILED', 'REGRESSION'):
            failed.append(case)
if not failed:
    print('No test failures!')
else:
    print(f'Failed tests ({len(failed)}):')
    for f in failed:
        print(f'  - {f[\"className\"]}.{f[\"name\"]}')
        err = (f.get('errorDetails') or '')[:200]
        if err:
            print(f'    Error: {err}')
"
```

## Phase 3 — Get Full stdout/stderr for a Specific Failure

The `errorStackTrace` can be incomplete when errors originate from subprocesses. In that case, fetch `stdout` and `stderr` for the specific test case to find the real error:
```bash
curl -s "${JENKINS_BASE}/${BUILD_NUM}/testReport/api/json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for suite in data.get('suites', []):
    for case in suite.get('cases', []):
        if case.get('status') in ('FAILED', 'REGRESSION'):
            name = f'{case[\"className\"]}.{case[\"name\"]}'
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
"
```

## Available Fields per Failed Test Case (Jenkins testReport API)

- `className`, `name`: test identifier
- `status`: `FAILED` or `REGRESSION`
- `errorDetails`: error message
- `errorStackTrace`: full stack trace (may be incomplete for subprocess errors)
- `stdout`, `stderr`: full test output (can be large, check these when stack trace is insufficient)

## Anti-Patterns

- Do not guess Jenkins URLs; always resolve dynamically via the internal shortcut.
- Do not stop at `errorStackTrace` if it mentions generic wrapper failures like `Process exited with status 1`; check `stdout` and `stderr` for the real error.
- Do not fetch all test cases when looking for a specific failure; use the `<search_term>` filter in Phase 3.
