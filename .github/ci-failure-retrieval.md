# Retrieving CI Test Failures from a PR

CI tests run on internal NVIDIA Jenkins infrastructure (`blossom-ci`). To retrieve failed test cases from a PR:

## Step 1: Get the Jenkins build number from PR comments

The CI bot (`tensorrt-cicd`) posts comments with links to the Jenkins build. Extract the `L0_MergeRequest_PR` build number:
```bash
PR_NUM=<pr_number>
BUILD_NUM=$(gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --jq \
  '[.[] | select(.user.login == "tensorrt-cicd") | select(.body | test("L0_MergeRequest_PR"))] | last | .body' \
  | grep -oP 'L0_MergeRequest_PR/\K\d+')
```

## Step 2: Query the Jenkins testReport API for failures

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

## Step 3 (if needed): Get full stdout/stderr for a specific failure

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

## Available fields per failed test case (from Jenkins testReport API)
- `className`, `name`: test identifier
- `status`: `FAILED` or `REGRESSION`
- `errorDetails`: error message
- `errorStackTrace`: full stack trace (may be incomplete for subprocess errors)
- `stdout`, `stderr`: full test output (can be large, check these when stack trace is insufficient)
