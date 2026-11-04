# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a private native HTTP completion from inside an isolated container."""

import ipaddress
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def validate_address(address: str) -> str:
    """Validate the numeric loopback endpoint retained across process restore.

    Args:
        address: IPv4 address and port published by the native serving launcher.

    Returns:
        The validated endpoint without surrounding whitespace.

    Raises:
        ValueError: If the endpoint is not a numeric IPv4 loopback and valid port.
    """
    address = address.strip()
    host, port = address.split(":")
    if not ipaddress.IPv4Address(host).is_loopback or not port.isdecimal():
        raise ValueError("Expected a numeric loopback endpoint")
    if not 1 <= int(port) <= 65535:
        raise ValueError("Invalid loopback port")
    return address


def wait_ready(opener: urllib.request.OpenerDirector, base: str, timeout: float) -> None:
    """Wait for HTTP 200 within a deadline, including non-error non-200 replies.

    Args:
        opener: Private, proxy-free HTTP client.
        base: Validated loopback URL.
        timeout: Total readiness timeout in seconds.

    Raises:
        TimeoutError: If the server does not become ready before the deadline.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(
                base + "/health", timeout=min(3, max(0.01, deadline - time.monotonic()))
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError("Private server readiness deadline")


def reference() -> None:
    """Verify the upstream workload's semantic results and captured-memory token.

    Raises:
        RuntimeError: If any answer or the captured-memory identity is incorrect.
    """
    config = json.loads(Path("/snapshot-control/experiment.json").read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for country, city in (("France", "paris"), ("Germany", "berlin"), ("Japan", "tokyo")):
        request = urllib.request.Request(
            "http://127.0.0.1:8000/generate",
            data=json.dumps({"prompt": f"The capital city of {country} is"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=60) as response:
            result = json.load(response)
        if (
            result.get("checkpoint_token") != config["template"]
            or city not in result.get("text", "").lower()
        ):
            raise RuntimeError("Upstream reference semantic/token check failed")
        results.append(result)
    print(json.dumps({"status": "PASS", "outputs": results}))


def main() -> None:
    """Wait for HTTP readiness, then validate one deterministic multi-token response.

    Raises:
        RuntimeError: If the response is incomplete or cannot exercise decoding.
        TimeoutError: If the private HTTP server never becomes ready.
    """
    address = validate_address(Path("/snapshot-control/address").read_text())
    base = "http://" + address
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    wait_ready(opener, base, 90)
    payload = {
        "model": "/model",
        "prompt": "Count the integers from one to ten:",
        "max_tokens": 16,
        "temperature": 0,
        "seed": 42,
        "stream": False,
    }
    start = time.monotonic()
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=60) as response:
        result = json.load(response)
    if (
        len(result.get("choices", [])) != 1
        or not result["choices"][0].get("text")
        or result["choices"][0].get("finish_reason") not in ("stop", "length")
        or result.get("usage", {}).get("completion_tokens", 0) < 2
    ):
        raise RuntimeError("Completion did not validate multi-token decoding")
    print(
        json.dumps(
            {
                "request_seconds": time.monotonic() - start,
                "completion": {
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0]["finish_reason"],
                    "usage": result["usage"],
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--reference"]:
        reference()
    else:
        main()
