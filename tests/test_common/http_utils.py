import os
import subprocess
import time

import requests

ERROR_KEYWORDS = ["RuntimeError", "out of memory", "ValueError"]


def wait_for_endpoint_ready(
    url: str,
    timeout: int = 300,
    check_files: list[str] | None = None,
    server_proc: subprocess.Popen = None,
):
    start = time.monotonic()
    iteration = 0
    while time.monotonic() - start < timeout:
        if server_proc is not None:
            exit_code = server_proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Server process exited with code {exit_code} before becoming ready."
                )
        iteration += 1
        if check_files and iteration % 300 == 0:
            for check_file in check_files:
                if os.path.exists(check_file):
                    try:
                        with open(check_file, "r") as f:
                            content = f.read()
                        for line in content.splitlines():
                            for keyword in ERROR_KEYWORDS:
                                if keyword in line:
                                    raise RuntimeError(
                                        f"Found '{keyword}' in server file {check_file}: {line}"
                                    )
                    except RuntimeError:
                        raise
                    except Exception as err:
                        print(f"Failed to read server file {check_file}: {err}")
        try:
            time.sleep(1)
            if requests.get(url, timeout=5).status_code == 200:
                print(f"endpoint {url} is ready")
                return
        except Exception as err:
            print(f"endpoint {url} is not ready, with exception: {err}")
    raise RuntimeError(f"Endpoint {url} did not become ready within {timeout} seconds")


def wait_for_endpoint_down(url: str, timeout: int = 300):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code >= 100:
                print(f"endpoint {url} returned status code {requests.get(url).status_code}")
                time.sleep(1)
        except Exception as err:
            print(f"endpoint {url} is down, with exception: {err}")
            return
    raise RuntimeError(f"Endpoint {url} did not become down within {timeout} seconds")
