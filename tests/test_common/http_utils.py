import subprocess
import time

import requests


def wait_for_endpoint_ready(url: str, timeout: int = 300, server_proc: subprocess.Popen = None):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if server_proc is not None:
            exit_code = server_proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Server process exited with code {exit_code} before becoming ready."
                )
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
