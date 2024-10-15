# Adapted from
# https://github.com/vllm-project/vllm/blob/baaedfdb2d3f1d70b7dbcde08b083abfe6017a92/tests/utils.py
import os
import subprocess
import sys
import time
from typing import List

import openai
import requests

LLM_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SERVER_PATH = os.path.join(LLM_ROOT, "examples", "apps", "openai_server.py")


class RemoteOpenAIServer:
    DUMMY_API_KEY = "tensorrt_llm"
    MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds

    def __init__(
        self,
        model: str,
        cli_args: List[str],
    ) -> None:
        self.host = '0.0.0.0'
        self.port = 8000

        self.proc = subprocess.Popen(["python3", SERVER_PATH] + [model] +
                                     cli_args,
                                     stdout=sys.stdout,
                                     stderr=sys.stderr)
        self._wait_for_server(url=self.url_for("health"),
                              timeout=self.MAX_SERVER_START_WAIT_S)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    break
            except Exception as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self):
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
        )

    def get_async_client(self):
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
        )
