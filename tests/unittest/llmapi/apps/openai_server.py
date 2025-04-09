# Adapted from
# https://github.com/vllm-project/vllm/blob/baaedfdb2d3f1d70b7dbcde08b083abfe6017a92/tests/utils.py
import os
import subprocess
import sys
import time
from typing import List

import openai
import requests

from tensorrt_llm.llmapi.mpi_session import find_free_port


class RemoteOpenAIServer:
    DUMMY_API_KEY = "tensorrt_llm"
    MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 600 seconds

    def __init__(self,
                 model: str,
                 cli_args: List[str] = None,
                 llmapi_launch: bool = False,
                 port: int = None) -> None:
        self.host = "localhost"
        self.port = port if port is not None else find_free_port()
        self.rank = os.environ.get("SLURM_PROCID", 0)

        args = ["--host", f"{self.host}", "--port", f"{self.port}"]
        if cli_args:
            args += cli_args
        launch_cmd = ["trtllm-serve"] + [model] + args
        if llmapi_launch:
            # start server with llmapi-launch on multi nodes
            launch_cmd = ["trtllm-llmapi-launch"] + launch_cmd
        self.proc = subprocess.Popen(launch_cmd,
                                     stdout=sys.stdout,
                                     stderr=sys.stderr)
        self._wait_for_server(url=self.url_for("health"),
                              timeout=self.MAX_SERVER_START_WAIT_S)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired as e:
            self.proc.kill()
            self.proc.wait(timeout=30)

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check on the first rank only.
        start = time.time()
        while True:
            try:
                if self.rank == 0:
                    if requests.get(url).status_code == 200:
                        break
                else:
                    time.sleep(timeout)
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

    def get_async_client(self, **kwargs):
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY,
                                  **kwargs)
