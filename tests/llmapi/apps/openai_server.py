# Adapted from
# https://github.com/vllm-project/vllm/blob/baaedfdb2d3f1d70b7dbcde08b083abfe6017a92/tests/utils.py
import os
import subprocess
import sys
from typing import List


from tensorrt_llm.llmapi.mpi_session import find_free_port


class RemoteOpenAIServer:
    DUMMY_API_KEY = "tensorrt_llm"
    MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 600 seconds

    def __init__(self,
                 model: str,
                 cli_args: List[str],
                 llmapi_launch: bool = False,
                 port: int = None) -> None:
        self.host = "localhost"
        self.port = port if port is not None else find_free_port()
        self.rank = os.environ.get("SLURM_PROCID", 0)

        cli_args += ["--host", f"{self.host}", "--port", f"{self.port}"]
        launch_cmd = ["trtllm-serve"] + [model] + cli_args
        if llmapi_launch:
            # start server with llmapi-launch on multi nodes
            launch_cmd = ["trtllm-llmapi-launch"] + launch_cmd
        self.proc = subprocess.Popen(launch_cmd,
                                     stdout=sys.stdout,
                                     stderr=sys.stderr)
        self._wait_for_server(url=self.url_for("health"),
                              timeout=self.MAX_SERVER_START_WAIT_S)
