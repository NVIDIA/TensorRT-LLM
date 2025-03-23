import os
import subprocess  # nosec B404
import time

from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored


def main():
    tasks = [0]
    client = RemoteMpiCommSessionClient(
        os.environ['TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'])
    for task in tasks:
        client.submit(print_colored, f"{task}\n", "green")
    time.sleep(10)
    client.shutdown()


if __name__ == "__main__":
    main()
