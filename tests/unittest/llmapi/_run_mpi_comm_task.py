import os
from typing import Literal

import click

from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored


@click.command()
@click.option("--task_type",
              type=click.Choice(["submit", "submit_sync"]),
              default="submit")
def main(task_type: Literal["submit", "submit_sync"]):
    tasks = [0]
    assert os.environ[
        'TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'] is not None, "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set"
    client = RemoteMpiCommSessionClient(
        os.environ['TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'])
    for task in tasks:
        if task_type == "submit":
            client.submit(print_colored, f"{task}\n", "green")
        elif task_type == "submit_sync":
            res = client.submit_sync(print_colored, f"{task}\n", "green")
            print(res)


if __name__ == "__main__":
    main()
