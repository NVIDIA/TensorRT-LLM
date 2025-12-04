import os
from typing import Literal

import click

from tensorrt_llm.executor.utils import LlmLauncherEnvs
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored


def run_task(task_type: Literal["submit", "submit_sync"]):
    tasks = range(10)
    assert os.environ[
        LlmLauncherEnvs.
        TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR] is not None, "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set"
    client = RemoteMpiCommSessionClient(
        os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR])

    for task in tasks:
        if task_type == "submit":
            client.submit(print_colored, f"{task}\n", "green")
        elif task_type == "submit_sync":
            res = client.submit_sync(print_colored, f"{task}\n", "green")
            print(res)


def run_multi_tasks(task_type: Literal["submit", "submit_sync"]):
    for i in range(3):
        print_colored(f"Running MPI comm task {i}\n", "green")
        run_task(task_type)
        print_colored(f"MPI comm task {i} completed\n", "green")


@click.command()
@click.option("--task_type",
              type=click.Choice(["submit", "submit_sync"]),
              default="submit")
def main(task_type: Literal["submit", "submit_sync"]):
    run_multi_tasks(task_type)


if __name__ == "__main__":
    main()
