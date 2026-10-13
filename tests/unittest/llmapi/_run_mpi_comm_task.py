# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal

import click

from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_hmac_key_env
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored


@click.command()
@click.option("--task_type",
              type=click.Choice(["submit", "submit_sync"]),
              default="submit")
def main(task_type: Literal["submit", "submit_sync"]) -> None:
    """Run the requested remote MPI session test task."""
    # TODO(dlfw-26.08): drop once the nested-spawn failure is settled. The DVM
    # that MPI_Comm_spawn starts inherits this process's environment, so this is
    # the environment that decides whether PRRTE will fork as root and which
    # hostname PMIx hands out. The shell that launched mpirun dumps the same
    # variables; a difference between the two means mpirun dropped them.
    mpi_env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("PRTE_", "OMPI_", "PMIX_", "PMI_"))
    }
    print(
        f"[pid {os.getpid()}] MPI environment in rank process: "
        f"{mpi_env or '(none set)'}",
        flush=True)

    tasks = [0]
    assert os.environ[
        'TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'] is not None, "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set"
    hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
    client = RemoteMpiCommSessionClient(
        os.environ['TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'], hmac_key=hmac_key)
    for task in tasks:
        if task_type == "submit":
            client.submit(print_colored, f"{task}\n", "green")
        elif task_type == "submit_sync":
            res = client.submit_sync(print_colored, f"{task}\n", "green")
            print(res)


if __name__ == "__main__":
    main()
