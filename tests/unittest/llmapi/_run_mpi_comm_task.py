# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Literal

import click

from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_hmac_key_env
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored


def get_flashinfer_workspace() -> tuple[str | None, str | None]:
    """Return the FlashInfer workspace and cubin cache configured for this rank."""
    return (os.environ.get("FLASHINFER_WORKSPACE_BASE"),
            os.environ.get("FLASHINFER_CUBIN_DIR"))


@click.command()
@click.option("--task_type",
              type=click.Choice(
                  ["submit", "submit_sync", "flashinfer_workspace"]),
              default="submit")
def main(
    task_type: Literal["submit", "submit_sync",
                       "flashinfer_workspace"]) -> None:
    """Run the requested remote MPI session test task."""
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
        elif task_type == "flashinfer_workspace":
            worker_envs = client.submit_sync(get_flashinfer_workspace)
            workspaces = {workspace for workspace, _ in worker_envs}
            cubin_dirs = {cubin_dir for _, cubin_dir in worker_envs}
            assert None not in workspaces
            assert len(workspaces) == 2
            workspace_root = (Path.home() / ".cache" / "tensorrt_llm" /
                              "flashinfer")
            assert all(
                Path(workspace).parent == workspace_root
                for workspace in workspaces)
            assert cubin_dirs == {
                str(Path.home() / ".cache" / "flashinfer" / "cubins")
            }


if __name__ == "__main__":
    main()
