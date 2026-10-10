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

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

import click

from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_hmac_key_env
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient
from tensorrt_llm.llmapi.utils import print_colored
from tensorrt_llm.sampling_params import SamplingParams


def _check_client_sys_path(client: RemoteMpiCommSessionClient) -> None:
    from _mpi_session_test_tasks import receive_logits_processor

    client.SYNC_IDLE_INTERVAL = 0.1
    worker_pids = client.submit_sync(os.getpid)
    assert len(set(worker_pids)) == 2, worker_pids
    with tempfile.TemporaryDirectory(
            prefix="trtllm-client-module-") as directory:
        module_name = "client_only_logits_processor"
        Path(directory, f"{module_name}.py").write_text(
            "from tensorrt_llm.sampling_params import LogitsProcessor\n"
            "class ForceTokenLogitsProcessor(LogitsProcessor):\n"
            "    def __init__(self, token_id):\n"
            "        self.token_id = token_id\n"
            "    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id):\n"
            "        logits.fill_(float('-inf'))\n"
            "        logits[..., self.token_id] = 0\n",
            encoding="utf-8")
        worker_specs = client.submit_sync(importlib.util.find_spec, module_name)
        assert worker_specs == [None, None], worker_specs
        queues = [ZeroMqQueue(is_server=True) for _ in range(2)]
        original_path = sys.path.copy()
        try:
            # Only the client learns this path, after MPI workers have started.
            sys.path.append(directory)
            module = importlib.import_module(module_name)
            params = SamplingParams(
                max_tokens=1,
                logits_processor=module.ForceTokenLogitsProcessor(22))
            addresses = [(queue.address_endpoint, queue.hmac_key)
                         for queue in queues]
            client.submit(receive_logits_processor, addresses)
            for queue in queues:
                queue.put(params)
            results = [queue.get(timeout=15) for queue in queues]
            assert results == [(0, 22), (1, 22)], results
            assert len(client.submit_sync(os.getpid)) == 2
        finally:
            sys.path[:] = original_path
            sys.modules.pop(module_name, None)
            for queue in queues:
                queue.close()


@click.command()
@click.option("--task_type",
              type=click.Choice([
                  "submit", "submit_sync", "flashinfer_workspace",
                  "flashinfer_temporary_cleanup", "task_kwargs",
                  "client_sys_path"
              ]),
              default="submit")
def main(
    task_type: Literal["submit", "submit_sync", "flashinfer_workspace",
                       "flashinfer_temporary_cleanup", "task_kwargs",
                       "client_sys_path"]
) -> None:
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
        elif task_type in ("submit_sync", "flashinfer_temporary_cleanup"):
            res = client.submit_sync(print_colored, f"{task}\n", "green")
            print(res)
        elif task_type == "task_kwargs":
            expected = {"client_sys_path": "task-owned value"}
            assert client.submit_sync(dict, **expected) == [expected, expected]
        elif task_type == "client_sys_path":
            _check_client_sys_path(client)
        elif task_type == "flashinfer_workspace":
            workspaces = set(
                client.submit_sync(os.getenv, "FLASHINFER_WORKSPACE_BASE"))
            cubin_dirs = set(
                client.submit_sync(os.getenv, "FLASHINFER_CUBIN_DIR"))
            assert None not in workspaces
            assert len(workspaces) == 2
            workspace_root = (Path.home() / ".cache" / "tensorrt_llm" /
                              "flashinfer")
            assert all(
                Path(workspace).parent == workspace_root
                for workspace in workspaces)
            # Unset means FlashInfer derives the artifact cache from each
            # worker's isolated workspace, keeping downloaded compiler inputs
            # per-rank.
            assert cubin_dirs == {None}


if __name__ == "__main__":
    main()
