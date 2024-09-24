# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# # Force resource release after test
import gc
import multiprocessing.connection
import os
import sys
import time

import pytest

memory_profiling_enabled = os.environ.get("LLM_MEMORY_PROFILING", False)

if memory_profiling_enabled:

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(session):
        import xdist
        session.stash["reporter"] = multiprocessing.connection.Client(
            "/tmp/profiling_scribe.unix", "AF_UNIX")
        session.stash["worker_id"] = xdist.get_xdist_worker_id(session)
        session.stash["reporter"].send({
            "type": "identity",
            "identifier": "unittest",
            "pid": os.getpid(),
            "worker_id": session.stash["worker_id"]
        })

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(session, config, items):
        for item in items:
            item.stash["reporter"] = session.stash["reporter"]
            item.stash["worker_id"] = session.stash["worker_id"]

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(session):
        session.stash["reporter"].close()


@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if memory_profiling_enabled:
        path, line, name = item.reportinfo()
        item.stash["reporter"].send({
            "type": "unit_case",
            "timestamp": time.time(),
            "case": {
                "path": str(path),
                "line": line,
                "name": name
            },
            "worker_id": item.stash["worker_id"],
            "pid": os.getpid()
        })

    result = yield

    if not any(module == 'torch' or module.startswith('torch.')
               for module in sys.modules):
        return result

    import torch

    if memory_profiling_enabled:
        item.stash["reporter"].send({
            "type": "torch_report",
            "timestamp": time.time(),
            "case": {
                "path": str(path),
                "line": line,
                "name": name
            },
            "context": "unit",
            "worker_id": item.stash["worker_id"],
            "pid": os.getpid(),
            "report": {
                "allocated": torch.cuda.memory_allocated(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_reserved": torch.cuda.max_memory_reserved(),
            }
        })

        torch.cuda.reset_peak_memory_stats()

    worker_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', 1))

    if (torch.cuda.memory_reserved(0) + torch.cuda.memory_allocated(0)
        ) >= (torch.cuda.get_device_properties(0).total_memory //
              worker_count) * 0.9:
        gc.collect()
        torch.cuda.empty_cache()

    return result
