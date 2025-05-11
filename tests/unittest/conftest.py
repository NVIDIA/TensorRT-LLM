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
import pytest
import tqdm


def pytest_configure(config):
    # avoid thread leak of tqdm's TMonitor
    tqdm.tqdm.monitor_interval = 0


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    yield

    import sys
    for m in sys.modules:
        if m == 'torch' or m.startswith('torch.'):
            import gc
            import os

            import torch
            worker_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', 1))

            if (torch.cuda.memory_reserved(0) + torch.cuda.memory_allocated(0)
                ) >= (torch.cuda.get_device_properties(0).total_memory //
                      worker_count) * 0.9:
                gc.collect()
                print("torch.cuda.memory_allocated: %fGB" %
                      (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
                print("torch.cuda.memory_reserved: %fGB" %
                      (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
                print("torch.cuda.max_memory_reserved: %fGB" %
                      (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

                torch.cuda.empty_cache()
            break
