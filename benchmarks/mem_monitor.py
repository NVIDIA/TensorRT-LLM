# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

import pynvml


def get_memory_info(handle):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = round(mem_info.total / 1024 / 1024 / 1024, 2)
    used = round(mem_info.used / 1024 / 1024 / 1024, 2)
    free = round(mem_info.used / 1024 / 1024 / 1024, 2)
    return total, used, free


def mem_monitor(q1, q2):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    peak_used = 0
    while q1.empty():
        _, used, _ = get_memory_info(handle)
        peak_used = max(used, peak_used)
        time.sleep(0.1)

    pynvml.nvmlShutdown()
    q2.put(peak_used)
