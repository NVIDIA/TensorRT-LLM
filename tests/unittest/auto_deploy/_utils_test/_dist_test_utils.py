# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
import torch


def get_device_counts(num_gpu_list=None):
    num_gpu_list = [1, 2] if num_gpu_list is None else num_gpu_list
    return [param_with_device_count(n) for n in num_gpu_list]


def param_with_device_count(n: int, *args, marks_extra=None):
    gpu_count = torch.cuda.device_count()
    marks = [pytest.mark.skipif(gpu_count < n, reason=f"need {n} GPUs!")]
    marks.extend(marks_extra or [])
    return pytest.param(n, *args, marks=marks)
