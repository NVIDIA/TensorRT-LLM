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

import torch


class BenchmarkProfiler(object):
    cuda_event_dict: dict
    timer_dict: dict
    aux_info: dict
    started: bool

    def __init__(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}
        self.started = False

    def clean(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def get_cuda_event(self, name: str):
        if name not in self.cuda_event_dict.keys():
            event = torch.cuda.Event(enable_timing=True)
            self.cuda_event_dict[name] = event
        return self.cuda_event_dict[name]

    def record_cuda_event(self, name: str):
        if not self.started:
            return
        event = self.get_cuda_event(name)
        event.record()

    def get_timer_value(self, timer_name: str):
        # timer is in milliseconds
        return self.timer_dict[timer_name]

    def record_elapsed_time(self, start_event_name: str, end_event_name: str,
                            timer_name: str):
        if timer_name not in self.timer_dict.keys():
            self.timer_dict[timer_name] = 0.0
        if not self.started:
            return
        self.get_cuda_event(start_event_name).synchronize()
        self.get_cuda_event(end_event_name).synchronize()
        self.timer_dict[timer_name] += self.get_cuda_event(
            start_event_name).elapsed_time(self.get_cuda_event(end_event_name))

    def get_aux_info(self, aux_name):
        return self.aux_info[aux_name]

    def add_aux_info(self, aux_name: str, add_value):
        if aux_name not in self.aux_info.keys():
            self.aux_info[aux_name] = 0
        if not self.started:
            return
        self.aux_info[aux_name] += add_value
