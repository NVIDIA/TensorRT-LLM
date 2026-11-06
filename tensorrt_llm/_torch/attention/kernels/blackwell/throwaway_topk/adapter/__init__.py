# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental MQA TopK backends; imported only on explicit use."""

import functools
import threading


@functools.cache
def _continuous_plan_type(backend):
    outcome = []

    def load():
        try:
            # Load capture helpers without retaining caller tensors.
            from . import _capture_status  # noqa: F401

            from .selective_prefill import _CapacitySelectiveTopKPlan as Plan

            outcome.append(Plan)
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(
        target=load, name="flashinfer-continuous-topk-definitions"
    )
    thread.start()
    thread.join()
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]
