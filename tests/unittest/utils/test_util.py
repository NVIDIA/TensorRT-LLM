# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import pytest
import torch

from .util import (DeviceSleepCtl, assert_no_cuda_sync, device_sleep,
                   force_ampere)


@force_ampere
@pytest.mark.parametrize(
    "cancel",
    [False, True],
)
def test_device_sleep(cancel: bool):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    sleep_ctl = DeviceSleepCtl()
    sleep_time = 0.3

    start_event.record()
    device_sleep(sleep_time, ctl=sleep_ctl, spin_s=0.01)
    end_event.record()

    if cancel:
        sleep_ctl.cancel()
    end_event.synchronize()
    # NB: torch.cuda.Event.elapsed_time returns millis
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    if cancel:
        assert elapsed_time < sleep_time
    else:
        assert elapsed_time >= sleep_time


@force_ampere
@pytest.mark.parametrize(
    "uut_syncs",
    [False, True],
)
def test_assert_no_cuda_sync(uut_syncs: bool):

    def _uut():
        if uut_syncs:
            torch.cuda.synchronize()

    ctx = pytest.raises(AssertionError, match="sync code should return quickly"
                        ) if uut_syncs else nullcontext()
    with ctx:
        with assert_no_cuda_sync(sync_timeout_s=0.2):
            _uut()
