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
"""``SessionResult`` on device-placed media: the owned copy against slot reuse."""

import pytest
import torch

from tensorrt_llm.visual_gen.session import SessionResult

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device media needs CUDA")


def test_copy_of_device_media_is_complete_when_it_returns():
    """A device clone is queued on the caller's stream; the worker reuses the slot
    with no view of that stream. ``copy()`` returns only once its reads are done,
    so a slot overwritten right after the return cannot leak into the owned copy."""
    slot = torch.full((16 << 20,), 1, dtype=torch.uint8, device="cuda")
    result = SessionResult(1, 0, 0, {"video": slot}, {"video": (0, 1)}, False, None)
    worker = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)  # the caller's own pending work, ahead of the clone
    pending = torch.cuda.Event()
    pending.record()
    keep = result.copy()
    assert pending.query(), "copy() returned with its source read still queued"
    with torch.cuda.stream(worker):
        slot.fill_(2)  # the worker's next advance writing the reused slot
    torch.cuda.synchronize()
    assert bool((keep.outputs["video"] == 1).all())
    assert bool((result.outputs["video"] == 2).all())
