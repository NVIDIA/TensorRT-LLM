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


def test_copy_of_device_media_is_complete_when_it_returns(monkeypatch):
    """A device clone is queued on the caller's stream; the worker reuses the slot
    with no view of that stream. ``copy()`` returns only once its reads are done,
    so a slot overwritten right after the return cannot leak into the owned copy.
    The event recorded right behind the clone on that stream says whether the
    clone had finished when ``copy()`` returned; the same event, queried when the
    clone is issued, proves the clone was still waiting behind the pending work."""
    slot = torch.full((16 << 20,), 1, dtype=torch.uint8, device="cuda")
    result = SessionResult(1, 0, 0, {"video": slot}, {"video": (0, 1)}, False, None)
    worker = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)  # the caller's own pending work, ahead of the clone
    clone = torch.Tensor.clone
    cloned = []
    waiting_when_issued = []

    def clone_then_mark(tensor, *args, **kwargs):
        out = clone(tensor, *args, **kwargs)
        cloned.append(torch.cuda.Event())
        cloned[-1].record()
        waiting_when_issued.append(not cloned[-1].query())
        return out

    monkeypatch.setattr(torch.Tensor, "clone", clone_then_mark)
    keep = result.copy()
    monkeypatch.undo()
    assert waiting_when_issued == [True], "the clone did not wait behind the pending work"
    assert cloned[0].query(), "copy() returned before its clone finished"
    with torch.cuda.stream(worker):
        slot.fill_(2)  # the worker's next advance writing the reused slot
    torch.cuda.synchronize()
    assert bool((keep.outputs["video"] == 1).all())
    assert bool((result.outputs["video"] == 2).all())
