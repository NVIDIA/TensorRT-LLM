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
"""Host <-> device page copies are exact for pages that span several CTAs."""

import pytest
import torch

copy_helpers = pytest.importorskip(
    "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
    reason="the device copy helpers are C++-backend only",
)

MemToMemTask = copy_helpers.MemToMemTask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# Page sizes around the 8 MiB per-CTA split and beyond the 16-CTA cap; the odd sizes leave a
# partial last CTA.
@pytest.mark.parametrize(
    "page_bytes",
    [16, 4096, (8 << 20) - 16, (8 << 20) + 16, 40 << 20, (200 << 20) + 48],
)
@pytest.mark.parametrize("num_pages", [1, 3])
def test_host_device_round_trip(page_bytes: int, num_pages: int) -> None:
    stream = torch.cuda.Stream()
    generator = torch.Generator().manual_seed(page_bytes + num_pages)
    host_src = torch.randint(
        0, 256, (num_pages, page_bytes), dtype=torch.uint8, generator=generator
    ).pin_memory()
    device = torch.full((num_pages, page_bytes), 0xA5, dtype=torch.uint8, device="cuda")
    host_dst = torch.zeros((num_pages, page_bytes), dtype=torch.uint8).pin_memory()

    to_device = [
        MemToMemTask(dst=device[i].data_ptr(), src=host_src[i].data_ptr()) for i in range(num_pages)
    ]
    to_host = [
        MemToMemTask(dst=host_dst[i].data_ptr(), src=device[i].data_ptr()) for i in range(num_pages)
    ]
    assert copy_helpers.copy_host_to_device(to_device, page_bytes, stream.cuda_stream) == 0
    assert copy_helpers.copy_device_to_host(to_host, page_bytes, stream.cuda_stream) == 0
    stream.synchronize()

    assert torch.equal(device.cpu(), host_src)
    assert torch.equal(host_dst, host_src)
