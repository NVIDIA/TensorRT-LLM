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

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.native.bounce import gather_scatter


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gather_scatter._HAVE_TRITON,
    reason="Bounce metadata reuse requires CUDA and Triton",
)
def test_scatter_metadata_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gather_scatter, "_meta_buffers", {})
    monkeypatch.setattr(gather_scatter, "prefer_pinned", lambda: True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(32, dtype=torch.uint8, device="cuda").reshape(2, 16)
        destination = torch.full_like(source, 255)
    sizes, offsets = np.array([16], dtype=np.int64), np.array([0], dtype=np.int64)

    # Warm the kernel and metadata allocation before delaying the next H2D copy.
    gather_scatter.scatter_contiguous(
        source[0].data_ptr(),
        np.array([destination[0].data_ptr()], dtype=np.int64),
        sizes,
        offsets,
        stream=stream.cuda_stream,
    )
    stream.synchronize()
    try:
        with torch.cuda.stream(stream):
            destination.fill_(255)
            torch.cuda._sleep(200_000_000)
        for src, dst in zip(source, destination, strict=True):
            gather_scatter.scatter_contiguous(
                src.data_ptr(),
                np.array([dst.data_ptr()], dtype=np.int64),
                sizes,
                offsets,
                stream=stream.cuda_stream,
            )
    finally:
        stream.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)
