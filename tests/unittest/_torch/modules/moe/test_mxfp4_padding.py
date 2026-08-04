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

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.quantization import maybe_pad_for_mxfp4


@pytest.mark.parametrize(
    ("shape", "col_alignment", "row_alignment"),
    [
        ((6, 8), 4, 3),
        ((8,), 4, None),
    ],
)
def test_maybe_pad_for_mxfp4_reuses_aligned_tensor(
    shape: tuple[int, ...], col_alignment: int, row_alignment: int | None
) -> None:
    weight = torch.arange(torch.Size(shape).numel(), dtype=torch.uint8).reshape(shape)

    padded = maybe_pad_for_mxfp4(weight, col_alignment, row_alignment)

    assert padded is weight


def test_maybe_pad_for_mxfp4_pads_unaligned_tensor() -> None:
    weight = torch.arange(15, dtype=torch.uint8).reshape(3, 5)

    padded = maybe_pad_for_mxfp4(weight, col_alignment=4, row_alignment=4)

    assert padded.shape == (4, 8)
    torch.testing.assert_close(padded[:3, :5], weight)
    assert torch.count_nonzero(padded[3, :]) == 0
    assert torch.count_nonzero(padded[:, 5:]) == 0
