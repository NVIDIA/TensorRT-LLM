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
"""``k3_l2_prefetch``: byte ranges of dense tensors."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)


def _mod():
    from tensorrt_llm._torch.kernels.blackwell import k3_l2_prefetch

    return k3_l2_prefetch


def test_tensor_range():
    m = _mod()
    w = torch.empty(256, 384, dtype=torch.float8_e4m3fn, device="cuda")
    assert m.tensor_range(w) == (w.data_ptr(), 256 * 384)
    # DeepGEMM-packed scale: int32 [N, words] with strides (1, N), dense in memory.
    packed = torch.empty(3, 256, dtype=torch.int32, device="cuda").t()
    assert m.tensor_range(packed) == (packed.data_ptr(), 3 * 256 * 4)
    # sizes are rounded inwards to 16 bytes
    odd = torch.empty(37, dtype=torch.bfloat16, device="cuda")
    addr, nbytes = m.tensor_range(odd)
    assert addr == odd.data_ptr() and nbytes == 64
    # a strided slice is not dense
    with pytest.raises(ValueError):
        m.tensor_range(w[:, :128])
