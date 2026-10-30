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
"""trtllm::rotate_rows_ against torch.roll: same sign convention, in place, any shift."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  # registers the trtllm:: ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.float32,
        torch.int64,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize("shape", [(7,), (1, 1), (3, 1), (8, 1235), (5, 64), (3, 0), (0, 4)])
@pytest.mark.parametrize("shift", [0, 1, 3, 63, 64, 65, -1, -1300, 1300])
def test_matches_torch_roll(dtype, shape, shift):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, device="cuda").to(dtype)
    # torch.roll lacks some dtypes (fp8); the reference rolls the raw bytes instead.
    raw = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64, 16: torch.complex128}
    as_raw = x.view(raw[x.element_size()])
    expected = torch.roll(as_raw, shift, dims=-1).view(dtype)
    torch.ops.trtllm.rotate_rows_(x, shift)
    torch.cuda.synchronize()
    torch.testing.assert_close(x.view(raw[x.element_size()]), expected.view(raw[x.element_size()]))


def test_strided_rows_view():
    """A column slice of a wider table, the way a cache rotates its ring pages."""
    table = torch.arange(8 * 20, device="cuda", dtype=torch.int32).view(8, 20)
    before = table.clone()
    ring = table[:, 5:]  # row stride 20, 15 columns
    torch.ops.trtllm.rotate_rows_(ring, -4)
    torch.cuda.synchronize()
    torch.testing.assert_close(table[:, :5], before[:, :5])  # untouched prefix
    torch.testing.assert_close(table[:, 5:], torch.roll(before[:, 5:], -4, dims=1))


def test_rejects_bad_inputs():
    x = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.rotate_rows_(x.t(), 1)  # last dim not contiguous
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.rotate_rows_(x.cpu(), 1)


def test_capturable_in_cuda_graph():
    x = torch.arange(3 * 10, device="cuda", dtype=torch.int64).view(3, 10)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.ops.trtllm.rotate_rows_(x, 1)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.rotate_rows_(x, 1)
    start = x.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(x, torch.roll(start, 1, dims=1))


def test_overlapping_rows_are_refused():
    base = torch.arange(20, device="cuda", dtype=torch.int32)
    overlapping = base.as_strided((3, 8), (4, 1))  # row i shares 4 elements with row i+1
    with pytest.raises(RuntimeError, match="rows overlap"):
        torch.ops.trtllm.rotate_rows_(overlapping, 1)


def test_traces_under_torch_compile():
    """The fake registration lets dynamo trace the op without a graph break."""
    x = torch.arange(64, device="cuda", dtype=torch.int32)
    expected = torch.roll(x, -5)

    @torch.compile(fullgraph=True, backend="aot_eager")
    def rotate(t):
        torch.ops.trtllm.rotate_rows_(t, -5)
        return t

    torch.testing.assert_close(rotate(x), expected)


def test_more_rows_than_one_grid_dimension_holds():
    """Rows beyond the 65535 the launch grid's y dimension allows are rotated too."""
    x = torch.arange(70_000 * 4, device="cuda", dtype=torch.int32).view(70_000, 4)
    expected = torch.roll(x, 1, dims=-1)
    torch.ops.trtllm.rotate_rows_(x, 1)
    torch.testing.assert_close(x, expected)


def test_extreme_shift():
    x = torch.arange(10, device="cuda", dtype=torch.int64)
    shift = -(2**63)
    expected = torch.roll(x, shift % 10)
    torch.ops.trtllm.rotate_rows_(x, shift)
    torch.testing.assert_close(x, expected)
