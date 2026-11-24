# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The autotuner's random dummy inputs: a large dummy is filled in slices.

A dummy is drawn as ``(torch.rand(shape) * 10 - 5).to(dtype)``; in one call that
holds two float32 temporaries of the dummy's full size at once. An attention-DP
context server with 16 ranks of 16384 tokens tunes its NVFP4 MoE FC2 (top 16 of
896 experts, 56 per rank, tile 128) at 4,201,344 permuted rows of 1536 bytes:
a 6 GiB activation whose one-call staging is 2 x 24 GiB. Dummies of more than
``_DUMMY_ONE_CALL_MAX_NUMEL`` elements are allocated in their own dtype and
filled in slices of ``_DUMMY_SLICE_NUMEL`` elements; smaller ones keep the one
call and its values. CPU only: the full-size shapes run on the meta device.
"""

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, DynamicDim, StaticDim

pytestmark = pytest.mark.cpu_only

ONE_CALL_MAX = AutoTuner._DUMMY_ONE_CALL_MAX_NUMEL
SLICE = AutoTuner._DUMMY_SLICE_NUMEL
FP4 = torch.float4_e2m1fn_x2

# The FC2 profile at the top bucket (262,144 received tokens): the tile-padded
# permuted rows, and the tokens they hold (the helper's infer_num_tokens).
FC2_ROWS = (262144 * 16 + 127 * 56) // 128 * 128
FC2_TOKENS = (FC2_ROWS - 127 * 55) // 16


def _tuner() -> AutoTuner:
    # The dummy path reads only class constants: skip the CUDA-facing __init__.
    return AutoTuner.__new__(AutoTuner)


def _origin(shape, dtype, device):
    if dtype == FP4:
        return torch.empty(shape, dtype=torch.uint8, device=device).view(FP4)
    return torch.empty(shape, dtype=dtype, device=device)


def _dims(shape):
    return [DynamicDim(shape[0], shape[0], shape[0])] + [StaticDim(s) for s in shape[1:]]


def _create(shape, dtype, device="cpu"):
    return _tuner()._create_tensor_like(_origin(shape, dtype, device), _dims(shape))


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == FP4 else t


@pytest.fixture
def rand_numels(monkeypatch):
    """Record the element count of every torch.rand call."""
    numels = []
    real_rand = torch.rand

    def rand(*args, **kwargs):
        out = real_rand(*args, **kwargs)
        numels.append(out.numel())
        return out

    monkeypatch.setattr(torch, "rand", rand)
    return numels


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((FC2_ROWS, 1536), FP4),  # the activation: 6.01 GiB
        ((FC2_ROWS * 192,), torch.uint8),  # its block scales: 0.75 GiB
        ((FC2_TOKENS, 3584), torch.bfloat16),  # the finalize output: 1.75 GiB
    ],
    ids=["activation", "block_scales", "output"],
)
def test_dep16_fc2_top_bucket_dummies_are_filled_in_slices(rand_numels, shape, dtype):
    out = _create(shape, dtype, device="meta")
    numel = out.numel()
    assert out.shape == torch.Size(shape) and out.dtype == dtype
    assert numel > ONE_CALL_MAX
    # Every element drawn once, never more than one slice per draw: the float32
    # staging is two slices (512 MiB), not two copies of the whole dummy.
    assert sum(rand_numels) == numel
    assert max(rand_numels) <= SLICE
    assert 2 * 4 * max(rand_numels) <= 512 << 20 < 2 * 4 * numel


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((ONE_CALL_MAX // 1024, 1024), torch.uint8),  # exactly the limit
        # A TEP8 context server's FC2 top bucket (16384 tokens, 112 experts per rank).
        (((16384 * 16 + 127 * 112) // 128 * 128, 1536), FP4),
        ((16384, 7168), torch.bfloat16),
    ],
    ids=["limit", "tep8_fc2_activation", "dense_16k"],
)
def test_dummies_up_to_the_limit_are_drawn_in_one_call(rand_numels, shape, dtype):
    out = _create(shape, dtype, device="meta")
    assert out.shape == torch.Size(shape) and out.dtype == dtype
    assert rand_numels == [out.numel()]


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((64, 48), torch.bfloat16),
        ((64, 48), torch.float32),
        ((96, 32), FP4),
        ((3000,), torch.uint8),
        ((257,), torch.int32),
    ],
    ids=lambda v: str(v),
)
def test_small_dummies_keep_their_values(shape, dtype):
    torch.manual_seed(1234)
    staged = torch.rand(shape) * 10 - 5
    expected = staged.to(torch.uint8).view(FP4) if dtype == FP4 else staged.to(dtype)
    torch.manual_seed(1234)
    out = _create(shape, dtype)
    assert out.dtype == dtype and out.shape == torch.Size(shape)
    assert torch.equal(_bytes(out), _bytes(expected))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8, FP4], ids=str)
def test_a_sliced_dummy_draws_each_slice_as_one_call(monkeypatch, rand_numels, dtype):
    monkeypatch.setattr(AutoTuner, "_DUMMY_ONE_CALL_MAX_NUMEL", 1000)
    monkeypatch.setattr(AutoTuner, "_DUMMY_SLICE_NUMEL", 256)
    shape = (37, 50)  # 1850 elements: seven slices of 256 and one of 58
    torch.manual_seed(99)
    out = _create(shape, dtype)
    assert rand_numels == [256] * 7 + [58]
    assert out.dtype == dtype and out.shape == torch.Size(shape) and out.is_contiguous()
    torch.manual_seed(99)
    cast = torch.uint8 if dtype == FP4 else dtype
    expected = torch.cat([(torch.rand(n) * 10 - 5).to(cast) for n in [256] * 7 + [58]])
    assert torch.equal(_bytes(out).reshape(-1), expected)
