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
"""VSA predictor kernels (tile + cube mean, row sort, coarse/fine blend) against references.

Every kernel is checked on CUDA (Triton path) and CPU (PyTorch fallback) over cube layouts
with ragged fills and head layouts whose row width is not a power of two.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.kernels import (
    blend_coarse_fine,
    sort_last_dim,
    tile_and_pool_cubes,
)

CUBE_SIZE = 64

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_HEAD_LAYOUTS = [
    pytest.param(6, 4, 32, id="row128"),
    pytest.param(5, 5, 21, id="row105_odd"),
    pytest.param(30, 40, 128, id="wan14b_row5120"),
    pytest.param(3, 64, 128, id="row8192"),
]


def _random_cube_layout(num_cubes: int, device: str, generator: torch.Generator):
    """Assign consecutive compact tokens to cubes with a random fill per cube.

    Returns the padded-slot source index (-1 for pad slots), the valid count per cube, the
    padded slot of every compact token, and the compact sequence length.
    """
    counts = torch.randint(1, CUBE_SIZE + 1, (num_cubes,), generator=generator)
    counts[0] = CUBE_SIZE
    counts[-1] = CUBE_SIZE // 2
    seq_len = int(counts.sum())
    tile_source_index = torch.full((num_cubes * CUBE_SIZE,), -1, dtype=torch.long)
    untile_index = torch.empty(seq_len, dtype=torch.long)
    token = 0
    for cube, count in enumerate(counts.tolist()):
        slots = torch.arange(cube * CUBE_SIZE, cube * CUBE_SIZE + count)
        tile_source_index[slots] = torch.arange(token, token + count)
        untile_index[token : token + count] = slots
        token += count
    return tile_source_index.to(device), counts.to(device), untile_index.to(device), seq_len


def _reference_tile_and_pool(x, tile_source_index, counts, num_cubes):
    batch, _, heads, head_dim = x.shape
    tiled = x.index_select(1, tile_source_index.clamp(min=0))
    valid = (tile_source_index >= 0).view(1, -1, 1, 1)
    tiled = torch.where(valid, tiled, torch.zeros((), dtype=x.dtype, device=x.device))
    pooled = tiled.view(batch, num_cubes, CUBE_SIZE, heads, head_dim).float().sum(dim=2)
    pooled = pooled / counts.view(1, -1, 1, 1).float()
    return tiled, pooled.to(x.dtype)


def _reference_blend(fine, coarse, gate_compress, gate_fine, untile_index, fine_is_tiled):
    coarse_per_token = coarse.index_select(1, untile_index // CUBE_SIZE)
    fine_compact = fine.index_select(1, untile_index) if fine_is_tiled else fine
    if gate_fine is not None:
        fine_compact = gate_fine * fine_compact
    return gate_compress * coarse_per_token + fine_compact


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("num_cubes", "heads", "head_dim"), _HEAD_LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_tile_and_pool_cubes_matches_reference(device, num_cubes, heads, head_dim, dtype):
    generator = torch.Generator().manual_seed(num_cubes * 31 + heads)
    source, counts, _untile, seq_len = _random_cube_layout(num_cubes, device, generator)
    x = torch.randn(2, seq_len, heads, head_dim, device=device, dtype=dtype)

    tiled, pooled = tile_and_pool_cubes(x, source, counts, cube_size=CUBE_SIZE)

    ref_tiled, ref_pooled = _reference_tile_and_pool(x, source, counts, num_cubes)
    assert tiled.shape == (2, num_cubes * CUBE_SIZE, heads, head_dim)
    assert pooled.shape == (2, num_cubes, heads, head_dim)
    assert torch.equal(tiled, ref_tiled)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(pooled, ref_pooled, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device", _DEVICES)
def test_tile_and_pool_cubes_accepts_strided_sequence_layout(device):
    """Q/K/V unbound from a packed [B, S, 3, H, D] tensor must work without a copy."""
    source, counts, _untile, seq_len = _random_cube_layout(
        8, device, torch.Generator().manual_seed(7)
    )
    packed = torch.randn(2, seq_len, 3, 4, 32, device=device, dtype=torch.float16)
    q = packed[:, :, 1]
    assert not q.is_contiguous()

    tiled, pooled = tile_and_pool_cubes(q, source, counts, cube_size=CUBE_SIZE)

    ref_tiled, ref_pooled = _reference_tile_and_pool(q.contiguous(), source, counts, 8)
    assert torch.equal(tiled, ref_tiled)
    torch.testing.assert_close(pooled, ref_pooled, rtol=1e-3, atol=1e-3)


def test_tile_and_pool_cubes_rejects_split_head_dims():
    source, counts, _untile, seq_len = _random_cube_layout(
        2, "cpu", torch.Generator().manual_seed(1)
    )
    x = torch.randn(1, seq_len, 32, 4).transpose(2, 3)

    with pytest.raises(ValueError, match="contiguous"):
        tile_and_pool_cubes(x, source, counts, cube_size=CUBE_SIZE)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("row_length", [1, 30, 144, 257, 2048, 5000])
def test_sort_last_dim_matches_torch_sort(device, row_length):
    generator = torch.Generator().manual_seed(row_length)
    values = torch.randint(0, 4096, (2, 3, 5, row_length), generator=generator, dtype=torch.int32)
    values = values.to(device)

    assert torch.equal(sort_last_dim(values), torch.sort(values, dim=-1).values)


@pytest.mark.parametrize("device", _DEVICES)
def test_blend_coarse_fine_reads_head_major_fine_output(device):
    """Fine output stored as [B, H, S, D] (the CuTe layout) is consumed through its strides."""
    num_cubes, batch, heads, head_dim = 6, 2, 4, 32
    source, _counts, untile, seq_len = _random_cube_layout(
        num_cubes, device, torch.Generator().manual_seed(11)
    )
    fine = torch.randn(batch, heads, num_cubes * CUBE_SIZE, head_dim, device=device).transpose(1, 2)
    assert not fine.is_contiguous()
    coarse = torch.randn(batch, num_cubes, heads, head_dim, device=device)
    gate_compress = torch.randn(batch, seq_len, heads, head_dim, device=device)

    out = blend_coarse_fine(
        fine, coarse, gate_compress, None, untile, cube_size=CUBE_SIZE, fine_is_tiled=True
    )

    ref = _reference_blend(fine, coarse, gate_compress, None, untile, True)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_sort_last_dim_requires_int32():
    with pytest.raises(TypeError, match="int32"):
        sort_last_dim(torch.zeros(2, 4, dtype=torch.int64))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fine_is_tiled", [True, False], ids=["tiled_fine", "compact_fine"])
@pytest.mark.parametrize("with_gate_fine", [True, False], ids=["gate_fine", "no_gate_fine"])
@pytest.mark.parametrize(
    ("heads", "head_dim", "dtype"),
    [(5, 21, torch.float32), (40, 128, torch.bfloat16)],
    ids=["row105_fp32", "wan14b_bf16"],
)
def test_blend_coarse_fine_matches_reference(
    device, fine_is_tiled, with_gate_fine, heads, head_dim, dtype
):
    num_cubes, batch = 6, 2
    source, _counts, untile, seq_len = _random_cube_layout(
        num_cubes, device, torch.Generator().manual_seed(3)
    )
    fine_len = num_cubes * CUBE_SIZE if fine_is_tiled else seq_len
    fine = torch.randn(batch, fine_len, heads, head_dim, device=device, dtype=dtype)
    coarse = torch.randn(batch, num_cubes, heads, head_dim, device=device, dtype=dtype)
    gate_compress = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    gate_fine = torch.randn_like(gate_compress) if with_gate_fine else None

    out = blend_coarse_fine(
        fine,
        coarse,
        gate_compress,
        gate_fine,
        untile,
        cube_size=CUBE_SIZE,
        fine_is_tiled=fine_is_tiled,
    )

    ref = _reference_blend(fine, coarse, gate_compress, gate_fine, untile, fine_is_tiled)
    assert out.shape == gate_compress.shape
    tolerance = 1e-6 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out, ref, rtol=tolerance, atol=tolerance)
