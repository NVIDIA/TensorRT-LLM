# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for cache readback and reference-state isolation."""

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".", "_components/dense_block"]

import _dense_block_reference as reference  # noqa: E402
from _beam_component_support import TorchCallTrace, cache_rows, compare  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("num_layers,local_layer", [(1, 0), (2, 1), (3, 2)])
def test_cache_rows_reads_nonzero_slots(num_layers: int, local_layer: int) -> None:
    slots, heads, block, dim = 4, 2, 4, 3
    pool = torch.arange(slots * num_layers * 2 * heads * block * dim).reshape(
        slots, num_layers, 2, heads, block, dim
    )
    # get_buffers starts at the requested layer's K page; rows are K/V pairs.
    view = pool.flatten(0, 1)[local_layer:]

    def get_buffers(layer_idx: int, kv_layout: str) -> torch.Tensor:
        assert layer_idx == 7 and kv_layout == "HND"
        return view

    selected = [2, 0, 1]
    scale = 2 * num_layers
    offsets = torch.tensor(selected) * scale
    manager = SimpleNamespace(
        get_buffers=get_buffers, get_layer_page_index_scale=lambda layer: scale
    )
    md = SimpleNamespace(
        kv_cache_manager=manager, kv_cache_block_offsets=offsets.reshape(1, 1, 1, -1)
    )
    length = 2 * block + 2
    keys, values = cache_rows(md, length, 7, block)
    for actual, role in ((keys, 0), (values, 1)):
        expected = torch.cat([pool[slot, local_layer, role].permute(1, 0, 2) for slot in selected])[
            :length
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [17, 41])
def test_dense_reference_preserves_rng(seed: int) -> None:
    dims = reference.Dims(hidden=8, intermediate=16, heads=2, kv_heads=1, head_dim=4)
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state().clone()
        first = reference.generate_state(seed, dims)
        assert torch.equal(before, torch.get_rng_state())
        second = reference.generate_state(seed, dims)
        assert torch.equal(before, torch.get_rng_state())
        assert first.keys() == second.keys()
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "unequal,fraction,passed", [(1, 0.25, True), (2, 0.25, False), (1, 0.24, False)]
)
def test_compare_unequal_limit(unequal: int, fraction: float, passed: bool) -> None:
    expected = torch.zeros(4)
    actual = expected.clone()
    actual[:unequal] = 0.01  # Within absolute tolerance, but still unequal.
    result = compare(
        "fraction", actual, expected, "test", (1.0, 0.0), actual.dtype, max_unequal=fraction
    )
    assert result["passed"] is passed
    assert result["violations"] == 0


def test_projection_trace_retains_precision_and_outputs() -> None:
    x, w = torch.ones(1, 2), torch.ones(3, 2)
    with TorchCallTrace(projection=torch.nn.functional.linear) as trace:
        output = torch.nn.functional.linear(x, w)
    projection = next(call for call in trace.calls if "math" in call)
    assert projection["math"] == {
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "autocast_cuda": torch.is_autocast_enabled("cuda"),
    }
    assert projection["operand_dtypes"] == [str(x.dtype), str(w.dtype)]
    assert projection["outputs"] == [["cpu", str(output.dtype), [1, 3]]]
