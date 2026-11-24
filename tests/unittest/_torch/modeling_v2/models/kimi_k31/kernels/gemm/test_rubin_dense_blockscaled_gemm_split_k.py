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
"""Host-side split-K check of the family's SM107 dense block-scaled GEMM runners
(``RubinBlockScaledGemmRunner.forward``, run through ``RubinMxfp8BaseGemmRunner`` and the
fine-grained ``RubinMxfp8GemmRunner`` of ``rubin_dense_blockscaled_gemm``).

A tactic whose split_k the kernel cannot run (K tiles that split_k does not divide, or FP8
output) raises ValueError before anything is allocated or launched; every other tactic goes on
through forward with what it carries. All but the last test use meta tensors and stop forward at
its first step after the check; the last one launches.
"""

import inspect

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU with a Rubin-capable CuTe DSL",
)

M, N, K = 256, 1024, 512  # MXFP8: 4 K tiles of 128
TILE = ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "m")
TABLE_TILE = ("base", (256, 256, 128), (256, 256, 64), (2, 1), False, False, "static", "m")
MIXED = ("mixed_clusters", (256, 256, 128), (256, 256, 64), (4, 2), (2, 1), False, False, "m")
# forward's view of a tactic, recorded where the check has passed.
PARSED = (
    "kernel_variant",
    "mma_tiler_mnk",
    "mma_inst_shape",
    "cluster_shape_mn",
    "swap_ab",
    "use_prefetch",
    "scheduler_mode",
    "raster_order",
    "split_k",
    "epi_tile_n",
    "m",
    "k",
    "n",
)


class _Reached(Exception):
    """forward got past the split-K check; ``parsed`` is its view of the tactic there."""

    def __init__(self, parsed):
        super().__init__(parsed)
        self.parsed = parsed


def _runner_classes():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_dense_blockscaled_gemm as gemm,
    )

    # The fine-grained MXFP8 runner keeps FP32 split-K partials; its parent does not.
    assert gemm.RubinMxfp8GemmRunner.split_k_fp32_partials
    assert not gemm.RubinMxfp8BaseGemmRunner.split_k_fp32_partials
    return gemm.RubinMxfp8BaseGemmRunner, gemm.RubinMxfp8GemmRunner


def _stopped_runner(cls, output_dtype):
    """A runner whose forward stops at its first step after the check (the FP32-partials query
    of the workspace bound) and reports what it parsed."""
    runner = cls(output_dtype)

    def stop(split_k):
        frame = inspect.currentframe().f_back
        raise _Reached({name: frame.f_locals.get(name) for name in PARSED})

    runner._fp32_split_k_partials = stop
    return runner


def _meta_inputs(m, k, n, packed_k_multiplier=1):
    def empty(*shape):
        return torch.empty(shape, dtype=torch.uint8, device="meta")

    k_bytes = k // packed_k_multiplier
    return [empty(m, k_bytes), empty(n, k_bytes), empty(1), empty(1), empty(1)]


def _forward(runner, tactic, m=M, k=K, n=N):
    """forward's parsed view where it passed the check, or the ValueError it raised."""
    inputs = _meta_inputs(m, k, n, runner.packed_k_multiplier)
    try:
        runner(inputs, tactic=tactic)
    except _Reached as reached:
        return reached.parsed
    except ValueError as err:
        return err
    raise AssertionError("forward neither stopped nor raised")


def _expected(tactic, split_k, m=M, k=K, n=N, epi_tile_n=None):
    (_, mma_tiler_mnk, mma_inst_shape, cluster_shape_mn, swap_ab, use_prefetch) = tactic[:6]
    return dict(
        kernel_variant="base",
        mma_tiler_mnk=mma_tiler_mnk,
        mma_inst_shape=mma_inst_shape,
        cluster_shape_mn=cluster_shape_mn,
        swap_ab=swap_ab,
        use_prefetch=use_prefetch,
        scheduler_mode=tactic[6],
        raster_order=tactic[7],
        split_k=split_k,
        epi_tile_n=epi_tile_n,
        m=m,
        k=k,
        n=n,
    )


@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_legal_bf16_split_k_passes_untouched(split_k):
    mxfp8, fine = _runner_classes()
    for cls in (mxfp8, fine):
        runner = _stopped_runner(cls, torch.bfloat16)
        assert _forward(runner, TILE + (split_k,)) == _expected(TILE, split_k), cls
    # The check runs before the FP32-partials workspace bound, which may still drop a legal
    # split_k to 1 later for a large M (here the stop shows the tactic's own split_k).
    runner = _stopped_runner(fine, torch.bfloat16)
    assert _forward(runner, TILE + (split_k,), m=16384) == _expected(TILE, split_k, m=16384)


@pytest.mark.parametrize("split_k", [3, 8, 32, 64])
def test_split_k_that_does_not_divide_the_k_tiles_raises(split_k):
    mxfp8, fine = _runner_classes()
    for cls in (mxfp8, fine):
        for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            result = _forward(_stopped_runner(cls, output_dtype), TILE + (split_k,))
            assert isinstance(result, ValueError), (cls, output_dtype)
            assert f"split_k={split_k} needs K={K} in whole K tiles of 128" in str(result)


def test_split_k_on_a_partial_k_tile_raises():
    mxfp8, _ = _runner_classes()
    runner = _stopped_runner(mxfp8, torch.bfloat16)
    assert _forward(runner, TILE + (1,), k=448) == _expected(TILE, 1, k=448)
    result = _forward(runner, TILE + (2,), k=448)
    assert isinstance(result, ValueError) and "K=448 in whole K tiles of 128" in str(result)


@pytest.mark.parametrize("split_k", [2, 4])
def test_fp8_output_split_k_raises(split_k):
    mxfp8, fine = _runner_classes()
    for cls in (mxfp8, fine):
        result = _forward(_stopped_runner(cls, torch.float8_e4m3fn), TILE + (split_k,))
        assert isinstance(result, ValueError), cls
        assert f"FP8 output runs split_k=1 only, got split_k={split_k}" in str(result)


def test_fp8_output_split_1_passes_untouched():
    mxfp8, _ = _runner_classes()
    runner = _stopped_runner(mxfp8, torch.float8_e4m3fn)
    assert _forward(runner, TILE + (1,)) == _expected(TILE, 1)
    # The FP8-output table tactic with an epilogue subtile (10 elements).
    for epi_tile_n in (128, 32, 64):
        assert _forward(runner, TABLE_TILE + (1, epi_tile_n)) == _expected(
            TABLE_TILE, 1, epi_tile_n=epi_tile_n
        )
    # No tactic (an autotuner cache miss) below the table's M: the 128x128 fallback, split 1.
    assert _forward(runner, -1) == _expected(TILE, 1)


def test_tactics_without_split_k_pass_untouched():
    mxfp8, _ = _runner_classes()
    for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        runner = _stopped_runner(mxfp8, output_dtype)
        assert _forward(runner, TILE) == _expected(TILE, 1)
        assert _forward(runner, TILE[1:6]) == _expected(TILE, 1)
        parsed = _forward(runner, MIXED)
        assert parsed["kernel_variant"] == "mixed_clusters" and parsed["split_k"] == 1


@pytest.mark.parametrize("split_k", [0, -1, True, 2.0])
def test_split_k_must_be_a_positive_integer(split_k):
    mxfp8, _ = _runner_classes()
    result = _forward(_stopped_runner(mxfp8, torch.bfloat16), TILE + (split_k,))
    assert isinstance(result, ValueError)
    assert "split_k must be a positive integer" in str(result)


def _gemm_inputs(seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(M, K, generator=g, device="cuda")
    w = torch.randn(N, K, generator=g, device="cuda") / 16
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x.to(torch.bfloat16))
    b, b_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w.to(torch.bfloat16))
    return [a, b, a_sf, b_sf, torch.ones((), dtype=torch.float32, device="cuda")]


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def test_refused_tactics_leave_the_gpu_alone(monkeypatch):
    """Refused tactics launch nothing (the GPU stays usable and a legal call gives the bytes it
    gave before them), and a legal tactic gives the same bytes with the check bypassed."""
    import tensorrt_llm  # noqa: F401  (registers the quantization ops)

    mxfp8, fine = _runner_classes()
    inputs = _gemm_inputs(seed=0)
    cases = (
        # (runner class, output dtype, legal split_k, refused split_k)
        (mxfp8, torch.bfloat16, (1,), (3, 8, 32)),
        (mxfp8, torch.float8_e4m3fn, (1,), (2, 4, 32)),
        (fine, torch.bfloat16, (1, 2, 4), (3, 8, 32)),
    )
    for cls, output_dtype, legal, refused in cases:
        runner = cls(output_dtype)
        before = runner(inputs, tactic=TILE + (1,))
        torch.cuda.synchronize()
        for split_k in refused:
            with pytest.raises(ValueError):
                runner(inputs, tactic=TILE + (split_k,))
        torch.cuda.synchronize()
        after = runner(inputs, tactic=TILE + (1,))
        torch.cuda.synchronize()
        assert torch.equal(_bytes(after), _bytes(before)), (cls, output_dtype)
        for split_k in legal:
            checked = runner(inputs, tactic=TILE + (split_k,))
            with monkeypatch.context() as ctx:
                ctx.setattr(cls, "_check_split_k", lambda *args: None)
                unchecked = runner(inputs, tactic=TILE + (split_k,))
            torch.cuda.synchronize()
            assert torch.equal(_bytes(checked), _bytes(unchecked)), (cls, output_dtype, split_k)
