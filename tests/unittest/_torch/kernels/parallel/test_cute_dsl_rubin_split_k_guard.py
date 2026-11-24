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
"""Host-side split-K check of the SM107 block-scaled dense GEMM runners
(``CuteDSLBlockScaledRubinLinear.forward``: the runners of ``cute_dsl_mxfp8_gemm_rubin``, its
in-place variant, and the NVFP4 runners).

A tactic whose split_k the kernel cannot run (K tiles that split_k does not divide, or FP8
output) raises ValueError before anything is allocated or launched; every other tactic goes on
through forward with what it carries. The first tests need the CuTe DSL package but no GPU: the
inputs are meta tensors, a caller-owned meta output keeps forward from allocating, and forward is
stopped where it makes its first CuTe pointer. The last test launches on SM107.
"""

import inspect

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

requires_cute_dsl_rubin = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE, reason="needs the CuTe DSL package with SM107 support"
)
requires_sm107 = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="launches the SM107 dense GEMM",
)

M, N, K = 256, 1024, 512  # MXFP8: 4 K tiles of 128
TILE = ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "m")
TABLE_TILE = ("base", (256, 256, 128), (256, 256, 64), (2, 1), False, False, "static", "m")
MIXED = ("mixed_clusters", (256, 256, 128), (256, 256, 64), (4, 2), (2, 1), False, False, "m")
# forward's view of a tactic, recorded where it makes its first CuTe pointer.
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


def _ops():
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    return ops


def _stopped_runner(cls, output_dtype):
    """A runner whose forward stops where it makes its first CuTe pointer (after the check and the
    output setup) and reports what it parsed."""
    runner = cls(output_dtype)

    def stop(*args, **kwargs):
        frame = inspect.currentframe().f_back
        raise _Reached({name: frame.f_locals.get(name) for name in PARSED})

    runner.make_cute_dsl_global_pointer = stop
    return runner


def _pad(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def _meta_inputs(runner, m, k, n, own_output):
    """Meta inputs with the scale sizes forward expects. With ``own_output`` the output is a
    caller-owned [m, 2n] tensor that forward writes as locality-domain half 0, so forward
    allocates nothing on the way to the stop."""

    def empty(*shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="meta")

    k_bytes = k // runner.packed_k_multiplier
    sf_k = _pad(k // runner.sf_vec_size, 4)
    inputs = [
        empty(m, k_bytes),
        empty(n, k_bytes),
        empty(_pad(m, 128) * sf_k),
        empty(_pad(n, 128) * sf_k),
        empty(1, dtype=torch.float32),
    ]
    if own_output:
        inputs.append(empty(m, 2 * n, dtype=runner.output_dtype))
    return inputs


def _forward(runner, tactic, m=M, k=K, n=N, own_output=True):
    """forward's parsed view at the stop, or the ValueError it raised."""
    inputs = _meta_inputs(runner, m, k, n, own_output)
    kwargs = {"partition_id": 0} if own_output else {}
    try:
        runner(inputs, tactic=tactic, **kwargs)
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


@requires_cute_dsl_rubin
def test_every_runner_takes_the_checked_forward():
    ops = _ops()
    base = ops.CuteDSLBlockScaledRubinLinear
    seen, todo = [], list(base.__subclasses__())
    while todo:
        cls = todo.pop()
        seen.append(cls)
        todo.extend(cls.__subclasses__())
    assert ops.CuteDSLMXFP8RubinLinear in seen and ops.CuteDSLNVFP4RubinLinear in seen
    for cls in seen:
        assert cls.forward is base.forward, cls
        assert cls._check_split_k is base._check_split_k, cls


@requires_cute_dsl_rubin
@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_legal_bf16_split_k_passes_untouched(split_k):
    runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, torch.bfloat16)
    assert _forward(runner, TILE + (split_k,)) == _expected(TILE, split_k)


@requires_cute_dsl_rubin
@pytest.mark.parametrize("split_k", [3, 8, 32, 64])
def test_split_k_that_does_not_divide_the_k_tiles_raises(split_k):
    for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, output_dtype)
        result = _forward(runner, TILE + (split_k,))
        assert isinstance(result, ValueError), output_dtype
        assert f"split_k={split_k} needs K={K} in whole K tiles of 128" in str(result)


@requires_cute_dsl_rubin
def test_split_k_on_a_partial_k_tile_raises():
    runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, torch.bfloat16)
    assert _forward(runner, TILE + (1,), k=448) == _expected(TILE, 1, k=448)
    result = _forward(runner, TILE + (2,), k=448)
    assert isinstance(result, ValueError) and "K=448 in whole K tiles of 128" in str(result)


@requires_cute_dsl_rubin
@pytest.mark.parametrize("split_k", [2, 4])
def test_fp8_output_split_k_raises(split_k):
    runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, torch.float8_e4m3fn)
    result = _forward(runner, TILE + (split_k,))
    assert isinstance(result, ValueError)
    assert f"FP8 output runs split_k=1 only, got split_k={split_k}" in str(result)


@requires_cute_dsl_rubin
def test_refused_before_the_output_is_allocated():
    # No caller-owned output: forward would allocate C on the GPU, but the check comes first.
    for output_dtype, split_k in ((torch.bfloat16, 3), (torch.float8_e4m3fn, 2)):
        runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, output_dtype)
        result = _forward(runner, TILE + (split_k,), own_output=False)
        assert isinstance(result, ValueError), output_dtype


@requires_cute_dsl_rubin
def test_fp8_output_split_1_passes_untouched():
    runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, torch.float8_e4m3fn)
    assert _forward(runner, TILE + (1,)) == _expected(TILE, 1)
    # The FP8-output table tactic with an epilogue subtile (10 elements).
    for epi_tile_n in (128, 32, 64):
        assert _forward(runner, TABLE_TILE + (1, epi_tile_n)) == _expected(
            TABLE_TILE, 1, epi_tile_n=epi_tile_n
        )
    # No tactic (an autotuner cache miss) below the table's M: the 128x128 fallback, split 1.
    assert _forward(runner, -1) == _expected(TILE, 1)


@requires_cute_dsl_rubin
def test_tactics_without_split_k_pass_untouched():
    for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, output_dtype)
        assert _forward(runner, TILE) == _expected(TILE, 1)
        assert _forward(runner, TILE[1:6]) == _expected(TILE, 1)
        parsed = _forward(runner, MIXED)
        assert parsed["kernel_variant"] == "mixed_clusters" and parsed["split_k"] == 1


@requires_cute_dsl_rubin
def test_nvfp4_runner_takes_the_same_check():
    runner = _stopped_runner(_ops().CuteDSLNVFP4RubinLinear, torch.bfloat16)
    tile = ("base", (128, 128, 256), (128, 128, 128), (1, 1), False, False, "static", "m")
    k = 1024  # 4 K tiles of 256, 512 packed bytes per row
    for split_k in (1, 2, 4):
        assert _forward(runner, tile + (split_k,), k=k) == _expected(tile, split_k, k=k // 2)
    for split_k in (3, 8):
        result = _forward(runner, tile + (split_k,), k=k)
        assert isinstance(result, ValueError) and "in whole K tiles of 256" in str(result)


@requires_cute_dsl_rubin
@pytest.mark.parametrize("split_k", [0, -1, True, 2.0])
def test_split_k_must_be_a_positive_integer(split_k):
    runner = _stopped_runner(_ops().CuteDSLMXFP8RubinLinear, torch.bfloat16)
    result = _forward(runner, TILE + (split_k,))
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


@requires_sm107
def test_refused_tactics_leave_the_gpu_alone():
    """On SM107: refused tactics launch nothing (the GPU stays usable and split 1 gives the bytes
    it gave before them), and legal BF16 split-K still runs."""
    import tensorrt_llm  # noqa: F401  (registers the quantization ops)

    mxfp8 = _ops().CuteDSLMXFP8RubinLinear
    inputs = _gemm_inputs(seed=0)
    cases = (
        # (output dtype, refused split_k)
        (torch.bfloat16, (3, 8, 32)),
        (torch.float8_e4m3fn, (2, 4, 32)),
    )
    for output_dtype, refused in cases:
        runner = mxfp8(output_dtype)
        before = runner(inputs, tactic=TILE + (1,))
        torch.cuda.synchronize()
        for split_k in refused:
            with pytest.raises(ValueError):
                runner(inputs, tactic=TILE + (split_k,))
        torch.cuda.synchronize()
        after = runner(inputs, tactic=TILE + (1,))
        torch.cuda.synchronize()
        assert torch.equal(_bytes(after), _bytes(before)), output_dtype
    runner = mxfp8(torch.bfloat16)
    split_1 = runner(inputs, tactic=TILE + (1,)).float()
    for split_k in (2, 4):
        # The splits reduce-add BF16 partials in no fixed order: close to split 1, not bitwise.
        out = runner(inputs, tactic=TILE + (split_k,)).float()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, split_1, rtol=2e-2, atol=5e-2)
