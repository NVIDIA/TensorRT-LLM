# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 targets' decoder glue (each target's ``modeling.py``) against main's helpers it replaced:
bitwise.

* ``_Fp8BlockScaleWeightReadLinear.from_linear`` on a ``core.Fp8BlockScaleLinear`` (loaded, or
  loaded and prepared; one rank or a COLUMN / ROW shard) gives the projection built from the codes
  and scales that main's ``Linear(FP8_BLOCK_SCALES)`` loads from the same checkpoint tensors.
* ``_shard_dim`` slices as main's ``load_weight_shard`` on every split K3 loads through it.
"""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm import (
    sm107_linear as linear,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode, load_weight_shard
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

FP8 = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
COL, ROW = TensorParallelMode.COLUMN, TensorParallelMode.ROW
ROWS = (1, 3, 8, 16, 31, 64, 100, 128, 129, 256, 1000)

requires_block_fp8 = pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107)
    or (getSMVersion() == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE),
    reason="needs a supported block-scaled FP8 GPU and toolchain",
)


def _fp8_weight_read():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix8_moeep8 import (  # noqa: E501
        modeling as fp8_weight_read,
    )

    return fp8_weight_read


def _weights():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix8_moeep8 import (  # noqa: E501
        weights,
    )

    return weights


def _checkpoint_fp8(n, k, seed):
    """E4M3 codes and FP32 (non power-of-two) 128x128 block scales of a random ``[n, k]`` weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = torch.nn.functional.pad(weight, (0, -k % 128, 0, -n % 128)).view(
        -(-n // 128), 128, -(-k // 128), 128
    )
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return (weight / expanded).to(torch.float8_e4m3fn), scale


def _swapped(source):
    proj = _fp8_weight_read()._Fp8BlockScaleWeightReadLinear.from_linear(source)
    proj.post_load_weights()
    return proj


def _from_main_linear(main):
    """The FP8 weight read over the codes, scales and shard that main's ``Linear`` loaded."""
    proj = _fp8_weight_read()._Fp8BlockScaleWeightReadLinear(
        main.weight.detach(), main.weight_scale.detach(), main.out_features
    )
    proj.quant_config = main.quant_config
    proj.tp_size, proj.tp_rank = main.tp_size, main.tp_rank
    # main's ``Linear`` holds the checkpoint codes and FP32 grid as loaded: post-load prepares them.
    proj._weights_transformed = False
    proj.post_load_weights()
    return proj


def _assert_same_projection(got, ref, k):
    assert (got.out_features, got.in_features) == (ref.out_features, ref.in_features)
    assert (got.tp_size, got.tp_rank) == (ref.tp_size, ref.tp_rank)
    assert got.quant_config.quant_algo == ref.quant_config.quant_algo
    assert torch.equal(got.weight.view(torch.uint8), ref.weight.view(torch.uint8))
    assert torch.equal(got.weight_scale, ref.weight_scale)
    # The helix8 glue has no generic low-M GEMM: the K3 low-M GEMM reads ``lowm_weight_scale``.
    a, b = got.lowm_weight_scale, ref.lowm_weight_scale
    assert (a is None) == (b is None)
    if a is not None:
        assert torch.equal(a, b)
    for m in ROWS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(got(x), ref(x)), m


# K3's checkpoint FP8 projection shapes (q_a + kv_a fused, q_b / kv_b / o / g heads, KDA q/k/v/g/o).
FP8_SHAPES = [(2112, 7168), (1152, 1536), (2304, 512), (7168, 1024), (512, 7168), (4096, 7168)]


@requires_block_fp8
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("n,k", FP8_SHAPES)
@torch.inference_mode()
def test_weight_read_from_core_linear_matches_main_linear(n, k, prepared):
    codes, scale = _checkpoint_fp8(n, k, seed=n + k)
    with torch.device("cuda"):
        main = Linear(k, n, bias=False, dtype=torch.bfloat16, quant_config=FP8, reduce_output=False)
    main.load_weights([{"weight": codes.clone(), "weight_scale": scale.clone()}])
    core = linear.Fp8BlockScaleLinear(k, n, device="cuda")
    core.load(codes.clone(), scale.clone())
    if prepared:
        core.prepare()
    _assert_same_projection(_swapped(core), _from_main_linear(main), k)


# MLA projections K3 splits over TP ranks: kv_b / q_b COLUMN over 64 heads x 256 / 192, o ROW over
# 64 heads x 128.
SHARDED = [(16384, 512, COL), (12288, 1536, COL), (7168, 8192, ROW)]


@requires_block_fp8
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("n,k,mode", SHARDED)
@torch.inference_mode()
def test_weight_read_from_sharded_core_linear_matches_main_linear(n, k, mode, tp):
    codes, scale = _checkpoint_fp8(n, k, seed=n * k + tp)
    for rank in range(tp):
        mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
        with torch.device("cuda"):
            main = Linear(
                k,
                n,
                bias=False,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=mode,
                quant_config=FP8,
                reduce_output=False,
            )
        main.load_weights([{"weight": codes.clone(), "weight_scale": scale.clone()}])
        core = linear.Fp8BlockScaleLinear(
            k, n, tp_size=tp, tp_rank=rank, tp_mode=mode, device="cuda"
        )
        core.load(codes.clone(), scale.clone())
        _assert_same_projection(_swapped(core), _from_main_linear(main), core.in_features)


# (shape, dim) of the splits K3 loads through ``_shard_dim``: KDA head-major rows (dim 0) and o_proj
# columns (dim 1), 1-D head vectors, width-1 dims, the dense / shared-expert gate, up (dim 0) and
# down (dim 1) halves.
SPLITS = [
    ((4096, 7168), 0),
    ((7168, 4096), 1),
    ((64,), 0),
    ((64,), 1),
    ((1, 1, 64, 1), 0),
    ((18432, 7168), 0),
    ((7168, 18432), 1),
]


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("shape,dim", SPLITS)
def test_shard_dim_matches_main_load_weight_shard(shape, dim, tp):
    tensor = torch.randn(shape)
    mode = COL if dim == 0 else ROW
    for rank in range(tp):
        got = _weights()._shard_dim(tensor, dim, tp, rank, torch.device("cpu"))
        assert torch.equal(got, load_weight_shard(tensor, tp, rank, mode)), (rank, tp)
