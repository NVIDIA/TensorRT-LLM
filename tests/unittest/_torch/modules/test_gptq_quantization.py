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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.gptq import convert_gptq_weights
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def _checkpoint(k, n, group_size, dtype, checkpoint_format, seed=0):
    """Pack independently generated logical weights in the public GPTQ format."""
    rng = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 16, (k, n), generator=rng, dtype=torch.int32)
    # Exercise every zero point, including v1's wrapped zero and signed int32 MSB.
    zeros = torch.arange(k // group_size * n).reshape(-1, n) % 16
    scales = (torch.rand(k // group_size, n, generator=rng) * 0.125 + 0.01).to(dtype)
    qweight = torch.zeros(k // 8, n, dtype=torch.int32)
    qzeros = torch.zeros(k // group_size, n // 8, dtype=torch.int32)
    stored_zeros = (zeros - int(checkpoint_format == "gptq")) & 15
    for i in range(8):
        qweight |= values[i::8] << (4 * i)
        qzeros |= stored_zeros[:, i::8].to(torch.int32) << (4 * i)
    weights = {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "g_idx": torch.arange(k, dtype=torch.int32) // group_size,
        "bias": torch.randn(n, generator=rng, dtype=dtype),
    }
    reference = values.float() - zeros.repeat_interleave(group_size, dim=0)
    reference *= scales.float().repeat_interleave(group_size, dim=0)
    return weights, reference


@pytest.mark.cpu_only
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_conversion(group_size, checkpoint_format):
    raw, expected = _checkpoint(256, 128, group_size, torch.float32, checkpoint_format)
    original = {key: value.clone() for key, value in raw.items()}
    result = convert_gptq_weights(raw, group_size, checkpoint_format)
    packed = result["weight"].T
    signed = torch.empty(256, 128, dtype=torch.int8)
    signed[:, 0::2] = ((packed & 15) ^ 8) - 8
    signed[:, 1::2] = (((packed >> 4) & 15) ^ 8) - 8
    actual = signed * result["weight_scale"].T.repeat_interleave(group_size, dim=0)
    actual += result["weight_zero"].T.repeat_interleave(group_size, dim=0)
    torch.testing.assert_close(actual, expected)
    assert result["weight"].dtype == torch.int8
    for key in raw:
        assert torch.equal(raw[key], original[key])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "Missing GPTQ tensors"),
        ("packing", "int32 packing"),
        ("shape", "shape does not match"),
        ("g_idx", "activation-order"),
        ("format", "checkpoint_format"),
        ("group", "group_size"),
    ],
)
def test_gptq_conversion_rejects_invalid_checkpoint(change, error):
    raw, _ = _checkpoint(256, 128, 128, torch.float16, "gptq")
    group, fmt = 128, "gptq"
    if change == "missing":
        del raw["qzeros"]
    elif change == "packing":
        raw["qweight"] = raw["qweight"].to(torch.int64)
    elif change == "shape":
        raw["scales"] = raw["scales"][:1]
    elif change == "g_idx":
        raw["g_idx"] = raw["g_idx"].flip(0)
    elif change == "format":
        fmt = "marlin"
    else:
        group = 32
    with pytest.raises(ValueError, match=error):
        convert_gptq_weights(raw, group, fmt)


def _mapper(checkpoint_format, tp_size=1):
    mapper = HfWeightMapper()
    mapper._model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={"checkpoint_format": checkpoint_format},
            num_key_value_heads=1,
        )
    )
    mapper._tp_size = tp_size
    mapper.map_weights()
    return mapper


def _quant_config(group_size):
    return QuantConfig(quant_algo=QuantAlgo.W4A16_GPTQ, group_size=group_size, has_zero_point=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
@pytest.mark.parametrize("tp_mode", [None, TensorParallelMode.ROW, TensorParallelMode.COLUMN])
def test_gptq_linear(dtype, group_size, checkpoint_format, tp_mode):
    k, n = 384, 384
    raw, reference = _checkpoint(k, n, group_size, dtype, checkpoint_format)
    mapper = _mapper(checkpoint_format)
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(7, k, generator=rng, dtype=dtype).cuda()
    tp_size = 1 if tp_mode is None else 2
    results = []
    for rank in range(tp_size):
        linear = Linear(
            k,
            n,
            bias=tp_mode != TensorParallelMode.ROW,
            dtype=dtype,
            quant_config=_quant_config(group_size),
            mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=rank),
            tensor_parallel_mode=tp_mode,
            reduce_output=False,
        ).cuda()
        assert mapper.is_special_instance_module(linear)
        mapper.handle_special_instance_module(linear, "o_proj", raw)
        pointers = [p.data_ptr() for p in linear.parameters()]
        local_x = x
        if tp_mode == TensorParallelMode.ROW:
            start, end = linear.tp_sharding
            local_x = x[:, start:end].contiguous()
        results.append(linear(local_x))
        # Reloads must preserve parameter storage (CUDA graph addresses).
        mapper.handle_special_instance_module(linear, "o_proj", raw)
        assert pointers == [p.data_ptr() for p in linear.parameters()]
    output = sum(results) if tp_mode == TensorParallelMode.ROW else torch.cat(results, dim=-1)
    expected = x.float() @ reference.cuda()
    if tp_mode != TensorParallelMode.ROW:
        expected += raw["bias"].float().cuda()
    # BF16 additive-zero rounding differs from direct (q - zp) * scale.
    torch.testing.assert_close(
        output.float(),
        expected,
        rtol=0.005 if dtype == torch.float16 else 0.03,
        atol=0.03 if dtype == torch.float16 else 0.25,
    )


@pytest.mark.parametrize(
    "weight_mode", [WeightMode.FUSED_QKV_LINEAR, WeightMode.FUSED_GATE_UP_LINEAR]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_fused_linear(weight_mode, dtype, checkpoint_format):
    names = (
        ["q_proj", "k_proj", "v_proj"]
        if weight_mode == WeightMode.FUSED_QKV_LINEAR
        else ["gate_proj", "up_proj"]
    )
    raws, refs = zip(
        *[_checkpoint(256, 128, 128, dtype, checkpoint_format, i) for i in range(len(names))]
    )
    linear = Linear(
        256,
        128 * len(names),
        dtype=dtype,
        quant_config=_quant_config(128),
        weights_loading_config=WeightsLoadingConfig(weight_mode=weight_mode),
    ).cuda()
    mapper = _mapper(checkpoint_format)
    fused_name = "qkv_proj" if len(names) == 3 else "gate_up_proj"
    checkpoint = {
        f"block.{name}.{key}": value for name, raw in zip(names, raws) for key, value in raw.items()
    }
    weights = mapper.apply_callbacks(linear, fused_name, ["block"], checkpoint)
    linear.load_weights(weights)
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(3, 256, generator=rng, dtype=dtype).cuda()
    expected = x.float() @ torch.cat(refs, dim=1).cuda()
    expected += torch.cat([raw["bias"] for raw in raws]).float().cuda()
    torch.testing.assert_close(
        linear(x).float(),
        expected,
        rtol=0.005 if dtype == torch.float16 else 0.03,
        atol=0.03 if dtype == torch.float16 else 0.2,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_gptq_qkv_tp_replicates_kv_group_parameters(rank):
    dtype, k, group_size = torch.float16, 256, 128
    mapper = _mapper("gptq", tp_size=2)
    names = ["q_proj", "k_proj", "v_proj"]
    data = [_checkpoint(k, n, group_size, dtype, "gptq", i) for i, n in enumerate([256, 128, 128])]
    checkpoint = {
        f"block.{name}.{key}": value
        for name, (raw, _) in zip(names, data)
        for key, value in raw.items()
    }
    linear = Linear(
        k,
        768,
        dtype=dtype,
        quant_config=_quant_config(group_size),
        mapping=Mapping(world_size=2, tp_size=2, rank=rank),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
        override_tp_sharding={key: (rank * 128, (rank + 1) * 128) for key in ("q", "k", "v")},
    ).cuda()
    weights = mapper.apply_callbacks(linear, "qkv_proj", ["block"], checkpoint)
    linear.load_weights(weights)
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(4, k, generator=rng, dtype=dtype).cuda()
    refs = [data[0][1][:, rank * 128 : (rank + 1) * 128], data[1][1], data[2][1]]
    biases = [
        data[0][0]["bias"][rank * 128 : (rank + 1) * 128],
        data[1][0]["bias"],
        data[2][0]["bias"],
    ]
    expected = x.float() @ torch.cat(refs, dim=1).cuda() + torch.cat(biases).float().cuda()
    torch.testing.assert_close(linear(x).float(), expected, rtol=0.005, atol=0.03)


@pytest.mark.cpu_only
def test_gptq_partial_loading_rejected():
    """Partial updates must fail before converting or copying GPTQ weights."""
    raw, _ = _checkpoint(256, 256, 128, torch.float16, "gptq")
    linear = Linear(256, 256, dtype=torch.float16, quant_config=_quant_config(128))
    with pytest.raises(ValueError, match="GPTQ does not support partial weight loading"):
        _mapper("gptq").handle_special_instance_module(
            linear, "o_proj", raw, allow_partial_loading=True
        )


def test_gptq_cuda_graph():
    raw, _ = _checkpoint(256, 256, 128, torch.float16, "gptq")
    linear = Linear(256, 256, dtype=torch.float16, quant_config=_quant_config(128)).cuda()
    _mapper("gptq").handle_special_instance_module(linear, "o_proj", raw)
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(4, 256, generator=rng, dtype=torch.float16).cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            linear(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = linear(x)
    x.add_(1)
    graph.replay()
    torch.testing.assert_close(output, linear(x))


def test_gptq_model_weight_loader():
    from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl_v2

    model = torch.nn.Module()
    model.qkv_proj = Linear(
        256,
        384,
        dtype=torch.float16,
        quant_config=_quant_config(128),
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
    )
    model.o_proj = Linear(384, 256, dtype=torch.float16, quant_config=_quant_config(128))
    model.cuda()
    checkpoint, reference, biases = {}, {}, {}
    for i, (name, k, n) in enumerate(
        [("q_proj", 256, 128), ("k_proj", 256, 128), ("v_proj", 256, 128), ("o_proj", 384, 256)]
    ):
        raw, ref = _checkpoint(k, n, 128, torch.float16, "gptq", i)
        checkpoint.update({f"{name}.{key}": value for key, value in raw.items()})
        reference[name], biases[name] = ref.cuda(), raw["bias"].float().cuda()
    _load_weights_impl_v2(model, checkpoint, _mapper("gptq"))
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 256, generator=rng, dtype=torch.float16).cuda()
    qkv_ref = torch.cat(
        [x.float() @ reference[name] + biases[name] for name in ("q_proj", "k_proj", "v_proj")],
        dim=-1,
    )
    torch.testing.assert_close(model.qkv_proj(x).float(), qkv_ref, rtol=0.005, atol=0.03)
    qkv = model.qkv_proj(x)
    out_ref = qkv.float() @ reference["o_proj"] + biases["o_proj"]
    torch.testing.assert_close(model.o_proj(qkv).float(), out_ref, rtol=0.005, atol=0.1)
