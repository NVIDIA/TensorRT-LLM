# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU checkpoint loading correctness for Kimi K3."""

import math
from pathlib import Path

import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

pytest.importorskip("fla")

from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig  # noqa: E402

_SHAPES = {
    "q_a_proj": (1536, 128),
    "kv_a_proj_with_mqa": (576, 128),
    "q_b_proj": (8 * 192, 1536),
    "kv_b_proj": (8 * 256, 512),
    "g_proj": (8 * 128, 128),
    "o_proj": (128, 8 * 128),
}


def _config(
    fp8: bool, *, mapping: Mapping, prefix: str, deferred: bool, cute_bmm: bool
) -> ModelConfig:
    return ModelConfig(
        pretrained_config=KimiLinearConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=8,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rms_norm_eps=1e-6,
            attn_res_block_size=1,
            mla_use_output_gate=True,
            max_position_embeddings=128,
            torch_dtype=torch.bfloat16,
            linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 8},
        ),
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            f"{prefix}model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in _SHAPES
        }
        if fp8
        else None,
        skip_create_weights_in_init=deferred,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=cute_bmm,
    )


def load_mla_checkpoint(
    model: KimiLinearForCausalLM, projections: dict[str, torch.Tensor], prefix: str
) -> None:
    # Supply a synthetic one-layer checkpoint through the public model loader.
    name_map, _, _ = model.checkpoint_name_plan(prefix)
    config = model.model_config.pretrained_config
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        if name.endswith("gate_up_proj.weight"):
            for source in ("gate_proj", "up_proj"):
                weights[key.replace("gate_up_proj", source)] = torch.zeros(
                    config.intermediate_size, config.hidden_size, dtype=param.dtype
                )
        else:
            shape = param.shape
            if name == "lm_head.weight":
                shape = (config.vocab_size, config.hidden_size)
            elif name.endswith(".mlp.down_proj.weight"):
                shape = (config.hidden_size, config.intermediate_size)
            weights[key] = torch.zeros(shape, dtype=param.dtype)
    weights.update(
        {f"{prefix}model.layers.0.self_attn.{name}": value for name, value in projections.items()}
    )
    model.load_weights(weights)


def _weights(fp8: bool, *, scale_name: str = "weight_scale") -> dict:
    generator = torch.Generator().manual_seed(123)
    result = {}
    for name, shape in _SHAPES.items():
        weight = torch.randn(shape, generator=generator)
        result[f"{name}.weight"] = weight.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
        if fp8:
            # Deliberately non-power-of-two scales, with the ModelOpt singleton axes.
            grid = tuple(math.ceil(dim / 128) for dim in shape)
            scale = 0.002 + torch.rand(grid, generator=generator) * 0.001
            result[f"{name}.{scale_name}"] = scale[:, None, :, None]
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize("deferred,cute_bmm", [(False, True), (True, False)])
@pytest.mark.parametrize(
    "fp8,scale_name", [(False, "weight_scale"), (True, "weight_scale"), (True, "weight_scale_inv")]
)
@pytest.mark.parametrize("tp,cp,adp", [(1, 1, False), (2, 1, False), (2, 2, False), (2, 1, True)])
def test_checkpoint_loads_fp8_mla_values(
    prefix: str,
    deferred: bool,
    cute_bmm: bool,
    fp8: bool,
    scale_name: str,
    tp: int,
    cp: int,
    adp: bool,
) -> None:
    world_size = tp * cp
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    args = (prefix, deferred, cute_bmm, fp8, scale_name, tp, cp, adp)
    if world_size == 1:
        _check_mla_checkpoint_values(*args)
    else:
        with MPIPoolExecutor(
            max_workers=world_size, path=[str(Path(__file__).resolve().parent)]
        ) as executor:
            futures = [
                executor.submit(_check_mla_checkpoint_values, *args) for _ in range(world_size)
            ]
            for future in futures:
                future.result()


def _check_mla_checkpoint_values(
    prefix: str,
    deferred: bool,
    cute_bmm: bool,
    fp8: bool,
    scale_name: str,
    tp: int,
    cp: int,
    adp: bool,
) -> None:
    rank = MPI.COMM_WORLD.Get_rank()
    assert MPI.COMM_WORLD.Get_size() == tp * cp
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=tp * cp,
        tp_size=tp,
        cp_size=cp,
        rank=rank,
        cp_config={"cp_type": "HELIX"} if cp > 1 else None,
        enable_attention_dp=adp,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(
            _config(fp8, mapping=mapping, prefix=prefix, deferred=deferred, cute_bmm=cute_bmm)
        )
    attention = model.model.layers[0].self_attn.mixer
    weights = _weights(fp8, scale_name=scale_name)
    load_mla_checkpoint(model, weights, prefix)
    effective_tp = 1 if adp else tp
    tp_rank = 0 if adp else mapping.tp_rank
    heads = 8 // effective_tp
    head_start = tp_rank * heads
    v_start = head_start + mapping.cp_rank * heads // cp
    v_end = v_start + heads // cp
    expected_a = torch.cat(
        (weights["q_a_proj.weight"].float(), weights["kv_a_proj_with_mqa.weight"].float())
    )
    torch.testing.assert_close(attention.kv_a_proj_with_mqa.weight.float().cpu(), expected_a)
    kv_b = weights["kv_b_proj.weight"].float().view(8, 256, 512)
    k, v = kv_b.split(128, dim=1)
    expected_k = k[head_start : head_start + heads]
    expected_v = v[head_start : head_start + heads]
    torch.testing.assert_close(
        attention.kv_b_proj.weight.float().cpu(),
        torch.cat((expected_k.flatten(0, 1), expected_v.flatten(0, 1))),
    )
    torch.testing.assert_close(attention.k_b_proj_trans.float().cpu(), expected_k.transpose(1, 2))
    torch.testing.assert_close(attention.v_b_proj.float().cpu(), v[v_start:v_end])
    if not fp8:
        assert (
            attention.v_b_proj.untyped_storage().data_ptr()
            == attention.kv_b_proj.weight.untyped_storage().data_ptr()
        )
    for name in ("q_b_proj", "g_proj", "o_proj"):
        expected = weights[f"{name}.weight"].float()
        if name == "q_b_proj":
            expected = expected[head_start * 192 : (head_start + heads) * 192]
        elif name == "g_proj":
            expected = expected[v_start * 128 : v_end * 128]
        else:
            expected = expected[:, v_start * 128 : v_end * 128]
        torch.testing.assert_close(getattr(attention, name).weight.float().cpu(), expected)
        if fp8:
            scale = weights[f"{name}.{scale_name}"].squeeze(1).squeeze(-1)
            if name == "q_b_proj":
                scale = scale[head_start * 192 // 128 : (head_start + heads) * 192 // 128]
            elif name == "g_proj":
                scale = scale[v_start:v_end]
            else:
                scale = scale[:, v_start:v_end]
            torch.testing.assert_close(getattr(attention, name).weight_scale.cpu(), scale)
    if fp8:
        assert attention.v_b_proj.dtype == torch.float8_e4m3fn
        expected_scale = (
            torch.cat(
                (weights[f"q_a_proj.{scale_name}"], weights[f"kv_a_proj_with_mqa.{scale_name}"])
            )
            .squeeze(1)
            .squeeze(-1)
        )
        torch.testing.assert_close(attention.kv_a_proj_with_mqa.weight_scale.cpu(), expected_scale)
        scales = weights[f"kv_b_proj.{scale_name}"].view(8, 2, 4)
        torch.testing.assert_close(
            attention.k_b_proj_trans_scale.cpu(),
            scales[head_start : head_start + heads, :1].transpose(1, 2),
        )
        torch.testing.assert_close(attention.v_b_proj_scale.cpu(), scales[v_start:v_end, 1:])
        local_scales = scales[head_start : head_start + heads]
        torch.testing.assert_close(
            attention.kv_b_proj.weight_scale.cpu(),
            torch.cat((local_scales[:, :1].flatten(0, 1), local_scales[:, 1:].flatten(0, 1))),
        )
        if attention.k_b_proj_trans_dequant is not None:
            expanded = scales.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
            reference_k, reference_v = (kv_b * expanded).split(128, dim=1)
            for loaded, expected in (
                (
                    attention.k_b_proj_trans_dequant,
                    reference_k[head_start : head_start + heads].transpose(1, 2),
                ),
                (attention.v_b_proj_dequant, reference_v[v_start:v_end]),
            ):
                torch.testing.assert_close(loaded.cpu(), expected.to(loaded.dtype))
