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

"""Optional real-checkpoint DeepSeek V4 prefill checks.

A full 5-layer eager AutoDeploy model cannot be materialized densely from the
real checkpoint without the quantization/lowering transforms because routed FP4
experts expand to very large BF16 tensors. These checks keep the scope small:
layer-0 attention exercises the AD FineGrained FP8 transform, while the
single-layer logits check remaps the fixed input's hash-routed experts into a
reduced AD model and only dequantizes those selected FP4 experts.
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from examples import deepseek_v4_5layer_forward as demo_dsv4
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FineGrainedFP8LinearQuantization,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_DEFAULT_CHECKPOINT_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
    "bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash"
)
_RUN_ENV = "RUN_DEEPSEEK_V4_REAL_CHECKPOINT"
_LAYER0_ATTENTION_RMSE_RATIO_TOL = 0.08
_LAYER0_LOGITS_INPUT_IDS = ((1,),)
_LAYER0_LOGITS_RMSE_RATIO_TOL = 0.03


def _checkpoint_dir() -> Path:
    return Path(os.environ.get("DEEPSEEK_V4_CHECKPOINT_DIR", _DEFAULT_CHECKPOINT_DIR))


def _require_real_checkpoint() -> Path:
    checkpoint_dir = _checkpoint_dir()
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run the real DeepSeek V4 checkpoint check")
    if not checkpoint_dir.is_dir():
        pytest.skip(f"DeepSeek V4 checkpoint directory not found: {checkpoint_dir}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the FineGrained FP8 Triton reference op")
    return checkpoint_dir


def _load_checkpoint_tensors(
    checkpoint_dir: Path,
    keys: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    keys_by_shard: dict[str, list[str]] = {}
    for key in keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    tensors = {}
    device_arg = str(device)
    for shard_name, shard_keys in sorted(keys_by_shard.items()):
        with safe_open(checkpoint_dir / shard_name, framework="pt", device=device_arg) as shard:
            for key in shard_keys:
                tensors[key] = shard.get_tensor(key)
    return tensors


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    return json.loads(index_path.read_text())["weight_map"]


def _hf_layer0_ad_config(checkpoint_dir: Path, seq_len: int) -> DeepseekV4Config:
    config_values = json.loads((checkpoint_dir / "config.json").read_text())
    config_values["num_hidden_layers"] = 1
    config_values["ad_rope_cache_len"] = seq_len
    config_values["ad_compress_max_seq_len"] = seq_len
    return DeepseekV4Config(**config_values)


def _hf_layer0_reduced_expert_ad_config(
    checkpoint_dir: Path, seq_len: int, num_routed_experts: int
) -> DeepseekV4Config:
    config_values = json.loads((checkpoint_dir / "config.json").read_text())
    config_values["num_hidden_layers"] = 1
    config_values["ad_rope_cache_len"] = seq_len
    config_values["ad_compress_max_seq_len"] = seq_len
    config_values["n_routed_experts"] = num_routed_experts
    return DeepseekV4Config(**config_values)


def _configure_demo_quant_globals(model_module, model_config: dict) -> None:
    """Mirror the checkpoint Transformer quantization globals for attention-only tests."""
    model_module.default_dtype = (
        torch.float8_e4m3fn if model_config.get("dtype", "fp8") == "fp8" else torch.bfloat16
    )
    if model_config.get("scale_dtype", "fp8") == "fp8":
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            raise RuntimeError("torch.float8_e8m0fnu is required for DeepSeek V4 FP8 scales.")
        model_module.scale_fmt = "ue8m0"
        model_module.scale_dtype = e8m0_dtype
    else:
        model_module.scale_fmt = model_config.get("scale_fmt")
        model_module.scale_dtype = torch.float32


class _AttentionLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.attn = DeepseekV4Attention(config, layer_idx=0)


class _Layer0AttentionWrapper(nn.Module):
    """Keep exported parameter names under ``layers.0.attn`` for DSV4 transforms."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttentionLayer(config)])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0].attn(x, position_ids)


class _DeepSeekV4QuantFactory:
    @staticmethod
    def get_quant_config() -> dict:
        return {
            "quant_method": "deepseek_v4_fp8",
            "linear_quant_method": "finegrained_fp8",
            "weight_block_size": [128, 128],
        }


def _build_quantized_ad_layer0_attention(
    checkpoint_dir: Path,
    device: torch.device,
    seq_len: int,
) -> tuple[torch.fx.GraphModule, int]:
    config = _hf_layer0_ad_config(checkpoint_dir, seq_len)
    with demo_dsv4._torch_defaults(torch.bfloat16, device):
        model = _Layer0AttentionWrapper(config).eval().to(device)

    x = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    gm = torch_export_to_gm(model, args=(x, position_ids), kwargs={}, dynamic_shapes=None)

    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    gm, info = transform._apply(gm, None, _DeepSeekV4QuantFactory(), SharedConfig())

    weight_map = _load_weight_map(checkpoint_dir)
    state_dict = {}
    for key in gm.state_dict():
        if key in weight_map:
            state_dict[key] = _load_checkpoint_tensors(checkpoint_dir, [key], device)[key]
        if key.endswith(".weight"):
            scale_alias = key.removesuffix(".weight") + ".scale"
            if scale_alias in weight_map:
                state_dict[scale_alias] = _load_checkpoint_tensors(
                    checkpoint_dir, [scale_alias], device
                )[scale_alias]

    missing, unexpected = gm.load_state_dict(state_dict, strict=False)
    assert missing == []
    assert unexpected == []
    gm.to(device)
    return gm, info.num_matches


def _build_demo_layer0_attention(
    checkpoint_dir: Path,
    device: torch.device,
    seq_len: int,
) -> nn.Module:
    model_module = demo_dsv4._import_checkpoint_model(checkpoint_dir)
    demo_dsv4._patch_checkpoint_model(model_module)
    model_config = demo_dsv4._load_model_args(
        checkpoint_dir, layers=1, batch_size=1, seq_len=seq_len
    )
    _configure_demo_quant_globals(model_module, model_config)
    with demo_dsv4._torch_defaults(torch.bfloat16, device):
        model_args = model_module.ModelArgs(**model_config)
        attention = model_module.Attention(0, model_args).eval()
        layer = SimpleNamespace(attn=attention)
        demo_dsv4._convert_wo_a_to_fp8(SimpleNamespace(layers=[layer]))

    state_keys = [
        f"layers.0.attn.{key}"
        for key in attention.state_dict()
        if f"layers.0.attn.{key}" in _load_weight_map(checkpoint_dir)
    ]
    state_dict = {
        key.removeprefix("layers.0.attn."): tensor
        for key, tensor in _load_checkpoint_tensors(checkpoint_dir, state_keys, device).items()
    }
    attention.load_state_dict(state_dict, strict=False)
    for linear in (attention.wq_a, attention.wq_b, attention.wkv, attention.wo_a, attention.wo_b):
        if getattr(linear, "scale", None) is not None:
            linear.weight.scale = linear.scale
    return attention


def _layer0_hash_selected_experts(
    checkpoint_dir: Path,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    tid2eid = _load_checkpoint_tensors(checkpoint_dir, ["layers.0.ffn.gate.tid2eid"], device)[
        "layers.0.ffn.gate.tid2eid"
    ]
    selected = tid2eid[input_ids].reshape(-1)
    return tid2eid, torch.unique(selected, sorted=True).tolist()


def _load_demo_single_layer_logits(
    checkpoint_dir: Path,
    input_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    model_module = demo_dsv4._import_checkpoint_model(checkpoint_dir)
    demo_dsv4._patch_checkpoint_model(model_module)
    model_config = demo_dsv4._load_model_args(
        checkpoint_dir, layers=1, batch_size=input_ids.shape[0], seq_len=seq_len
    )
    _configure_demo_quant_globals(model_module, model_config)

    with demo_dsv4._torch_defaults(torch.bfloat16, device):
        model_args = model_module.ModelArgs(**model_config)
        model = model_module.Transformer(model_args).eval()
        demo_dsv4._convert_wo_a_to_fp8(model)
        demo_dsv4._load_checkpoint_subset(model, checkpoint_dir, device)
        with torch.inference_mode():
            return model(input_ids, start_pos=0).detach().float().cpu()


def _load_ad_checkpoint_tensor(
    checkpoint_dir: Path,
    checkpoint_key: str,
    target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tensor = _load_checkpoint_tensors(checkpoint_dir, [checkpoint_key], device)[checkpoint_key]
    if checkpoint_key.startswith("layers.0.ffn.experts.") and checkpoint_key.endswith(".weight"):
        scale_key = checkpoint_key.removesuffix(".weight") + ".scale"
        scale = _load_checkpoint_tensors(checkpoint_dir, [scale_key], device)[scale_key]
        return demo_dsv4._dequant_fp4_weight(tensor, scale, target.dtype)
    if tensor.dtype == torch.float8_e4m3fn:
        scale_key = checkpoint_key.removesuffix(".weight") + ".scale"
        scale = _load_checkpoint_tensors(checkpoint_dir, [scale_key], device)[scale_key]
        return demo_dsv4._dequant_fp8_weight(tensor, scale, target.dtype)
    return tensor.to(dtype=target.dtype, device=target.device)


def _remap_layer0_tid2eid(
    tid2eid: torch.Tensor,
    target: torch.Tensor,
    input_ids: torch.Tensor,
    remap: dict[int, int],
) -> torch.Tensor:
    remapped = torch.zeros_like(target)
    for token_id in input_ids.flatten().tolist():
        remapped[token_id] = torch.tensor(
            [remap[int(expert)] for expert in tid2eid[token_id].tolist()],
            dtype=target.dtype,
            device=target.device,
        )
    return remapped


def _build_reduced_ad_single_layer_logits(
    checkpoint_dir: Path,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    selected_experts: list[int],
    device: torch.device,
) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    config = _hf_layer0_reduced_expert_ad_config(checkpoint_dir, seq_len, len(selected_experts))
    with demo_dsv4._torch_defaults(torch.bfloat16, device):
        model = DeepseekV4ForCausalLM(config).eval().to(device)

    remap = {expert_id: local_id for local_id, expert_id in enumerate(selected_experts)}
    state_dict = {}
    for key, target in model.state_dict().items():
        if key == "layers.0.ffn.gate.tid2eid":
            state_dict[key] = _remap_layer0_tid2eid(tid2eid, target, input_ids, remap)
            continue
        if key == "layers.0.ffn.gate.weight":
            tensor = _load_checkpoint_tensors(checkpoint_dir, [key], device)[key]
            state_dict[key] = tensor[selected_experts].to(dtype=target.dtype, device=target.device)
            continue
        if key.startswith("layers.0.ffn.experts."):
            parts = key.split(".")
            local_expert = int(parts[4])
            checkpoint_key = ".".join(parts[:4] + [str(selected_experts[local_expert])] + parts[5:])
        else:
            checkpoint_key = key
        state_dict[key] = _load_ad_checkpoint_tensor(checkpoint_dir, checkpoint_key, target, device)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.inference_mode(), demo_dsv4._torch_defaults(torch.bfloat16, device):
        logits = model(input_ids=input_ids, position_ids=position_ids).logits[:, -1]
        return logits.detach().float().cpu()


def _rmse_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    return (rmse_diff / rmse_ref.clamp_min(1e-12)).item()


def test_real_checkpoint_layer0_attention_quant_transform_loads() -> None:
    checkpoint_dir = _require_real_checkpoint()
    device = torch.device("cuda:0")
    seq_len = 4

    gm, num_matches = _build_quantized_ad_layer0_attention(checkpoint_dir, device, seq_len)

    quant_nodes = [
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
        or is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    ]
    state = gm.state_dict()

    assert num_matches == 5
    assert len(quant_nodes) == 5
    assert state["layers.0.attn.wq_a.weight"].dtype == torch.float8_e4m3fn
    assert state["layers.0.attn.wq_a.weight_scale_inv"].is_floating_point()
    assert state["layers.0.attn.wo_a.weight"].dtype == torch.float8_e4m3fn
    assert state["layers.0.attn.wo_a.weight_scale_inv"].is_floating_point()


def test_real_checkpoint_layer0_attention_prefill_matches_demo_reference() -> None:
    checkpoint_dir = _require_real_checkpoint()
    device = torch.device("cuda:0")
    seq_len = 4
    torch.manual_seed(1234)

    ad_attention, _ = _build_quantized_ad_layer0_attention(checkpoint_dir, device, seq_len)
    demo_attention = _build_demo_layer0_attention(checkpoint_dir, device, seq_len)
    x = torch.randn(1, seq_len, 4096, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.inference_mode(), demo_dsv4._torch_defaults(torch.bfloat16, device):
        expected = demo_attention(x.clone(), start_pos=0)
        actual = ad_attention(x.clone(), position_ids)
        if isinstance(actual, tuple):
            actual = actual[0]

    ratio = _rmse_ratio(actual, expected)
    assert ratio < _LAYER0_ATTENTION_RMSE_RATIO_TOL, (
        "transformed AD FineGrained FP8 layer-0 attention diverges from the demo reference "
        f"(RMSE ratio {ratio:.6f}, tolerance {_LAYER0_ATTENTION_RMSE_RATIO_TOL})"
    )


def test_real_checkpoint_single_layer_prefill_logits_match_demo_reference() -> None:
    checkpoint_dir = _require_real_checkpoint()
    device = torch.device("cuda:0")
    input_ids = torch.tensor(_LAYER0_LOGITS_INPUT_IDS, device=device, dtype=torch.long)

    tid2eid, selected_experts = _layer0_hash_selected_experts(checkpoint_dir, input_ids, device)
    expected = _load_demo_single_layer_logits(checkpoint_dir, input_ids, device)
    gc.collect()
    torch.cuda.empty_cache()
    actual = _build_reduced_ad_single_layer_logits(
        checkpoint_dir, input_ids, tid2eid, selected_experts, device
    )

    diff = actual.float() - expected.float()
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ratio = _rmse_ratio(actual, expected)
    print(
        "single-layer logits selected_experts="
        f"{selected_experts}, RMSE={rmse:.8f}, RMSE ratio={ratio:.8f}"
    )
    assert ratio < _LAYER0_LOGITS_RMSE_RATIO_TOL, (
        "reduced-expert dense-dequant AD single-layer logits diverge from the demo reference "
        f"(selected experts {selected_experts}, RMSE {rmse:.6f}, RMSE ratio {ratio:.6f}, "
        f"tolerance {_LAYER0_LOGITS_RMSE_RATIO_TOL})"
    )
