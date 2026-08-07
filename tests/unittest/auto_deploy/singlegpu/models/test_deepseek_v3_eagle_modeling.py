# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the DeepSeek-V3/R1 native MTP (Eagle one-model) drafter in AutoDeploy.

These are the model-specific onboarding ladder for ``model_type == "deepseek_v3"``. They
protect the *new* MTP-drafter code added on top of the already-validated DeepSeek-V3 base
components (RMSNorm / MLA attention / MoE / decoder layer are covered by
``test_deepseek_custom.py`` and ``test_deepseek_v2_modeling.py``):

1. EagleConfig defaults for deepseek_v3 (incl. ``normalize_target_hidden_state=True``).
2. The ``model.layers.{N}.*`` -> drafter checkpoint-key conversion mapping.
3. The MTP layer prologue math (enorm/hnorm/concat/eh_proj) vs an AutoDeploy-op-free reference.
4. The full ``DeepSeekV3EagleLayer.forward`` composition order vs an independent reference.
5. The drafter export IO contract (``inputs_embeds``, ``position_ids``, ``hidden_states``).
6. The real-checkpoint key contract against DeepSeek-V3-Lite's MTP weight names.
"""

import json
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from test_common.llm_data import hf_id_to_local_model_dir
from transformers import PretrainedConfig

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (register custom ops)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek import (
    DeepSeekV3EagleLayer,
    build_deepseek_v3_eagle_layers,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleConfig,
    EagleDrafterForCausalLM,
)

DEEPSEEK_V3_LITE_HUB_ID = "deepseek-ai/DeepSeek-V3-Lite"


# =============================================================================
# Config helpers
# =============================================================================


class _SmallDeepSeekV3Config(PretrainedConfig):
    """Config subclass so ``model_type`` survives ``to_dict()`` (which reads the class attr)."""

    model_type = "deepseek_v3"


def _make_small_deepseek_v3_config(
    q_lora_rank=None,
    num_hidden_layers: int = 4,
    num_nextn_predict_layers: int = 1,
    torch_dtype: str = "float32",
) -> PretrainedConfig:
    """A small ``deepseek_v3`` base config that keeps the MLA / MoE structure intact.

    ``q_lora_rank`` selects the attention variant: ``None`` mirrors DeepSeek-V3-Lite
    (direct ``q_proj``); a positive value mirrors DeepSeek-V3/R1 (compressed
    ``q_a_proj``/``q_b_proj``). The MTP-specific code is identical for both.
    """
    config = _SmallDeepSeekV3Config()
    # attention / MLA
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.qk_nope_head_dim = 64
    config.qk_rope_head_dim = 32
    config.v_head_dim = 64
    config.kv_lora_rank = 128
    config.q_lora_rank = q_lora_rank
    config.hidden_size = 256
    config.rope_theta = 10000.0
    config.max_position_embeddings = 512
    config.attention_bias = False
    config.rope_scaling = None
    config.rms_norm_eps = 1e-6
    # MLP / MoE
    config.intermediate_size = 512
    config.hidden_act = "silu"
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 256
    config.n_shared_experts = 1
    config.routed_scaling_factor = 1.0
    config.n_group = 1
    config.topk_group = 1
    config.first_k_dense_replace = 1
    config.moe_layer_freq = 1
    # model
    config.num_hidden_layers = num_hidden_layers
    config.num_nextn_predict_layers = num_nextn_predict_layers
    config.vocab_size = 1000
    config.pad_token_id = 0
    config.initializer_range = 0.02
    config.torch_dtype = torch_dtype
    return config


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain DeepSeek-style RMSNorm reference (no AutoDeploy custom ops)."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * x).to(dtype)


# =============================================================================
# 1. EagleConfig defaults / checkpoint-conversion-mapping contract  (CPU)
# =============================================================================


def test_deepseek_v3_eagle_config_defaults():
    """from_base_config must expose the MTP drafter contract, incl. target final-norm.

    ``normalize_target_hidden_state=True`` is load-bearing: the PyTorch backend feeds the
    MTP layer the target's post-``model.norm`` hidden state, while AutoDeploy captures the
    pre-norm residual add, so the wrapper must apply the target final norm.
    """
    base = _make_small_deepseek_v3_config()
    cfg = EagleConfig.from_base_config(base, "deepseek_v3")

    assert cfg.load_embedding_from_target is True
    assert cfg.load_lm_head_from_target is True
    assert cfg.num_capture_layers == 1
    assert cfg.normalize_target_hidden_state is True
    assert cfg.layers_handle_final_norm is True


def test_deepseek_v3_checkpoint_conversion_mapping_single_mtp():
    """Single-MTP checkpoints remap ``model.layers.{num_hidden_layers}.*`` onto the drafter."""
    base = _make_small_deepseek_v3_config(num_hidden_layers=4, num_nextn_predict_layers=1)
    mapping = EagleConfig._deepseek_v3_checkpoint_conversion_mapping(base.to_dict())

    assert mapping[r"^model\.layers\.4\.enorm\."] == "model.layers.enorm."
    assert mapping[r"^model\.layers\.4\.hnorm\."] == "model.layers.hnorm."
    assert mapping[r"^model\.layers\.4\.eh_proj\."] == "model.layers.eh_proj."
    assert mapping[r"^model\.layers\.4\.shared_head\.norm\."] == "model.layers.shared_head_norm."
    # Catch-all routes the embedded decoder block under mtp_block.
    assert mapping[r"^model\.layers\.4\."] == "model.layers.mtp_block."


def test_deepseek_v3_checkpoint_conversion_mapping_multi_mtp():
    """Multi-MTP checkpoints index each drafter layer by its position."""
    base = _make_small_deepseek_v3_config(num_hidden_layers=4, num_nextn_predict_layers=2)
    mapping = EagleConfig._deepseek_v3_checkpoint_conversion_mapping(base.to_dict())

    assert mapping[r"^model\.layers\.4\.enorm\."] == "model.layers.0.enorm."
    assert mapping[r"^model\.layers\.4\."] == "model.layers.0.mtp_block."
    assert mapping[r"^model\.layers\.5\.enorm\."] == "model.layers.1.enorm."
    assert mapping[r"^model\.layers\.5\."] == "model.layers.1.mtp_block."


def test_deepseek_v3_checkpoint_conversion_mapping_requires_num_hidden_layers():
    """A checkpoint config without num_hidden_layers cannot place the MTP layer."""
    with pytest.raises(ValueError, match="num_hidden_layers"):
        EagleConfig._deepseek_v3_checkpoint_conversion_mapping({})


# =============================================================================
# 2. MTP layer math + composition equivalence  (CUDA)
# =============================================================================


@torch.no_grad()
def test_deepseek_v3_eagle_layer_prologue_matches_reference():
    """enorm/hnorm/concat/eh_proj must match a plain-PyTorch reference (no AutoDeploy ops).

    This is the faithful math check for the MTP prologue: the embedding branch is normed by
    enorm, the hidden branch by hnorm, they are concatenated embeds-first, then projected by
    the ``2*hidden -> hidden`` eh_proj.
    """
    device, dtype = "cuda", torch.float32
    torch.manual_seed(0)
    config = _make_small_deepseek_v3_config(torch_dtype="float32")
    layer = DeepSeekV3EagleLayer(config, layer_idx=config.num_hidden_layers).to(device, dtype)

    B, S, H = 2, 4, config.hidden_size
    hidden_states = torch.randn(B, S, H, device=device, dtype=dtype)
    inputs_embeds = torch.randn(B, S, H, device=device, dtype=dtype)
    eps = config.rms_norm_eps

    # AutoDeploy prologue (exactly what DeepSeekV3EagleLayer.forward computes before mtp_block)
    emb = layer.enorm(inputs_embeds)
    hid = layer.hnorm(hidden_states)
    ad_proj = torch.ops.auto_deploy.torch_linear_simple(
        torch.cat([emb, hid], dim=-1), layer.eh_proj.weight, None, tp_mode="rowwise", layer_type="mlp"
    )

    # AutoDeploy-op-free reference
    ref_emb = _ref_rms_norm(inputs_embeds, layer.enorm.weight, eps)
    ref_hid = _ref_rms_norm(hidden_states, layer.hnorm.weight, eps)
    ref_proj = F.linear(torch.cat([ref_emb, ref_hid], dim=-1), layer.eh_proj.weight)

    torch.testing.assert_close(ad_proj, ref_proj, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("q_lora_rank", [None, 128], ids=["no_qlora_v3lite", "qlora_r1"])
@torch.no_grad()
def test_deepseek_v3_eagle_layer_forward_wiring_matches_reference(q_lora_rank):
    """The full layer must compose prologue -> decoder block -> shared_head_norm in order.

    The reference re-expresses the intended MTP forward (per the PyTorch backend
    ``DeepseekV3MTP``) using the layer's own submodules. A dense decoder block (layer_idx
    below first_k_dense_replace) keeps the comparison free of MoE routing nondeterminism;
    the prologue/epilogue wiring is identical for dense and MoE blocks. Both q-LoRA variants
    are covered because the MTP machinery is q-projection agnostic.
    """
    device, dtype = "cuda", torch.float32
    torch.manual_seed(0)
    config = _make_small_deepseek_v3_config(q_lora_rank=q_lora_rank, torch_dtype="float32")
    # layer_idx=0 < first_k_dense_replace=1 -> dense mtp_block (smooth, deterministic).
    layer = DeepSeekV3EagleLayer(config, layer_idx=0).to(device, dtype)

    B, S, H = 2, 4, config.hidden_size
    hidden_states = torch.randn(B, S, H, device=device, dtype=dtype)
    inputs_embeds = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    eps = config.rms_norm_eps

    out = layer(hidden_states, inputs_embeds, position_ids)

    ref_emb = _ref_rms_norm(inputs_embeds, layer.enorm.weight, eps)
    ref_hid = _ref_rms_norm(hidden_states, layer.hnorm.weight, eps)
    ref_proj = F.linear(torch.cat([ref_emb, ref_hid], dim=-1), layer.eh_proj.weight)
    ref_block = layer.mtp_block(ref_proj, position_ids)
    ref_out = _ref_rms_norm(ref_block, layer.shared_head_norm.weight, eps)

    assert out.shape == (B, S, H)
    assert_rmse_close(out, ref_out, rmse_ratio_tol=0.02, msg="DeepSeek-V3 MTP layer wiring: ")


@torch.no_grad()
def test_deepseek_v3_eagle_layer_moe_block_runs_and_is_finite():
    """The production MTP layer_idx selects a MoE decoder block; it must run end-to-end."""
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    config = _make_small_deepseek_v3_config(torch_dtype="bfloat16")
    layer_idx = config.num_hidden_layers  # always >= first_k_dense_replace -> MoE
    layer = DeepSeekV3EagleLayer(config, layer_idx=layer_idx).to(device, dtype)
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek import DeepSeekV3MoE

    assert isinstance(layer.mtp_block.mlp, DeepSeekV3MoE)
    layer.mtp_block.mlp.gate.weight = torch.nn.Parameter(
        torch.randn_like(layer.mtp_block.mlp.gate.weight)
    )

    B, S, H = 2, 4, config.hidden_size
    hidden_states = torch.randn(B, S, H, device=device, dtype=dtype)
    inputs_embeds = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    out = layer(hidden_states, inputs_embeds, position_ids)
    assert out.shape == (B, S, H)
    assert torch.isfinite(out).all()


def test_deepseek_v3_build_eagle_layers_count_and_indexing():
    """build_deepseek_v3_eagle_layers must emit num_nextn_predict_layers MTP layers."""
    config = _make_small_deepseek_v3_config(num_hidden_layers=4, num_nextn_predict_layers=2)
    layers = build_deepseek_v3_eagle_layers(config)
    assert len(layers) == 2
    assert all(isinstance(layer, DeepSeekV3EagleLayer) for layer in layers)
    # MTP blocks are indexed starting at num_hidden_layers.
    assert [layer.mtp_block.layer_idx for layer in layers] == [4, 5]


# =============================================================================
# 3. Drafter export IO contract  (CUDA)
# =============================================================================


@torch.no_grad()
def test_deepseek_v3_eagle_drafter_exports():
    """The drafter EagleModel must export with (inputs_embeds, position_ids, hidden_states)."""
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    base = _make_small_deepseek_v3_config(torch_dtype="bfloat16")
    eagle_config = EagleConfig.from_base_config(base, "deepseek_v3")

    drafter = EagleDrafterForCausalLM(eagle_config).to(device, dtype).eval()
    inner = drafter.model  # EagleModel; mirrors DraftModelExportInfo's exported submodule

    B, S, H = 2, 4, base.hidden_size
    inputs_embeds = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    hidden_states = torch.randn(B, S, H, device=device, dtype=dtype)

    gm = torch_export_to_gm(inner, args=(inputs_embeds, position_ids, hidden_states), clone=True)
    out = gm(inputs_embeds, position_ids, hidden_states)
    out_t = out[0] if isinstance(out, (tuple, list)) else out
    assert out_t.shape == (B, S, H)
    assert torch.isfinite(out_t).all()


# =============================================================================
# 4. Real-checkpoint key contract  (CPU; reads weight index only, no weights)
# =============================================================================


def _apply_conversion_mapping(key: str, mapping: dict) -> str:
    """Mirror transformers' _checkpoint_conversion_mapping application (ordered re.sub)."""
    for pattern, repl in mapping.items():
        key = re.sub(pattern, repl, key)
    return key


def test_deepseek_v3_eagle_real_checkpoint_key_contract():
    """The conversion mapping must route real DeepSeek-V3-Lite MTP keys onto drafter names.

    Reads the bf16 checkpoint's safetensors index (key names only) and routes the MTP layer's
    keys through the production conversion mapping. Asserts the MTP-specific keys land on their
    dedicated drafter modules, the embedded decoder block lands under ``mtp_block.*``, and the
    checkpoint's own ``embed_tokens`` / ``shared_head.head`` (which the drafter loads from the
    target instead) are the documented extras that fall under ``mtp_block.*`` harmlessly. This
    protects the MTP remapping itself; base decoder/MoE param-name fidelity is covered by
    ``test_deepseek_custom.py``.
    """
    model_dir = hf_id_to_local_model_dir(DEEPSEEK_V3_LITE_HUB_ID)
    index = None
    if model_dir is not None:
        index = Path(model_dir) / "bf16" / "model.safetensors.index.json"
    if index is None or not index.is_file():
        pytest.skip(f"{DEEPSEEK_V3_LITE_HUB_ID} bf16 weight index not found (LLM_MODELS_ROOT)")

    weight_map = json.loads(index.read_text())["weight_map"]
    cfg = json.loads((Path(model_dir) / "bf16" / "config.json").read_text())
    num_hidden_layers = cfg["num_hidden_layers"]
    mtp_layer = num_hidden_layers  # single MTP layer at index num_hidden_layers
    prefix = f"model.layers.{mtp_layer}."

    src_keys = [k for k in weight_map if k.startswith(prefix)]
    assert src_keys, f"No MTP keys at model.layers.{mtp_layer} in {DEEPSEEK_V3_LITE_HUB_ID}/bf16"

    base = _make_small_deepseek_v3_config(num_hidden_layers=num_hidden_layers)
    mapping = EagleConfig._deepseek_v3_checkpoint_conversion_mapping(base.to_dict())
    mapped = {k: _apply_conversion_mapping(k, mapping) for k in src_keys}

    # MTP-specific modules get their own dedicated drafter names.
    expected_specific = {
        f"{prefix}enorm.weight": "model.layers.enorm.weight",
        f"{prefix}hnorm.weight": "model.layers.hnorm.weight",
        f"{prefix}eh_proj.weight": "model.layers.eh_proj.weight",
        f"{prefix}shared_head.norm.weight": "model.layers.shared_head_norm.weight",
    }
    for src, dst in expected_specific.items():
        assert src in mapped, f"checkpoint missing MTP key {src}"
        assert mapped[src] == dst, (src, mapped[src], dst)

    # The drafter loads embedding + lm_head from the target, so the checkpoint's own copies are
    # the only keys that route into the (nonexistent) mtp_block submodules and are ignored on load.
    target_loaded_extras = {f"{prefix}embed_tokens.weight", f"{prefix}shared_head.head.weight"}

    # Everything else (the embedded decoder block) routes under mtp_block.* preserving its suffix.
    for src, dst in mapped.items():
        if src in expected_specific or src in target_loaded_extras:
            continue
        assert dst == "model.layers.mtp_block." + src[len(prefix) :], (src, dst)

    # The decoder block's attention/MoE/layernorm keys are actually present.
    block_suffixes = {dst for src, dst in mapped.items() if ".mtp_block." in dst}
    assert any("self_attn" in s for s in block_suffixes)
    assert any("mlp.experts." in s for s in block_suffixes)
    assert any("input_layernorm" in s for s in block_suffixes)
