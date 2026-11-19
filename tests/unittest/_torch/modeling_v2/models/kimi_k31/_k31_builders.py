# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVIDIA NVFP4 export of Kimi K3.1 as routing and the model loader read it, for the family tests that build a
server from it (the routing tests and the NVFP4 default target's cache-layout tests).

``checkpoint_dir`` writes the export's config.json and hf_quant_config.json (93 layers, the ModelOpt per-layer
recipes and exclusions); ``on_gpu`` reports a GPU of a given capability, also on a host without one;
``HfConfigOnly`` reads the config as the HF checkpoint loader does, and nothing else.
"""

import json

import torch

import tensorrt_llm._utils


def on_gpu(monkeypatch, capability):
    """Report a GPU of ``capability``, also on a host without one: routing reads the capability,
    config validators read the SM through ``get_sm_version``, which is fixed to -1 in a process
    that started without a GPU (the CPU stage)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(
        tensorrt_llm._utils, "get_sm_version", lambda: capability[0] * 10 + capability[1]
    )


# The export's exclusions, as its hf_quant_config.json lists them.
EXCLUDE = [
    "*block_sparse_moe.gate*",
    "*block_sparse_moe.routed_expert_up_proj*",
    "*mlp.routed_expert_up_proj*",
    "*block_sparse_moe.routed_expert_down_proj*",
    "*mlp.routed_expert_down_proj*",
    "*block_sparse_moe.routed_expert_norm*",
    "*mlp.routed_expert_norm*",
    "*self_attn.q_conv1d*",
    "*self_attn.k_conv1d*",
    "*self_attn.v_conv1d*",
    "*self_attn.f_b_proj*",
    "*self_attn.o_norm*",
    "*self_attn.q_a_layernorm*",
    "*self_attn.kv_a_layernorm*",
    "*_res_proj*",
    "*_res_norm*",
    "*layernorm*",
    "language_model.model.layers.0.mlp.*",
    "layers.0.mlp.*",
    "model.layers.0.mlp.*",
    "language_model.lm_head",
    "lm_head",
    "vision_tower*",
    "mm_projector*",
]


PRODUCER = {"name": "modelopt", "version": "0.49.0"}


def quantized_layers(num_layers=8, mla_layers=(4, 8), shared=True) -> dict:
    """Per-layer recipes under the module spellings the export lists (three prefixes, both MoE
    block names). ``mla_layers`` numbers layers from 1; ``shared`` False leaves the shared experts
    out (BF16)."""
    kda = [i for i in range(num_layers) if i + 1 not in mla_layers]
    layers = {}
    for prefix in ("language_model.model.", "model.", ""):
        for i in range(1, num_layers):
            for block in ("block_sparse_moe", "mlp"):
                layers[f"{prefix}layers.{i}.{block}.experts"] = {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                }
                for proj in ("gate_proj", "up_proj", "gate_up_proj", "down_proj") if shared else ():
                    layers[f"{prefix}layers.{i}.{block}.shared_experts.{proj}"] = {
                        "quant_algo": "MXFP8"
                    }
        for i in kda:
            for proj in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "b_proj", "f_a_proj"):
                layers[f"{prefix}layers.{i}.self_attn.{proj}"] = {"quant_algo": "MXFP8"}
            layers[f"{prefix}layers.{i}.self_attn.fused_qkvg_proj"] = {"quant_algo": "MXFP8"}
            layers[f"{prefix}layers.{i}.self_attn.in_proj_qkvgfab"] = {"quant_algo": "MXFP8"}
    return layers


def inline_quantization_config(layers: dict) -> dict:
    """config.json's quantization_config: the flat ModelOpt form."""
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "producer": PRODUCER,
        "group_size": 16,
        "ignore": list(EXCLUDE),
        "quantized_layers": layers,
    }


def hf_quant_config(layers: dict) -> dict:
    return {
        "producer": PRODUCER,
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": list(EXCLUDE),
            "quantized_layers": layers,
        },
    }


SM103, SM107 = (10, 3), (10, 7)


#: The MLA layers of the checkpoint (1-indexed), each with a DSA indexer; KDA on the rest.
CHECKPOINT_MLA_LAYERS = [*range(4, 93, 4), 93]


def checkpoint_dir(tmp_path) -> str:
    """A directory with the export's config.json and hf_quant_config.json, 93 layers."""
    kda_layers = [i for i in range(1, 94) if i not in CHECKPOINT_MLA_LAYERS]
    layers = quantized_layers(93, CHECKPOINT_MLA_LAYERS)
    text = dict(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        hidden_size=7168,
        num_hidden_layers=93,
        num_attention_heads=96,
        num_key_value_heads=96,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=2,
        first_k_dense_replace=1,
        moe_intermediate_size=3072,
        routed_expert_hidden_size=3584,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=12,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vocab_size=163840,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            full_attn_layers=CHECKPOINT_MLA_LAYERS,
            kda_layers=kda_layers,
            head_dim=128,
            num_heads=96,
        ),
        dsa_config=dict(
            dsa_layers=CHECKPOINT_MLA_LAYERS,
            index_topk=2048,
            index_head_dim=128,
            index_num_heads=32,
            index_rope_dim=0,
            index_k_norm_type="layernorm",
            indexer_use_fp8=False,
            index_topk_freq=1,
            index_skip_topk_offset=1,
        ),
    )
    config = dict(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=text,
        vision_config=dict(vt_hidden_size=1024, vt_num_attention_heads=12, vt_num_hidden_layers=27),
        quantization_config=inline_quantization_config(layers),
    )
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    (path / "hf_quant_config.json").write_text(json.dumps(hf_quant_config(layers)))
    return str(path)


class HfConfigOnly:
    """A checkpoint loader that reads the config as the HF loader does, and nothing else."""

    def load_config(self, checkpoint_dir, **kwargs):
        from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader

        return HfConfigLoader().load(checkpoint_dir, **kwargs)
