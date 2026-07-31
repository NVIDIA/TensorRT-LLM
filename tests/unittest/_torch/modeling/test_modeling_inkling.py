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
"""CPU-only structural tests for the Inkling text-tower bring-up.

These pin the in-repo config parsing, registration, and the exact
consumed/deferred weight accounting against the REAL NVFP4 checkpoint index (no
GPU, no 591 GB load). They are the cheapest tier of the bring-up validation
ladder and guarantee that no required text tensor (q/k norm, relative-bias,
short-conv, route/global-scale, unpadded-logit) can silently go missing before
the GPU load/replay stages run.

Run (inside the task container, after bootstrap):
    python -m pytest tests/unittest/_torch/modeling/test_modeling_inkling.py -v
"""

import json
import os
import struct

import pytest

CHECKPOINT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT),
    reason=f"Inkling checkpoint not present at {CHECKPOINT}")


def _load_index_keys(ckpt: str) -> set:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return set(json.load(f)["weight_map"].keys())


def _load_exclude_modules(ckpt: str) -> set:
    with open(os.path.join(ckpt, "hf_quant_config.json")) as f:
        q = json.load(f)["quantization"]
    return set(q.get("exclude_modules", []))


def _safetensors_shape(ckpt: str, key: str):
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][key]
    with open(os.path.join(ckpt, shard), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header[key]["shape"], header[key]["dtype"]


def test_config_registers_and_parses():
    """InklingConfig is registered and parses the real checkpoint config.json."""
    from tensorrt_llm._torch.configs.inkling import (InklingConfig,
                                                     InklingTextConfig)
    # Importing the model module registers the auto-model + weight mapper.
    import tensorrt_llm._torch.models.modeling_inkling  # noqa: F401
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
    assert "InklingForConditionalGeneration" in MODEL_CLASS_MAPPING

    cfg = InklingConfig.from_pretrained(CHECKPOINT)
    assert cfg.model_type == "inkling_mm_model"
    tc = cfg.text_config
    assert isinstance(tc, InklingTextConfig)
    assert tc.num_hidden_layers == 66
    assert tc.hidden_size == 6144
    assert tc.num_attention_heads == 64
    assert tc.num_key_value_heads == 8
    assert tc.head_dim == 128
    assert tc.vocab_size == 201024
    assert tc.unpadded_vocab_size == 200058
    assert tc.logits_mup_width_multiplier == 24.0
    assert tc.use_embed_norm is True
    assert tc.dense_mlp_idx == 2
    assert tc.n_routed_experts == 256
    assert tc.num_experts_per_tok == 6
    assert tc.n_shared_experts == 2
    assert tc.sliding_window_size == 512
    assert tc.swa_num_key_value_heads == 16


def test_layer_classification():
    """Dense = {0,1}; global (full-attention) = every 6th layer at offset 5."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    tc = InklingConfig.from_pretrained(CHECKPOINT).text_config
    assert [n for n in range(tc.num_hidden_layers)
            if tc.is_dense_layer(n)] == [0, 1]
    globals_ = [
        n for n in range(tc.num_hidden_layers) if not tc.is_local_layer(n)
    ]
    assert globals_ == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
    # local geometry: 16 kv-heads + 512 window; global: 8 kv-heads + no window.
    assert tc.layer_num_kv_heads(0) == 16 and tc.layer_window(0) == 512
    assert tc.layer_num_kv_heads(5) == 8 and tc.layer_window(5) is None


def test_text_weight_accounting():
    """Every checkpoint key is consumed-text or intentionally-deferred; the text
    tower is exactly and fully covered (no missing, no unaccounted)."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        inkling_account_checkpoint, inkling_nvfp4_expert_layers)

    tc = InklingConfig.from_pretrained(CHECKPOINT).text_config
    exclude = _load_exclude_modules(CHECKPOINT)
    all_keys = _load_index_keys(CHECKPOINT)

    acct = inkling_account_checkpoint(all_keys, tc, exclude)
    assert not acct["unaccounted"], sorted(acct["unaccounted"])[:10]
    assert not acct["missing"], sorted(acct["missing"])[:10]
    # Every deferred key is audio / vision / mtp.
    assert all(
        k.startswith(("model.audio.", "model.visual.", "model.mtp."))
        for k in acct["deferred"])
    assert len(acct["consumed_text"]) + len(acct["deferred"]) == len(all_keys)

    # NVFP4 routed experts are exactly layers 3..65 (layer 2 experts are bf16).
    nvfp4_layers = inkling_nvfp4_expert_layers(tc, exclude)
    assert nvfp4_layers == list(range(3, 66))
    assert f"model.llm.layers.2.mlp.experts" in exclude
    assert f"model.llm.layers.3.mlp.experts" not in exclude


def test_checkpoint_tensor_shapes_match_geometry():
    """Sample checkpoint tensors have the shapes the modules will construct."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    tc = InklingConfig.from_pretrained(CHECKPOINT).text_config
    hd, hidden = tc.head_dim, tc.hidden_size

    # q/k/v/r projection out-dims (layer 0 is local: 16 kv-heads).
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wq_du.weight")
    assert shape == [tc.num_attention_heads * hd, hidden]
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wk_dv.weight")
    assert shape == [tc.swa_num_key_value_heads * hd, hidden]
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.5.attn.wk_dv.weight")
    assert shape == [tc.num_key_value_heads * hd, hidden]  # global layer
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wr_du.weight")
    assert shape == [tc.num_attention_heads * tc.d_rel, hidden]

    # short conv depthwise weight: [channels, 1, kernel].
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.k_sconv.weight")
    assert shape == [tc.swa_num_key_value_heads * hd, 1, tc.sconv_kernel_size]

    # NVFP4 routed experts: [E, 2*inter, hidden/2] packed uint8 + block scale.
    shape, dtype = _safetensors_shape(
        CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight")
    assert shape == [tc.n_routed_experts, 2 * tc.intermediate_size, hidden // 2]
    assert dtype in ("U8", "UINT8")
    shape, _ = _safetensors_shape(
        CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight.scale")
    assert shape == [tc.n_routed_experts, 2 * tc.intermediate_size, hidden // 16]

    # router weight covers routed + shared experts.
    shape, _ = _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.gate.weight")
    assert shape == [tc.n_routed_experts + tc.n_shared_experts, hidden]


# --------------------------------------------------------------------------- #
# Stage 9 MTP -- static-tier (validation_tier=static) load-accounting.         #
# These pin the surfaced mtp_config + the exact consumed/deferred MTP weight    #
# accounting against the REAL checkpoint index (no GPU), exactly mirroring the  #
# text-tower accounting above. They are acceptance-criteria Stage 9 item 1:     #
# all 8 draft depths load-accounted, dtypes recorded (BF16), the MTP block is   #
# not silently quantized from exclude_modules, and the text weights remain      #
# exactly accounted. The runtime module build + spec-decode wiring (criteria    #
# 2-8, GPU tiers) are the next milestones and are NOT exercised here.           #
# --------------------------------------------------------------------------- #


def test_mtp_config_parses():
    """InklingMTPConfig surfaces the checkpoint mtp_config + per-depth banding."""
    from tensorrt_llm._torch.configs.inkling import (InklingConfig,
                                                     InklingMTPConfig)
    cfg = InklingConfig.from_pretrained(CHECKPOINT)
    assert cfg.has_mtp is True
    mtp = cfg.mtp_config
    assert isinstance(mtp, InklingMTPConfig)
    assert mtp.num_nextn_predict_layers == 8
    assert mtp.chain_hidden_post_norm is False
    assert mtp.local_layer_ids == [0, 2, 4, 5, 6, 7]
    # Per-depth banded attention: depths in local_layer_ids are sliding-window
    # (SWA: 16 kv-heads, window 512); depths 1 and 3 are full attention (8 kv).
    tc = cfg.text_config
    assert mtp.mtp_local_extent == tc.sliding_window_size == 512
    for d in (0, 2, 4, 5, 6, 7):
        assert mtp.is_local_depth(d) is True
        assert mtp.depth_num_kv_heads(d) == tc.swa_num_key_value_heads == 16
        assert mtp.depth_window(d) == 512
    for d in (1, 3):
        assert mtp.is_local_depth(d) is False
        assert mtp.depth_num_kv_heads(d) == tc.num_key_value_heads == 8
        assert mtp.depth_window(d) is None


def test_mtp_weight_accounting():
    """All 8 MTP draft depths (160 keys) are load-accounted; nothing missing or
    unaccounted; the text tower stays exactly accounted."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        inkling_account_checkpoint, inkling_expected_mtp_keys)

    cfg = InklingConfig.from_pretrained(CHECKPOINT)
    tc = cfg.text_config
    exclude = _load_exclude_modules(CHECKPOINT)
    all_keys = _load_index_keys(CHECKPOINT)

    # 8 depths x 20 keys/depth = 160 expected MTP keys.
    expected_mtp = inkling_expected_mtp_keys(cfg.mtp_config)
    assert len(expected_mtp) == 160
    ckpt_mtp = {k for k in all_keys if k.startswith("model.mtp.")}
    assert ckpt_mtp == expected_mtp  # exact match: none missing, none extra

    acct = inkling_account_checkpoint(all_keys, tc, exclude,
                                      mtp_config=cfg.mtp_config)
    assert not acct["unaccounted"], sorted(acct["unaccounted"])[:10]
    assert not acct["missing"], sorted(acct["missing"])[:10]
    assert acct["consumed_mtp"] == expected_mtp
    # With MTP accounted, `deferred` no longer holds any MTP key (audio/visual only).
    assert all(k.startswith(("model.audio.", "model.visual."))
               for k in acct["deferred"])
    # Whole checkpoint is exactly partitioned: text + mtp + (audio/visual).
    assert (len(acct["consumed_text"]) + len(acct["consumed_mtp"])
            + len(acct["deferred"]) == len(all_keys))

    # The text tower accounting is UNCHANGED by the MTP extension (backward-compat:
    # calling without mtp_config keeps MTP in `deferred`, text identical).
    acct_text = inkling_account_checkpoint(all_keys, tc, exclude)
    assert acct_text["consumed_text"] == acct["consumed_text"]
    assert not acct_text["unaccounted"] and not acct_text["missing"]


def test_mtp_tensors_are_bf16_and_unquantized():
    """Every MTP tensor is BF16 in the checkpoint and NONE is in exclude_modules
    -- so the draft MUST be built UNQUANTIZED (quant state is not inferable from
    the exclude list; the quantizer simply never touched the MTP block)."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        inkling_expected_mtp_keys)

    cfg = InklingConfig.from_pretrained(CHECKPOINT)
    exclude = _load_exclude_modules(CHECKPOINT)
    expected_mtp = inkling_expected_mtp_keys(cfg.mtp_config)

    # dtype recorded as loaded from the checkpoint: all 160 are BF16.
    for key in sorted(expected_mtp):
        _, dtype = _safetensors_shape(CHECKPOINT, key)
        assert dtype == "BF16", f"{key} is {dtype}, expected BF16"

    # not silently quantized: no MTP module path appears in exclude_modules
    # (and, symmetrically, no exclude entry targets an mtp path).
    assert not any("mtp" in e.lower() for e in exclude)


def test_mtp_per_depth_banding_matches_checkpoint_shapes():
    """The checkpoint tensor shapes encode the per-depth banding the surfaced
    config derives: local depths carry 16 SWA kv-heads (wk_dv out = 2048), global
    depths 1 and 3 carry 8 kv-heads (wk_dv out = 1024). input_proj is
    [hidden, 2*hidden]."""
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    cfg = InklingConfig.from_pretrained(CHECKPOINT)
    mtp, hidden, hd = cfg.mtp_config, cfg.text_config.hidden_size, cfg.text_config.head_dim

    for d in range(mtp.num_nextn_predict_layers):
        wk = f"model.mtp.layers.{d}.transformer_block.attn.wk_dv.weight"
        shape, _ = _safetensors_shape(CHECKPOINT, wk)
        assert shape == [mtp.depth_num_kv_heads(d) * hd, hidden], (d, shape)
        # input_proj consumes concat(hidden_norm(hidden), embed_norm(embed)).
        ip = f"model.mtp.layers.{d}.input_proj.weight"
        shape, _ = _safetensors_shape(CHECKPOINT, ip)
        assert shape == [hidden, 2 * hidden], (d, shape)
