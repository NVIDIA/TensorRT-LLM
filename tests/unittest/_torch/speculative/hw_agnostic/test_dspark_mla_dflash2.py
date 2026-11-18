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
"""A DFlash 2 drafter on the MLA DSpark backbone, from the released checkpoint (CPU-only).

The reference is RadixArk/Kimi-K3.1-Dflash2-Preview as published: block_size 5, no
``embed_tokens`` / ``lm_head`` tensors, no ``shift_label``, ``is_causal`` true and a 1024-token
sliding window. These tests cover the decisions the loader and the worker make from that config
-- slot layout, block width, embedding sharing, block-causal mask and window -- without building
the drafter or loading weights. Every non-DFlash 2 drafter must resolve exactly as before.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, dspark_layer_window_size
from tensorrt_llm._torch.models.modeling_dspark import (
    GQADSparkForCausalLM,
    MLADSparkForCausalLM,
    _causal_block_kv_bounds,
    _dflash2_shares_target_embedding,
    _DSparkHeadMixin,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs, is_dflash2_checkpoint

pytestmark = pytest.mark.cpu_only

# RadixArk/Kimi-K3.1-Dflash2-Preview config.json, as published (fields the drafter reads).
RELEASED_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "block_size": 5,
    "dflash_config": {
        "attention_mode": "mla",
        "aux_hidden_stream": "attn_res",
        "block_size": 5,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 163824,
        "selector_rank": 256,
        "selector_top_k": 32,
        "target_layer_ids": [7, 23, 51, 67, 83, 91],
    },
    "hidden_size": 7168,
    "is_causal": True,
    "kv_lora_rank": 512,
    "layer_types": ["sliding_attention"] * 5,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_hidden_layers": 5,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "sliding_window": 1024,
    "use_sliding_window": True,
    "v_head_dim": 128,
    "vocab_size": 163840,
}

# The tensor names of the released model.safetensors, one layer shown (86 tensors in all).
RELEASED_WEIGHT_NAMES = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "layers.0.attention_conv.base_kernel",
    "layers.0.attention_conv.kernel_projection.weight",
    "layers.0.mlp_conv.base_kernel",
    "layers.0.mlp_conv.kernel_projection.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_a_proj.weight",
    "layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "layers.0.mlp.gate_proj.weight",
)

# A DSpark (not DFlash 2) MLA drafter config: no block convolutions, no selector.
DSPARK_CONFIG = {
    "architectures": ["K3DSparkModel"],
    "block_size": 7,
    "dflash_config": {
        "block_size": 7,
        "mask_token_id": 163837,
        "projector_type": "dspark",
        "target_layer_ids": [23, 47, 71, 87, 91],
    },
    "num_hidden_layers": 5,
    "sliding_window": 1024,
    "layer_types": ["sliding_attention"] * 5,
}


def _is_dflash2(config: dict) -> bool:
    return is_dflash2_checkpoint(config.get("architectures"), config.get("dflash_config") or {})


class _MlaDraft:
    """The MLADSparkForCausalLM state the DFlash 2 resolution reads, without the module.

    Reuses the real methods: the block attention resolution of the DFlash base and the MLA
    drafter's own DFlash 2 resolution, window choice and target sharing.
    """

    _get_attention_mask_args = DFlashForCausalLM._get_attention_mask_args
    _resolve_block_attention = DFlashForCausalLM._resolve_block_attention
    _init_dflash2_block_attention = MLADSparkForCausalLM._init_dflash2_block_attention
    _dflash2_attn_window = MLADSparkForCausalLM._dflash2_attn_window
    _check_dflash2_window_backend = MLADSparkForCausalLM._check_dflash2_window_backend
    load_weights_from_target_model = MLADSparkForCausalLM.load_weights_from_target_model

    def __init__(
        self, config: dict, attention_backend: str = "CUTEDSL", context_kv_only: bool = False
    ):
        dflash_config = config.get("dflash_config") or {}
        num_layers = config["num_hidden_layers"]
        self.config = SimpleNamespace(**config)
        self._is_dflash2 = _is_dflash2(config)
        self._uses_lilicorr = False
        self._sliding_layers_causal = False
        self._layer_windows = [
            dspark_layer_window_size(
                bool(dflash_config.get("use_swa", False)),
                int(dflash_config.get("swa_window_size", 0) or 0),
                config.get("layer_types"),
                i,
            )
            for i in range(num_layers)
        ]
        self.model = SimpleNamespace(layers=[None] * num_layers)
        self.dflash_attention_backend = attention_backend
        self.context_kv_only = context_kv_only
        self._shares_target_embedding = False
        self.draft_model_full = SimpleNamespace(
            model=SimpleNamespace(embed_tokens="own embedding"), lm_head=None
        )
        self.lm_head = None
        self._init_dflash2_block_attention()
        self._check_dflash2_window_backend()


class _Heads(_DSparkHeadMixin):
    """The DSpark heads of the MLA drafter."""

    _dflash2_defaults_to_dflash_layout = MLADSparkForCausalLM._dflash2_defaults_to_dflash_layout

    def __init__(self, config: dict):
        self._is_dflash2 = _is_dflash2(config)
        self._init_dspark_heads(SimpleNamespace(**config))


class _GqaHeads(_Heads):
    """The DSpark heads of the GQA drafter."""

    _dflash2_defaults_to_dflash_layout = GQADSparkForCausalLM._dflash2_defaults_to_dflash_layout


# The released config without the MLA shape: a DFlash 2 checkpoint the GQA drafter would serve.
_MLA_KEYS = ("kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
GQA_DFLASH2_CONFIG = {k: v for k, v in RELEASED_CONFIG.items() if k not in _MLA_KEYS}


@pytest.fixture
def draft_attn_window_env(monkeypatch):
    """Set or clear TLLM_DSPARK_DRAFT_ATTN_WINDOW around the lru_cached readers."""

    def set_window(value):
        if value is None:
            monkeypatch.delenv("TLLM_DSPARK_DRAFT_ATTN_WINDOW", raising=False)
        else:
            monkeypatch.setenv("TLLM_DSPARK_DRAFT_ATTN_WINDOW", str(value))
        modeling_dspark._draft_attn_window.cache_clear()
        modeling_dspark._draft_attn_window_env.cache_clear()

    yield set_window
    modeling_dspark._draft_attn_window.cache_clear()
    modeling_dspark._draft_attn_window_env.cache_clear()


# --- detection ----------------------------------------------------------------------------------


def test_released_checkpoint_is_dflash2_and_dspark_is_not():
    assert _is_dflash2(RELEASED_CONFIG)
    assert not _is_dflash2(DSPARK_CONFIG)


# --- slot layout (shift_label) -----------------------------------------------------------------


def test_dflash2_without_shift_label_uses_the_dflash_layout():
    """The released config sets no shift_label: slots 1..K, as the checkpoint was trained."""
    assert _Heads(RELEASED_CONFIG)._dspark_shift_label is False


@pytest.mark.parametrize("value", [True, False])
def test_explicit_shift_label_wins_for_dflash2(value):
    config = json.loads(json.dumps(RELEASED_CONFIG))
    config["dflash_config"]["shift_label"] = value
    assert _Heads(config)._dspark_shift_label is value


def test_dspark_without_shift_label_keeps_the_shift_label_default():
    assert _Heads(DSPARK_CONFIG)._dspark_shift_label is True


def test_gqa_dflash2_without_shift_label_keeps_the_shift_label_default():
    """Only the MLA drafter defaults a DFlash 2 checkpoint to the DFlash layout."""
    assert _is_dflash2(GQA_DFLASH2_CONFIG)
    assert _GqaHeads(GQA_DFLASH2_CONFIG)._dspark_shift_label is True


def test_dspark_explicit_false_still_selects_the_legacy_layout():
    config = json.loads(json.dumps(DSPARK_CONFIG))
    config["dflash_config"]["shift_label"] = False
    assert _Heads(config)._dspark_shift_label is False


# --- embedding sharing --------------------------------------------------------------------------


def test_released_checkpoint_shares_the_target_embedding():
    assert _dflash2_shares_target_embedding(True, RELEASED_WEIGHT_NAMES)


@pytest.mark.parametrize("name", ["embed_tokens.weight", "model.embed_tokens.weight"])
def test_dflash2_checkpoint_with_its_own_embedding_keeps_it(name):
    assert not _dflash2_shares_target_embedding(True, RELEASED_WEIGHT_NAMES + (name,))


def test_non_dflash2_drafter_never_shares_the_target_embedding():
    """The 'own embedding required' rule of the DSpark MLA drafters is unchanged."""
    assert not _dflash2_shares_target_embedding(False, RELEASED_WEIGHT_NAMES)


def test_target_sharing_takes_the_embedding_only_when_flagged():
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens="target embedding"), lm_head="target head"
    )

    shared = _MlaDraft(RELEASED_CONFIG)
    shared._shares_target_embedding = True
    shared.load_weights_from_target_model(target)
    assert shared.draft_model_full.model.embed_tokens == "target embedding"
    assert shared.lm_head == shared.draft_model_full.lm_head == "target head"

    own = _MlaDraft(DSPARK_CONFIG)
    own.load_weights_from_target_model(target)
    assert own.draft_model_full.model.embed_tokens == "own embedding"
    assert own.lm_head == "target head"


def test_target_sharing_fails_loudly_without_a_target_embedding():
    draft = _MlaDraft(RELEASED_CONFIG)
    draft._shares_target_embedding = True
    with pytest.raises(ValueError, match="no model.embed_tokens to share"):
        draft.load_weights_from_target_model(SimpleNamespace(model=None, lm_head="head"))


def test_context_kv_only_drafter_builds_no_dflash2_modules():
    """A context server's drafter keeps fc / hidden_norm / kv_a only and never drafts."""
    draft = _MlaDraft(RELEASED_CONFIG)
    draft.context_kv_only = True
    weights = {
        "fc.weight": "fc",
        "hidden_norm.weight": "norm",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "kv_a",
    }
    assert MLADSparkForCausalLM._load_dflash2_weights(draft, weights) is weights


# --- block attention: causal mask and window ----------------------------------------------------


def test_released_config_resolves_block_causal_with_the_config_window():
    draft = _MlaDraft(RELEASED_CONFIG)
    assert draft._dflash2_block_causal is True
    # HF sliding_window 1024: the current token plus 1023 to the left.
    assert draft._dflash2_window_left == 1023
    assert draft.serves_wider_block is True


def test_non_causal_dflash2_does_not_serve_a_wider_block():
    config = dict(RELEASED_CONFIG, is_causal=False)
    draft = _MlaDraft(config)
    assert draft._dflash2_block_causal is False
    assert draft.serves_wider_block is False


def test_layers_that_disagree_are_rejected():
    config = dict(
        RELEASED_CONFIG,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
    )
    with pytest.raises(NotImplementedError, match="different"):
        _MlaDraft(config)


def test_non_dflash2_drafter_resolves_nothing(draft_attn_window_env):
    draft_attn_window_env(None)
    draft = _MlaDraft(DSPARK_CONFIG)
    assert draft._dflash2_block_causal is False
    assert draft._dflash2_window_left is None
    assert draft.serves_wider_block is False
    assert draft._dflash2_attn_window() == (0, "env")


def test_config_window_is_the_dflash2_default(draft_attn_window_env):
    draft_attn_window_env(None)
    assert _MlaDraft(RELEASED_CONFIG)._dflash2_attn_window() == (1023, "config")


@pytest.mark.parametrize("value", [0, 4096])
def test_env_window_overrides_the_config_window(draft_attn_window_env, value):
    draft_attn_window_env(value)
    assert _MlaDraft(RELEASED_CONFIG)._dflash2_attn_window() == (value, "env")


def test_trtllm_rejects_the_config_window(draft_attn_window_env):
    """The TRTLLM decode attends the whole context: not what the checkpoint was trained with."""
    draft_attn_window_env(None)
    with pytest.raises(ValueError, match="TLLM_DSPARK_DRAFT_ATTN_WINDOW=0"):
        _MlaDraft(RELEASED_CONFIG, attention_backend="TRTLLM")


@pytest.mark.parametrize("backend", ["CUTEDSL", "VANILLA"])
def test_window_backends_accept_the_config_window(draft_attn_window_env, backend):
    draft_attn_window_env(None)
    assert _MlaDraft(RELEASED_CONFIG, attention_backend=backend)._dflash2_window_left == 1023


def test_trtllm_runs_when_the_window_is_not_from_the_config(draft_attn_window_env):
    """The env var takes the window out of the config; other drafters have no config window."""
    draft_attn_window_env(None)
    _MlaDraft(DSPARK_CONFIG, attention_backend="TRTLLM")
    draft_attn_window_env(0)
    _MlaDraft(RELEASED_CONFIG, attention_backend="TRTLLM")


def test_trtllm_context_kv_only_drafter_is_not_rejected(draft_attn_window_env):
    """A context server's drafter only builds the context KV and never drafts."""
    draft_attn_window_env(None)
    _MlaDraft(RELEASED_CONFIG, attention_backend="TRTLLM", context_kv_only=True)


def test_env_window_still_applies_to_other_drafters(draft_attn_window_env):
    draft_attn_window_env(4096)
    assert _MlaDraft(DSPARK_CONFIG)._dflash2_attn_window() == (4096, "env")


# --- block-causal kv bounds ---------------------------------------------------------------------


def _visible_keys(ctx_len: int, owned_slots: list, query: int) -> int:
    """Brute force: the keys a block query sees, counted one by one in local key order."""
    keys = [("ctx", i) for i in range(ctx_len)] + [("blk", t) for t in owned_slots]
    return sum(1 for kind, t in keys if kind == "ctx" or t <= query)


@pytest.mark.parametrize("block_size", [1, 5, 8])
def test_causal_bounds_without_helix_match_brute_force(block_size):
    ctx_lens = torch.tensor([0, 3, 17, 1024], dtype=torch.int32)
    # The non-causal bound every block query starts with: the whole context and the whole block.
    kv_bounds = (ctx_lens + block_size).repeat_interleave(block_size)
    out = _causal_block_kv_bounds(kv_bounds, block_size).view(-1, block_size)
    assert out.dtype == torch.int32
    for b, ctx_len in enumerate(ctx_lens.tolist()):
        for j in range(block_size):
            assert out[b, j].item() == _visible_keys(ctx_len, list(range(block_size)), j)


def test_causal_bounds_with_owned_slots_match_brute_force():
    """Helix / context split: a rank holds some context keys and some of the block's keys."""
    block_size = 8
    generator = torch.Generator().manual_seed(0)
    local_ctx = torch.randint(0, 300, (6,), generator=generator, dtype=torch.int32)
    owned = torch.rand(6, block_size, 1, generator=generator) < 0.5
    owned[0] = False
    owned[1] = True
    kv_bounds = (local_ctx + owned.sum(dim=(1, 2)).to(torch.int32)).repeat_interleave(block_size)
    out = _causal_block_kv_bounds(kv_bounds, block_size, owned).view(-1, block_size)
    for b, ctx_len in enumerate(local_ctx.tolist()):
        slots = [t for t in range(block_size) if owned[b, t, 0]]
        for j in range(block_size):
            assert out[b, j].item() == _visible_keys(ctx_len, slots, j)


def test_causal_bounds_are_a_new_tensor():
    kv_bounds = torch.full((10,), 12, dtype=torch.int32)
    out = _causal_block_kv_bounds(kv_bounds, 5)
    assert out.data_ptr() != kv_bounds.data_ptr()
    assert kv_bounds.eq(12).all()


# --- block width --------------------------------------------------------------------------------


def _resolve_widths(max_draft_len: int, **draft_model):
    worker = object.__new__(DSparkWorker)
    worker.spec_config = MagicMock(max_draft_len=max_draft_len)
    draft = SimpleNamespace(**draft_model)
    worker._resolve_block_widths(draft)
    return worker._resolved_block_size, worker._compute_block_size


def test_block_causal_dflash2_runs_wider_than_its_checkpoint():
    """max_draft_len 7 in the DFlash layout computes 8 slots over a block_size 5 checkpoint."""
    assert _resolve_widths(7, block_size=5, _dspark_shift_label=False, serves_wider_block=True) == (
        5,
        8,
    )


def test_trained_width_needs_no_override():
    assert _resolve_widths(
        4, block_size=5, _dspark_shift_label=False, serves_wider_block=False
    ) == (5, 5)


def test_other_drafters_still_reject_a_wider_block():
    with pytest.raises(ValueError, match="trained with block_size=5"):
        _resolve_widths(7, block_size=5, _dspark_shift_label=False, serves_wider_block=False)
    with pytest.raises(ValueError, match="trained with block_size=5"):
        _resolve_widths(7, block_size=5, _dspark_shift_label=False)
    with pytest.raises(ValueError, match="trained with block_size=5"):
        _resolve_widths(7, block_size=5, _dspark_shift_label=True)


def _checkpoint(tmp_path, name, config):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _spec_config(draft_dir, max_draft_len, block_size=None):
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        speculative_config=DSparkDecodingConfig(
            max_draft_len=max_draft_len, speculative_model=draft_dir, block_size=block_size
        ),
    )
    return args.speculative_config


@pytest.mark.parametrize("max_draft_len", [4, 5, 7])
def test_dflash2_checkpoint_block_size_does_not_pin_the_draft_length(tmp_path, max_draft_len):
    spec = _spec_config(_checkpoint(tmp_path, "dflash2", RELEASED_CONFIG), max_draft_len)
    assert spec.block_size == max_draft_len
    assert spec.mask_token_id == 163824
    assert spec.target_layer_ids == [7, 23, 51, 67, 83, 91]


def test_explicit_block_size_must_still_equal_the_draft_length(tmp_path):
    with pytest.raises(ValueError, match="block_size must equal max_draft_len"):
        _spec_config(_checkpoint(tmp_path, "dflash2", RELEASED_CONFIG), 7, block_size=5)


def test_gqa_dflash2_checkpoint_block_size_still_pins_the_draft_length(tmp_path):
    """Only an MLA DFlash 2 checkpoint leaves block_size to max_draft_len."""
    draft_dir = _checkpoint(tmp_path, "gqa_dflash2", GQA_DFLASH2_CONFIG)
    assert _spec_config(draft_dir, 5).block_size == 5
    with pytest.raises(ValueError, match="block_size must equal max_draft_len"):
        _spec_config(draft_dir, 7)


def test_dspark_checkpoint_block_size_still_pins_the_draft_length(tmp_path):
    draft_dir = _checkpoint(tmp_path, "dspark", DSPARK_CONFIG)
    assert _spec_config(draft_dir, 7).block_size == 7
    with pytest.raises(ValueError, match="block_size must equal max_draft_len"):
        _spec_config(draft_dir, 4)
