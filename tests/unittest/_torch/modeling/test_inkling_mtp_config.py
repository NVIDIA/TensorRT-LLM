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
"""The MTP draft chain's geometry, which is not the trunk's.

A banded MTP depth runs at the HEAD's window, not the trunk's: the
checkpoint's ``rel_logits_proj`` for that depth was trained at that window, so
reusing the trunk geometry silently applies the wrong relative-position extent
-- wrong numbers, no crash.

The checkpoint declares the chain in a top-level ``mtp_config``; ``text_config``
is the single source of truth every consumer reads, so the canonicalization is
asserted here rather than assumed.
"""

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig, InklingTextConfig

# The real checkpoints' declaration, as shipped in both Inkling-NVFP4-full and
# Inkling-small-NVFP4 (verified against the on-disk config.json).
_CKPT_MTP_CONFIG = {
    "num_nextn_predict_layers": 8,
    "chain_hidden_post_norm": False,
    "local_layer_ids": [0, 2, 4, 5, 6, 7],
}


def test_mtp_config_is_canonicalized_onto_text_config():
    """A checkpoint-level ``mtp_config`` must reach ``text_config``.

    Three consumers read this -- the MTP blocks, the KV-pool routing and the
    weight mapper. Leaving it only on the top-level config is how they drift.
    """
    cfg = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP_CONFIG))
    assert cfg.text_config.mtp_local_layer_ids == [0, 2, 4, 5, 6, 7]
    # Retained verbatim as well, so the checkpoint still round-trips.
    assert cfg.mtp_config.num_nextn_predict_layers == 8


def test_local_extent_defaults_to_the_trunk_window():
    """Absent an explicit extent, a banded depth uses the trunk's window.

    Both shipped checkpoints omit ``local_extent``, so this default is the one
    actually in use -- not a fallback nobody hits.
    """
    cfg = InklingConfig(text_config={"sliding_window_size": 512}, mtp_config=dict(_CKPT_MTP_CONFIG))
    text = cfg.text_config
    assert text.mtp_local_extent is None
    assert text.mtp_depth_window(0) == 512


def test_explicit_local_extent_overrides_the_trunk_window():
    cfg = InklingConfig(
        text_config={"sliding_window_size": 512},
        mtp_config={**_CKPT_MTP_CONFIG, "local_extent": 2048},
    )
    assert cfg.text_config.mtp_depth_window(0) == 2048


@pytest.mark.parametrize(
    "depth,banded",
    [(0, True), (1, False), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True)],
)
def test_depth_bandedness_matches_the_checkpoint(depth, banded):
    """Depths 1 and 3 are global; the rest are banded, per the checkpoint."""
    text = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP_CONFIG)).text_config
    assert text.is_mtp_local_depth(depth) is banded
    assert (text.mtp_depth_window(depth) is not None) is banded


def test_global_depth_uses_the_full_attention_geometry():
    """A global depth must not inherit the SWA head counts."""
    text = InklingConfig(
        text_config={
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "swa_num_attention_heads": 64,
            "swa_num_key_value_heads": 16,
            "swa_head_dim": 64,
        },
        mtp_config=dict(_CKPT_MTP_CONFIG),
    ).text_config
    # depth 1 is global
    assert text.mtp_depth_num_heads(1) == 48
    assert text.mtp_depth_num_kv_heads(1) == 8
    assert text.mtp_depth_head_dim(1) == 128
    # depth 0 is banded -> SWA geometry
    assert text.mtp_depth_num_heads(0) == 64
    assert text.mtp_depth_num_kv_heads(0) == 16
    assert text.mtp_depth_head_dim(0) == 64


def test_mtp_swa_geometry_can_be_overridden_independently():
    """The head's SWA geometry may differ from the trunk's.

    SGLang allows this per depth; defaulting to the trunk is only a default.
    """
    text = InklingTextConfig(
        swa_num_attention_heads=64,
        swa_num_key_value_heads=16,
        swa_head_dim=64,
        mtp_local_layer_ids=[0],
        mtp_swa_num_attention_heads=32,
        mtp_swa_num_key_value_heads=4,
        mtp_swa_head_dim=256,
    )
    assert text.mtp_depth_num_heads(0) == 32
    assert text.mtp_depth_num_kv_heads(0) == 4
    assert text.mtp_depth_head_dim(0) == 256


def test_no_mtp_config_leaves_the_chain_empty():
    """A checkpoint without a draft chain must not look like it has one."""
    text = InklingConfig(text_config={}).text_config
    assert text.mtp_local_layer_ids == []
    assert text.is_mtp_local_depth(0) is False
    assert text.mtp_depth_window(0) is None


def test_the_chain_depth_is_recovered_from_the_largest_listed_depth():
    """``MTPForCausalLM`` reads the chain depth with a BARE attribute access.

    ``checkpoint_mtp_num_layers = pretrained_config.num_nextn_predict_layers``
    -- no getattr, no default. A checkpoint that describes its chain only by
    naming the banded depths would reach that line with the attribute absent and
    die on an AttributeError from inside framework code, naming neither Inkling
    nor the field.

    The count comes from the largest index, not from how many are listed:
    ``local_layer_ids`` says WHICH depths are banded, and
    ``is_mtp_local_depth`` uses it as a membership set. The shipped small
    checkpoint is the proof -- it declares 8 depths as [0, 2, 4, 5, 6, 7], so
    the length is 6 and only ``max + 1`` recovers the 8.
    """
    cfg = InklingConfig(
        text_config={},
        mtp_config={"local_layer_ids": [0, 2, 4, 5, 6, 7]},
    )
    assert cfg.text_config.num_nextn_predict_layers == 8
    assert cfg.num_nextn_predict_layers == 8


def test_a_declared_depth_count_wins_over_the_listed_depths():
    """The explicit field is authoritative; the list is only a fallback."""
    cfg = InklingConfig(
        text_config={},
        mtp_config={"num_nextn_predict_layers": 3, "local_layer_ids": [0, 2]},
    )
    assert cfg.text_config.num_nextn_predict_layers == 3


def test_no_chain_leaves_the_framework_field_unset():
    """Nothing is fabricated for a checkpoint that ships no chain.

    The model's own guard turns that into a sentence about the checkpoint;
    mirroring a made-up 1 here would build a one-deep chain out of nothing.
    """
    cfg = InklingConfig(text_config={})
    assert getattr(cfg.text_config, "num_nextn_predict_layers", None) is None


def test_the_chain_depth_reaches_both_framework_readers():
    """Two readers, two places, and only one of them descends into text_config.

    ``MTPForCausalLM`` is handed the TEXT sub-config, but
    ``update_spec_config_from_model_config`` is handed
    ``config.pretrained_config`` -- the TOP-LEVEL object for a multimodal
    checkpoint -- and reads the field off it directly. Missing it there, it
    falls back to 1, and ``MTPDecodingConfig.spec_dec_mode`` then resolves 1
    depth plus the default flags to MTP_EAGLE_ONE_MODEL instead of vanilla MTP:
    one draft block replayed, rather than Inkling's per-depth chain. Nothing
    raises; the two sides simply disagree about how many depths exist.
    """
    cfg = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP_CONFIG))
    depth = _CKPT_MTP_CONFIG["num_nextn_predict_layers"]
    assert cfg.text_config.num_nextn_predict_layers == depth
    assert cfg.num_nextn_predict_layers == depth, (
        "the top-level config is what update_spec_config_from_model_config reads"
    )


def test_the_resolved_spec_mode_is_vanilla_mtp_not_eagle():
    """End-to-end on the field that decides it, through the real resolver.

    Pinning the mode rather than the field, because the field is only
    interesting for what it resolves to -- and the failure was silent precisely
    because the field looked absent rather than wrong.
    """
    from tensorrt_llm._torch.speculative.utils import update_spec_config_from_model_config
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    cfg = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP_CONFIG))
    spec_config = MTPDecodingConfig(max_draft_len=3)
    update_spec_config_from_model_config(spec_config, cfg)

    assert spec_config.num_nextn_predict_layers == _CKPT_MTP_CONFIG["num_nextn_predict_layers"]
    assert spec_config.spec_dec_mode.is_mtp_vanilla(), (
        f"resolved to {spec_config.spec_dec_mode}; Inkling's depths have their "
        "own weights and geometry, so a replayed single block is a different model"
    )
