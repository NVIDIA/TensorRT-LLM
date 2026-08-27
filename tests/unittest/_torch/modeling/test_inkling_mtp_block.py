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
"""The derived config that makes a decoder layer behave as an MTP block.

The draft block reuses ``InklingDecoderLayer`` unchanged: the layer asks its
config which layers are dense and which are banded, and ``mtp_block_config``
answers with the CHAIN's geometry instead of the trunk's.

That indirection is the whole risk. If a banded depth were built with the
trunk's window, the checkpoint's ``rel_logits_proj`` for that depth -- trained
at the head's window -- would be applied at the wrong extent: wrong numbers,
no crash, nothing at runtime to notice it. So the derived config is asserted
field by field rather than trusted.
"""

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig

# As shipped in both Inkling-NVFP4-full and Inkling-small-NVFP4.
_CKPT_MTP = {
    "num_nextn_predict_layers": 8,
    "chain_hidden_post_norm": False,
    "local_layer_ids": [0, 2, 4, 5, 6, 7],
}

_TRUNK = {
    "num_hidden_layers": 66,
    "dense_mlp_idx": 2,
    "local_layer_ids": [1, 3, 5],  # deliberately different from the chain's
    "sliding_window_size": 512,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 16,
    "swa_head_dim": 64,
}


def _text(**overrides):
    cfg = {**_TRUNK, **overrides}
    return InklingConfig(text_config=cfg, mtp_config=dict(_CKPT_MTP)).text_config


@pytest.mark.parametrize("depth", range(8))
def test_every_depth_is_dense(depth):
    """MTP blocks always use the dense MLP, at any depth index.

    Depth 5 sits well past the trunk's ``dense_mlp_idx`` of 2, so without the
    override it would be built as MoE and then fail to find expert weights the
    checkpoint does not have.
    """
    block = _text().mtp_block_config(depth)
    assert block.is_dense_layer(depth) is True


@pytest.mark.parametrize(
    "depth,banded",
    [(0, True), (1, False), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True)],
)
def test_bandedness_follows_the_chain_not_the_trunk(depth, banded):
    """The trunk's banded layers are [1,3,5]; the chain's are [0,2,4,5,6,7].

    Depths 1 and 3 are banded in the trunk and global in the chain, so a config
    that leaked the trunk's list would get exactly these two backwards.
    """
    block = _text().mtp_block_config(depth)
    assert block.is_local_layer(depth) is banded


def test_banded_depth_uses_the_chain_window_and_heads():
    text = _text()
    block = text.mtp_block_config(0)  # banded
    assert block.is_local_layer(0)
    assert block.layer_window(0) == 512  # defaults to the trunk window
    assert block.layer_num_heads(0) == 64
    assert block.layer_num_kv_heads(0) == 16
    assert block.layer_head_dim(0) == 64


def test_explicit_chain_extent_reaches_the_block():
    """An overridden ``local_extent`` must land on the layer's window."""
    cfg = InklingConfig(text_config=dict(_TRUNK), mtp_config={**_CKPT_MTP, "local_extent": 2048})
    block = cfg.text_config.mtp_block_config(0)
    assert block.layer_window(0) == 2048


def test_global_depth_keeps_the_full_attention_geometry():
    """A global depth must not be given the SWA head counts."""
    block = _text().mtp_block_config(1)  # global in the chain
    assert block.is_local_layer(1) is False
    assert block.layer_window(1) is None
    assert block.layer_num_heads(1) == 48
    assert block.layer_num_kv_heads(1) == 8
    assert block.layer_head_dim(1) == 128


def test_deriving_does_not_mutate_the_trunk_config():
    """Building a block must not disturb the trunk the model is still using."""
    text = _text()
    before = (
        text.dense_mlp_idx,
        list(text.local_layer_ids),
        text.sliding_window_size,
        text.swa_num_attention_heads,
    )
    for depth in range(8):
        text.mtp_block_config(depth)
    after = (
        text.dense_mlp_idx,
        list(text.local_layer_ids),
        text.sliding_window_size,
        text.swa_num_attention_heads,
    )
    assert before == after


# --- the draft chain's own KV cache geometry -------------------------------
# The draft chain gets a SEPARATE cache manager, built with num_layers = the
# number of built depths. KVCacheManagerV2 asserts len(num_kv_heads) equals
# that count, so handing it the trunk's per-layer list is an outright failure
# -- which is how this was found, several minutes into a 4-GPU job.


@pytest.mark.parametrize("depths", [1, 3, 8])
def test_draft_kv_head_list_has_one_entry_per_built_depth(depths):
    """Length must follow the chain the runtime built, not the checkpoint's 8.

    A server asking for 3 draft tokens builds 3 blocks; the manager is created
    with num_layers=3 and asserts the list matches.
    """
    assert len(_text().mtp_num_kv_heads_per_layer(depths)) == depths


def test_draft_kv_heads_follow_the_chain_banded_pattern():
    """Chain banded depths are [0,2,4,5,6,7]; the trunk's are [1,3,5].

    On the full checkpoint banded layers carry 16 KV heads and global ones 8,
    so a slice of the trunk's list would size depths 1 and 3 for 16 heads and
    depth 1 for 8 -- pages allocated against the wrong head count, with nothing
    at runtime to report it.
    """
    assert _text().mtp_num_kv_heads_per_layer(8) == [16, 8, 16, 8, 16, 16, 16, 16]


def test_uniform_checkpoint_gives_a_uniform_draft_list():
    """Inkling-small has swa_num_key_value_heads == num_key_value_heads == 8.

    Banded and global depths then agree, and the list must simply be uniform
    rather than accidentally picking up the trunk's 16 from a stale default.
    """
    text = _text(num_key_value_heads=8, swa_num_key_value_heads=8)
    assert text.mtp_num_kv_heads_per_layer(4) == [8, 8, 8, 8]


def test_chain_swa_geometry_can_differ_from_the_trunk():
    text = _text()
    text.mtp_swa_num_attention_heads = 32
    text.mtp_swa_num_key_value_heads = 4
    text.mtp_swa_head_dim = 256
    block = text.mtp_block_config(0)
    assert block.layer_num_heads(0) == 32
    assert block.layer_num_kv_heads(0) == 4
    assert block.layer_head_dim(0) == 256


# --- the two indices a draft block lives under -----------------------------
# Geometry is indexed by CHAIN depth (0..7); the KV cache is keyed by GLOBAL
# layer index (trunk layers + depth), because the draft manager's layer offsets
# are global. Folding one into the other is a KeyError in the first draft
# forward, several minutes into a multi-GPU run.


def test_block_config_answers_for_the_global_index():
    """Built with the global index, the config must still say dense and banded."""
    text = _text()
    trunk = text.num_hidden_layers
    for depth in range(8):
        cfg = text.mtp_block_config(depth, trunk + depth)
        assert cfg.is_dense_layer(trunk + depth) is True
        assert cfg.is_local_layer(trunk + depth) is text.is_mtp_local_depth(depth)


def test_global_index_geometry_matches_the_chain_depth_geometry():
    """Same block, two indices, identical answers.

    The window and head counts come from the chain depth; only the index the
    layer is addressed by changes. If these ever diverge, a banded depth would
    be built with the wrong window and its rel_logits_proj -- trained at the
    head's window -- would be applied at the wrong extent, with no crash.
    """
    text = _text()
    trunk = text.num_hidden_layers
    for depth in range(8):
        by_depth = text.mtp_block_config(depth)
        by_global = text.mtp_block_config(depth, trunk + depth)
        assert by_global.layer_window(trunk + depth) == by_depth.layer_window(depth)
        assert by_global.layer_num_heads(trunk + depth) == by_depth.layer_num_heads(depth)
        assert by_global.layer_num_kv_heads(trunk + depth) == by_depth.layer_num_kv_heads(depth)


def test_draft_conv_pool_widths_come_from_the_chain_not_the_trunk():
    """Addressed globally, sized from the chain.

    The draft pool's rows are indexed by the global layer number, but their
    channel width is the chain's: the trunk's accessor at index 42 answers for
    a layer the trunk does not have, so a banded depth would be given global
    widths (or the reverse). The two differ by the banded/global head ratio,
    which shows up as a channel-width mismatch inside the verify capture -- and
    only on a checkpoint where those head counts differ, which is why the
    short-prompt end-to-end run never caught it.
    """
    text = _text()
    trunk = text.num_hidden_layers
    for depth in range(8):
        chain_heads = text.mtp_depth_num_kv_heads(depth)
        trunk_answer = text.layer_num_kv_heads(trunk + depth)
        # depth 0 is banded in the chain (16 heads here) and the trunk's
        # accessor at 66 says global (8): exactly the mismatch.
        if depth in (0, 2, 4, 5, 6, 7):
            assert chain_heads == 16
            assert trunk_answer == 8
        else:
            assert chain_heads == 8


# --- the chain as MTPForCausalLM actually builds it -------------------------
# Everything above asserts the config's answers. This builds the blocks through
# the framework's own constructor and asks each one which depth it thinks it is
# -- the step that was wrong, and the only one nothing was checking. Real chain
# SHAPE (42 trunk over 8 depths, so trunk % depths == 2), tiny dimensions, no
# weights.


def _tiny_mtp_model_config(max_draft_len):
    import copy as _copy

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    text = InklingConfig(
        text_config={
            "num_hidden_layers": 42,  # the shipped small checkpoint's trunk
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 4,
            "swa_head_dim": 16,
            "intermediate_size": 32,
            "dense_intermediate_size": 32,
            "d_rel": 4,
            "rel_extent": 16,
            "vocab_size": 128,
            "unpadded_vocab_size": 128,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        },
        mtp_config=dict(_CKPT_MTP),
    ).text_config

    spec_config = MTPDecodingConfig(max_draft_len=max_draft_len)
    from tensorrt_llm._torch.speculative.utils import update_spec_config_from_model_config

    update_spec_config_from_model_config(spec_config, text)
    model_config = ModelConfig(pretrained_config=text)
    model_config = _copy.copy(model_config)
    model_config.spec_config = spec_config
    return model_config, text


@pytest.mark.parametrize("max_draft_len", [1, 3, 5])
def test_each_built_block_knows_its_own_depth(max_draft_len):
    """42 % 8 == 2, so a modulo puts block b on depth b + 2.

    It survived at max_draft_len 3 because the banded set [0, 2, 4, 5, 6, 7]
    gives 0,1,2 and 2,3,4 the same banded/global pattern. 5 is past that: block
    3 wants depth 3 (global) and a modulo hands it depth 5 (banded), which is a
    different KV-head count from the one the conv pool sizes that layer with.
    """
    import torch.nn as nn

    from tensorrt_llm._torch.models.modeling_speculative import MTPForCausalLM

    model_config, text = _tiny_mtp_model_config(max_draft_len)

    class _Model(nn.Module):
        aux_stream_dict: dict = {}

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 8)

    chain = MTPForCausalLM(model_config, text.num_hidden_layers, nn.Identity(), _Model())

    assert len(chain.mtp_layers) == min(max_draft_len, _CKPT_MTP["num_nextn_predict_layers"])
    for expected, block in enumerate(chain.mtp_layers):
        assert block.depth == expected, (
            f"block {expected} thinks it is depth {block.depth}; a modulo over "
            f"{_CKPT_MTP['num_nextn_predict_layers']} depths under a "
            f"{text.num_hidden_layers}-layer trunk gives {expected + 2}"
        )
        assert block.transformer_block.layer_idx == text.num_hidden_layers + expected


def test_the_chain_is_not_one_replayed_block():
    """Vanilla MTP builds one block per depth; EAGLE builds one and replays it.

    Both shipped releases resolved to EAGLE, so this is the assertion that the
    chain has distinct blocks at all -- distinct objects, distinct parameters.
    """
    import torch.nn as nn

    from tensorrt_llm._torch.models.modeling_speculative import MTPForCausalLM

    model_config, text = _tiny_mtp_model_config(3)
    assert model_config.spec_config.spec_dec_mode.is_mtp_vanilla()

    class _Model(nn.Module):
        aux_stream_dict: dict = {}

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 8)

    chain = MTPForCausalLM(model_config, text.num_hidden_layers, nn.Identity(), _Model())
    assert len({id(b) for b in chain.mtp_layers}) == 3
    first = dict(chain.mtp_layers[0].named_parameters())
    second = dict(chain.mtp_layers[1].named_parameters())
    shared = [k for k in first if first[k] is second.get(k)]
    assert not shared, f"depths share parameters: {shared[:3]}"
