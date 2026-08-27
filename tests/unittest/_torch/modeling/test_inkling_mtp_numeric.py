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
"""What the draft chain's weights actually become, on real tensors.

Everything checked about the MTP loader so far has been structural: which keys
exist, which names map to which, that a strict load is used. None of it reads a
number. That matters because speculative decoding was found to be lossless-in-
principle and lossy-in-fact, and a mis-loaded draft chain is one of the few
remaining explanations that no existing test could rule out.

These read the shipped checkpoint and check the transforms the mapper applies,
tensor by tensor. They skip when the checkpoint is not mounted.
"""

import glob
import json
import os

import pytest
import torch

_HF_ROOT = os.environ.get(
    "INKLING_HF_ROOT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data",
)
_CHECKPOINTS = ["Inkling-NVFP4-full", "Inkling-small-NVFP4"]


def _shard_for(ckpt, key):
    path = os.path.join(_HF_ROOT, ckpt)
    index = glob.glob(os.path.join(path, "*.index.json"))
    if not index:
        pytest.skip(f"{ckpt} not mounted")
    with open(index[0]) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        pytest.skip(f"{ckpt} has no {key}")
    return os.path.join(path, weight_map[key])


def _load_tensor(ckpt, key):
    from safetensors import safe_open

    shard = _shard_for(ckpt, key)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(key)


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_the_draft_dense_mlp_is_de_interleaved_not_chunked(ckpt):
    """``w13_dn`` is gate/up INTERLEAVED along the output dim.

    A contiguous ``chunk(2)`` produces two tensors of the right shape that pair
    the wrong gate and up channels in every SwiGLU of the chain. Nothing about
    that raises; the draft blocks simply propose badly, and speculative decoding
    then looks like a throughput disappointment rather than a bug. The trunk
    already de-interleaves, so this asserts the chain gets the same treatment.
    """
    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        InklingHfWeightMapper,
    )

    key = "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight"
    t = _load_tensor(ckpt, key)

    out = {}
    InklingHfWeightMapper._map_mtp(
        None,
        __import__("re").match(r"^model\.mtp\.layers\.(\d+)\.(.*)$", key),
        t,
        out,
    )
    gate = out["mtp_layers.0.transformer_block.mlp.gate_proj.weight"]
    up = out["mtp_layers.0.transformer_block.mlp.up_proj.weight"]

    assert torch.equal(gate, t[0::2]), "gate must be the EVEN output rows"
    assert torch.equal(up, t[1::2]), "up must be the ODD output rows"
    # And explicitly not the contiguous split, which has the same shapes.
    half = t.shape[0] // 2
    assert not torch.equal(gate, t[:half]) or half <= 1


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_every_draft_tensor_keeps_its_values_through_the_mapper(ckpt):
    """The mapper renames; only w13_dn is allowed to transform.

    A rename that silently transposed or sliced would be invisible in the key
    set, which is all the structural tests look at.
    """
    import re

    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        InklingHfWeightMapper,
    )

    for tail, expect_split in [
        ("transformer_block.attn.wq_du.weight", False),
        ("transformer_block.attn.rel_logits_proj.proj", False),
        ("input_proj.weight", False),
        ("embed_norm.weight", False),
        ("transformer_block.mlp.w2_md.weight", False),
        ("transformer_block.mlp.w13_dn.weight", True),
    ]:
        key = f"model.mtp.layers.0.{tail}"
        t = _load_tensor(ckpt, key)
        out = {}
        InklingHfWeightMapper._map_mtp(
            None, re.match(r"^model\.mtp\.layers\.(\d+)\.(.*)$", key), t, out
        )
        if expect_split:
            assert len(out) == 2
            continue
        assert len(out) == 1, f"{tail} produced {sorted(out)}"
        (mapped,) = out.values()
        assert torch.equal(mapped, t), f"{tail} was altered, not just renamed"


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_the_input_proj_takes_the_concatenated_pair(ckpt):
    """``input_proj`` maps [hidden_norm || embed_norm] back to hidden.

    So its in_features is 2 * hidden_size. A checkpoint that stored the two
    halves the other way round would have the same shape, which is why the
    ORDER of the concatenation in the block is asserted separately -- here we
    only pin that the width is the fused one and not a single hidden.
    """
    path = os.path.join(_HF_ROOT, ckpt, "config.json")
    if not os.path.exists(path):
        pytest.skip(f"{ckpt} not mounted")
    with open(path) as f:
        raw = json.load(f)
    hidden = (raw.get("text_config") or raw)["hidden_size"]

    t = _load_tensor(ckpt, "model.mtp.layers.0.input_proj.weight")
    assert t.shape == (hidden, 2 * hidden), t.shape


def test_the_block_concatenates_hidden_then_embed():
    """Order matters and the shapes cannot tell the two apart.

    SGLang builds ``cat(hidden_norm(h), embed_norm(e))``; swapping them feeds
    every draft block's input_proj the two halves transposed -- a valid tensor,
    a wrong projection, and no error anywhere.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock

    src = inspect.getsource(InklingMTPBlock.forward)
    i_hidden = src.index("self.hidden_norm(")
    i_embed = src.index("self.embed_norm(")
    assert i_hidden < i_embed, "hidden_norm must come first in the concatenation"


# --- what the shipped config.json actually resolves to ----------------------
# Cheap enough to belong here despite the file's name: it reads config.json and
# no tensors. It is the only check that runs the REAL declaration through the
# REAL resolver, and it is the one that would have caught the chain being built
# as a single replayed block on both releases.


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_the_shipped_config_resolves_to_vanilla_mtp(ckpt):
    """A whole-chain check, from config.json to the resolved decoding mode.

    Two framework readers want the chain depth and only one descends into
    text_config: MTPForCausalLM gets the TEXT sub-config, while
    update_spec_config_from_model_config gets the TOP-LEVEL object. Both shipped
    releases are `InklingForConditionalGeneration` / `inkling_mm_model`, whose
    top-level config.json carries no `num_nextn_predict_layers` at all -- so the
    second reader found nothing, fell back to 1, and the mode resolved to
    MTP_EAGLE_ONE_MODEL: ONE draft block replayed max_draft_len times, against a
    chain that declares eight depths with their own weights and their own
    banded/global attention geometry.

    Nothing raised. The model side read the text config and believed it had
    eight depths while the framework had decided there was one.
    """
    from tensorrt_llm._torch.configs.inkling import InklingConfig
    from tensorrt_llm._torch.speculative.utils import (
        get_num_spec_layers,
        update_spec_config_from_model_config,
    )
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    path = os.path.join(_HF_ROOT, ckpt, "config.json")
    if not os.path.exists(path):
        pytest.skip(f"{ckpt} not mounted")
    raw = json.load(open(path))

    assert raw["architectures"] == ["InklingForConditionalGeneration"]
    declared = raw["mtp_config"]["num_nextn_predict_layers"]
    assert raw.get("num_nextn_predict_layers") is None, (
        "the top level is where the resolver looks and where the checkpoint is silent"
    )

    cfg = InklingConfig(**{k: v for k, v in raw.items() if k != "architectures"})
    spec_config = MTPDecodingConfig(max_draft_len=3)
    update_spec_config_from_model_config(spec_config, cfg)

    assert spec_config.num_nextn_predict_layers == declared
    assert spec_config.spec_dec_mode.is_mtp_vanilla(), (
        f"{ckpt} resolved to {spec_config.spec_dec_mode}"
    )
    # What MTPForCausalLM will build, and what the draft KV cache is sized for.
    assert min(spec_config.max_draft_len, cfg.text_config.num_nextn_predict_layers) == 3
    assert get_num_spec_layers(spec_config) == declared
