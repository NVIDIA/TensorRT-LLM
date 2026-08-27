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
"""The MTP draft chain's checkpoint keys, checked against the real weights.

A key set derived by reading code is a guess until it is compared with a
checkpoint. These tests take the actual ``*.index.json`` of the two shipped
releases and require an EXACT match -- no missing keys (a silently
unloaded draft tensor is wrong numbers, not a crash) and no unaccounted ones
(a tensor nobody claims means the derivation is incomplete).

Skipped, not failed, when the weights are not mounted: the derivation is still
covered by the shape assertions below.
"""

import glob
import json
import os

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig
from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
    inkling_expected_mtp_keys,
)

_HF_ROOT = os.environ.get(
    "INKLING_HF_ROOT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data",
)
_CHECKPOINTS = ["Inkling-NVFP4-full", "Inkling-small-NVFP4"]


def _load(ckpt):
    path = os.path.join(_HF_ROOT, ckpt)
    index = glob.glob(os.path.join(path, "*.index.json"))
    cfg_path = os.path.join(path, "config.json")
    if not index or not os.path.exists(cfg_path):
        pytest.skip(f"{ckpt} not mounted")
    with open(index[0]) as f:
        keys = set(json.load(f)["weight_map"])
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = InklingConfig(text_config=raw.get("text_config") or {}, mtp_config=raw.get("mtp_config"))
    return keys, cfg, raw


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_mtp_keys_match_the_checkpoint_exactly(ckpt):
    """Every ``model.mtp.*`` key is claimed, and every claimed key exists."""
    keys, cfg, raw = _load(ckpt)
    depths = raw["mtp_config"]["num_nextn_predict_layers"]
    expected = inkling_expected_mtp_keys(cfg.text_config, depths)
    actual = {k for k in keys if k.startswith("model.mtp.")}

    assert not (expected - actual), (
        f"{ckpt}: derived keys the checkpoint does not have "
        f"(sample: {sorted(expected - actual)[:5]})"
    )
    assert not (actual - expected), (
        f"{ckpt}: checkpoint keys nobody claims -- the derivation is "
        f"incomplete (sample: {sorted(actual - expected)[:5]})"
    )


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_chain_depth_and_geometry_come_from_the_checkpoint(ckpt):
    """The declared depth matches the weights, and the bandedness is read."""
    keys, cfg, raw = _load(ckpt)
    depths = raw["mtp_config"]["num_nextn_predict_layers"]
    present = {
        int(k.split("model.mtp.layers.")[1].split(".")[0])
        for k in keys
        if k.startswith("model.mtp.layers.")
    }
    assert present == set(range(depths)), (
        f"{ckpt}: declared {depths} depths, checkpoint has {sorted(present)}"
    )
    assert cfg.text_config.mtp_local_layer_ids == raw["mtp_config"]["local_layer_ids"]


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_every_depth_is_dense_not_moe(ckpt):
    """MTP blocks use the dense MLP, so no expert tensors may appear.

    SGLang forces the dense MLP for every depth. If a checkpoint ever ships
    expert tensors here, the derivation above would silently drop them, so
    assert the assumption rather than rely on it.
    """
    keys, _, _ = _load(ckpt)
    experts = {k for k in keys if k.startswith("model.mtp.") and (".experts." in k or "gate." in k)}
    assert not experts, f"{ckpt}: unexpected MoE tensors in the draft chain: {sorted(experts)[:5]}"


def test_derivation_scales_with_depth_without_a_checkpoint():
    """The key count is linear in depth and includes the three fold tensors."""
    cfg = InklingConfig(
        text_config={}, mtp_config={"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2]}
    )
    one = inkling_expected_mtp_keys(cfg.text_config, 1)
    two = inkling_expected_mtp_keys(cfg.text_config, 2)
    assert len(two) == 2 * len(one)
    for name in ("embed_norm.weight", "hidden_norm.weight", "input_proj.weight"):
        assert f"model.mtp.layers.0.{name}" in one


# --- the chain is not quantized --------------------------------------------


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_the_checkpoint_carries_no_nvfp4_scales_for_the_draft_chain(ckpt):
    """The chain is BF16, and the checkpoint says so by omission.

    Under ``model.mtp`` the only scale tensor is the dense MLP's
    ``global_scale``, which BF16 dense layers carry too. No ``weight_scale``,
    ``weight_scale_2`` or ``input_scale`` exists anywhere in the chain, so
    building the draft blocks NVFP4 asks for tensors that were never written --
    which surfaced first as a strict-load miss and then, once loading was fixed,
    as "fp4_quantize only supports fp16/bf16/e4m3" from the quantize op: a dtype
    complaint at the activation rather than "this was never quantized".
    """
    keys, _, _ = _load(ckpt)
    scales = [
        k
        for k in keys
        if k.startswith("model.mtp.") and ("weight_scale" in k or "input_scale" in k or "amax" in k)
    ]
    assert scales == [], f"unexpected NVFP4 scales in the draft chain: {sorted(scales)[:3]}"


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_the_exclude_list_cannot_carve_the_chain_out(ckpt):
    """``exclude_modules`` names only ``model.llm.*`` entries.

    So the chain sits outside the quantized subtree rather than being excluded
    from it, and no exclusion-list matching would make a quantized draft block
    behave. The block's config has to be built unquantized.
    """
    path = os.path.join(_HF_ROOT, ckpt, "hf_quant_config.json")
    if not os.path.exists(path):
        pytest.skip(f"{ckpt} has no hf_quant_config.json")
    with open(path) as f:
        excludes = json.load(f)["quantization"].get("exclude_modules", [])
    assert excludes, "no exclude_modules to reason about"
    assert not [e for e in excludes if e.startswith("model.mtp")]


def test_the_draft_block_builds_its_config_unquantized():
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock, _unquantized_like

    src = inspect.getsource(InklingMTPBlock.__init__)
    assert "_unquantized_like" in src
    assert _unquantized_like(None) is None


def test_unquantized_config_actually_reports_no_quantization():
    """Clearing ``quant_algo`` on a copy is not enough.

    ``QuantConfig.quant_mode`` and ``layer_quant_mode`` are cached_property, so
    a copy keeps whatever mode was already computed: the algo field reads None
    while every Linear still builds quantized. The mode is what Linear consults,
    so that is what this asserts.
    """
    from tensorrt_llm._torch.models.modeling_inkling import _unquantized_like
    from tensorrt_llm.models.modeling_utils import QuantConfig

    src = QuantConfig(quant_algo="NVFP4", kv_cache_quant_algo="FP8")
    _ = src.quant_mode  # populate the cache, as the real config has by now
    out = _unquantized_like(src)
    assert out.quant_algo is None
    assert not out.layer_quant_mode.has_nvfp4()
    # The draft KV cache still follows the target's KV quantization.
    assert out.kv_cache_quant_algo == src.kv_cache_quant_algo


def test_the_draft_chain_is_looked_up_under_its_mapped_names():
    """The keys `_load_mtp_weights` searches must be the ones the mapper emits.

    The chain arrives as ``model.mtp.layers.N.*`` and the loader walks the
    module tree as ``mtp_layers.N.*``; the mapper is what converts one to the
    other. Searching the RAW dict for the MAPPED prefix matches nothing, and
    nothing is a silent success: every draft block keeps its initial values,
    the drafter proposes token 0 forever, every draft is rejected, and
    speculative decoding costs a forward per step while changing no output.
    That is what shipped until it was probed at the drafter's proposals.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._load_mtp_weights)
    # The raw checkpoint prefix has to be selected, and the mapper has to run,
    # BEFORE anything looks for the mapped prefix.
    assert 'startswith("model.mtp.")' in src
    assert "preprocess_weights" in src
    assert src.index("preprocess_weights") < src.index('startswith("mtp_layers.")')


def test_an_unloaded_draft_block_is_refused():
    """All-zero weight matrices after loading must raise, not run.

    A trained projection is never all zeros; a block that was built and never
    loaded is exactly that (its norms sit at their init 1.0). The state costs
    nothing at load and everything at runtime, so it is checked at load.
    """
    import pytest
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import _assert_draft_chain_loaded

    class _Block(torch.nn.Module):
        def __init__(self, zeroed: bool):
            super().__init__()
            self.norm = torch.nn.Parameter(torch.ones(8))
            w = torch.zeros(8, 8) if zeroed else torch.randn(8, 8)
            self.proj = torch.nn.Parameter(w)

    _assert_draft_chain_loaded([_Block(False), _Block(False)])
    with pytest.raises(RuntimeError, match="all zeros after loading"):
        _assert_draft_chain_loaded([_Block(False), _Block(True)])
