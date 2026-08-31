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
"""EPLB wiring for the one-model DSpark drafter (CPU-only, no weights).

Covers the three P0 guarantees:
  1. the target's ``moe_load_balancer`` reaches the DSpark draft config -- and
     ONLY DSpark's (PARD / DFlash / draft-target must be untouched);
  2. an ``initial_global_assignments`` map that predates DSpark fails early with
     every missing stage index listed, not a bare ``KeyError``;
  3. online EPLB is rejected at config time instead of deadlocking at runtime.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    validate_dspark_eplb_layer_base,
    validate_dspark_eplb_stage_layers,
)
from tensorrt_llm._torch.models.modeling_speculative import external_drafter_config_kwargs
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

NUM_HIDDEN_LAYERS = 61
NUM_STAGES = 3
NUM_EXPERTS = 384
# DSpark stages register as EPLB layers 61, 62, 63.
DSPARK_LAYERS = list(range(NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS + NUM_STAGES))


def _assignments(layer_ids):
    """A structurally valid placement (one permutation of the experts) per layer."""
    return {layer_id: list(range(NUM_EXPERTS)) for layer_id in layer_ids}


def _lb_config(layer_ids=None, layer_updates_per_iter=0):
    return MoeLoadBalancerConfig(
        num_slots=NUM_EXPERTS,
        initial_global_assignments=(_assignments(layer_ids) if layer_ids is not None else None),
        layer_updates_per_iter=layer_updates_per_iter,
    )


def _model_config(lb_config=None, num_hidden_layers=NUM_HIDDEN_LAYERS):
    return SimpleNamespace(
        moe_load_balancer=lb_config,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
        mapping=object(),
        max_num_tokens=8192,
        moe_max_num_tokens=8192,
        pretrained_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
    )


def _spec_config(mode, *, embedded=True):
    # Only the embedded DSpark draft shares the target's EPLB layer namespace:
    # its stages are target decoder blocks registered into the target's
    # balancer. A standalone DSpark drafter is an independent checkpoint and is
    # treated like the other external drafters.
    return SimpleNamespace(spec_dec_mode=mode, draft_is_embedded_in_target=embedded)


@pytest.fixture
def eplb_active():
    """Pretend an engine-wide MoeLoadBalancer is live during model construction."""
    with patch.object(modeling_dspark, "_active_moe_load_balancer", return_value=object()):
        yield


# --------------------------------------------------------------------------
# 1. config propagation -- DSpark only
# --------------------------------------------------------------------------


def test_dspark_draft_config_inherits_load_balancer():
    lb_config = _lb_config(DSPARK_LAYERS)
    kwargs = external_drafter_config_kwargs(
        _model_config(lb_config), _spec_config(SpeculativeDecodingMode.DSPARK)
    )
    # The very same object, so MoeLoadBalancerConfig.setup() done on the target
    # side is already visible to the draft.
    assert kwargs["moe_load_balancer"] is lb_config


def test_dspark_draft_config_does_not_recurse_into_spec_dec():
    # Regression guard: propagating moe_load_balancer must not tempt anyone into
    # also forwarding spec_config, which would recursively build a drafter.
    kwargs = external_drafter_config_kwargs(
        _model_config(_lb_config(DSPARK_LAYERS)), _spec_config(SpeculativeDecodingMode.DSPARK)
    )
    assert kwargs["spec_config"] is None


@pytest.mark.parametrize(
    "mode",
    [
        SpeculativeDecodingMode.PARD,
        SpeculativeDecodingMode.DFLASH,
        SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL,
    ],
    # SpeculativeDecodingMode is an IntEnum, so the default ids would be the raw
    # numbers -- name the cases so a CI failure says which drafter regressed.
    ids=["pard", "dflash", "draft_target_one_model"],
)
def test_non_dspark_external_drafters_do_not_inherit_load_balancer(mode):
    # These are independent checkpoints whose expert topology and layer-index
    # namespace need not match the target's. Keep the DSpark fix from being
    # silently generalized by a future refactor.
    kwargs = external_drafter_config_kwargs(
        _model_config(_lb_config(DSPARK_LAYERS)), _spec_config(mode)
    )
    assert "moe_load_balancer" not in kwargs


def test_standalone_dspark_drafter_does_not_inherit_load_balancer():
    # The flavour, not the mode, decides: a standalone DSpark drafter is its own
    # checkpoint, so forwarding the target's EPLB config would key its experts
    # against a layer namespace that is not the drafter's.
    kwargs = external_drafter_config_kwargs(
        _model_config(_lb_config(DSPARK_LAYERS)),
        _spec_config(SpeculativeDecodingMode.DSPARK, embedded=False),
    )
    assert "moe_load_balancer" not in kwargs


def test_external_drafter_kwargs_are_stable_across_modes():
    common = external_drafter_config_kwargs(
        _model_config(_lb_config(DSPARK_LAYERS)), _spec_config(SpeculativeDecodingMode.PARD)
    )
    dspark = external_drafter_config_kwargs(
        _model_config(_lb_config(DSPARK_LAYERS)), _spec_config(SpeculativeDecodingMode.DSPARK)
    )
    assert set(dspark) - set(common) == {"moe_load_balancer"}


# --------------------------------------------------------------------------
# 2. stage-layer placement coverage
# --------------------------------------------------------------------------


def test_complete_assignments_accepted(eplb_active):
    lb_config = _lb_config(list(range(NUM_HIDDEN_LAYERS)) + DSPARK_LAYERS)
    validate_dspark_eplb_stage_layers(_model_config(lb_config), NUM_HIDDEN_LAYERS, NUM_STAGES)


def test_missing_stage_layers_listed_in_one_error(eplb_active):
    # A config generated back when the drafter was 1-layer MTP: covers layer 61
    # (the old MTP layer) but not the two extra DSpark stages.
    lb_config = _lb_config(list(range(NUM_HIDDEN_LAYERS + 1)))
    with pytest.raises(ValueError) as excinfo:
        validate_dspark_eplb_stage_layers(_model_config(lb_config), NUM_HIDDEN_LAYERS, NUM_STAGES)
    message = str(excinfo.value)
    assert "[62, 63]" in message
    assert "61" not in message.split("missing DSpark layer(s)")[1].split("]")[0]


def test_auto_placement_needs_no_assignments(eplb_active):
    # initial_global_assignments omitted -> auto-generated placement covers every
    # registered layer, including the DSpark stages.
    validate_dspark_eplb_stage_layers(
        _model_config(_lb_config(None)), NUM_HIDDEN_LAYERS, NUM_STAGES
    )


def test_no_validation_when_eplb_inactive():
    # Without an active balancer the config is never consumed, so an incomplete
    # (or online) one must not break the non-EPLB DSpark path.
    lb_config = _lb_config(list(range(NUM_HIDDEN_LAYERS)), layer_updates_per_iter=4)
    with patch.object(modeling_dspark, "_active_moe_load_balancer", return_value=None):
        validate_dspark_eplb_stage_layers(_model_config(lb_config), NUM_HIDDEN_LAYERS, NUM_STAGES)


# --------------------------------------------------------------------------
# 3. online EPLB rejected at config time
# --------------------------------------------------------------------------


def test_online_eplb_rejected(eplb_active):
    lb_config = _lb_config(list(range(NUM_HIDDEN_LAYERS)) + DSPARK_LAYERS, layer_updates_per_iter=2)
    with pytest.raises(ValueError, match="static EPLB only"):
        validate_dspark_eplb_stage_layers(_model_config(lb_config), NUM_HIDDEN_LAYERS, NUM_STAGES)


# --------------------------------------------------------------------------
# 4. draft/target layer namespace must line up
# --------------------------------------------------------------------------


def test_layer_base_mismatch_rejected(eplb_active):
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_dspark_eplb_layer_base(
            _model_config(_lb_config(DSPARK_LAYERS)),
            _model_config(_lb_config(DSPARK_LAYERS), num_hidden_layers=3),
        )


def test_layer_base_match_accepted(eplb_active):
    validate_dspark_eplb_layer_base(
        _model_config(_lb_config(DSPARK_LAYERS)), _model_config(_lb_config(DSPARK_LAYERS))
    )


def test_layer_base_not_checked_without_eplb():
    # A draft-only checkpoint config with its own depth stays valid when EPLB is
    # off -- the layer namespace only has to line up for EPLB placement keys.
    with patch.object(modeling_dspark, "_active_moe_load_balancer", return_value=None):
        validate_dspark_eplb_layer_base(
            _model_config(None), _model_config(None, num_hidden_layers=3)
        )


# ---------------------------------------------------------------------------
# Deployment-form probe. Every DSpark dispatch -- draft model, worker, spec
# metadata, draft-KV decision -- reads this one flag, so a misread does not
# degrade gracefully: it routes a standalone drafter into the V4 worker.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index_name,weight_map,model_type,expected",
    [
        # A parsed index is authoritative in BOTH directions. Falling through
        # to model_type on a standalone V4-shaped drafter would classify it as
        # embedded, and that only surfaces inside count_dspark_stages.
        ("model.safetensors.index.json", {"mtp.0.layers.0.weight": "a"}, "deepseek_v4", True),
        (
            "model.safetensors.index.json",
            {"layers.0.self_attn.q_proj.weight": "a"},
            "deepseek_v4",
            False,
        ),
        # The bin index is probed too; only the safetensors one used to be.
        ("pytorch_model.bin.index.json", {"mtp.1.mlp.weight": "a"}, "qwen3", True),
        # No index at all -> the model_type fallback.
        (None, None, "deepseek_v4", True),
        (None, None, "qwen3", False),
    ],
    ids=["mtp_index", "standalone_index", "bin_index", "no_index_v4", "no_index_qwen3"],
)
def test_draft_form_probe_reads_the_checkpoint(
    tmp_path, index_name, weight_map, model_type, expected
):
    import json

    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    if index_name is not None:
        (tmp_path / index_name).write_text(json.dumps({"weight_map": weight_map}))

    cfg = DSparkDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path))

    assert cfg.draft_is_embedded_in_target is expected
