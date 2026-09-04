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

"""Unit tests for Eagle3 model with AutoDeploy."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from _model_test_utils import get_small_model_config
from test_common.llm_data import hf_id_to_local_model_dir

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleRMSNorm, EagleWrapper
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    get_weight_shape,
    infer_draft_embedding_size,
    is_any_lin_op,
)

EAGLE_MODEL_HUB_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
NEMOTRON_SUPER_MODEL_HUB_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

NEMOTRON_SUPER_DRAFT_MODEL_KWARGS = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "layer_norm_epsilon": 1e-5,
    "residual_in_fp32": False,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "moe_shared_expert_intermediate_size": 64,
    "moe_latent_size": 16,
    "n_group": 1,
    "topk_group": 1,
}


def _build_small_draft_factory(
    model_hub_id: str, model_kwargs: dict | None = None
) -> EagleDrafterFactory:
    draft_model_path = hf_id_to_local_model_dir(model_hub_id)
    if draft_model_path is None or not Path(draft_model_path).is_dir():
        pytest.skip(
            f"Draft model {model_hub_id} not found (LLM_MODELS_ROOT not set or model missing)"
        )

    return EagleDrafterFactory(
        model=str(draft_model_path),
        model_kwargs=model_kwargs,
        skip_loading_weights=True,
        max_seq_len=64,
    )


@pytest.mark.cpu_only
def test_eagle_rmsnorm_keeps_fp32_weights():
    norm = EagleRMSNorm(hidden_size=16)

    assert norm.weight.dtype == torch.float32


def test_eagle_model_torch_export():
    """Test that Eagle3Model can be exported with torch.export.

    This validates that the model architecture is compatible with
    torch.export for potential TensorRT compilation.

    Note: We skip loading weights since torch.export only traces the computation
    graph (model architecture).
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel torch.export")
    print("=" * 80)

    eagle_model_path = hf_id_to_local_model_dir(EAGLE_MODEL_HUB_ID)
    if eagle_model_path is None:
        pytest.skip("Eagle model not found (LLM_MODELS_ROOT not set or model missing)")

    eagle_path = Path(eagle_model_path)

    # Setup
    device = torch.device("cuda")
    dtype = torch.float16

    # Create model via EagleDrafterFactory (creates EagleDrafterForCausalLM)
    factory = EagleDrafterFactory(model=str(eagle_path), skip_loading_weights=True)
    model = factory.build_model(device)
    config = model.config

    # Create inputs for export
    batch_size = 1
    seq_len = 8
    hidden_dim = config.hidden_size

    inputs_embeds = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    mock_hidden_states = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)

    print("Export input shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {mock_hidden_states.shape}")

    example_args = (
        inputs_embeds,
        position_ids,
    )

    # Attempt torch.export
    try:
        exported_program = torch.export.export(
            model, args=example_args, kwargs={"hidden_states": mock_hidden_states}
        )
        print("✅ torch.export successful!")
        print("Graph module code preview (first 20 lines):")
        code_lines = exported_program.graph_module.code.split("\n")[:20]
        print("\n".join(code_lines))
    except Exception as e:
        pytest.fail(f"torch.export failed: {e}")


@pytest.mark.parametrize(
    ("model_hub_id", "model_kwargs", "expected_is_eagle"),
    [
        (
            EAGLE_MODEL_HUB_ID,
            get_small_model_config(EAGLE_MODEL_HUB_ID)["args"]["model_kwargs"],
            True,
        ),
        (NEMOTRON_SUPER_MODEL_HUB_ID, NEMOTRON_SUPER_DRAFT_MODEL_KWARGS, False),
    ],
)
def test_infer_draft_hidden_size_from_exported_draft_graph(
    model_hub_id, model_kwargs, expected_is_eagle
):
    factory = _build_small_draft_factory(model_hub_id, model_kwargs=model_kwargs)
    model = factory.build_model("cuda")
    inner_model = model.model.eval()
    hidden_size = model.config.hidden_size
    dtype = model.config.torch_dtype

    batch_size = 2
    seq_len = 4
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    position_ids = (
        torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    )
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)

    gm = torch_export_to_gm(
        inner_model,
        args=(inputs_embeds, position_ids, hidden_states),
        clone=True,
    )
    gm.is_draft = True

    linear_nodes = [node for node in gm.graph.nodes if is_any_lin_op(node)]
    assert linear_nodes, "Expected exported draft graph to contain linear nodes"
    assert get_weight_shape(linear_nodes[-1], dim=0) == hidden_size

    embd, in_eagle_drafter = infer_draft_embedding_size(gm, linear_nodes)
    assert embd == hidden_size
    assert in_eagle_drafter is expected_is_eagle


###############################################################################
# sample_greedy broadcast gating (regression for #13134 + attention-DP fix).
#
# - TP replication: argmax across ranks can diverge due to FP noise; the
#   broadcast keeps acceptance patterns consistent.
# - Attention-DP: each rank holds a different slice of the global batch, so
#   per-rank tokens are legitimately different and the broadcast would
#   corrupt peers' state. The gate is `EagleWrapperConfig.enable_attention_dp`,
#   plumbed through to `EagleWrapper.enable_attention_dp`.
###############################################################################


def _make_bare_wrapper(enable_attention_dp: bool) -> EagleWrapper:
    """Construct a minimally-initialized EagleWrapper for testing sample_greedy.

    sample_greedy only reads self.enable_attention_dp; bypass __init__ to avoid
    requiring a real target/draft submodule pair.
    """
    wrapper = EagleWrapper.__new__(EagleWrapper)
    wrapper.enable_attention_dp = enable_attention_dp
    return wrapper


def test_sample_greedy_skips_broadcast_under_attention_dp():
    wrapper = _make_bare_wrapper(enable_attention_dp=True)
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    with patch("tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle.broadcast") as mock_bc:
        ret = wrapper.sample_greedy(logits)

    mock_bc.assert_not_called()
    assert ret.tolist() == [1, 0]


def test_sample_greedy_broadcasts_under_tp_replication():
    wrapper = _make_bare_wrapper(enable_attention_dp=False)
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    with patch("tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle.broadcast") as mock_bc:
        ret = wrapper.sample_greedy(logits)

    mock_bc.assert_called_once()
    args, kwargs = mock_bc.call_args
    # `broadcast(ret, src=0)` -- positional ret, src may be kwarg or positional.
    assert args[0] is ret
    src = kwargs.get("src", args[1] if len(args) > 1 else None)
    assert src == 0
    assert ret.tolist() == [1, 0]
