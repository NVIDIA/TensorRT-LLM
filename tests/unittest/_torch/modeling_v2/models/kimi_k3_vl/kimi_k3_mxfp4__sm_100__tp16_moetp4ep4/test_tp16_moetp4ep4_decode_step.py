# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the tp16_moetp4ep4 target classifies a step for its decode kernels (host-side, no GPU): ``decode_step``, the
attention-residual epilogue ceiling it sets, which steps the MoE layers push on, the KDA verify's dense output, the one
path the text model runs for a step, and the DSpark worker reading the current batch-to-slot buffer."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAIntermediateLayerCache,
    KDAReplayLayerCache,
)
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import _KdaPrequantizedCore

__extra_import_path__ = [".."]
from _kimi_k3_decode_step import (  # noqa: E402
    ATTN_RES_CEILINGS,
    GENERIC_STEPS,
    LATENT_PUSH_STEPS,
    PURE_DECODE_STEPS,
    SHORT_STEPS,
    SLOT_ORDERS,
    STEP_PATHS,
    check_attn_res_epilogue_ceiling,
    check_drafter_inputs_use_current_slot_buffer,
    check_each_step_takes_one_path_through_every_layer,
    check_latent_push,
    check_pure_decode_step,
    check_short_step_is_small_only,
    check_spec_accept_uses_current_slot_buffer,
    check_step_takes_the_generic_path,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        modeling as target,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        spec_worker,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("requests,tokens", PURE_DECODE_STEPS)
def test_pure_decode_steps(requests, tokens):
    """R <= 8 generation requests of the same T <= 8 tokens: a decode step, small up to one token tile, wide above."""
    check_pure_decode_step(target, requests, tokens)


@pytest.mark.parametrize("seq_lens,num_contexts,rows", SHORT_STEPS)
def test_short_steps_are_small_only(seq_lens, num_contexts, rows):
    check_short_step_is_small_only(target, seq_lens, num_contexts, rows)


@pytest.mark.parametrize("seq_lens,num_contexts,rows", GENERIC_STEPS)
def test_other_steps_take_the_generic_path(seq_lens, num_contexts, rows):
    check_step_takes_the_generic_path(target, seq_lens, num_contexts, rows)


@pytest.mark.parametrize("step,ceiling", ATTN_RES_CEILINGS)
def test_attn_res_epilogue_ceiling(step, ceiling):
    check_attn_res_epilogue_ceiling(target, step, ceiling)


@pytest.mark.parametrize("step,given,pushes", LATENT_PUSH_STEPS)
def test_latent_push(step, given, pushes):
    check_latent_push(target, step, given, pushes)


def _kda_verify_module(monkeypatch):
    module = object.__new__(target.K3DecodeKDA)
    torch.nn.Module.__init__(module)
    module.num_heads = 2
    module.head_dim = 128
    module.k3_proj_weight = torch.empty(1)
    monkeypatch.setattr(module, "k3_buffers", object(), raising=False)
    return module


@pytest.mark.parametrize("allow_prequantized_output", [False, True])
@pytest.mark.parametrize("use_output_buffer", [False, True])
def test_kda_verify_accepts_prequantized_permission_with_dense_replay_output(
    monkeypatch, allow_prequantized_output, use_output_buffer
):
    module = _kda_verify_module(monkeypatch)
    core = torch.arange(8 * 2 * 128, dtype=torch.float32).to(torch.bfloat16).reshape(8, 2, 128)
    output = torch.empty_like(core) if use_output_buffer else None
    cache = types.SimpleNamespace(has_kda_replay_caches=True, kda_state_tok=torch.empty(1))
    monkeypatch.setattr(module, "_k3_verify", lambda *args: core)
    result = module.forward_verify(
        torch.empty(8, 16),
        8,
        cache,
        None,
        None,
        torch.tensor([0]),
        output=output,
        allow_prequantized_output=allow_prequantized_output,
    )
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, core, rtol=0, atol=0)
    if output is not None:
        assert result is output


@pytest.mark.parametrize("allow_prequantized_output", [False, True])
@pytest.mark.parametrize("use_output_buffer", [False, True])
@pytest.mark.parametrize("has_replay_caches", [False, True])
def test_kda_verify_keeps_builtin_fallback_dense(
    monkeypatch, allow_prequantized_output, use_output_buffer, has_replay_caches
):
    module = _kda_verify_module(monkeypatch)
    monkeypatch.setattr(
        module, "_dispatch", types.SimpleNamespace(verify_kernel_path="optimized"), raising=False
    )
    core = torch.arange(8 * 2 * 128, dtype=torch.float32).to(torch.bfloat16).reshape(8, 2, 128)
    output = torch.empty_like(core) if use_output_buffer else None
    cache = (
        KDAReplayLayerCache(
            conv=torch.empty(0),
            temporal=torch.empty(0),
            kda_conv_q=torch.empty(0),
        )
        if has_replay_caches
        else KDAIntermediateLayerCache(conv=torch.empty(0), temporal=torch.empty(0))
    )

    mxfp8_input = (torch.empty(8, 16, dtype=torch.uint8), torch.empty(4, dtype=torch.uint8))
    seen_inputs = []

    def verify_fused(
        self, *args, output=None, allow_prequantized_output=False, prequantized_input=None
    ):
        seen_inputs.append(prequantized_input)
        if allow_prequantized_output:
            return _KdaPrequantizedCore(core.reshape(8, -1), torch.ones(1, dtype=torch.uint8))
        return self._store_core(core, output)

    monkeypatch.setattr(target.KimiKDALinearAttention, "forward_verify_fused", verify_fused)
    monkeypatch.setattr(
        target.KimiKDALinearAttention,
        "forward_verify_sequential",
        lambda self, *args, output=None, prequantized_input=None: (
            seen_inputs.append(prequantized_input) or self._store_core(core, output)
        ),
    )
    result = module.forward_verify(
        torch.empty(8, 16),
        8,
        cache,
        None,
        None,
        torch.tensor([0]),
        output=output,
        allow_prequantized_output=allow_prequantized_output,
        prequantized_input=mxfp8_input,
    )
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, core, rtol=0, atol=0)
    if output is not None:
        assert result is output
    # The MXFP8 input reaches the built-in verify path.
    assert len(seen_inputs) == 1 and seen_inputs[0] is mxfp8_input


@pytest.mark.parametrize("seq_lens,num_contexts,rows,path", STEP_PATHS)
def test_each_step_takes_one_path_through_every_layer(seq_lens, num_contexts, rows, path):
    check_each_step_takes_one_path_through_every_layer(target, seq_lens, num_contexts, rows, path)


@pytest.mark.parametrize("request_ids,expected", SLOT_ORDERS)
def test_spec_accept_uses_current_slot_buffer(monkeypatch, request_ids, expected):
    check_spec_accept_uses_current_slot_buffer(monkeypatch, spec_worker, request_ids, expected)


@pytest.mark.parametrize("num_contexts", [0, 1])
@pytest.mark.parametrize("paged", [False, True])
def test_drafter_inputs_use_current_slot_buffer(monkeypatch, num_contexts, paged):
    check_drafter_inputs_use_current_slot_buffer(monkeypatch, spec_worker, num_contexts, paged)
