# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the tp16_moetp4ep4 target classifies a step for its decode kernels (host-side, no GPU): ``decode_step``, the
attention-residual epilogue ceiling it sets, which steps the MoE layers push on, the KDA verify's dense output, the one
path the text model runs for a step, and the DSpark worker reading the current batch-to-slot buffer."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAIntermediateLayerCache,
    KDAReplayLayerCache,
)

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
        kda as target_kda,
    )
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


def _kda_verify_module(monkeypatch, seen):
    """A K3DecodeKDA shell whose three verify paths record their call and store a known core."""
    module = object.__new__(target_kda.K3DecodeKDA)
    torch.nn.Module.__init__(module)
    module.num_heads = 2
    module.head_dim = 128
    module.layer_idx = 0
    module._decode_weights_built = True
    module.k3_buffers = object()
    core = torch.arange(8 * 2 * 128, dtype=torch.float32).to(torch.bfloat16).reshape(8, 2, 128)

    def k3_verify(x, num_steps, layer_cache, ssm_pool, slots):
        seen.append("_k3_verify")
        return core.reshape(8, -1)

    def path(name):
        def run(x, num_steps, layer_cache, slots, output, replay_slots=None):
            seen.append(name)
            return module._store_core(core.reshape(8, -1), output)

        return run

    monkeypatch.setattr(module, "_k3_verify", k3_verify)
    monkeypatch.setattr(module, "_verify_replay", path("_verify_replay"))
    monkeypatch.setattr(module, "_verify_sequential", path("_verify_sequential"))
    return module, core


def _verify_cache(kind):
    empty = torch.empty(0)
    if kind == "records":
        return KDAReplayLayerCache(
            conv=empty, temporal=empty, kda_conv_q=empty, kda_state_tok=torch.empty(1)
        )
    if kind == "replay":
        return KDAReplayLayerCache(conv=empty, temporal=empty, kda_conv_q=empty)
    return KDAIntermediateLayerCache(conv=empty, temporal=empty)


@pytest.mark.parametrize("use_output_buffer", [False, True])
@pytest.mark.parametrize(
    "kind,path",
    [
        ("records", "_k3_verify"),  # the per-draft records: k3_kda_attn / k3_kda_verify
        ("replay", "_verify_replay"),  # the replay caches alone: kda_mtp_decode
        ("sequential", "_verify_sequential"),  # the sequential-verify buffers: FLA token by token
    ],
)
def test_kda_verify_takes_the_path_of_the_managers_state(
    monkeypatch, kind, path, use_output_buffer
):
    """The KDA verify path follows the manager's speculative state, and every path returns the dense core
    [N T, H, 128], in the breakable CUDA graph's buffer when one is given."""
    seen = []
    module, core = _kda_verify_module(monkeypatch, seen)
    output = torch.empty_like(core) if use_output_buffer else None
    result = module._verify(
        torch.empty(8, 16), 8, _verify_cache(kind), torch.tensor([0], dtype=torch.int32), output
    )
    assert seen == [path]
    torch.testing.assert_close(result, core, rtol=0, atol=0)
    if output is not None:
        assert result is output


def test_kda_verify_records_need_the_decode_kernels(monkeypatch):
    """A manager keeping the per-draft records, which only the K3 kernels write, with a layer whose decode weights
    are not built: refused, not run on another path."""
    seen = []
    module, _ = _kda_verify_module(monkeypatch, seen)
    module._decode_weights_built = False
    with pytest.raises(AssertionError, match="per-token verify states"):
        module._verify(
            torch.empty(8, 16),
            8,
            _verify_cache("records"),
            torch.tensor([0], dtype=torch.int32),
            None,
        )
    assert seen == []


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
