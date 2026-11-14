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
"""CPU contracts for the opt-in MegaMoE uGPU localization path.

The backend is the real class. VMM placement runs on CPU tensors through
MockVmm and the kernel launch is a mock op, so these tests claim no physical
locality and no GPU numerics.
"""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.moe.megamoe_ugpu_test_utils import MockVmm
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl, ugpu_weights
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as backend_module
from tensorrt_llm.mapping import Mapping

_PACKED_SHAPES = ((2, 1024, 512), (2, 65536), (2, 1024, 256), (2, 32768))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the CUDA/driver boundaries; record placement calls."""
    driver = MockVmm(granularity=128)
    state = SimpleNamespace(driver=driver, placements=[])

    def place(*tensors: torch.Tensor, mode: str) -> ugpu_weights.LocalizedMegaMoEWeights:
        state.placements.append(mode)
        return ugpu_weights._place_weights(
            tuple(t.detach() for t in tensors), mode=mode, driver=driver
        )

    monkeypatch.delenv("MEGAMOE_UGPU_LAYOUT", raising=False)
    monkeypatch.delenv("MEGAMOE_COMBINE_FORMAT", raising=False)
    monkeypatch.delenv("MEGAMOE_TACTIC_AUTOTUNE", raising=False)
    monkeypatch.setattr(backend_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(backend_module, "localize_megamoe_weights", place)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    return state


def _backend() -> MegaMoECuteDsl:
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    return MegaMoECuteDsl(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )


def _packed(backend: MegaMoECuteDsl) -> dict:
    """Register a packed bank and per-expert vectors as normal loading would."""
    generator = torch.Generator().manual_seed(7)
    for name, shape in zip(backend._UGPU_PACKED_PARAMS, _PACKED_SHAPES):
        data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        backend.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    for name in ("fc31_alpha", "fc2_alpha", "fc1_norm_const"):
        backend.register_parameter(name, torch.nn.Parameter(torch.ones(2), requires_grad=False))
    backend.quant_method = SimpleNamespace(transform_weights=Mock(), cache_derived_state=Mock())
    return {name: getattr(backend, name) for name in backend._UGPU_PACKED_PARAMS}


def _localized(env: SimpleNamespace, layout: str = "n_split") -> MegaMoECuteDsl:
    backend = _backend()
    backend.ep_size = 2
    backend._configure_ugpu_localization(layout)
    return backend


def test_default_off_and_invalid_layout_rejected(env, monkeypatch) -> None:
    backend = _backend()
    assert backend._ugpu_weight_layout == "none" and backend._ugpu_weight_owner is None
    monkeypatch.setenv("MEGAMOE_UGPU_LAYOUT", "bogus")
    with pytest.raises(ValueError, match="expert_half or n_split"):
        backend.create_weights()
    assert backend._ugpu_weight_layout == "none" and not env.placements


def test_post_load_installs_once_and_releases_original_bank(env) -> None:
    backend = _localized(env)
    originals = _packed(backend)
    expected = {name: tensor.detach().clone() for name, tensor in originals.items()}
    vectors = [backend.fc31_alpha, backend.fc2_alpha, backend.fc1_norm_const]
    released = [weakref.ref(tensor) for tensor in originals.values()]
    del originals
    backend.post_load_weights()
    backend.transform_weights()
    assert env.placements == ["n_split"]
    for name, original in expected.items():
        placed = getattr(backend, name)
        halves = placed.view(2, 2, original.shape[1] // 2, *original.shape[2:])
        torch.testing.assert_close(torch.cat((halves[0], halves[1]), dim=1), original)
    assert [backend.fc31_alpha, backend.fc2_alpha, backend.fc1_norm_const] == vectors
    gc.collect()
    assert all(ref() is None for ref in released)
    backend._ugpu_weight_owner.close()
    env.driver.assert_empty()


def test_failed_commit_restores_original_bank(env, monkeypatch) -> None:
    backend = _localized(env)
    originals = _packed(backend)
    real_setattr = type(backend).__setattr__

    def fail_second_swap(self, name, value):
        if name == "mega_fc1_weight_sf" and value is not originals[name]:
            raise RuntimeError("commit failed")
        real_setattr(self, name, value)

    monkeypatch.setattr(type(backend), "__setattr__", fail_second_swap)
    with pytest.raises(RuntimeError, match="commit failed"):
        backend._install_ugpu_localized_weights()
    assert all(getattr(backend, name) is value for name, value in originals.items())
    assert backend._ugpu_weight_owner is None
    env.driver.assert_empty()


@pytest.mark.parametrize(
    "field,value,ep_size",
    [("combine_format", "32e4m3xe8m0", 2), ("layer_load_balancer", object(), 2), (None, None, 1)],
)
def test_unsupported_configuration_fails_closed(env, field, value, ep_size) -> None:
    backend = _backend()
    backend.ep_size = ep_size
    if field is not None:
        setattr(backend, field, value)
    with pytest.raises(ValueError):
        backend._configure_ugpu_localization("n_split")
    assert backend._ugpu_weight_layout == "none" and not env.placements


def test_layout_reaches_query_and_provider(env, monkeypatch) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op_module

    query = Mock(return_value=(128, True))
    provider = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(op_module, "query_megamoe_kernel_properties", query, raising=False)
    monkeypatch.setattr(op_module, "get_megamoe_symm_provider", provider, raising=False)
    backend = _localized(env)
    backend._ep_pg = object()
    backend._alloc_symm_provider()
    assert query.call_count == provider.call_count == len(backend._maxt_buckets)
    for call in (*query.call_args_list, *provider.call_args_list):
        assert call.kwargs["ugpu_weight_layout"] == "n_split"


def _launch(backend: MegaMoECuteDsl, capacity: int, tokens: int) -> torch.Tensor:
    return backend._launch_megamoe_kernel(
        activation=torch.empty((capacity, 128), dtype=torch.uint8),
        activation_sf=torch.empty((capacity, 16), dtype=torch.uint8),
        topk_idx=torch.zeros((capacity, 2), dtype=torch.int32),
        topk_weights=torch.ones((capacity, 2)),
        weight_view=backend._build_weight_view(),
        combine_output=torch.zeros((capacity, 1, 256), dtype=torch.bfloat16),
        shared_workspace=torch.zeros(128, dtype=torch.uint8),
        world_size=2,
        local_rank=0,
        top_k=2,
        hidden=256,
        peer_offsets=[0, 0],
        num_tokens=tokens,
        output_dtype=torch.bfloat16,
        launch_max_T=capacity,
    )


def test_installed_bank_and_table_reach_the_op(env, monkeypatch) -> None:
    backend = _localized(env)
    _packed(backend)
    with pytest.raises(RuntimeError, match="before its weights were installed"):
        backend._run_moe(
            x=None,
            x_sf=None,
            topk_idx=None,
            topk_weights=None,
            weight_view=None,
            num_tokens=0,
            top_k=2,
            hidden=256,
            device=torch.device("cpu"),
            output_dtype=torch.bfloat16,
        )
    backend.post_load_weights()
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op_module

    launch = Mock()
    # Eager launches call the implementation directly; traced code uses the registered op.
    monkeypatch.setattr(op_module, "run_cute_dsl_megamoe_nvfp4_blackwell", launch, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_megamoe_nvfp4_blackwell", launch, raising=False)
    for capacity, tokens in ((64, 4), (1024, 0), (1025, 4)):
        _launch(backend, capacity, tokens)
        call = launch.call_args.kwargs
        assert call["ugpu_weight_layout"] == "n_split"
        assert "smid_die_table" not in call
        assert call["fc1_weight"].data_ptr() == backend.mega_fc1_weight.data_ptr()
    backend._ugpu_weight_owner.close()
    env.driver.assert_empty()
