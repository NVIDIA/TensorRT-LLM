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

import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivationSupport,
    SiTuActivation,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import _BACKEND_SYNC_ATTRS, ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.interface import moe_custom_op
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor, model_extra_attrs
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Nothing here allocates: the quant-config tests drive ``create_weights`` with
# construction stubbed out, and the activation test only builds a carrier. The
# marker is also what makes the file reachable: the CPU stage lists this
# directory but collects only files that carry it.
pytestmark = pytest.mark.cpu_only


def _trtllm_gen_fake_wrapper() -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.backend = TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl)
    torch.nn.Module.__init__(wrapper.backend)
    wrapper.backend.hidden_size = 2880
    wrapper.backend.mapping = SimpleNamespace(tp_rank=1)
    return wrapper


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_tokens", [0, 7])
@pytest.mark.parametrize("counts", [None, [3, 5]])
@pytest.mark.parametrize("output_dtype", [None, torch.float16])
@pytest.mark.parametrize("custom_op", [False, True])
def test_trtllm_gen_nvfp4_forward_fake_logical_output(
    num_tokens: int,
    counts: list[int] | None,
    output_dtype: torch.dtype | None,
    custom_op: bool,
) -> None:
    wrapper = _trtllm_gen_fake_wrapper()
    with FakeTensorMode(), model_extra_attrs({"moe_layers": {"0": weakref.ref(wrapper)}}):
        fp4 = torch.empty((num_tokens, 3072 // 2), dtype=torch.uint8)
        scales = torch.empty((num_tokens, 3072 // 16), dtype=torch.uint8)
        router_logits = torch.empty((num_tokens, 8), dtype=torch.bfloat16)
        if custom_op:
            # The real compile boundary reconstructs a carrier without a BF16
            # view, so shape inference must use the backend's logical width.
            output = moe_custom_op(
                "0", fp4, scales, False, router_logits, True, output_dtype, counts, False
            )[0]
        else:
            output = wrapper.forward_fake(
                Fp4QuantizedTensor(fp4, scales, is_sf_swizzled=False),
                router_logits,
                output_dtype=output_dtype,
                all_rank_num_tokens=counts,
                use_dp_padding=False,
            )
        expected_tokens = counts[1] if counts is not None else num_tokens
        assert output.shape == (expected_tokens, 2880)
        assert output.dtype == torch.bfloat16
        assert output.device == fp4.device
        assert output.is_contiguous()


@pytest.mark.cpu_only
def test_trtllm_gen_nvfp4_custom_op_fake_symbolic_tokens() -> None:
    wrapper = _trtllm_gen_fake_wrapper()

    def forward(
        fp4: torch.Tensor, scales: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        return moe_custom_op(
            "0", fp4, scales, False, router_logits, True, torch.bfloat16, None, False
        )[0]

    with model_extra_attrs({"moe_layers": {"0": weakref.ref(wrapper)}}):
        graph = make_fx(forward, tracing_mode="symbolic")(
            torch.empty((7, 1536), dtype=torch.uint8),
            torch.empty((7, 192), dtype=torch.uint8),
            torch.empty((7, 8), dtype=torch.bfloat16),
        )

    calls = [
        node
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.trtllm.moe_custom_op.default
    ]
    assert len(calls) == 1
    packed_input = calls[0].args[1].meta["val"]
    output = calls[0].meta["val"][0]
    assert isinstance(output.shape[0], torch.SymInt)
    assert output.shape[0] == packed_input.shape[0]
    assert output.shape[1] == 2880
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("sections", [[1, 3], [0, 2, 2], [4]])
def test_nvfp4_token_split_preserves_dynamic_scale(sections: list[int]) -> None:
    scale = torch.tensor([0.125])
    carrier = Fp4QuantizedTensor(
        torch.arange(32, dtype=torch.uint8).view(4, 8),
        torch.arange(4, dtype=torch.uint8).view(4, 1),
        is_sf_swizzled=False,
        reciprocal_scale=scale,
    )
    chunks = carrier.split(sections)
    assert [chunk.shape[0] for chunk in chunks] == sections
    assert all(chunk.reciprocal_scale is scale for chunk in chunks)
    torch.testing.assert_close(
        torch.cat([chunk.fp4_tensor for chunk in chunks]), carrier.fp4_tensor, rtol=0, atol=0
    )
    torch.testing.assert_close(scale, torch.tensor([0.125]), rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("post_quant_dispatch", [None, False, True])
def test_nvfp4_input_quant_params_require_post_quant_dispatch(
    post_quant_dispatch: bool | None,
) -> None:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.comm = (
        None
        if post_quant_dispatch is None
        else Mock(supports_post_quant_dispatch=Mock(return_value=post_quant_dispatch))
    )
    params = (torch.ones(1), 3072)
    wrapper.backend = Mock(get_nvfp4_input_quant_params=Mock(return_value=params))

    result = wrapper.get_nvfp4_input_quant_params()

    if post_quant_dispatch is False:
        assert result is None
        wrapper.backend.get_nvfp4_input_quant_params.assert_not_called()
    else:
        assert result is params
        wrapper.backend.get_nvfp4_input_quant_params.assert_called_once_with()


@pytest.mark.cpu_only
@pytest.mark.parametrize("counts", [[1, 7], [7, 1], [2, 9, 5]])
@pytest.mark.parametrize("alltoall", [False, True])
def test_nvfp4_input_carrier_chunking_with_uneven_dp_ranks(
    counts: list[int],
    alltoall: bool,
) -> None:
    num_chunks = 4
    rank_chunk_counts = []
    for rank, num_tokens in enumerate(counts):
        wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
        torch.nn.Module.__init__(wrapper)
        wrapper.use_dp = True
        wrapper.rank = rank
        wrapper.comm = Mock() if alltoall else None
        wrapper.aux_stream = None
        wrapper.repeat_idx = 0
        wrapper.repeat_count = 1
        wrapper.backend = SimpleNamespace(
            input_requirement=SimpleNamespace(requires_run_moe_workspace=False)
        )
        scheduler = ExternalCommMoEScheduler(wrapper)
        norm_out = torch.arange(num_tokens * 16, dtype=torch.bfloat16).view(num_tokens, 16)
        carrier = Fp4QuantizedTensor(
            torch.arange(num_tokens * 24, dtype=torch.uint8).view(num_tokens, 24),
            torch.arange(num_tokens * 3, dtype=torch.uint8).view(num_tokens, 3),
            is_sf_swizzled=False,
            unquantized_hidden_states=norm_out,
        )
        scheduler._forward_chunk_impl = Mock(
            side_effect=lambda x, *args, **kwargs: x.unquantized_hidden_states
        )

        output = scheduler._forward_multiple_chunks(
            carrier, torch.zeros((num_tokens, 4)), num_chunks, torch.bfloat16, counts, False
        )

        torch.testing.assert_close(output, norm_out)
        calls = scheduler._forward_chunk_impl.call_args_list
        assert len(calls) == num_chunks
        rank_chunk_counts.append([call.args[3] for call in calls])
        chunks = carrier.split(wrapper.split_chunk(num_tokens, num_chunks))
        for call, chunk in zip(calls, chunks):
            expected = chunk if chunk.numel() else chunks[0]
            actual = call.args[0]
            assert isinstance(actual, Fp4QuantizedTensor)
            assert actual.shape[0] == call.args[3][rank] > 0
            assert not actual.is_sf_swizzled
            torch.testing.assert_close(actual.fp4_tensor, expected.fp4_tensor)
            torch.testing.assert_close(actual.scaling_factor, expected.scaling_factor)
            torch.testing.assert_close(
                actual.unquantized_hidden_states, expected.unquantized_hidden_states
            )

    assert all(chunk_counts == rank_chunk_counts[0] for chunk_counts in rank_chunk_counts)


def _wrapper() -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.num_experts = 8
    # Divides ``num_experts`` so ``_reject_non_divisible_ep_backend`` returns at
    # the divisibility check. It must not fall past it: the branch below reads
    # ``type(self.backend)._supports_non_divisible_ep``, and the backend here is
    # a ``Mock`` *instance*, so that lookup lands on the ``Mock`` class and
    # raises. Assigned because ``__new__`` skipped the ``MoE.__init__`` that
    # normally derives it from the mapping.
    wrapper.ep_size = 1
    wrapper.hidden_size = 16
    wrapper.intermediate_size = 32
    wrapper.dtype = torch.bfloat16
    wrapper.reduce_results = False
    wrapper.aux_stream_dict = None
    wrapper.weight_loading_mode = None
    wrapper.apply_router_weight_on_input = False
    # Both are read while the backend is being built: the carrier goes to
    # ``resolve_moe_cls`` and to ``install_activation_params``, the kind to
    # ``infer_swiglu_gptoss_style``. ``__new__`` skipped the ``__init__`` that
    # normally assigns them, so a fixture that sets neither fails before
    # reaching the quant-config behaviour these tests are about.
    wrapper.activation = DEFAULT_MOE_ACTIVATION
    wrapper.activation_type = ActivationType(DEFAULT_MOE_ACTIVATION.kind)
    wrapper.routing_method = Mock()
    wrapper._override_quant_config = None
    for attr in _BACKEND_SYNC_ATTRS:
        setattr(wrapper, attr, None)
    return wrapper


def _backend_mock() -> Mock:
    """A ``Mock`` backend that survives ``install_activation_params``.

    That call cannot be patched out here: ``ConfigurableMoE`` makes it on the
    backend after the EPLB sync *and* inside ``create_weights``, which is the
    code path under test. A bare ``Mock`` slips past
    ``resolve_activation_support`` -- every attribute of a Mock is callable, so
    the override branch is taken -- and then fails inside
    ``materialize_activation_params``, which uses the returned Mock as a real
    declaration. Declaring a real one keeps the failure surface at the quant
    config. Still GPU-free: ``DEFAULT_MOE_ACTIVATION`` carries no constants, so
    every register short-circuits to None without allocating.
    """
    backend = Mock()
    backend.activation = DEFAULT_MOE_ACTIVATION
    backend.resolve_activation_support = Mock(
        return_value=MoEActivationSupport(
            kinds=frozenset({ActivationType.Swiglu}),
            alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
            limit=ActivationParamShape.PER_EXPERT_TENSOR,
        )
    )
    return backend


def _create_backend(
    wrapper: ConfigurableMoE,
    model_config: ModelConfig,
    override_quant_config: QuantConfig | None = None,
) -> Mock:
    backend = _backend_mock()
    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.resolve_moe_cls",
            return_value=Mock(),
        ),
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe_backend",
            return_value=backend,
        ),
    ):
        wrapper._create_and_sync_backend(
            model_config=model_config,
            routing_method=Mock(),
            override_quant_config=override_quant_config,
        )
    return backend


def test_layerwise_quant_config_is_applied_before_weight_creation() -> None:
    global_config = QuantConfig()
    layer_config = QuantConfig()
    model_config = ModelConfig(
        quant_config=global_config,
        quant_config_dict={"model.layers.0.mlp.experts": layer_config},
    )
    wrapper = _wrapper()
    wrapper.quant_config = global_config

    backend = _create_backend(wrapper, model_config)

    backend.create_weights.assert_not_called()
    wrapper.quant_config = layer_config
    wrapper.create_weights()

    assert backend.quant_config is layer_config
    backend.create_weights.assert_called_once_with()


def test_exclusions_only_recreate_matching_moe_weights() -> None:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        exclude_modules=["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
    )
    model_config = ModelConfig(quant_config=quant_config)
    wrapper = _wrapper()
    wrapper.quant_config = quant_config

    backend = _create_backend(wrapper, model_config)

    assert backend.quant_config is quant_config
    backend.create_weights.assert_called_once_with()
    backend.create_weights.reset_mock()
    backend._weights_created = True

    root = torch.nn.Module()
    root.model_config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8,
            exclude_modules=["experts"],
        )
    )
    root.experts = wrapper

    DecoderModelForCausalLM.apply_quant_config_exclude_modules(root)

    assert not backend._weights_created
    wrapper.create_weights()
    assert wrapper.quant_config.quant_algo is None
    assert backend.quant_config.quant_algo is None
    backend.create_weights.assert_called_once_with()


@pytest.mark.parametrize("missing", ["gate_softcap", "linear_softcap"])
def test_situ_activation_rejects_a_missing_soft_cap(missing: str) -> None:
    """Absence is a rejection for a soft cap, unlike for a clamp.

    ``swiglu_limit`` has a value that encodes "unclamped", so omitting it is
    how a caller asks for plain SwiGLU. A soft cap has no such value -- the
    kernel divides by it -- so the two neighbouring constants cannot be given
    one rule.
    """
    caps = {"gate_softcap": 1.0, "linear_softcap": 1.0}
    caps[missing] = None

    with pytest.raises(ValueError, match=f"SiTu {missing}"):
        SiTuActivation(**caps)
