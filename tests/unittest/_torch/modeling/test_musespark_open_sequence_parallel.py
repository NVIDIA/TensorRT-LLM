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
"""Run with four MPI ranks to check token order and the MoE gather contract."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.distributed import AllReduce, allgather, reducescatter
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import (
    LayerNvtx,
    MuseSparkOpenAttention,
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
    MuseSparkOpenMoE,
    MuseSparkOpenRMSNormWeightFirst,
    _pad_sequence_parallel_tokens,
    _prefill_sp_min_tokens,
    _reserve_mnnvl_all_reduce_workspace,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping


class _Routing:
    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (logits.shape[0], 1)
        return logits.new_zeros(shape, dtype=torch.int32), logits.new_ones(shape)

    def clear_routing_memo(self) -> None:
        pass


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.comm = None
        self.routing_method = _Routing()

    def forward(self, pooled: torch.Tensor, logits: torch.Tensor, **kwargs) -> tuple:
        assert pooled.shape[0] == logits.shape[0]
        return pooled, None, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [4096, 4097, 4100])
@pytest.mark.parametrize("owner", [0, 3])
@torch.no_grad()
def test_prefill_moe_token_shards(tokens: int, owner: int) -> None:
    if mpi_world_size() != 4:
        pytest.skip("Requires four MPI ranks")
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=4, rank=rank, tp_size=4, moe_ep_size=4)
    module = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    nn.Module.__init__(module)
    module.mapping = mapping
    module.experts = _Experts()
    module.nvtx = LayerNvtx(
        "moe", 0, ("router", "pool", "experts", "finalize", "all_reduce", "unpool")
    )
    module.all_reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    module.router_logits = lambda hidden: hidden[:, :4].float()
    module.pooler = nn.Identity()
    module.unpooler = nn.Identity()
    module._pool_gather_workspace = None
    module._unpool_gather_workspace = None
    prepared_rows = []

    def prepare(pooled: torch.Tensor) -> torch.Tensor:
        prepared_rows.append(pooled.shape[0])
        return pooled * 0.5

    module.prepare_expert_input = prepare
    # Only one rank owns each synthetic expert contribution. This avoids
    # conflating collective row-order checks with BF16 summation-order changes.
    module.finalize = lambda output, *_: output if rank == owner else torch.zeros_like(output)
    for seed in (41, 42):
        torch.manual_seed(seed)
        hidden = torch.randint(-16, 17, (tokens, 16), device="cuda").to(torch.bfloat16)
        expected = module(hidden, None)
        local = _pad_sequence_parallel_tokens(hidden, mapping.tp_size).chunk(4)[rank].contiguous()
        actual_local = module(local, None, sp_num_tokens=tokens)
        actual = allgather(actual_local, mapping, dim=0)[:tokens]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, hidden * 0.5, rtol=0, atol=0)
        assert actual_local.shape == ((tokens + 3) // 4, 16)
        # The real fused norm/quantizer must receive global, unpadded rows.
        assert prepared_rows[-1] == tokens


def test_sequence_parallel_padding_preserves_rows() -> None:
    original = torch.arange(30).reshape(5, 6)
    padded = _pad_sequence_parallel_tokens(original, 4)
    torch.testing.assert_close(padded[:5], original, rtol=0, atol=0)
    assert padded.shape == (8, 6)
    assert torch.count_nonzero(padded[5:]) == 0
    assert _pad_sequence_parallel_tokens(padded, 4) is padded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens,contexts", [(128, 1), (2049, 1), (4097, 0), (4097, 1), (8193, 1)])
@pytest.mark.parametrize("projection", ["default", "padding", "overlap"])
@pytest.mark.parametrize("min_tokens", [4096, 2048])
@torch.no_grad()
def test_prefill_model_boundary_shards(
    tokens: int, contexts: int, projection: str, min_tokens: int
) -> None:
    if mpi_world_size() != 4:
        pytest.skip("Requires four MPI ranks")
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(1271)
    mapping = Mapping(world_size=4, rank=rank, tp_size=4, moe_ep_size=4)
    reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._sp_mapping = mapping
    model._prefill_sp_enabled = False
    model._prefill_sp_min_tokens = min_tokens
    model.embed_norm = RMSNorm(
        hidden_size=256, eps=1e-5, dtype=torch.bfloat16, has_weights=False
    ).cuda()
    model.norm = RMSNorm(hidden_size=256, eps=1e-5, dtype=torch.bfloat16).cuda()
    attn_calls = []
    ffn_rows = []

    def attention(
        hidden_states,
        all_reduce_params=None,
        overlap_sp_output_projection=False,
        pad_sp_output_projection=False,
        **kwargs,
    ):
        attn_calls.append((hidden_states.shape[0], all_reduce_params is not None))
        partial = hidden_states * 0.7 if rank == 0 else torch.zeros_like(hidden_states)
        if all_reduce_params is None:
            return reduce(partial)
        assert all_reduce_params.enable_allreduce is False
        if overlap_sp_output_projection or pad_sp_output_projection:
            return reducescatter(_pad_sequence_parallel_tokens(partial, 4), mapping, dim=0)
        return partial

    def feed_forward(hidden, metadata, *, sp_num_tokens=None):
        ffn_rows.append(hidden.shape[0])
        return hidden * 0.9

    layers = []
    for index in range(2):
        layer = MuseSparkOpenDecoderLayer.__new__(MuseSparkOpenDecoderLayer)
        nn.Module.__init__(layer)
        layer.mapping = mapping
        layer._overlap_sp_projection_enabled = projection == "overlap"
        layer._pad_sp_projection_enabled = projection == "padding"
        layer.layer_idx = index
        layer.dtype = torch.bfloat16
        layer.nvtx = LayerNvtx(
            "layer", index, ("pre_qkv_proj_norm", "attn", "norm_add_attn", "moe", "norm_add_ffn")
        )
        layer.pre_qkv_proj_norm = RMSNorm(
            hidden_size=256, eps=1e-5, dtype=torch.bfloat16, use_gemma=True
        ).cuda()
        layer.pre_ffn_norm = RMSNorm(
            hidden_size=256, eps=1e-5, dtype=torch.bfloat16, use_gemma=True
        ).cuda()
        layer.post_attn_norm = RMSNorm(
            hidden_size=256, eps=1e-5, dtype=torch.bfloat16, has_weights=False
        ).cuda()
        layer.post_ffn_norm = MuseSparkOpenRMSNormWeightFirst(256, 1e-5, torch.bfloat16).cuda()
        layer.add_attn = MuseSparkOpenGatedResidual(256, torch.bfloat16).cuda()
        layer.add_ffn = MuseSparkOpenGatedResidual(256, torch.bfloat16).cuda()
        layer.self_attn = attention
        layer.feed_forward = feed_forward
        for parameter in layer.parameters():
            parameter.normal_(0, 0.03)
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    for module in model.modules():
        if isinstance(module, (MuseSparkOpenGatedResidual, MuseSparkOpenRMSNormWeightFirst)):
            module.cache_derived_state()
    metadata = SimpleNamespace(num_contexts=contexts)
    for seed in (17, 18):
        torch.manual_seed(seed)
        inputs = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        model._prefill_sp_enabled = False
        expected = model(attn_metadata=metadata, inputs_embeds=inputs)
        model._prefill_sp_enabled = True
        actual = model(attn_metadata=metadata, inputs_embeds=inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        enabled = contexts > 0 and tokens >= min_tokens
        assert attn_calls[-2:] == [(tokens, enabled)] * 2
        assert ffn_rows[-2:] == [((tokens + 3) // 4 if enabled else tokens)] * 2


class _Projection(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, inputs: torch.Tensor, all_reduce_params=None, **kwargs) -> torch.Tensor:
        if all_reduce_params is not None:
            assert not all_reduce_params.enable_allreduce
        return torch.nn.functional.linear(inputs, self.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shared_stream", [False, True])
@pytest.mark.parametrize("strategy", [AllReduceStrategy.ONESHOT, AllReduceStrategy.NCCL])
@torch.no_grad()
def test_overlapped_projection_uneven_rows_and_stream_reuse(
    shared_stream: bool, strategy: AllReduceStrategy
) -> None:
    if mpi_world_size() != 4:
        pytest.skip("Requires four MPI ranks")
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    module = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(module)
    module.mapping = Mapping(world_size=4, rank=rank, tp_size=4, moe_ep_size=4)
    module._sp_projection_stream = torch.cuda.Stream() if shared_stream else None
    torch.manual_seed(523 + rank)
    module.o_proj = Linear(
        8192,
        8192,
        bias=False,
        dtype=torch.bfloat16,
        mapping=module.mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=strategy,
    ).cuda()
    module.o_proj.weight.copy_(torch.randn_like(module.o_proj.weight) / 32)
    other = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(other)
    other.mapping = module.mapping
    other.o_proj = module.o_proj
    other._sp_projection_stream = module._sp_projection_stream
    modules = (module, other)
    main = torch.cuda.current_stream()
    alternate = torch.cuda.Stream()
    for index, tokens in enumerate((4097, 8193, 16385, 8192, 32769)):
        module = modules[index % 2]
        inputs = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
        partial = module.o_proj(inputs, all_reduce_params=AllReduceParams(enable_allreduce=False))
        expected = reducescatter(_pad_sequence_parallel_tokens(partial, 4), module.mapping, dim=0)
        stream = main if index % 2 == 0 else alternate
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            actual = module._overlap_sp_output_projection(inputs)
            padded_actual = module._pad_sp_output_projection(inputs)
        main.wait_stream(stream)
        actual.record_stream(main)
        padded_actual.record_stream(main)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(padded_actual, expected, rtol=0, atol=0)
        assert actual.shape == ((tokens + 3) // 4, 8192)

    # Keep outputs until after allocator pressure on both streams. Compare only
    # after every call has been queued, so assertions cannot hide lifetime races.
    queued_outputs = []
    for index in range(24):
        caller = main if index % 2 == 0 else alternate
        caller.wait_stream(main)
        module = modules[index % 2]
        with torch.cuda.stream(caller):
            actual = module._overlap_sp_output_projection(inputs)
            partial_rows = ((inputs.shape[0] + 3) // 4 + 1) // 2 * 4
            poison_main = torch.empty(partial_rows, 8192, device=inputs.device, dtype=inputs.dtype)
            poison_main.fill_(float("nan"))
            with torch.cuda.stream(module._sp_projection_stream):
                poison_aux = torch.empty(
                    partial_rows // 4, 8192, device=inputs.device, dtype=inputs.dtype
                )
                poison_aux.fill_(float("nan"))
            caller.wait_stream(module._sp_projection_stream)
        main.wait_stream(caller)
        actual.record_stream(main)
        queued_outputs.append(actual)
        del poison_main, poison_aux
    for actual in queued_outputs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_projection_callback_stays_out_of_backend_arguments() -> None:
    module = Attention.__new__(Attention)
    nn.Module.__init__(module)
    module.mapping = Mapping()
    module.mapping_o = module.mapping
    module.layer_idx = 0
    module.rope_fusion = True
    module.sparse_attn_hooks = None
    module.qkv_proj = nn.Identity()
    module.o_proj = _Projection(torch.eye(4) * 3)
    module.preprocess_qkv = lambda qkv, _: (qkv, None, None, None)
    module.convert_qkv = lambda q, k, v: (q, k, v)

    def forward_impl(q, *args, **kwargs):
        assert "output_projection" not in kwargs
        return q * 2

    module.forward_impl = forward_impl
    inputs = torch.arange(12).reshape(3, 4).float()
    metadata = SimpleNamespace()
    baseline = module(None, inputs, metadata)
    custom = module(None, inputs, metadata, output_projection=lambda value: value * 2)
    torch.testing.assert_close(baseline, inputs * 6, rtol=0, atol=0)
    torch.testing.assert_close(custom, inputs * 4, rtol=0, atol=0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 4096),
        ("", 4096),
        ("2048", 2048),
        ("1024", 1024),
        ("1023", 4096),
        ("512", 4096),
        ("0", 4096),
        ("many", 4096),
    ],
)
def test_prefill_sp_min_tokens_from_env(monkeypatch, value, expected: int) -> None:
    if value is None:
        monkeypatch.delenv("MUSESPARK_OPEN_PREFILL_SP_MIN_TOKENS", raising=False)
    else:
        monkeypatch.setenv("MUSESPARK_OPEN_PREFILL_SP_MIN_TOKENS", value)
    assert _prefill_sp_min_tokens() == expected


def _reducer_layer(o_proj_mapping, moe_mapping) -> SimpleNamespace:
    """A decoder layer reduced to the all-reduce modules the MNNVL reservation reads."""

    def reducer(mapping):
        mnnvl = None if mapping is None else SimpleNamespace(mapping=mapping, dtype=torch.bfloat16)
        return SimpleNamespace(mnnvl_allreduce=mnnvl)

    return SimpleNamespace(
        self_attn=SimpleNamespace(
            o_proj=SimpleNamespace(all_reduce=reducer(o_proj_mapping), out_features=8192)
        ),
        feed_forward=SimpleNamespace(all_reduce=reducer(moe_mapping), expert_hidden_size=4096),
    )


@pytest.mark.parametrize("moe_mnnvl", [False, True], ids=["moe_nccl", "moe_mnnvl"])
def test_mnnvl_reservation_uses_each_reducer_mapping(monkeypatch, moe_mnnvl: bool) -> None:
    # The attention output projection builds its own mapping without expert
    # parallelism, so it keys a different shared workspace than the model mapping.
    model_mapping = Mapping(world_size=4, rank=0, gpus_per_node=4, tp_size=4, moe_ep_size=4)
    o_proj_mapping = Mapping(world_size=4, rank=0, gpus_per_node=4, tp_size=4)
    assert o_proj_mapping != model_mapping
    calls = []
    monkeypatch.setattr(
        modeling_musespark_open,
        "get_or_scale_allreduce_mnnvl_workspace",
        lambda mapping, dtype, buffer_size_bytes: calls.append((mapping, dtype, buffer_size_bytes)),
    )
    layers = [
        _reducer_layer(o_proj_mapping, model_mapping if moe_mnnvl else None) for _ in range(3)
    ]
    _reserve_mnnvl_all_reduce_workspace(layers, 16384)
    expected = [
        (
            o_proj_mapping,
            torch.bfloat16,
            MNNVLAllReduce.get_required_workspace_size(16384, 8192, 4, torch.bfloat16),
        )
    ]
    if moe_mnnvl:
        expected.append(
            (
                model_mapping,
                torch.bfloat16,
                MNNVLAllReduce.get_required_workspace_size(16384, 4096, 4, torch.bfloat16),
            )
        )
    assert calls == expected


def test_mnnvl_reservation_skips_non_mnnvl_reducers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        modeling_musespark_open,
        "get_or_scale_allreduce_mnnvl_workspace",
        lambda *args, **kwargs: calls.append(args),
    )
    _reserve_mnnvl_all_reduce_workspace([_reducer_layer(None, None)], 16384)
    assert calls == []


def test_mnnvl_reservation_runs_once_outside_capture(monkeypatch) -> None:
    # The reservation waits for the first forward that is not captured, then
    # never runs again; with the split off nothing is pending.
    calls = []
    capturing = [True]
    monkeypatch.setattr(
        modeling_musespark_open,
        "_reserve_mnnvl_all_reduce_workspace",
        lambda layers, tokens: calls.append((layers, tokens)),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    model = SimpleNamespace(layers="layers", _mnnvl_reserve_tokens=16384)
    MuseSparkOpenModel._reserve_mnnvl_workspaces_once(model)
    assert calls == [] and model._mnnvl_reserve_tokens == 16384
    capturing[0] = False
    MuseSparkOpenModel._reserve_mnnvl_workspaces_once(model)
    MuseSparkOpenModel._reserve_mnnvl_workspaces_once(model)
    assert calls == [("layers", 16384)] and model._mnnvl_reserve_tokens is None
    assert MuseSparkOpenModel._mnnvl_reserve_tokens is None


def test_split_off_keeps_the_default_threshold_and_no_reservation() -> None:
    # Without MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL the constructor reads neither the
    # threshold nor sets a reservation: the class defaults stand.
    assert MuseSparkOpenModel._prefill_sp_min_tokens == 4096
    assert MuseSparkOpenModel._mnnvl_reserve_tokens is None
