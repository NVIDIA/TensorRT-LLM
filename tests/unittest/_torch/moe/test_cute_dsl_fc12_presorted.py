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
"""``CuteDslFc12FusedMoE`` on routing tables attached by the Kimi-K3 routing tail.

``trtllm::k3_route_tail`` (registered by the Kimi-K3 Rubin package) returns the top-16 of
``noaux_tc_op`` together with this rank's ``moe_sort`` outputs (128-row tiles, rows of an expert
in ascending token order) and the model attaches them to the top-k ids. FC12 then runs at tile 128
without ``moe_sort``.

Both paths compute every routed row identically; only the row order within an expert's tile
and the order of the BF16 scatter-adds into a token's output row can differ. So output rows
with at most two local experts (``0 + a + b`` rounds the same in either order) must be
bit-identical, the others equal up to the BF16 accumulation order.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.moe.fused_moe.cutedsl import fused_fc12 as fc12
from tensorrt_llm._torch.moe.fused_moe.cutedsl.kernel_inputs import (
    NvFp4WeightView,
    presorted_routing,
    sort_routing,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version

# The Kimi-K3 Rubin package: the routing tail op and the MoE runtime.
K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin"

NUM_EXPERTS = 896
TOP_K = 16
TILE = 128
# Small FC12 geometry: FC1 [2 * INTER, HIDDEN], FC2 [HIDDEN, INTER].
HIDDEN = 512
INTER = 256
SCALE = 2.5
# (local_expert_offset, local_num_experts): EP16 ranks 0 and 7, EP8 rank 3, EP4 rank 1, no EP.
LAYOUTS = [(0, 56), (7 * 56, 56), (3 * 112, 112), (224, 224), (0, 896)]
MS = [8, 16, 32, 64, 128]


def _rubin_fc12_available() -> bool:
    if not torch.cuda.is_available() or get_sm_version() != 107:
        return False
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return IS_CUTLASS_DSL_RUBIN_AVAILABLE


rubin_fc12 = pytest.mark.skipif(
    not _rubin_fc12_available(), reason="CUTEDSL_FC12 needs Rubin (SM107) and CuTe DSL Rubin"
)


def _k3_module(name):
    """A module of the Kimi-K3 Rubin package (``kernels.k3_route_tail``, ``text_model``)."""
    import importlib

    return importlib.import_module(f"{K3_RUBIN}.{name}")


# ---------------------------------------------------------------------------
# Dispatch (no GPU)
# ---------------------------------------------------------------------------


def _dispatch_backend(monkeypatch, tuned_tile):
    """FC12 stand-in whose fused impl and outer AutoTuner record what they are asked."""
    calls = {"tiles": [], "tuned": 0}

    def choose_one(op, runners, config, inputs):
        calls["tuned"] += 1
        return runners[0], tuned_tile

    monkeypatch.setattr(
        fc12.AutoTuner, "get", staticmethod(lambda: SimpleNamespace(choose_one=choose_one))
    )

    def impl(*args, enable_alltoall=False, tile_size=TILE):
        calls["tiles"].append(tile_size)
        return args[4]

    backend = SimpleNamespace(
        has_nvfp4=True,
        num_slots=NUM_EXPERTS,
        hidden_size=8,
        use_fused_finalize=True,
        _epilogue_identity=lambda: (int(ActivationType.SiTu), 4.0, 25.0),
        run_moe_nvfp4_impl=impl,
    )
    return backend, calls


def _run_dispatch(backend, ids, slot_start, num_local):
    num_tokens = ids.shape[0]
    return CuteDslFc12FusedMoE.run_moe_nvfp4(
        backend,
        x=torch.empty(num_tokens, 4, dtype=torch.uint8),
        token_selected_experts=ids,
        token_final_scales=torch.ones(num_tokens, TOP_K),
        x_sf=torch.empty(0),
        moe_output=torch.empty(num_tokens, 8, dtype=torch.bfloat16),
        weight_view=SimpleNamespace(expert_size_per_partition=num_local, slot_start=slot_start),
    )


def _attach(ids, key):
    ids._k3_presorted = (key, tuple(torch.empty(1, dtype=torch.int32) for _ in range(6)))
    return ids


def test_fc12_runs_attached_tables_at_their_tile(monkeypatch):
    """Tables of this rank: FC12 runs at their tile and does not consult the tuner."""
    backend, calls = _dispatch_backend(monkeypatch, tuned_tile=256)
    ids = _attach(torch.zeros(4, TOP_K, dtype=torch.int32), (336, 112, TILE))
    _run_dispatch(backend, ids, 336, 112)
    assert calls == {"tiles": [TILE], "tuned": 0}


@pytest.mark.parametrize(
    "key",
    [None, (0, 112, TILE), (336, 56, TILE), (336, 112, 512)],
    ids=["none", "other_offset", "other_range", "unsupported_tile"],
)
def test_fc12_tunes_without_matching_tables(monkeypatch, key):
    """No tables, another rank's tables or a tile FC12 lacks: the tuned tile as before."""
    backend, calls = _dispatch_backend(monkeypatch, tuned_tile=256)
    ids = torch.zeros(4, TOP_K, dtype=torch.int32)
    if key is not None:
        _attach(ids, key)
    _run_dispatch(backend, ids, 336, 112)
    assert calls == {"tiles": [256], "tuned": 1}


def test_sort_routing_uses_matching_tables_only(monkeypatch):
    """``sort_routing`` returns attached tables for this rank and tile, else runs ``moe_sort``."""
    import tensorrt_llm  # noqa: F401  (registers trtllm ops)

    sorted_calls = []

    def fake_moe_sort(**kwargs):
        sorted_calls.append(kwargs)
        return ("moe_sort",)

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", fake_moe_sort)
    backend = SimpleNamespace(num_slots=NUM_EXPERTS)

    def sort(*args):
        return sort_routing(backend, *args)

    view = SimpleNamespace(expert_size_per_partition=112, slot_start=336)
    ids = torch.zeros(4, TOP_K, dtype=torch.int32)
    scales = torch.ones(4, TOP_K)

    tables = tuple(torch.full((1,), i, dtype=torch.int32) for i in range(6))
    ids._k3_presorted = ((336, 112, TILE), tables)
    assert sort(ids, scales, view, TILE) is tables
    tile, attached = presorted_routing(ids, 336, 112)
    assert tile == TILE and attached is tables
    assert presorted_routing(ids, 0, 112) is None
    assert presorted_routing(torch.zeros(4, TOP_K, dtype=torch.int32), 336, 112) is None
    assert sorted_calls == []

    assert sort(ids, scales, view, 256) == ("moe_sort",)
    assert sorted_calls[-1]["tile_tokens_dim"] == 256
    assert sort(
        ids, scales, SimpleNamespace(expert_size_per_partition=112, slot_start=0), TILE
    ) == ("moe_sort",)
    assert sorted_calls[-1]["local_expert_offset"] == 0
    assert sorted_calls[-1]["top_k"] == TOP_K and sorted_calls[-1]["num_experts"] == NUM_EXPERTS


def test_k3_fused_latent_input_accepts_cutedsl_nvfp4_leaves():
    """The Kimi-K3 runtime fuses the experts' NVFP4 input quantization into latent_down
    (``TRTLLM_K3_LD_FUSED``) for backends that quantize with
    ``fp4_quantize`` themselves: both CuteDSL NVFP4 leaves, FC12 included, and not with a
    pre-quant activation scale."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl import TrtllmCutedslGroupedGemmNvfp4Impl

    probe = _k3_module("text_model")._quantizes_input_with_fp4_quantize
    for cls in (CuteDslFc12FusedMoE, TrtllmCutedslGroupedGemmNvfp4Impl):
        backend = cls.__new__(cls)
        nn.Module.__init__(backend)
        assert probe(backend), cls.__name__
        backend.fc31_act_scale = torch.ones(1)
        assert not probe(backend), cls.__name__
    assert not probe(SimpleNamespace(quantize_input=lambda x: x))


# ---------------------------------------------------------------------------
# Rubin: FC12 outputs on the tail's tables vs moe_sort
# ---------------------------------------------------------------------------


def _logits(m, seed):
    """Router-like FP32 logits (some popular experts, tokens of a request alike) and a bias."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = 256
    directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
    directions = directions / directions.norm(dim=1, keepdim=True)
    popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
    requests = torch.randn(((m + 7) // 8, dim), device="cuda", generator=gen)
    hidden = requests.repeat_interleave(8, dim=0)[:m] * 0.85
    hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
    hidden = hidden / hidden.norm(dim=1, keepdim=True)
    logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
    bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
    return logits.contiguous(), bias


class _RankFc12(CuteDslFc12FusedMoE):
    """``CuteDslFc12FusedMoE`` of one expert-parallel rank with random NVFP4 weights.

    Only the state ``run_moe_nvfp4`` reads is set up; routed rows are well conditioned (no
    FP4/FP8 saturation) so the comparison is on finite values.
    """

    has_nvfp4 = True

    @classmethod
    def build(cls, slot_start: int, num_local: int, seed: int = 0) -> "_RankFc12":
        backend = cls.__new__(cls)
        nn.Module.__init__(backend)
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def nibbles(*shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

        def block_scales(*shape):
            values = torch.rand(shape, device="cuda", generator=gen) * 1.5 + 0.5
            return values.to(torch.float8_e4m3fn).view(torch.uint8)

        backend.num_slots = NUM_EXPERTS
        backend.hidden_size = HIDDEN
        backend.use_fused_finalize = True
        backend.act_clamp = float("inf")
        backend.mapping = SimpleNamespace(moe_ep_size=NUM_EXPERTS // num_local)
        backend.activation_type = ActivationType.SiTu
        # SiTU soft-caps as the leaf holds them: gate_softcap, linear_softcap.
        backend.act_alpha = 4.0
        backend.act_beta = 25.0
        backend.fc2_input_scale = torch.tensor([4.0], device="cuda")
        backend.scaling_vector_size = 16
        backend.rank_weights = NvFp4WeightView(
            w3_w1_weight=nibbles(num_local, 2 * INTER, HIDDEN // 2),
            fc1_weight_scale=block_scales(num_local, 2 * INTER, HIDDEN // 16),
            fc1_global_scale=torch.full((num_local,), 2.5e-5, device="cuda"),
            w2_weight=nibbles(num_local, HIDDEN, INTER // 2),
            fc2_weight_scale=block_scales(num_local, HIDDEN, INTER // 16),
            fc2_global_scale=torch.full((num_local,), 2e-4, device="cuda"),
            expert_size_per_partition=num_local,
            slot_start=slot_start,
        )
        return backend

    def run(self, x, x_sf, ids, weights, tile_size=None):
        """``run_moe_nvfp4`` (``tile_size=None``) or ``run_moe_nvfp4_impl`` at ``tile_size``."""
        if tile_size is None:
            return self.run_moe_nvfp4(x, ids, weights, x_sf, weight_view=self.rank_weights)
        out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device="cuda")
        return self.run_moe_nvfp4_impl(
            x, ids, weights, x_sf, out, self.rank_weights, tile_size=tile_size
        )


def _quantized_input(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    global_scale = (448 * 6) / x.abs().max().float()
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, global_scale.reshape(1), 16, False, False)
    return x_fp4, x_sf.view(m, HIDDEN // 16)


def _tail_route(logits, bias, offset, num_local):
    _k3_module("kernels.k3_route_tail")  # registers trtllm::k3_route_tail
    outs = torch.ops.trtllm.k3_route_tail(logits, bias, SCALE, offset, num_local, TILE)
    ids = outs[0]
    ids._k3_presorted = ((offset, num_local, TILE), tuple(outs[2:8]))
    return ids, outs[1]


def _noaux_route(logits, bias):
    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)
    return ids.to(torch.int32), weights.to(torch.float32)


def _local_counts(ids, offset, num_local):
    return ((ids >= offset) & (ids < offset + num_local)).sum(dim=1)


def _assert_same_rows(actual, expected, local_counts):
    """Rows with <= 2 local experts bit-identical, the others up to the BF16 add order."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(expected.float()).all()
    if local_counts.sum() > 0:
        assert expected.abs().amax() > 0, "degenerate test: all-zero MoE output"
    exact = local_counts <= 2
    mismatch = (actual[exact].view(torch.int16) != expected[exact].view(torch.int16)).any(dim=1)
    assert not mismatch.any(), f"{int(mismatch.sum())} of {int(exact.sum())} exact rows differ"
    if (~exact).any():
        a, e = actual[~exact].float(), expected[~exact].float()
        # Each of the up to 16 BF16 adds rounds by at most half an ulp of the running sum.
        tol = (local_counts[~exact].float() * 2.0**-8).unsqueeze(1) * e.abs().amax(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        assert ((a - e).abs() <= tol).all(), f"max diff {(a - e).abs().max().item():.3e}"


@rubin_fc12
@pytest.mark.parametrize("offset,num_local", LAYOUTS)
@pytest.mark.parametrize("m", MS)
def test_fc12_tail_tables_match_moe_sort(monkeypatch, m, offset, num_local):
    """FC12 on the tail's tables equals FC12 after ``noaux_tc_op`` + ``moe_sort``."""
    import tensorrt_llm  # noqa: F401

    _k3_module("kernels.k3_route_tail")

    backend = _RankFc12.build(offset, num_local, seed=offset + num_local)
    logits, bias = _logits(m, seed=m)
    x, x_sf = _quantized_input(m, seed=m + 1)

    ref_ids, ref_weights = _noaux_route(logits, bias)
    expected = backend.run(x, x_sf, ref_ids, ref_weights, tile_size=TILE)
    ids, weights = _tail_route(logits, bias, offset, num_local)
    assert torch.equal(ids, ref_ids) and torch.equal(
        weights.view(torch.int32), ref_weights.view(torch.int32)
    )

    real_moe_sort = torch.ops.trtllm.moe_sort
    sorts = []

    def counting_moe_sort(**kwargs):
        sorts.append(kwargs["tile_tokens_dim"])
        return real_moe_sort(**kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", counting_moe_sort)
    actual = backend.run(x, x_sf, ids, weights)
    assert sorts == [], "FC12 ran moe_sort although the tail's tables were attached"
    _assert_same_rows(actual, expected, _local_counts(ref_ids, offset, num_local))

    # At another tile the attached tables do not apply: moe_sort runs instead.
    at_256 = backend.run(x, x_sf, ids, weights, tile_size=256)
    assert sorts == [256]
    expected_256 = backend.run(x, x_sf, ref_ids, ref_weights, tile_size=256)
    _assert_same_rows(at_256, expected_256, _local_counts(ref_ids, offset, num_local))


@rubin_fc12
def test_fc12_tail_tables_cuda_graph_replay():
    """Tail + FC12 captured in one CUDA graph follow new logits on replay."""
    import tensorrt_llm  # noqa: F401

    _k3_module("kernels.k3_route_tail")

    m, offset, num_local = 32, 3 * 112, 112
    backend = _RankFc12.build(offset, num_local, seed=5)
    logits, bias = _logits(m, seed=40)
    x, x_sf = _quantized_input(m, seed=41)
    static_logits = logits.clone()
    backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))
    for seed in range(3):
        fresh, _ = _logits(m, seed=50 + seed)
        static_logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_weights = _noaux_route(static_logits, bias)
        expected = backend.run(x, x_sf, ref_ids, ref_weights, tile_size=TILE)
        _assert_same_rows(out, expected, _local_counts(ref_ids, offset, num_local))


# ---------------------------------------------------------------------------
# Rubin: KimiK3MoERuntime routes a real CUTEDSL_FC12 ConfigurableMoE with the tail
# ---------------------------------------------------------------------------


def _fc12_moe(routing):
    """Single-rank ``CUTEDSL_FC12`` ConfigurableMoE (NVFP4, SiTU) with random experts."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN
    pretrained_config.intermediate_size = INTER
    pretrained_config.torch_dtype = torch.bfloat16
    pretrained_config.activation_situ_beta = 4.0
    pretrained_config.activation_situ_linear_beta = 25.0
    moe = create_moe(
        routing_method=routing,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=ModelConfig(
            pretrained_config=pretrained_config, mapping=Mapping(), moe_backend="CUTEDSL_FC12"
        ),
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        layer_idx=0,
        communication_method=None,
        activation=SiTuActivation(gate_softcap=4.0, linear_softcap=25.0),
    ).cuda()
    assert isinstance(moe, ConfigurableMoE)
    assert type(moe.backend) is CuteDslFc12FusedMoE

    gen = torch.Generator().manual_seed(17)
    backend = moe.backend
    quant_method = backend.quant_method
    quant_method.prepare_streaming_expert_load(backend)
    for expert_id in range(NUM_EXPERTS):
        tensors = {}
        for name, rows, cols in (
            ("w1", INTER, HIDDEN),
            ("w3", INTER, HIDDEN),
            ("w2", HIDDEN, INTER),
        ):
            tensors[f"{name}_weight"] = torch.randint(
                0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8
            )
            tensors[f"{name}_weight_scale"] = (
                torch.randint(120, 132, (rows, cols // 16), generator=gen, dtype=torch.int32)
                .to(torch.float32)
                .div(126.0)
                .to(torch.float8_e4m3fn)
            )
            # Gate/up/down outputs of order one, so SiTU and the FP4 requantization of
            # its output do not flush to zero.
            tensors[f"{name}_weight_scale_2"] = torch.tensor(0.03, dtype=torch.float32)
            tensors[f"{name}_input_scale"] = torch.tensor(1.0, dtype=torch.float32)
        quant_method.load_streaming_nvfp4_expert(
            backend, global_expert_id=expert_id, local_slot_id=expert_id, **tensors
        )
    backend.process_weights_after_loading()
    backend._weights_transformed = False
    moe.post_load_weights()
    return moe


@rubin_fc12
@pytest.mark.parametrize("m", [8, 64])
def test_moe_runtime_tail_feeds_fc12(monkeypatch, m):
    """``TRTLLM_K3_ROUTE_TAIL=1``: ``KimiK3MoERuntime`` + ``CUTEDSL_FC12`` skip ``moe_sort``,
    run FC12 at tile 128 and match the regular routing."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    modeling_kimi_linear = _k3_module("text_model")
    _KimiK3RoutingMethod = modeling_kimi_linear._KimiK3RoutingMethod

    logits, bias = _logits(m, seed=60 + m)
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=SCALE,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    moe = _fc12_moe(routing)

    class _Identity(nn.Module):
        def forward(self, value, **kwargs):
            return value

    class _Zeros(nn.Module):
        def forward(self, value, **kwargs):
            return torch.zeros_like(value)

    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=lambda value: logits,
        e_score_correction_bias=bias,
        routed_scaling_factor=SCALE,
        num_experts=NUM_EXPERTS,
    )
    runtime.routed_experts = moe
    runtime.routed_expert_down_proj = _Identity()
    runtime.routed_expert_norm = _Identity()
    runtime.routed_expert_up_proj = _Identity()
    runtime.shared_experts = _Zeros()
    # The runtime reads the shared expert's down projection; a plain module here.
    runtime.shared_experts.down_proj = nn.Identity()
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime.shared_expert_stream = None
    runtime.latent_down_stream = None
    runtime.moe_main_event = runtime.moe_shared_event = runtime.moe_routed_event = None
    runtime.moe_latent_fork_event = runtime.moe_latent_ready_event = None
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value)

    real_moe_sort = torch.ops.trtllm.moe_sort
    sorts, tiles = [], []

    def counting_moe_sort(**kwargs):
        sorts.append(kwargs["tile_tokens_dim"])
        return real_moe_sort(**kwargs)

    real_impl = moe.backend.run_moe_nvfp4_impl

    def recording_impl(*args, **kwargs):
        tiles.append(kwargs.get("tile_size"))
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", counting_moe_sort)
    monkeypatch.setattr(moe.backend, "run_moe_nvfp4_impl", recording_impl)

    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    outputs = {}
    for mode in ("0", "1"):
        monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", mode)
        sorts.clear()
        tiles.clear()
        with torch.inference_mode(), with_multi_stream(False):
            outputs[mode] = runtime(hidden)
        torch.cuda.synchronize()
        assert routing.deferred is None
        assert tiles == [TILE]
        assert sorts == ([TILE] if mode == "0" else []), f"mode {mode}: moe_sort calls {sorts}"

    # Every token has its 16 experts on this single rank.
    ids, _ = _noaux_route(logits, bias)
    _assert_same_rows(outputs["1"], outputs["0"], _local_counts(ids, 0, NUM_EXPERTS))


@rubin_fc12
@pytest.mark.parametrize("order", ["router", "fork"])
@pytest.mark.parametrize("m", [8, 64])
def test_moe_runtime_tail_order_fc12(monkeypatch, m, order):
    """``TRTLLM_K3_ROUTE_TAIL_ORDER``: with the latent and shared-expert streams, FC12 on the tail's
    tables waits for the shared expert's GEMMs and takes the routed input the latent chain
    quantized; the output equals the previous order's."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    modeling_kimi_linear = _k3_module("text_model")
    _KimiK3RoutingMethod = modeling_kimi_linear._KimiK3RoutingMethod

    logits, bias = _logits(m, seed=80 + m)
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=SCALE,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    moe = _fc12_moe(routing)

    class _Identity(nn.Module):
        def forward(self, value, **kwargs):
            return value

    class _AllReduce(nn.Module):
        def forward(self, value, all_reduce_params=None):
            if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
                return value
            return value + 1.0

    class _SharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Module()
            self.down_proj.reduce_output = True
            self.down_proj.use_fused_gemm_allreduce = False
            self.down_proj.all_reduce = _AllReduce()

        def forward(self, value, final_all_reduce_params=None):
            return self.down_proj.all_reduce(value * 0.5, all_reduce_params=final_all_reduce_params)

    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=lambda value: logits,
        e_score_correction_bias=bias,
        routed_scaling_factor=SCALE,
        num_experts=NUM_EXPERTS,
    )
    runtime.top_k = TOP_K
    runtime.num_experts = NUM_EXPERTS
    runtime.routed_experts = moe
    runtime.routed_expert_down_proj = _Identity()
    runtime.routed_expert_norm = _Identity()
    runtime.routed_expert_up_proj = _Identity()
    runtime.shared_experts = _SharedExperts()
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = torch.cuda.Stream()
    for name in (
        "moe_main_event",
        "moe_shared_event",
        "moe_latent_fork_event",
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime.moe_routed_event = None
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value * 1.0)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_ORDER_MIN_ROWS", 1)

    real_moe_sort = torch.ops.trtllm.moe_sort
    sorts, backend_quant = [], []

    def counting_moe_sort(**kwargs):
        sorts.append(kwargs["tile_tokens_dim"])
        return real_moe_sort(**kwargs)

    real_quantize = moe.backend.quantize_input

    def counting_quantize(x, *args, **kwargs):
        # The backend quantizes a BF16 input; a quantized one it only unpacks.
        if not isinstance(x, Fp4QuantizedTensor):
            backend_quant.append(1)
        return real_quantize(x, *args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", counting_moe_sort)
    monkeypatch.setattr(moe.backend, "quantize_input", counting_quantize)

    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    outputs = {}
    for key, mode in (("0", "0"), ("0 again", "0"), (order, order)):
        monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_ORDER", mode)
        backend_quant.clear()
        with torch.inference_mode(), with_multi_stream(True):
            outputs[key] = runtime(hidden)
        torch.cuda.synchronize()
        assert sorts == [] and routing.deferred is None
        # The previous order quantizes in the backend, the new one on the latent chain.
        assert backend_quant == ([1] if mode == "0" else [])

    # Same kernels on the same inputs: bit-identical wherever FC12 itself reproduces its output
    # (its scatter-adds of tokens with several local experts may land in a different order).
    ids, _ = _noaux_route(logits, bias)
    if torch.equal(outputs["0 again"], outputs["0"]):
        assert torch.equal(outputs[order], outputs["0"])
    else:
        _assert_same_rows(outputs[order], outputs["0"], _local_counts(ids, 0, NUM_EXPERTS))
