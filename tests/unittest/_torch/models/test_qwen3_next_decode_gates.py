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
"""Where the Qwen3.5 prefill / decode kernels engage, decided on the host (no GPU): the architecture and layout
gates, the resident band, the shared expert's low-M path and its tuning, the expert prefetch's layers, the row
gates under torch.compile, the drafter's view of the target's declarations and the fast paths' runtime hooks.
Each reference layout (GB300 NVL72: CTX TEP2 / DEP4, GEN TEP8 / TEP16 / DEP4, MTP 6) keeps its decisions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_qwen3_next as m
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextLinearDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MOE_DECODE_OPTIONS, MoEDecodeOptions
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _StopInit(Exception):
    pass


def _stop(model_config) -> None:
    raise _StopInit


def _config(mapping: Mapping, extra_attrs=None, spec_rows=None, batch_sizes=(1, 2)):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type="qwen3_5_moe_text", torch_dtype=torch.bfloat16
        ),
        extra_attrs={} if extra_attrs is None else extra_attrs,
        spec_config=None if spec_rows is None else SimpleNamespace(tokens_per_gen_step=spec_rows),
        cuda_graph_batch_sizes=list(batch_sizes),
        mapping=mapping,
    )


# Qwen3.5's declarations on the shared class (the Qwen3.5 classes' own __init__ only normalizes the quant config).
_Q35 = type("_Q35", (Qwen3NextForCausalLM,), {"decode_kernels": True, "prefill_kernels": True})


def _build(model_cls: type, config) -> None:
    """Run the model class's __init__ up to the decoder model (what it declares in ``extra_attrs``)."""
    with pytest.raises(_StopInit):
        model_cls(config)


# (name, mapping, aarch64): GB300 / GB200 NVL racks are aarch64 with an NVLink fabric, HGX nodes x86.
_GB300_TEP8 = Mapping(world_size=8, tp_size=8, moe_ep_size=8, gpus_per_node=4)
_GB300_TEP16 = Mapping(world_size=16, tp_size=16, moe_ep_size=16, gpus_per_node=4)
_GB300_CTX_TEP2 = Mapping(world_size=2, tp_size=2, moe_ep_size=2, gpus_per_node=4)
_DEP4 = Mapping(world_size=4, tp_size=4, moe_ep_size=4, gpus_per_node=4, enable_attention_dp=True)
_HGX_TEP8 = Mapping(world_size=8, tp_size=8, moe_ep_size=8, gpus_per_node=8)
_HGX_TP2 = Mapping(world_size=2, tp_size=2, gpus_per_node=8)
_TP1 = Mapping(world_size=1, tp_size=1)


@pytest.fixture
def blackwell(monkeypatch) -> None:
    monkeypatch.setattr(m, "is_blackwell_datacenter", lambda: True)
    monkeypatch.setattr(m, "Qwen3NextModel", _stop)


def _fabric(monkeypatch, aarch64: bool) -> None:
    """The MNNVL hardware check of MNNVLAllReduce.is_mnnvl: an NVLink fabric on an aarch64 (Grace) node."""
    from tensorrt_llm._torch.distributed import mnnvl_memory, ops

    monkeypatch.delenv("TLLM_TEST_MNNVL", raising=False)
    monkeypatch.setattr(mnnvl_memory.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: True))
    monkeypatch.setattr(ops.platform, "machine", lambda: "aarch64" if aarch64 else "x86_64")


@pytest.mark.parametrize(
    "mapping,aarch64,expected",
    [
        (_GB300_TEP8, True, True),  # GEN TEP8 (reference)
        (_GB300_TEP16, True, True),  # GEN TEP16 (reference)
        (_GB300_CTX_TEP2, True, True),  # CTX TEP2 (reference; its TWOSHOT strategy does not matter)
        (_DEP4, True, True),  # GEN / CTX DEP4 (reference)
        (_DEP4, False, True),  # attention DP on an HGX node
        (_HGX_TEP8, False, False),  # TEP8 on 8x B200 HGX: no MNNVL
        (_HGX_TP2, False, False),  # Qwen3.6 TP2 on B200 HGX
        (_TP1, False, False),  # Qwen3.6 TP1 on B200 HGX
        (_TP1, True, True),  # TP1 on a GB300 node (as before)
    ],
    ids=[
        "gb300_tep8",
        "gb300_tep16",
        "gb300_ctx_tep2",
        "gb300_dep4",
        "hgx_dep4",
        "hgx_tep8",
        "hgx_tp2",
        "hgx_tp1",
        "gb300_tp1",
    ],
)
def test_decode_kernels_follow_the_layout(
    monkeypatch, blackwell, mapping, aarch64, expected
) -> None:
    """The declared decode kernels (and the fused-norm all-reduces, the MoE decode options, the fast paths) engage
    under attention DP or in a TP layout with MNNVL, where they were tuned; a TP layout without one (an HGX node)
    keeps the stock paths. Every GB300 reference layout keeps them."""
    _fabric(monkeypatch, aarch64)
    config = _config(mapping)
    _build(_Q35, config)
    assert m._uses_decode_kernels(config) is expected
    assert _Q35.supports_spec_decode_fast_paths(config) is expected
    assert MoEDecodeOptions.of(config.extra_attrs).tuning is expected
    assert config.extra_attrs.get("allreduce_mnnvl_fused_norm", False) is expected
    assert config.extra_attrs.get("allreduce_tp2_fused_norm", False) is expected
    assert m._uses_tp_decode_kernels(config) is (expected and not mapping.enable_attention_dp)


@pytest.mark.parametrize("sm", [90, 100, 103, 107, 120])
@pytest.mark.parametrize("declared", [True, False])
def test_prefill_kernels_need_blackwell_datacenter(monkeypatch, sm, declared) -> None:
    """The declared prefill kernels (fused GDN prefill conv and splits, MoE input dual quant, FP8 output gate,
    staged-KV FP8 context attention) run on SM 10.0 / 10.3 only, as the decode kernels."""
    import tensorrt_llm._utils as utils

    monkeypatch.setattr(utils, "get_sm_version", lambda: sm)
    monkeypatch.setattr(m, "Qwen3NextModel", _stop)
    monkeypatch.setattr(m, "_decode_kernels_layout", lambda model_config: True)
    config = _config(_GB300_CTX_TEP2)
    assert Qwen3_5MoeForCausalLM.prefill_kernels and not Qwen3_5ForCausalLM.prefill_kernels
    _build(type("_Declared", (Qwen3NextForCausalLM,), {"prefill_kernels": declared}), config)
    assert m._uses_prefill_kernels(config) is (declared and sm in (100, 103))


def _layer_with_projection(n: int, k: int) -> Qwen3NextLinearDecoderLayer:
    layer = Qwen3NextLinearDecoderLayer.__new__(Qwen3NextLinearDecoderLayer)
    nn.Module.__init__(layer)
    projection = SimpleNamespace(
        weight=torch.empty((n, k), dtype=torch.float8_e4m3fn, device="meta"),
        _weights_removed=False,
        resident_fp8_min_rows=0,
    )
    layer.linear_attn = SimpleNamespace(in_proj_qkvz=projection)
    return layer


def _feeding_layer(tp_size: int, post_moe_fusion: bool, scale) -> SimpleNamespace:
    return SimpleNamespace(
        next_layer_input_scale=scale,
        fusion_config=SimpleNamespace(POST_MOE_FUSION=post_moe_fusion),
        mapping=SimpleNamespace(tp_size=tp_size),
        allreduce=SimpleNamespace(mnnvl_allreduce=None),
        next_layer_resident_rows=(1, 0),
    )


@pytest.mark.parametrize(
    "tp_size,post_moe_fusion,scale,n,k,band",
    [
        # TEP8: GDN in_proj_qkvz (20480 / 8) and attention qkv_proj
        (8, True, True, 2560, 4096, True),
        (16, True, True, 1280, 4096, True),  # TEP16: GDN in_proj_qkvz
        (16, True, True, 1536, 4096, True),  # TEP16: attention qkv_proj (q 1024 + replicated k / v)
        (1, True, True, 2560, 4096, False),  # TP1: no all-reduce hands over E4M3
        # no POST_MOE_FUSION: the next layer quantizes its own input
        (8, False, True, 2560, 4096, False),
        (8, True, False, 2560, 4096, False),  # BF16 next projection
        (8, True, True, 2624, 4096, False),  # N not in whole 128-feature CTA pairs
        (8, True, True, 2560, 4608, False),  # K not in whole x TMA boxes
    ],
)
def test_resident_band_needs_an_e4m3_handover_and_a_covered_shape(
    tp_size, post_moe_fusion, scale, n, k, band
) -> None:
    """The input projection after a MoE-output all-reduce takes the resident-weight GEMM from its resident rows only
    where that all-reduce hands it an E4M3 input and resident_fp8_gemm covers (K, N) at every row of the band; the
    reference TEP8 / TEP16 shapes keep the band."""
    from tensorrt_llm._torch.kernels.blackwell.resident_fp8_gemm import MAX_ROWS

    layer = _feeding_layer(tp_size, post_moe_fusion, torch.ones(1) if scale else None)
    next_layer = _layer_with_projection(n, k)
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            extra_attrs={"qwen3_next_decode_kernels": True},
            mapping=SimpleNamespace(enable_attention_dp=False),
            spec_config=SimpleNamespace(tokens_per_gen_step=7),
        )
    )
    Qwen3NextModel._cache_tp_decode_gates(model, [layer, next_layer])
    projection = next_layer.linear_attn.in_proj_qkvz
    assert projection.resident_fp8_min_rows == (14 if band else 0)
    assert layer.next_layer_resident_rows == ((14, MAX_ROWS) if band else (1, 0))


@pytest.mark.parametrize("attention_dp,rows", [(False, 14), (True, 0)])
def test_resident_rows_come_from_one_helper(monkeypatch, attention_dp, rows) -> None:
    """The GDN layer's resident rows and the model's resident band derive from _resident_min_rows."""
    config = SimpleNamespace(
        extra_attrs={"qwen3_next_decode_kernels": True},
        mapping=SimpleNamespace(enable_attention_dp=attention_dp),
        spec_config=SimpleNamespace(tokens_per_gen_step=7),
        cuda_graph_batch_sizes=[1, 2, 4, 8],
    )
    assert m._resident_min_rows(config) == rows
    built = {}

    def record(self, model_config, aux_stream, layer_idx, **kwargs) -> None:
        nn.Module.__init__(self)
        built.update(kwargs)
        raise _StopInit

    monkeypatch.setattr(m.Qwen3NextGatedDeltaNet, "__init__", record)
    with pytest.raises(_StopInit):
        Qwen3NextLinearDecoderLayer(config, 0, None)
    assert built["resident_min_rows"] == rows
    assert built["resident_ba_min_rows"] == (rows + 1 if rows else 0)


class _Dispatcher:
    """A low-M dispatcher that records its FP8 race calls."""

    def __init__(self) -> None:
        self.calls = []

    def takes_rows(self, module, rows: int) -> bool:
        return True

    def apply_fp8(self, x, weight, weight_scale, input_scale, out_dtype):
        self.calls.append((tuple(x.shape), x.dtype, weight))


def _shared_expert_block(raced_rows: int, hidden: int = 256, inter: int = 64):
    dispatcher = _Dispatcher()
    scale = torch.ones(())

    def projection(n: int, k: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=torch.zeros((n, k), dtype=torch.float8_e4m3fn),
            weight_scale=scale,
            quant_method=SimpleNamespace(get_static_input_scale=lambda proj: scale),
            _low_m_gemm_dispatcher=dispatcher,
        )

    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block._shared_expert_low_m = (torch.zeros(1), torch.zeros(1))
    block._shared_expert_raced_rows = raced_rows
    block.shared_expert = SimpleNamespace(
        gate_up_proj=projection(2 * inter, hidden), down_proj=projection(hidden, inter)
    )
    return block, dispatcher


@pytest.mark.parametrize(
    "raced_rows,tunes",
    [
        # MTP 3, CUDA graphs up to 4 requests: eager 17..32-row steps take fused_shared_expert
        (16, True),
        # TEP16 b1 (reference): its steps never pass 7 rows, so nothing reads these races
        (7, True),
        (32, False),  # the raced path covers every low-M row count
        (56, False),  # TEP2 / TEP8 with 8+ requests at MTP 6 (reference)
        (0, False),  # attention DP (reference DEP4): the stock calls tune the races
    ],
)
def test_shared_expert_tuning_also_tunes_the_fused_path(monkeypatch, raced_rows, tunes) -> None:
    """In tuning mode the raced path takes every call, so the low-M races that fused_shared_expert reads for the
    rows above the raced ones (gate_up_proj with a BF16 input, down_proj with an E4M3 input) are timed beside it."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    block, dispatcher = _shared_expert_block(raced_rows)
    monkeypatch.setattr(m, "raced_shared_expert", lambda x, weights, **kwargs: "raced")
    monkeypatch.setattr(m, "fused_shared_expert", lambda dispatcher, x, weights, **kwargs: "fused")
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
    x = torch.zeros((64, 256), dtype=torch.bfloat16)
    out = block._shared_expert_low_m_forward(x)
    assert out == ("raced" if raced_rows else "fused")
    mlp = block.shared_expert
    if tunes:
        assert dispatcher.calls == [
            ((32, 256), torch.bfloat16, mlp.gate_up_proj.weight),
            ((32, 64), torch.float8_e4m3fn, mlp.down_proj.weight),
        ]
    else:
        assert dispatcher.calls == []
    # Outside tuning nothing extra runs.
    dispatcher.calls.clear()
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", False)
    block._shared_expert_low_m_forward(x[: max(raced_rows, 1)])
    assert dispatcher.calls == []


@pytest.mark.parametrize(
    "gate_up,down,qualifies",
    [
        ((256, 4096), (4096, 128), True),  # TEP8 shared expert (I 1024 / 8)
        ((128, 4096), (4096, 64), True),  # TEP16
        ((2048, 4096), (4096, 1024), True),  # attention DP (whole shared expert)
        ((256, 1088), (1088, 128), False),  # K of 17 64-wide chunks: no default gate_up tactic
    ],
)
def test_shared_expert_low_m_declines_shapes_without_a_default_tactic(
    monkeypatch, gate_up, down, qualifies
) -> None:
    """An untuned launch of the shared expert's low-M path runs low_m_mma_gemm.default_tactic, which raises for
    some shapes: the path declines them when its weights are built."""
    assert (
        m._shared_expert_default_tactics_exist(torch.empty(gate_up), torch.empty(down)) is qualifies
    )

    class _Fp8:
        def get_static_input_scale(self, proj):
            return torch.ones(())

    def projection(shape):
        # BF16 stands in for the E4M3 weights: only their shapes decide (CPU tensor ops).
        return SimpleNamespace(
            weight=torch.zeros(shape, dtype=torch.bfloat16), bias=None, quant_method=_Fp8()
        )

    monkeypatch.setattr(m, "FP8QDQLinearMethod", _Fp8)
    monkeypatch.setattr(m, "warp_mma_supported", lambda: True)
    monkeypatch.setattr(m, "low_m_gemm_dispatcher", lambda module: object())
    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block._shared_expert_low_m = None
    block.shared_expert = SimpleNamespace(
        gate_up_proj=projection(gate_up),
        down_proj=projection(down),
        activation=torch.nn.functional.silu,
        _is_plain_swiglu=lambda: True,
        swiglu_limit=None,
    )
    block.shared_expert_gate = SimpleNamespace(
        weight=torch.zeros((1, gate_up[1]), dtype=torch.bfloat16), bias=None
    )
    weights = block._shared_expert_low_m_weights()
    assert (weights is not None) is qualifies


def _moe_layer(idx: int, post_moe_fusion: bool, defers: bool, fold_rows) -> SimpleNamespace:
    mlp = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(mlp)
    mlp._experts_defer_finalize = lambda: defers
    return SimpleNamespace(
        layer_idx=idx,
        mlp=mlp,
        hidden_size=4096,
        fusion_config=SimpleNamespace(POST_MOE_FUSION=post_moe_fusion),
        allreduce=SimpleNamespace(supports_moe_fold=lambda rows, hidden: rows in fold_rows),
    )


def test_expert_prefetch_keeps_only_layers_whose_front_records() -> None:
    """The decode front records a layer's experts only in an un-finalized MoE call (the MoE-output all-reduce folds
    and finalizes it): a layer that never takes one is unregistered, so it launches no prefetch on an empty or stale
    selection. The reference TEP layouts (MNNVL fold at every prefetched row count) keep every layer."""
    from tensorrt_llm._torch.modules.expert_l2_prefetch import ExpertL2Prefetch

    prefetch = ExpertL2Prefetch(6, min_rows=14, max_rows=63, top_k=10, stream=None)
    every_row = range(1, 129)
    layers = [
        _moe_layer(0, True, True, every_row),  # reference TEP8 / TEP16 layer
        _moe_layer(1, True, True, range(64, 129)),  # folds only above the prefetched rows
        _moe_layer(2, True, True, ()),  # no MNNVL all-reduce (e.g. AUTO TEP4 within one tray)
        _moe_layer(3, True, False, every_row),  # experts that cannot defer their finalize
        _moe_layer(4, False, True, every_row),  # no POST_MOE_FUSION
        _moe_layer(5, True, True, range(30, 31)),  # folds at one prefetched row count
    ]
    for layer in layers:
        prefetch.register(layer.layer_idx, object(), 0)
    Qwen3NextModel._drop_expert_prefetch_without_record(
        SimpleNamespace(expert_prefetch=prefetch), layers
    )
    assert [prefetch.registered(i) for i in range(6)] == [True, False, False, False, False, True]
    # An unregistered layer launches nothing (the backend lookup returns first, before any device work).
    prefetch.prefetch(2, 20)
    assert not prefetch._forked


def _compile_count(fn, rows) -> int:
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    compiled = torch.compile(fn, backend=counter, dynamic=True)
    for n in rows:
        compiled(torch.zeros((n, 8)))
    return counter.frame_count


def test_expert_prefetch_rows_do_not_guard_under_torch_compile() -> None:
    """Under torch.compile the prefetch row check tests is_compiling() first, so a symbolic row count crossing its
    bounds compiles once."""
    from tensorrt_llm._torch.modules.expert_l2_prefetch import ExpertL2Prefetch

    torch._dynamo.reset()
    prefetch = ExpertL2Prefetch(4, min_rows=14, max_rows=63, top_k=10, stream=None)

    def fn(x):
        return x * 2 if prefetch.takes_rows(x.shape[0]) else x + 1

    assert _compile_count(fn, (2, 3, 20, 40, 100, 300)) == 1


def test_moe_input_dual_quant_rows_do_not_guard_under_torch_compile() -> None:
    """The MoE input dual quant tests is_compiling() before its 1024-row bound: a symbolic row count crossing it
    compiles once."""
    torch._dynamo.reset()
    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block._dual_quant_input = True

    def fn(x):
        return x * 2 if block._moe_input_dual_quant(x, None) is None else x + 1

    assert _compile_count(fn, (2, 3, 900, 1100, 2000)) == 1


def _spec_config(kind: str) -> SimpleNamespace:
    mode = SimpleNamespace(
        use_one_engine=lambda: True,
        is_sa=lambda: False,
        is_eagle3_one_model=lambda: False,
        is_external_drafter=lambda: kind == "pard",
        is_parallel_draft=lambda: False,
    )
    return SimpleNamespace(
        spec_dec_mode=mode,
        uses_external_draft_model=kind == "external_mtp",
        uses_replacement_heads=False,
        speculative_model="drafter",
        tokens_per_gen_step=7,
    )


@pytest.mark.parametrize("kind", ["external_mtp", "pard", "own_mtp"])
def test_drafters_of_another_checkpoint_do_not_take_the_target_decode_options(
    monkeypatch, kind
) -> None:
    """The target's MoE decode options and fused-norm all-reduce declarations stay out of a drafter built from
    another checkpoint (external MTP, PARD / draft-target), whose config shares the target's extra_attrs; Qwen3.5's
    own MTP layer (the reference) keeps them, and the target's dict is unchanged afterwards."""
    from tensorrt_llm._torch.models import modeling_speculative as ms
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

    def base_init(self, model, *, config, hidden_size, vocab_size) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.lm_head = None
        self.epilogue = []

    seen = {}

    def get_draft_model(model_config, draft_config, lm_head, model):
        attrs = model_config.extra_attrs
        seen["options"] = MoEDecodeOptions.of(attrs)
        seen["fused_norm"] = attrs.get("allreduce_mnnvl_fused_norm", False)
        seen["layers"] = "attn_layers" in attrs
        seen["shared"] = draft_config is None or draft_config.extra_attrs is attrs
        return None

    monkeypatch.setattr(DecoderModelForCausalLM, "__init__", base_init)
    monkeypatch.setattr(ms, "_get_requested_draft_moe_backend", lambda *args: "TRTLLM")
    monkeypatch.setattr(ms, "_set_draft_kv_cache_quant_algo", lambda *args: None)
    monkeypatch.setattr(ms, "external_drafter_config_kwargs", lambda *args: {})
    monkeypatch.setattr(ms, "should_use_separate_draft_kv_cache", lambda *args: False)
    monkeypatch.setattr(ms, "get_draft_model", get_draft_model)
    monkeypatch.setattr(ms, "get_spec_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ms.ModelConfig, "from_pretrained", classmethod(lambda cls, *a, **k: SimpleNamespace())
    )

    options = MoEDecodeOptions(tuning=True, fc2_weights_evict_first_rows=1344)
    extra_attrs = {
        MOE_DECODE_OPTIONS: options,
        "allreduce_mnnvl_fused_norm": True,
        "allreduce_tp2_fused_norm": True,
        "attn_layers": {},
    }
    config = SimpleNamespace(
        spec_config=_spec_config(kind),
        extra_attrs=extra_attrs,
        attn_backend="TRTLLM",
        mapping=SimpleNamespace(mapping_for_full_rows=lambda: None),
        max_num_tokens=8,
        moe_max_num_tokens=8,
        pretrained_config=SimpleNamespace(hidden_size=8, vocab_size=8),
    )
    target = Qwen3_5MoeForCausalLM.__new__(Qwen3_5MoeForCausalLM)
    ms.SpecDecOneEngineForCausalLM.__init__(target, None, config)

    own = kind == "own_mtp"
    assert seen["shared"] and seen["layers"]
    assert seen["options"] == (options if own else MoEDecodeOptions())
    assert seen["fused_norm"] is own
    assert extra_attrs[MOE_DECODE_OPTIONS] is options
    assert extra_attrs["allreduce_mnnvl_fused_norm"] and extra_attrs["allreduce_tp2_fused_norm"]


def test_fast_path_runtime_hooks_are_declared_by_the_model() -> None:
    """Qwen3Next declares its fast paths' step-loop hooks; another fast-path model gets none from the base class."""
    from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    expected = RuntimeHooks(
        loop_on_execution_stream=True,
        pack_spec_rng_uploads=True,
        iteration_timing_one_event=True,
        perf_metrics_timing_side_stream=True,
    )
    for cls in (Qwen3NextForCausalLM, Qwen3_5MoeForCausalLM, Qwen3_5ForCausalLM):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.spec_decode_fast_paths = True
        assert model.runtime_hooks == expected
        model.spec_decode_fast_paths = False
        assert model.runtime_hooks is None
    other_cls = type("_OtherFastPathModel", (SpecDecOneEngineForCausalLM,), {})
    other = other_cls.__new__(other_cls)
    nn.Module.__init__(other)
    other.spec_decode_fast_paths = True
    assert other.runtime_hooks is None
