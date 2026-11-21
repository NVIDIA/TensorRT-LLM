# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile real prefill host dispatch, substituting CPU math for GPU kernels."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling as tp2,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (
    modeling as tp4,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8 import (
    modeling as tp8,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.mxfp8_state import (
    _mx_mode,
    _Mxfp8GemmState,
)
from tensorrt_llm._torch.modules.linear import (
    flashinfer_mxfp8_autotune,
    flashinfer_mxfp8_decode_graph_capture,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import enable_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.engine import model_setup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import PrefillCompiledModel
from tensorrt_llm._torch.utils import model_extra_attrs, torch_compiling
from tensorrt_llm.llmapi import PrefillCudaGraphBackend, TorchCompileConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only
_TARGETS = {
    tp2: (
        tp2.MiniMaxM3Tp2ModelingV2,
        tp2.ModelingV2MiniMaxM3Nvfp4Sm107Tp2,
        2,
        "SPARSE_DECODE_LC_PREFETCH_SELF_KV",
    ),
    tp4: (
        tp4.MiniMaxM3Tp4ModelingV2,
        tp4.ModelingV2MiniMaxM3Nvfp4Sm107Tp4,
        4,
        "SPARSE_DECODE_LC_PREFETCH_NEXT_KV",
    ),
    tp8: (
        tp8.MiniMaxM3Tp8ModelingV2,
        tp8.ModelingV2MiniMaxM3Nvfp4Sm107Tp8,
        8,
        "SPARSE_DECODE_LC8_PREFETCH_NEXT_KV",
    ),
}


@pytest.fixture
def make_cpu_model(monkeypatch, make_m3_config):
    def make_model(modeling, *, fused_projection=False, shell=False):
        core_type, shell_type, tp, prefetch_flag = _TARGETS[modeling]
        config = make_m3_config(tp, head_dim=16, fused_projection=fused_projection)
        # The constructors allocate CUDA resources that the host dispatch tests never use.
        with monkeypatch.context() as setup:
            setup.setattr(modeling, prefetch_flag, False)
            setup.setattr(torch.cuda, "Stream", lambda: None)
            setup.setattr(torch.cuda, "Event", lambda: None)
            setup.setattr(
                RopeParams,
                "create_rope_const_params",
                lambda self, **kwargs: (None, torch.empty(self.max_positions, self.dim)),
            )
            setup.setattr(
                modeling,
                "create_attention",
                lambda *args, **kwargs: Mock(spec=modeling.MiniMaxM3MsaSparseAttention),
            )
            if shell:
                setup.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 7))
                setup.setattr(modeling, "_require", [False])
                return shell_type(config)
            return core_type(config)

    return make_model


@dataclass
class _AttentionStep:
    num_tokens: int
    index_cache: torch.Tensor
    msa_out_cache_loc: torch.Tensor
    kv_cache_manager: object = None
    num_contexts: int = 1
    num_ctx_tokens: int = 0
    padded_num_tokens: int | None = None

    def msa_idx_k_cache(self, layer: int) -> torch.Tensor:
        return self.index_cache


@torch.library.custom_op("trtllm_test::m3_write_index_cache", mutates_args=("cache",))
def _write_index_cache(
    idx_qk: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    cache.index_copy_(0, slots, idx_qk)
    return idx_qk.clone()


@_write_index_cache.register_fake
def _write_index_cache_fake(
    idx_qk: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(idx_qk)


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
@pytest.mark.parametrize("attention_kind", ["dense", "fused_sparse", "unfused_sparse"])
def test_prefill_fullgraph_keeps_token_rows_dynamic(
    monkeypatch: pytest.MonkeyPatch, make_cpu_model, modeling: ModuleType, attention_kind: str
) -> None:
    fused = attention_kind == "fused_sparse"
    sparse = attention_kind != "dense"
    layer = int(sparse)
    core = make_cpu_model(modeling, fused_projection=fused)
    sentinel = -123
    metadata = _AttentionStep(
        num_tokens=0,
        index_cache=torch.full((105, 16), sentinel, dtype=torch.bfloat16),
        msa_out_cache_loc=torch.arange(105),
    )

    # Use identity projections and observe the arguments at the GPU kernel boundary.
    identity = torch.eye(16, dtype=torch.bfloat16)
    core.w[f"l{layer}_qkv_w"] = torch.nn.Parameter(identity.repeat(3, 1), requires_grad=False)
    core.w[f"l{layer}_idx_w"] = torch.nn.Parameter(identity, requires_grad=False)
    core.w[f"l{layer}_o_w"] = torch.nn.Parameter(identity, requires_grad=False)

    def attention(backend, q, k, v, idx_q, step, out, is_sparse, inv_scale, scale):
        assert is_sparse is sparse
        assert (k is None) is fused
        assert (v is None) is fused
        assert (idx_q is not None) is sparse
        assert q.shape[0] == step.num_tokens
        if sparse and not fused:
            torch.testing.assert_close(metadata.index_cache[: step.num_tokens], idx_q)
            assert (metadata.index_cache[step.num_tokens :] == sentinel).all()
        out.copy_(q + 1)

    def fused_producer(core, layer, packed, positions, *cache_args):
        assert packed.shape[0] == positions.shape[-1] == metadata.num_tokens
        return packed[:, :16], packed[:, :16]

    def write_index(core, layer, projected, positions, cache_for_layer, slots):
        assert positions.shape[-1] == projected.shape[0]
        return _write_index_cache(projected, cache_for_layer(layer), slots[: projected.shape[0]])

    monkeypatch.setattr(
        modeling, "_mxfp8_linear", lambda core, hidden, weight, scale, mode: hidden @ weight.t()
    )
    monkeypatch.setattr(modeling, "m3_msa_attention", attention)
    monkeypatch.setattr(modeling, "_qk_norm_rope", lambda core, layer, qkv, *args: qkv)
    if modeling in (tp2, tp8) and not sparse:
        monkeypatch.setattr(modeling, "_dense_qk_norm_rope_fp8", lambda core, layer, qkv, pos: qkv)
    if fused:
        monkeypatch.setattr(modeling, "_sparse_qkv_producer", fused_producer)
    elif sparse:
        monkeypatch.setattr(modeling, "m3_bf16_linear", lambda x, w: x @ w.t())
        monkeypatch.setattr(modeling, "_fp8_index_qk_norm_rope", write_index)
    target = modeling.PrefillTarget(core)

    def forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs_embeds.shape[0])[None]
        step = {
            "metadata": metadata,
            "kv_cache_manager": metadata.kv_cache_manager,
            "msa_idx_k_cache": metadata.msa_idx_k_cache,
            "msa_out_cache_loc": metadata.msa_out_cache_loc,
        }
        output, _ = target._attention(layer, positions, inputs_embeds, step)
        return output

    # Use the production graph split with CPU execution of each segment.
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    monkeypatch.setattr(Backend, "build_custom_passes", lambda *args: [])
    compiler = Backend(enable_inductor=False, enable_piecewise_cuda_graph=True)
    counter = CompileCounter()

    def compile_piecewise(gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Callable:
        counter(gm, inputs)
        return compiler(gm, inputs)

    torch._dynamo.reset()
    compiled = torch.compile(forward, fullgraph=True, dynamic=True, backend=compile_piecewise)
    generator = torch.Generator().manual_seed(7090784)
    with (
        model_extra_attrs({"attention_metadata": lambda: metadata}),
        torch_compiling(True),
        enable_breakable_cuda_graph(),
    ):
        # More than Dynamo's recompile limit, with one padding row each time.
        for rows in range(65, 105):
            metadata.num_tokens = rows
            metadata.index_cache.fill_(sentinel)
            hidden = torch.randn(rows + 1, 16, generator=generator, dtype=torch.bfloat16)
            torch.testing.assert_close(compiled(hidden)[:rows], hidden[:rows] + 1)
    assert counter.frame_count == 1
    assert len(compiler._piecewise_runners) == 2

    with torch_compiling(False):
        torch.testing.assert_close(forward(hidden[:rows]), hidden[:rows] + 1)


def test_compiled_mxfp8_mode_is_fixed_at_construction() -> None:
    state = _Mxfp8GemmState()
    method = state.quant_method
    method.backend = "auto"
    with flashinfer_mxfp8_decode_graph_capture(), flashinfer_mxfp8_autotune():
        with torch_compiling(False):
            eager = _mx_mode(state)
            assert eager.decode_graph and eager.autotune and eager.flashinfer
        with torch_compiling(True):
            assert _mx_mode(state) is state.compiled_mode
    # The engine's warmup flips the tuning flags after the first compile and before capture.
    method.enable_native_autotune()
    method.tune_mxfp8_backends = True
    method.mark_native_autotuned()
    method.mark_flashinfer_autotuned()
    with torch_compiling(True):
        compiled = _mx_mode(state)
    assert compiled is state.compiled_mode
    assert not (compiled.decode_graph or compiled.autotune or compiled.tuned_quantize_gemm)
    assert not compiled.flashinfer


@pytest.fixture
def make_meta_core(make_cpu_model):
    """A real core with its weights on the meta device. Tracing it runs every op's fake, so the compiled prefill
    dispatch runs exactly as written, with no GPU kernel and nothing stubbed."""

    def make_core(modeling: ModuleType, *, fused_projection: bool) -> torch.nn.Module:
        core = make_cpu_model(modeling, fused_projection=fused_projection)
        for weight in core.w.values():
            weight.data.fill_(1)
        # post_load_weights derives the MoE kernel scalars on the host before the targets are built.
        moe = {i: core._derive_moe(i) for i in sorted(core.moe_layers)}
        core.to("meta")
        core.moe = {
            i: dataclasses.replace(
                scalars,
                **{
                    field.name: getattr(scalars, field.name).to("meta")
                    for field in dataclasses.fields(scalars)
                },
            )
            for i, scalars in moe.items()
        }
        core._prefill = modeling.PrefillTarget(core)
        core._decode = modeling.DecodeTarget(core)
        return core

    return make_core


@dataclass
class _UnreadCaches:
    """The metadata handles ``step_args`` borrows. Compiled prefill reads them only inside the attention boundary."""

    kv_cache_manager: object = None
    msa_out_cache_loc: object = None

    def msa_idx_k_cache(self, layer: int) -> torch.Tensor:
        raise AssertionError("traced prefill code read the index-K cache")


class _GraphRecorder:
    def __init__(self) -> None:
        self.graphs: list[torch.fx.GraphModule] = []

    def __call__(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Callable:
        self.graphs.append(gm)
        return gm.forward

    def ops(self, graph: torch.fx.GraphModule) -> list[str]:
        return [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]


_TARGET_IDS = ((tp2, "tp2"), (tp4, "tp4"), (tp8, "tp8"))
_ATTENTION = "trtllm.modeling_v2_m3_prefill_attention.default"


@pytest.mark.parametrize(
    ("modeling", "fused_projection", "core_attrs", "buckets", "op", "calls_per_graph"),
    [
        # Descending buckets across the eager dispatch's row thresholds (8, 9..48, 32) share one graph.
        *(
            pytest.param(
                modeling,
                fused,
                {"moe_max_num_tokens": 64},
                (64, 48, 33, 32, 9, 8, 2),
                _ATTENTION,
                [2],
                id=f"{name}-{'fused' if fused else 'unfused'}",
            )
            for modeling, name in _TARGET_IDS
            for fused in (True, False)
        ),
        # TP2's tensor-core router takes 9..48 rows, splitting those buckets off as the eager dispatch does.
        pytest.param(
            tp2,
            True,
            {"moe_max_num_tokens": 64, "router_tc_cluster16": True},
            (64, 48, 9),
            "trtllm.minimax_m3_router_tc",
            [0, 1],
            id="tp2-router-tc",
        ),
        # 16-row MoE chunks compile once per chunk count, whatever the last chunk's size. Eager prefill overlaps
        # chunks on a stream the compiled graph must not trace.
        *(
            pytest.param(
                modeling,
                True,
                {"moe_max_num_tokens": 16, "aux_stream_moe_chunk": object()},
                (64, 48, 40, 34),
                "trtllm.fp4_block_scale_moe_runner",
                [4, 3],
                id=f"{name}-moe-chunks",
            )
            for modeling, name in _TARGET_IDS
        ),
    ],
)
def test_compiled_prefill_core_graphs(
    make_meta_core,
    modeling: ModuleType,
    fused_projection: bool,
    core_attrs: dict,
    buckets: tuple[int, ...],
    op: str,
    calls_per_graph: list[int],
) -> None:
    core = make_meta_core(modeling, fused_projection=fused_projection)
    for name, value in core_attrs.items():
        setattr(core, name, value)
    recorder = _GraphRecorder()
    torch._dynamo.reset()
    compiled = torch.compile(core, fullgraph=True, dynamic=True, backend=recorder)
    with torch_compiling(True), model_extra_attrs({}):
        for rows in buckets:
            output = compiled(
                attn_metadata=_UnreadCaches(),
                input_ids=torch.empty(rows, dtype=torch.long, device="meta"),
                position_ids=torch.empty(1, rows, dtype=torch.int32, device="meta"),
                has_context=True,
            )
            assert output.shape == (rows, core.hidden)

    assert [recorder.ops(graph).count(op) for graph in recorder.graphs] == calls_per_graph
    for graph in recorder.graphs:
        ops = recorder.ops(graph)
        # Request-dependent attention runs at a piecewise boundary, once per layer.
        assert ops.count(_ATTENTION) == core.num_layers
        # Eager-only kernels stay out: the C++ dense SwiGLU and directly launched Triton kernels.
        assert not [op for op in ops if "swiglu_oai" in op or "triton_kernel_wrapper" in op]


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
def test_shell_compiles_only_eligible_prefill(make_cpu_model, modeling: ModuleType) -> None:
    shell = make_cpu_model(modeling, shell=True)
    args = TorchLlmArgs.model_construct(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        torch_compile_config=TorchCompileConfig(),
    )
    model, prefill_only = model_setup.apply_torch_compile(
        shell, "eager", llm_args=args, fullgraph=True
    )
    # Decode and above-ceiling prefill keep their eager dispatch.
    assert model is shell and prefill_only
    assert isinstance(shell.model, PrefillCompiledModel)


@dataclass
class _Batch:
    num_tokens: int
    num_contexts: int


class _TargetProbe:
    def __init__(self, name: str) -> None:
        self.name = name

    def forward(self, *args: object) -> str:
        return self.name


class _NoCounts:
    """Attention metadata whose step counts must not be read."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"the core read attn_metadata.{name}")


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
def test_core_dispatches_on_the_shell_phase(modeling: ModuleType) -> None:
    core = Mock(_prefill=_TargetProbe("prefill"), _decode=_TargetProbe("decode"))
    forward = _TARGETS[modeling][0].forward
    input_ids = torch.ones(4, dtype=torch.long)

    # Direct calls derive the phase from the context count: mixed steps belong to prefill.
    for num_contexts, expected in ((0, "decode"), (1, "prefill"), (7, "prefill")):
        metadata = _Batch(num_tokens=4, num_contexts=num_contexts)
        assert forward(core, metadata, input_ids) == expected

    # The shell's phase wins, and the core then reads no step count.
    assert forward(core, _NoCounts(), input_ids, has_context=True) == "prefill"
    assert forward(core, _NoCounts(), input_ids, has_context=False) == "decode"
