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
from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import modeling_support
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
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.common import pad_input_tokens
from tensorrt_llm._torch.utils import (
    get_global_attrs,
    is_torch_compiling,
    model_extra_attrs,
    torch_compiling,
)
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


def test_compiled_prefill_disables_mxfp8_decode_and_autotune_modes() -> None:
    state = _Mxfp8GemmState()
    state.quant_method.backend = "auto"
    with (
        torch_compiling(False),
        flashinfer_mxfp8_decode_graph_capture(),
        flashinfer_mxfp8_autotune(),
    ):
        eager = _mx_mode(state)
        assert eager.decode_graph and eager.autotune and eager.flashinfer
        with torch_compiling(True):
            compiled = _mx_mode(state)
            assert not compiled.decode_graph and not compiled.autotune and not compiled.flashinfer
        assert _mx_mode(state) == eager


def test_compiled_mxfp8_mode_ignores_warmup_tuning_flags() -> None:
    state = _Mxfp8GemmState()
    method = state.quant_method

    def forward(x: torch.Tensor) -> torch.Tensor:
        mode = _mx_mode(state)
        assert not (mode.decode_graph or mode.autotune or mode.tuned_quantize_gemm)
        return x + mode.native_tuning + 2 * mode.flashinfer

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(forward, fullgraph=True, backend=counter)
    x = torch.zeros(4)
    with torch_compiling(True):
        expected = compiled(x)
        # The engine's warmup flips every tuning flag after the first compile and before capture.
        method.enable_native_autotune()
        method.backend = "auto"
        method.tune_mxfp8_backends = True
        with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
            torch.testing.assert_close(compiled(x), expected)
        method.mark_native_autotuned()
        method.mark_flashinfer_autotuned()
        torch.testing.assert_close(compiled(x), expected)
    assert counter.frame_count == 1


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


def _run_compiled_prefill(compiled: Callable, core: torch.nn.Module, rows: int) -> None:
    output = compiled(
        attn_metadata=_UnreadCaches(),
        input_ids=torch.empty(rows, dtype=torch.long, device="meta"),
        position_ids=torch.empty(1, rows, dtype=torch.int32, device="meta"),
        has_context=True,
    )
    assert output.shape == (rows, core.hidden)


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
@pytest.mark.parametrize("fused_projection", [True, False], ids=["fused", "unfused"])
def test_compiled_prefill_core_shares_one_graph_across_buckets(
    make_meta_core, modeling: ModuleType, fused_projection: bool
) -> None:
    core = make_meta_core(modeling, fused_projection=fused_projection)
    core.moe_max_num_tokens = 64
    recorder = _GraphRecorder()
    torch._dynamo.reset()
    compiled = torch.compile(core, fullgraph=True, dynamic=True, backend=recorder)
    with torch_compiling(True), model_extra_attrs({}):
        # Descending buckets across the eager dispatch's row thresholds (8, 9..48, 32).
        for rows in (64, 48, 33, 32, 9, 8, 2):
            _run_compiled_prefill(compiled, core, rows)
        assert len(recorder.graphs) == 1
        # Dynamo specializes a one-row input on its own.
        _run_compiled_prefill(compiled, core, 1)
    assert len(recorder.graphs) == 2

    for graph in recorder.graphs:
        ops = recorder.ops(graph)
        # Request-dependent attention runs at a piecewise boundary, once per layer.
        assert ops.count("trtllm.modeling_v2_m3_prefill_attention.default") == core.num_layers
        # Eager-only kernels stay out: the C++ dense SwiGLU and directly launched Triton kernels.
        assert not [op for op in ops if "swiglu_oai" in op or "triton_kernel_wrapper" in op]


def test_compiled_tp2_prefill_takes_the_tensor_core_router_on_its_rows(make_meta_core) -> None:
    core = make_meta_core(tp2, fused_projection=True)
    core.router_tc_cluster16 = True
    recorder = _GraphRecorder()
    torch._dynamo.reset()
    compiled = torch.compile(core, fullgraph=True, dynamic=True, backend=recorder)
    with torch_compiling(True), model_extra_attrs({}):
        # The 9..48-row router splits the buckets, as it splits the eager dispatch.
        for rows in (64, 48, 9):
            _run_compiled_prefill(compiled, core, rows)
    router_calls = [
        recorder.ops(graph).count("trtllm.minimax_m3_router_tc") for graph in recorder.graphs
    ]
    assert router_calls == [0, len(core.moe_layers)]


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
def test_compiled_prefill_moe_chunks_compile_once_per_chunk_count(
    make_meta_core, modeling: ModuleType
) -> None:
    core = make_meta_core(modeling, fused_projection=True)
    core.moe_max_num_tokens = 16
    # Eager prefill overlaps chunks on this stream; the compiled graph must not trace it.
    core.aux_stream_moe_chunk = object()
    recorder = _GraphRecorder()
    torch._dynamo.reset()
    compiled = torch.compile(core, fullgraph=True, dynamic=True, backend=recorder)
    with torch_compiling(True), model_extra_attrs({}):
        # Four chunks, then three chunks whose last chunk has 16, 8 or 2 rows.
        for rows in (64, 48, 40, 34):
            _run_compiled_prefill(compiled, core, rows)
    runner_calls = [
        recorder.ops(graph).count("trtllm.fp4_block_scale_moe_runner") for graph in recorder.graphs
    ]
    assert runner_calls == [4 * len(core.moe_layers), 3 * len(core.moe_layers)]


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8])
def test_value_dump_skips_symbolic_counts_during_fullgraph_trace(
    monkeypatch: pytest.MonkeyPatch, modeling: ModuleType
) -> None:
    values = set()
    monkeypatch.setattr(modeling, "_VALUE_DUMP", True)
    monkeypatch.setattr(modeling, "_VALUES", values)

    def forward(x: torch.Tensor) -> torch.Tensor:
        modeling._record("prefill", int(x.shape[0]), 1)
        return x + 1

    counter = CompileCounter()
    compiled = torch.compile(forward, fullgraph=True, dynamic=True, backend=counter)
    for rows in range(9, 49):
        x = torch.ones(rows)
        torch.testing.assert_close(compiled(x), x + 1)
    assert counter.frame_count == 1
    assert not values
    forward(x)
    assert values == {("prefill", 48, 1)}


class _CompileProbe(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        traced = torch.full_like(input_ids, torch.compiler.is_compiling(), dtype=torch.bool)
        compile_policy = torch.full_like(input_ids, is_torch_compiling(), dtype=torch.bool)
        return torch.stack((traced, compile_policy), dim=-1)


class _PassThroughLogits(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, *args: object) -> torch.Tensor:
        return hidden


@dataclass
class _Batch:
    num_tokens: int
    num_contexts: int
    padded_num_tokens: int | None = None


@dataclass
class _EagleCapture:
    hidden_states: torch.Tensor
    hidden_size: int
    gather_ids: torch.Tensor
    num_capture_tokens: int = 0
    layers_to_capture: tuple[int, ...] = (0,)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self.layers_to_capture


class _EagleWorker:
    _m3_capture_norm = None
    _m3_fc_early_drafter = None

    def arm_capture_norm(self, *args) -> None:
        pass

    def capture_hidden_states(self, *args) -> None:
        raise AssertionError("compiled prefill should use bucket capture")

    def capture_target(self, *args) -> None:
        raise AssertionError("prefill does not use decode boundary capture")

    def tail_eligible(self, *args) -> bool:
        return False

    def __call__(self, *, hidden_states, **kwargs) -> torch.Tensor:
        return hidden_states


@pytest.mark.parametrize("modeling", [tp2, tp4, tp8], ids=["tp2", "tp4", "tp8"])
@pytest.mark.parametrize("eagle3", [False, True], ids=["plain", "eagle3"])
def test_prefill_trace_reuses_bucket_across_request_counts(
    monkeypatch: pytest.MonkeyPatch, make_cpu_model, modeling: ModuleType, eagle3: bool
) -> None:
    shell = make_cpu_model(modeling, shell=True)
    core = shell.model
    core._prefill = modeling.PrefillTarget(core)
    core._decode = modeling.DecodeTarget(core)
    shell.logits_processor = _PassThroughLogits()
    monkeypatch.setattr(core._prefill, "_attention", lambda layer, pos, x, step: (x, False))
    monkeypatch.setattr(core._prefill, "_swiglu_mlp", lambda x, *args, **kwargs: x + 1)
    monkeypatch.setattr(core._prefill, "_moe", lambda layer, x, **kwargs: (x + 1, None))
    monkeypatch.setattr(modeling, "_native_allreduce_norm", lambda core, x, r, w: (x, x))
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", lambda x, *args: x)

    def decode(metadata, input_ids, positions, hidden, *args):
        return hidden - 1

    monkeypatch.setattr(core._decode, "forward", decode)
    monkeypatch.setattr(
        get_global_attrs(), "per_request_prefill_cuda_graph_flag", True, raising=False
    )
    args = TorchLlmArgs.model_construct(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        torch_compile_config=TorchCompileConfig(),
    )
    torch._dynamo.reset()
    counter = CompileCounter()
    model, prefill_only = model_setup.apply_torch_compile(
        shell, counter, llm_args=args, fullgraph=True
    )
    caller = ModelCaller(model, prefill_compile_only=prefill_only, nccl_window_scopes_enabled=False)
    hidden = torch.ones((8, core.hidden), dtype=torch.bfloat16)
    metadata = _AttentionStep(0, torch.empty(0), torch.arange(8), padded_num_tokens=8)
    spec_metadata = None
    if eagle3:
        shell.spec_worker = _EagleWorker()
        spec_metadata = _EagleCapture(torch.empty_like(hidden), core.hidden, torch.arange(1))

        def cpu_slice_copy(dest, src, start, end):
            dest[: src.shape[0], start:end].copy_(src)

        monkeypatch.setattr(modeling_support, "inplace_slice_copy", cpu_slice_copy)

    with model_extra_attrs({}):
        for contexts, live_tokens in ((1, 6), (4, 7), (1, 5)):
            metadata.num_contexts = contexts
            metadata.num_tokens = metadata.num_ctx_tokens = live_tokens
            hidden.fill_(contexts + live_tokens)
            if spec_metadata is not None:
                spec_metadata.num_capture_tokens = live_tokens
            output = caller(
                attn_metadata=metadata, inputs_embeds=hidden, spec_metadata=spec_metadata
            )
            torch.testing.assert_close(output, hidden[:live_tokens] + core.num_layers)
            if spec_metadata is not None:
                torch.testing.assert_close(
                    spec_metadata.hidden_states[:live_tokens], hidden[:live_tokens] + 1
                )
        assert counter.frame_count == 1

        # An eligible call with zero local contexts still selects decode.
        shell.spec_worker = None
        metadata.num_contexts = metadata.num_ctx_tokens = 0
        output = caller(attn_metadata=metadata, inputs_embeds=hidden)
        torch.testing.assert_close(output, hidden[: metadata.num_tokens] - 1)


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


@pytest.mark.parametrize(
    ("modeling", "num_ctx_tokens", "num_gen_tokens", "expected_compiled"),
    [
        pytest.param(tp2, 8, 0, True, id="tp2-prefill"),
        pytest.param(tp2, 0, 4, False, id="tp2-decode"),
        pytest.param(tp4, 8, 0, True, id="tp4-prefill-at-ceiling"),
        pytest.param(tp4, 4, 2, True, id="tp4-mixed"),
        pytest.param(tp4, 0, 4, False, id="tp4-decode"),
        pytest.param(tp4, 9, 0, False, id="tp4-prefill-above-ceiling"),
        pytest.param(tp8, 8, 0, True, id="tp8-prefill"),
        pytest.param(tp8, 0, 4, False, id="tp8-decode"),
    ],
)
def test_shell_routes_by_prefill_graph_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    make_cpu_model,
    modeling: ModuleType,
    num_ctx_tokens: int,
    num_gen_tokens: int,
    expected_compiled: bool,
) -> None:
    shell = make_cpu_model(modeling, shell=True)
    shell.model = _CompileProbe()
    shell.logits_processor = _PassThroughLogits()
    args = TorchLlmArgs.model_construct(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        torch_compile_config=TorchCompileConfig(compile_only_piecewise_graphs=False),
    )
    torch._dynamo.reset()
    counter = CompileCounter()
    model, prefill_only = model_setup.apply_torch_compile(
        shell, counter, llm_args=args, fullgraph=True
    )
    assert model is shell and prefill_only
    caller = ModelCaller(model, prefill_compile_only=prefill_only, nccl_window_scopes_enabled=False)

    capture_ceiling = 8
    num_tokens = num_ctx_tokens + num_gen_tokens
    num_contexts = int(num_ctx_tokens > 0)
    metadata = _Batch(num_tokens=num_tokens, num_contexts=num_contexts)
    input_ids = torch.ones(max(num_tokens, capture_ceiling), dtype=torch.long)
    position_ids = torch.arange(input_ids.numel())
    prior_policy = not expected_compiled
    monkeypatch.setattr(
        get_global_attrs(), "per_request_prefill_cuda_graph_flag", prior_policy, raising=False
    )
    padded_tokens = pad_input_tokens(
        metadata,
        num_tokens,
        num_contexts,
        None,
        input_ids=input_ids,
        position_ids=position_ids,
        dist=None,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        prefill_cuda_graph_num_tokens=[capture_ceiling],
    )
    with model_extra_attrs({}), torch_compiling(prior_policy):
        output = caller(
            attn_metadata=metadata,
            input_ids=input_ids[:padded_tokens],
            position_ids=position_ids[:padded_tokens],
        )
        assert is_torch_compiling() is prior_policy

    assert counter.frame_count == int(expected_compiled)
    expected = torch.full((num_tokens, 2), expected_compiled, dtype=torch.bool)
    torch.testing.assert_close(output, expected)
