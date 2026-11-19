# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile real prefill host dispatch, substituting CPU math for GPU kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import prefill_attention as boundary
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling as tp2,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (
    modeling as tp4,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8 import (
    modeling as tp8,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m3_dense_swiglu import (
    ALPHA,
    BETA,
    LIMIT,
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
_Prefill = tp2.PrefillTarget | tp4.PrefillTarget | tp8.PrefillTarget


@dataclass
class _Metadata:
    num_contexts: int
    num_ctx_tokens: int
    num_tokens: int
    kv_cache_manager: object = None
    msa_idx_k_cache: object = None
    msa_out_cache_loc: object = None


@pytest.mark.parametrize("attention_kind", ["dense", "fused_sparse", "unfused_sparse"])
@pytest.mark.parametrize("modeling,tp", [(tp2, 2), (tp4, 4), (tp8, 8)])
def test_prefill_fullgraph_keeps_token_rows_dynamic(
    monkeypatch: pytest.MonkeyPatch, modeling: ModuleType, tp: int, attention_kind: str
) -> None:
    core = object.__new__(getattr(modeling, f"MiniMaxM3Tp{tp}ModelingV2"))
    torch.nn.Module.__init__(core)
    core.num_layers = 1
    core.heads, core.head_dim = 1, 16
    core.q_size = core.kv_size = 16
    core.sparse_layers = set() if attention_kind == "dense" else {0}
    core.fused_projection = attention_kind == "fused_sparse"
    core.main_kv_is_fp8 = False
    core.attn_events, core.aux_stream_attn = [(None, None)], None
    core.main_kv_is_nvfp4 = False
    core.attn = [None]
    live_metadata = _Metadata(1, 0, 0)
    core.mxfp8 = _Mxfp8GemmState()
    core.mx_alpha = torch.ones(1)
    core.swiglu_alpha = ALPHA
    core.swiglu_limit = LIMIT
    core.swiglu_beta = BETA
    generator = torch.Generator().manual_seed(7090784)
    core.w = {
        "mlp_gu_w": torch.randn(32, 16, generator=generator, dtype=torch.bfloat16),
        "mlp_gu_s": torch.empty(0),
        "mlp_dn_w": torch.randn(16, 16, generator=generator, dtype=torch.bfloat16),
        "mlp_dn_s": torch.empty(0),
    }

    identity = torch.eye(16, dtype=torch.bfloat16)
    core.w.update(
        l0_qkv_w=identity.repeat(3, 1),
        l0_qkv_s=torch.empty(0),
        l0_o_w=identity,
        l0_o_s=torch.empty(0),
        l0_idx_w=identity,
    )

    def gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        alpha: torch.Tensor,
        dtype: torch.dtype,
        tuned: bool,
        tune: bool,
        flashinfer: bool,
        native_tuning: bool,
    ) -> torch.Tensor:
        assert not tuned and not tune and not flashinfer
        return x @ weight.t()

    def swiglu(x: torch.Tensor, limit: float, alpha: float, beta: float) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + beta)

    def msa(
        backend: object,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        idx_q: torch.Tensor | None,
        metadata: _Metadata,
        output: torch.Tensor,
        sparse: bool,
        inv_scale: torch.Tensor | None,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        assert sparse is (attention_kind != "dense")
        assert (k is None) is (attention_kind == "fused_sparse")
        assert (v is None) is (attention_kind == "fused_sparse")
        assert (idx_q is None) is (attention_kind == "dense")
        assert q.shape[0] == metadata.num_tokens
        output.copy_(q + 1)
        return output

    # Exercise real attention dispatch with CPU math and collective stand-ins.
    def producer(
        packed: torch.Tensor, position_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed.shape[0] == live_metadata.num_tokens
        assert position_ids is None or position_ids.shape[-1] == live_metadata.num_tokens
        return packed[:, :16], packed[:, :16]

    def layer(
        self: _Prefill,
        i: int,
        position_ids: torch.Tensor | None,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        step: dict[str, object],
        spec_metadata: object,
        capture: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, _ = self._attention(i, position_ids, hidden, step)
        return self._swiglu_mlp(attended, "mlp", False, dense=True), residual

    monkeypatch.setattr(modeling, "m3_msa_attention", msa)
    monkeypatch.setattr(modeling, "_qk_norm_rope", lambda core, i, qkv, *args: qkv)
    monkeypatch.setattr(
        modeling,
        "_sparse_qkv_producer",
        lambda core, i, packed, position_ids, *args: producer(packed, position_ids),
    )
    monkeypatch.setattr(modeling, "_producer_covered", lambda core: False)
    monkeypatch.setattr(modeling, "m3_bf16_linear", lambda x, w: x @ w.t())
    monkeypatch.setattr(modeling, "_fp8_index_qk_norm_rope", lambda core, i, value, *args: value)
    monkeypatch.setattr(
        boundary, "get_model_extra_attrs", lambda: {"attention_metadata": lambda: live_metadata}
    )
    monkeypatch.setattr(modeling, "m3_mxfp8_linear", gemm)
    monkeypatch.setattr(modeling, "m3_silu_and_mul", swiglu)
    monkeypatch.setattr(modeling.PrefillTarget, "_layer", layer)
    monkeypatch.setattr(modeling, "_require", [False])
    target = modeling.PrefillTarget(core)

    def forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        rows = inputs_embeds.shape[0]
        md = _Metadata(1, rows, rows)
        pos = torch.arange(rows)[None]
        return target.forward(md, None, pos, inputs_embeds, None, None, None)

    torch._dynamo.reset()
    counter = CompileCounter()
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    monkeypatch.setattr(Backend, "build_custom_passes", lambda *args: [])
    compiler = Backend(enable_inductor=False, enable_piecewise_cuda_graph=True)

    def compile_piecewise(gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Callable:
        counter(gm, inputs)
        return compiler(gm, inputs)

    compiled = torch.compile(forward, fullgraph=True, dynamic=True, backend=compile_piecewise)
    # Keep CPU SwiGLU scalars static; compilation overrides decode-graph tuning scopes.
    with (
        torch._dynamo.config.patch(specialize_float=True),
        torch_compiling(False),
        flashinfer_mxfp8_decode_graph_capture(),
    ):
        assert _mx_mode(core.mxfp8).decode_graph
        with torch_compiling(True), enable_breakable_cuda_graph():
            assert not _mx_mode(core.mxfp8).decode_graph
            # Stay above fixture weight dimensions: the piecewise token-axis
            # detector guards on equal dimensions independently of M3 dispatch.
            for rows in range(65, 105):
                live_metadata.num_tokens = rows
                x = torch.randn(rows + 1, 16, generator=generator, dtype=torch.bfloat16)
                gu = (x[:rows] + 1) @ core.w["mlp_gu_w"].t()
                activated = swiglu(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
                expected = activated @ core.w["mlp_dn_w"].t()
                torch.testing.assert_close(compiled(x)[:rows], expected)
        assert _mx_mode(core.mxfp8).decode_graph
    assert counter.frame_count == 1
    assert len(compiler._piecewise_runners) == 2
    with torch_compiling(False):
        eager, _ = target._attention(0, None, x[:rows], target.step_args(live_metadata))
        torch.testing.assert_close(eager[:rows], x[:rows] + 1)

    core.mxfp8.quant_method.backend = "auto"
    with torch_compiling(False), flashinfer_mxfp8_autotune():
        eager = _mx_mode(core.mxfp8)
        assert eager.autotune and eager.flashinfer
        with torch_compiling(True):
            compiled_mode = _mx_mode(core.mxfp8)
            assert not compiled_mode.autotune and not compiled_mode.flashinfer
        restored = _mx_mode(core.mxfp8)
        assert restored.autotune and restored.flashinfer


@pytest.mark.parametrize("modeling,tp", [(tp2, 2), (tp4, 4), (tp8, 8)])
def test_shell_installs_prefill_only_wrapper(
    monkeypatch: pytest.MonkeyPatch, modeling: ModuleType, tp: int
) -> None:
    shell_type = getattr(modeling, f"ModelingV2MiniMaxM3Nvfp4Sm107Tp{tp}")
    shell = object.__new__(shell_type)
    torch.nn.Module.__init__(shell)
    shell.model = torch.nn.Identity()
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: torch.nn.ReLU())
    args = TorchLlmArgs.model_construct(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        torch_compile_config=TorchCompileConfig(compile_only_piecewise_graphs=False),
    )
    result, prefill_only = model_setup.apply_torch_compile(
        shell, "eager", llm_args=args, fullgraph=True
    )
    assert result is shell and prefill_only
    runner = "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner"
    x = torch.tensor([-1.0, 1.0])
    for eligible in [True, False]:
        monkeypatch.setattr(f"{runner}.get_per_request_prefill_cuda_graph_flag", lambda: eligible)
        torch.testing.assert_close(shell.model(x), x.relu() if eligible else x)


@pytest.mark.parametrize(
    ("num_ctx_tokens", "num_gen_tokens", "expected_compiled"),
    [
        pytest.param(8, 0, True, id="prefill-at-ceiling"),
        pytest.param(4, 2, True, id="mixed"),
        pytest.param(0, 4, False, id="decode-only"),
        pytest.param(9, 0, False, id="prefill-above-ceiling"),
    ],
)
def test_tp4_routes_by_prefill_graph_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    num_ctx_tokens: int,
    num_gen_tokens: int,
    expected_compiled: bool,
) -> None:
    observed_compile_policy = []

    def model_body(input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        observed_compile_policy.append(is_torch_compiling())
        return input_ids

    eager_forward = Mock(side_effect=model_body)
    compiled_forward = Mock(side_effect=model_body)
    eager_model = torch.nn.Module()
    eager_model.forward = eager_forward
    compiled_model = torch.nn.Module()
    compiled_model.forward = compiled_forward
    monkeypatch.setattr(torch, "compile", Mock(return_value=compiled_model))

    shell = object.__new__(tp4.ModelingV2MiniMaxM3Nvfp4Sm107Tp4)
    torch.nn.Module.__init__(shell)
    shell.model = eager_model
    shell.model_config = Mock(extra_attrs={})
    shell.spec_worker = None
    shell.lm_head = torch.nn.Identity()
    shell.logits_processor = Mock()
    args = TorchLlmArgs.model_construct(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        torch_compile_config=TorchCompileConfig(compile_only_piecewise_graphs=False),
    )
    model, prefill_only = model_setup.apply_torch_compile(
        shell, "eager", llm_args=args, fullgraph=True
    )
    caller = ModelCaller(model, prefill_compile_only=prefill_only, nccl_window_scopes_enabled=False)

    capture_ceiling = 8
    num_tokens = num_ctx_tokens + num_gen_tokens
    num_contexts = int(num_ctx_tokens > 0)
    metadata = Mock(spec=AttentionMetadata, num_tokens=num_tokens, num_contexts=num_contexts)
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
        caller(
            attn_metadata=metadata,
            input_ids=input_ids[:padded_tokens],
            position_ids=position_ids[:padded_tokens],
        )
        assert is_torch_compiling() is prior_policy

    selected, skipped = (
        (compiled_forward, eager_forward)
        if expected_compiled
        else (eager_forward, compiled_forward)
    )
    selected.assert_called_once()
    skipped.assert_not_called()
    assert observed_compile_policy == [expected_compiled]
