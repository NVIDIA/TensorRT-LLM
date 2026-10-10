# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE-CuteDSL deferred finalize (skip_topk_reduce) tests.

Contract tests (CPU): kernel constructor guards, compile-cache key split,
runner unique_id split, scheduler do_finalize gating, backend env-knob wiring.

GPU tests (SM100 family + cutlass-dsl): drive the
``trtllm::cute_dsl_megamoe_nvfp4_blackwell`` op directly with synthetic NVFP4
weights (all-1.0 scale factors, random fp4 codes -- every fp4 bit pattern is
finite) and check, in ladder order:

1. deferred-finalize triple vs the standalone TopkReduce output (and that the
   exposed rows carry NO routing weight);
2. the model finalize (``post_expert_norm_combine``) consumes the dense
   identity-mapped triple identically to the op-by-op reference composition;
3. ``apply_topk_in_fc1=False`` (weights in the reduce) vs the default
   ``apply_topk_in_fc1=True`` (weights folded pre-fc1-quant), tolerance-based
   -- the two differ only by where the fc1-out NVFP4 quantization sees the
   routing weight.
"""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import IS_MEGAMOE_OP_AVAILABLE
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import FusedCommMoEScheduler
from tensorrt_llm._torch.moe.fused_moe.routing import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

requires_megamoe_op = pytest.mark.skipif(
    not IS_MEGAMOE_OP_AVAILABLE,
    reason="MegaMoE CuteDSL op is not available (cutlass-dsl[cu13] missing)",
)


def _requires_sm100_gpu() -> Optional[str]:
    if not IS_MEGAMOE_OP_AVAILABLE:
        return "MegaMoE CuteDSL op is not available"
    if not torch.cuda.is_available():
        return "CUDA required"
    if get_sm_version() not in (100, 103):
        return f"requires SM100/SM103, got SM{get_sm_version()}"
    return None


# ---------------------------------------------------------------------------
# Kernel constructor guards + cache-key split
# ---------------------------------------------------------------------------


def _make_kernel(**overrides):
    import cutlass

    from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4 import SfPaddingBlock, import_kernel
    from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4.token_comm import CombineFormat

    kwargs = dict(
        mma_tiler_mnk=(256, 128, 256),
        cluster_shape_mnk=(2, 1, 1),
        use_2cta_instrs=True,
        group_hint=512,
        token_padding_block=64,
        sf_padding_block=SfPaddingBlock,
        static_expert_shape=(8, 1024, 512),
        world_size=1,
        num_topk=4,
        max_tokens_per_rank=128,
        hidden=512,
        fc2_output_dtype=cutlass.BFloat16,
        combine_format=CombineFormat.parse("bf16"),
    )
    kwargs.update(overrides)
    return import_kernel()(**kwargs)


@requires_megamoe_op
def test_kernel_skip_topk_reduce_rejects_form_b():
    with pytest.raises(ValueError, match="skip_topk_reduce requires in_kernel_fc2_reduce=False"):
        _make_kernel(
            skip_topk_reduce=True,
            in_kernel_fc2_reduce=True,
            apply_topk_in_fc1=True,
        )


@requires_megamoe_op
def test_kernel_skip_topk_reduce_rejects_quantized_wire():
    from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4.token_comm import CombineFormat

    with pytest.raises(ValueError, match="skip_topk_reduce requires the bf16 combine format"):
        _make_kernel(
            skip_topk_reduce=True,
            combine_format=CombineFormat.parse("32e4m3xe8m0"),
        )


@requires_megamoe_op
def test_kernel_name_splits_compile_cache_on_skip():
    base = _make_kernel(apply_topk_in_fc1=False)
    skip = _make_kernel(apply_topk_in_fc1=False, skip_topk_reduce=True)
    assert base.name() != skip.name(), "skip_topk_reduce must split the compiled-kernel cache key"
    assert "_skipredu" in skip.name() and "_skipredu" not in base.name()


@requires_megamoe_op
def test_query_combine_quant_region_matches_bf16_tensor_size():
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        query_megamoe_combine_quant_region,
    )

    max_t, top_k, hidden = 128, 4, 512
    offset, nbytes = query_megamoe_combine_quant_region(
        world_size=1,
        num_topk=top_k,
        num_experts_per_rank=8,
        hidden_size=hidden,
        intermediate_size_per_partition=512,
        expand_intermediate_size_per_partition=1024,
        max_tokens_per_rank=max_t,
    )
    assert nbytes == max_t * top_k * hidden * 2
    assert offset % 128 == 0, "combine_quant must stay 128 B aligned for the STG target"
    with pytest.raises(ValueError, match="bf16 combine format"):
        query_megamoe_combine_quant_region(
            world_size=1,
            num_topk=top_k,
            num_experts_per_rank=8,
            hidden_size=hidden,
            intermediate_size_per_partition=512,
            expand_intermediate_size_per_partition=1024,
            max_tokens_per_rank=max_t,
            combine_format="32e4m3xe8m0",
        )


# ---------------------------------------------------------------------------
# Runner unique_id split (needs a Blackwell GPU only for the ctor SM check)
# ---------------------------------------------------------------------------


def test_runner_unique_id_splits_on_skip_topk_reduce():
    reason = _requires_sm100_gpu()
    if reason:
        pytest.skip(reason)
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        Sm100MegaMoENvfp4Runner,
    )

    common = dict(
        world_size=1,
        local_rank=0,
        num_topk=4,
        num_experts_per_rank=8,
        hidden_size=512,
        intermediate_size_per_partition=512,
        expand_intermediate_size_per_partition=1024,
        max_tokens_per_rank=128,
        output_dtype=torch.bfloat16,
        apply_topk_in_fc1=False,
    )
    base = Sm100MegaMoENvfp4Runner(**common)
    skip = Sm100MegaMoENvfp4Runner(skip_topk_reduce=True, **common)
    assert base.unique_id() != skip.unique_id(), (
        "skip_topk_reduce must enter unique_id so deferred and default runs "
        "never share compiled kernels, tactics or local workspaces"
    )


# ---------------------------------------------------------------------------
# Scheduler gating
# ---------------------------------------------------------------------------


class _StubBackend:
    scheduler_kind = None

    def __init__(self, supports_deferred: bool):
        if supports_deferred:
            self.supports_deferred_finalize = True


def _split_chunk(split_token_num: int, split_num_chunks: int):
    # Mirrors ConfigurableMoE.split_chunk (even partition).
    val_div = split_token_num // split_num_chunks
    val_mod = split_token_num % split_num_chunks
    return [val_div + 1] * val_mod + [val_div] * (split_num_chunks - val_mod)


def _stub_scheduler(supports_deferred: bool, moe_max_num_tokens: int = 4096):
    moe = types.SimpleNamespace(
        backend=_StubBackend(supports_deferred),
        use_dp=False,
        moe_max_num_tokens=moe_max_num_tokens,
        mapping=types.SimpleNamespace(moe_ep_rank=0),
        split_chunk=_split_chunk,
    )
    return FusedCommMoEScheduler(moe)


def test_scheduler_rejects_unfinalized_without_backend_capability():
    sched = _stub_scheduler(supports_deferred=False)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    logits = torch.randn(4, 8)
    with pytest.raises(NotImplementedError, match="supports_deferred_finalize"):
        sched.forward(
            x,
            logits,
            do_finalize=False,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=[4],
            use_dp_padding=False,
            input_ids=None,
        )


def test_scheduler_rejects_chunked_unfinalized_forward():
    # 8 tokens with moe_max_num_tokens=4 -> num_chunks=2: the second chunk
    # would overwrite the staging the first chunk's triple aliases.
    sched = _stub_scheduler(supports_deferred=True, moe_max_num_tokens=4)
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    logits = torch.randn(8, 8)
    with pytest.raises(NotImplementedError, match="chunked"):
        sched.forward(
            x,
            logits,
            do_finalize=False,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=[8],
            use_dp_padding=False,
            input_ids=None,
        )


def _scheduler_finalize_hook_keywords() -> set:
    """Keyword names the scheduler passes to ``unfinalized_combine_fn``.

    Read from the source of ``_finalize_before_combine`` so the contract the
    model's finalize must satisfy is the one actually in the tree.
    """
    import ast
    import inspect

    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler

    module = ast.parse(inspect.getsource(moe_scheduler))
    for func in ast.walk(module):
        if not (isinstance(func, ast.FunctionDef) and func.name == "_finalize_before_combine"):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "finalize_fn"
            ):
                return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("no finalize_fn(...) call found in _finalize_before_combine")


def test_model_finalize_accepts_scheduler_hook_keywords():
    """The registered finalize must bind every keyword the scheduler passes.

    A keyword the scheduler adds that the model's override does not accept
    only shows up as a TypeError inside engine warmup; check the signatures
    against each other here instead.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

    keywords = _scheduler_finalize_hook_keywords()
    assert keywords, "scheduler passes no keywords to finalize_fn"
    signature = inspect.signature(MuseSparkOpenMoE.finalize)
    # ``self`` plus one placeholder per scheduler keyword; bind() raises
    # TypeError on an unexpected or missing-required name.
    signature.bind(None, **{name: None for name in keywords})


# ---------------------------------------------------------------------------
# Backend knob wiring
# ---------------------------------------------------------------------------


def _make_backend() -> MegaMoECuteDsl:
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend="MEGAMOE_CUTEDSL",
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


def test_backend_deferred_finalize_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MEGAMOE_DEFERRED_FINALIZE", raising=False)
    moe = _make_backend()
    assert moe.deferred_finalize is False
    assert moe.supports_deferred_finalize is False
    assert moe.apply_topk_in_fc1 is True, "default path must stay byte-identical"


def test_backend_deferred_finalize_selects_transformers_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEGAMOE_DEFERRED_FINALIZE", "1")
    moe = _make_backend()
    assert moe.deferred_finalize is True
    assert moe.supports_deferred_finalize is True
    assert moe.apply_topk_in_fc1 is False, (
        "deferred finalize must apply routing weights post-norm (RMSNorm is "
        "scale-invariant; a folded weight would be erased)"
    )


def test_backend_deferred_finalize_rejects_quantized_wire(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEGAMOE_DEFERRED_FINALIZE", "1")
    monkeypatch.setenv("MEGAMOE_COMBINE_FORMAT", "32e4m3xe8m0")
    with pytest.raises(ValueError, match="bf16 combine wire"):
        _make_backend()


def test_backend_run_moe_rejects_unfinalized_without_knob(monkeypatch: pytest.MonkeyPatch):
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

    monkeypatch.delenv("MEGAMOE_DEFERRED_FINALIZE", raising=False)
    moe = _make_backend()
    ctx = MoERunContext(
        token_selected_experts=torch.zeros(1, 2, dtype=torch.int32),
        token_final_scales=torch.ones(1, 2),
        x=torch.zeros(1, 256, dtype=torch.uint8),
        x_sf=torch.zeros(1, 32, dtype=torch.uint8),
        output_dtype=torch.bfloat16,
        do_finalize=False,
    )
    with pytest.raises(NotImplementedError, match="MEGAMOE_DEFERRED_FINALIZE"):
        moe.run_moe(ctx)


# ---------------------------------------------------------------------------
# GPU: op-level parity ladder
# ---------------------------------------------------------------------------


class _OpHarness:
    """Synthetic single-rank problem driving the MegaMoE op directly.

    Weight scale factors are exactly 1.0 (byte 0x38), activation SFs 0.125
    (see below), and per-expert alphas / norm_consts are 1.0, so both launch
    variants see identical quantization inputs; fp4 weight/activation codes
    are random bytes (every fp4 bit pattern is finite).
    """

    def __init__(self, *, seed: int = 0):
        from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

        torch.manual_seed(seed)
        self.device = torch.device("cuda")
        self.max_t = 128
        self.top_k = 4
        self.hidden = 512
        self.inter = 512
        self.experts = 8
        d = self.device
        self.activation = torch.randint(
            0, 256, (self.max_t, self.hidden // 2), dtype=torch.uint8, device=d
        ).view(torch.float4_e2m1fn_x2)
        sf_cols = self.hidden // 16
        # 0x20 = fp8 0.125, deliberately NOT 1.0: unit activation SFs push the
        # fc1 output to ~1e5 magnitudes, deep into NVFP4 saturation (real
        # checkpoints carry calibrated scales that keep it in range).
        # Saturation is harmless for same-kernel parity tests but nonlinearly
        # amplifies the weight-placement difference in the cross-route test
        # (rel_l2 ~0.7 at SF=1.0 vs ~0.1 at SF=0.125; exact at uniform
        # weights in both regimes).
        self.activation_sf = torch.full(
            (self.max_t, sf_cols), 0x20, dtype=torch.uint8, device=d
        ).view(torch.float8_e4m3fn)
        self.topk_idx = (
            torch.arange(self.max_t * self.top_k, dtype=torch.int64, device=d) % self.experts
        ).view(self.max_t, self.top_k)
        w = torch.rand(self.max_t, self.top_k, dtype=torch.float32, device=d) + 0.05
        self.topk_weights = w / w.sum(-1, keepdim=True)
        self.fc1_weight = torch.randint(
            0, 256, (self.experts, 2 * self.inter, self.hidden // 2), dtype=torch.uint8, device=d
        ).view(torch.float4_e2m1fn_x2)
        self.fc1_weight_sf = torch.full(
            (self.experts, NVFP4MegaMoECuteDslMethod.fc1_sf_flat_size(self.inter, self.hidden)),
            0x38,
            dtype=torch.uint8,
            device=d,
        ).view(torch.float8_e4m3fn)
        self.fc2_weight = torch.randint(
            0, 256, (self.experts, self.hidden, self.inter // 2), dtype=torch.uint8, device=d
        ).view(torch.float4_e2m1fn_x2)
        self.fc2_weight_sf = torch.full(
            (self.experts, NVFP4MegaMoECuteDslMethod.fc2_sf_flat_size(self.hidden, self.inter)),
            0x38,
            dtype=torch.uint8,
            device=d,
        ).view(torch.float8_e4m3fn)
        self.fc1_alpha = torch.ones(self.experts, dtype=torch.float32, device=d)
        self.fc2_alpha = torch.ones(self.experts, dtype=torch.float32, device=d)
        self.fc1_norm_const = torch.ones(self.experts, dtype=torch.float32, device=d)

    def shared_workspace(self) -> torch.Tensor:
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            query_megamoe_shared_workspace_bytes,
        )

        nbytes = query_megamoe_shared_workspace_bytes(
            world_size=1,
            num_topk=self.top_k,
            num_experts_per_rank=self.experts,
            hidden_size=self.hidden,
            intermediate_size_per_partition=self.inter,
            expand_intermediate_size_per_partition=2 * self.inter,
            max_tokens_per_rank=self.max_t,
            apply_topk_in_fc1=False,
        )
        return torch.zeros(nbytes, dtype=torch.uint8, device=self.device)

    def run(
        self,
        *,
        apply_topk_in_fc1: bool,
        skip_topk_reduce: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        """Launch once; return (combine_output_2d, shared_workspace)."""
        combine_output = torch.zeros(
            self.max_t, 1, self.hidden, dtype=torch.bfloat16, device=self.device
        )
        shared_workspace = self.shared_workspace()
        torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
            activation=self.activation,
            activation_sf=self.activation_sf,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_weights if topk_weights is None else topk_weights,
            fc1_weight=self.fc1_weight,
            fc1_weight_sf=self.fc1_weight_sf,
            fc2_weight=self.fc2_weight,
            fc2_weight_sf=self.fc2_weight_sf,
            fc1_alpha=self.fc1_alpha,
            fc2_alpha=self.fc2_alpha,
            fc1_norm_const=self.fc1_norm_const,
            combine_output=combine_output,
            shared_workspace=shared_workspace,
            world_size=1,
            local_rank=0,
            num_topk=self.top_k,
            num_experts_per_rank=self.experts,
            hidden_size=self.hidden,
            intermediate_size_per_partition=self.inter,
            expand_intermediate_size_per_partition=2 * self.inter,
            max_tokens_per_rank=self.max_t,
            peer_offsets=[0],
            apply_topk_in_fc1=apply_topk_in_fc1,
            skip_topk_reduce=skip_topk_reduce,
        )
        torch.cuda.synchronize()
        return combine_output.squeeze(1), shared_workspace

    def combine_quant_view(self, shared_workspace: torch.Tensor) -> torch.Tensor:
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            query_megamoe_combine_quant_region,
        )

        offset, nbytes = query_megamoe_combine_quant_region(
            world_size=1,
            num_topk=self.top_k,
            num_experts_per_rank=self.experts,
            hidden_size=self.hidden,
            intermediate_size_per_partition=self.inter,
            expand_intermediate_size_per_partition=2 * self.inter,
            max_tokens_per_rank=self.max_t,
        )
        return (
            shared_workspace[offset : offset + nbytes]
            .view(torch.bfloat16)
            .view(self.max_t, self.top_k, self.hidden)
        )


@pytest.fixture()
def op_harness():
    reason = _requires_sm100_gpu()
    if reason:
        pytest.skip(reason)
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        reset_megamoe_workspace_state,
    )

    reset_megamoe_workspace_state()
    yield _OpHarness()
    reset_megamoe_workspace_state()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-6)).item()


def test_deferred_triple_matches_topk_reduce(op_harness):
    """Ladder step 1: manual weighted reduce over the exposed combine_quant
    rows must reproduce the standalone TopkReduce output, and the exposed
    rows must NOT carry the routing weight."""
    h = op_harness
    ref, _ = h.run(apply_topk_in_fc1=False, skip_topk_reduce=False)
    _, ws = h.run(apply_topk_in_fc1=False, skip_topk_reduce=True)
    rows = h.combine_quant_view(ws)  # (max_t, top_k, hidden) bf16
    manual = (rows.float() * h.topk_weights.unsqueeze(-1)).sum(dim=1)
    rel = _rel_l2(manual, ref)
    assert rel < 2e-2, f"deferred reduce mismatch vs TopkReduce: rel_l2={rel}"

    # Weight-independence: rerunning with different routing weights must not
    # change the exposed rows (they are applied only in the reduce).
    other_w = torch.rand_like(h.topk_weights) + 0.05
    other_w = other_w / other_w.sum(-1, keepdim=True)
    _, ws2 = h.run(apply_topk_in_fc1=False, skip_topk_reduce=True, topk_weights=other_w)
    rows2 = h.combine_quant_view(ws2)
    assert torch.equal(rows, rows2), (
        "combine_quant rows changed with the routing weights; the weight "
        "leaked upstream of the deferred finalize"
    )


def test_apply_topk_in_fc1_false_matches_default(op_harness):
    """Ladder step 3: the transformers route (weights in the reduce) vs the
    deepgemm route (weights folded into fc1-out before its NVFP4 quant).

    Two-part check. With UNIFORM weights the fold is a no-op, so the two
    routes compute the same function and must match to rounding -- this is the
    route-correctness assertion. With random weights the routes genuinely
    differ by where the fc1-out NVFP4 quantization sees the weight; the
    residual is quantization asymmetry, bounded loosely here and adjudicated
    for real magnitudes by the accuracy gate (uniform-w is exact at every SF
    magnitude; random-w rel_l2 ~0.1 at SF=0.125)."""
    h = op_harness
    ones = torch.ones_like(h.topk_weights)
    exact_fold = h.run(apply_topk_in_fc1=True, topk_weights=ones)[0]
    exact_reduce = h.run(apply_topk_in_fc1=False, topk_weights=ones)[0]
    rel_exact = _rel_l2(exact_reduce, exact_fold)
    assert rel_exact < 1e-3, (
        f"transformers route computes a different function even with uniform "
        f"weights (fold is a no-op there): rel_l2={rel_exact}"
    )

    out_fold = h.run(apply_topk_in_fc1=True)[0]
    out_reduce = h.run(apply_topk_in_fc1=False)[0]
    assert torch.isfinite(out_reduce.float()).all()
    rel = _rel_l2(out_reduce, out_fold)
    assert rel < 0.2, f"apply_topk_in_fc1=False diverged from default: rel_l2={rel}"


def test_post_expert_norm_combine_consumes_deferred_triple(op_harness):
    """Ladder step 2 (unit slice): the model's fused Triton finalize over the
    deferred triple (dense rows, identity index map) matches the op-by-op
    reference composition."""
    from tensorrt_llm._torch.models.modeling_musespark_open import (
        post_expert_norm_combine,
        post_expert_norm_rows,
    )

    h = op_harness
    _, ws = h.run(apply_topk_in_fc1=False, skip_topk_reduce=True)
    rows = h.combine_quant_view(ws)
    gemm2_output = rows.reshape(h.max_t * h.top_k, h.hidden)
    idx = torch.arange(h.max_t * h.top_k, dtype=torch.int32, device=h.device)
    norm_weight = torch.randn(h.hidden, dtype=torch.float32, device=h.device) * 0.1
    eps = 1e-6

    fused = post_expert_norm_combine(
        gemm2_output,
        idx,
        h.topk_weights,
        norm_weight,
        eps,
        num_tokens=h.max_t,
        top_k=h.top_k,
        hidden_size=h.hidden,
    )
    normed, valid = post_expert_norm_rows(
        gemm2_output,
        idx,
        norm_weight,
        eps,
        num_tokens=h.max_t,
        top_k=h.top_k,
        hidden_size=h.hidden,
    )
    assert bool(valid.all()), "identity map must mark every slot valid"
    ref = (normed * h.topk_weights.unsqueeze(-1)).sum(dim=1)
    rel = _rel_l2(fused, ref)
    assert rel < 1e-4, f"fused finalize diverged from reference composition: rel_l2={rel}"
