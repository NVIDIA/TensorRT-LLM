# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Rubin FP8 split-K context FMHA (k3_rubin/kernels/k3_context_fmha*.py).

The old-O TMA stage regression of the kernel (with an artificially delayed correction warp),
and the ``k3_context_fmha`` entry through ``run_k3_split_k_context_partition`` against an
FP32 reference for both cumulative-KV-length dtypes, every merge op and per-rank head counts."""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../../.."]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="K3 FP8 context FMHA requires SM107",
)

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"


@pytest.fixture(scope="module", params=[None, 1, 3], ids=["normal", "delay-warp1", "delay-warp3"])
def run_fmha(request, tmp_path_factory):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx import (
        k3_context_fmha,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx.k3_context_fmha import (  # noqa: E501
        RubinFusedMultiHeadAttentionForward,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx.k3_context_fmha_helpers import (  # noqa: E501
        MaskEnum,
    )

    kernel_type = RubinFusedMultiHeadAttentionForward
    if request.param is not None and not hasattr(cute.arch, "nanosleep"):
        # Without cute.arch.nanosleep the instrumented warp would spin on
        # clock64 inside a divergent branch.
        pytest.skip("the delayed-warp instrumentation needs cute.arch.nanosleep")
    if request.param is not None:
        # Delay a non-issuer warp immediately before its O0 wait, after all
        # preceding pipeline releases. O1 can then finish before that wait.
        # Instrument a temporary copy so production needs no testing hook.
        source = Path(k3_context_fmha.__file__).read_text()
        wait = (
            "                    cute.arch.mbarrier_wait(old_o_tma_args[3], phase=old_o_tma_phase)"
        )
        assert source.count(wait) == 1
        delay = (
            f"                    if correction_warp_idx == {request.param}:\n"
            "                        if (row_idx - thread_idx) % self.cta_tiler[0] == 0:\n"
            "                            delay_start = cute.arch.clock64()\n"
            "                            while cute.arch.clock64() - delay_start < 2_000_000:\n"
            "                                cute.arch.nanosleep(sleep_time=1000)\n"
        )
        source_path = tmp_path_factory.mktemp("old_o_delay") / "delayed_fmha.py"
        source_path.write_text(source.replace(wait, delay + wait))
        module_name = f"{k3_context_fmha.__package__}._old_o_delay_{request.param}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        class DelayedFmha(RubinFusedMultiHeadAttentionForward):
            correction_epilog = module.RubinFusedMultiHeadAttentionForward.correction_epilog

        kernel_type = DelayedFmha

    fmha = kernel_type(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (192, 128),
        False,
        MaskEnum.RESIDUAL_MASK,
        False,
        False,
        use_tma_store=False,
        split_k_dims=(128, 64),
    )
    compiled = None

    def run(q, k, k_pe, v, output, stats, op):
        nonlocal compiled
        nq, heads, _ = q.shape
        nk = k.shape[0]
        cu_q = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, nk], dtype=torch.int32, device="cuda")
        scale = 1.0 / math.sqrt(192)
        args = (
            q.view(1, nq, heads, 1, 192),
            k.view(1, nk, heads, 1, 128),
            k_pe.view(1, nk, 1, 1, 64),
            v.view(1, nk, heads, 1, 128),
            output.view(1, nq, heads, 1, 128),
            tuple(cutlass.Int32(x) for x in (1, nq, nq, nk, heads, heads, 192, 128)),
            cu_q,
            cu_k,
            None,
            None,
            stats,
            op,
            cutlass.Float32(scale * math.log2(math.e)),
            cutlass.Float32(scale),
            cutlass.Float32(1.0),
            None,
            None,
            None,
            None,
            None,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            False,
        )
        if compiled is None:
            compile_args = tuple(
                from_dlpack(
                    x, assumed_align=x.element_size() if x.ndim == 1 else 16
                ).mark_layout_dynamic(leading_dim=x.ndim - 1)
                if isinstance(x, torch.Tensor)
                else x
                for x in args
            )
            compiled = cute.compile(
                fmha, *compile_args, options="--opt-level=2 --ptxas-options '--uumn'"
            )
            print(f"Compiled old-O regression kernel: delay_warp={request.param}", flush=True)
        compiled(*args)
        torch.cuda.synchronize()

    return run


@pytest.mark.parametrize("q_len", [256, 257, 512, 8192])
def test_k3_context_old_o_barrier(run_fmha, q_len: int) -> None:
    torch.manual_seed(q_len)
    heads, kv_len = 96, 256

    def fp8(*shape):
        return (torch.randn(*shape, device="cuda") * 0.125).to(torch.float8_e4m3fn)

    q = fp8(q_len, heads, 192)
    k = fp8(kv_len, heads, 128)
    k_pe = fp8(kv_len, 1, 64)
    v = fp8(kv_len, heads, 128)
    full_k = torch.cat((k.float(), k_pe.float().expand(-1, heads, -1)), dim=-1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), full_k) / math.sqrt(192)
    current_max = scores.amax(dim=-1).transpose(0, 1)
    weights = (scores - current_max.transpose(0, 1).unsqueeze(-1)).exp()
    current_sum = weights.sum(dim=-1).transpose(0, 1)
    current_output = torch.einsum("hqk,khd->qhd", weights, v.float()) / current_sum[..., None]
    del scores, weights, full_k

    output = torch.full((q_len, heads, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    stats = torch.full((q_len, heads, 2), torch.nan, device="cuda")
    op = torch.tensor([2], dtype=torch.int64, device="cuda")
    run_fmha(q, k, k_pe, v, output, stats, op)
    torch.testing.assert_close(output.float(), current_output, atol=2e-3, rtol=2e-2)

    op.fill_(1)
    for _ in range(3):
        # Give the old result substantial weight, so a wrong old-O load cannot
        # hide behind a tiny merge coefficient. Each launch starts fresh.
        old_output = (torch.randn_like(output.float()) * 0.125).to(torch.bfloat16)
        old_max = torch.empty_like(current_max).uniform_(-0.5, 0.5)
        old_sum = torch.empty_like(current_sum).uniform_(128, 384)
        output.copy_(old_output)
        stats[..., 0].copy_(old_max)
        stats[..., 1].copy_(old_sum)
        merged_max = torch.maximum(old_max, current_max)
        old_weight = old_sum * (old_max - merged_max).exp()
        current_weight = current_sum * (current_max - merged_max).exp()
        merged_sum = old_weight + current_weight
        expected = (
            old_output.float() * old_weight[..., None] + current_output * current_weight[..., None]
        ) / merged_sum[..., None]

        run_fmha(q, k, k_pe, v, output, stats, op)
        torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            stats[..., 0] + stats[..., 1].log(),
            merged_max + merged_sum.log(),
            atol=3e-4,
            rtol=1e-5,
        )

    op.zero_()
    saved_output, saved_stats = output.clone(), stats.clone()
    run_fmha(q, k, k_pe, v, output, stats, op)
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(stats, saved_stats, atol=0, rtol=0)


def _attention(heads: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_mla_enable=True,
        kv_lora_rank=512,
        num_heads=heads,
        num_kv_heads=heads,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_scaling=1.0,
    )


@pytest.mark.parametrize(
    "q_len,is_causal", [(256, False), (257, False), (512, False), (8192, False), (257, True)]
)
@pytest.mark.parametrize("heads", [96, 48, 24, 12, 6, 3])
def test_k3_context_fmha_cu_kv_dtypes(q_len: int, is_causal: bool, heads: int) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx.k3_context_fmha_op import (  # noqa: E501
        CuteDSLK3ContextFmhaRunner,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__tp1.modeling import (  # noqa: E501
        run_k3_split_k_context_partition,
        supports_k3_fp8_split_k_context,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_context_fmha import (  # noqa: E501
        k3_context_fmha,
    )

    torch.manual_seed(q_len)
    q_lens = [q_len, 17]
    kv_lens = [384, 128]
    nq, nk = sum(q_lens), sum(kv_lens)

    attn = _attention(heads)
    assert supports_k3_fp8_split_k_context(attn)

    def fp8(*shape):
        return (torch.randn(*shape, device="cuda") * 0.125).to(torch.float8_e4m3fn)

    q, k_pe = fp8(nq, heads, 192), fp8(nk, 1, 64)
    # Match the noncompact K/V views produced by the combined KV-B projection.
    kv = fp8(nk, 2 * heads * 128)
    k, v = (part.view(nk, heads, 128) for part in kv.chunk(2, dim=-1))
    cu_q = torch.tensor([0, q_len, nq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, kv_lens[0], nk], dtype=torch.int64, device="cuda")
    output = torch.empty((nq, heads, 128), dtype=torch.bfloat16, device="cuda")
    stats = torch.empty((nq, heads, 2), device="cuda")
    old_output = torch.randn_like(output) * 0.125
    old_max = torch.zeros_like(stats[..., 0])
    old_sum = torch.full_like(old_max, 256)
    expected, expected_lse = [], []
    expected_current, expected_current_lse = [], []
    q_start = k_start = 0
    for q_size, k_size in zip(q_lens, kv_lens):
        full_k = torch.cat(
            (
                k[k_start : k_start + k_size].float(),
                k_pe[k_start : k_start + k_size].float().expand(-1, heads, -1),
            ),
            -1,
        )
        scores = torch.einsum(
            "qhd,khd->hqk", q[q_start : q_start + q_size].float(), full_k
        ) / math.sqrt(192)
        if is_causal:
            # The op uses WINDOW_MASK (top-left aligned), not
            # WINDOW_MASK_INFERENCE (bottom-right aligned).
            mask = (
                torch.arange(k_size, device="cuda")[None, :]
                > torch.arange(q_size, device="cuda")[:, None]
            )
            scores.masked_fill_(mask[None], -torch.inf)
        # Include the old partition directly in a stable reference merge.
        cur_max = scores.amax(-1).transpose(0, 1)
        merged_max = torch.maximum(cur_max, old_max[q_start : q_start + q_size])
        weights = (scores - merged_max.T[..., None]).exp()
        old_weight = old_sum[q_start : q_start + q_size] * (-merged_max).exp()
        merged_sum = weights.sum(-1).T + old_weight
        merged = (
            torch.einsum("hqk,khd->qhd", weights, v[k_start : k_start + k_size].float())
            + old_output[q_start : q_start + q_size].float() * old_weight[..., None]
        ) / merged_sum[..., None]
        expected.append(merged)
        expected_lse.append(merged_max + merged_sum.log())
        current_sum = weights.sum(-1).T
        current_output = (
            torch.einsum("hqk,khd->qhd", weights, v[k_start : k_start + k_size].float())
            / current_sum[..., None]
        )
        if is_causal:
            # Short causal prefixes fit in one 128-key tile. Model the kernel's
            # E4M3 P rounding (prescale=256, unrounded denominator) explicitly:
            # the FP32-only reference misses this error even with 96 heads.
            prefix = min(q_size, 128)
            p = (scores[:, :prefix] - cur_max[:prefix].T[..., None]).exp()
            p_quantized = (p * 256).to(torch.float8_e4m3fn).float() / 256
            current_output[:prefix] = (
                torch.einsum("hqk,khd->qhd", p_quantized, v[k_start : k_start + k_size].float())
                / p.sum(-1).T[..., None]
            )
        expected_current.append(current_output)
        expected_current_lse.append(merged_max + current_sum.log())
        q_start += q_size
        k_start += k_size
    expected, expected_lse = torch.cat(expected), torch.cat(expected_lse)

    def run(op, cu_kv):
        run_k3_split_k_context_partition(
            attn,
            q.flatten(1),
            k.flatten(1),
            k_pe,
            v.flatten(1),
            output.flatten(1),
            stats,
            op,
            cu_q,
            cu_kv,
            max_q_len=max(q_lens),
            max_kv_len=max(kv_lens),
            is_causal=is_causal,
        )
        torch.cuda.synchronize()

    op = torch.ones(2, dtype=torch.int64, device="cuda")
    cache_entries = {}
    for dtype in (torch.int32, torch.int64, torch.int32):
        output.copy_(old_output)
        stats[..., 0].copy_(old_max)
        stats[..., 1].copy_(old_sum)
        run(op, cu_k.to(dtype))
        torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            stats[..., 0] + stats[..., 1].log(), expected_lse, atol=3e-4, rtol=1e-5
        )
        key = (q.dtype, heads, is_causal, dtype)
        compiled = CuteDSLK3ContextFmhaRunner.kernel_cache[key]
        if dtype in cache_entries:
            assert compiled is cache_entries[dtype]
        cache_entries[dtype] = compiled
    assert cache_entries[torch.int32] is not cache_entries[torch.int64]

    # Exercise initialization and skip as well as the merge path above.
    output.fill_(torch.nan)
    stats.fill_(torch.nan)
    op.fill_(2)
    run(op, cu_k)
    torch.testing.assert_close(output.float(), torch.cat(expected_current), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
        stats[..., 0] + stats[..., 1].log(),
        torch.cat(expected_current_lse),
        atol=3e-4,
        rtol=1e-5,
    )
    if is_causal:
        # Same logical heads through the original 96-head specialization must
        # agree exactly, independently of the FP8-vs-FP32 reference tolerance.
        repeats = 96 // heads
        baseline_output = torch.empty((nq, 96, 128), dtype=output.dtype, device="cuda")
        baseline_stats = torch.empty((nq, 96, 2), device="cuda")
        k3_context_fmha(
            torch.cat([q] * repeats, dim=1),
            torch.cat([k] * repeats, dim=1),
            k_pe,
            torch.cat([v] * repeats, dim=1),
            baseline_output,
            cu_q,
            cu_k,
            baseline_stats,
            op,
            max(q_lens),
            max(kv_lens),
            1.0 / math.sqrt(192),
            True,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output, baseline_output[:, :heads], atol=0, rtol=0)
        torch.testing.assert_close(stats, baseline_stats[:, :heads], atol=0, rtol=0)
    saved_output, saved_stats = output.clone(), stats.clone()
    op.zero_()
    run(op, cu_k)
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(stats, saved_stats, atol=0, rtol=0)


def _entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_context_fmha import (  # noqa: E501
        k3_context_fmha,
    )

    return k3_context_fmha


@pytest.mark.parametrize("cell", _entry().CELLS, ids=[c.why[:48] for c in _entry().CELLS])
def test_k3_context_fmha_cells(cell) -> None:
    """The ``k3_context_fmha`` entry's ``.CELLS`` under ``validating`` against its FP32 reference:
    output within the FP8 FMHA band (wider on causal rows over few keys, where the kernel's E4M3
    rounding of P is not averaged out), max + log(sum) of the stats, and rows of op-0 requests
    untouched."""
    import zlib

    from _validating import validating

    op = _entry()
    spec = cell.spec
    heads = 96
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    q_lens, kv_lens = spec["q_lens"], spec["kv_lens"]
    nq, nk = sum(q_lens), sum(kv_lens)

    def fp8(*shape):
        return (torch.randn(*shape, device="cuda") * 0.125).to(torch.float8_e4m3fn)

    q, k_pe = fp8(nq, heads, 192), fp8(nk, 1, 64)
    kv = fp8(nk, 2 * heads * 128)
    k_nope, v = (part.view(nk, heads, 128) for part in kv.chunk(2, dim=-1))
    cu_q = torch.tensor([0, *q_lens], device="cuda").cumsum(0).to(torch.int32)
    kv_dtype = torch.int64 if spec["cu_kv"] == "int64" else torch.int32
    cu_kv = torch.tensor([0, *kv_lens], device="cuda").cumsum(0).to(kv_dtype)
    merge_op = torch.tensor(spec["merge_ops"], dtype=torch.int64, device="cuda")
    output = (torch.randn((nq, heads, 128), device="cuda") * 0.125).to(torch.bfloat16)
    stats = torch.empty((nq, heads, 2), device="cuda")
    stats[..., 0].uniform_(-0.5, 0.5)
    stats[..., 1].uniform_(128, 384)
    args = (
        q,
        k_nope,
        k_pe,
        v,
        output,
        cu_q,
        cu_kv,
        stats,
        merge_op,
        max(q_lens),
        max(kv_lens),
        1.0 / math.sqrt(192),
        spec["causal"],
    )
    expected_out, expected_stats = op.reference(*args)
    with validating(op):
        op(*args)
    torch.cuda.synchronize()
    q0 = 0
    for q_len, merge in zip(q_lens, spec["merge_ops"]):
        rows = slice(q0, q0 + q_len)
        q0 += q_len
        if merge == 0:
            assert torch.equal(output[rows], expected_out[rows]), cell.why
            assert torch.equal(stats[rows], expected_stats[rows]), cell.why
            continue
        got, ref = output[rows].float(), expected_out[rows].float()
        if spec["causal"]:
            # E4M3 P (3 mantissa bits): each probability within 2^-4 relative, so a row over few
            # keys moves by up to 2^-4 of max |V|; rows past one 128-key tile average it out.
            few = min(q_len, 128)
            band = 2.0**-4 * float(v.float().abs().max())
            torch.testing.assert_close(got[:few], ref[:few], atol=band, rtol=0)
            got, ref = got[few:], ref[few:]
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            stats[rows, :, 0] + stats[rows, :, 1].log(),
            expected_stats[rows, :, 0] + expected_stats[rows, :, 1].log(),
            atol=3e-4,
            rtol=1e-5,
        )
