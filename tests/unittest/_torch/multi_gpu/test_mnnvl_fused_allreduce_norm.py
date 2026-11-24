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
"""The CuTe DSL all-reduce + residual + RMSNorm kernels of MNNVLAllReduce against an FP32 reference over
all ranks' inputs: the slot (one-shot) and two-shot launches, the shared-expert gate fold (with or
without the in-kernel MoE finalize), the FP8 output, the FP8 GEMM + all-reduce + norm launches (one-shot and
two-shot), and the TP2 large-message all-reduce + norm (Tp2AllReduceNorm)."""

import operator
import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, MoEAllReduceParams
from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import MnnvlAllReduceNorm
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="the kernels are enabled on SM100/SM103",
    ),
]

H = 4096
EPS = 1e-6
# Kernel launch: the slot kernel up to 128 rows (with the two-shot arithmetic from 64), the two-shot kernel above
# (SLOT_MAX_ROWS, TWO_SHOT_ROWS). ``ar(...)`` takes the CuTe kernels only up to the 1 MiB
# one-shot message (rows * H * tp * 2 bytes), the library op above, so the two-shot rows are also checked through
# ``ar.fused_norm`` directly. Consecutive calls also cycle the Lamport buffers.
ROWS = (1, 7, 32, 63, 64, 96, 200)


def _rand(shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)


def _ref_norm(total, residual, norm_weight):
    """(norm, residual_out) in FP32 of the all-reduced ``total`` [M, H] FP32."""
    res = total + residual.float()
    var = res.pow(2).mean(-1, keepdim=True)
    return res * torch.rsqrt(var + EPS) * norm_weight.float(), res


def _check(got, ref, tag, rtol=2e-2):
    torch.testing.assert_close(
        got.float(), ref, rtol=rtol, atol=rtol * ref.abs().max().item(), msg=lambda m: f"{tag}: {m}"
    )


def _library(ar, x, params):
    """The library op's outputs for this call (MNNVLAllReduce with its CuTe kernels set aside)."""
    fused = ar.fused_norm
    ar.fused_norm = None
    try:
        out = ar(x, all_reduce_params=params)
    finally:
        ar.fused_norm = fused
    return out if isinstance(out, tuple) else (out,)


def _assert_bitwise(got, ref, tag, norm_ulps=0):
    """Outputs equal bit for bit (BF16 / E4M3 compared as integers); with ``norm_ulps``, all but the last (the
    residual, still bit for bit) within that many ulps / E4M3 codes."""
    assert len(got) == len(ref), tag
    for i, (g, r) in enumerate(zip(got, ref)):
        assert g.dtype == r.dtype and g.shape == r.shape, (tag, i)
        as_int = torch.int16 if g.element_size() == 2 else torch.uint8
        if norm_ulps and i < len(got) - 1:
            ulps = (g.view(as_int).int() - r.view(as_int).int()).abs().max().item()
            assert ulps <= norm_ulps, (tag, i, ulps)
        else:
            assert torch.equal(g.view(as_int), r.view(as_int)), (tag, i)


def _library_arithmetic(m, tp, hidden=H):
    """Which arithmetic the library op runs for ``m`` rows ("one-shot" up to its message size, else "two-shot"),
    the one the CuTe kernels run for ``m`` rows (the two-shot's from TWO_SHOT_ROWS), and whether the norm's
    reduction tree is the library's too: always for the two-shot (it replays the library's launch shape), for the
    slot kernel's one-shot only where the library's one-shot norm takes its SLOT_CTAS CTAs per token (else the FP32
    sum of squares is added in another order and the norm may differ by an ulp)."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import TWO_SHOT_ROWS, lib_norm_tree
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES as one_shot

    lib = "one-shot" if m * hidden * tp * 2 <= one_shot else "two-shot"
    cute = "two-shot" if m >= TWO_SHOT_ROWS else "one-shot"
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    same_tree = (
        cute == "two-shot" or lib_norm_tree(m, hidden, sms)[1] == MnnvlAllReduceNorm.SLOT_CTAS
    )
    return lib, cute, same_tree


def _mnnvl(tp, hidden=H, max_num_tokens=max(ROWS)):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    MPI.COMM_WORLD.barrier()
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    ar = MNNVLAllReduce(mapping, torch.bfloat16, max_num_tokens=max_num_tokens, hidden_size=hidden)
    assert MnnvlAllReduceNorm.is_supported(mapping, torch.bfloat16, hidden)
    assert ar.fused_norm is not None, "CuTe DSL kernels not built"
    return ar, rank


def _worker(fn, tp):
    try:
        with torch.inference_mode():
            fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
    return True


def _norm_cases(tp):
    ar, rank = _mnnvl(tp)
    norm_weight = _rand((H,), 1, 0.1) + 1
    scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
    for m in ROWS:
        assert ar.fused_norm.supports_rows(m), m
        xs = [_rand((m, H), 100 * m + r) for r in range(tp)]
        residual = _rand((m, H), 7 * m)
        ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
        )
        norm, res = ar(xs[rank], all_reduce_params=params)
        _check(norm, ref_norm, ("norm", m))
        _check(res, ref_res, ("residual", m))
        norm, res = ar.fused_norm(residual, norm_weight, EPS, x=xs[rank])
        _check(norm, ref_norm, ("kernel norm", m))
        _check(res, ref_res, ("kernel residual", m))
        norm, fp8, res = ar.fused_norm(
            residual, norm_weight, EPS, x=xs[rank], scale=scale, norm_out=True
        )
        _check(norm, ref_norm, ("norm with fp8", m))
        # E4M3 holds 3 mantissa bits: half an ulp is 6.25 % relative.
        _check(fp8.float() * scale, ref_norm.clamp(-448 * 0.05, 448 * 0.05), ("fp8", m), rtol=7e-2)
        # Bit for bit the library op wherever it runs the arithmetic the kernels run for m rows.
        # Elsewhere (the one-shot rows whose library norm takes fewer CTAs per token) the norm within an ulp.
        lib, cute, same_tree = _library_arithmetic(m, tp)
        ulps = 0 if same_tree else 1
        if lib == cute:
            _assert_bitwise(
                ar.fused_norm(residual, norm_weight, EPS, x=xs[rank]),
                _library(ar, xs[rank], params),
                ("library", lib, m),
                ulps,
            )
            fp8_params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
                residual=residual,
                norm_weight=norm_weight,
                eps=EPS,
                scale=scale,
            )
            _assert_bitwise(
                ar.fused_norm(residual, norm_weight, EPS, x=xs[rank], scale=scale, norm_out=True),
                _library(ar, xs[rank], fp8_params),
                ("library fp8", lib, m),
                ulps,
            )
    # Exercise two and 14 tokens per CTA, including a partially filled final CTA. From 15 tokens an SM holds
    # one of these CTAs instead of two, and the dispatch declines.
    wave = 2 * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    assert not ar.fused_norm.supports_rows(14 * wave + 1)
    for m in (wave + 8, 13 * wave + 1):
        assert ar.fused_norm.supports_rows(m), m
        xs = [_rand((m, H), 100 * m + r) for r in range(tp)]
        residual = _rand((m, H), 7 * m)
        ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
        norm, res = ar.fused_norm(residual, norm_weight, EPS, x=xs[rank])
        _check(norm, ref_norm, ("kernel norm", m))
        _check(res, ref_res, ("kernel residual", m))


@pytest.mark.parametrize(("hidden", "tokens_per_cta"), [(2048, 28), (4096, 28), (8192, 14)])
def test_mnnvl_fused_allreduce_norm_compiled_shared_memory(
    hidden: int, tokens_per_cta: int
) -> None:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.memory import get_kernel_smem_size

    from tensorrt_llm._torch.cute_dsl_utils import graph_safe_cute_tensor
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import _example_operands, _rows
    from tensorrt_llm._torch.kernels.blackwell.mnnvl_allreduce_norm import MnnvlAllReduceNormKernel

    kernel = MnnvlAllReduceNormKernel(
        hidden=hidden,
        norm_eps=EPS,
        nranks=2,
        ctas=1,
        use_pdl=False,
        se_fold=False,
        norm_out=True,
        quant=False,
        tpc=tokens_per_cta,
    )

    @cute.kernel
    def store_size(output: cute.Tensor, size: cutlass.Int64) -> None:
        output[0] = size

    @cute.jit
    def shared_memory_size(
        execute: cutlass.Boolean,
        x: cute.Tensor,
        gate: cute.Tensor,
        weight: cute.Tensor,
        scale: cute.Tensor,
        quant: cute.Tensor,
        table: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        # Trace the real kernel without executing a collective on the dummy workspace.
        if execute:
            kernel(
                x,
                x,
                x,
                gate,
                x,
                weight,
                scale,
                x,
                x,
                quant,
                table,
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int32(0),
                cutlass.Int64(0),  # no L2 prefetch
                cutlass.Int64(0),
                cutlass.Int64(0),
                stream,
            )
        store_size(output, cutlass.Int64(get_kernel_smem_size(kernel.kernel_mt))).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    x, gate, weight, scale, quant, table = _example_operands(hidden, 2)
    output = torch.empty((1,), dtype=torch.int64, device="cuda")
    args = (
        cutlass.Boolean(False),
        _rows(x),
        _rows(gate),
        graph_safe_cute_tensor(weight),
        graph_safe_cute_tensor(scale),
        _rows(quant),
        graph_safe_cute_tensor(table),
        graph_safe_cute_tensor(output),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    cute.compile(shared_memory_size, *args)(*args)
    assert output.item() == kernel.mt_smem_bytes(hidden, tokens_per_cta)


def _large_hidden_norm_cases(tp: int) -> None:
    hidden = 8192
    ar, rank = _mnnvl(tp, hidden)
    wave = 2 * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    m = 13 * wave + 1
    assert ar.fused_norm.supports_rows(m), m
    norm_weight = _rand((hidden,), 1, 0.1) + 1
    xs = [_rand((m, hidden), 100 * m + r) for r in range(tp)]
    residual = _rand((m, hidden), 7 * m)
    ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
    norm, res = ar.fused_norm(residual, norm_weight, EPS, x=xs[rank])
    _check(norm, ref_norm, ("large hidden norm", m))
    _check(res, ref_res, ("large hidden residual", m))


def _moe_fold_cases(
    tp: int,
    top_k: int = 10,
    hidden: int = H,
    rows: tuple[int, ...] = ROWS,
    quantize: bool = False,
) -> None:
    ar, rank = _mnnvl(tp, hidden)
    norm_weight = _rand((hidden,), 2, 0.1) + 1
    for m in rows:
        assert ar.supports_moe_fold(m, hidden), m
        routed, finalized, args = [], [], None
        for r in range(tp):
            s = 1000 * m + 10 * r
            # FC2 rows of the permuted (token, k) pairs plus padding rows, with a padded stride.
            fc2 = _rand((m * top_k + 16, hidden + 64), s)[:, :hidden]
            g = torch.Generator(device="cuda").manual_seed(s + 1)
            perm = torch.randperm(m * top_k + 16, generator=g, device="cuda")[: m * top_k]
            perm = perm.view(m, top_k).to(torch.int32)
            topw = torch.softmax(_rand((m, top_k), s + 2, dtype=torch.float32), -1).to(
                torch.bfloat16
            )
            shared = _rand((m, hidden), s + 3)
            gate = _rand((m, 1), s + 4)
            fin = (fc2.float()[perm.long()] * topw.float().unsqueeze(-1)).sum(1)
            routed.append(fin + torch.sigmoid(gate.float()) * shared.float())
            fin_bf16 = fin.to(torch.bfloat16)
            finalized.append(fin_bf16.float() + torch.sigmoid(gate.float()) * shared.float())
            if r == rank:
                args = (fc2, topw, perm, shared, gate, fin_bf16)
        residual = _rand((m, hidden), 9 * m)
        ref_norm, ref_res = _ref_norm(sum(routed), residual, norm_weight)
        fc2, topw, perm, shared, gate, fin_bf16 = args
        params = MoEAllReduceParams(
            expanded_idx_to_permuted_idx=perm,
            expert_scale_factor=topw,
            shared_expert_output=shared,
            shared_expert_gate=gate,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        )
        norm, res = ar(fc2, all_reduce_params=params)
        _check(norm, ref_norm, ("moe fold norm", m))
        _check(res, ref_res, ("moe fold residual", m))
        if quantize:
            params.fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8
            params.scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
            norm, fp8, res = ar(fc2, all_reduce_params=params)
            _check(norm, ref_norm, ("moe fold norm with fp8", m))
            _check(res, ref_res, ("moe fold residual with fp8", m))
            _check(
                fp8.float() * params.scale,
                ref_norm.clamp(-448 * 0.05, 448 * 0.05),
                ("moe fold fp8", m),
                rtol=7e-2,
            )
        # A finalized routed output: only the shared-expert combine is folded.
        ref_norm, ref_res = _ref_norm(sum(finalized), residual, norm_weight)
        params = MoEAllReduceParams(
            shared_expert_output=shared,
            shared_expert_gate=gate,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        )
        norm, res = ar(fin_bf16, all_reduce_params=params)
        _check(norm, ref_norm, ("finalized fold norm", m))
        _check(res, ref_res, ("finalized fold residual", m))


def _moe_fold_lamport_cases(tp, top_k=10):
    """fc2 rows in a Lamport FC2 buffer (0xFFFFFFFF where not written; MoEAllReduceParams.fc2_lamport): the fold
    polls them instead of waiting, gives the plain fold's outputs bit for bit and writes the rows it read back to
    0xFFFFFFFF. Consecutive calls reuse the buffer, as consecutive MoE layers do."""
    ar, rank = _mnnvl(tp)
    norm_weight = _rand((H,), 4, 0.1) + 1
    cap = 128 * top_k + 16
    buf = torch.full((cap, H), -1, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    for m in (1, 7, 32, 63, 64, 96, 128):
        assert ar.supports_moe_fold(m, H), m
        s = 2000 * m + 10 * rank
        g = torch.Generator(device="cuda").manual_seed(s + 1)
        perm = torch.randperm(m * top_k + 16, generator=g, device="cuda")[: m * top_k]
        perm = perm.view(m, top_k).to(torch.int32)
        rows = perm.flatten().long()
        fc2 = _rand((m * top_k + 16, H), s)
        topw = torch.softmax(_rand((m, top_k), s + 2, dtype=torch.float32), -1).to(torch.bfloat16)
        shared = _rand((m, H), s + 3)
        gate = _rand((m, 1), s + 4)
        residual = _rand((m, H), 9 * m)
        outs = []
        for lamport in (False, True):
            src = fc2
            if lamport:
                buf[rows] = fc2[rows]
                src = buf
            params = MoEAllReduceParams(
                expanded_idx_to_permuted_idx=perm,
                expert_scale_factor=topw,
                shared_expert_output=shared,
                shared_expert_gate=gate,
                residual=residual,
                norm_weight=norm_weight,
                eps=EPS,
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                fc2_lamport=lamport,
            )
            outs.append(ar(src, all_reduce_params=params))
        torch.cuda.synchronize()
        for plain, polled, tag in zip(outs[0], outs[1], ("norm", "residual")):
            assert torch.equal(plain.view(torch.int16), polled.view(torch.int16)), (
                "lamport",
                tag,
                m,
            )
        assert bool((buf.view(torch.int32) == -1).all()), ("lamport rows not reset", m)
    # A Lamport buffer needs the un-finalized fold; the call refuses before any launch.
    for fold in (None, (shared, shared, gate)):
        with pytest.raises(ValueError, match="un-finalized MoE fold"):
            ar.fused_norm(residual, norm_weight, EPS, x=shared, moe_fold=fold, fc2_lamport=True)


def _large_moe_fold_cases(tp: int) -> None:
    wave = 2 * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    _moe_fold_cases(tp, hidden=8192, rows=(13 * wave + 1,), quantize=True)


def _gemm_norm_cases(tp, k=1024):
    from tensorrt_llm._torch.modules.low_m_mma_gemm import default_tactic

    ar, rank = _mnnvl(tp)
    assert ar.gemm_norm is not None, "GEMM + all-reduce + norm kernel not built"
    norm_weight = _rand((H,), 3, 0.1) + 1
    weight_scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    input_scale = torch.tensor([4.0 / 448], dtype=torch.float32, device="cuda")
    for m in (1, 5, 8, 16, 32):
        outs, args = [], None
        for r in range(tp):
            x = _rand((m, k), 77 * m + r)
            w = _rand((H, k), 5000 + r, 4.0).to(torch.float8_e4m3fn)
            xq = (x.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            outs.append((xq.float() @ w.float().t()) * input_scale * weight_scale)
            if r == rank:
                args = (x, w)
        residual = _rand((m, H), 11 * m)
        ref_norm, ref_res = _ref_norm(sum(outs), residual, norm_weight)
        x, w = args
        # fp8_payload only changes two-shot calls; this one-launch kernel exchanges BF16 either way.
        for fp8_payload in (False, True):
            params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=EPS,
                fp8_payload=fp8_payload,
            )
            out = ar.gemm_allreduce_norm(
                x, w, weight_scale, input_scale, params, default_tactic(m, H, k)[:3]
            )
            assert out is not None, ("GEMM + all-reduce + norm declined", m, fp8_payload)
            norm, res = out
            _check(norm, ref_norm, ("gemm norm", m, fp8_payload))
            _check(res, ref_res, ("gemm residual", m, fp8_payload))


def _gemm_twoshot_cases(
    tp, k=1024, max_num_tokens=max(ROWS), rows=(9, 17, 33, 56, 64, 65, 96, 128), required=False
):
    """The tcgen05 FP8 GEMM + two-shot all-reduce + norm launch (GemmTwoShotAllReduceNorm) over its rows, with and
    without the FP8 output, and an E4M3 or BF16 input."""
    ar, rank = _mnnvl(tp, max_num_tokens=max_num_tokens)
    assert ar.gemm_twoshot is not None, "GEMM + two-shot all-reduce + norm kernel not built"
    norm_weight = _rand((H,), 5, 0.1) + 1
    weight_scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    input_scale = torch.tensor([4.0 / 448], dtype=torch.float32, device="cuda")
    scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
    probe = torch.empty((H, k), dtype=torch.float8_e4m3fn, device="cuda")
    for m in rows:
        if not ar.gemm_twoshot.takes(m, probe):
            assert not required, ("GEMM + two-shot refused", m)
            continue
        outs, args = [], None
        for r in range(tp):
            x = _rand((m, k), 91 * m + r)
            w = _rand((H, k), 6000 + r, 4.0).to(torch.float8_e4m3fn)
            xq = (x.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            outs.append((xq.float() @ w.float().t()) * input_scale * weight_scale)
            if r == rank:
                args = (x, xq, w)
        residual = _rand((m, H), 13 * m)
        ref_norm, ref_res = _ref_norm(sum(outs), residual, norm_weight)
        x, xq, w = args
        for inp in (xq, x):
            params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=EPS,
            )
            out = ar.gemm_twoshot_allreduce_norm(inp, w, weight_scale, input_scale, params)
            assert out is not None, ("GEMM + two-shot declined", m, inp.dtype)
            norm, res = out
            _check(norm, ref_norm, ("gemm two-shot norm", m, inp.dtype))
            _check(res, ref_res, ("gemm two-shot residual", m, inp.dtype))
        # Bit for bit the unfused calls: the projection on cuBLAS (as FP8QDQLinearMethod.apply), then the library
        # op, which runs two-shot at every row count the kernel takes.
        # Looked up by name: cloudpickle pickles this module by value and cannot pickle the torch.ops module that an
        # attribute access (or import) of ``ops`` here would make it carry along.
        cublas_scaled_mm = operator.attrgetter("ops.trtllm.cublas_scaled_mm")(torch)
        partial = cublas_scaled_mm(
            xq,
            w.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=torch.bfloat16,
        )
        assert _library_arithmetic(m, tp)[0] == "two-shot", m
        _assert_bitwise(
            ar.gemm_twoshot_allreduce_norm(xq, w, weight_scale, input_scale, params),
            _library(ar, partial, params),
            ("gemm two-shot vs cuBLAS + library", m),
        )
        # The kernel exchanges BF16: an E4M3-payload call stays on the library two-shot.
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            fp8_payload=True,
        )
        assert ar.gemm_twoshot_allreduce_norm(xq, w, weight_scale, input_scale, params) is None, m
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            scale=scale,
        )
        norm, fp8, res = ar.gemm_twoshot_allreduce_norm(xq, w, weight_scale, input_scale, params)
        _check(norm, ref_norm, ("gemm two-shot norm with fp8", m))
        _check(fp8.float() * scale, ref_norm.clamp(-448 * 0.05, 448 * 0.05), ("fp8", m), rtol=7e-2)


def _gemm_twoshot_large_workspace_cases(tp, k=1024):
    """The GEMM + two-shot launch on a Lamport buffer of 2 GiB (max_num_tokens 131072 at H 4096), whose size and
    offsets the kernel must read as unsigned, over consecutive calls that cycle through all three buffers."""
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES as one_shot
    from tensorrt_llm._torch.kernels.blackwell import gemm_twoshot_allreduce_norm as kern

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    min_rows = one_shot // (H * tp * 2) + 1
    fits = [m for m in range(1, kern.MAX_ROWS + 1) if kern.supports_rows(m, H, tp, sms, min_rows)]
    if not fits:  # TP2: no row takes the kernel
        return
    rows = (fits[0], fits[len(fits) // 2], fits[-1])
    _gemm_twoshot_cases(tp, k, max_num_tokens=131072, rows=rows, required=True)


def _shifted(t, offset):
    """A copy of contiguous ``t`` at ``offset`` bytes past a 128-byte boundary."""
    buf = torch.empty(t.numel() * t.element_size() + 128, dtype=torch.uint8, device="cuda")
    assert buf.data_ptr() % 128 == 0
    view = buf[offset : offset + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
    view.copy_(t)
    return view


def _gemm_twoshot_alignment_cases(tp, k=1024):
    """The GEMM + two-shot launch takes an E4M3 x and W at any 16-byte aligned address (their TMA descriptors'
    requirement) with the aligned call's outputs bit for bit, and declines (before any launch, on every rank) an x or
    W that is not 16-byte aligned."""
    ar, rank = _mnnvl(tp)
    assert ar.gemm_twoshot is not None, "GEMM + two-shot all-reduce + norm kernel not built"
    probe = torch.empty((H, k), dtype=torch.float8_e4m3fn, device="cuda")
    fits = [m for m in range(1, max(ROWS) + 1) if ar.gemm_twoshot.takes(m, probe)]
    if not fits:  # TP2: no row takes the kernel
        return
    m = fits[0]
    norm_weight = _rand((H,), 5, 0.1) + 1
    weight_scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    input_scale = torch.tensor([4.0 / 448], dtype=torch.float32, device="cuda")
    xq = (_rand((m, k), 91 * m + rank).float() / input_scale).clamp(-448, 448)
    xq = xq.to(torch.float8_e4m3fn)
    w = _rand((H, k), 6000 + rank, 4.0).to(torch.float8_e4m3fn)
    residual = _rand((m, H), 13 * m)
    params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=norm_weight,
        eps=EPS,
    )
    ref = ar.gemm_twoshot_allreduce_norm(xq, w, weight_scale, input_scale, params)
    assert ref is not None, m
    for x_off, w_off in ((16, 0), (0, 16), (48, 80)):
        got = ar.gemm_twoshot_allreduce_norm(
            _shifted(xq, x_off), _shifted(w, w_off), weight_scale, input_scale, params
        )
        assert got is not None, ("16-byte aligned operands declined", m, x_off, w_off)
        _assert_bitwise(got, ref, ("16-byte aligned operands", x_off, w_off))
    for x_off, w_off in ((8, 0), (0, 8)):
        x8, w8 = _shifted(xq, x_off), _shifted(w, w_off)
        assert not ar.gemm_twoshot.supports(x8, w8), (x_off, w_off)
        assert ar.gemm_twoshot_allreduce_norm(x8, w8, weight_scale, input_scale, params) is None


def test_gemm_twoshot_rows() -> None:
    """The two-shot GEMM kernel takes a call from the library's first two-shot row count (one more than the one-shot
    message size holds: 17 at TP8 / H 4096) and only where every owner item has a thread: all of its rows at TP8 /
    TP16, fewer at TP4 / TP2 (a rank owns more slices)."""
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES as one_shot
    from tensorrt_llm._torch.kernels.blackwell import gemm_twoshot_allreduce_norm as kern

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    min_rows = {tp: one_shot // (H * tp * 2) + 1 for tp in (2, 4, 8, 16)}
    for tp, m in min_rows.items():
        assert (m - 1) * H * tp * 2 <= one_shot < m * H * tp * 2
    if one_shot == 64 * 1024 * 8 * 2:
        assert min_rows == {2: 65, 4: 33, 8: 17, 16: 9}
    fits = {
        tp: [
            m
            for m in range(1, kern.MAX_ROWS + 1)
            if kern.supports_rows(m, H, tp, sms, min_rows[tp])
        ]
        for tp in (2, 4, 8, 16)
    }
    for tp in (8, 16):
        assert fits[tp] == [
            m for m in range(min_rows[tp], kern.MAX_ROWS + 1) if kern._replayable(m, H, sms)
        ]
    assert not fits[2] and all(m <= 64 for m in fits[4])
    assert not kern.supports_hidden(2 * H, 8, sms)
    assert not kern.supports_hidden(H // 2, 8, sms)


def test_l2_prefetch_args() -> None:
    """The L2 prefetch splits a weight into 16-byte multiple slices that cover it, and is off for a weight whose
    address or size is not a 16-byte multiple."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import _l2pf_args

    weight = torch.empty((3 * 1024 + 8,), dtype=torch.bfloat16, device="cuda")
    l2pf, ptr, nbytes, per = _l2pf_args(weight, 7)
    assert l2pf and ptr == weight.data_ptr() and nbytes == weight.numel() * 2
    assert per % 16 == 0 and 7 * per >= nbytes > 6 * per
    assert _l2pf_args(None, 7) == (False, 0, 0, 0)
    assert _l2pf_args(weight, 0) == (False, 0, 0, 0)
    assert _l2pf_args(weight[:9], 7) == (False, 0, 0, 0)
    assert _l2pf_args(weight[1:9], 7) == (False, 0, 0, 0)


def test_slot_kernel_takes_every_supported_hidden_size() -> None:
    """Every hidden size MnnvlAllReduceNorm.is_supported accepts builds the slot kernel at every row count it takes
    (from TWO_SHOT_ROWS with the two-shot CTA split, which stays within 1024 threads per CTA as the library's), and
    the others are rejected up front."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import (
        SLOT_MAX_ROWS,
        TWO_SHOT_ROWS,
        _ctas_per_token,
    )
    from tensorrt_llm._torch.kernels.blackwell.mnnvl_slot_allreduce_norm import (
        MnnvlSlotAllReduceNormKernel,
    )

    mapping = Mapping(world_size=8, tp_size=8, rank=0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert _ctas_per_token(SLOT_MAX_ROWS, 16384, sms) == 2
    assert not MnnvlAllReduceNorm.is_supported(mapping, torch.bfloat16, 6144)
    for hidden in range(2048, 65536 + 1, 2048):
        supported = MnnvlAllReduceNorm.is_supported(mapping, torch.bfloat16, hidden)
        built = True
        for m in range(1, SLOT_MAX_ROWS + 1):
            ts = _ctas_per_token(m, hidden, sms) if m >= TWO_SHOT_ROWS else 0
            try:
                kernel = MnnvlSlotAllReduceNormKernel(
                    hidden=hidden,
                    norm_eps=EPS,
                    nranks=8,
                    ctas=MnnvlAllReduceNorm.SLOT_CTAS,
                    use_pdl=True,
                    se_fold=False,
                    norm_out=True,
                    quant=False,
                    region_bytes=1,
                    ts_ctas=ts,
                )
            except ValueError:
                built = False
                continue
            # The launch reserves the kernel's shared memory: warp sums, the cluster sums (one per warp of the
            # token with the two-shot arithmetic), flags and mbarrier; up to hidden 16384 they fit in the 512-byte
            # floor smem_bytes keeps for the measured launches.
            nsum = kernel.ctas * kernel.num_warps if ts else kernel.ctas
            assert kernel.smem_bytes() >= 4 * (kernel.num_warps + nsum) + 32 + 8, (hidden, m)
            assert hidden > 16384 or kernel.smem_bytes() == 512, (hidden, m)
        warps = hidden // 2048
        assert supported == (warps & (warps - 1) == 0), hidden
        assert not supported or built, hidden


def _tp2_cases(tp, max_tokens=4096):
    """Tp2AllReduceNorm (TP2 without MNNVL, large messages): residual RMSNorm with and without the FP8 output, at
    hidden sizes of 8 and 16 reference warps."""
    from tensorrt_llm._torch.distributed.tp2_allreduce import Tp2AllReduceNorm

    if tp != 2:
        return
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    assert not Tp2AllReduceNorm.is_supported(mapping, H + 128, max_tokens)
    assert not Tp2AllReduceNorm.is_supported(mapping, 4 * H, max_tokens)
    for hidden in (H, H // 2):
        _tp2_hidden_cases(tp, rank, mapping, hidden, max_tokens)


def _tp2_hidden_cases(tp, rank, mapping, hidden, max_tokens):
    from tensorrt_llm._torch.distributed.ll_exchange import _PeerBuffers
    from tensorrt_llm._torch.distributed.tp2_allreduce import Tp2AllReduceNorm

    assert Tp2AllReduceNorm.is_supported(mapping, hidden, max_tokens)
    ar = Tp2AllReduceNorm(mapping, hidden, max_tokens)
    # Peer memory without a multicast object (the kernel reads only the two unicast mappings): IPC inside a node.
    assert isinstance(ar._bufs, _PeerBuffers) and ar._bufs.transport == "IPC"
    norm_weight = _rand((hidden,), 6, 0.1) + 1
    scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
    for m in (Tp2AllReduceNorm.MIN_TOKENS, 3001, max_tokens):
        xs = [_rand((m, hidden), 300 * m + r) for r in range(tp)]
        residual = _rand((m, hidden), 17 * m)
        ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
        op = AllReduceFusionOp.RESIDUAL_RMS_NORM
        assert ar.supports(xs[rank], op, residual, norm_weight)
        assert not ar.supports(xs[rank], op, residual.float(), norm_weight)
        assert not ar.supports(xs[rank], op, residual[:-1], norm_weight)
        assert not ar.supports(xs[rank], op, residual, norm_weight.float())
        norm, res = ar(
            xs[rank], residual, norm_weight, EPS, AllReduceFusionOp.RESIDUAL_RMS_NORM, None
        )
        _check(norm, ref_norm, ("tp2 norm", m))
        _check(res, ref_res, ("tp2 residual", m))
        fp8, res = ar(
            xs[rank],
            residual,
            norm_weight,
            EPS,
            AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
            scale,
        )
        _check(
            fp8.float() * scale, ref_norm.clamp(-448 * 0.05, 448 * 0.05), ("tp2 fp8", m), rtol=7e-2
        )
        _check(res, ref_res, ("tp2 fp8 residual", m))
    # The table, dummy-scale and norm-weight views are built once and reused by every later call; another
    # layer's norm weight gets its own view.
    assert len(ar._norm_weight_views) == 1
    other_weight = _rand((hidden,), 7, 0.1) + 1
    m = Tp2AllReduceNorm.MIN_TOKENS
    xs = [_rand((m, hidden), 11 + r) for r in range(tp)]
    residual = _rand((m, hidden), 13)
    ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, other_weight)
    norm, res = ar(xs[rank], residual, other_weight, EPS, AllReduceFusionOp.RESIDUAL_RMS_NORM, None)
    _check(norm, ref_norm, ("tp2 norm, second weight", m))
    _check(res, ref_res, ("tp2 residual, second weight", m))
    assert len(ar._norm_weight_views) == 2
    big = _rand((max_tokens + 1, hidden), 1)
    assert not ar.supports(big, AllReduceFusionOp.RESIDUAL_RMS_NORM, big, norm_weight)


def _meta_init_cases(tp):
    """Kernels built under MetaInitMode, as the model loader builds the model, hold no meta tensor and run."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import (
        GemmAllReduceNorm,
        _LamportTable,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    ar, rank = _mnnvl(tp)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)["comm"]
    with MetaInitMode():
        table = _LamportTable(mapping, torch.bfloat16)
        norm_kernel = MnnvlAllReduceNorm(mapping, comm, H, table)
        gemm_kernel = GemmAllReduceNorm(mapping, H, table)
    for kernel in (norm_kernel, gemm_kernel):
        meta = [k for k, v in vars(kernel).items() if isinstance(v, torch.Tensor) and v.is_meta]
        assert not meta, (type(kernel).__name__, meta)
    m = 7
    norm_weight = _rand((H,), 3, 0.1) + 1
    xs = [_rand((m, H), 300 + r) for r in range(tp)]
    residual = _rand((m, H), 301)
    ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
    norm, res = norm_kernel(residual, norm_weight, EPS, x=xs[rank])
    _check(norm, ref_norm, ("meta-init norm", m))
    _check(res, ref_res, ("meta-init residual", m))


def _opt_in_cases(tp):
    """AllReduce builds the kernels only where the model declares them (extra_attrs
    ``allreduce_mnnvl_fused_norm``); otherwise its MNNVL path keeps the library op."""
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.utils import model_extra_attrs
    from tensorrt_llm.functional import AllReduceStrategy

    _, rank = _mnnvl(tp)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    shape = {"allreduce_max_num_tokens": max(ROWS), "allreduce_hidden_size": H}
    for declared in (False, True):
        with model_extra_attrs({**shape, "allreduce_mnnvl_fused_norm": declared}):
            ar = AllReduce(mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        assert ar.mnnvl_allreduce is not None
        assert (ar.mnnvl_allreduce.fused_norm is not None) == declared


CASES = {
    "norm": _norm_cases,
    "large_hidden_norm": _large_hidden_norm_cases,
    "moe_fold": _moe_fold_cases,
    "moe_fold_lamport": _moe_fold_lamport_cases,
    "large_moe_fold": _large_moe_fold_cases,
    "gemm_norm": _gemm_norm_cases,
    "gemm_twoshot": _gemm_twoshot_cases,
    "gemm_twoshot_large_workspace": _gemm_twoshot_large_workspace_cases,
    "gemm_twoshot_alignment": _gemm_twoshot_alignment_cases,
    "tp2": _tp2_cases,
    "meta_init": _meta_init_cases,
    "opt_in": _opt_in_cases,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_fused_allreduce_norm(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)


def _tp2_setup_failure(tp, error_type=RuntimeError):
    """A failure past the peer buffers' collective setup on one rank (the pointer-table init is not collective) fails
    Tp2AllReduceNorm's construction on every rank with RuntimeError, naming the failed rank's cause, whatever
    ``error_type`` that rank raised. Returns this rank's outcome as text (exceptions stay in the worker)."""
    from tensorrt_llm._torch.distributed.ll_exchange import _PeerBuffers
    from tensorrt_llm._torch.distributed.tp2_allreduce import Tp2AllReduceNorm

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    real = _PeerBuffers.__dict__["create"]

    @classmethod
    def create(cls, *args, **kwargs):
        bufs = real.__func__(cls, *args, **kwargs)
        if rank == 1:
            raise error_type("injected pointer-table failure")
        return bufs

    _PeerBuffers.create = create
    try:
        Tp2AllReduceNorm(mapping, H, 4096)
        outcome = "constructed"
    except RuntimeError as e:
        outcome = str(e)
    except BaseException as e:  # noqa: BLE001 - reported to the parent as text
        outcome = f"unexpected {type(e).__name__}: {e}"
    finally:
        _PeerBuffers.create = real
    MPI.COMM_WORLD.barrier()
    return outcome


@pytest.mark.timeout(300)
# MemoryError: a failure other than RuntimeError (e.g. a host allocation) must not leave the peer waiting.
@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_tp2_setup_failure_on_one_rank(mpi_pool_executor, error_type):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    outcomes = list(mpi_pool_executor.map(_tp2_setup_failure, [2] * 2, [error_type] * 2))
    assert all("rank 1: injected pointer-table failure" in o for o in outcomes), outcomes
