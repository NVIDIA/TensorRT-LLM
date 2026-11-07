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
"""Pushed MNNVL all-reduce (``AllReduce.push_target`` / ``finish_push``, the FP8 block-scale
low-M GEMM's push epilogue, the copy push, the small-batch MoE kernel's output push): bitwise
equal to the C++ MNNVL all-reduce, with pushed and C++ calls interleaved on one workspace, eagerly
and in CUDA graph replays; also where the decode attention-residual mix finishes the call itself
(``MnnvlPendingResult``)."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from _torch.helpers import bf16_bits_differ as _differ
from _torch.helpers import mnnvl_torch_push as _torch_push
from _torch.helpers import (
    mpi_worker_env,
    r128c4_scale_offsets,
    run_small_batch_moe,
    small_batch_moe_operands,
)
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def _special(m, h, gen):
    """BF16 ``[m, h]`` with -0, +0, denormal, large, inf and NaN lanes."""
    x = torch.randn(m, h, device="cuda", generator=gen) * torch.exp2(
        torch.randint(-20, 20, (m, 1), device="cuda", generator=gen).float()
    )
    bits = x.to(torch.bfloat16).view(torch.int16)
    r = torch.rand(m, h, device="cuda", generator=gen)
    bits[r < 0.05] = -32768
    bits[(r >= 0.05) & (r < 0.08)] = 0
    den = (r >= 0.08) & (r < 0.11)
    bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
        torch.int16
    )
    bits[(r >= 0.11) & (r < 0.115)] = 0x7F80
    bits[(r >= 0.115) & (r < 0.117)] = 0x7FC1
    return bits.view(torch.bfloat16)


def _input(m, h, seed, rank, kind=0):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    if kind == 0:
        return (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    if kind == 1:
        return _special(m, h, gen)
    if kind == 3:
        # Denormals of either sign, no NaN (whose payload a torch add and a kernel add round
        # differently).
        mag = torch.randint(1, 127, (m, h), device="cuda", generator=gen).to(torch.int16)
        sign = torch.randint(0, 2, (m, h), device="cuda", generator=gen).to(torch.int16) << 15
        return (mag | sign).view(torch.bfloat16)
    # Denormals only: the sums stay denormal (no flush to zero anywhere).
    bits = torch.randint(-127, 127, (m, h), device="cuda", generator=gen).to(torch.int16)
    return bits.view(torch.bfloat16)


def _left_pending(m, h, k, world, in_place_max_rows):
    """Whether an o_proj ``[h, k]`` of ``m`` rows leaves its all-reduce pending for the mix after
    attention: where its GEMM pushes, up to ``in_place_max_rows`` rows (either transport)."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    return low_m.all_reduce_push_applies(m, h, k) and m <= in_place_max_rows


def _projection(plain, x, gate=None):
    """``plain`` (no all-reduce) on ``x`` (``* gate``) as the pushing o_proj computes it: above
    the GEMM's range where its push is selected (``all_reduce_push_max_m_for``), the mid-M GEMM
    instead of the default GEMM."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    m, (n, k) = x.shape[0], plain.weight.shape
    if low_m.max_m_for(n, k) < m and low_m.all_reduce_push_applies(m, n, k):
        y = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        rows = x if gate is None else x * gate
        return low_m.run_low_m_gemm(rows, plain.weight, plain._low_m_weight_scale, y)
    return plain(x) if gate is None else plain.forward_gated(x, gate)


def _scale_rows(xs, m, h):
    """The UE8M0 bytes of the first ``m`` rows of R128c4 scales of ``[*, h]`` rows."""
    return xs[r128c4_scale_offsets(m, h, xs.device)].contiguous()


def _result(out):
    from tensorrt_llm._torch.distributed import MnnvlWorkspaceResult

    return out.materialize() if isinstance(out, MnnvlWorkspaceResult) else out


def _mxfp8_operand(x):
    """``(xq, xs)`` of ``x`` with the scales in the R128c4 layout (rows padded to 128), the
    operand ``forward_prequantized`` takes."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    m, k = x.shape
    xq, xs_rows = low_m.mxfp8_quantize_reference(x)
    xs = torch.full((-(-m // 128) * 128 * (k // 32),), 0x7F, dtype=torch.uint8, device="cuda")
    xs[r128c4_scale_offsets(m, k, "cuda")] = xs_rows
    return xq, xs


def _weight(n, k, seed):
    """FP8 weight and its scale as ``_Fp8BlockScaleWeightReadLinear`` loads them."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    return _Fp8BlockScaleWeightReadLinear.quantize_weight(weight)


def _gemm_weight(n, k, seed):
    """FP8 weight and the DeepGEMM-packed scale the low-M GEMM reads."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    w, w_scale = _weight(n, k, seed)
    if torch.cuda.get_device_capability() == (10, 7):
        w_scale = low_m.pack_block_scale_from_mxfp8_layout(w_scale, n, k)
    return w, w_scale


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank(world_size):
    from tensorrt_llm._torch.distributed import (
        AllReduce,
        AllReduceFusionOp,
        AllReduceParams,
        AllReduceStrategy,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        # Only where forward runs the plain sum.
        for params in (
            AllReduceParams(enable_allreduce=False),
            AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM),
        ):
            assert allreduce.push_target(8, 7168, all_reduce_params=params) is None
        assert allreduce.push_target(8, 7168, all_reduce_params=AllReduceParams()) is not None
        bad = 0

        # A pushed input (special values) finished against the C++ all-reduce, with C++ calls
        # in between; one-shot and two-shot sizes, in place where the transport allows it, and
        # left pending (finished by the finish kernel into a tensor); pushed by a torch stand-in
        # and by the copy kernel.
        for m, h in (
            (1, 7168),
            (9, 7168),
            (14, 7168),
            (64, 7168),
            (300, 7168),
            (14, 3584),
            (5, 64),
        ):
            for kind in (0, 1, 2):
                x = _input(m, h, 3 * m + kind, rank, kind)
                ref = allreduce(x)
                for in_place, pending in ((False, False), (True, False), (False, True)):
                    target = allreduce.push_target(
                        m, h, in_place=in_place, consumer_finishes=pending
                    )
                    assert target.consumer_finishes == pending
                    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
                    _torch_push(workspace, target, x, rank)
                    bad += _differ(ref, _result(allreduce.finish_push(target)))
                    bad += _differ(ref, allreduce(x))
                    target = allreduce.push_target(m, h, consumer_finishes=pending)
                    allreduce.push_input(target, x)
                    bad += _differ(ref, _result(allreduce.finish_push(target)))

        # GEMM push epilogue (plain and gated; split-K at (7168, 1536) on SM100 / SM103; the
        # mid-M kernel above the low-M kernel's range) against the GEMM followed by the C++
        # all-reduce.
        for n, k in ((7168, 768), (7168, 1536)):
            w, w_scale = _gemm_weight(n, k, 11 + rank)
            for m in (1, 7, 14, 32, 64, 65, 96, 120, 128, 144, 192, 200):
                if not low_m.all_reduce_push_applies(m, n, k):
                    continue
                x = _input(m, k, 5 * m, rank)
                gate = _input(m, k, 7 * m, rank)
                gated = low_m.all_reduce_push_applies(m, n, k, gated=True)
                gates = (None, gate) if gated and low_m.gate_fits(x, gate, n) else (None,)
                for g in gates:
                    y = low_m.run_low_m_gemm(
                        x,
                        w,
                        w_scale,
                        torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
                        gate=g,
                    )
                    ref = allreduce(y)
                    for in_place, pending in ((False, False), (True, False), (False, True)):
                        target = allreduce.push_target(
                            m, n, in_place=in_place, consumer_finishes=pending
                        )
                        low_m.run_low_m_gemm_all_reduce_push(
                            x, w, w_scale, target.view, target.one_shot, gate=g
                        )
                        bad += _differ(ref, _result(allreduce.finish_push(target)))
                    if g is None:
                        # The same push on the MXFP8 operand of ``x`` (low-M and mid-M kernels).
                        xq, xs = _mxfp8_operand(x)
                        target = allreduce.push_target(m, n)
                        low_m.run_low_m_gemm_prequantized_all_reduce_push(
                            xq, xs, w, w_scale, target.view, target.one_shot
                        )
                        bad += _differ(ref, _result(allreduce.finish_push(target)))

        # One CUDA graph with pushed and C++ calls on the workspace, replayed with new inputs.
        n, k, m = 7168, 768, 14
        w, w_scale = _gemm_weight(n, k, 99 + rank)
        xs = [_input(m, k, 700 + i, rank) for i in range(4)]
        ys = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

        def calls():
            res = []
            for i in range(4):
                if i % 2 == 0:
                    target = allreduce.push_target(m, n)
                    low_m.run_low_m_gemm_all_reduce_push(
                        xs[i], w, w_scale, target.view, target.one_shot
                    )
                    res.append(allreduce.finish_push(target))
                else:
                    low_m.run_low_m_gemm(xs[i], w, w_scale, ys[i])
                    res.append(allreduce(ys[i]))
            return res

        # A C++ call grows the workspace; as in the CUDA graph runners, one eager run of the
        # captured calls precedes the capture and builds the new workspace's push view.
        allreduce(torch.zeros(2048, n, device="cuda", dtype=torch.bfloat16))
        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i in range(4):
                xs[i].copy_(_input(m, k, 900 + 10 * rep + i, rank))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(4):
                y = low_m.run_low_m_gemm(
                    xs[i], w, w_scale, torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                )
                bad += _differ(allreduce(y), results[i])
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank_o_proj(world_size):
    """An o_proj that owns its all-reduce (pushed, finished by the mix after attention) against
    the projection followed by the C++ all-reduce and the tensor mix; also its prequantized
    forward, whose output a copy kernel pushes."""
    from torch import nn

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy, MnnvlPendingResult
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        hidden = 7168
        gen = torch.Generator(device="cuda").manual_seed(3)
        proj = nn.Linear(hidden, 1, bias=False, dtype=torch.bfloat16, device="cuda")
        proj.weight.data.copy_(torch.randn(1, hidden, device="cuda", generator=gen) * 0.05)
        norm = kimi.KimiK3RMSNorm(hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda")
        norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        out_norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16).cuda()
        out_norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        bad = 0
        for k in (768, 1536):
            w, w_scale = _weight(hidden, k, 21 + rank)
            plain = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden)
            plain.post_load_weights()
            for in_place_max_rows in (0, 64, 128):
                o_proj = kimi._Fp8BlockScaleWeightReadLinear(
                    w, w_scale, hidden, allreduce, in_place_max_rows
                )
                o_proj.post_load_weights()
                for m in (1, 8, 14, 32, 64, 65, 96, 120, 128, 192, 200):
                    x = _input(m, k, 9 * m, rank)
                    gate = _input(m, k, 13 * m, rank)
                    prefix = torch.randn(m, hidden, device="cuda", generator=gen).to(torch.bfloat16)
                    block = torch.randn(4, m, hidden, device="cuda", generator=gen).to(
                        torch.bfloat16
                    )
                    for g in (None, gate):
                        ref_out = allreduce(_projection(plain, x, g))
                        ref = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, ref_out, block, proj, norm, out_norm
                        )
                        out = o_proj(x) if g is None else o_proj.forward_gated(x, g)
                        # Pushed calls are left to the mix up to the row count asked for.
                        assert isinstance(out, MnnvlPendingResult) == _left_pending(
                            m, hidden, k, world_size, in_place_max_rows
                        )
                        got = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, out, block, proj, norm, out_norm
                        )
                        for a, b in zip(ref, got):
                            bad += _differ(a, b)
                    if o_proj.takes_prequantized_input(m):
                        xq, xs = _mxfp8_operand(x)
                        ref_out = allreduce(plain.forward_prequantized(xq, xs))
                        ref = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, ref_out, block, proj, norm, out_norm
                        )
                        out = o_proj.forward_prequantized(xq, xs)
                        assert isinstance(out, MnnvlPendingResult) == (m <= in_place_max_rows)
                        got = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, out, block, proj, norm, out_norm
                        )
                        for a, b in zip(ref, got):
                            bad += _differ(a, b)

        # One CUDA graph of two decoder layers in their production order: the o_proj pushes and
        # leaves the call pending, the mix after attention finishes it (before its grid wait
        # where one-shot), and the next all-reduce on the workspace (the MoE's, a C++ call)
        # re-arms that buffer. Replayed with new inputs against the projection + C++
        # all-reduce + tensor mix.
        k = 1536
        w, w_scale = _weight(hidden, k, 31 + rank)
        plain = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden)
        plain.post_load_weights()
        o_proj = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden, allreduce, 128)
        o_proj.post_load_weights()
        for m in (1, 8, 64, 96):
            xs = [_input(m, k, 40 + i, rank) for i in range(2)]
            prefix = torch.randn(m, hidden, device="cuda", generator=gen).to(torch.bfloat16)
            block = torch.randn(4, m, hidden, device="cuda", generator=gen).to(torch.bfloat16)

            def layers(projection):
                res = []
                for x in xs:
                    out = projection(x)
                    if projection is plain:
                        out = allreduce(out)
                    else:
                        assert isinstance(out, MnnvlPendingResult) == _left_pending(
                            m, hidden, k, world_size, 128
                        )
                    updated, normed = kimi._apply_attn_res_add_and_rmsnorm(
                        prefix, out, block, proj, norm, out_norm
                    )
                    res += [updated, normed, allreduce(normed)]
                return res

            layers(o_proj)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                results = layers(o_proj)
            for rep in range(3):
                for i, x in enumerate(xs):
                    x.copy_(_input(m, k, 60 + 10 * rep + i, rank))
                torch.cuda.synchronize()
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                for a, b in zip(layers(plain), results):
                    bad += _differ(a, b)
            del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank_transport(world_size):
    """The transport and Lamport buffer size ``mnnvl_allreduce_transport`` gives are the ones the
    C++ MNNVL all-reduce uses (read back from the rotation state it leaves: its stage count and
    bytes to clear), and ``push_target`` / ``check_transport`` take that transport."""
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import (
        get_or_scale_allreduce_mnnvl_workspace,
        mnnvl_allreduce_transport,
    )
    from tensorrt_llm._torch.kernels.blackwell import mnnvl_pushed_allreduce as pushed
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=dtype)
            assert allreduce.mnnvl_allreduce is not None
            for h in (64, 1536, 4096, 7168):
                for m in (1, 2, 7, 9, 16, 17, 18, 19, 31, 32, 33, 63, 64, 65, 128, 300):
                    one_shot, need = mnnvl_allreduce_transport(m, h, world_size, dtype)
                    allreduce(torch.ones(m, h, device="cuda", dtype=dtype))
                    torch.cuda.synchronize()
                    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)
                    flags = workspace["buffer_flags"].cpu().long().tolist()
                    stages, clear0 = flags[3], flags[4]
                    assert stages == (1 if one_shot else 2), (dtype, m, h, flags)
                    assert clear0 * stages == need, (dtype, m, h, flags)
                    if dtype is torch.bfloat16:
                        target = allreduce.push_target(m, h)
                        assert target.one_shot == one_shot
                        pushed.check_transport(target.view, m, h, one_shot)
                        with pytest.raises(ValueError):
                            pushed.check_transport(target.view, m, h, not one_shot)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _k3_layer(kind, layer_idx, world_size, rank):
    """One Kimi-K3 decoder layer (KDA or MLA attention, dense MLP) of a small config with FP8
    attention projections, loaded through ``_finalize_weight_load`` with random weights."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.distributed import AllReduceStrategy
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    hidden, heads = 7168, 24
    cfg = SimpleNamespace(
        hidden_size=hidden,
        rms_norm_eps=1e-6,
        num_attention_heads=heads,
        q_lora_rank=128,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        max_position_embeddings=128,
        num_hidden_layers=layer_idx + 1,
        num_experts=None,
        first_k_dense_replace=layer_idx + 1,
        intermediate_size=512,
        attn_res_block_size=4,
        linear_attn_config={
            "num_heads": heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "kda_layers": [layer_idx + 1] if kind == "kda" else [],
            "full_attn_layers": [layer_idx + 1] if kind == "mla" else [],
        },
    )
    names = (
        ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "f_a_proj", "f_b_proj", "b_proj")
        if kind == "kda"
        else ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "o_proj")
    )
    config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=world_size, tp_size=world_size, rank=rank),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={
            f"language_model.model.layers.{layer_idx}.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES
            )
            for name in names
        },
        allreduce_strategy=AllReduceStrategy.MNNVL,
        skip_create_weights_in_init=kind == "mla",
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    stream = torch.cuda.Stream()
    aux_streams = {stream_type: stream for stream_type in AuxStreamType}
    layer = kimi.KimiLinearDecoderLayer(config, cfg, layer_idx, aux_streams)
    model = kimi.KimiLinearForCausalLM.__new__(kimi.KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model._fp8_weight_read_moe_mlp = False
    model.model_config = config
    model.spec_config = config.spec_config
    model._repurposed_tp_mapping = None
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    if kind == "mla":
        model.__post_init__()
    model.cuda()
    gen = torch.Generator(device="cuda").manual_seed(100 * layer_idx + rank)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                codes = torch.randint(-32, 33, param.shape, device="cuda", generator=gen)
                param.copy_(codes.to(torch.float8_e4m3fn))
            elif not param.is_floating_point():
                continue
            elif "scale" in name.rsplit(".", 1)[-1]:
                param.copy_(torch.rand(param.shape, device="cuda", generator=gen) * 0.002 + 0.001)
            else:
                param.copy_(torch.randn(param.shape, device="cuda", generator=gen) * 0.02)
    model._finalize_weight_load(0, 0)
    for module in model.modules():
        if isinstance(module, (Linear, kimi._Fp8BlockScaleWeightReadLinear)):
            module.post_load_weights()
    if kind == "mla":
        layer.self_attn.mixer.cache_derived_state()
    return layer


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank_k3_attention(world_size):
    """Kimi-K3 attention as the FP8 swap builds it: only ``o_proj`` gets the attention's
    all-reduce, the attention (KDA ``_project_output``, ``KimiMLARuntime.forward``) skips its own
    reduction exactly when ``o_proj`` reduces, and ``o_proj`` leaves the all-reduce pending only
    up to the rows the mix after attention finishes (none on snapshot layers). Each output is
    bitwise the projection followed by one C++ all-reduce (not unreduced, not reduced twice)."""
    from tensorrt_llm._torch.distributed import MnnvlPendingResult
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res as attn_res
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.projection_capabilities import reduces_output

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        device = torch.cuda.current_device()
        bad = 0
        pending_calls = 0
        # Layer 0 is a snapshot layer; layer 1 mixes over 2 candidates, layer 49 over 14 (more
        # than the decode attention-residual kernel takes).
        for kind in ("kda", "mla"):
            for layer_idx in (0, 1, 49):
                layer = _k3_layer(kind, layer_idx, world_size, rank)
                hidden = layer.hidden_size
                owner = layer.linear_attn if kind == "kda" else layer.self_attn
                attention = layer.linear_attn if kind == "kda" else layer.self_attn.mixer
                allreduce = owner.output_all_reduce
                assert allreduce is not None
                o_proj = attention.o_proj
                for name, child in attention.named_children():
                    if isinstance(child, kimi._Fp8BlockScaleWeightReadLinear):
                        assert child.output_all_reduce is (allreduce if name == "o_proj" else None)
                assert reduces_output(o_proj)
                candidates = layer_idx // layer.attn_res_block_size + 2
                decode_mix = (
                    layer_idx % layer.attn_res_block_size != 0
                    and kimi._FUSED_ATTN_RES_ENABLED
                    and kimi._decode_attn_res_on_device(device)
                    and attn_res.supports(attn_res.MAX_TOKENS, hidden, candidates)
                )
                expected_rows = attn_res.MAX_TOKENS if decode_mix else 0
                assert layer.attention_output_reduction() == (allreduce, expected_rows)
                assert o_proj._in_place_max_rows == expected_rows

                # The same weight without the all-reduce: the attention then reduces itself.
                plain = kimi._Fp8BlockScaleWeightReadLinear(
                    o_proj.weight, o_proj.weight_scale, o_proj.out_features
                )
                plain.post_load_weights()
                if kind == "mla":
                    assert attention._fused_output_gate
                gen = torch.Generator(device="cuda").manual_seed(7 + rank)
                for m in (1, 8, 64, 65):
                    core = (
                        torch.randn(m, o_proj.in_features, device="cuda", generator=gen) * 0.25
                    ).to(torch.bfloat16)
                    states = (torch.randn(m, hidden, device="cuda", generator=gen) * 0.25).to(
                        torch.bfloat16
                    )

                    def attend(projection):
                        attention.o_proj = projection
                        if kind == "kda":
                            return attention._project_output(core)
                        attention.forward = (
                            lambda position_ids, hidden_states, metadata, **kwargs: (
                                attention._gate_and_project_output(
                                    hidden_states, [core], position_ids, metadata, None
                                )
                            )
                        )
                        return owner(states, None)

                    out = attend(o_proj)
                    pending = m <= expected_rows
                    assert isinstance(out, MnnvlPendingResult) == pending, (kind, layer_idx, m)
                    if pending:
                        pending_calls += 1
                        prefix = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
                        assert (
                            kimi._attn_res_kernel(prefix, candidates) is kimi._AttnResKernel.DECODE
                        )
                    got = _result(out)
                    unfused = attend(plain)
                    if kind == "kda":
                        ref = allreduce(plain(core))
                    else:
                        gate = attention.g_proj.forward_sigmoid(states)
                        ref = allreduce(plain.forward_gated(core, gate))
                    attention.o_proj = o_proj
                    bad += _differ(ref, got) + _differ(ref, unfused)
                del layer
                torch.cuda.synchronize()
        if kimi._decode_attn_res_on_device(device) and kimi._FUSED_ATTN_RES_ENABLED:
            assert pending_calls > 0
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


_SENTINEL = -(2**31)  # 0x80000000 as int32: an empty Lamport word


def _small_batch_moe_operands(world, rank, m, owners, seed):
    """Operands of the small-batch MoE kernel on ``rank`` of an expert-parallel group: the same
    top-k on every rank, drawn from the experts of the ``owners`` ranks only (the other ranks then
    have no local expert), and this rank's NVFP4 tokens and experts."""
    hidden, inter, num_experts, top_k = 1024, 256, 64, 8
    num_local = num_experts // world
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(m, num_experts, device="cuda", generator=gen)
    owned = torch.zeros(num_experts, dtype=torch.bool, device="cuda")
    for r in owners:
        owned[r * num_local : (r + 1) * num_local] = True
    scales, ids = logits.masked_fill(~owned, float("-inf")).topk(top_k, dim=-1)
    wgen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    return small_batch_moe_operands(
        ids.to(torch.int32),
        scales.softmax(dim=-1).float(),
        hidden,
        inter,
        num_experts,
        num_local,
        rank * num_local,
        gen,
        wgen,
    )


def _unpushed_ranks(target, m, h):
    """Ranks with a slot of some token still empty in this rank's buffer of the one-shot call
    ``target`` (its finish would poll them forever)."""
    view = target.view
    cur, _, nbytes = (int(v) for v in view.flags.cpu().tolist()[:3])
    words = view.buffer.view(torch.uint8)[cur * nbytes : cur * nbytes + m * view.world * h * 2]
    words = words.view(torch.int32).view(m, view.world, h // 2)
    return sorted(set((words == _SENTINEL).any(-1).nonzero()[:, 1].tolist()))


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank_small_batch_moe_push(world_size):
    """The small-batch MoE kernel's output push on every rank of an expert-parallel group,
    finished by the finish kernel, against the C++ all-reduce of its unpushed outputs; with the
    experts of every rank and of rank 0 only (the others push the rows of a rank without a local
    expert), every routing mode, the default and a single CTA. With the red combine (whose bits
    depend on the order of its reds) the pushed slots are the kernel's own output: the finish
    equals the C++ all-reduce of that output, which is the slice combine's within BF16 rounding."""
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_small_batch_moe_custom_op import (
        new_small_batch_moe_counters,
    )
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        counters = new_small_batch_moe_counters(torch.device("cuda"))
        bad = 0
        for m in (3, 8):
            for owners in ((0,), tuple(range(world_size))):
                d = _small_batch_moe_operands(world_size, rank, m, owners, 41 * m + len(owners))
                if owners == (0,):
                    assert (int(d["nnet"].item()) == 0) == (rank != 0)
                h = d["hidden"]
                for mode in ("permuted", "token_indexed", "published"):
                    kwargs = dict(
                        token_indexed=mode != "permuted",
                        published=mode == "published",
                    )
                    out = torch.full((m, h), float("nan"), dtype=torch.bfloat16, device="cuda")
                    run_small_batch_moe(d, out, counters, deterministic=True, **kwargs)
                    ref = allreduce(out)
                    for deterministic, num_ctas in ((True, 0), (True, 1), (False, 0), (False, 1)):
                        target = allreduce.push_target(m, h)
                        assert target is not None and target.one_shot
                        pushed = torch.full_like(out, float("nan"))
                        run_small_batch_moe(
                            d,
                            pushed,
                            counters,
                            deterministic=deterministic,
                            num_ctas=num_ctas,
                            push=target.view,
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                        comm.Barrier()
                        missing = _unpushed_ranks(target, m, h)
                        assert not missing, (
                            f"m={m} owners={owners} {mode} deterministic={deterministic} "
                            f"ctas={num_ctas}: slots of ranks {missing} never pushed"
                        )
                        result = allreduce.finish_push(target)
                        if deterministic:
                            bad += _differ(out, pushed)
                            bad += _differ(ref, result)
                        else:
                            bad += _differ(allreduce(pushed), result)
                            a, b = pushed.float(), out.float()
                            rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
                            assert rel < 1e-2, f"red combine off the slice combine by {rel}"
                        torch.cuda.synchronize()
                        assert int(counters.abs().sum().item()) == 0, "counters not reset"
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _finish_reference(slots, chunk):
    """The one-shot sum of ``slots [m, W, h]`` (BF16) in the C++ order: FP32 partial sums of
    ``chunk`` ranks from zero, added to an FP32 accumulator from zero, rounded to BF16."""
    acc = torch.zeros(slots.shape[0], slots.shape[2], device=slots.device)
    for c0 in range(0, slots.shape[1], chunk):
        part = torch.zeros_like(acc)
        for r in range(c0, c0 + chunk):
            part = part + slots[:, r].float()
        acc = acc + part
    return acc.to(torch.bfloat16)


def test_lamport_state_layout(monkeypatch):
    """The finish kernel's view of the rotation state (``LamportFlags`` in ``lamportUtils.cuh``)
    matches the state ``ops.py`` initializes for the C++ kernels."""
    from tensorrt_llm._torch.distributed import ops
    from tensorrt_llm._torch.kernels.blackwell import mnnvl_pushed_allreduce as pushed

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    nbytes = 1 << 20
    workspace = {
        "uc_buffer": torch.empty(16),
        "buffer_flags": torch.zeros(pushed.FLAG_ARRIVALS + 1, dtype=torch.uint32),
        "buffer_size_bytes": nbytes,
        "comm": None,
    }
    ops._initialize_allreduce_mnnvl_protocol(workspace, converge_errors=False)
    cur, dirty, size, stages, *clears, arrivals = workspace["buffer_flags"].long().tolist()
    assert len(clears) == 4 and arrivals == 0
    assert size == nbytes and stages == 0 and clears == [0] * 4
    assert dirty == (cur + pushed.NUM_BUFFERS - 1) % pushed.NUM_BUFFERS
    with pytest.raises(ValueError):
        pushed.PushedAllReduceFinishKernel(12, 7168, True, True, True)


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_pushed_allreduce_matches_cpp(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_o_proj_pushed_allreduce_finished_by_mix(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_o_proj, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_small_batch_moe_output_push_without_local_experts(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the small-batch MoE kernel targets the SM100 family")
    results = mpi_pool_executor.map(
        _run_rank_small_batch_moe_push, *zip(*[(world_size,)] * world_size)
    )
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("world", [4, 16, 32, 64])
def test_one_shot_finish_sums_rank_chunks(world):
    """The one-shot finish kernel at world sizes whose sum spans several chunks of ranks (polled
    in one round up to ``POLL_ALL_MAX_WORLD`` ranks, one chunk at a time above) against the C++
    order, on one GPU: the pushed slots of every rank are written into the local Lamport buffer,
    which is all a one-shot finish reads."""
    from tensorrt_llm._torch.distributed.ops import LamportWorkspaceView, mnnvl_allreduce_transport
    from tensorrt_llm._torch.kernels.blackwell import mnnvl_pushed_allreduce as pushed

    m, h = 3, 1024
    one_shot, nbytes = mnnvl_allreduce_transport(m, h, world, torch.bfloat16)
    assert one_shot
    workspace = torch.zeros(pushed.NUM_BUFFERS * nbytes, dtype=torch.uint8, device="cuda")
    cur = 1
    flags = torch.tensor(
        [cur, (cur + 2) % 3, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
    ).view(torch.uint32)
    gen = torch.Generator(device="cuda").manual_seed(world)
    slots = (torch.randn(m, world, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    bits = slots.view(torch.int16)
    slots = torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)
    workspace[cur * nbytes : (cur + 1) * nbytes].view(torch.bfloat16).copy_(slots.flatten())
    view = LamportWorkspaceView(
        world=world,
        rank=0,
        buffer=workspace,
        mc_base=0,
        peer_ptrs=torch.tensor([workspace.data_ptr()], dtype=torch.int64, device="cuda"),
        flags=flags,
        bytes_per_buffer=nbytes,
    )
    out = pushed.run_finish(view, torch.empty(m, h, device="cuda", dtype=torch.bfloat16), True)
    torch.cuda.synchronize()
    ref = _finish_reference(slots, world if world <= 8 else 8)
    assert _differ(ref, out) == 0
    state = flags.cpu().long().tolist()
    assert state[:5] == [(cur + 1) % 3, cur, nbytes, 1, m * world * h * 2]


@skip_pre_blackwell
@pytest.mark.parametrize("world, m", [(4, 3), (16, 2), (16, 8)])
@pytest.mark.parametrize("cpp_one_shot", [True, False])
def test_all_gather_call_finish_on_one_gpu(world, m, cpp_one_shot):
    """An all-gather call finished by the finish kernel (``bf16(y + bf16(sum))``) and by the
    decode MoE-residual mix, with the sum order of either C++ transport, on one GPU: every
    rank's slots and the all-gather rows are written into the local buffer."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("needs the CuTe DSL")
    from tensorrt_llm._torch.distributed.ops import LamportWorkspaceView
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res as attn_res
    from tensorrt_llm._torch.kernels.blackwell import mnnvl_pushed_allreduce as pushed

    h, k = 7168, 3
    nbytes = max(pushed.all_gather_call_bytes(m, h, world), 1 << 16)
    chunk = pushed.reduce_chunk(world, cpp_one_shot)
    gen = torch.Generator(device="cuda").manual_seed(world + m)
    slots = (torch.randn(m, world, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    bits = slots.view(torch.int16)
    slots = torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)
    y = (torch.randn(m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    y_bits = y.view(torch.int16)
    y = torch.where(y_bits == -32768, torch.zeros_like(y_bits), y_bits).view(torch.bfloat16)
    shared = _finish_reference(slots, chunk)
    if not cpp_one_shot:
        shared_bits = shared.view(torch.int16)
        shared = torch.where(shared_bits == -32768, torch.zeros_like(shared_bits), shared_bits)
        shared = shared.view(torch.bfloat16)
    cur = 1

    def workspace():
        ws = torch.zeros(pushed.NUM_BUFFERS * nbytes, dtype=torch.uint8, device="cuda")
        buf = ws[cur * nbytes : cur * nbytes + pushed.all_gather_call_bytes(m, h, world)]
        buf.view(torch.bfloat16).copy_(torch.cat([slots.flatten(), y.flatten()]))
        flags = torch.tensor(
            [cur, (cur + 2) % 3, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
        ).view(torch.uint32)
        view = LamportWorkspaceView(
            world=world,
            rank=0,
            buffer=ws,
            mc_base=0,
            peer_ptrs=torch.tensor([ws.data_ptr()], dtype=torch.int64, device="cuda"),
            flags=flags,
            bytes_per_buffer=nbytes,
        )
        return ws, view

    expected_state = [(cur + 1) % 3, cur, nbytes, 1, m * (world + 1) * h * 2, 0, 0, 0, 0]
    ws, view = workspace()
    out = pushed.run_finish(
        view,
        torch.empty(m, h, device="cuda", dtype=torch.bfloat16),
        True,
        all_gather=True,
        chunk=chunk,
        sanitize=not cpp_one_shot,
    )
    torch.cuda.synchronize()
    assert _differ(y + shared, out) == 0
    assert view.flags.cpu().long().tolist() == expected_state

    if not attn_res.is_supported_device(torch.cuda.current_device()):
        return
    residual = (torch.randn(m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    block = (torch.randn(k, m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    weights = [
        (1 + 0.1 * torch.randn(h, device="cuda", generator=gen)).to(torch.bfloat16)
        for _ in range(3)
    ]
    ref_out, ref_upd = torch.empty_like(residual), torch.empty_like(residual)
    attn_res.run_attn_res(residual, (y, shared), block, *weights, 1e-6, 1e-6, ref_out, ref_upd)
    ws, view = workspace()
    got_out, got_upd = torch.empty_like(residual), torch.empty_like(residual)
    attn_res.run_attn_res(
        residual,
        (),
        block,
        *weights,
        1e-6,
        1e-6,
        got_out,
        got_upd,
        in_place_addend=(view.uc_local, view.flags.data_ptr()),
        finish=attn_res.AllReduceFinish(
            world, 0, 0, False, True, chunk=chunk, sanitize=not cpp_one_shot, all_gather=True
        ),
    )
    torch.cuda.synchronize()
    assert _differ(ref_upd, got_upd) == 0 and _differ(ref_out, got_out) == 0
    assert view.flags.cpu().long().tolist() == expected_state


@skip_pre_blackwell
@pytest.mark.parametrize("early_poll", [False, True])
@pytest.mark.parametrize("world, m", [(2, 1), (8, 5), (16, 3), (16, 8), (64, 1)])
def test_attn_res_finishes_one_shot_call(world, m, early_poll):
    """The decode attention-residual mix finishing a pushed one-shot call (sums in the C++ order,
    chunks of eight ranks above eight) against the mix of the finished all-reduce, on one GPU:
    every rank's slots are written into the local buffer, which is all a one-shot finish reads.
    The mix also advances the rotation state like the finish kernel and re-arms the previous
    call's buffer."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("needs the CuTe DSL")
    from tensorrt_llm._torch.distributed.ops import mnnvl_allreduce_transport
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res as attn_res
    from tensorrt_llm._torch.kernels.blackwell import mnnvl_pushed_allreduce as pushed

    if not attn_res.is_supported_device(torch.cuda.current_device()):
        pytest.skip("the decode attention-residual kernel does not run on this GPU")
    h, k = 7168, 3
    # One-shot up to 2 MiB (TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES of the min-latency deployments):
    # 8 rows over 16 ranks, one decode step of one request with 7 draft tokens.
    one_shot, nbytes = mnnvl_allreduce_transport(m, h, world, torch.bfloat16, 2 << 20)
    assert one_shot
    nbytes = max(nbytes, 1 << 16)
    workspace = torch.zeros(pushed.NUM_BUFFERS * nbytes, dtype=torch.uint8, device="cuda")
    cur, dirty = 1, 0
    # The previous call (one stage of 4 KiB in buffer ``dirty``) is re-armed by the mix.
    flags = torch.tensor(
        [cur, dirty, nbytes, 1, 4096, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
    ).view(torch.uint32)
    gen = torch.Generator(device="cuda").manual_seed(world + m)
    slots = (torch.randn(m, world, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    bits = slots.view(torch.int16)
    slots = torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)
    workspace[cur * nbytes : cur * nbytes + slots.numel() * 2].view(torch.bfloat16).copy_(
        slots.flatten()
    )
    addend = _finish_reference(slots, pushed.reduce_chunk(world, True))
    prefix = (torch.randn(m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    block = (torch.randn(k, m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    weights = [
        (1 + 0.1 * torch.randn(h, device="cuda", generator=gen)).to(torch.bfloat16)
        for _ in range(3)
    ]
    ref_out, ref_upd = torch.empty_like(prefix), torch.empty_like(prefix)
    attn_res.run_attn_res(prefix, (addend,), block, *weights, 1e-6, 1e-6, ref_out, ref_upd)
    out, upd = torch.empty_like(prefix), torch.empty_like(prefix)
    attn_res.run_attn_res(
        prefix,
        (),
        block,
        *weights,
        1e-6,
        1e-6,
        out,
        upd,
        in_place_addend=(workspace.data_ptr(), flags.data_ptr()),
        finish=attn_res.AllReduceFinish(world, 0, 0, False, early_poll),
    )
    torch.cuda.synchronize()
    assert _differ(ref_upd, upd) == 0 and _differ(ref_out, out) == 0
    state = flags.cpu().long().tolist()
    assert state == [(cur + 1) % 3, cur, nbytes, 1, m * world * h * 2, 0, 0, 0, 0]
    rearmed = workspace[dirty * nbytes : dirty * nbytes + 4096].view(torch.int32)
    assert bool((rearmed == _SENTINEL).all())


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank_all_gather(world_size):
    """An all-gather call (the column-parallel MoE tail): the shared partial in the slots and
    every rank's columns of ``y`` in the all-gather rows, finished into ``bf16(y + all-reduce)``
    and by the decode MoE-residual mix, against the C++ all-reduce of the partials and the tensor
    mix; one-shot and two-shot sizes of that all-reduce, eagerly and in a CUDA graph with C++
    calls on the workspace in between."""
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy, MnnvlPendingResult
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        from torch import nn

        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        hidden = 7168
        blocks = hidden // 128
        first = rank * (blocks // world_size) + min(rank, blocks % world_size)
        count = blocks // world_size + (rank < blocks % world_size)
        c0, n = first * 128, count * 128
        gen = torch.Generator(device="cuda").manual_seed(5)
        proj = nn.Linear(hidden, 1, bias=False, dtype=torch.bfloat16, device="cuda")
        proj.weight.data.copy_(torch.randn(1, hidden, device="cuda", generator=gen) * 0.05)
        norm = kimi.KimiK3RMSNorm(hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda")
        norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        out_norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16).cuda()
        out_norm.weight.data.copy_(1 + 0.1 * torch.randn(hidden, device="cuda", generator=gen))
        bad = 0

        def tail(y, shared_partial):
            target = allreduce.push_target(y.shape[0], hidden, all_gather=True)
            allreduce.push_input(target, shared_partial)
            allreduce.push_input(target, y[:, c0 : c0 + n].contiguous(), columns=c0)
            return allreduce.finish_push(target)

        # 64 rows: the all-reduce of the partials is two-shot at 2 and 4 ranks.
        for m in (1, 8, 64):
            for kind in (0, 3):
                y = _input(m, hidden, 3 * m + kind, 0, kind)  # replicated, as latent_up's output
                shared_partial = _input(m, hidden, 7 * m + kind, rank, kind)
                residual = _input(m, hidden, 11 * m, 0)
                block = _input(4 * m, hidden, 13 * m, 0).view(4, m, hidden)
                shared = allreduce(shared_partial)
                ref_sum = y + shared
                pending = tail(y, shared_partial)
                assert isinstance(pending, MnnvlPendingResult) and pending.all_gather
                bad += _differ(ref_sum, pending.materialize())
                bad += _differ(shared, allreduce(shared_partial))
                ref = kimi._attn_res_decode_kernel(
                    residual, (y, shared), block, proj, norm, out_norm, mxfp8=True
                )
                if ref is None:  # no decode mix on this device
                    continue
                got = kimi._attn_res_decode_kernel(
                    residual, (tail(y, shared_partial), None), block, proj, norm, out_norm, True
                )
                for a, b in zip(ref[:3], got[:3]):
                    bad += _differ(a, b)
                # The R128c4 scales hold 128-row blocks; only the first m rows are written.
                bad += _differ(_scale_rows(ref[3], m, hidden), _scale_rows(got[3], m, hidden))
                bad += _differ(shared, allreduce(shared_partial))

        # A CUDA graph: tail call, C++ call, tail call finished by the mix, C++ call.
        m = 8
        ys = [_input(m, hidden, 300 + i, 0) for i in range(2)]
        parts = [_input(m, hidden, 400 + i, rank) for i in range(2)]
        residual = _input(m, hidden, 500, 0)
        block = _input(4 * m, hidden, 501, 0).view(4, m, hidden)

        def calls():
            return [
                tail(ys[0], parts[0]).materialize(),
                allreduce(parts[1]),
                *kimi._attn_res_decode_kernel(
                    residual, (tail(ys[1], parts[1]), None), block, proj, norm, out_norm
                ),
                allreduce(parts[0]),
            ]

        if kimi._attn_res_decode_kernel(residual, (ys[1], parts[1]), block, proj, norm, out_norm):
            calls()
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                results = calls()
            for rep in range(3):
                for i in range(2):
                    parts[i].copy_(_input(m, hidden, 600 + 10 * rep + i, rank))
                torch.cuda.synchronize()
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                s0, s1 = allreduce(parts[0]), allreduce(parts[1])
                ref = [ys[0] + s0, s1]
                ref += list(
                    kimi._attn_res_decode_kernel(residual, (ys[1], s1), block, proj, norm, out_norm)
                )
                ref.append(s0)
                for a, b in zip(ref, results):
                    bad += _differ(a, b)
            del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_all_gather_call_matches_replicated_tail(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_all_gather, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("m", [1, 7, 16, 32])
def test_latent_up_output_rows_match_whole_projection(m):
    """latent_up's row slice (whole 128-row blocks) projects bitwise like the matching columns
    of the whole projection, prequantized input, on every rank's slice of 8 and 16 ranks."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    hidden, latent = 7168, 3072
    weight, scale = _weight(hidden, latent, 17)
    x = _input(m, latent, 19 + m, 0)
    xq, xs = _mxfp8_operand(x)
    whole = _Fp8BlockScaleWeightReadLinear(weight, scale, hidden)
    whole.post_load_weights()
    if not whole.takes_prequantized_input(m):
        pytest.skip("latent_up's low-M GEMM does not serve these rows here")
    ref = whole.forward_prequantized(xq, xs)
    blocks = hidden // 128
    for world in (8, 16):
        for rank in range(world):
            first = rank * (blocks // world) + min(rank, blocks % world)
            rows = (first * 128, (blocks // world + (rank < blocks % world)) * 128)
            part = _Fp8BlockScaleWeightReadLinear(weight, scale, hidden)
            part.set_output_rows(rows)
            part.post_load_weights()
            if not part.serves_output_rows(m):
                continue
            got = part.forward_rows_prequantized(xq, xs)
            torch.cuda.synchronize()
            assert _differ(ref[:, rows[0] : rows[0] + rows[1]].contiguous(), got) == 0


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_transport_matches_cpp(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_transport, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_attention_o_proj_reduces_once(mpi_pool_executor):
    pytest.importorskip("fla")
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the Kimi-K3 FP8 checkpoint attention needs SM100/SM103")
    results = mpi_pool_executor.map(_run_rank_k3_attention, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
