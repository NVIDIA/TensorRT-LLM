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
low-M GEMM's push epilogue): bitwise equal to the C++ MNNVL all-reduce, with pushed and C++
calls interleaved on one workspace, eagerly and in CUDA graph replays."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
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
    # Denormals only: the sums stay denormal (no flush to zero anywhere).
    bits = torch.randint(-127, 127, (m, h), device="cuda", generator=gen).to(torch.int16)
    return bits.view(torch.bfloat16)


def _differ(ref, out):
    return int((ref.view(torch.int16) != out.view(torch.int16)).sum())


def _torch_push(workspace, target, x, rank):
    """Producer stand-in: store ``x`` into the Lamport slots of ``target`` with plain copies
    (``-0`` sent as ``+0``), through every rank's unicast mapping."""
    torch.cuda.synchronize()  # the previous call's rotation state is final
    flags = target.view.flags.cpu().tolist()
    cur, nbytes = int(flags[0]), int(flags[2])
    world = target.view.world
    m, h = x.shape
    bits = x.view(torch.int16)
    clean = torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)
    handle = workspace["handle"]
    elems = 3 * nbytes // 2
    for peer in range(world):
        buf = handle.get_uc_buffer(peer, (elems,), torch.bfloat16, 0)
        base = cur * nbytes // 2
        if target.one_shot:
            slots = buf[base : base + m * world * h].view(m, world, h)
            slots[:, rank].copy_(clean)
        else:
            for t in range(peer, m, world):
                slot = (t // world) * world + rank
                buf[base + slot * h : base + (slot + 1) * h].copy_(clean[t])
    torch.cuda.synchronize()


def _stays_in_place(m, h, world, in_place_max_rows):
    """Whether a pushed o_proj of ``m`` rows leaves its all-reduce result in place: up to
    ``in_place_max_rows`` rows, where the C++ rule picks the two-shot transport."""
    from tensorrt_llm._torch.distributed.ops import mnnvl_allreduce_transport

    one_shot, _ = mnnvl_allreduce_transport(m, h, world, torch.bfloat16)
    return m <= in_place_max_rows and not one_shot


def _result(out):
    from tensorrt_llm._torch.distributed import MnnvlInPlaceResult

    return out.materialize() if isinstance(out, MnnvlInPlaceResult) else out


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


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
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
        # in between; one-shot and two-shot sizes, in place where the transport allows it.
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
                for in_place in (False, True):
                    target = allreduce.push_target(m, h, in_place=in_place)
                    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
                    _torch_push(workspace, target, x, rank)
                    bad += _differ(ref, _result(allreduce.finish_push(target)))
                    bad += _differ(ref, allreduce(x))

        # GEMM push epilogue (plain and gated; split-K at (7168, 1536) on SM100 / SM103)
        # against the GEMM followed by the C++ all-reduce.
        for n, k in ((7168, 768), (7168, 1536)):
            w, w_scale = _gemm_weight(n, k, 11 + rank)
            for m in (1, 7, 14, 32, 64):
                if not low_m.all_reduce_push_applies(m, n, k):
                    continue
                x = _input(m, k, 5 * m, rank)
                gate = _input(m, k, 7 * m, rank)
                gates = (None, gate) if low_m.gate_fits(x, gate, n) else (None,)
                for g in gates:
                    y = low_m.run_low_m_gemm(
                        x,
                        w,
                        w_scale,
                        torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
                        gate=g,
                    )
                    ref = allreduce(y)
                    for in_place in (False, True):
                        target = allreduce.push_target(m, n, in_place=in_place)
                        low_m.run_low_m_gemm_all_reduce_push(
                            x, w, w_scale, target.view, target.one_shot, gate=g
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

        # A C++ call grows the workspace; the capture follows without an eager pushed call on
        # the new workspace, whose device view is built with it.
        allreduce(torch.zeros(2048, n, device="cuda", dtype=torch.bfloat16))
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


def _run_rank_o_proj(world_size):
    """An o_proj that owns its all-reduce (pushed, result read in place by the mix after
    attention) against the projection followed by the C++ all-reduce and the tensor mix."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from torch import nn

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy, MnnvlInPlaceResult
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
            for in_place_max_rows in (0, 64):
                o_proj = kimi._Fp8BlockScaleWeightReadLinear(
                    w, w_scale, hidden, allreduce, in_place_max_rows
                )
                o_proj.post_load_weights()
                for m in (1, 8, 14, 32, 64, 65, 200):
                    x = _input(m, k, 9 * m, rank)
                    gate = _input(m, k, 13 * m, rank)
                    prefix = torch.randn(m, hidden, device="cuda", generator=gen).to(torch.bfloat16)
                    block = torch.randn(4, m, hidden, device="cuda", generator=gen).to(
                        torch.bfloat16
                    )
                    for g in (None, gate):
                        ref_out = allreduce(plain(x) if g is None else plain.forward_gated(x, g))
                        ref = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, ref_out, block, proj, norm, out_norm
                        )
                        out = o_proj(x) if g is None else o_proj.forward_gated(x, g)
                        # Pushed calls stay in place up to the row count asked for, where the
                        # transport is two-shot.
                        assert isinstance(out, MnnvlInPlaceResult) == _stays_in_place(
                            m, hidden, world_size, in_place_max_rows
                        )
                        got = kimi._apply_attn_res_add_and_rmsnorm(
                            prefix, out, block, proj, norm, out_norm
                        )
                        for a, b in zip(ref, got):
                            bad += _differ(a, b)

        # One CUDA graph of two decoder layers in their production order: the o_proj pushes and
        # leaves its result in place, the mix after attention reads it there, and the next
        # all-reduce on the workspace (the MoE's, a C++ call) re-arms that buffer. Replayed with
        # new inputs against the projection + C++ all-reduce + tensor mix.
        k = 1536
        w, w_scale = _weight(hidden, k, 31 + rank)
        plain = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden)
        plain.post_load_weights()
        o_proj = kimi._Fp8BlockScaleWeightReadLinear(w, w_scale, hidden, allreduce, 64)
        o_proj.post_load_weights()
        for m in (8, 64):
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
                        assert isinstance(out, MnnvlInPlaceResult) == _stays_in_place(
                            m, hidden, world_size, 64
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


def _run_rank_transport(world_size):
    """The transport and Lamport buffer size ``mnnvl_allreduce_transport`` gives are the ones the
    C++ MNNVL all-reduce uses (read back from the rotation state it leaves: its stage count and
    bytes to clear), and ``push_target`` / ``check_transport`` take that transport."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
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


def _run_rank_k3_attention(world_size):
    """Kimi-K3 attention as the FP8 swap builds it: only ``o_proj`` gets the attention's
    all-reduce, the attention (KDA ``_project_output``, ``KimiMLARuntime.forward``) skips its own
    reduction exactly when ``o_proj`` reduces, and ``o_proj`` leaves its result in place only up
    to the rows the mix after attention reads in place (none on snapshot layers). Each output is
    bitwise the projection followed by one C++ all-reduce (not unreduced, not reduced twice)."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.distributed import MnnvlInPlaceResult
    from tensorrt_llm._torch.distributed.ops import mnnvl_allreduce_transport
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res as attn_res
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.projection_capabilities import reduces_output

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        device = torch.cuda.current_device()
        bad = 0
        in_place_calls = 0
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
                    one_shot, _ = mnnvl_allreduce_transport(m, hidden, world_size, torch.bfloat16)
                    stays = m <= expected_rows and not one_shot
                    assert isinstance(out, MnnvlInPlaceResult) == stays, (kind, layer_idx, m)
                    if stays:
                        in_place_calls += 1
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
            assert in_place_calls > 0
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
        "buffer_flags": torch.zeros(pushed._FLAG_ARRIVALS + 1, dtype=torch.uint32),
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
def test_o_proj_pushed_allreduce_in_place_mix(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank_o_proj, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("world", [4, 16, 64])
def test_one_shot_finish_sums_rank_chunks(world):
    """The one-shot finish kernel at world sizes whose sum spans several chunks of ranks
    (polled and summed one chunk at a time) against the C++ order, on one GPU: the pushed slots
    of every rank are written into the local Lamport buffer, which is all a one-shot finish
    reads."""
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
        uc_local=workspace.data_ptr(),
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
