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
"""Collective tests for TokenShardedTP and the token-sharded adapters.

The ``*_gloo`` tests run on CPU (``-m cpu_only``), the ``*_nccl`` tests on GPUs; each test
runs several checks in one spawn. The file keeps its own spawn harness (``_worker``) so a
failing rank cannot leave its peers hanging in a collective. The copy-engine gather and the
copy-engine reduce-scatter (symmetric memory, SM100 FP8 block-scale path) have their own
``test_fp8_ce_*_nccl`` entry points.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_token_sharded_tp_collectives.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import functools
import sys
import traceback
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, gelu_tanh
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import (
    TokenShardedColumn,
    TokenShardedMLP,
    TokenShardedRow,
    convert_to_token_sharded_tp,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    Fp8BlockScaledActivation,
    TokenShardedSequenceSharder,
    TokenShardedTP,
    fp8_block_scale_prequant_ok,
    fp8_scale_cols,
    fp8_scales_mn_major,
    quantize_fp8_block,
    quantize_nvfp4,
    swizzled_sf_numel,
)
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# token_sharded_tp_test_utils is in tests/unittest/_torch/visual_gen (spawned workers inherit
# pytest's sys.path entry for it).
__extra_import_path__ = [".."]

from token_sharded_tp_test_utils import (
    fp8_block_checkpoint,
    padded_rows,
    swizzle_ref,
    unswizzle_ref,
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed harness
# =============================================================================


def _worker(rank, world_size, backend, test_fn, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size, device)
        dist.barrier()
    except BaseException:
        # Report and exit without a collective teardown: peers may be blocked in a
        # collective, and this process exiting lets mp.spawn terminate them.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    # Drop the DeviceMesh singleton a VisualGenMapping may have built before the process
    # group goes away (avoids NCCL destructor crashes at exit).
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None
    DeviceMeshTopologyImpl.tp_mesh = None
    dist.destroy_process_group()


def _run(world_size, test_fn, backend):
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs, have {torch.cuda.device_count()}")
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _worker, args=(world_size, backend, test_fn, port), nprocs=world_size, join=True
        )
    )


def _requires_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 quantize/GEMM requires SM100+")


def _requires_sm100f():
    # The rule (fp8_block_scale_prequant_ok) and the pinned quantizer need the SM100 family;
    # SM120 is Blackwell too but takes neither.
    if not torch.cuda.is_available() or not is_sm_100f():
        pytest.skip("The FP8 block-scale DeepGEMM path requires an SM100-family GPU")


# =============================================================================
# Reference helpers (identical on all ranks)
# =============================================================================

# (B, S): unpadded, a rank straddling the sample boundary, interior padding (B >= 2),
# tail padding (B = 1) with fully padded ranks at tp = 4, and a longer sequence.
_SHAPES = [(1, 8), (1, 5), (2, 5), (2, 8), (3, 7), (2, 256)]
# FP8 block scales also: unpadded m with m % 4 != 0 and m % 128 != 0 (418 -> 209 at tp 2,
# 420 -> 105 at tp 4, 53 at tp 8).
_FP8_SHAPES = _SHAPES + [(1, 418), (1, 420)]


def _rel_l2(a, b):
    """Relative L2 distance of ``a`` from ``b`` (fp32), 0 when both are empty."""
    if b.numel() == 0:
        return 0.0
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _real_row_mask(plan, device):
    return padded_rows(torch.ones(plan.batch_size, plan.seq_len, 1, device=device), plan)[
        plan.row_start : plan.row_start + plan.local_rows, 0
    ].bool()


def _helper():
    return TokenShardedTP(dist.group.WORLD)


def _check(ok, msg, device):
    """Assert on every rank together: a rank-local failure must not leave the other ranks
    waiting in the next collective (which would hang until the NCCL watchdog fires)."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert ok and flag.item() == 1, (
        f"rank {dist.get_rank()}: {msg if not ok else 'another rank failed'}"
    )


def _check_close(got, ref, device, rtol, atol, what):
    ok = got.shape == ref.shape and torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
    diff = None
    if got.shape == ref.shape and got.numel() > 0:  # a fully padded rank compares no rows
        diff = (got.float() - ref.float()).abs().max().item()
    _check(ok, f"{what}: max abs diff {diff}", device)


# =============================================================================
# reduce_scatter == all-reduce then slice (bitwise on integer-valued partials)
# =============================================================================


def _logic_reduce_scatter(rank, world_size, device):
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tp = _helper()
    for b, s in _SHAPES:
        plan = tp.begin(b, s)
        gen = torch.Generator().manual_seed(b * 1000 + s)
        partials = torch.randint(-8, 9, (world_size, b * s, 24), generator=gen)
        partial = partials[rank].to(device=device, dtype=dtype)
        rows = slice(plan.row_start, plan.row_start + plan.local_rows)
        ref = padded_rows(partial.view(b, s, -1), plan).clone()  # independent of the helper
        dist.all_reduce(ref)
        # [B * S] input (padded inside) and [B * S_pad] input.
        got = tp.reduce_scatter(partial)
        _check(torch.equal(got, ref[rows]), f"reduce_scatter [B*S] input {(b, s)}", device)
        got = tp.reduce_scatter(tp._add_padding(partial))
        _check(torch.equal(got, ref[rows]), f"reduce_scatter [B*S_pad] input {(b, s)}", device)


# =============================================================================
# all_gather (NVFP4 payload + SF, plain)
# =============================================================================


def _logic_all_gather(rank, world_size, device):
    tp = _helper()
    for b, s in _SHAPES:
        plan = tp.begin(b, s)
        rows = slice(plan.row_start, plan.row_start + plan.local_rows)
        for k in (128, 80):  # 80: sf_cols = 5, not a multiple of 4
            sf_cols = k // 16
            gen = torch.Generator().manual_seed(b * 100 + s + k)
            payload = torch.randint(0, 256, (b, s, k // 2), dtype=torch.uint8, generator=gen)
            sf_lin = torch.randint(0, 256, (b, s, sf_cols), dtype=torch.uint8, generator=gen)
            # Token-pad rows carry garbage on a real shard: poison them.
            payload_pad = padded_rows(payload, plan)
            sf_pad = padded_rows(sf_lin, plan)
            real = padded_rows(torch.ones(b, s, 1, dtype=torch.bool), plan)[:, 0]
            payload_pad[~real] = 0xAB
            sf_pad[~real] = 0xAB
            loc = Fp4QuantizedTensor(
                payload_pad[rows].to(device), swizzle_ref(sf_pad[rows], 0xAB).to(device)
            )
            got = tp.all_gather(loc)
            ok = (
                isinstance(got, Fp4QuantizedTensor)
                and got.is_sf_swizzled
                and torch.equal(got.fp4_tensor.cpu(), payload.reshape(b * s, -1))
                and got.scaling_factor.numel() == swizzled_sf_numel(b * s, sf_cols)
                and torch.equal(
                    unswizzle_ref(got.scaling_factor, b * s, sf_cols).cpu(),
                    sf_lin.reshape(b * s, -1),
                )
            )
            _check(ok, f"NVFP4 all_gather {(b, s, k)}", device)
        # Plain tensors (strided shard input).
        x = torch.randn(b, s, 40, generator=torch.Generator().manual_seed(s)).to(device)
        x_loc = tp.shard(x.transpose(0, 1).contiguous().transpose(0, 1))
        ok = torch.equal(x_loc, padded_rows(x, plan)[rows])
        ok = torch.equal(tp.all_gather(x_loc), x.reshape(b * s, -1)) and ok
        _check(ok, f"plain all_gather {(b, s)}", device)


# =============================================================================
# all_gather of an FP8 block-scale pair (bytes + packed scales, re-laid MN-major)
# =============================================================================


def _fp8_shards(b, s, k, plan, gen):
    """A global FP8 payload + packed scales and this rank's shard of them, as the quantizer
    returns them (fp8 ``[m, K]``, MN-major int32 ``[m, P]``); token-pad rows are poisoned."""
    cols = fp8_scale_cols(k)
    payload = torch.randint(0, 256, (b, s, k), dtype=torch.uint8, generator=gen)
    scales = torch.randint(0, 2**31 - 1, (b, s, cols), generator=gen).to(torch.int32)
    payload_pad, scales_pad = padded_rows(payload, plan), padded_rows(scales, plan)
    real = padded_rows(torch.ones(b, s, 1, dtype=torch.bool), plan)[:, 0]
    payload_pad[~real] = 0xAB
    scales_pad[~real] = -7
    rows = slice(plan.row_start, plan.row_start + plan.local_rows)
    loc = Fp8BlockScaledActivation(
        payload_pad[rows].view(torch.float8_e4m3fn), fp8_scales_mn_major(scales_pad[rows])
    )
    return payload.reshape(b * s, k), scales.reshape(b * s, cols), loc


def _logic_all_gather_fp8(rank, world_size, device):
    for row_align in (1, 4):
        tp = TokenShardedTP(dist.group.WORLD, row_align=row_align)
        for b, s in _FP8_SHAPES:
            plan = tp.begin(b, s)
            n, g = len(plan.entry_batch), plan.rows_per_entry
            for k in (128, 640):  # P = 1 and 2 packed scale columns
                gen = torch.Generator().manual_seed(b * 100 + s + k + row_align)
                payload, scales, loc = _fp8_shards(b, s, k, plan, gen)
                loc = Fp8BlockScaledActivation(loc.fp8.to(device), loc.scale.to(device))
                given = Fp8BlockScaledActivation(loc.fp8.view(n, g, k), loc.scale)  # [n, g, K]
                for got in (tp.all_gather(loc), tp.gather_input(None, given)):
                    ok = (
                        isinstance(got, Fp8BlockScaledActivation)
                        and got.fp8.dtype == torch.float8_e4m3fn
                        and torch.equal(got.fp8.view(torch.uint8).cpu(), payload)
                        and got.scale.dtype == torch.int32
                        and torch.equal(got.scale.cpu(), scales)
                        and got.scale.stride() == (1, pad_up(b * s, 4))
                    )
                    _check(ok, f"FP8 all_gather {(b, s, k)} row_align={row_align}", device)


# =============================================================================
# Rank disagreement on the token layout raises on every rank (no hang)
# =============================================================================


def _logic_rank_disagreement(rank, world_size, device):
    tp = _helper()
    seq = 9 if rank == world_size - 1 else 8
    with pytest.raises(ValueError, match="TP ranks disagree on the token layout"):
        tp.begin(1, seq)
    # The group is still usable afterwards, and an agreeing shape works.
    plan = tp.begin(1, 8)
    assert plan.local_rows == pad_up(8, world_size) // world_size


# =============================================================================
# Real fp4_quantize on the shards, gathered == fp4_quantize of all rows (NCCL)
# =============================================================================


def _logic_fp4_quantize_gather(rank, world_size, device):
    tp = _helper()
    for b, s in _SHAPES:
        tp.begin(b, s)
        for k in (256, 80):
            gen = torch.Generator().manual_seed(b * 10 + s + k)
            x = (torch.randn(b, s, k, generator=gen) * 3).to(device, torch.bfloat16)
            scale = (448.0 * 6.0 / x.float().abs().amax()).reshape(1)
            got = tp.all_gather(quantize_nvfp4(tp.shard(x), scale))
            ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(x.reshape(b * s, k), scale, 16, False)
            ok = torch.equal(got.fp4_tensor, ref_fp4) and torch.equal(
                unswizzle_ref(got.scaling_factor, b * s, k // 16),
                unswizzle_ref(ref_sf.reshape(-1), b * s, k // 16),
            )
            _check(ok, f"fp4_quantize shards gathered {(b, s, k)}", device)


# =============================================================================
# Converted real Linear / MLP / GatedMLP vs the plain modules (NCCL)
# =============================================================================


def _mapping(rank, world_size):
    """TP mapping as VisualGen builds it (VisualGenMapping sets up the TP communicators
    that TRT-LLM Linear/AllReduce construction relies on)."""
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

    vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
    assert vgm.tp_rank == rank
    return vgm.to_llm_mapping()


def _as_model(**modules):
    """A one-block model (``blocks`` container) holding ``modules``, for the converter."""
    block = nn.Module()
    for name, module in modules.items():
        setattr(block, name, module)
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    return model


def _logic_real_adapters(rank, world_size, device):
    """Real TRT-LLM modules built as for plain TP and converted by the rules: the row Linear
    reduce-scatters; MLP / GatedMLP gather, run on all tokens and reduce-scatter."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.mlp import MLP

    k_in, n_out = 384, 96  # 384 splits evenly over tp in {2, 3, 4}
    torch.manual_seed(0)
    # Small magnitudes keep the per-rank bf16 rounding of the K-partials (inherent to any
    # row-parallel split, all-reduce included) well inside the tolerance.
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.02
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.05 + 0.25  # counted twice -> caught
    ref_lin = Linear(k_in, n_out, bias=True, dtype=torch.bfloat16).to(device)
    ref_lin.load_weights([{"weight": weight, "bias": bias}])
    mapping = _mapping(rank, world_size)
    nccl = AllReduceStrategy.NCCL
    row_lin = Linear(
        k_in,
        n_out,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,
        allreduce_strategy=nccl,
    ).to(device)
    row_lin.load_weights([{"weight": weight, "bias": bias}])
    config = ModelConfig(mapping=mapping, allreduce_strategy=nccl)
    mlp = MLP(
        hidden_size=k_in,
        intermediate_size=k_in,
        bias=True,
        activation=gelu_tanh,  # Wan's FFN activation
        dtype=torch.bfloat16,
        config=config,
    ).to(device)
    gated = GatedMLP(
        hidden_size=k_in, intermediate_size=k_in, bias=False, dtype=torch.bfloat16, config=config
    ).to(device)
    gen = torch.Generator().manual_seed(3)
    for m in (mlp, gated):
        for name, prm in m.named_parameters():
            prm.data.copy_((torch.randn(prm.shape, generator=gen) * 0.02).to(prm.dtype))
    tp = _helper()
    mlps = (("mlp", mlp), ("gated", gated))
    mlp_refs = {}  # plain TP on all rows, before the conversion
    for b, s in _SHAPES:
        act = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(100 + s))
        act = act.to(device, torch.bfloat16)
        mlp_refs[(b, s)] = act, {n: m(act.reshape(b * s, k_in)).view(b, s, -1) for n, m in mlps}
    convert_to_token_sharded_tp(_as_model(row=row_lin, mlp=mlp, gated=gated), tp)
    k_loc = k_in // world_size
    for b, s in _SHAPES:
        plan = tp.begin(b, s)
        act = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        got = row_lin(act[..., rank * k_loc : (rank + 1) * k_loc]).reshape(plan.local_rows, -1)
        ref = padded_rows(ref_lin(act).view(b, s, n_out), plan)[
            plan.row_start : plan.row_start + plan.local_rows
        ]
        mask = _real_row_mask(plan, device)
        _check_close(got[mask], ref[mask], device, 1e-2, 1e-2, f"row adapter {(b, s)}")
    for b, s in _SHAPES:
        plan = tp.begin(b, s)
        act, refs = mlp_refs[(b, s)]
        mine = slice(plan.row_start, plan.row_start + plan.local_rows)
        mask = _real_row_mask(plan, device)
        for name, m in mlps:
            got = m(tp.local_view(tp.shard(act))).reshape(plan.local_rows, -1)
            want = padded_rows(refs[name], plan)[mine]
            _check_close(got[mask], want[mask], device, 2e-2, 2e-2, f"{name} adapter {(b, s)}")


# =============================================================================
# Converted static-NVFP4 column Linear and MLP vs the same modules on all rows (NCCL)
# =============================================================================


def _nvfp4_checkpoint(weight, act_amax):
    """ModelOpt static-NVFP4 layout for a bf16 [N, K] weight (uint8 payload, fp8 SF)."""
    e2m1_max, fp8_max = 6.0, 448.0
    weight = weight.cuda()
    weight_scale_2 = (weight.float().abs().amax() / (fp8_max * e2m1_max)).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(weight, 1.0 / weight_scale_2, 16, False, False)
    return {
        "weight": fp4.cpu(),
        "weight_scale": sf.view(torch.float8_e4m3fn).reshape(weight.shape[0], -1).cpu(),
        "weight_scale_2": weight_scale_2.cpu(),
        "input_scale": (act_amax / (fp8_max * e2m1_max)).reshape(1).float().cpu(),
    }


def _spy_all_gather(tp):
    """Record what each of ``tp``'s all-gathers moves."""
    moved = []
    gather = tp.all_gather

    def spy(act):
        moved.append(act)
        return gather(act)

    tp.all_gather = spy
    return moved


def _logic_adapters_nvfp4(rank, world_size, device):
    """Static-NVFP4 column Linear and MLP, converted before their weights load: bf16 rows are
    quantized with the consumer's input_scale before the all-gather, so the column GEMM
    matches the plain module bitwise; the row projection reduce-scatters."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    k_in, n_out = 256, 128 * world_size
    torch.manual_seed(1)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.1
    w_row = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    w_up = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    w_down = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    mapping = _mapping(rank, world_size)
    nccl = AllReduceStrategy.NCCL
    nvfp4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    def build():
        col = Linear(
            k_in,
            n_out,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=nvfp4,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        row = Linear(
            n_out,
            k_in,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            allreduce_strategy=nccl,
        ).to(device)
        config = ModelConfig(mapping=mapping, allreduce_strategy=nccl, quant_config=nvfp4)
        mlp = MLP(
            hidden_size=k_in,
            intermediate_size=n_out,
            bias=False,
            activation=gelu_tanh,  # Wan's FFN activation (MLP fuses GELU + NVFP4 for it)
            dtype=torch.bfloat16,
            config=config,
        ).to(device)
        return col, row, mlp

    def load(col, row, mlp, ckpt, up_ckpt, down_ckpt):
        col.load_weights([{**ckpt, "bias": bias}])
        row.load_weights([{"weight": w_row}])
        mlp.up_proj.load_weights([up_ckpt])
        mlp.down_proj.load_weights([down_ckpt])
        for lin in (col, mlp.up_proj, mlp.down_proj):
            lin.post_load_weights()

    for b, s in _SHAPES:
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        amax = x.float().abs().amax().cpu()
        ckpts = (
            _nvfp4_checkpoint(weight, amax),
            _nvfp4_checkpoint(w_up, amax),
            _nvfp4_checkpoint(w_down, torch.tensor(4.0)),
        )
        col_ref, row_ref, mlp_ref = build()
        load(col_ref, row_ref, mlp_ref, *ckpts)
        h_ref = col_ref(x.reshape(b * s, k_in)).view(b, s, -1)  # the Linear quantizes all rows
        y_ref = row_ref(h_ref)  # all-reduced
        f_ref = mlp_ref(x.reshape(b * s, k_in)).view(b, s, -1)  # all-reduced
        col, row, mlp = build()
        tp = _helper()
        convert_to_token_sharded_tp(
            _as_model(col=col, row=row, mlp=mlp), tp, exceptions={"col": "column"}
        )
        load(col, row, mlp, *ckpts)
        moved = _spy_all_gather(tp)
        plan = tp.begin(b, s)
        x_loc = tp.local_view(tp.shard(x))
        mask = _real_row_mask(plan, device)
        mine = slice(plan.row_start, plan.row_start + plan.local_rows)
        h = col(x_loc)
        _check(
            isinstance(moved[-1], Fp4QuantizedTensor) and torch.equal(h, h_ref),
            f"NVFP4 column adapter {(b, s)}",
            device,
        )
        y = row(h).reshape(plan.local_rows, -1)
        _check_close(
            y[mask], padded_rows(y_ref, plan)[mine][mask], device, 1e-2, 1e-2, f"row {(b, s)}"
        )
        f = mlp(x_loc).reshape(plan.local_rows, -1)
        _check(isinstance(moved[-1], Fp4QuantizedTensor), f"NVFP4 MLP gather {(b, s)}", device)
        _check_close(
            f[mask], padded_rows(f_ref, plan)[mine][mask], device, 2e-2, 2e-2, f"MLP {(b, s)}"
        )


# =============================================================================
# Converted FP8 block-scale column Linear / MLP == the same modules on all rows (NCCL)
# =============================================================================


def _fp8_quant_tactics(x):
    """``(pinned, cuda, triton)`` packed 1x128 quantizations of bf16 ``[M, K]`` ``x``: the
    helper's, and the two tactics the Linear's ``fp8_swap_ab_gemm`` autotunes between."""
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils import fp8_quantize

    pinned = quantize_fp8_block(x)
    cuda = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    a, sf = fp8_quantize.triton_fp8_quantize_1x128(x, use_ue8m0=True)
    triton = (a, deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf.transpose(0, 1)))
    return pinned, cuda, triton


def _same_quant(p, q):
    return torch.equal(p[0], q[0]) and torch.equal(p[1], q[1])


def _logic_fp8_quant_tactics(rank, world_size, device):
    """The pinned quantize is the CUDA tactic, bitwise; whether the Triton tactic agrees is
    reported (the stock path is bitwise with the gathered bytes only when it does)."""
    for m, k in ((8, 128), (209, 256), (105, 640), (4096, 5120)):
        x = (torch.randn(m, k, generator=torch.Generator().manual_seed(m + k)) * 3).to(
            device, torch.bfloat16
        )
        pinned, cuda, triton = _fp8_quant_tactics(x)
        ok = (
            _same_quant(pinned, cuda)
            and pinned.fp8.shape == (m, k)
            and pinned.scale.shape == (m, fp8_scale_cols(k))
            and pinned.scale.stride() == (1, pad_up(m, 4))
        )
        _check(ok, f"pinned FP8 quantize == CUDA tactic {(m, k)}", device)
        agree = torch.tensor([int(_same_quant(triton, cuda))], device=device)
        dist.all_reduce(agree, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(
                f"FP8_QUANT_TACTICS m={m} k={k} triton_equals_cuda={bool(agree.item())}",
                flush=True,
            )


def _logic_adapters_fp8_block_scales(rank, world_size, device):
    """Converted real FP8 block-scale modules (DeepGEMM path): a column projection and an MLP
    fed this rank's bf16 rows quantize them to FP8 + packed scales before the all-gather, so
    the GEMMs see exactly the bytes they would quantize themselves on all rows: the column
    output is bitwise equal to the plain Linear's (untuned, and after autotuning unless the
    tuner picked the Triton quantizer and it disagrees with the CUDA one, which is reported);
    a BF16 consumer never receives the pair."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    k_in, n_out = 256, 128 * world_size
    torch.manual_seed(5)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.1
    w_up = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    w_down = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    ckpt_col = {**fp8_block_checkpoint(weight), "bias": bias}
    ckpt_up, ckpt_down = fp8_block_checkpoint(w_up), fp8_block_checkpoint(w_down)
    mapping = _mapping(rank, world_size)
    fp8bs = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    def build():
        col = Linear(
            k_in,
            n_out,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=fp8bs,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        config = ModelConfig(
            mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL, quant_config=fp8bs
        )
        mlp = MLP(
            hidden_size=k_in,
            intermediate_size=n_out,
            bias=False,
            activation=gelu_tanh,  # Wan's FFN activation
            dtype=torch.bfloat16,
            config=config,
        ).to(device)
        return col, mlp

    def load(col, mlp):
        col.load_weights([ckpt_col])
        mlp.up_proj.load_weights([ckpt_up])
        mlp.down_proj.load_weights([ckpt_down])
        for lin in (col, mlp.up_proj, mlp.down_proj):
            lin.post_load_weights()

    col_ref, mlp_ref = build()
    load(col_ref, mlp_ref)
    _check(
        fp8_block_scale_prequant_ok(col_ref) and fp8_block_scale_prequant_ok(mlp_ref.up_proj),
        "the FP8 block-scale rule engages on this GPU",
        device,
    )
    adapters = {}
    for row_align in (1, 4):
        col, mlp = build()
        tp = TokenShardedTP(dist.group.WORLD, row_align=row_align)
        convert_to_token_sharded_tp(_as_model(col=col, mlp=mlp), tp, exceptions={"col": "column"})
        load(col, mlp)
        adapters[row_align] = (tp, col)
        moved = _spy_all_gather(tp)
        for b, s in _FP8_SHAPES:
            x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
                device, torch.bfloat16
            )
            x2d = x.reshape(b * s, k_in)
            h_ref = col_ref(x2d).view(b, s, -1)  # today's path: the Linear quantizes all rows
            u_ref = mlp_ref.up_proj(x2d)
            f_ref = mlp_ref(x2d).view(b, s, -1)  # all-reduced
            plan = tp.begin(b, s)
            x_loc = tp.local_view(tp.shard(x))
            h = col(x_loc)
            a = moved[-1]
            ok = (
                isinstance(a, Fp8BlockScaledActivation)
                and a.fp8.dtype == torch.float8_e4m3fn
                and a.fp8.shape == (plan.local_rows, k_in)
                and 2 * a.fp8.numel() * a.fp8.element_size()
                == x_loc.numel() * x_loc.element_size()  # half the bf16 bytes
                and a.scale.shape == (plan.local_rows, fp8_scale_cols(k_in))
                and torch.equal(h, h_ref)
            )
            _check(ok, f"FP8 column adapter {(b, s)} row_align={row_align}", device)
            # The MLP's up-projection on the gathered pair (what TokenShardedMLP feeds it).
            u = mlp.up_proj(tp.gather_input(mlp.up_proj, x_loc))
            _check(
                isinstance(moved[-1], Fp8BlockScaledActivation) and torch.equal(u, u_ref),
                f"FP8 up_proj {(b, s)} row_align={row_align}",
                device,
            )
            f = mlp(x_loc).reshape(plan.local_rows, -1)
            _check(
                isinstance(moved[-1], Fp8BlockScaledActivation), f"FP8 MLP gather {(b, s)}", device
            )
            mask = _real_row_mask(plan, device)
            mine = slice(plan.row_start, plan.row_start + plan.local_rows)
            _check_close(
                f[mask],
                padded_rows(f_ref, plan)[mine][mask],
                device,
                2e-2,
                2e-2,
                f"FP8 MLP {(b, s)}",
            )
    # Today's path after VisualGen's warmup: the Linear autotunes its quantizer per row count
    # and may switch to the Triton kernel at large M. The gathered FP8 (pinned to the CUDA
    # kernel) then matches the tuned Linear bitwise exactly when the two kernels agree.
    from tensorrt_llm._torch.autotuner import autotune

    tp, col = adapters[1]  # the served plans: from_model_config leaves row_align at 1
    b, s = 2, 4096
    x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
        device, torch.bfloat16
    )
    x2d = x.reshape(b * s, k_in)
    _, cuda, triton = _fp8_quant_tactics(x2d)
    agree = torch.tensor([int(_same_quant(triton, cuda))], device=device)
    dist.all_reduce(agree, op=dist.ReduceOp.MIN)
    with autotune(skip_dynamic_tuning_buckets=True):  # as the VisualGen pipeline warms up
        col_ref(x2d)
    h_ref_tuned = col_ref(x2d).view(b, s, -1)
    tp.begin(b, s)
    h = col(tp.local_view(tp.shard(x)))
    equal_tuned = torch.equal(h, h_ref_tuned)
    if rank == 0:
        print(
            f"FP8_TUNED_REFERENCE m={b * s} k={k_in} equal_tuned={equal_tuned} "
            f"triton_equals_cuda={bool(agree.item())}",
            flush=True,
        )
    # Falsifiable form of the claim: the Linear fed the pinned CUDA quantize of ALL rows must
    # match the gathered path bitwise (same bytes into the same GEMM); the autotuned Linear may
    # pick the Triton quantizer at this M and then differs by the two kernels' rounding only.
    h_ref_cuda = col_ref(quantize_fp8_block(x2d)).view(b, s, -1)
    _check(
        torch.equal(h, h_ref_cuda),
        "the gathered-FP8 path differs from the Linear on the CUDA-quantized rows",
        device,
    )
    _check_close(
        h, h_ref_tuned, device, 2e-2, 2e-2, "gathered-FP8 path vs the autotuned FP8 Linear"
    )
    _check(
        equal_tuned or not bool(agree.item()),
        "the autotuned FP8 Linear differs from the gathered-FP8 path although the Triton and "
        "CUDA quantizers agree on this input",
        device,
    )
    bf16_col = Linear(
        k_in,
        n_out,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    plan = tp.begin(2, 8)
    pair = quantize_fp8_block(
        torch.randn(plan.local_rows, k_in, device=device, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="not an FP8 block-scale Linear on the DeepGEMM path"):
        tp.gather_input(bf16_col, pair)


# =============================================================================
# torch.compile(fullgraph=True) and CUDA-graph capture of a boundary chain
# =============================================================================


def _boundary_chain_parts(rank, world_size, device, d=256, n_col=128):
    torch.manual_seed(2)
    col_w = torch.randn(n_col * world_size, d, dtype=torch.bfloat16) * 0.05
    row_w = torch.randn(d, n_col * world_size, dtype=torch.bfloat16) * 0.05
    mapping = _mapping(rank, world_size)
    col = Linear(
        d,
        n_col * world_size,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    col.load_weights([{"weight": col_w, "bias": torch.zeros(n_col * world_size)}])
    row = Linear(
        n_col * world_size,
        d,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,  # built as for plain TP; the conversion turns it into a reduce-scatter
        allreduce_strategy=AllReduceStrategy.NCCL,
    ).to(device)
    row.load_weights([{"weight": row_w, "bias": torch.full((d,), 0.1)}])
    return col, row


def _converted_chain_parts(rank, world_size, device, d):
    tp = _helper()
    col, row = _boundary_chain_parts(rank, world_size, device, d)
    convert_to_token_sharded_tp(_as_model(col=col, row=row), tp, exceptions={"col": "column"})
    return tp, col, row


def _make_chain(tp, col, row, ln_w, ln_b, fp4_scale):
    """One block boundary on this rank's [n, g, D] sample groups: AdaLN norm -> converted
    column -> converted row -> gated residual -> LayerNorm + NVFP4 quantize -> FP4 all-gathers."""

    def chain(x_loc, fp4_payload, fp4_sf, table, h_given):
        d = x_loc.shape[-1]
        shift, scale, gate = tp.per_sample_table(table)[:, :, None].unbind(1)  # [n, 1, D] each
        h = F.layer_norm(x_loc.float(), (d,)) * (1 + scale) + shift
        q = col(h.to(x_loc.dtype))  # all tokens, this rank's features
        x = (x_loc.float() + row(q).float() * gate).to(x_loc.dtype)
        h2 = F.layer_norm(x.float(), (d,), ln_w, ln_b).to(x.dtype)
        g2 = tp.all_gather(quantize_nvfp4(h2, fp4_scale))  # the chain's own quantize
        # A given FP4 input as fused norms emit it ([n, g, K/2]), through the adapters' path.
        given = Fp4QuantizedTensor(fp4_payload.view(*x_loc.shape[:2], -1), fp4_sf)
        g = tp.gather_input(None, given)
        g3 = tp.all_gather(quantize_nvfp4(h_given, fp4_scale))  # quantize of a given bf16 input
        return (
            x,
            g2.fp4_tensor,
            g2.scaling_factor,
            g.fp4_tensor,
            g.scaling_factor,
            g3.fp4_tensor,
            g3.scaling_factor,
        )

    return chain


def _chain_inputs(tp, b, s, d, device):
    gen = torch.Generator().manual_seed(b * 7 + s)
    x = torch.randn(b, s, d, generator=gen).to(device, torch.bfloat16)
    table = torch.randn(b, 3, d, generator=gen).to(device) * 0.1
    plan = tp.plan
    payload = torch.randint(0, 256, (plan.local_rows, d // 2), dtype=torch.uint8, generator=gen)
    sf = torch.randint(
        0, 256, (swizzled_sf_numel(plan.local_rows, d // 16),), dtype=torch.uint8, generator=gen
    )
    h_given = (torch.randn(plan.local_rows, d, generator=gen) * 2).to(device, torch.bfloat16)
    return tp.local_view(tp.shard(x)), payload.to(device), sf.to(device), table, h_given


def _logic_compile_fullgraph(rank, world_size, device):
    import torch._dynamo

    d = 256
    tp, col, row = _converted_chain_parts(rank, world_size, device, d)
    ln_w = torch.ones(d, device=device)
    ln_b = torch.zeros(d, device=device)
    fp4_scale = torch.tensor([448.0 * 6.0 / 8.0], device=device)
    chain = _make_chain(tp, col, row, ln_w, ln_b, fp4_scale)

    def compare(got, ref, b, s, what):
        # Residual stream (bf16 rows): Inductor may round the fused LayerNorm / residual
        # one bf16 ULP differently from eager, so compare at bf16 resolution.
        _check_close(got[0], ref[0], device, 1e-2, 1e-2, f"{what} residual {(b, s)}")
        # Gathered FP4 of a given FP4 input, and compiled quantize + gather of a given bf16
        # input (payload + regrouped SF content): bitwise.
        rows, sf_cols = b * s, d // 16
        for i, name in ((3, "given FP4"), (5, "quantized given bf16")):
            ok = torch.equal(got[i], ref[i]) and torch.equal(
                unswizzle_ref(got[i + 1], rows, sf_cols), unswizzle_ref(ref[i + 1], rows, sf_cols)
            )
            _check(ok, f"{what} {name} all_gather {(b, s)}", device)
        # The chain's own quantize of its LayerNorm output: Inductor may round the LN one
        # ULP differently from eager, which can move an FP4 code (or a 16-element block's
        # scale), so check the layout and that only a small fraction of codes differ.
        same_layout = got[1].shape == ref[1].shape and got[2].numel() == ref[2].numel()
        mismatch = (got[1] != ref[1]).float().mean().item() if same_layout else 1.0
        _check(mismatch < 0.05, f"{what} own-quantize {(b, s)}: FP4 codes {mismatch:.4f}", device)

    for b, s in [(2, 256), (2, 5), (1, 5)]:  # SF fast path, SF regroup, padded
        tp.begin(b, s)
        args = _chain_inputs(tp, b, s, d, device)
        torch._dynamo.reset()
        compiled = torch.compile(chain, fullgraph=True)  # raises on any graph break
        compare(compiled(*args), chain(*args), b, s, "compiled")

    # As the pipeline does: one compiled callable, a new shape without a reset (the plan's
    # ints are guarded, so it recompiles), then a revisit of the first shape.
    torch._dynamo.reset()
    compiled = torch.compile(chain, fullgraph=True)
    for b, s in [(2, 256), (1, 5), (2, 256)]:
        tp.begin(b, s)
        args = _chain_inputs(tp, b, s, d, device)
        compare(compiled(*args), chain(*args), b, s, "recompiled")


def _logic_cuda_graph(rank, world_size, device):
    d = 256
    tp, col, row = _converted_chain_parts(rank, world_size, device, d)
    ln_w = torch.ones(d, device=device)
    ln_b = torch.zeros(d, device=device)
    fp4_scale = torch.tensor([448.0 * 6.0 / 8.0], device=device)
    chain = _make_chain(tp, col, row, ln_w, ln_b, fp4_scale)
    for b, s in [(2, 256), (2, 5), (1, 5)]:
        tp.begin(b, s)
        args = _chain_inputs(tp, b, s, d, device)
        static_args = [a.clone() for a in args]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # eager warmups (as CUDAGraphRunner.WARMUP_STEPS)
                chain(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = chain(*static_args)
        new_args = [
            args[0] * 0.5 + 0.25,
            args[1] ^ 0x5A,
            args[2] ^ 0x3C,
            args[3] * 0.5,
            args[4] * 0.5,
        ]
        for dst, src in zip(static_args, new_args):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        ref = chain(*new_args)
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"CUDA graph replay vs eager {(b, s)}", device)
        del graph


# =============================================================================
# The same, with an FP8 block-scale consumer (FP8 + scales all-gather)
# =============================================================================


def _fp8_chain_parts(rank, world_size, device, d=256, n_col=128, gather_mode=None):
    """A converted FP8 block-scale column Linear (gathers FP8 + packed scales) and a converted
    bf16 row Linear (reduce-scatters); ``gather_mode`` selects the helper's gather."""
    torch.manual_seed(4)
    col_w = torch.randn(n_col * world_size, d, dtype=torch.bfloat16) * 0.05
    col = Linear(
        d,
        n_col * world_size,
        bias=True,
        dtype=torch.bfloat16,
        mapping=_mapping(rank, world_size),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    _, row = _boundary_chain_parts(rank, world_size, device, d, n_col)
    tp = (
        _helper()
        if gather_mode is None
        else TokenShardedTP(dist.group.WORLD, gather_mode=gather_mode)
    )
    convert_to_token_sharded_tp(_as_model(col=col, row=row), tp, exceptions={"col": "column"})
    col.load_weights([{**fp8_block_checkpoint(col_w), "bias": torch.zeros(n_col * world_size)}])
    col.post_load_weights()
    return tp, col, row


def _make_fp8_chain(tp, col, row):
    """AdaLN norm (row-local) -> converted FP8 block-scale column Linear on a given bf16 input
    (FP8 + scales all-gather, then DeepGEMM) -> converted row Linear -> gated residual, plus
    the quantize + gather of that input on its own."""

    def chain(x_loc, table, h_given):
        d = x_loc.shape[-1]
        shift, scale, gate = tp.per_sample_table(table)[:, :, None].unbind(1)  # [n, 1, D] each
        h = (F.layer_norm(x_loc.float(), (d,)) * (1 + scale) + shift).to(x_loc.dtype)
        q = col(h_given)  # all tokens, this rank's features; exact input -> bitwise GEMM
        x = (h.float() + row(q).float() * gate).to(x_loc.dtype)
        g = tp.all_gather(quantize_fp8_block(h_given))
        return x, q, g.fp8, g.scale

    return chain


def _fp8_chain_inputs(tp, b, s, d, device):
    gen = torch.Generator().manual_seed(b * 11 + s)
    x = torch.randn(b, s, d, generator=gen).to(device, torch.bfloat16)
    table = torch.randn(b, 3, d, generator=gen).to(device) * 0.1
    x_loc = tp.local_view(tp.shard(x))
    h_given = (torch.randn(*x_loc.shape, generator=gen) * 2).to(device, torch.bfloat16)
    return x_loc, table, h_given


def _logic_compile_fullgraph_fp8(rank, world_size, device):
    import torch._dynamo

    d = 256
    tp, col, row = _fp8_chain_parts(rank, world_size, device, d)
    chain = _make_fp8_chain(tp, col, row)

    def compare(got, ref, b, s, what):
        # The residual carries the Inductor-fused LayerNorm: bf16 resolution.
        _check_close(got[0], ref[0], device, 1e-2, 1e-2, f"{what} residual {(b, s)}")
        # The GEMM on the gathered FP8 pair and the pair itself: bitwise. Values only: Inductor
        # lays the gathered scale out with strides (1, M), which the GEMM runner re-strides.
        ok = all(torch.equal(g, r) for g, r in zip(got[1:], ref[1:]))
        _check(ok, f"{what} FP8 gather + GEMM {(b, s)}", device)

    for b, s in [(2, 256), (2, 5), (1, 5), (1, 418)]:  # m % 4 == 0 and != 0, padded
        tp.begin(b, s)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        torch._dynamo.reset()
        compiled = torch.compile(chain, fullgraph=True)  # raises on any graph break
        compare(compiled(*args), chain(*args), b, s, "compiled")

    torch._dynamo.reset()
    compiled = torch.compile(chain, fullgraph=True)
    for b, s in [(2, 256), (1, 5), (2, 256)]:
        tp.begin(b, s)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        compare(compiled(*args), chain(*args), b, s, "recompiled")


def _logic_cuda_graph_fp8(rank, world_size, device):
    d = 256
    tp, col, row = _fp8_chain_parts(rank, world_size, device, d)
    chain = _make_fp8_chain(tp, col, row)
    for b, s in [(2, 256), (2, 5), (1, 5), (1, 418)]:
        tp.begin(b, s)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        static_args = [a.clone() for a in args]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # eager warmups (as CUDAGraphRunner.WARMUP_STEPS)
                chain(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = chain(*static_args)
        new_args = [args[0] * 0.5 + 0.25, args[1] * 0.5, args[2] * 0.75 - 0.5]
        for dst, src in zip(static_args, new_args):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        ref = chain(*new_args)
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"FP8 CUDA graph replay vs eager {(b, s)}", device)
        del graph


# =============================================================================
# Copy-engine all-gather (symmetric memory): transport, per-chunk GEMM, adapters, graphs
# =============================================================================
#
# The copy-engine gather needs the FP8 block-scale DeepGEMM path (SM100 family) and a TP group
# whose symmetric-memory probe accepts it; a rejected probe fails these checks loudly (it names
# the reason) rather than skipping, since the product path would silently lose the gather.


def _ce_module():
    from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_gather

    return token_sharded_ce_gather


def _ce_reason(tp):
    """Why ``tp``'s copy-engine gather is not effective (for a failure message)."""
    return getattr(getattr(tp, "_ce_state", None), "reason", "no gather state")


def _logic_fp8_block_gemm_out(rank, world_size, device):
    """The per-chunk GEMM into a row block of one output is bitwise the stock op on all rows
    (same kernel, same operands), including chunks whose scale slice must be re-strided."""
    tsc = _ce_module()
    for k, n in ((640, 256), (5120, 384)):  # P = 2 and 10 packed scale columns
        weight = torch.randn(n, k, generator=torch.Generator().manual_seed(k)) * 0.05
        col = Linear(
            k,
            n,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        ).to(device)
        col.load_weights([fp8_block_checkpoint(weight.to(torch.bfloat16))])
        col.post_load_weights()
        tsc.validate_fp8_block_consumer(col)
        for m_total, chunks in ((420, 4), (418, 2), (8192, 4)):  # m % 4 != 0 chunks too
            x = (torch.randn(m_total, k, generator=torch.Generator().manual_seed(m_total)) * 2).to(
                device, torch.bfloat16
            )
            fp8, sf = quantize_fp8_block(x)
            ref = torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
                fp8, sf, col.weight, col.weight_scale, torch.bfloat16, True
            )
            out = torch.empty_like(ref)
            m = m_total // chunks
            for c in range(chunks):
                r0, r1 = c * m, (c + 1) * m
                tsc.fp8_block_gemm_out(
                    fp8[r0:r1],
                    tsc.mn_major_scales(sf[r0:r1]),
                    col.weight,
                    col.weight_scale,
                    out[r0:r1],
                )
            _check(
                torch.equal(out, ref),
                f"per-chunk GEMM != stock op (K={k}, N={n}, M={m_total}, chunks={chunks})",
                device,
            )


def _logic_ce_transport(rank, world_size, device):
    """CeAllGather vs tp.all_gather: 20 boundaries per shape with slot alternation, the chunk
    views in DeepGEMM's layout, a reset after a rank-symmetric failure, the pool growing across
    shapes and never shrinking, protocol misuse raising, and release emptying the registry."""
    tsc = _ce_module()
    group = dist.group.WORLD
    state = tsc.register_state(tsc.CeGatherState(group, group.group_name))
    tp = _helper()
    k = 5120
    state.note_consumer(k)
    slot_bytes = []

    def boundary(m, seed):
        x = (torch.randn(m, k, generator=torch.Generator().manual_seed(seed)) * 3).to(
            device, torch.bfloat16
        )
        loc = quantize_fp8_block(x)
        ref = tp.all_gather(loc)  # unpadded plan: [tp * m, K] fp8, (tp * m, P) scales
        slot = state.next_slot()
        transport = state.transport
        transport.issue(loc.fp8, loc.scale, slot)
        got_fp8 = torch.empty(world_size * m, k, dtype=torch.float8_e4m3fn, device=device)
        got_sf = torch.empty(world_size * m, loc.scale.shape[1], dtype=torch.int32, device=device)
        layout_ok = True
        for src in (rank, *transport.consume_order):
            fp8_s, sf_s = transport.chunk(src, slot)
            layout_ok &= (
                fp8_s.shape == (m, k)
                and fp8_s.is_contiguous()
                and fp8_s.data_ptr() % 16 == 0
                and sf_s.shape == (m, loc.scale.shape[1])
                and sf_s.stride() == (1, pad_up(m, 4))
                and sf_s.data_ptr() % 16 == 0
            )
            got_fp8[src * m : (src + 1) * m].copy_(fp8_s)
            got_sf[src * m : (src + 1) * m].copy_(sf_s)
        transport.done(slot)
        equal = torch.equal(got_fp8.view(torch.uint8), ref.fp8.view(torch.uint8)) and torch.equal(
            got_sf, ref.scale
        )
        return slot, layout_ok, equal, loc

    for m in (105, 209, 18900):
        plan = tp.begin(1, m * world_size)
        assert plan.local_rows == m
        state.prepare(plan)
        _check(
            state.effective, f"copy-engine gather probe rejected this group: {state.reason}", device
        )
        transport = state.transport
        layout = tsc.RegionLayout(m, k)
        _check(
            transport.covers(layout) and transport.slot_nbytes >= layout.slot_nbytes(world_size),
            f"pool holds {transport.slot_nbytes} bytes per slot; m={m} needs {layout.slot_nbytes(world_size)}",
            device,
        )
        slot_bytes.append(transport.slot_nbytes)
        for i in range(20):
            slot, layout_ok, equal, _ = boundary(m, rank * 1000 + m + i)
            _check(
                slot == i % 2, f"slot {slot} at boundary {i} (prepare resets the sequence)", device
            )
            _check(layout_ok, f"chunk views are not in DeepGEMM's layout at m={m}", device)
            _check(equal, f"CE gather != NCCL all_gather at m={m}, boundary {i}", device)
        # A rank-symmetric failure after issue(): reset() releases the slot on every rank and the
        # protocol continues in step.
        _, _, _, loc = boundary(m, 7)
        transport.issue(loc.fp8, loc.scale, state.next_slot())
        transport.reset()
        state.begin_forward()  # must not raise
        for i in range(2):
            _, _, equal, _ = boundary(m, 100 + i)
            _check(
                equal, f"CE gather != NCCL all_gather after reset() at m={m}, boundary {i}", device
            )
    _check(
        slot_bytes[0] < slot_bytes[1] < slot_bytes[2], f"pool did not grow: {slot_bytes}", device
    )
    plan = tp.begin(1, 105 * world_size)
    state.prepare(plan)
    _check(
        state.transport.slot_nbytes == slot_bytes[2], "the pool shrank for a smaller shape", device
    )
    transport = state.transport
    with pytest.raises(RuntimeError, match="needs an issue"):
        transport.chunk(rank, 0)
    with pytest.raises(RuntimeError, match="without an issue"):
        transport.done(0)
    too_big = torch.zeros(2 * 18900, k, device=device, dtype=torch.bfloat16)  # 2x the pool
    with pytest.raises(RuntimeError, match="reserve"):
        transport.issue(*quantize_fp8_block(too_big), 0)
    del too_big
    state.close()
    with pytest.raises(KeyError):
        tsc.get_state(group.group_name)
    w = torch.zeros(128, k, dtype=torch.float8_e4m3fn, device=device)
    ws = torch.zeros(tsc.packed_weight_scale_cols(k), 128, dtype=torch.int32, device=device).t()
    with pytest.raises(RuntimeError, match="no CeGatherState is registered"):
        torch.ops.trtllm.token_sharded_fp8_ce_gather_gemm(
            loc.fp8,
            loc.scale,
            w,
            ws,
            group.group_name,
            rank,
            world_size,
            1,
            105 * world_size,
            105 * world_size,
        )


def _logic_adapters_ce_gather(rank, world_size, device):
    """FP8 block-scale column Linear and MLP converted with ``gather_mode="copy_engine"`` are
    bitwise the NCCL-gather adapters, eager and torch.compile'd; the helper's all_gather is
    bypassed at those boundaries and the engagement rule holds."""
    import torch._dynamo

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    k_in, n_out = 640, 128 * world_size  # P = 2 packed scale columns
    torch.manual_seed(6)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.1
    w_up = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    w_down = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    ckpt_col = {**fp8_block_checkpoint(weight), "bias": bias}
    ckpt_up, ckpt_down = fp8_block_checkpoint(w_up), fp8_block_checkpoint(w_down)
    mapping = _mapping(rank, world_size)
    fp8bs = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    def build(gather_mode):
        col = Linear(
            k_in,
            n_out,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=fp8bs,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        config = ModelConfig(
            mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL, quant_config=fp8bs
        )
        mlp = MLP(
            hidden_size=k_in,
            intermediate_size=n_out,
            bias=False,
            activation=gelu_tanh,
            dtype=torch.bfloat16,
            config=config,
        ).to(device)
        tp = TokenShardedTP(dist.group.WORLD, gather_mode=gather_mode)
        convert_to_token_sharded_tp(_as_model(col=col, mlp=mlp), tp, exceptions={"col": "column"})
        col.load_weights([ckpt_col])
        mlp.up_proj.load_weights([ckpt_up])
        mlp.down_proj.load_weights([ckpt_down])
        for lin in (col, mlp.up_proj, mlp.down_proj):
            lin.post_load_weights()
        return tp, col, mlp

    tp_ref, col_ref, mlp_ref = build("nccl")
    tp_ce, col_ce, mlp_ce = build("copy_engine")
    moved = _spy_all_gather(tp_ce)
    for b, s in _FP8_SHAPES:
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        tp_ref.begin(b, s)
        tp_ce.begin(b, s)
        _check(
            tp_ce.effective_gather_mode == "copy_engine",
            f"copy-engine gather not effective: {_ce_reason(tp_ce)}",
            device,
        )
        x_loc = tp_ce.local_view(tp_ce.shard(x))
        h_ref, f_ref = col_ref(x_loc), mlp_ref(x_loc)
        n_moved = len(moved)
        h, f = col_ce(x_loc), mlp_ce(x_loc)
        _check(torch.equal(h, h_ref), f"CE column adapter != NCCL {(b, s)}", device)
        _check(torch.equal(f, f_ref), f"CE MLP adapter != NCCL {(b, s)}", device)
        _check(len(moved) == n_moved, f"the CE adapters still all-gathered {(b, s)}", device)
        rule = (
            tp_ce.uses_ce_gather(col_ce, x_loc)
            and tp_ce.uses_ce_gather(mlp_ce.up_proj, quantize_fp8_block(x_loc))
            and not tp_ce.uses_ce_gather(col_ce, x_loc, {"active": True})  # LoRA: dense input
            and not tp_ce.uses_ce_gather(mlp_ce.down_proj, x_loc)  # not a column consumer
            and not tp_ce.uses_ce_gather(col_ce, x_loc.float())  # not bf16
            and not tp_ref.uses_ce_gather(col_ref, x_loc)  # NCCL mode
        )
        _check(rule, f"uses_ce_gather rule {(b, s)}", device)
    del tp_ce.all_gather  # drop the spy (an instance attribute) before compiling
    torch._dynamo.reset()
    col_c = torch.compile(lambda t: col_ce(t))
    mlp_c = torch.compile(lambda t: mlp_ce(t))
    for b, s in [(2, 256), (1, 5), (1, 418), (2, 256)]:
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(2 * s)).to(
            device, torch.bfloat16
        )
        tp_ref.begin(b, s)
        tp_ce.begin(b, s)
        x_loc = tp_ce.local_view(tp_ce.shard(x))
        _check(
            torch.equal(col_c(x_loc), col_ref(x_loc)),
            f"compiled CE column != NCCL {(b, s)}",
            device,
        )
        _check(
            torch.equal(mlp_c(x_loc), mlp_ref(x_loc)), f"compiled CE MLP != NCCL {(b, s)}", device
        )
    tp_ce.close()
    tp_ce.close()  # idempotent


def _logic_compile_fullgraph_ce(rank, world_size, device):
    """torch.compile(fullgraph=True) of the FP8 boundary chain with the copy-engine gather on:
    one graph (the fused op is opaque), bitwise the eager chain on the GEMM output."""
    import torch._dynamo

    d = 256
    tp, col, row = _fp8_chain_parts(rank, world_size, device, d, gather_mode="copy_engine")
    chain = _make_fp8_chain(tp, col, row)

    def compare(got, ref, b, s, what):
        _check_close(got[0], ref[0], device, 1e-2, 1e-2, f"{what} CE residual {(b, s)}")
        ok = all(torch.equal(g, r) for g, r in zip(got[1:], ref[1:]))
        _check(ok, f"{what} CE gather + GEMM {(b, s)}", device)

    def run_compiled(compiled, args):
        try:
            return compiled(*args)
        except torch._dynamo.exc.Unsupported as e:
            if "all_reduce" in str(e).lower() or "allreduce" in str(e).lower():
                print(
                    f"SKIP _logic_compile_fullgraph_ce: the AllReduce in the overlay is not "
                    f"traceable under fullgraph=True ({e})",
                    flush=True,
                )
                return None
            raise

    for b, s in [(2, 256), (2, 5), (1, 5), (1, 418)]:
        tp.begin(b, s)
        _check(tp.effective_gather_mode == "copy_engine", _ce_reason(tp), device)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        torch._dynamo.reset()
        got = run_compiled(torch.compile(chain, fullgraph=True), args)
        if got is None:
            tp.close()
            return
        compare(got, chain(*args), b, s, "compiled")
    torch._dynamo.reset()
    compiled = torch.compile(chain, fullgraph=True)
    for b, s in [(2, 256), (1, 5), (2, 256)]:
        tp.begin(b, s)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        compare(run_compiled(compiled, args), chain(*args), b, s, "recompiled")
    tp.close()


def _logic_cuda_graph_ce(rank, world_size, device):
    """CUDA-graph capture + replay of the FP8 boundary chain with the copy-engine gather on:
    begin() reserves the pool eagerly, two eager boundaries precede the capture (the consumed
    waits are in the graph), the replay is bitwise the eager chain on new inputs, and a larger
    shape after the captures is refused rather than growing the pool under the graphs."""
    d = 256
    tp, col, row = _fp8_chain_parts(rank, world_size, device, d, gather_mode="copy_engine")
    chain = _make_fp8_chain(tp, col, row)
    for b, s in [(2, 256), (2, 5), (1, 5), (1, 418)]:
        tp.begin(b, s)  # eager: probes once, sizes the pool for this shape
        _check(tp.effective_gather_mode == "copy_engine", _ce_reason(tp), device)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        static_args = [a.clone() for a in args]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # eager warmups (as CUDAGraphRunner.WARMUP_STEPS), >= slots
                chain(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = chain(*static_args)
        new_args = [args[0] * 0.5 + 0.25, args[1] * 0.5, args[2] * 0.75 - 0.5]
        for dst, src in zip(static_args, new_args):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        ref = chain(*new_args)
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"CE CUDA graph replay vs eager {(b, s)}", device)
        # A second replay reuses the slot protocol in steady state.
        graph.replay()
        torch.cuda.synchronize()
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"CE CUDA graph second replay vs eager {(b, s)}", device)
        del graph
    # The pool holds the captured graphs' buffers: a shape that needs a larger pool must now
    # be refused (rank-symmetrically, before any collective) instead of reallocating.
    with pytest.raises(RuntimeError, match="would grow after a CUDA graph"):
        tp.begin(2, 512)
    tp.close()


# =============================================================================
# Copy-engine reduce-scatter (symmetric memory): the fused GEMM + RS op, adapters, graphs
# =============================================================================
#
# The copy-engine reduce-scatter takes the FP8 block-scale row Linears (SM100 DeepGEMM path) of
# a TP group whose symmetric-memory probe accepts it; a rejected probe fails these checks loudly
# (it names the reason) rather than skipping. Its destination reduce sums the bf16 K-partials in
# fp32 in source-rank order and rounds once, so the result is bitwise the explicit fp32 chain
# (``reference_fixed_order_reduce``), bitwise NCCL's reduce-scatter at world size 2 (one add)
# and, at tp >= 3, within U(tp) ulps of the largest partial of NCCL's bf16 ring, which rounds
# per hop (U(3) = 5, U(4) = 7, U(8) = 25). The bias of a row Linear enters as PR #9's partial
# carries it: added in bf16 to rank 0's partial before the sum.


def _rs_module():
    from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_reduce_scatter

    return token_sharded_ce_reduce_scatter


def _rs_reason(tp):
    """Why ``tp``'s copy-engine reduce-scatter is not effective (for a failure message after
    ``begin()``, which creates the state)."""
    return tp._rs_state.reason


def _ring_ulp_bound(tp_size):
    """``U(tp) = 0.5 * sum_{j=2..tp} 2^ceil(log2 j) + 0.5 * 2^ceil(log2 tp)``: the bound on
    ``|fixed-order fp32 sum - bf16 ring sum|`` in ulps of the largest partial (tp >= 3)."""

    def pow2_ceil(j):
        return 1 << (j - 1).bit_length()

    return 0.5 * sum(pow2_ceil(j) for j in range(2, tp_size + 1)) + 0.5 * pow2_ceil(tp_size)


def _bf16_ulp(x):
    """Elementwise ulp of ``x`` at bf16 resolution, as fp32: ``2^(floor(log2 |x|) - 7)`` (the
    smallest normal's for 0)."""
    _, exponent = torch.frexp(x.float().abs().clamp_min(2.0**-126))
    return torch.exp2((exponent - 8).to(torch.float32))


def _rs_partials(partial_padded, plan, group):
    """Every source rank's block of this rank's rows, bf16 ``[tp, m, N]``: what a reduce-scatter
    of the padded ``[B * S_pad, N]`` partials sums here (a test-only all-to-all)."""
    src = partial_padded.contiguous()
    parts = torch.empty_like(src)
    dist.all_to_all_single(parts, src, group=group)
    return parts.view(plan.tp_size, plan.local_rows, -1)


def _rs_stats(c5, nccl, parts):
    """Design 5.2 (C) statistics of a copy-engine reduce-scatter ``c5`` against NCCL's ``nccl``
    on the same partials ``parts`` (``[tp, m, N]`` bf16, rank 0's carrying the bias)."""
    exact = parts.double().sum(0)  # <= 8 bf16 addends: exact in f64
    tiny = torch.finfo(torch.float64).tiny
    diff = (c5.float() - nccl.float()).abs()
    denom = exact.norm().clamp_min(tiny)
    exact_bf16 = exact.to(torch.bfloat16)
    # Result ulps are unbounded under cancellation (|sum| << |partials|): reported only, over
    # the elements whose exact sum is not 0 at bf16 resolution.
    result_ulps = torch.where(exact_bf16 != 0, diff / _bf16_ulp(exact_bf16), 0.0)
    return {
        "max_ulps_of_max_partial": (diff / _bf16_ulp(parts.abs().amax(0))).max().item(),
        "max_result_ulps": result_ulps.max().item(),
        "differ_frac": (c5 != nccl).float().mean().item(),
        "rel_l2_c5_nccl": (diff.double().norm() / nccl.double().norm().clamp_min(tiny)).item(),
        "rel_l2_c5_exact": ((c5.double() - exact).norm() / denom).item(),
        "rel_l2_nccl_exact": ((nccl.double() - exact).norm() / denom).item(),
        "c5_ne_bf16_exact": int((c5 != exact_bf16).sum().item()),
        "numel": c5.numel(),
    }


def _check_rs_vs_nccl(c5, nccl, parts, device, what):
    """The copy-engine result against NCCL's reduce-scatter of the same partials: bitwise at
    tp = 2 (one add), else within U(tp) ulps of the largest partial, rel-L2 <= 1e-2, at least
    as close to the f64 sum as NCCL, and equal to bf16(f64 sum) everywhere (tp <= 4, where the
    fp32 chain is exact for every realistic exponent spread; on > 16M elements a 1e-5 fraction
    is allowed) or all but a 1e-5 fraction, at least 4 elements (tp = 8: the seven-add chain can
    double-round against the f64 sum; 1 of 271,360 observed). Returns the statistics (None at
    tp = 2)."""
    tp = parts.shape[0]
    if tp == 2:
        _check(torch.equal(c5, nccl), f"{what}: copy-engine RS != NCCL RS at tp=2", device)
        return None
    s = _rs_stats(c5, nccl, parts)
    bound = _ring_ulp_bound(tp)
    # tp <= 4: the fp32 chain equals bf16(f64 sum) everywhere (every realistic exponent spread).
    # tp = 8: the chain's seventh fp32 add can double-round against the f64 sum; measured 1 of
    # 271,360 elements at K_local 3456 / m 53 (8x B200), so allow a 1e-5 fraction, at least 4.
    exact_limit = 0 if tp <= 4 and s["numel"] <= 1 << 24 else max(4.0, 1e-5 * s["numel"])
    ok = (
        s["max_ulps_of_max_partial"] <= bound
        and s["rel_l2_c5_nccl"] <= 1e-2
        and s["rel_l2_c5_exact"] <= s["rel_l2_nccl_exact"] + 1e-9
        and s["c5_ne_bf16_exact"] <= exact_limit
    )
    _check(ok, f"{what}: copy-engine RS vs NCCL RS out of bound U({tp})={bound}: {s}", device)
    return s


def _spy_reduce_scatter(tp):
    """Record what each of ``tp``'s NCCL reduce-scatters reduces."""
    moved = []
    scatter = tp.reduce_scatter

    def spy(partial):
        moved.append(partial)
        return scatter(partial)

    tp.reduce_scatter = spy
    return moved


def _fp8_row_linear(rank, world_size, device, k_local, n, seed):
    """An FP8 block-scale row-parallel Linear (``[n, k_local * tp]`` weight sharded over K, with
    a bias) built as for plain TP, and its checkpoint (load it after the conversion)."""
    k = k_local * world_size
    gen = torch.Generator().manual_seed(seed)
    weight = (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16)
    bias = (torch.randn(n, generator=gen) * 0.5 + 0.25).to(torch.bfloat16)  # O(1), as the partials
    row = Linear(
        k,
        n,
        bias=True,
        dtype=torch.bfloat16,
        mapping=_mapping(rank, world_size),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,  # built as for plain TP; the conversion turns it into a reduce-scatter
        allreduce_strategy=AllReduceStrategy.NCCL,
    ).to(device)
    return row, {**fp8_block_checkpoint(weight), "bias": bias}


def _logic_ce_reduce_scatter_op(rank, world_size, device):
    """The fused op ``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter`` on real FP8 block-scale
    row Linears (K_local 1280 / 3456, N 5120) over small-M and 32K/64K-token plans, including
    a one-row tail chunk: bitwise the explicit fp32 chain over the all-to-all'd bf16 partials
    with the bias entering as rank 0's bf16 partial carries it (A), within design 5.2's bound of
    NCCL's reduce-scatter of PR #9's partials and bitwise at tp = 2 (C), deterministic over 24
    boundaries with a rotating source's push to a rotating destination delayed on its side
    stream (B); a rank-symmetric raise inside the op (an injected GEMM failure before, between
    and after the pushes) resets the transport and the next boundary is right; the pool grows
    across shapes and never shrinks; close() unregisters the state and the op refuses to run
    without one."""
    rs = _rs_module()
    n = 5120
    rows = {k: _fp8_row_linear(rank, world_size, device, k, n, seed=k) for k in (1280, 3456)}
    tp = TokenShardedTP(dist.group.WORLD, reduce_scatter_mode="copy_engine")
    convert_to_token_sharded_tp(_as_model(**{f"row{k}": row for k, (row, _) in rows.items()}), tp)
    for row, ckpt in rows.values():
        row.load_weights([ckpt])
        row.post_load_weights()
        rs.validate_fp8_block_row_consumer(row)
    group_name = tp.group_name

    def op(fp8, sf, row, bias, plan):
        return torch.ops.trtllm.token_sharded_fp8_ce_gemm_reduce_scatter(
            fp8,
            sf,
            row.weight,
            row.weight_scale,
            bias,
            group_name,
            plan.tp_rank,
            plan.tp_size,
            plan.batch_size,
            plan.seq_len,
            plan.padded_seq_len,
        )

    def boundary(row, k_local, plan, seed, with_bias):
        """One boundary on this rank's ``[B, S, K_local]`` slice: the quantized padded input,
        the raw GEMM partial, PR #9's partial (bias on rank 0 when ``with_bias``) and the fused
        op's ``[m, N]`` output (the op takes the bias on every rank and adds it to source 0)."""
        b, s = plan.batch_size, plan.seq_len
        x = (torch.randn(b, s, k_local, generator=torch.Generator().manual_seed(seed)) * 2).to(
            device, torch.bfloat16
        )
        fp8, sf = quantize_fp8_block(tp.pad_row_input(x))
        raw = torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            fp8, sf, row.weight, row.weight_scale, torch.bfloat16, True
        )
        pr9 = raw + row.bias if with_bias and rank == 0 else raw
        c5 = op(fp8, sf, row, row.bias if with_bias else None, plan)  # every rank passes the bias
        return fp8, sf, raw, pr9, c5

    shapes = _FP8_SHAPES + [(1, 768), (1, (rs.CHUNK_ROWS + 1) * world_size), (1, 32768), (1, 65536)]
    pool_bytes = []
    for b, s in shapes:
        plan = tp.begin(b, s)
        state = rs.get_rs_state(group_name)
        _check(
            state.effective,
            f"copy-engine reduce-scatter probe rejected this group: {state.reason}",
            device,
        )
        _check(tp.effective_reduce_scatter_mode == "copy_engine", _rs_reason(tp), device)
        pool_bytes.append(state.pool.nbytes)
        m = plan.local_rows
        for k_local, (row, _) in rows.items():
            for with_bias in (True, False):
                what = f"K_local={k_local} {(b, s)} m={m} bias={with_bias}"
                # Per rank (its own K slice), per bias variant.
                seed = 1000 * b + 7 * s + k_local + 31 * rank + int(with_bias)
                _, _, raw, pr9, c5 = boundary(row, k_local, plan, seed, with_bias)
                _check(
                    c5.shape == (m, n) and c5.dtype == torch.bfloat16,
                    f"{what}: output {c5.dtype} {tuple(c5.shape)}",
                    device,
                )
                parts_raw = _rs_partials(raw, plan, tp.group)
                parts = parts_raw.clone()  # PR #9's partials: rank 0's carries the bias
                if with_bias:
                    parts[0] = parts_raw[0] + row.bias  # as Linear.forward adds it (bf16)
                bias = row.bias if with_bias else None
                ref = rs.reference_fixed_order_reduce(list(parts_raw.unbind(0)), bias)
                # (A) Bitwise the sequential fp32 chain over the same partial bytes.
                _check(
                    torch.equal(c5, ref),
                    f"{what}: copy-engine RS != the fixed-order fp32 chain "
                    f"({(c5 != ref).sum().item()} of {c5.numel()} differ)",
                    device,
                )
                _check(
                    torch.equal(ref, rs.reference_fixed_order_reduce(list(parts.unbind(0)), None)),
                    f"{what}: the reference's bias is not rank 0's bf16(p0 + bias)",
                    device,
                )
                # (C) vs NCCL's reduce-scatter of PR #9's partials, itself repeated once (the
                # baseline's determinism is measured per node, not assumed).
                nccl = tp.reduce_scatter(pr9)
                _check(
                    torch.equal(nccl, tp.reduce_scatter(pr9)),
                    f"{what}: the NCCL reduce-scatter is not repeatable on this node",
                    device,
                )
                stats = _check_rs_vs_nccl(c5, nccl, parts, device, what)
                if rank == 0 and stats is not None:
                    print(f"CE_RS_VS_NCCL tp={world_size} {what} {stats}", flush=True)
        if m <= 512:
            # The bias is applied: with and without differ (bias != 0) at the same input.
            seed = 1000 * b + 7 * s + 1280 + 31 * rank
            _, _, _, _, with_b = boundary(rows[1280][0], 1280, plan, seed, True)
            _, _, _, _, without = boundary(rows[1280][0], 1280, plan, seed, False)
            _check(not torch.equal(with_b, without), f"{(b, s)}: the bias had no effect", device)

    # (B) Determinism: 24 boundaries at one shape (the pool already holds a larger one, so the
    # partials view runs at the pool's region stride); at boundary i source i % tp sleeps on the
    # side stream of a rotating destination before the op, so that one push lands last there
    # while its other pushes and every other source's are unaffected (design 5.2 B); every
    # output equals the first, which is the reference chain.
    plan = tp.begin(2, 256)
    state = rs.get_rs_state(group_name)
    transport = state.transport
    for k_local, (row, _) in rows.items():
        first = None
        for i in range(24):
            if rank == i % world_size:
                dst = transport.dest_order[i % len(transport.dest_order)]
                with torch.cuda.stream(transport.streams.stream(dst)):
                    torch.cuda._sleep(2_000_000)
            _, _, raw, _, c5 = boundary(row, k_local, plan, 5 + k_local + 31 * rank, True)
            if first is None:
                first = c5
                ref = rs.reference_fixed_order_reduce(
                    list(_rs_partials(raw, plan, tp.group).unbind(0)), row.bias
                )
                _check(
                    torch.equal(first, ref),
                    f"K_local={k_local}: boundary 0 of the determinism run != the fixed-order "
                    f"chain ({(first != ref).sum().item()} of {ref.numel()} differ)",
                    device,
                )
            else:
                _check(
                    torch.equal(c5, first),
                    f"K_local={k_local}: boundary {i} differs from boundary 0 "
                    f"(source {i % world_size} delayed its push to destination "
                    f"{transport.dest_order[i % len(transport.dest_order)]})",
                    device,
                )
        if rank == 0:
            print(f"CE_RS_DETERMINISM tp={world_size} K_local={k_local} 24 boundaries equal")

    # A rank-symmetric raise inside the op -- a GEMM failing before the first push, between two
    # pushes (tp >= 3) and after the last one -- resets the transport: the pending ready waits
    # are trimmed to the sources that pushed, so the release does not spin to the deadline, and
    # the next boundary is the reference chain again.
    row, _ = rows[1280]
    real_gemm = rs.fp8_block_gemm_out
    for fail_at in sorted({0, 1, world_size - 1}):
        gemms = []

        def failing_gemm(*args, _fail_at=fail_at, **kwargs):
            gemms.append(1)
            if len(gemms) - 1 == _fail_at:
                raise RuntimeError(f"injected GEMM failure at GEMM {_fail_at}")
            return real_gemm(*args, **kwargs)

        rs.fp8_block_gemm_out = failing_gemm
        try:
            with pytest.raises(RuntimeError, match="injected GEMM failure"):
                boundary(row, 1280, plan, 700 + fail_at, True)
        finally:
            rs.fp8_block_gemm_out = real_gemm
        _check(len(gemms) == fail_at + 1, f"the injected failure did not fire at {fail_at}", device)
        _, _, raw, _, c5 = boundary(row, 1280, plan, 800 + fail_at + 31 * rank, True)
        ref = rs.reference_fixed_order_reduce(
            list(_rs_partials(raw, plan, tp.group).unbind(0)), row.bias
        )
        torch.cuda.synchronize()  # a trapped context (a wait spun to the deadline) raises here
        _check(
            torch.equal(c5, ref),
            f"the boundary after a rank-symmetric raise at GEMM {fail_at} != the fixed-order "
            f"chain ({(c5 != ref).sum().item()} of {ref.numel()} differ)",
            device,
        )

    # The pool: sized by the forwards, grow-only, never shrinking for a smaller shape.
    _check(
        pool_bytes[-1] == pool_bytes[-2] == pool_bytes[-3],
        f"the reduce-scatter pool grew past the row chunk: {pool_bytes}",
        device,
    )
    _check(
        all(a <= b for a, b in zip(pool_bytes, pool_bytes[1:])) and pool_bytes[-1] > pool_bytes[0],
        f"the reduce-scatter pool did not grow monotonically: {pool_bytes}",
        device,
    )
    plan = tp.begin(1, 8)
    _check(
        rs.get_rs_state(group_name).pool.nbytes == max(pool_bytes),
        "the reduce-scatter pool shrank for a smaller shape",
        device,
    )
    tp.close()
    with pytest.raises(KeyError):
        rs.get_rs_state(group_name)
    tp.close()  # idempotent
    fp8, sf = quantize_fp8_block(
        torch.zeros(plan.padded_rows, 1280, device=device, dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError, match="no CeReduceScatterState is registered"):
        op(fp8, sf, row, None, plan)


@patch("tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp.MIN_CE_REDUCE_SCATTER_ROWS", 0)
def _logic_adapters_ce_reduce_scatter(rank, world_size, device):
    """FP8 block-scale row Linear and MLP converted with ``reduce_scatter_mode="copy_engine"``
    (and the copy-engine gather: both pools alive in one group) against the NCCL adapters: the
    outputs are bitwise the fixed-order fp32 chain over the NCCL adapters' own partials (rank 0's
    with the bias) and within design 5.2's bound of the NCCL result (bitwise at tp = 2); the
    helper's reduce_scatter is bypassed; on the rows the stock Linear runs main's FP8 block-scale
    low-M GEMM (``fp8_low_m_gemm.serves``: decode-sized, not DeepGEMM) the adapters keep the
    module's GEMM and the NCCL reduce-scatter instead, bitwise the NCCL adapters; the engagement
    rule holds; the torch.compile'd adapters are bitwise their eager selves; close() is
    idempotent."""
    import torch._dynamo

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    rs = _rs_module()
    k_in, n_out = 640, 128 * world_size  # row and down_proj: [640, 128 * tp], K_local 128, N 640
    torch.manual_seed(8)
    w_row = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    w_up = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    w_down = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    ckpt_row = {**fp8_block_checkpoint(w_row), "bias": torch.randn(k_in) * 0.5}
    ckpt_up = {**fp8_block_checkpoint(w_up), "bias": torch.randn(n_out) * 0.1}
    ckpt_down = {**fp8_block_checkpoint(w_down), "bias": torch.randn(k_in) * 0.5}
    mapping = _mapping(rank, world_size)
    fp8bs = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)

    def build(mode):
        row = Linear(
            n_out,
            k_in,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=fp8bs,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            allreduce_strategy=AllReduceStrategy.NCCL,
        ).to(device)
        config = ModelConfig(
            mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL, quant_config=fp8bs
        )
        mlp = MLP(
            hidden_size=k_in,
            intermediate_size=n_out,
            bias=True,
            activation=gelu_tanh,
            dtype=torch.bfloat16,
            config=config,
        ).to(device)
        tp = TokenShardedTP(dist.group.WORLD, gather_mode=mode, reduce_scatter_mode=mode)
        convert_to_token_sharded_tp(_as_model(row=row, mlp=mlp), tp)
        row.load_weights([ckpt_row])
        mlp.up_proj.load_weights([ckpt_up])
        mlp.down_proj.load_weights([ckpt_down])
        for lin in (row, mlp.up_proj, mlp.down_proj):
            lin.post_load_weights()
        return tp, row, mlp

    tp_ref, row_ref, mlp_ref = build("nccl")
    tp_ce, row_ce, mlp_ce = build("copy_engine")
    bf16_row = Linear(
        n_out,
        k_in,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=False,
    ).to(device)
    scattered = _spy_reduce_scatter(tp_ce)
    down_partials = []  # the NCCL MLP's down_proj output on the real rows (its RS input)
    down_inputs = []  # and its input: the rows the stock GEMM runs on
    mlp_ref.down_proj.register_forward_pre_hook(lambda mod, args: down_inputs.append(args[0]))
    mlp_ref.down_proj.register_forward_hook(lambda mod, args, out: down_partials.append(out))
    k_loc = n_out // world_size

    def inputs(b, s, seed):
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(seed))
        x_row = torch.randn(b, s, k_loc, generator=torch.Generator().manual_seed(seed + rank))
        return x.to(device, torch.bfloat16), (2 * x_row).to(device, torch.bfloat16)

    def low_m_served(linear, x):
        """Whether the stock ``linear`` runs main's FP8 block-scale low-M GEMM on ``x``'s rows as
        its prologue flattens them (``Fp8BlockScaleLowMGemmRange.serves`` over the weight's range
        on this device; the rule of ``TokenShardedTP.uses_ce_reduce_scatter``)."""
        return linear.fp8_low_m_gemm.serves(x.reshape(-1, x.shape[-1]))

    for b, s in _FP8_SHAPES:
        x, x_row = inputs(b, s, 100 * s + b)  # the token stream; this rank's K slice
        plan = tp_ref.begin(b, s)
        tp_ce.begin(b, s)
        _check(
            tp_ce.effective_reduce_scatter_mode == "copy_engine"
            and tp_ce.effective_gather_mode == "copy_engine",
            f"copy-engine modes not effective: RS {_rs_reason(tp_ce)}; gather {_ce_reason(tp_ce)}",
            device,
        )
        x_loc = tp_ce.local_view(tp_ce.shard(x))
        y_ref, f_ref = row_ref(x_row), mlp_ref(x_loc)
        # Where the stock Linear runs main's low-M GEMM (not DeepGEMM, so not bitwise with the
        # fused op's per-block GEMM) the CE adapters keep the module's GEMM and the NCCL
        # reduce-scatter, each deciding on the tensor its GEMM takes: the row adapter on its
        # padded input (B * S_pad rows), the MLP adapter on its down_proj input (the B * S real
        # rows). Rank-identical (plan ints, the weight's range on this device; 0 where the
        # shape or device is not served: every shape is a copy-engine check then, as the shapes
        # above 256 rows always are).
        x_row_padded = tp_ce.pad_row_input(x_row)
        served = {
            "row adapter": low_m_served(row_ce, x_row_padded),
            "MLP adapter": low_m_served(mlp_ce.down_proj, down_inputs[-1]),
        }
        n_scattered = len(scattered)
        y_ce, f_ce = row_ce(x_row), mlp_ce(x_loc)
        _check(
            len(scattered) - n_scattered == sum(served.values()),
            f"the CE adapters reduce-scattered with NCCL {len(scattered) - n_scattered} times "
            f"{(b, s)}, expected {sum(served.values())} (low-M GEMM rows: {served})",
            device,
        )
        n_scattered = len(scattered)
        shared_ce = mlp_ce(tp_ce.share_input(x_loc, (mlp_ce,)))
        _check(
            torch.equal(shared_ce, f_ce),
            f"shared gather + CE reduce-scatter != independent MLP {(b, s)}",
            device,
        )
        _check(
            len(scattered) - n_scattered == served["MLP adapter"],
            f"shared MLP did not preserve low-M NCCL fallback {(b, s)}",
            device,
        )
        _check(
            y_ce.shape == y_ref.shape and f_ce.shape == f_ref.shape,
            f"adapter output shapes {tuple(y_ce.shape)} vs {tuple(y_ref.shape)}, "
            f"{tuple(f_ce.shape)} vs {tuple(f_ref.shape)} {(b, s)}",
            device,
        )
        # The NCCL adapters' own partials (rank 0's with the bias): the row Linear's GEMM on the
        # padded input, the MLP's down_proj output on the real rows, padded as the adapter does.
        # Only the real rows are compared: pad rows are dropped at the next gather by contract,
        # and their content differs by construction (the fused op leaves rank 0's bias in them,
        # as the NCCL row path does; the NCCL MLP path appends zero rows after its bias).
        row_partial = Linear.forward(row_ref, tp_ref.pad_row_input(x_row))
        down_partial = tp_ref.pad_row_input(down_partials[-1])
        m = plan.local_rows
        real = _real_row_mask(plan, device)
        for what, got, ref, partial in (
            ("row adapter", y_ce, y_ref, row_partial),
            ("MLP adapter", f_ce, f_ref, down_partial),
        ):
            parts = _rs_partials(partial, plan, tp_ref.group)
            got2, ref2 = got.reshape(m, -1)[real], ref.reshape(m, -1)[real]
            if got2.numel() == 0:
                # Every row of this rank is padding (e.g. (1, 5) at tp 4): nothing to compare, but
                # _check is collective, so take part in the checks the other ranks run.
                for _ in range(1 if served[what] else 2):
                    _check(True, f"{what} {(b, s)}: no real rows on this rank", device)
                continue
            if served[what]:
                # The module's own GEMM and the NCCL reduce-scatter, as the NCCL adapter ran them.
                _check(
                    torch.equal(got2, ref2),
                    f"{what} {(b, s)}: low-M GEMM rows != the NCCL adapter "
                    f"({(got2 != ref2).sum().item()} of {ref2.numel()} differ)",
                    device,
                )
                continue
            chain = rs.reference_fixed_order_reduce(list(parts.unbind(0)), None)[real]
            _check(
                torch.equal(got2, chain),
                f"{what} {(b, s)}: copy-engine RS != the fixed-order chain over the NCCL "
                f"adapter's partials ({(got2 != chain).sum().item()} of {chain.numel()} differ)",
                device,
            )
            _check_rs_vs_nccl(got2, ref2, parts[:, real], device, f"{what} {(b, s)}")
        rule = (
            tp_ce.uses_ce_reduce_scatter(row_ce, x_row_padded) is not served["row adapter"]
            and tp_ce.uses_ce_reduce_scatter(mlp_ce.down_proj, x_row)
            is not low_m_served(mlp_ce.down_proj, x_row)
            and not tp_ce.uses_ce_reduce_scatter(row_ce, x_row, {"active": True})  # LoRA
            and not tp_ce.uses_ce_reduce_scatter(row_ce, x_row.float())  # not bf16
            and not tp_ce.uses_ce_reduce_scatter(bf16_row, x_row)  # not an FP8 block-scale row
            and not tp_ref.uses_ce_reduce_scatter(row_ref, x_row)  # NCCL mode
        )
        _check(rule, f"uses_ce_reduce_scatter rule {(b, s)}", device)
    del tp_ce.reduce_scatter  # drop the spy (an instance attribute) before compiling
    torch._dynamo.reset()
    row_c = torch.compile(lambda t: row_ce(t))
    mlp_c = torch.compile(lambda t: mlp_ce(t))
    mlp_ref_c = torch.compile(lambda t: mlp_ref(t))
    for b, s in [(2, 256), (1, 5), (1, 418), (2, 256)]:
        x, x_row = inputs(b, s, 200 * s + b)
        tp_ref.begin(b, s)
        tp_ce.begin(b, s)
        x_loc = tp_ce.local_view(tp_ce.shard(x))
        # The row adapter is the op alone (the module's GEMM and the NCCL reduce-scatter on
        # low-M GEMM rows): compiled == eager bitwise.
        _check(
            torch.equal(row_c(x_row), row_ce(x_row)),
            f"compiled CE row adapter != eager {(b, s)}",
            device,
        )
        # The MLP's activation and the quantize prologue run under Inductor, whose bf16
        # rounding differs from the eager kernels before the FP8 quantize; that drift exists on
        # the NCCL path too, so the copy-engine MLP's compiled-vs-eager drift is bounded by the
        # NCCL MLP's (same inputs, same compile), with the design's 1e-2 floor.
        drift_ref = _rel_l2(mlp_ref_c(x_loc), mlp_ref(x_loc))
        drift_ce = _rel_l2(mlp_c(x_loc), mlp_ce(x_loc))
        _check(
            drift_ce <= max(1e-2, 2.0 * drift_ref),
            f"compiled CE MLP adapter drifts from eager by rel-L2 {drift_ce:.3e} {(b, s)} "
            f"(NCCL MLP compiled vs eager: {drift_ref:.3e})",
            device,
        )
    # Exercise production dispatch for the real-time shape and a larger bias-free projection.
    with patch(
        "tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp.MIN_CE_REDUCE_SCATTER_ROWS", 8192
    ):
        for tokens in (768, world_size * 8192):
            if tokens != 768:
                row_ce.bias = row_ref.bias = None
            _, x_row = inputs(1, tokens, 1234)
            tp_ref.begin(1, tokens)
            tp_ce.begin(1, tokens)
            scattered = _spy_reduce_scatter(tp_ce)
            got = row_ce(x_row)
            _check(len(scattered) == 1, "fallback row adapter bypassed NCCL", device)
            _check(torch.equal(got, row_ref(x_row)), "fallback != NCCL row adapter", device)
            del tp_ce.reduce_scatter
    tp_ce.close()
    tp_ce.close()  # idempotent


def _fp8_rs_chain_parts(rank, world_size, device, d=256, n_col=128, mode="copy_engine"):
    """A converted FP8 block-scale column Linear and a converted FP8 block-scale row Linear
    (``[d, n_col * tp]`` weight, K_local = n_col, bias) with the gather and the reduce-scatter
    of the helper both in ``mode``."""
    torch.manual_seed(4)
    col_w = torch.randn(n_col * world_size, d, dtype=torch.bfloat16) * 0.05
    row_w = torch.randn(d, n_col * world_size, dtype=torch.bfloat16) * 0.05
    mapping = _mapping(rank, world_size)
    fp8bs = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    col = Linear(
        d,
        n_col * world_size,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        quant_config=fp8bs,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    row = Linear(
        n_col * world_size,
        d,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        quant_config=fp8bs,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,  # built as for plain TP; the conversion turns it into a reduce-scatter
        allreduce_strategy=AllReduceStrategy.NCCL,
    ).to(device)
    tp = TokenShardedTP(dist.group.WORLD, gather_mode=mode, reduce_scatter_mode=mode)
    convert_to_token_sharded_tp(_as_model(col=col, row=row), tp, exceptions={"col": "column"})
    col.load_weights([{**fp8_block_checkpoint(col_w), "bias": torch.zeros(n_col * world_size)}])
    row.load_weights([{**fp8_block_checkpoint(row_w), "bias": torch.full((d,), 0.1)}])
    col.post_load_weights()
    row.post_load_weights()
    return tp, col, row


def _make_rs_chain(tp, col, row):
    """AdaLN norm (row-local) -> converted FP8 column Linear on a given bf16 input (the gather)
    -> converted FP8 row Linear (the reduce-scatter) -> gated residual; returns the residual,
    the column output and the reduce-scatter output."""

    def chain(x_loc, table, h_given):
        d = x_loc.shape[-1]
        shift, scale, gate = tp.per_sample_table(table)[:, :, None].unbind(1)  # [n, 1, D] each
        h = (F.layer_norm(x_loc.float(), (d,)) * (1 + scale) + shift).to(x_loc.dtype)
        q = col(h_given)  # all tokens, this rank's features; exact input -> bitwise GEMM
        y = row(q)  # this rank's reduced rows, [n, g, D]
        x = (h.float() + y.float() * gate).to(x_loc.dtype)
        return x, q, y

    return chain


def _check_rs_modes(tp, device):
    _check(
        tp.effective_reduce_scatter_mode == "copy_engine"
        and tp.effective_gather_mode == "copy_engine",
        f"copy-engine modes not effective: RS {_rs_reason(tp)}; gather {_ce_reason(tp)}",
        device,
    )


@patch("tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp.MIN_CE_REDUCE_SCATTER_ROWS", 0)
def _logic_compile_fullgraph_ce_rs(rank, world_size, device):
    """torch.compile(fullgraph=True) of a gather + reduce-scatter chain with both copy-engine
    modes on: one graph (the fused ops are opaque; on the rows the stock Linear runs main's
    low-M GEMM the row adapter traces its module GEMM + NCCL reduce-scatter instead), the
    column and reduce-scatter outputs bitwise the eager chain's."""
    import torch._dynamo

    d = 256
    tp, col, row = _fp8_rs_chain_parts(rank, world_size, device, d)
    chain = _make_rs_chain(tp, col, row)

    def compare(got, ref, b, s, what):
        # The residual carries the Inductor-fused LayerNorm: bf16 resolution.
        _check_close(got[0], ref[0], device, 1e-2, 1e-2, f"{what} CE residual {(b, s)}")
        ok = torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])
        _check(ok, f"{what} CE gather + GEMM / GEMM + RS {(b, s)}", device)

    def run_compiled(compiled, args):
        try:
            return compiled(*args)
        except torch._dynamo.exc.Unsupported as e:
            if "all_reduce" in str(e).lower() or "allreduce" in str(e).lower():
                print(
                    f"SKIP _logic_compile_fullgraph_ce_rs: the AllReduce in the overlay is not "
                    f"traceable under fullgraph=True ({e})",
                    flush=True,
                )
                return None
            raise

    for b, s in [(2, 256), (2, 5), (1, 5), (1, 418)]:  # m % 4 == 0 and != 0, padded
        tp.begin(b, s)
        _check_rs_modes(tp, device)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        torch._dynamo.reset()
        got = run_compiled(torch.compile(chain, fullgraph=True), args)
        if got is None:
            tp.close()
            return
        compare(got, chain(*args), b, s, "compiled")
    torch._dynamo.reset()
    compiled = torch.compile(chain, fullgraph=True)
    for b, s in [(2, 256), (1, 5), (2, 256)]:
        tp.begin(b, s)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        compare(run_compiled(compiled, args), chain(*args), b, s, "recompiled")
    tp.close()


@patch("tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp.MIN_CE_REDUCE_SCATTER_ROWS", 0)
def _logic_cuda_graph_ce_rs(rank, world_size, device):
    """CUDA-graph capture + replay of a gather + reduce-scatter chain with both copy-engine
    modes on: begin() reserves both pools eagerly, two eager boundaries precede the capture (the
    consumed waits are in the graph), the replays are bitwise the eager chain on new inputs --
    which is itself bitwise the fixed-order chain over the row Linear's partials (the smaller
    shapes run at the pool's region stride of the first, larger one), or, on the rows the stock
    Linear runs main's low-M GEMM (``fp8_low_m_gemm.serves`` on the row adapter's padded
    input), bitwise the module's GEMM + NCCL reduce-scatter the adapter keeps there -- and a
    larger shape after the captures is refused rather than growing a pool under the graphs."""
    rs = _rs_module()
    d = 256
    tp, col, row = _fp8_rs_chain_parts(rank, world_size, device, d)
    chain = _make_rs_chain(tp, col, row)
    for b, s in [(1, (rs.CHUNK_ROWS + 1) * world_size), (2, 256), (2, 5), (1, 5), (1, 418)]:
        plan = tp.begin(b, s)  # eager: probes once, sizes both pools for this shape
        _check_rs_modes(tp, device)
        args = _fp8_chain_inputs(tp, b, s, d, device)
        static_args = [a.clone() for a in args]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # eager warmups (as CUDAGraphRunner.WARMUP_STEPS), >= slots
                chain(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = chain(*static_args)
        new_args = [args[0] * 0.5 + 0.25, args[1] * 0.5, args[2] * 0.75 - 0.5]
        for dst, src in zip(static_args, new_args):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        ref = chain(*new_args)
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"CE RS CUDA graph replay vs eager {(b, s)}", device)
        # Eager (hence the replay) against the reference chain over the row Linear's own GEMM
        # partials on the chain's column output (rank 0's with the bias, as Linear.forward
        # places it): a reduce reading the slot at the wrong stride would still replay
        # consistently, so eager-vs-replay alone cannot see it. On the rows the stock Linear
        # runs main's low-M GEMM the row adapter keeps the module's GEMM and the NCCL
        # reduce-scatter (not the fixed-order chain at tp > 2), so the reference is that path
        # on the same partials. Rank-identical; one collective _check on either branch.
        padded = tp.pad_row_input(ref[1])
        partial = Linear.forward(row, padded)
        if row.fp8_low_m_gemm.serves(padded):
            _check(
                torch.equal(ref[2].reshape(plan.local_rows, -1), tp.reduce_scatter(partial)),
                f"CE RS eager chain != the module's GEMM + NCCL reduce-scatter on low-M GEMM "
                f"rows {(b, s)}",
                device,
            )
        else:
            parts = _rs_partials(partial, plan, tp.group)
            chain_ref = rs.reference_fixed_order_reduce(list(parts.unbind(0)), None)
            _check(
                torch.equal(ref[2].reshape(plan.local_rows, -1), chain_ref),
                f"CE RS eager chain != the fixed-order chain over the row partials {(b, s)}",
                device,
            )
        graph.replay()  # steady state of the slot protocol
        torch.cuda.synchronize()
        ok = all(torch.equal(got_t, ref_t) for got_t, ref_t in zip(static_out, ref))
        _check(ok, f"CE RS CUDA graph second replay vs eager {(b, s)}", device)
        del graph
    with pytest.raises(RuntimeError, match="CUDA graph"):
        tp.begin(1, (rs.CHUNK_ROWS + 2) * world_size)
    tp.close()


# =============================================================================
# Sharder round trip and adapters vs the full computation (exact)
# =============================================================================
#
# Projections with integer-valued fp32 weights and inputs make every sum exact, so the
# adapters' all-gather / reduce-scatter must reproduce the full computation bit for bit on
# any backend. The fakes stand in for TRT-LLM Linear / MLP (whose TP construction needs a
# CUDA device mesh; the NCCL kernel checks and the Wan tests convert real modules).


class _FakeColumn(nn.Module):
    """Column-parallel GEMM: this rank's output features."""

    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return x @ self.w.t()


class _FakeRow(nn.Module):
    """Row-parallel GEMM: K-partial sums, the bias added on rank 0 only."""

    def __init__(self, w, b):
        super().__init__()
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, x):
        self.rows_seen = x.shape[0]
        return x @ self.w.t() + self.b


class _FakeMLP(nn.Module):
    def __init__(self, up, down):
        super().__init__()
        self.up, self.down = up, down

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _fake(adapter):
    """``adapter`` for a fake module: no Linear / MLP to check, no all-reduce to stop."""
    return type(f"Fake{adapter.__name__}", (adapter,), {"prepare": classmethod(lambda *a: None)})


_FAKE_ADAPTERS = {
    "col": _fake(TokenShardedColumn),
    "row": _fake(TokenShardedRow),
    "mlp": _fake(TokenShardedMLP),
}


def _ints(gen, *shape):
    return torch.randint(-3, 4, shape, generator=gen).float()


def _logic_layout_round_trip(rank, world_size, device):
    sharder = TokenShardedSequenceSharder(_helper())
    for b, s in _SHAPES:
        x = _ints(torch.Generator().manual_seed(b * 100 + s), b, s, 8).to(device)
        x_loc = sharder.shard(x, dim=1)
        _check(
            torch.equal(sharder.gather(x_loc, dim=1), x),
            f"shard/gather round trip {(b, s)}",
            device,
        )


def _logic_adapters(rank, world_size, device):
    k, n, hidden = 8, 4 * world_size, 6
    gen = torch.Generator().manual_seed(7)
    w_col, w_row, bias = _ints(gen, n, k), _ints(gen, hidden, n), _ints(gen, hidden)
    w_up, w_down, b_down = _ints(gen, n, k), _ints(gen, hidden, n), _ints(gen, hidden)
    cols = slice(rank * n // world_size, (rank + 1) * n // world_size)
    rank0_bias = bias if rank == 0 else torch.zeros_like(bias)
    block = nn.Module()
    block.col = _FakeColumn(w_col[cols])
    block.row = _FakeRow(w_row[:, cols], rank0_bias)
    block.mlp = _FakeMLP(
        _FakeColumn(w_up[cols]), _FakeRow(w_down[:, cols], b_down if rank == 0 else 0 * b_down)
    )
    root = nn.Module()
    root.blocks = nn.ModuleList([block]).to(device)
    tp = _helper()
    convert_to_token_sharded_tp(root, tp, exceptions=_FAKE_ADAPTERS)
    for b, s in _SHAPES:
        plan = tp.begin(b, s)
        x = _ints(torch.Generator().manual_seed(b * 10 + s), b, s, k).to(device)
        x_loc = tp.local_view(tp.shard(x))
        mask = _real_row_mask(plan, device)
        h = block.col(x_loc)  # all tokens, this rank's features
        _check(torch.equal(h, x @ w_col[cols].t().to(device)), f"column adapter {(b, s)}", device)
        mine = slice(plan.row_start, plan.row_start + plan.local_rows)
        want = padded_rows(x @ w_col.t().to(device) @ w_row.t().to(device) + bias.to(device), plan)
        got = block.row(h)
        _check(
            got.shape[:2] == x_loc.shape[:2]
            and torch.equal(got.reshape(plan.local_rows, -1)[mask], want[mine][mask]),
            f"row adapter {(b, s)}",
            device,
        )
        # The GEMM runs on the padded stream (pad rows carry only rank 0's bias, then drop).
        _check(block.row.rows_seen == plan.padded_rows, f"row GEMM input rows {(b, s)}", device)
        ref = torch.relu(x @ w_up.t().to(device)) @ w_down.t().to(device) + b_down.to(device)
        got = block.mlp(x_loc)
        _check(
            torch.equal(got.reshape(plan.local_rows, -1)[mask], padded_rows(ref, plan)[mine][mask]),
            f"MLP adapter {(b, s)}",
            device,
        )


def _shared_input_parts(rank, world_size, device, mode, *, tp=None):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.mlp import MLP

    torch.manual_seed(41)
    k, n = 640, 128 * world_size
    config = ModelConfig(
        mapping=_mapping(rank, world_size),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        allreduce_strategy=AllReduceStrategy.NCCL,
    )
    columns = [
        Linear(
            k,
            n,
            bias=True,
            dtype=torch.bfloat16,
            mapping=config.mapping,
            quant_config=config.quant_config,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        for _ in range(3)
    ]
    mlps = [
        cls(
            hidden_size=k,
            intermediate_size=n,
            bias=False,
            dtype=torch.bfloat16,
            activation=gelu_tanh if cls is MLP else F.silu,
            config=config,
        ).to(device)
        for cls in (MLP, GatedMLP)
    ]
    root = _as_model(qkv=columns[0], gate=columns[1], up=columns[2], mlp=mlps[0], gated=mlps[1])
    if tp is None:
        tp = TokenShardedTP(dist.group.WORLD, gather_mode=mode)
    convert_to_token_sharded_tp(
        root, tp, exceptions={"qkv": "column", "gate": "column", "up": "column"}
    )
    for col in columns:
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.03
        col.load_weights([{**fp8_block_checkpoint(w), "bias": torch.randn(n) * 0.01}])
        col.post_load_weights()
    for mlp in mlps:
        w_up = torch.randn(n, k, dtype=torch.bfloat16) * 0.03
        up = mlp.up_proj if isinstance(mlp, MLP) else mlp.gate_up_proj
        checkpoint = [fp8_block_checkpoint(w_up)]
        if isinstance(mlp, GatedMLP):
            checkpoint.append(fp8_block_checkpoint(torch.randn(n, k, dtype=torch.bfloat16) * 0.03))
        up.load_weights(checkpoint)
        mlp.down_proj.load_weights(
            [fp8_block_checkpoint(torch.randn(k, n, dtype=torch.bfloat16) * 0.03)]
        )
        up.post_load_weights()
        mlp.down_proj.post_load_weights()
    return tp, tuple(columns + mlps), root


def _logic_shared_input_adapters(rank, world_size, device):
    import torch._dynamo

    for mode in ("nccl", "copy_engine"):
        tp, consumers, root = _shared_input_parts(rank, world_size, device, mode)
        keys = tuple(root.state_dict())

        def grouped(x):
            shared = tp.share_input(x, consumers)
            return tuple(consumer(shared) for consumer in consumers)

        def independent(x):
            return tuple(consumer(x) for consumer in consumers)

        for b, s in _FP8_SHAPES:
            tp.begin(b, s)
            x = torch.randn(b, s, 640, device=device, dtype=torch.bfloat16)
            loc = tp.local_view(tp.shard(x))
            reference = independent(loc)
            moved = _spy_all_gather(tp)
            issued = []
            transport = tp._ce_state.transport if mode == "copy_engine" else None
            if transport is not None:
                issue = transport.issue

                def counted(*a, **kw):
                    issued.append(1)
                    return issue(*a, **kw)

                transport.issue = counted
            got = grouped(loc)
            if transport is not None:
                del transport.issue
            del tp.all_gather
            _check(
                all(torch.equal(g, r) for g, r in zip(got, reference)),
                f"shared {mode} != independent {(b, s)}",
                device,
            )
            _check(
                len(moved) == (mode == "nccl") and len(issued) == (mode == "copy_engine"),
                f"shared {mode} gather count: NCCL {len(moved)}, CE {len(issued)}",
                device,
            )
            packed = quantize_fp8_block(loc)
            got_packed = grouped(packed)
            _check(
                all(torch.equal(g, r) for g, r in zip(got_packed, reference)),
                f"shared prequantized {mode} {(b, s)}",
                device,
            )
        _check(tuple(root.state_dict()) == keys, "sharing changed state_dict keys", device)
        if world_size in (2, 4):
            tp.begin(2, 5)
            loc = tp.local_view(
                tp.shard(torch.randn(2, 5, 640, device=device, dtype=torch.bfloat16))
            )
            compiled = torch.compile(grouped, fullgraph=True)
            compiled_ref = torch.compile(independent, fullgraph=True)
            actual, reference = compiled(loc), compiled_ref(loc)
            _check(
                all(torch.equal(g, r) for g, r in zip(actual, reference)),
                f"shared compiled {mode}",
                device,
            )
            # Both slots must be warm before capture.
            grouped(loc)
            reference = grouped(loc)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = grouped(loc)
            graph.replay()
            _check(
                all(torch.equal(g, r) for g, r in zip(captured, reference)),
                f"shared CUDA graph {mode}",
                device,
            )
            loc.add_(0.125)
            reference = grouped(loc)
            graph.replay()
            _check(
                all(torch.equal(g, r) for g, r in zip(captured, reference)),
                f"shared CUDA graph fresh rows {mode}",
                device,
            )
            del graph
        tp.close()


def _logic_shared_input_dense_fallbacks(rank, world_size, device):
    """Real BF16/FP8 consumers and an active native LoRA share one dense gather."""
    from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
    from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import _convert

    torch.manual_seed(91)
    k, n = 256, 128 * world_size
    mapping = _mapping(rank, world_size)
    for mode in ("nccl", "copy_engine"):
        tp = TokenShardedTP(dist.group.WORLD, gather_mode=mode)
        columns = [
            Linear(
                k,
                n,
                bias=True,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False,
                quant_config=QuantConfig(quant_algo=algo) if algo else QuantConfig(),
            ).to(device)
            for algo in (QuantAlgo.FP8_BLOCK_SCALES, None, QuantAlgo.FP8_BLOCK_SCALES)
        ]
        for name, col in zip(("fp8", "dense", "fp8_sibling"), columns):
            _convert(col, "column", tp, name)
        for index, col in enumerate(columns):
            weight = torch.randn(n, k, dtype=torch.bfloat16) * 0.03
            checkpoint = fp8_block_checkpoint(weight) if index != 1 else {"weight": weight}
            checkpoint["bias"] = torch.randn(n, dtype=torch.bfloat16) * 0.01
            col.load_weights([checkpoint])
            col.post_load_weights()
        plan = tp.begin(1, 127)
        loc = tp.local_view(tp.shard(torch.randn(1, 127, k, device=device, dtype=torch.bfloat16)))
        for with_lora in (False, True):
            consumers = (columns[0], columns[2]) if with_lora else tuple(columns)
            options = {}
            if with_lora:
                col = columns[0]
                col.lora = LoraLayer([LoraModuleType.ATTENTION_Q], [128])
                lora_a = torch.randn(16, k, device=device, dtype=torch.bfloat16) * 0.1
                lora_b = torch.randn(128, 16, device=device, dtype=torch.bfloat16) * 0.1
                params = {
                    "num_seqs": 1,
                    "host_request_types": torch.zeros(1, dtype=torch.int32),
                    "prompt_lens_cpu": torch.tensor([plan.seq_len], dtype=torch.int32),
                    0: {
                        int(LoraModuleType.ATTENTION_Q): {
                            "adapter_size": torch.tensor([16], dtype=torch.int32),
                            "weight_pointers": torch.tensor(
                                [[lora_a.data_ptr(), lora_b.data_ptr(), 0]], dtype=torch.int64
                            ),
                        }
                    },
                }
                options = {"lora_params": params, "layer_idx": 0}
            # Dense fallback must match independently gathered dense inputs. FP8 activation
            # quantization belongs to each consumer and may select its own quantizer.
            dense_rows = tp.all_gather(loc.reshape(-1, k))
            reference = tuple(
                Linear.forward(col, dense_rows, **(options if index == 0 else {})).view(
                    1, plan.seq_len, -1
                )
                for index, col in enumerate(consumers)
            )
            moved = _spy_all_gather(tp)
            shared = tp.share_input(loc, consumers, prequantize=not with_lora)
            actual = tuple(
                col(shared, **(options if index == 0 else {}))
                for index, col in enumerate(consumers)
            )
            del tp.all_gather
            _check(len(moved) == 1, "dense fallback did not share one gather", device)
            _check(isinstance(shared.activation, torch.Tensor), "fallback was not dense", device)
            _check(
                all(torch.equal(a, b) for a, b in zip(actual, reference)),
                f"mixed quantizers / active LoRA mismatch ({mode}, {with_lora})",
                device,
            )
            if with_lora:
                base = Linear.forward(columns[0], dense_rows).view(1, plan.seq_len, -1)
                _check(not torch.equal(actual[0], base), "LoRA was not active", device)
                delta = columns[0].lora(dense_rows, params, 0)
                expected_delta = (dense_rows @ lora_a.t()) @ lora_b.t()
                _check_close(delta, expected_delta, device, 2e-2, 2e-2, "native LoRA delta")
        tp.close()


def _shared_biased_mlp_parts(rank, world_size, device, tp):
    """A QKV column and biased MLP sharing rows, with a production-eligible FP8 down GEMM."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    torch.manual_seed(73)
    k, n = 256, 128 * world_size
    config = ModelConfig(
        mapping=_mapping(rank, world_size),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        allreduce_strategy=AllReduceStrategy.NCCL,
    )
    col = Linear(
        k,
        n,
        bias=True,
        dtype=torch.bfloat16,
        mapping=config.mapping,
        quant_config=config.quant_config,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    mlp = MLP(
        hidden_size=k,
        intermediate_size=n,
        bias=True,
        activation=gelu_tanh,
        dtype=torch.bfloat16,
        config=config,
    ).to(device)
    root = _as_model(qkv=col, mlp=mlp)
    convert_to_token_sharded_tp(root, tp, exceptions={"qkv": "column"})
    for linear, shape in ((col, (n, k)), (mlp.up_proj, (n, k)), (mlp.down_proj, (k, n))):
        checkpoint = fp8_block_checkpoint(torch.randn(*shape, dtype=torch.bfloat16) * 0.03)
        checkpoint["bias"] = torch.randn(shape[0], dtype=torch.bfloat16) * 0.02
        linear.load_weights([checkpoint])
        linear.post_load_weights()
    return col, mlp


def _logic_shared_input_chunked_graphs(rank, world_size, device):
    """Multiple shared consumers followed by real multi-chunk RS, compiled and replayed."""
    import torch._dynamo

    rs = _rs_module()
    tp = TokenShardedTP(
        dist.group.WORLD, gather_mode="copy_engine", reduce_scatter_mode="copy_engine"
    )
    col, mlp = _shared_biased_mlp_parts(rank, world_size, device, tp)
    # Exercise the production cutoff without overriding it, a one-row tail and padding.
    plan = tp.begin(1, (rs.CHUNK_ROWS + 1) * world_size - 1)
    _check_rs_modes(tp, device)
    x = torch.randn(1, plan.seq_len, 256, device=device, dtype=torch.bfloat16)
    loc = tp.local_view(tp.shard(x))

    def grouped(value):
        shared = tp.share_input(value, (col, mlp))
        return col(shared), mlp(shared)

    def independent(value):
        return col(value), mlp(value)

    _check(tp.uses_ce_reduce_scatter(mlp.down_proj, x), "production RS dispatch", device)
    reference = independent(loc)
    gather_transport = tp._ce_state.transport
    rs_transport = tp._rs_state.transport
    with patch.object(gather_transport, "issue", wraps=gather_transport.issue) as issued:
        with patch.object(rs_transport, "begin", wraps=rs_transport.begin) as rounds:
            with patch.object(tp, "reduce_scatter", wraps=tp.reduce_scatter) as nccl_rs:
                actual = grouped(loc)
    _check(issued.call_count == 1, "combined group repeated its gather", device)
    _check(rounds.call_count == 2 and nccl_rs.call_count == 0, "chunked RS not used", device)
    _check(
        all(torch.equal(a, b) for a, b in zip(actual, reference)),
        "shared consumers + chunked RS differ from independent consumers",
        device,
    )
    capacity = tp._rs_state.pool.nbytes
    expected = rs.Bf16RowsRegionLayout(rs.CHUNK_ROWS, 256).slot_nbytes(world_size) * rs.SLOTS
    _check(capacity == expected, "combined path exceeded capped RS payload", device)

    # Fullgraph failures are test failures: this check has no unsupported-op early return.
    compiled = torch.compile(grouped, fullgraph=True)
    compiled_reference = torch.compile(independent, fullgraph=True)
    actual, reference = compiled(loc), compiled_reference(loc)
    _check(
        all(torch.equal(a, b) for a, b in zip(actual, reference)),
        "compiled shared consumers + chunked RS differ",
        device,
    )
    for _ in range(2):
        compiled(loc)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(loc)
    for delta in (0.125, -0.25):
        loc.add_(delta)
        reference = compiled_reference(loc)
        graph.replay()
        _check(
            all(torch.equal(a, b) for a, b in zip(captured, reference)),
            "combined CUDA graph replay failed on changed rows",
            device,
        )
    _check(tp._rs_state.pool.nbytes == capacity, "replay grew RS pool", device)
    del graph, compiled, compiled_reference
    torch._dynamo.reset()
    tp.close()


def _logic_parallel_config_gather(rank, world_size, device):
    """Serving YAML controls shared gathers and aligned, padded token plans."""
    import tempfile
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    for gather_mode in ("nccl", "copy_engine"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serving.yaml"
            path.write_text(
                "model: /tmp/token-sharded-synthetic\nparallel_config:\n"
                f"  tp_size: {world_size}\n  tp_layout: token_sharded\n"
                f"  gather_mode: {gather_mode}\n  row_align: 64\n"
            )
            args = VisualGenArgs.from_yaml(path)
        vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
        config = DiffusionModelConfig(parallel=args.parallel_config, visual_gen_mapping=vgm)
        tp = TokenShardedTP.from_model_config(config)
        _, consumers, _ = _shared_input_parts(rank, world_size, device, gather_mode, tp=tp)
        for rows in (64, 192):
            plan = tp.begin(1, rows * world_size - 1)
            _check(
                plan.local_rows == rows and plan.is_padded and tp.row_align == 64,
                "YAML row alignment did not reach the token plan",
                device,
            )
            _check(tp.effective_gather_mode == gather_mode, "YAML gather backend", device)
            loc = tp.local_view(
                tp.shard(torch.randn(1, plan.seq_len, 640, device=device, dtype=torch.bfloat16))
            )
            reference = tuple(consumer(loc) for consumer in consumers)
            moved = _spy_all_gather(tp)
            if gather_mode == "copy_engine":
                transport = tp._ce_state.transport
                with patch.object(transport, "issue", wraps=transport.issue) as issued:
                    shared = tp.share_input(loc, consumers)
                _check(issued.call_count == 1, "YAML CE gather was not shared", device)
            else:
                shared = tp.share_input(loc, consumers)
            actual = tuple(consumer(shared) for consumer in consumers)
            del tp.all_gather
            _check(len(moved) == (gather_mode == "nccl"), "YAML gather count", device)
            _check(
                all(torch.equal(a, b) for a, b in zip(actual, reference)),
                "YAML-driven shared projections differ from independent projections",
                device,
            )
        tp.close()


def _logic_parallel_config_collectives(rank, world_size, device):
    """Serving YAML controls real grouped gathers, alignment and production RS dispatch."""
    import tempfile
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    rs = _rs_module()
    for gather_mode in ("nccl", "copy_engine"):
        for reduce_mode in ("nccl", "copy_engine"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "serving.yaml"
                path.write_text(
                    "model: /tmp/token-sharded-synthetic\nparallel_config:\n"
                    f"  tp_size: {world_size}\n  tp_layout: token_sharded\n"
                    f"  gather_mode: {gather_mode}\n  reduce_scatter_mode: {reduce_mode}\n"
                    "  row_align: 64\n"
                )
                args = VisualGenArgs.from_yaml(path)
            vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
            config = DiffusionModelConfig(parallel=args.parallel_config, visual_gen_mapping=vgm)
            tp = TokenShardedTP.from_model_config(config)
            col, mlp = _shared_biased_mlp_parts(rank, world_size, device, tp)
            for rows in (64, rs.CHUNK_ROWS + 64):
                # Both shapes pad, remain 64-row aligned, and straddle production dispatch.
                plan = tp.begin(1, rows * world_size - 1)
                _check(
                    plan.local_rows == rows and plan.is_padded and tp.row_align == 64,
                    "YAML row alignment did not reach the token plan",
                    device,
                )
                _check(
                    tp.effective_gather_mode == gather_mode
                    and tp.effective_reduce_scatter_mode == reduce_mode,
                    "YAML modes did not reach real transports",
                    device,
                )
                loc = tp.local_view(
                    tp.shard(torch.randn(1, plan.seq_len, 256, device=device, dtype=torch.bfloat16))
                )
                reference = col(loc), mlp(loc)
                moved = _spy_all_gather(tp)
                scattered = _spy_reduce_scatter(tp)
                if gather_mode == "copy_engine":
                    transport = tp._ce_state.transport
                    with patch.object(transport, "issue", wraps=transport.issue) as issued:
                        shared = tp.share_input(loc, (col, mlp))
                    _check(issued.call_count == 1, "YAML CE gather was not shared", device)
                else:
                    shared = tp.share_input(loc, (col, mlp))
                if reduce_mode == "copy_engine":
                    transport = tp._rs_state.transport
                    with patch.object(transport, "begin", wraps=transport.begin) as rounds:
                        actual = col(shared), mlp(shared)
                    _check(
                        rounds.call_count == (2 if rows > rs.CHUNK_ROWS else 0),
                        "YAML CE reduction did not follow chunked production dispatch",
                        device,
                    )
                else:
                    actual = col(shared), mlp(shared)
                del tp.all_gather, tp.reduce_scatter
                _check(len(moved) == (gather_mode == "nccl"), "YAML gather backend", device)
                expected_nccl = reduce_mode == "nccl" or rows < rs.CHUNK_ROWS
                _check(len(scattered) == expected_nccl, "YAML RS backend/fallback", device)
                _check(
                    all(torch.equal(a, b) for a, b in zip(actual, reference)),
                    "YAML-driven shared projections differ from independent projections",
                    device,
                )
            tp.close()


# =============================================================================
# Test entry points
# =============================================================================

_WORLD_SIZES = [2, 3, 4]

# Backend-agnostic logic: gloo at every world size (CPU lane), NCCL once at world size 3.
_LOGIC_CHECKS = (
    _logic_reduce_scatter,
    _logic_all_gather,
    _logic_all_gather_fp8,
    _logic_rank_disagreement,
    _logic_layout_round_trip,
    _logic_adapters,
)
# Real NVFP4 kernels and TRT-LLM Linear/MLP modules over NCCL.
_KERNEL_CHECKS = (_logic_fp4_quantize_gather, _logic_real_adapters, _logic_adapters_nvfp4)
# torch.compile(fullgraph=True) and CUDA-graph capture of a boundary chain.
_GRAPH_CHECKS = (_logic_compile_fullgraph, _logic_cuda_graph)
# The FP8 block-scale gather: real DeepGEMM Linear / MLP, the quantize tactics, and the graphs.
_FP8_KERNEL_CHECKS = (_logic_fp8_quant_tactics, _logic_adapters_fp8_block_scales)
_FP8_GRAPH_CHECKS = (_logic_compile_fullgraph_fp8, _logic_cuda_graph_fp8)
# The copy-engine gather: per-chunk GEMM, transport, adapters in both modes, and the graphs.
_CE_CHECKS = (
    _logic_fp8_block_gemm_out,
    _logic_ce_transport,
    _logic_adapters_ce_gather,
    _logic_shared_input_adapters,
)
_CE_GRAPH_CHECKS = (_logic_compile_fullgraph_ce, _logic_cuda_graph_ce)
# The copy-engine reduce-scatter: the fused GEMM + RS op, adapters in both modes, the graphs.
_RS_CHECKS = (_logic_ce_reduce_scatter_op, _logic_adapters_ce_reduce_scatter)
_RS_GRAPH_CHECKS = (_logic_compile_fullgraph_ce_rs, _logic_cuda_graph_ce_rs)


def _run_checks(rank, world_size, device, checks):
    """Run several checks in one spawn; a failure names the check."""
    for check in checks:
        try:
            check(rank, world_size, device)
        except BaseException as e:
            raise AssertionError(f"{check.__name__} (world_size={world_size}): {e}") from e


def _checks(*checks):
    return functools.partial(_run_checks, checks=checks)


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_logic_gloo(world_size):
    _run(world_size, _checks(*_LOGIC_CHECKS), "gloo")


def test_logic_nccl():
    _run(3, _checks(*_LOGIC_CHECKS), "nccl")


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_kernels_nccl(world_size):
    _requires_blackwell()
    checks = _KERNEL_CHECKS + (_GRAPH_CHECKS if world_size == 2 else ())
    _run(world_size, _checks(*checks), "nccl")


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_fp8_block_scales_nccl(world_size):
    _requires_sm100f()
    checks = _FP8_KERNEL_CHECKS + (_FP8_GRAPH_CHECKS if world_size == 2 else ())
    _run(world_size, _checks(*checks), "nccl")


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_fp8_ce_gather_nccl(world_size):
    """The copy-engine all-gather of FP8 block-scale inputs (symmetric memory): bitwise the
    NCCL gather at the transport, GEMM and adapter levels; fullgraph compile and CUDA-graph
    replay of a boundary chain at world size 2."""
    _requires_sm100f()
    checks = _CE_CHECKS + (_CE_GRAPH_CHECKS if world_size == 2 else ())
    _run(world_size, _checks(*checks), "nccl")


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_fp8_ce_reduce_scatter_nccl(world_size):
    """The copy-engine reduce-scatter fused with the FP8 block-scale row GEMM (symmetric
    memory): bitwise the fixed-order fp32 chain and within design 5.2's bound of NCCL's
    reduce-scatter at the op and adapter levels (bitwise at world size 2), deterministic under a
    perturbed arrival order; fullgraph compile and CUDA-graph replay of a gather + reduce-scatter
    chain at world sizes 2 and 4."""
    _requires_sm100f()
    checks = _RS_CHECKS + (_RS_GRAPH_CHECKS if world_size in (2, 4) else ())
    _run(world_size, _checks(*checks), "nccl")


@pytest.mark.parametrize("world_size", [2, 4])
def test_shared_input_dense_fallbacks_nccl(world_size):
    _requires_sm100f()
    _run(world_size, _checks(_logic_shared_input_dense_fallbacks), "nccl")


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_shared_input_chunked_graphs_nccl(world_size):
    _requires_sm100f()
    _run(world_size, _checks(_logic_shared_input_chunked_graphs), "nccl")


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_parallel_config_gather_nccl(world_size):
    _requires_sm100f()
    _run(world_size, _checks(_logic_parallel_config_gather), "nccl")


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_parallel_config_collectives_nccl(world_size):
    _requires_sm100f()
    _run(world_size, _checks(_logic_parallel_config_collectives), "nccl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
