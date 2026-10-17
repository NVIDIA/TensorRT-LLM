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
failing rank cannot leave its peers hanging in a collective.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_token_sharded_tp_collectives.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import functools
import sys
import traceback

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

from token_sharded_tp_test_utils import padded_rows, swizzle_ref, unswizzle_ref


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


def _fp8_block_checkpoint(weight):
    """128x128 block-scale FP8 layout for a bf16 [N, K] weight (N, K multiples of 128): the
    ``weight`` / ``weight_scale`` pair ``FP8BlockScalesLinearMethod`` loads."""
    n, k = weight.shape
    blocks = weight.float().view(n // 128, 128, k // 128, 128).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return {
        "weight": q.permute(0, 2, 1, 3).reshape(n, k),
        "weight_scale": scale.reshape(n // 128, k // 128),
    }


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
    ckpt_col = {**_fp8_block_checkpoint(weight), "bias": bias}
    ckpt_up, ckpt_down = _fp8_block_checkpoint(w_up), _fp8_block_checkpoint(w_down)
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


def _fp8_chain_parts(rank, world_size, device, d=256, n_col=128):
    """A converted FP8 block-scale column Linear (gathers FP8 + packed scales) and a converted
    bf16 row Linear (reduce-scatters)."""
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
    tp = _helper()
    convert_to_token_sharded_tp(_as_model(col=col, row=row), tp, exceptions={"col": "column"})
    col.load_weights([{**_fp8_block_checkpoint(col_w), "bias": torch.zeros(n_col * world_size)}])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
