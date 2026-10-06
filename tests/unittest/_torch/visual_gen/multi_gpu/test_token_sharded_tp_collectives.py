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
"""Collective tests for the token-sharded TP helper (TokenShardedTP).

Every case runs over a real process group: the ``*_gloo`` tests on CPU (gloo; selected
by ``-m cpu_only`` in the CPU lane) and the ``*_nccl`` tests on GPUs. The functional
collectives used by the helper (``all_gather_single`` / ``reduce_scatter_single``)
are supported by both backends. Each test runs several checks inside one spawn (the
spawn and ``tensorrt_llm`` import dominate the cost): backend-agnostic logic at every
world size on gloo and once (world size 3, a rank straddles samples) on NCCL; the
real-kernel / Linear checks at world sizes 2, 3 and 4 on NCCL.

The file keeps its own spawn harness instead of ``test_wan_tp.run_test_in_distributed``:
every assertion is rank-lockstep (``_check`` all-reduces the verdict), and a failing
worker exits *without* a collective teardown, so ``mp.spawn`` terminates peers that are
blocked in a collective instead of hanging until the NCCL watchdog
(``run_test_in_distributed`` tears the process group down in a ``finally``). Workers also
receive the device (CPU for gloo).

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

from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    RowNorm,
    TokenShardedTP,
    quantize_nvfp4,
    static_nvfp4_input_scale,
    swizzled_sf_numel,
)
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# tests/unittest/_torch/visual_gen: shared SF-layout references (pytest already puts this
# directory on sys.path for the multi_gpu package; spawned workers inherit it).
__extra_import_path__ = [".."]

from token_sharded_tp_test_utils import padded_rows, swizzle_ref, unswizzle_ref

from tensorrt_llm._torch.visual_gen.parallel.token_sharded_modules import (  # noqa: E402
    TokenShardedColumn,
    TokenShardedMLP,
    TokenShardedRow,
    convert_to_token_sharded_tp,
    register_token_sharded_adapter,
)
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder  # noqa: E402


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


# =============================================================================
# Reference helpers (identical on all ranks)
# =============================================================================

# (B, S): unpadded, a rank straddling the sample boundary, interior padding (B >= 2),
# tail padding (B = 1) with fully padded ranks at tp = 4, and a longer sequence.
_SHAPES = [(1, 8), (1, 5), (2, 5), (2, 8), (3, 7), (2, 256)]


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
# B1. reduce_scatter == all-reduce then slice (bitwise on integer-valued partials)
# =============================================================================


def _logic_reduce_scatter(rank, world_size, device):
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    sp = _helper()
    for b, s in _SHAPES:
        plan = sp.begin(b, s)
        gen = torch.Generator().manual_seed(b * 1000 + s)
        partials = torch.randint(-8, 9, (world_size, b * s, 24), generator=gen)
        partial = partials[rank].to(device=device, dtype=dtype)
        rows = slice(plan.row_start, plan.row_start + plan.local_rows)
        ref = padded_rows(partial.view(b, s, -1), plan).clone()  # independent of the helper
        dist.all_reduce(ref)
        # [B * S] input (padded inside) and [B * S_pad] input.
        got = sp.reduce_scatter(partial)
        _check(torch.equal(got, ref[rows]), f"reduce_scatter [B*S] input {(b, s)}", device)
        got = sp.reduce_scatter(sp._add_padding(partial))
        _check(torch.equal(got, ref[rows]), f"reduce_scatter [B*S_pad] input {(b, s)}", device)
        with pytest.raises(ValueError, match="expected .* rows for the current plan"):
            sp.reduce_scatter(partial[:-1])


# =============================================================================
# B2 / B3. all_gather (NVFP4 payload + SF, plain) and shard/unshard round trip
# =============================================================================


def _logic_all_gather(rank, world_size, device):
    sp = _helper()
    for b, s in _SHAPES:
        plan = sp.begin(b, s)
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
            got = sp.all_gather(loc)
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
        # Plain tensors and the shard -> unshard round trip.
        x = torch.randn(b, s, 40, generator=torch.Generator().manual_seed(s)).to(device)
        x_loc = sp.shard(x.transpose(0, 1).contiguous().transpose(0, 1))  # strided input
        ok = torch.equal(x_loc, padded_rows(x, plan)[rows])
        ok = torch.equal(sp.all_gather(x_loc), x.reshape(b * s, -1)) and ok
        ok = torch.equal(sp.unshard(x_loc), x) and ok
        _check(ok, f"plain all_gather / shard-unshard {(b, s)}", device)


def _logic_all_gather_rejects_bad_fp4(rank, world_size, device):
    sp = _helper()
    plan = sp.begin(2, 8)
    m = plan.local_rows
    payload = torch.zeros(m, 64, dtype=torch.uint8, device=device)
    sf = torch.zeros(swizzled_sf_numel(m, 8), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="per-rank dynamic scale"):
        sp.all_gather(
            Fp4QuantizedTensor(payload, sf, reciprocal_scale=torch.ones(1, device=device))
        )
    with pytest.raises(ValueError, match="must be 128x4-swizzled"):
        sp.all_gather(Fp4QuantizedTensor(payload, sf[:-1]))
    with pytest.raises(ValueError, match="must be 128x4-swizzled"):
        sp.all_gather(Fp4QuantizedTensor(payload, sf, is_sf_swizzled=False))
    with pytest.raises(ValueError, match="payload must be uint8"):
        sp.all_gather(Fp4QuantizedTensor(payload.float(), sf))
    with pytest.raises(ValueError, match="payload must be uint8"):
        sp.all_gather(Fp4QuantizedTensor(payload[:-1], sf))


# =============================================================================
# B4. Rank disagreement on the token layout raises on every rank (no hang)
# =============================================================================


def _logic_rank_disagreement(rank, world_size, device):
    sp = _helper()
    seq = 9 if rank == world_size - 1 else 8
    with pytest.raises(ValueError, match="TP ranks disagree on the token layout"):
        sp.begin(1, seq)
    # The group is still usable afterwards, and an agreeing shape works.
    plan = sp.begin(1, 8)
    assert plan.local_rows == pad_up(8, world_size) // world_size


# =============================================================================
# B5. Real fp4_quantize on the shards, gathered == fp4_quantize of all rows (NCCL)
# =============================================================================


def _logic_fp4_quantize_gather(rank, world_size, device):
    sp = _helper()
    for b, s in _SHAPES:
        sp.begin(b, s)
        for k in (256, 80):
            gen = torch.Generator().manual_seed(b * 10 + s + k)
            x = (torch.randn(b, s, k, generator=gen) * 3).to(device, torch.bfloat16)
            scale = (448.0 * 6.0 / x.float().abs().amax()).reshape(1)
            got = sp.all_gather(quantize_nvfp4(sp.shard(x), scale))
            ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(x.reshape(b * s, k), scale, 16, False)
            ok = torch.equal(got.fp4_tensor, ref_fp4) and torch.equal(
                unswizzle_ref(got.scaling_factor, b * s, k // 16),
                unswizzle_ref(ref_sf.reshape(-1), b * s, k // 16),
            )
            _check(ok, f"fp4_quantize shards gathered {(b, s, k)}", device)


# =============================================================================
# B6. row_linear with a row-parallel Linear == unsharded Linear, sliced (NCCL)
# =============================================================================


def _mapping(rank, world_size):
    """TP mapping as VisualGen builds it (VisualGenMapping sets up the TP communicators
    that TRT-LLM Linear/AllReduce construction relies on)."""
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

    vgm = VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)
    assert vgm.tp_rank == rank
    return vgm.to_llm_mapping()


def _logic_row_linear(rank, world_size, device):
    k_in, n_out = 384, 96  # 384 splits evenly over tp in {2, 3, 4}
    torch.manual_seed(0)
    # Small magnitudes keep the per-rank bf16 rounding of the K-partials (inherent to any
    # row-parallel split, all-reduce included) well inside the tolerance.
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.02
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.05 + 0.25  # counted twice -> caught
    ref_lin = Linear(k_in, n_out, bias=True, dtype=torch.bfloat16).to(device)
    ref_lin.load_weights([{"weight": weight, "bias": bias}])
    mapping = _mapping(rank, world_size)
    row_lin = Linear(
        k_in,
        n_out,
        bias=True,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=False,
    ).to(device)
    row_lin.load_weights([{"weight": weight, "bias": bias}])
    k_loc = k_in // world_size
    sp = _helper()
    for b, s in _SHAPES:
        plan = sp.begin(b, s)
        act = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        got = sp.row_linear(row_lin, act[..., rank * k_loc : (rank + 1) * k_loc])
        ref = padded_rows(ref_lin(act).view(b, s, n_out), plan)[
            plan.row_start : plan.row_start + plan.local_rows
        ]
        mask = _real_row_mask(plan, device)
        _check_close(got[mask], ref[mask], device, 1e-2, 1e-2, f"row_linear {(b, s)}")

    # Misuse: an all-reduced row Linear (would sum twice) and mode mismatches.
    row_lin.reduce_output = True
    with pytest.raises(ValueError, match="built with reduce_output=False"):
        sp.row_linear(row_lin, act[..., :k_loc])
    with pytest.raises(ValueError, match="built with reduce_output=False"):
        sp.row_linear_residual_norm(row_lin, act[..., :k_loc], None)
    row_lin.reduce_output = False
    col_lin = Linear(
        k_in,
        n_out * world_size,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        reduce_output=False,
    ).to(device)
    with pytest.raises(ValueError, match="expected linear to be a row-parallel Linear"):
        sp.row_linear(col_lin, act)
    with pytest.raises(ValueError, match="expected linear to be a column-parallel Linear"):
        sp.column_linear(row_lin, act)
    # A row Linear sharded for another TP size (e.g. built from a tp=1 mapping) computes
    # full outputs on every rank; the reduce-scatter would sum them tp times.
    from tensorrt_llm.mapping import Mapping

    unsharded = Linear(
        k_in,
        n_out,
        bias=False,
        dtype=torch.bfloat16,
        mapping=Mapping(),
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=False,
    ).to(device)
    with pytest.raises(
        ValueError, match=f"sharded for tp_size=1, but the helper's TP group has {world_size}"
    ):
        sp.row_linear(unsharded, act)

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP
    from tensorrt_llm.functional import AllReduceStrategy

    mlp = MLP(
        hidden_size=k_in,
        intermediate_size=k_in,
        bias=True,
        dtype=torch.bfloat16,
        config=ModelConfig(mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL),
        reduce_output=False,
    )
    mlp.down_proj.reduce_output = True
    with pytest.raises(ValueError, match="mlp.down_proj to be a row-parallel Linear built with"):
        sp.mlp_residual(mlp, act, act)

    # GatedMLP is not an MLP subclass: the down_proj check is duck-typed, so an
    # all-reducing GatedMLP is rejected too (it would come out tp times too large).
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    gated = GatedMLP(
        hidden_size=k_in,
        intermediate_size=k_in,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL),
        reduce_output=True,
    )
    with pytest.raises(ValueError, match="mlp.down_proj to be a row-parallel Linear built with"):
        sp.mlp_residual(gated, act, act)


# =============================================================================
# B7. column_linear with a static-NVFP4 Linear == same Linear on the full FP4 input
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


def _logic_column_linear_nvfp4(rank, world_size, device):
    k_in, n_out = 256, 128 * world_size
    torch.manual_seed(1)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.1
    mapping = _mapping(rank, world_size)
    sp = _helper()
    for b, s in _SHAPES:
        plan = sp.begin(b, s)
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        ckpt = _nvfp4_checkpoint(weight, x.float().abs().amax().cpu())  # 1-pass calibration
        lin = Linear(
            k_in,
            n_out,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        lin.load_weights([{**ckpt, "bias": bias}])
        lin.post_load_weights()
        scale = static_nvfp4_input_scale(lin)
        assert scale is not None
        got = sp.column_linear(lin, quantize_nvfp4(sp.shard(x), scale))
        ref_in = Fp4QuantizedTensor(
            *torch.ops.trtllm.fp4_quantize(x.reshape(b * s, k_in), scale, 16, False)
        )
        ref = lin(ref_in).view(b, s, -1)
        ok = got.shape == ref.shape == (b, s, n_out // world_size) and torch.equal(got, ref)
        _check(ok, f"NVFP4 column_linear {(b, s, plan.local_rows)}", device)


def _logic_adapters_nvfp4(rank, world_size, device):
    """Converted real Linears: a static-NVFP4 column projection fed this rank's bf16 rows
    quantizes them with its own scale before the all-gather, so its GEMM sees exactly the
    bytes it would produce on all rows; a row projection built to all-reduce reduce-scatters."""
    from tensorrt_llm.functional import AllReduceStrategy

    k_in, n_out = 256, 128 * world_size
    torch.manual_seed(1)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n_out, dtype=torch.bfloat16) * 0.1
    w_row = torch.randn(k_in, n_out, dtype=torch.bfloat16) * 0.05
    mapping = _mapping(rank, world_size)
    for b, s in _SHAPES:
        x = torch.randn(b, s, k_in, generator=torch.Generator().manual_seed(s)).to(
            device, torch.bfloat16
        )
        ckpt = _nvfp4_checkpoint(weight, x.float().abs().amax().cpu())
        col = Linear(
            k_in,
            n_out,
            bias=True,
            dtype=torch.bfloat16,
            mapping=mapping,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            reduce_output=False,
        ).to(device)
        col.load_weights([{**ckpt, "bias": bias}])
        col.post_load_weights()
        row = Linear(
            n_out,
            k_in,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            allreduce_strategy=AllReduceStrategy.NCCL,
        ).to(device)
        row.load_weights([{"weight": w_row}])
        h_ref = col(x.reshape(b * s, k_in)).view(b, s, -1)  # the Linear quantizes all rows
        y_ref = row(h_ref)  # all-reduced
        block = nn.Module()
        block.col, block.row = col, row
        root = nn.Module()
        root.blocks = nn.ModuleList([block])
        tp = _helper()
        convert_to_token_sharded_tp(root, tp, exceptions={"col": "column"})
        plan = tp.begin(b, s)
        h = col(tp.local_view(tp.shard(x)))
        _check(torch.equal(h, h_ref), f"NVFP4 column adapter {(b, s)}", device)
        mask = _real_row_mask(plan, device)
        mine = slice(plan.row_start, plan.row_start + plan.local_rows)
        y = row(h).reshape(plan.local_rows, -1)
        _check_close(
            y[mask],
            padded_rows(y_ref, plan)[mine][mask],
            device,
            1e-2,
            1e-2,
            f"row adapter {(b, s)}",
        )


# =============================================================================
# B8 / B9. torch.compile(fullgraph=True) and CUDA-graph capture of a boundary chain
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
        reduce_output=False,
    ).to(device)
    row.load_weights([{"weight": row_w, "bias": torch.full((d,), 0.1)}])
    return col, row


def _make_chain(sp, col, row, ln_w, ln_b, fp4_scale):
    """norm -> column_linear -> row_linear_residual_norm (+NVFP4 quantize) -> FP4 all-gathers."""

    def chain(x_loc, fp4_payload, fp4_sf, table, h_given):
        mod = sp.per_sample_table(table)  # [n, 3, D]
        shift, scale, gate = mod.unbind(1)
        h = sp.norm(x_loc, RowNorm(scale=scale, shift=shift))
        q = sp.column_linear(col, h)
        x, h2 = sp.row_linear_residual_norm(
            row, q, x_loc, gate=gate, norm=RowNorm(weight=ln_w, bias=ln_b, quant_scale=fp4_scale)
        )
        g2 = sp.all_gather(h2)  # NVFP4 payload + regrouped SF of the chain's own quantize
        g = sp.all_gather(Fp4QuantizedTensor(fp4_payload, fp4_sf))  # of a given FP4 input
        g3 = sp.all_gather(quantize_nvfp4(h_given, fp4_scale))  # quantize of a given bf16 input
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


def _chain_inputs(sp, b, s, d, device):
    gen = torch.Generator().manual_seed(b * 7 + s)
    x = torch.randn(b, s, d, generator=gen).to(device, torch.bfloat16)
    table = torch.randn(b, 3, d, generator=gen).to(device) * 0.1
    plan = sp.plan
    payload = torch.randint(0, 256, (plan.local_rows, d // 2), dtype=torch.uint8, generator=gen)
    sf = torch.randint(
        0, 256, (swizzled_sf_numel(plan.local_rows, d // 16),), dtype=torch.uint8, generator=gen
    )
    h_given = (torch.randn(plan.local_rows, d, generator=gen) * 2).to(device, torch.bfloat16)
    return sp.shard(x), payload.to(device), sf.to(device), table, h_given


def _logic_compile_fullgraph(rank, world_size, device):
    import torch._dynamo

    d = 256
    sp = _helper()
    col, row = _boundary_chain_parts(rank, world_size, device, d)
    ln_w = torch.ones(d, device=device)
    ln_b = torch.zeros(d, device=device)
    fp4_scale = torch.tensor([448.0 * 6.0 / 8.0], device=device)
    chain = _make_chain(sp, col, row, ln_w, ln_b, fp4_scale)

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
        sp.begin(b, s)
        args = _chain_inputs(sp, b, s, d, device)
        torch._dynamo.reset()
        compiled = torch.compile(chain, fullgraph=True)  # raises on any graph break
        compare(compiled(*args), chain(*args), b, s, "compiled")

    # As the pipeline does: one compiled callable, a new shape without a reset (the plan's
    # ints are guarded, so it recompiles), then a revisit of the first shape.
    torch._dynamo.reset()
    compiled = torch.compile(chain, fullgraph=True)
    for b, s in [(2, 256), (1, 5), (2, 256)]:
        sp.begin(b, s)
        args = _chain_inputs(sp, b, s, d, device)
        compare(compiled(*args), chain(*args), b, s, "recompiled")


def _logic_cuda_graph(rank, world_size, device):
    d = 256
    sp = _helper()
    col, row = _boundary_chain_parts(rank, world_size, device, d)
    ln_w = torch.ones(d, device=device)
    ln_b = torch.zeros(d, device=device)
    fp4_scale = torch.tensor([448.0 * 6.0 / 8.0], device=device)
    chain = _make_chain(sp, col, row, ln_w, ln_b, fp4_scale)
    for b, s in [(2, 256), (2, 5), (1, 5)]:
        sp.begin(b, s)
        args = _chain_inputs(sp, b, s, d, device)
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
# B8. The layout: sharder round trip and adapters vs the full computation (exact)
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
        return x @ self.w.t() + self.b


class _FakeMLP(nn.Module):
    def __init__(self, up, down):
        super().__init__()
        self.up, self.down = up, down

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


register_token_sharded_adapter(_FakeColumn, TokenShardedColumn)
register_token_sharded_adapter(_FakeRow, TokenShardedRow)
register_token_sharded_adapter(_FakeMLP, TokenShardedMLP)


def _ints(gen, *shape):
    return torch.randint(-3, 4, shape, generator=gen).float()


def _logic_layout_round_trip(rank, world_size, device):
    tp = _helper()
    sharder = SequenceSharder(size=1, rank=0, group=None)
    sharder.use_token_sharded_tp(tp)
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
    convert_to_token_sharded_tp(root, tp)
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
    _logic_all_gather_rejects_bad_fp4,
    _logic_rank_disagreement,
    _logic_layout_round_trip,
    _logic_adapters,
)
# Real NVFP4 kernels and TRT-LLM Linear/MLP modules over NCCL.
_KERNEL_CHECKS = (
    _logic_fp4_quantize_gather,
    _logic_row_linear,
    _logic_column_linear_nvfp4,
    _logic_adapters_nvfp4,
)
# torch.compile(fullgraph=True) and CUDA-graph capture of a boundary chain.
_GRAPH_CHECKS = (_logic_compile_fullgraph, _logic_cuda_graph)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
