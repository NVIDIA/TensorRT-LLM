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
"""attention_output_tail contract bridge with a synthetic latent, into the native reduction/norm.

The latent is ``tail_fixtures.mix_latent`` (softmax mixes of real compressed-KV
rows with random scores; no attention kernel runs), so this file checks only
the consumer side of the bridge. The native attention producer → attention_output_tail → native
reduction/norm chain runs in ``test_attn_tail_mla_forward_mpi.py``.

Launch one process per GPU under MPI (``srun --mpi=pmix -n <tp>``); every
rank runs the same tests in the same order. Each rank holds its own TP8 shard
of real layer weights (rank ``r`` uses TP8 shard ``r``; with fewer than eight
ranks the all-reduce sums a subset of shards, which still exercises the
bridge). Both destinations are covered:

* MoE layer (3): ``o_proj`` with ``enable_allreduce=False`` → decoder-level
  ``AllReduce(RESIDUAL_RMS_NORM)`` with the real ``post_attention_layernorm``.
* Dense layer (0): ``o_proj`` reduces inside ``Linear.forward``, then the real
  ``post_attention_layernorm(hidden, residual)``.

The native chain uses the native W_UV BMM and the native ROW ``Linear``; the
attention_output_tail chain goes through ``AttnTail.defer_w_uv`` / ``AttnTail.project``, which
returns the rank-local partial without communicating, and then reduces it as
``MLA.forward`` does (``o_proj.all_reduce`` with the caller's params, only
where ``o_proj`` reduces), with the same ``Linear`` and ``AllReduce``
modules. ``project`` must not call the all-reduce. Outputs must match within
the frozen attention_output_tail tolerance and be bit-identical across ranks.
"""

import hashlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
    AttnTail,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

TP8 = 8
HEADS, V_DIM, C_DIM, HIDDEN = 64, 256, 512, 6144
LOCAL_HEADS = HEADS // TP8
EPS = 1e-5

_CKPT = tail_fixtures.checkpoint_dir()


def _comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (_CKPT and os.path.isdir(_CKPT)), reason="checkpoint not available"),
    pytest.mark.skipif(_comm().Get_size() < 2, reason="needs >= 2 MPI ranks"),
]


_SETUP = {}


def _setup(layer):
    if layer not in _SETUP:
        _SETUP[layer] = _build(layer)
    return _SETUP[layer]


def _build(layer):
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = _comm()
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(
        world_size=world, tp_size=world, rank=rank, gpus_per_node=torch.cuda.device_count()
    )
    w_uv, w_o = tail_fixtures.load_tail_weights(_CKPT, layer, [rank], TP8)[rank]
    w_uv = w_uv.cuda()
    # A ROW Linear over `world` ranks whose local slice is this rank's TP8
    # shard: load the [N, world * K_local] matrix built from TP8 shards.
    o_proj = Linear(
        world * LOCAL_HEADS * V_DIM,
        HIDDEN,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=True,
    ).cuda()
    o_proj.weight.data.copy_(w_o.cuda())
    norm_w = tail_fixtures.read_tensor(
        _CKPT, f"model.layers.{layer}.post_attention_layernorm.weight"
    ).cuda()
    norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(norm_w)
    return rank, world, mapping, w_uv, o_proj, norm


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _tail_ns(w_uv, o_proj):
    tail = AttnTail("bridge")
    tail._static_ok = True  # The weights are real; MLA-type checks need a module.
    mla = SimpleNamespace(v_b_proj=w_uv, o_proj=o_proj)
    return tail, mla


def _run_tail(tail, mla, latent, out_rows, all_reduce_params):
    """The attention_output_tail, then the reduction ``MLA.forward`` applies to it."""
    rows = latent.shape[0]
    attn_out = torch.empty((out_rows, LOCAL_HEADS * V_DIM), dtype=torch.bfloat16, device="cuda")
    meta = SimpleNamespace(num_contexts=0, num_tokens=rows)
    o_proj = mla.o_proj
    # Count calls of o_proj's AllReduce module while the tail runs.
    reduce = o_proj.all_reduce
    native_forward = reduce.forward
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return native_forward(*args, **kwargs)

    reduce.forward = counting
    try:
        assert tail.arm(mla, meta)
        assert tail.defer_w_uv(mla, meta, latent, attn_out[:rows])
        partial = tail.project(mla, attn_out)
    finally:
        del reduce.forward
    assert not calls, "the attention_output_tail tail called the all-reduce"
    if o_proj.reduce_output:
        partial = o_proj.all_reduce(partial, all_reduce_params=all_reduce_params)
    return partial


@pytest.mark.parametrize("rows", [1, 6])
def test_moe_layer_bridge(rows):
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams

    rank, world, mapping, w_uv, o_proj, norm = _setup(3)
    allreduce = AllReduce(mapping=mapping, dtype=torch.bfloat16)
    comp = tail_fixtures.compressed_kv_rows(_CKPT, 3)
    latent = tail_fixtures.mix_latent(comp, rows, LOCAL_HEADS, seed=rank)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    residual = torch.randn(
        (rows, HIDDEN), generator=gen, device="cuda", dtype=torch.float32
    ).bfloat16()

    def fused(partial):
        return allreduce(
            partial,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm.weight,
                eps=EPS,
                trigger_completion_at_end=False,
            ),
        )

    no_ar = AllReduceParams(enable_allreduce=False)
    native_partial = tail_reference.native_attn_tail(
        latent, w_uv, None, rows, o_proj=lambda x: o_proj(x, all_reduce_params=no_ar)
    )
    hs_nat, res_nat = fused(native_partial)
    tail, mla = _tail_ns(w_uv, o_proj)
    attn_tail_partial = _run_tail(tail, mla, latent, rows, no_ar)
    hs_attn_tail, res_attn_tail = fused(attn_tail_partial)
    torch.cuda.synchronize()

    ref, _ = tail_reference.oracle_attn_tail(latent, w_uv, o_proj.weight, rows)
    e_attn_tail = tail_reference.error_stats(attn_tail_partial, ref)
    e_nat = tail_reference.error_stats(native_partial, ref)
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (e_attn_tail, e_nat)
    torch.testing.assert_close(res_attn_tail, res_nat, rtol=2**-6, atol=2**-6)
    torch.testing.assert_close(hs_attn_tail, hs_nat, rtol=2**-6, atol=2**-5)
    digests = _comm().allgather((_digest(hs_attn_tail), _digest(res_attn_tail)))
    assert len(set(digests)) == 1, "post-norm outputs differ across ranks"
    print(
        f"[attention_output_tail-bridge] moe rank={rank}/{world} rows={rows} "
        f"partial_err attention_output_tail={e_attn_tail['rel_rms']:.3e} native={e_nat['rel_rms']:.3e} "
        f"hs_maxdiff={(hs_attn_tail.float() - hs_nat.float()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("rows", [1, 6])
def test_dense_layer_bridge(rows):
    from tensorrt_llm._torch.distributed import AllReduceParams

    rank, world, mapping, w_uv, o_proj, norm = _setup(0)
    comp = tail_fixtures.compressed_kv_rows(_CKPT, 0)
    latent = tail_fixtures.mix_latent(comp, rows, LOCAL_HEADS, seed=rank)
    gen = torch.Generator(device="cuda").manual_seed(4321)
    residual = torch.randn(
        (rows, HIDDEN), generator=gen, device="cuda", dtype=torch.float32
    ).bfloat16()
    with_ar = AllReduceParams(enable_allreduce=True)

    native_reduced = tail_reference.native_attn_tail(
        latent, w_uv, None, rows, o_proj=lambda x: o_proj(x, all_reduce_params=with_ar)
    )
    hs_nat, res_nat = norm(native_reduced, residual.clone())
    tail, mla = _tail_ns(w_uv, o_proj)
    attn_tail_reduced = _run_tail(tail, mla, latent, rows, with_ar)
    hs_attn_tail, res_attn_tail = norm(attn_tail_reduced, residual.clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(
        attn_tail_reduced,
        native_reduced,
        rtol=2**-6,
        atol=2**-6 * native_reduced.float().pow(2).mean().sqrt().item(),
    )
    torch.testing.assert_close(hs_attn_tail, hs_nat, rtol=2**-6, atol=2**-5)
    digests = _comm().allgather((_digest(attn_tail_reduced), _digest(hs_attn_tail)))
    assert len(set(digests)) == 1, "reduced outputs differ across ranks"
    print(
        f"[attention_output_tail-bridge] dense rank={rank}/{world} rows={rows} "
        f"hs_maxdiff={(hs_attn_tail.float() - hs_nat.float()).abs().max().item():.3e}"
    )
