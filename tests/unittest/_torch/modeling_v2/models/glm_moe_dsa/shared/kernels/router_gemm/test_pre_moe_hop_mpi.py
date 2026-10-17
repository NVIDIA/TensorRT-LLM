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
"""Multi-rank tests of the router_gemm pre-MoE hop with the native MNNVL all-reduce.

Run one pytest process per GPU, every rank executing the same tests:

    srun --mpi=pmix -N2 -n8 python3 -m pytest --import-mode=importlib <this>

With 4 ranks on one node the MNNVL strategy is requested explicitly; with 8
ranks over two nodes ``AUTO`` selects it as in the served model. The chain
under test is: stock row-parallel o_proj -> native fused AR + residual +
RMSNorm -> router_gemm router -> native routing (``noaux_tc_op``), compared bitwise
against the same chain with the native gate, eagerly and under CUDA graph
replay, and across ranks.
"""

import hashlib
import os

import pytest
import torch
from mpi4py import MPI
from utils.llm_data import llm_models_root

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.router_gemm import (  # noqa: E501
    pre_moe_hop,
    router_gemm,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

COMM = MPI.COMM_WORLD
WORLD = COMM.Get_size()
RANK = COMM.Get_rank()

pytestmark = pytest.mark.skipif(WORLD < 2, reason="launch one process per GPU with srun --mpi=pmix")

HIDDEN = 6144
EXPERTS = 256
HEAD_SLICE = 2048  # 64 heads * 256 v_head / TP8 per rank (o_proj in-features)
EPS = 1e-5


def _bits_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.int32 if a.element_size() == 4 else torch.int16),
        b.contiguous().view(torch.int32 if b.element_size() == 4 else torch.int16),
    )


def _digest(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _checkpoint_dir():
    """The GLM-5.2 NVFP4 checkpoint: ``$GLM52_NVFP4_CKPT``, else under ``llm_models_root()``."""
    root = llm_models_root()
    default = os.path.join(str(root), "GLM-5.2-NVFP4") if root is not None else ""
    return os.environ.get("GLM52_NVFP4_CKPT", default)


@pytest.fixture(scope="module")
def env():
    ngpu = torch.cuda.device_count()
    local = int(os.environ.get("SLURM_LOCALID", RANK % ngpu))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=ngpu)
    strategy = AllReduceStrategy.AUTO if mapping.is_multi_node() else AllReduceStrategy.MNNVL
    allreduce = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
    assert allreduce.mnnvl_allreduce is not None, "MNNVL AR not engaged"
    return mapping, allreduce


@pytest.fixture(scope="module")
def layer_weights(env):
    """Real gate/norm/o_proj weights of one MoE layer if the checkpoint is set."""
    from safetensors import safe_open

    ckpt = _checkpoint_dir()
    dev = torch.device("cuda", torch.cuda.current_device())
    if ckpt and os.path.exists(os.path.join(ckpt, "model.safetensors.index.json")):
        import json

        with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]

        def load(name):
            with safe_open(os.path.join(ckpt, wmap[name]), framework="pt") as f:
                return f.get_tensor(name)

        pre = "model.layers.40"
        o_proj = load(f"{pre}.self_attn.o_proj.weight")
        return {
            "source": "real",
            "gate": load(f"{pre}.mlp.gate.weight").to(dev),
            "bias": load(f"{pre}.mlp.gate.e_score_correction_bias").to(dev).to(torch.bfloat16),
            "norm": load(f"{pre}.post_attention_layernorm.weight").to(dev),
            "o_proj_local": o_proj[:, RANK * HEAD_SLICE : (RANK + 1) * HEAD_SLICE]
            .contiguous()
            .to(dev),
        }
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    return {
        "source": "synthetic",
        "gate": (torch.randn(EXPERTS, HIDDEN, generator=g, device=dev) * 0.02).to(torch.bfloat16),
        "bias": torch.zeros(EXPERTS, dtype=torch.bfloat16, device=dev),
        "norm": (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=dev)).to(torch.bfloat16),
        "o_proj_local": (torch.randn(HIDDEN, HEAD_SLICE, generator=g, device=dev) * 0.02).to(
            torch.bfloat16
        ),
    }


def _inputs(rows, seed):
    """Per-rank attention output slice and the shared residual."""
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev)
    g.manual_seed(seed * 1000 + RANK)
    attn = torch.randn(rows, HEAD_SLICE, generator=g, device=dev).to(torch.bfloat16)
    g.manual_seed(seed * 1000 + 999)
    residual = (torch.randn(rows, HIDDEN, generator=g, device=dev) * 4).to(torch.bfloat16)
    return attn, residual


def _native_chain(allreduce, w, attn, residual):
    partial = torch.nn.functional.linear(attn, w["o_proj_local"])
    hidden, res = allreduce(
        partial,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=w["norm"],
            eps=EPS,
            trigger_completion_at_end=False,
        ),
    )
    logits = torch.ops.trtllm.dsv3_router_gemm_op(
        hidden, w["gate"].t(), bias=None, out_dtype=torch.float32
    )
    return hidden, res, logits


def _router_gemm_chain(allreduce, w, attn, residual):
    partial = torch.nn.functional.linear(attn, w["o_proj_local"])
    hidden, res, logits = pre_moe_hop(allreduce, partial, residual, w["norm"], EPS, w["gate"])
    assert logits is not None
    return hidden, res, logits


def _route(logits, bias):
    return torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, 8, 2.5)


def test_weight_source(layer_weights):
    """Real checkpoint weights are used whenever the checkpoint is set."""
    if os.path.exists(os.path.join(_checkpoint_dir(), "model.safetensors.index.json")):
        assert layer_weights["source"] == "real"


@pytest.mark.parametrize("rows", [1, 3, 6, 16])
def test_hop_bitwise_and_rank_agreement(env, layer_weights, rows):
    _, allreduce = env
    attn, residual = _inputs(rows, seed=rows)
    h_n, r_n, l_n = _native_chain(allreduce, layer_weights, attn, residual)
    h_p, r_p, l_p = _router_gemm_chain(allreduce, layer_weights, attn, residual)
    torch.cuda.synchronize()
    assert _bits_equal(h_p, h_n) and _bits_equal(r_p, r_n)
    assert _bits_equal(l_p, l_n)
    v_n, i_n = _route(l_n, layer_weights["bias"])
    v_p, i_p = _route(l_p, layer_weights["bias"])
    assert torch.equal(i_p, i_n) and _bits_equal(v_p, v_n)
    digests = COMM.allgather((_digest(l_p), _digest(i_p)))
    assert all(d == digests[0] for d in digests), digests


def test_hop_graph_replay(env, layer_weights):
    """Capture native and router_gemm chains (with the MoE-style aux fork) and replay."""
    _, allreduce = env
    rows = 6
    attn, residual = _inputs(rows, seed=77)
    aux = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
    w_shared = torch.randn(512, HIDDEN, device="cuda").to(torch.bfloat16)

    def body(chain):
        hidden, res, logits = chain(allreduce, layer_weights, attn, residual)
        ev0.record()
        ids = _route(logits, layer_weights["bias"])[1]
        with torch.cuda.stream(aux):
            ev0.wait()
            shared = torch.nn.functional.linear(hidden, w_shared)
            ev1.record()
        ev1.wait()
        return hidden, res, logits, ids, shared

    router_gemm(
        torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"), layer_weights["gate"]
    )
    outs = {}
    graphs = {}
    for name, chain in (("native", _native_chain), ("router_gemm", _router_gemm_chain)):
        body(chain)
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
            outs[name] = body(chain)
        torch.cuda.current_stream().wait_stream(s)
        graphs[name] = graph
    torch.cuda.synchronize()
    COMM.Barrier()
    for step in range(30):
        a, r = _inputs(rows, seed=1000 + step)
        attn.copy_(a)
        residual.copy_(r)
        graphs["router_gemm"].replay()
        graphs["native"].replay()
        torch.cuda.synchronize()
        for x, y in zip(outs["router_gemm"], outs["native"]):
            assert _bits_equal(x, y), f"step {step}"
        if step % 10 == 0:
            digests = COMM.allgather(_digest(outs["router_gemm"][2]))
            assert all(d == digests[0] for d in digests)
    COMM.Barrier()
