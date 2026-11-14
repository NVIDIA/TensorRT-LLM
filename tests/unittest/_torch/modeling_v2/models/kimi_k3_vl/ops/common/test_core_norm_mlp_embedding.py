# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``layers`` norm, gated MLP, embedding, conv weight and two-stream fork against main's
modules (one GPU, one rank): bitwise."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation import (
    sm107_mlp as mlp,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.common import (
    sm107_streams as streams,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.embedding import (
    sm107_embedding as embedding,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm import (
    sm107_norm as norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm import (
    sm107_short_conv as conv,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm as GatedRMSNorm
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.situ import SituAndMul
from tensorrt_llm.mapping import Mapping

ROWS = (1, 4, 16, 33, 128, 1000)


def _bf16(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


@pytest.mark.parametrize("hidden", [512, 1536, 7168])
@torch.inference_mode()
def test_rms_norm_matches_main(hidden):
    eps = 1e-6
    ref = RMSNorm(hidden_size=hidden, eps=eps, dtype=torch.bfloat16).cuda()
    ref.weight.data.copy_(_bf16(hidden))
    prim = norm.RMSNormWeight(hidden, eps, dtype=torch.bfloat16, device="cuda")
    prim.weight.data.copy_(ref.weight)
    assert dict(prim.named_parameters()).keys() == dict(ref.named_parameters()).keys()
    assert prim.variance_epsilon == ref.variance_epsilon == prim.eps
    for m in ROWS:
        x = _bf16(m, hidden, scale=3.0)
        assert torch.equal(prim(x), ref(x)), m
        r = _bf16(m, hidden)
        x_ref, r_ref = ref(x.clone(), r.clone())
        x_prim, r_prim = prim(x.clone(), r.clone())
        assert torch.equal(x_prim, x_ref) and torch.equal(r_prim, r_ref), m


def test_gated_norm_and_conv_weights_register_main_names():
    gated = GatedRMSNorm(128, eps=1e-5)
    prim = norm.RMSNormWeight(128, 1e-5)
    assert dict(prim.named_parameters()).keys() == dict(gated.named_parameters()).keys()
    assert prim.eps == gated.eps
    from fla.modules import ShortConvolution

    ref_conv = ShortConvolution(hidden_size=1536, kernel_size=4, activation="silu")
    prim_conv = conv.ShortConvWeight(1536, 4)
    ref_params = {n: p.shape for n, p in ref_conv.named_parameters()}
    assert {n: p.shape for n, p in prim_conv.named_parameters()} == ref_params
    assert prim_conv.activation == ref_conv.activation


def _main_mlp(hidden, inter, activation, mapping=None):
    config = ModelConfig(mapping=mapping or Mapping())
    with torch.device("cuda"):
        return GatedMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            bias=False,
            activation=activation,
            dtype=torch.bfloat16,
            config=config,
            reduce_output=False,
        )


@pytest.mark.parametrize("act", ["situ", "situ_linear", "silu"])
@torch.inference_mode()
def test_gated_mlp_matches_main(act):
    hidden, inter = 2048, 1536
    if act == "silu":
        activation, kwargs = torch.nn.functional.silu, {}
    else:
        linear_beta = 3.0 if act == "situ_linear" else None
        activation = SituAndMul(beta=2.0, linear_beta=linear_beta, use_fused_activation=True)
        kwargs = dict(beta=2.0, linear_beta=linear_beta)
    ref = _main_mlp(hidden, inter, activation)
    gate, up, down = (
        _bf16(inter, hidden, scale=0.02),
        _bf16(inter, hidden, scale=0.02),
        _bf16(hidden, inter, scale=0.02),
    )
    ref.gate_up_proj.load_weights([{"weight": gate}, {"weight": up}])
    ref.down_proj.load_weights([{"weight": down}])
    prim = mlp.GatedMlp(
        hidden, inter, activation="silu" if act == "silu" else "situ", device="cuda", **kwargs
    )
    prim.load(gate, up, down)
    assert torch.equal(prim.gate_up_proj.weight, ref.gate_up_proj.weight)
    assert torch.equal(prim.down_proj.weight, ref.down_proj.weight)
    for m in ROWS:
        x = _bf16(m, hidden)
        assert torch.equal(prim(x), ref(x)), m


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_gated_mlp_shards_match_main(tp):
    hidden, inter = 1024, 3000
    gate, up, down = (
        torch.randn(inter, hidden),
        torch.randn(inter, hidden),
        torch.randn(hidden, inter),
    )
    for rank in range(tp):
        mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
        ref = _main_mlp(hidden, inter, torch.nn.functional.silu, mapping)
        ref.gate_up_proj.load_weights([{"weight": gate}, {"weight": up}])
        ref.down_proj.load_weights([{"weight": down}])
        prim = mlp.GatedMlp(
            hidden, inter, activation="silu", mapping=mapping, reduce_output=False, device="cuda"
        )
        prim.load(gate, up, down)
        assert torch.equal(prim.gate_up_proj.weight, ref.gate_up_proj.weight)
        assert torch.equal(prim.down_proj.weight, ref.down_proj.weight)


def test_subgroup_mapping_matches_main():
    for world, sub in ((8, 4), (16, 4), (16, 2)):
        for rank in range(world):
            mapping = Mapping(world_size=world, tp_size=world, rank=rank, gpus_per_node=4)
            # GatedMLP keeps its subgroup mapping on the projections.
            config = ModelConfig(mapping=mapping)
            with torch.device("cuda"):
                ref = GatedMLP(
                    hidden_size=64,
                    intermediate_size=64,
                    bias=False,
                    dtype=torch.bfloat16,
                    config=config,
                    overridden_tp_size=sub,
                    reduce_output=False,
                )
            got = mlp.subgroup_mapping(mapping, sub)
            want = ref.down_proj.mapping
            assert (
                got.tp_size,
                got.tp_rank,
                got.tp_group,
                got.world_size,
                got.is_multi_node(),
            ) == (
                want.tp_size,
                want.tp_rank,
                want.tp_group,
                want.world_size,
                want.is_multi_node(),
            )


@pytest.mark.parametrize("vocab", [1000, 163840])
@pytest.mark.parametrize("tp", [1, 3, 4])
@torch.inference_mode()
def test_embedding_matches_main(vocab, tp):
    """Main's vocab-parallel ``Embedding`` (COLUMN, gather_output) before its all-reduce, or its
    whole forward with one rank, on ids that include -1 and both ends of the vocabulary."""
    from tensorrt_llm._torch.modules.embedding import pre_comm_embedding_ops

    hidden = 256
    table = _bf16(vocab, hidden)
    ids = torch.cat(
        [
            torch.randint(0, vocab, (37,), device="cuda"),
            torch.tensor([-1, vocab - 1, 0], device="cuda"),
        ]
    )
    for rank in range(tp):
        mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
        with torch.device("cuda"):
            ref = Embedding(
                vocab,
                hidden,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                reduce_output=False,
            )
        ref.load_weights([{"weight": table}])
        prim = embedding.EmbeddingTable(
            vocab,
            hidden,
            dtype=torch.bfloat16,
            mapping=mapping,
            vocab_parallel=True,
            reduce_output=False,
            device="cuda",
        )
        prim.load(table)
        assert torch.equal(prim.weight, ref.weight)
        assert (prim.vocab_start, prim.vocab_end) == (ref.vocab_start_index, ref.vocab_end_index)
        if tp == 1:
            want = ref(ids)
            assert torch.equal(embedding.gather_embedding(ids, prim.weight), want)
        else:
            want = pre_comm_embedding_ops(
                ids,
                ref.weight,
                tp,
                rank,
                TensorParallelMode.COLUMN,
                ref.vocab_start_index,
                ref.vocab_end_index,
                True,
                ref.padding_size,
            )
        assert torch.equal(prim(ids), want)


def test_fork_join_matches_maybe_execute_in_parallel():
    aux = torch.cuda.Stream()
    x = torch.randn(64, 64, device="cuda")
    seen = {}

    def fn0():
        seen["fn0"] = torch.cuda.current_stream()
        return x @ x

    def fn1():
        seen["fn1"] = torch.cuda.current_stream()
        return (x + 1, [x * 2])

    for multi in (False, True):
        with with_multi_stream(multi):
            ref0, ref1 = maybe_execute_in_parallel(
                fn0, fn1, torch.cuda.Event(), torch.cuda.Event(), aux
            )
            ref_streams = dict(seen)
            got0, got1 = streams.fork_join(fn0, fn1, torch.cuda.Event(), torch.cuda.Event(), aux)
            assert seen == ref_streams
            assert (seen["fn1"] == aux) == multi
        torch.cuda.synchronize()
        assert torch.equal(got0, ref0) and torch.equal(got1[0], ref1[0])
        assert torch.equal(got1[1][0], ref1[1][0])
