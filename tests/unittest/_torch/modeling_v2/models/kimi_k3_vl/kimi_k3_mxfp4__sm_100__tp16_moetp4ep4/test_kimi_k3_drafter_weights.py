# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight load of the tp16_moetp4ep4 target's DSpark drafters (``drafter.K3DSparkDrafter``, GQA, and
``drafter_mla.K3MLADSparkDrafter``, MLA) on CPU, at small widths.

* Both loaders read the checkpoint through ``drafter.CheckpointShards``. A float8 or packed (integer) tensor is
  refused with its name, wherever the loader reads it: a row or column shard, a whole tensor, ``fc`` /
  ``hidden_norm``, a DSpark head, the GQA drafter's trained mask row. An fp32 tensor loads, cast to the bf16
  parameter.
* The MLA drafter's load at TP 1, 2 and 4: each projection and the embedding cut as the built-in loader
  (``load_weight_shard``) cuts them; under TP, ``fc`` is this rank's block of input columns (``drafter.K3FcSlice``,
  ``drafter.fc_columns``). (The GQA drafter's cuts: ``test_kimi_k3_spec_shell.py``.)
* A vocabulary TP does not divide: the MLA drafter's embedding rows padded with zeros on the last rank, as the
  built-in ``Embedding`` pads them; an embedding of another row count is refused.
"""

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter as target_drafter,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter_mla as target_drafter_mla,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

HIDDEN, VOCAB, MARKOV_RANK, LAYERS = 256, 128, 4, 2
# The quantized dtypes a checkpoint could carry: float8, and packed integers (NVFP4 / MXFP4 pack two values a byte).
QUANTIZED_DTYPES = [torch.float8_e4m3fn, torch.uint8, torch.int32]


class _NoAllReduce:
    def __init__(self, *args, **kwargs):
        self.args = (args, kwargs)

    def __call__(self, x):
        return x


@pytest.fixture(autouse=True)
def drafters_build_on_cpu(monkeypatch):
    """The drafters' block attention reads as available, and their TP all-reduces as one-rank stand-ins."""
    monkeypatch.setattr(target_drafter._block_attn_op, "unavailability_reason", lambda: None)
    monkeypatch.setattr(
        target_drafter_mla._mla_ops, "mla_decode_unavailability_reason", lambda: None
    )
    monkeypatch.setattr(target_drafter, "K3AllReduce", _NoAllReduce)
    monkeypatch.setattr(target_drafter_mla, "K3AllReduce", _NoAllReduce)


def _rnd(g, *shape):
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------
# The GQA drafter.
# ----------------------------------------------------------------------------------------------------------------------

GQA_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM, GQA_INTER = 4, 1, 8, 64


def _gqa_drafter():
    cfg = Qwen3Config.from_dict(
        dict(
            architectures=["DSparkDraftModel"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=GQA_INTER,
            num_hidden_layers=LAYERS,
            num_attention_heads=GQA_HEADS,
            num_key_value_heads=GQA_KV_HEADS,
            head_dim=GQA_HEAD_DIM,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            attention_bias=False,
            torch_dtype="bfloat16",
            markov_rank=MARKOV_RANK,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            dflash_config={"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]},
        )
    )
    return target_drafter.K3DSparkDrafter(
        ModelConfig(pretrained_config=cfg), dflash_attention_backend="TRTLLM"
    )


def _gqa_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, kv = GQA_HEADS * GQA_HEAD_DIM, GQA_KV_HEADS * GQA_HEAD_DIM
    w = {
        "fc.weight": _rnd(g, HIDDEN, 2 * HIDDEN),
        "hidden_norm.weight": _rnd(g, HIDDEN),
        "norm.weight": _rnd(g, HIDDEN),
        "embed_tokens.weight": _rnd(g, VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": _rnd(g, VOCAB, MARKOV_RANK),
        "markov_head.markov_w2.weight": _rnd(g, VOCAB, MARKOV_RANK),
        "confidence_head.proj.weight": _rnd(g, 1, HIDDEN + MARKOV_RANK),
        "confidence_head.proj.bias": _rnd(g, 1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        for name, shape in (
            ("self_attn.q_proj", (q, HIDDEN)),
            ("self_attn.k_proj", (kv, HIDDEN)),
            ("self_attn.v_proj", (kv, HIDDEN)),
            ("self_attn.o_proj", (HIDDEN, q)),
            ("self_attn.q_norm", (GQA_HEAD_DIM,)),
            ("self_attn.k_norm", (GQA_HEAD_DIM,)),
            ("input_layernorm", (HIDDEN,)),
            ("post_attention_layernorm", (HIDDEN,)),
            ("mlp.gate_proj", (GQA_INTER, HIDDEN)),
            ("mlp.up_proj", (GQA_INTER, HIDDEN)),
            ("mlp.down_proj", (HIDDEN, GQA_INTER)),
        ):
            w[p + name + ".weight"] = _rnd(g, *shape)
    return w


# (checkpoint key, the name the error gives): a row shard, a column shard, a whole norm, fc, hidden_norm, a DSpark
# head (named by its canonical key), the embedding whose mask row the drafter keeps.
_GQA_READS = [
    ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight"),
    ("layers.1.mlp.down_proj.weight", "layers.1.mlp.down_proj.weight"),
    ("layers.1.input_layernorm.weight", "layers.1.input_layernorm.weight"),
    ("fc.weight", "fc.weight"),
    ("hidden_norm.weight", "hidden_norm.weight"),
    ("markov_head.markov_w1.weight", "markov_w1.weight"),
    ("confidence_head.proj.weight", "confidence_proj.weight"),
    ("embed_tokens.weight", "embed_tokens.weight"),
]


@pytest.mark.parametrize("dtype", QUANTIZED_DTYPES, ids=str)
@pytest.mark.parametrize("key, shown", _GQA_READS, ids=[read[0] for read in _GQA_READS])
def test_gqa_drafter_refuses_a_quantized_tensor(key, shown, dtype):
    weights = _gqa_weights()
    weights[key] = weights[key].to(dtype)
    drafter = _gqa_drafter()
    with pytest.raises(ValueError, match=f"draft checkpoint's {shown} is {dtype}"):
        drafter.load_weights(weights)


def test_gqa_drafter_loads_fp32_tensors_cast_to_bf16():
    weights = _gqa_weights()
    for key in ("layers.0.input_layernorm.weight", "layers.0.self_attn.o_proj.weight"):
        weights[key] = weights[key].float()
    drafter = _gqa_drafter()
    drafter.load_weights(weights)
    layer = drafter.model.layers[0]
    assert layer.input_layernorm.weight.dtype == torch.bfloat16
    assert torch.equal(
        layer.input_layernorm.weight, weights["layers.0.input_layernorm.weight"].bfloat16()
    )
    assert torch.equal(
        layer.self_attn.o_proj.weight, weights["layers.0.self_attn.o_proj.weight"].bfloat16()
    )


# ----------------------------------------------------------------------------------------------------------------------
# The MLA drafter.
# ----------------------------------------------------------------------------------------------------------------------

MLA_HEADS, Q_LORA, KV_LORA, NOPE, ROPE, V_DIM, MLA_INTER = 4, 64, 512, 128, 64, 128, 128


def _mla_drafter(mapping=None, vocab=VOCAB):
    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=MLA_INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=MLA_HEADS,
        num_key_value_heads=MLA_HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=vocab,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "mscale_all_dim": 1.0,
        },
        mask_token_id=vocab - 3,
        markov_rank=MARKOV_RANK,
        markov_head_type="vanilla",
        torch_dtype=torch.bfloat16,
    )
    return target_drafter_mla.K3MLADSparkDrafter(
        ModelConfig(pretrained_config=cfg, mapping=mapping or Mapping()),
        dflash_attention_backend="CUTEDSL",
    )


def _mla_weights(seed=1, vocab=VOCAB):
    """The TorchSpec spellings (``context_proj`` / ``context_norm`` / ``final_norm``)."""
    g = torch.Generator().manual_seed(seed)
    w = {
        "context_proj.weight": _rnd(g, HIDDEN, 2 * HIDDEN),
        "context_norm.weight": _rnd(g, HIDDEN),
        "final_norm.weight": _rnd(g, HIDDEN),
        "embed_tokens.weight": _rnd(g, vocab, HIDDEN),
        "markov_head.markov_w1.weight": _rnd(g, vocab, MARKOV_RANK),
        "markov_head.markov_w2.weight": _rnd(g, vocab, MARKOV_RANK),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        for name, shape in (
            ("self_attn.q_a_proj", (Q_LORA, HIDDEN)),
            ("self_attn.q_a_layernorm", (Q_LORA,)),
            ("self_attn.q_b_proj", (MLA_HEADS * (NOPE + ROPE), Q_LORA)),
            ("self_attn.kv_a_proj_with_mqa", (KV_LORA + ROPE, HIDDEN)),
            ("self_attn.kv_a_layernorm", (KV_LORA,)),
            ("self_attn.kv_b_proj", (MLA_HEADS * (NOPE + V_DIM), KV_LORA)),
            ("self_attn.o_proj", (HIDDEN, MLA_HEADS * V_DIM)),
            ("input_layernorm", (HIDDEN,)),
            ("post_attention_layernorm", (HIDDEN,)),
            ("mlp.gate_proj", (MLA_INTER, HIDDEN)),
            ("mlp.up_proj", (MLA_INTER, HIDDEN)),
            ("mlp.down_proj", (HIDDEN, MLA_INTER)),
        ):
            w[p + name + ".weight"] = _rnd(g, *shape)
    return w


_MLA_READS = [
    ("layers.0.self_attn.kv_b_proj.weight", "layers.0.self_attn.kv_b_proj.weight"),
    ("layers.1.self_attn.o_proj.weight", "layers.1.self_attn.o_proj.weight"),
    ("layers.0.self_attn.q_a_proj.weight", "layers.0.self_attn.q_a_proj.weight"),
    ("embed_tokens.weight", "embed_tokens.weight"),
    ("context_proj.weight", "fc.weight"),
    ("context_norm.weight", "hidden_norm.weight"),
    ("markov_head.markov_w2.weight", "markov_w2.weight"),
]


@pytest.mark.parametrize("dtype", QUANTIZED_DTYPES, ids=str)
@pytest.mark.parametrize("key, shown", _MLA_READS, ids=[read[0] for read in _MLA_READS])
def test_mla_drafter_refuses_a_quantized_tensor(key, shown, dtype):
    weights = _mla_weights()
    weights[key] = weights[key].to(dtype)
    drafter = _mla_drafter()
    with pytest.raises(ValueError, match=f"draft checkpoint's {shown} is {dtype}"):
        drafter.load_weights(weights)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_mla_drafter_load_cuts_each_tensor_like_the_builtin_loader(tp):
    """Each rank's shard: q_b / kv_b, gate / up (fused in that order) and the vocabulary-sharded embedding by rows;
    o_proj and down_proj by columns; the norms, q_a and kv_a whole. Under TP, ``fc`` holds the rank's block of input
    columns; on one rank it is the whole projection."""
    weights = _mla_weights()
    col, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
    for rank in range(tp):
        drafter = _mla_drafter(Mapping(world_size=tp, tp_size=tp, rank=rank))
        drafter.load_weights(dict(weights))

        def shard(name, mode):
            return load_weight_shard(weights[name], tp, rank, mode)

        assert torch.equal(drafter.model.embed_tokens.weight, shard("embed_tokens.weight", col))
        for i, layer in enumerate(drafter.model.layers):
            p = f"layers.{i}."
            attn, mlp = layer.self_attn, layer.mlp
            gate_up = torch.cat([shard(p + f"mlp.{n}_proj.weight", col) for n in ("gate", "up")])
            assert torch.equal(attn.q_a_proj.weight, weights[p + "self_attn.q_a_proj.weight"])
            assert torch.equal(
                attn.q_a_layernorm.weight, weights[p + "self_attn.q_a_layernorm.weight"]
            )
            assert torch.equal(attn.q_b_proj.weight, shard(p + "self_attn.q_b_proj.weight", col))
            assert torch.equal(
                attn.kv_a_proj_with_mqa.weight, weights[p + "self_attn.kv_a_proj_with_mqa.weight"]
            )
            assert torch.equal(
                attn.kv_a_layernorm.weight, weights[p + "self_attn.kv_a_layernorm.weight"]
            )
            assert torch.equal(attn.kv_b_proj.weight, shard(p + "self_attn.kv_b_proj.weight", col))
            assert torch.equal(attn.o_proj.weight, shard(p + "self_attn.o_proj.weight", row))
            assert torch.equal(mlp.gate_up_proj.weight, gate_up)
            assert torch.equal(mlp.down_proj.weight, shard(p + "mlp.down_proj.weight", row))
            assert torch.equal(layer.input_layernorm.weight, weights[p + "input_layernorm.weight"])
            assert torch.equal(
                layer.post_attention_layernorm.weight,
                weights[p + "post_attention_layernorm.weight"],
            )
        assert torch.equal(drafter.model.norm.weight, weights["final_norm.weight"])
        assert torch.equal(drafter.hidden_norm.weight, weights["context_norm.weight"])
        fc = weights["context_proj.weight"]
        if tp == 1:
            assert isinstance(drafter.fc, torch.nn.Linear)
            assert torch.equal(drafter.fc.weight, fc)
        else:
            start, end = target_drafter.fc_columns(fc.shape[1], tp, rank)
            assert isinstance(drafter.fc, target_drafter.K3FcSlice)
            assert (drafter.fc.start, drafter.fc.end) == (start, end)
            assert torch.equal(drafter.fc.weight, fc[:, start:end])


def test_mla_drafter_loads_the_torchspec_names_under_the_model_prefix():
    """``model.context_proj`` / ``model.context_norm`` / ``model.final_norm`` load as ``fc`` / ``hidden_norm`` /
    ``norm``, like the names without the prefix."""
    weights = _mla_weights()
    for name in ("context_proj.weight", "context_norm.weight", "final_norm.weight"):
        weights["model." + name] = weights.pop(name)
    drafter = _mla_drafter()
    drafter.load_weights(dict(weights))
    assert torch.equal(drafter.fc.weight, weights["model.context_proj.weight"])
    assert torch.equal(drafter.hidden_norm.weight, weights["model.context_norm.weight"])
    assert torch.equal(drafter.model.norm.weight, weights["model.final_norm.weight"])


@pytest.mark.parametrize("tp", [2, 4])
def test_mla_drafter_loads_a_vocabulary_tp_does_not_divide(tp):
    """A vocabulary of VOCAB + 1 rows over TP 2 and 4: each rank holds rows of its ceil(vocab / tp) block (the built-in
    loader's cut, ``load_weight_shard``), the last rank's rows past the vocabulary are zero (the built-in
    ``Embedding``'s padding), and the ranks' lookups sum to the table."""
    vocab = VOCAB + 1
    weights = _mla_weights(vocab=vocab)
    table = weights["embed_tokens.weight"]
    rows = -(-vocab // tp)
    ids = torch.arange(vocab)
    looked_up = torch.zeros(vocab, HIDDEN, dtype=torch.bfloat16)
    for rank in range(tp):
        drafter = _mla_drafter(Mapping(world_size=tp, tp_size=tp, rank=rank), vocab=vocab)
        drafter.load_weights(dict(weights))
        embed = drafter.model.embed_tokens
        start, end = rank * rows, min((rank + 1) * rows, vocab)
        assert (embed.vocab_start_index, embed.vocab_end_index) == (start, end)
        assert embed.weight.shape == (rows, HIDDEN)
        builtin = load_weight_shard(table, tp, rank, TensorParallelMode.COLUMN)
        assert torch.equal(embed.weight[: end - start], builtin)
        assert not embed.weight[end - start :].any()
        # The TP all-reduce is a one-rank stand-in here, so each rank returns its own partial lookup.
        looked_up += embed(ids)
    assert torch.equal(looked_up, table)


def test_mla_drafter_refuses_an_embedding_of_another_vocabulary():
    weights = _mla_weights()
    weights["embed_tokens.weight"] = weights["embed_tokens.weight"][:-1]
    drafter = _mla_drafter(Mapping(world_size=2, tp_size=2, rank=1))
    with pytest.raises(ValueError, match="not the vocabulary size"):
        drafter.load_weights(weights)
