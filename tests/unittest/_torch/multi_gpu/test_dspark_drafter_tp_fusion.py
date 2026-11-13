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
"""A TP-sharded MLA DSpark drafter matches the unsharded one.

Under TP the MLA DSpark drafter, which declares the TP fusions, splits its
context projection over the input features (hidden_norm fused into the
all-reduce) and fuses each layer's all-reduce with the residual add and
RMSNorm. Those change the rounding, not the math: the projected context and the
block decode stay within bf16 noise of the single-GPU drafter, which runs
neither fusion. A drafter that does not declare them keeps the unsplit
projection and the unfused norm under TP.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

VOCAB = 128
RANK = 8
NUM_CAPTURE = 2
HIDDEN = 256
INTERMEDIATE = 128
HEADS = 4
Q_LORA = 64
KV_LORA = 512
NOPE = 128
ROPE = 64
V_DIM = 128
LAYERS = 2
BLOCK = 3
LATENT = KV_LORA + ROPE


def _config():
    from transformers import PretrainedConfig

    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=50000.0,
        block_size=BLOCK,
        mask_token_id=VOCAB - 3,
        target_layer_ids=[0, 1],
        markov_rank=RANK,
        markov_head_type="vanilla",
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 64,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    cfg.dflash_config = {"mask_token_id": VOCAB - 3, "target_layer_ids": [0, 1]}
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _weights():
    g = torch.Generator().manual_seed(11)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    w = {
        "context_proj.weight": rnd(HIDDEN, HIDDEN * NUM_CAPTURE),
        "context_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "final_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "embed_tokens.weight": rnd(VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "self_attn.q_a_proj.weight"] = rnd(Q_LORA, HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = rnd(Q_LORA).abs() + 1.0
        w[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * (NOPE + ROPE), Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(LATENT, HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = rnd(KV_LORA).abs() + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = rnd(HEADS * (NOPE + V_DIM), KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * V_DIM)
        w[p + "mlp.gate_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTERMEDIATE)
    return w


def _dflash_config():
    from transformers import Qwen3Config

    return Qwen3Config.from_dict(
        dict(
            architectures=["DFlashDraftModel"],
            model_type="qwen3",
            block_size=BLOCK,
            hidden_size=HIDDEN,
            num_hidden_layers=LAYERS,
            num_attention_heads=HEADS,
            num_key_value_heads=2,
            head_dim=128,
            intermediate_size=INTERMEDIATE,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            max_position_embeddings=256,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            torch_dtype="bfloat16",
            num_target_layers=4,
            tie_word_embeddings=False,
            dflash_config={"target_layer_ids": [0, 1], "mask_token_id": VOCAB - 3},
        )
    )


def _dflash_weights():
    g = torch.Generator().manual_seed(13)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    kv, hd = 2, 128
    w = {
        "fc.weight": rnd(HIDDEN, HIDDEN * NUM_CAPTURE),
        "hidden_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "norm.weight": rnd(HIDDEN).abs() + 1.0,
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(HEADS * hd, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = rnd(kv * hd, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = rnd(kv * hd, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * hd)
        w[p + "self_attn.q_norm.weight"] = rnd(hd).abs() + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(hd).abs() + 1.0
        w[p + "input_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTERMEDIATE)
    return w


def _dflash_drafter(mapping):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    model_config = ModelConfig(
        pretrained_config=_dflash_config(), mapping=mapping, attn_backend="TRTLLM"
    )
    drafter = DFlashForCausalLM(model_config, dflash_attention_backend="VANILLA").to("cuda")
    drafter.load_weights(_dflash_weights())
    return drafter


@torch.inference_mode()
def _project(drafter, captured):
    with discard_nccl_window_tensor_outputs(captured):
        return drafter.project_target_hidden(captured).clone()


def _drafter(mapping, weights=None):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_config(), mapping=mapping, attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend="VANILLA").to("cuda")
    drafter.load_weights(_weights() if weights is None else weights)
    return drafter


@torch.inference_mode()
def _block_decode(drafter, captured, noise):
    """Eager block decode of one batch over the private context arena.

    The all-reduces may return NCCL-window tensors, whose leases must end in a
    tensor scope (the model engine opens one around every forward); the results
    are copied out before the scope releases them.
    """
    batch, ctx_len = captured.shape[0], captured.shape[1]
    with discard_nccl_window_tensor_outputs((captured, noise)):
        projected = drafter.project_target_hidden(captured.reshape(-1, captured.shape[-1]))
        positions = torch.arange(ctx_len, device="cuda").repeat(batch)
        latent, _ = drafter.precompute_context_kv(projected, positions)
        pool = torch.zeros(
            batch, LAYERS, ctx_len + BLOCK, 1, LATENT, dtype=torch.bfloat16, device="cuda"
        )
        pool[:, :, :ctx_len] = latent.view(batch, ctx_len, LAYERS, 1, LATENT).permute(0, 2, 1, 3, 4)
        out = drafter.dflash_forward(
            noise_embedding=noise,
            query_positions=torch.arange(ctx_len, ctx_len + BLOCK, device="cuda").expand(batch, -1),
            num_ctx_per_req=torch.full((batch,), ctx_len, device="cuda"),
            ctx_k_cache=pool,
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.arange(batch, device="cuda"),
        )
        return projected.clone(), out.clone()


def _check(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single = _drafter(Mapping())
        sharded = _drafter(Mapping(world_size=tp, tp_size=tp, rank=rank))
        assert single._fc_input_slice is None and not single._tp_split
        assert sharded._fc_input_slice is not None and sharded._tp_split
        assert sharded.fc.weight.shape[1] == HIDDEN * NUM_CAPTURE // tp

        g = torch.Generator(device="cuda").manual_seed(42)
        captured = torch.randn(3, 20, HIDDEN * NUM_CAPTURE, device="cuda", generator=g)
        captured = (captured * 0.5).to(torch.bfloat16)
        noise = (torch.randn(3, BLOCK, HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        ref_proj, ref_out = _block_decode(single, captured, noise)
        proj, out = _block_decode(sharded, captured, noise)
        # bf16 rounding of a differently ordered sum: ~1 ULP at these magnitudes.
        torch.testing.assert_close(proj, ref_proj, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
        # Every rank ends with the same reduced output.
        gathered = MPI.COMM_WORLD.allgather(out.float().cpu())
        assert all(torch.equal(o, gathered[0]) for o in gathered)

        # An fc input width that does not split over tp_size keeps the whole fc
        # and projects exactly as the unsharded drafter does.
        weights = _weights()
        fc = weights["context_proj.weight"]
        weights["context_proj.weight"] = torch.cat([fc, fc[:, :1]], dim=1)
        unsplit = _drafter(Mapping(world_size=tp, tp_size=tp, rank=rank), weights)
        assert unsplit._tp_split and unsplit._fc_input_slice is None
        assert torch.equal(unsplit.fc.weight.cpu(), weights["context_proj.weight"])
        odd = torch.randn(40, HIDDEN * NUM_CAPTURE + 1, device="cuda", generator=g)
        odd = (odd * 0.5).to(torch.bfloat16)
        assert torch.equal(_project(unsplit, odd), _project(_drafter(Mapping(), weights), odd))
    except Exception:
        traceback.print_exc()
        raise
    return True


def _check_undeclared(tp):
    """A drafter without the TP fusions keeps the unsplit fc and the unfused norm."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
    from tensorrt_llm._torch.models.modeling_dspark import (
        GQADSparkForCausalLM,
        MLADSparkForCausalLM,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        assert MLADSparkForCausalLM._tp_fusions
        assert not DFlashForCausalLM._tp_fusions
        assert not GQADSparkForCausalLM._tp_fusions
        sharded = _dflash_drafter(Mapping(world_size=tp, tp_size=tp, rank=rank))
        assert not sharded._tp_split
        assert sharded._fc_input_slice is None and sharded._fc_all_reduce is None
        assert sharded.fc.weight.shape[1] == HIDDEN * NUM_CAPTURE
        g = torch.Generator(device="cuda").manual_seed(43)
        captured = torch.randn(40, HIDDEN * NUM_CAPTURE, device="cuda", generator=g)
        captured = (captured * 0.5).to(torch.bfloat16)
        # hidden_norm(fc(x)) on every rank, as on a single GPU.
        assert torch.equal(
            _project(sharded, captured), _project(_dflash_drafter(Mapping()), captured)
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_tp_sharded_mla_drafter_matches_single_gpu(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_check, [tp] * tp)
    assert all(results)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_tp_drafter_without_fusions_keeps_unsplit_projection(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_check_undeclared, [tp] * tp)
    assert all(results)
