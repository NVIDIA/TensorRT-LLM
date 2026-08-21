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
"""Contract tests: a real standalone DSpark drafter driven by the real worker
that the routing selects for it.

``test_dspark_flavour_dispatch.py`` asserts *which class* each factory returns,
using stubs. That cannot catch a worker handed a draft model whose attributes it
does not have: the two agree on paper and diverge on first contact. The routing
bug this file pins surfaced only as

    AttributeError: 'GQADSparkForCausalLM' object has no attribute 'num_stages'

five minutes into a 16-GPU run, after the weights had loaded -- because nothing
below the factory was ever exercised.

So these tests build the drafter for real and drive ``DFlashWorker``'s lazy
init, which is where the drafter contract actually lives: it reaches for
``fc.weight``, ``block_size``, ``_build_fused_kv_buffers``, ``_num_attn_layers``,
``_num_heads``, ``_num_kv_heads``, ``_head_dim`` and ``_get_attention_mask_args``.
A worker routed by mode instead of by flavour fails here, in seconds, on one GPU.

The drafter is built with the TRTLLM block-decode backend, matching the K3
serving config; the VANILLA default would pull in flash-attn.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import utils as spec_utils
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dspark import DSparkWorker, DSv4DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.mapping import Mapping

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the drafter and the worker buffers are CUDA-resident"
)

VOCAB = 256
RANK = 8
BLOCK_SIZE = 4
MAX_REQUESTS = 4
MAX_SEQ_LEN = 128

TINY = dict(
    architectures=["DSparkDraftModel"],
    model_type="qwen3",
    block_size=BLOCK_SIZE,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    # The fused QK-norm-RoPE kernel the bf16 block decode uses rejects small
    # head dims; 128 is also the real K3 drafter's head_dim.
    head_dim=128,
    intermediate_size=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    vocab_size=VOCAB,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    torch_dtype="bfloat16",
    num_target_layers=2,
    tie_word_embeddings=False,
)

NUM_CAPTURE = 2


def _drafter_config():
    """A standalone DSpark drafter config: qwen3 backbone plus the head set."""
    from transformers import Qwen3Config

    cfg = dict(TINY)
    cfg["dflash_config"] = {
        "mask_token_id": VOCAB - 2,
        "target_layer_ids": [0, 1],
        "projector_type": "dspark",
        "causal": False,
        "shift_label": True,
        "markov_rank": RANK,
        "markov_head_type": "vanilla",
        "use_confidence_head": True,
    }
    return Qwen3Config.from_dict(cfg)


def _drafter_weights(seed=11, legacy_head_keys=False):
    """Synthetic drafter weights in the published key namespace.

    Head tensors are named the way both public checkpoints ship them
    (``markov_head.*`` / ``confidence_head.proj.*``, after the submodules that
    own them); ``legacy_head_keys`` switches to the bare spellings so the
    aliasing is covered from both sides.
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    h, inter = TINY["hidden_size"], TINY["intermediate_size"]
    nh, nkv, hd = (TINY["num_attention_heads"], TINY["num_key_value_heads"], TINY["head_dim"])
    head = {
        "markov_w1.weight": rnd(VOCAB, RANK),
        "markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_proj.weight": rnd(1, h + RANK),
        "confidence_proj.bias": rnd(1),
    }
    if not legacy_head_keys:
        head = {
            "markov_head.markov_w1.weight": head["markov_w1.weight"],
            "markov_head.markov_w2.weight": head["markov_w2.weight"],
            "confidence_head.proj.weight": head["confidence_proj.weight"],
            "confidence_head.proj.bias": head["confidence_proj.bias"],
        }
    weights = {
        "fc.weight": rnd(h, h * NUM_CAPTURE),
        "hidden_norm.weight": rnd(h) + 1.0,
        "norm.weight": rnd(h) + 1.0,
        **head,
    }
    for i in range(TINY["num_hidden_layers"]):
        p = f"layers.{i}."
        weights[p + "self_attn.q_proj.weight"] = rnd(nh * hd, h)
        weights[p + "self_attn.k_proj.weight"] = rnd(nkv * hd, h)
        weights[p + "self_attn.v_proj.weight"] = rnd(nkv * hd, h)
        weights[p + "self_attn.o_proj.weight"] = rnd(h, nh * hd)
        weights[p + "self_attn.q_norm.weight"] = rnd(hd) + 1.0
        weights[p + "self_attn.k_norm.weight"] = rnd(hd) + 1.0
        weights[p + "input_layernorm.weight"] = rnd(h) + 1.0
        weights[p + "post_attention_layernorm.weight"] = rnd(h) + 1.0
        weights[p + "mlp.gate_proj.weight"] = rnd(inter, h)
        weights[p + "mlp.up_proj.weight"] = rnd(inter, h)
        weights[p + "mlp.down_proj.weight"] = rnd(h, inter)
    return weights


def _drafter_config_top_level_spelling():
    """RadixArk/Kimi-K3-DSpark as published: head switches at the top level.

    ``dflash_config`` carries only mask_token_id / target_layer_ids and the
    confidence flag is spelled ``enable_confidence_head``.
    """
    from transformers import Qwen3Config

    cfg = dict(TINY)
    # Verbatim shape of the published config: no shift_label, no
    # projector_type, and dflash_config holding nothing but these two keys.
    cfg["dflash_config"] = {"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]}
    cfg["markov_rank"] = RANK
    cfg["markov_head_type"] = "vanilla"
    cfg["enable_confidence_head"] = True
    cfg["confidence_head_with_markov"] = True
    return Qwen3Config.from_dict(cfg)


@pytest.fixture(scope="module")
def standalone_drafter():
    """The real ``GQADSparkForCausalLM``, weights loaded, on the device."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_drafter_config(), attn_backend="TRTLLM")
    drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(_drafter_weights())
    return drafter


@needs_gpu
def test_top_level_head_spelling_activates_the_heads():
    # The published RadixArk revision spells the switches top-level. Reading
    # only dflash_config resolves markov_rank to 0, which drops markov_w1/w2
    # on the floor and falls back to the DFlash slot convention -- correct
    # output, lower acceptance, nothing raised.
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    model_config = ModelConfig(
        pretrained_config=_drafter_config_top_level_spelling(), attn_backend="TRTLLM"
    )
    drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(_drafter_weights())

    assert drafter._dspark_markov_rank == RANK
    assert drafter._dspark_use_confidence_head is True
    assert drafter.has_markov_head, "markov weights were dropped despite being in the checkpoint"
    # Declared nowhere in the published config, so it rides on the DSpark
    # default. False here would run slots 1..7 on a block_size-7 drafter and
    # read the next request's anchor slot.
    assert drafter._dspark_shift_label is True


@needs_gpu
def test_published_head_weight_names_load():
    # Both public drafters nest the head tensors under the module that owns
    # them. Matching only the bare names leaves markov_w1/w2 in the dict handed
    # to DFlash, which drops them -- or, once the rank resolves, raises
    # "missing markov_w1.weight" on a checkpoint that plainly ships it.
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    for legacy in (False, True):
        model_config = ModelConfig(
            pretrained_config=_drafter_config_top_level_spelling(), attn_backend="TRTLLM"
        )
        drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="TRTLLM").to("cuda")
        drafter.load_weights(_drafter_weights(legacy_head_keys=legacy))
        assert drafter.has_markov_head, f"legacy_head_keys={legacy}"
        assert drafter.confidence_proj_weight is not None, f"legacy_head_keys={legacy}"


@needs_gpu
def test_markov_weights_without_a_resolvable_rank_raise():
    # The inverse of the missing-weights check: weights present, rank resolved
    # to 0. Loading silently would cost acceptance with no signal.
    from transformers import Qwen3Config

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    cfg = dict(TINY)
    cfg["dflash_config"] = {"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]}
    model_config = ModelConfig(pretrained_config=Qwen3Config.from_dict(cfg), attn_backend="TRTLLM")
    drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="TRTLLM").to("cuda")

    with pytest.raises(ValueError, match="markov_rank resolved to 0"):
        drafter.load_weights(_drafter_weights())


def _spec_config(*, embedded):
    """Only the fields the routing and the worker read."""
    return SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        draft_is_embedded_in_target=embedded,
        _use_shared_kv_cache=False,
        _allow_separate_draft_kv_cache=True,
        # K == block_size under the dspark shift_label convention, which is
        # also what DSv4DSparkWorker validates before it touches anything else.
        max_draft_len=BLOCK_SIZE,
        attention_backend="TRTLLM",
    )


def _lazy_init_args():
    spec_metadata = SimpleNamespace(max_num_requests=MAX_REQUESTS)
    attn_metadata = SimpleNamespace(max_seq_len=MAX_SEQ_LEN)
    return spec_metadata, attn_metadata


@needs_gpu
def test_routed_worker_initializes_against_a_real_standalone_drafter(standalone_drafter):
    """The end-to-end contract, and the regression test for the routing bug.

    ``get_spec_worker`` picks the worker; the drafter is the real one the
    builder would produce for the same config. Driving lazy init proves the two
    agree on the draft-model interface. Route by mode instead of by flavour and
    this raises ``AttributeError: ... has no attribute 'num_stages'``.
    """
    spec_config = _spec_config(embedded=False)
    worker = spec_utils.get_spec_worker(
        spec_config, model_config=None, mapping=Mapping(), use_separate_draft_kv_cache=True
    )
    # DFlash lineage (all the drafting plumbing), DSpark leaf (the two heads).
    assert isinstance(worker, DSparkWorker)
    assert isinstance(worker, DFlashWorker)

    worker.set_draft_model(standalone_drafter)
    worker._lazy_init_ctx_buffers(standalone_drafter, *_lazy_init_args())

    assert worker._ctx_buf_inited
    # One scratch slot on top of the request slots, so dummy/padded writes
    # cannot land on a real request's context.
    assert worker._ctx_len.shape == (MAX_REQUESTS + 1,)
    assert worker._dummy_slot == MAX_REQUESTS
    assert worker._batch_to_slot.shape == (MAX_REQUESTS,)
    assert worker._resolved_block_size == BLOCK_SIZE
    assert sorted(worker._free_slots) == list(range(MAX_REQUESTS))


@needs_gpu
def test_routed_worker_sees_the_dspark_heads(standalone_drafter):
    """The heads moved to the DSpark subclass must stay visible to the worker.

    ``DSparkWorker`` probes them defensively (``getattr(..., False)``),
    so a drafter that lost them degrades to plain DFlash silently -- lower
    acceptance, no error. These are the two probes the block-draft step makes.
    """
    assert getattr(standalone_drafter, "has_markov_head", False) is True
    assert getattr(standalone_drafter, "_dspark_shift_label", False) is True
    assert standalone_drafter.markov_w1.shape == (VOCAB, RANK)
    assert standalone_drafter.markov_w2.shape == (VOCAB, RANK)


@needs_gpu
def test_embedded_worker_cannot_drive_a_standalone_drafter(standalone_drafter):
    """Witness for why the routing has to follow the flavour.

    ``DSv4DSparkWorker`` serves the embedded DeepSeek-V4-Pro draft and reads
    V4-draft-only attributes. Handing it a standalone drafter is the exact
    mis-route that reached production, so pin the failure rather than trusting
    the routing test alone to stay correct.
    """
    worker = DSv4DSparkWorker(_spec_config(embedded=True), Mapping())
    spec_metadata, _ = _lazy_init_args()
    with pytest.raises(AttributeError, match="num_stages"):
        worker._lazy_init(standalone_drafter, spec_metadata)


@needs_gpu
def test_dspark_overrides_read_the_drafter_not_a_hardcoded_policy(standalone_drafter):
    """The two DSpark policies must come off the drafter, not the class.

    ``DSparkWorker`` exists only to override the block-output slot
    convention and the Markov bias. Both are declared by the checkpoint, so a
    DSpark drafter trained with the legacy DFlash slot layout (or without a
    Markov head) must still get the base-class behaviour. Hardcoding the
    dspark answer in the subclass would pass every mode/flavour dispatch test
    while silently mis-slotting such a drafter.
    """
    worker = DSparkWorker(_spec_config(embedded=False), Mapping())
    worker.set_draft_model(standalone_drafter)

    # shift_label on -> slots 0..K-1 (anchor slot predicts the first draft
    # token); the DFlash base would return slots 1..K for the same inputs.
    ids = worker._draft_slot_ids(
        standalone_drafter, num_gens=2, block_size=BLOCK_SIZE, num_draft_tokens=3
    )
    assert ids.tolist() == [0, 1, 2, BLOCK_SIZE, BLOCK_SIZE + 1, BLOCK_SIZE + 2]
    base_ids = DFlashWorker._draft_slot_ids(
        worker, standalone_drafter, num_gens=2, block_size=BLOCK_SIZE, num_draft_tokens=3
    )
    assert base_ids.tolist() == [1, 2, 3, BLOCK_SIZE + 1, BLOCK_SIZE + 2, BLOCK_SIZE + 3]

    # A drafter without a Markov head falls through untouched, even on the
    # DSpark subclass.
    logits = torch.randn(2, 3, 8, device="cuda")
    object.__setattr__(standalone_drafter, "_dspark_markov_rank", 0)
    standalone_drafter.markov_w1 = None
    assert standalone_drafter.has_markov_head is False
    out = worker._refine_block_logits(standalone_drafter, logits, {}, None)
    assert out is logits


@needs_gpu
def test_tp_sharded_markov_chain_matches_the_unsharded_result():
    # The sharded branch is unreachable on the validated TEP16/attention-DP
    # config -- there the draft head is full-vocab, so vocab_slice is None and
    # no gather runs. Exercise it here with a fake two-rank split: each rank
    # sees its own logit columns and a markov_w2 sliced the same way, and the
    # chain must agree column-for-column with the unsharded computation.
    from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_chain_logits

    torch.manual_seed(0)
    vocab, rank, batch, block, tp = 32, 4, 3, 5, 2
    shard = vocab // tp
    w1 = torch.randn(vocab, rank, device="cuda")
    w2 = torch.randn(vocab, rank, device="cuda")
    base = torch.randn(batch, block, vocab, device="cuda")
    anchor = torch.randint(0, vocab, (batch,), device="cuda")

    full = dspark_markov_chain_logits(base, anchor, w1, w2)

    for tp_rank in range(tp):
        vocab_slice = slice(tp_rank * shard, (tp_rank + 1) * shard)
        # Stands in for greedy_sample_draft_with_tp_gather: the real one
        # all-gathers local (index, value) pairs, so the chain always advances
        # on the global argmax rather than this rank's local one.
        sharded = dspark_markov_chain_logits(
            base[..., vocab_slice],
            anchor,
            w1,
            w2[vocab_slice],
            argmax_fn=_make_global_argmax(full, tp_rank, shard),
        )
        torch.testing.assert_close(sharded, full[..., vocab_slice])


def _make_global_argmax(full_logits, tp_rank, shard):
    """Replay the global argmax the TP gather would have produced."""
    step = {"i": 0}

    def argmax_fn(_step_logits):
        i = step["i"]
        step["i"] += 1
        return full_logits[:, i].argmax(dim=-1)

    return argmax_fn
