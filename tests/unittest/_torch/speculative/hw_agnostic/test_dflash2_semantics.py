# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 drafter semantics (https://inco.ai/blog/dflash2/).

Covers the per-sublayer convolutions and the pairwise candidate selector
against nested-loop oracles, the config/weight contract of the released
drafter (incoai/Qwen3.8-27B-DFlash2), and the no-regression property that a
checkpoint without the DFlash 2 fields resolves to plain DFlash. The tiny
end-to-end block-decode parity test needs CUDA + flash_attn and is
skip-guarded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_dflash import (
    DFlash2BlockConv,
    DFlash2CandidateSelector,
    DFlashForCausalLM,
    _is_dflash2_architecture,
    dflash2_grouped_conv,
    dflash2_score_edges,
    dflash2_walk_candidate_paths,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

# ---------------------------------------------------------------------------
# Two-tap dynamic depthwise convolution
# ---------------------------------------------------------------------------


def _ref_grouped_conv(hidden, delta, base_kernel, block_size, group_size):
    """Nested-loop oracle for Conv(x)_t = sum_i k_{t,i} * x_{t-i}."""
    taps, hidden_size = base_kernel.shape
    num_groups = hidden_size // group_size
    batch = hidden.shape[0] // block_size
    x = hidden.view(batch, block_size, num_groups, group_size)
    base = base_kernel.view(taps, num_groups, group_size)
    d = delta.view(batch, block_size, taps, num_groups)
    out = torch.zeros_like(x)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            coefficient = base[tap] + d[:, position, tap, :, None]
            out[:, position] += coefficient * x[:, position - tap]
    return out.flatten(0, 1).flatten(-2)


def _conv_case(block_size, taps, num_groups=4, group_size=2, batch=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden_size = num_groups * group_size
    return (
        torch.randn(batch * block_size, hidden_size, generator=generator),
        torch.randn(batch * block_size, taps, num_groups, generator=generator),
        torch.randn(taps, hidden_size, generator=generator),
        group_size,
    )


@pytest.mark.parametrize("block_size", [5, 8])
@pytest.mark.parametrize("taps", [1, 2, 3])
def test_grouped_conv_matches_reference(block_size, taps):
    hidden, delta, base, group_size = _conv_case(block_size, taps)
    torch.testing.assert_close(
        dflash2_grouped_conv(hidden, delta, base, block_size, group_size),
        _ref_grouped_conv(hidden, delta, base, block_size, group_size),
    )


def test_grouped_conv_anchor_slot_has_no_predecessor():
    """Block slot 0 holds the anchor token, so only the tap-0 term applies."""
    block_size, taps, group_size = 4, 2, 2
    hidden, delta, base, _ = _conv_case(block_size, taps, seed=1)
    out = dflash2_grouped_conv(hidden, delta, base, block_size, group_size)
    coefficients = base[0] + delta[:, 0].repeat_interleave(group_size, dim=-1)
    torch.testing.assert_close(out[::block_size], (coefficients * hidden)[::block_size])


def test_grouped_conv_does_not_cross_request_boundary():
    """Requests are packed back to back; taps must not cross the boundary."""
    block_size, taps, group_size = 4, 2, 2
    hidden, delta, base, _ = _conv_case(block_size, taps, batch=2, seed=2)
    baseline = dflash2_grouped_conv(hidden, delta, base, block_size, group_size)

    perturbed = hidden.clone()
    perturbed[block_size - 1] += 10.0  # last slot of request 0
    out = dflash2_grouped_conv(perturbed, delta, base, block_size, group_size)

    torch.testing.assert_close(out[block_size:], baseline[block_size:])
    assert not torch.allclose(out[:block_size], baseline[:block_size])


def _block_conv(hidden_size=8, taps=2, group_size=2, seed=3):
    generator = torch.Generator().manual_seed(seed)
    num_groups = hidden_size // group_size
    return DFlash2BlockConv(
        base_kernel=torch.randn(2, taps, hidden_size, generator=generator),
        kernel_projection_weight=torch.randn(
            2 * taps * num_groups, hidden_size, generator=generator
        ),
        taps=taps,
        group_size=group_size,
        hidden_size=hidden_size,
    )


def test_block_conv_both_sides_share_the_input_projection():
    """``finish`` must use the coefficients ``prepare`` handed back, not a
    reprojection of the sublayer output."""
    hidden_size, taps, group_size, block_size, batch = 8, 2, 2, 4, 2
    conv = _block_conv(hidden_size, taps, group_size)
    generator = torch.Generator().manual_seed(9)
    sublayer_input = torch.randn(batch * block_size, hidden_size, generator=generator)
    sublayer_output = torch.randn(batch * block_size, hidden_size, generator=generator)

    convolved, coefficients = conv.prepare(sublayer_input, block_size)
    finished = conv.finish(sublayer_output, coefficients, block_size)

    projected = conv.kernel_projection(sublayer_input).view(
        batch * block_size, 2, taps, hidden_size // group_size
    )
    torch.testing.assert_close(
        convolved,
        _ref_grouped_conv(
            sublayer_input, projected[:, 0], conv.base_kernel[0], block_size, group_size
        ),
    )
    torch.testing.assert_close(
        finished,
        _ref_grouped_conv(
            sublayer_output, projected[:, 1], conv.base_kernel[1], block_size, group_size
        ),
    )
    # The output side must still depend on the output itself.
    assert not torch.allclose(
        finished, conv.finish(torch.randn_like(sublayer_output), coefficients, block_size)
    )


def test_block_conv_rejects_mismatched_checkpoint_shapes():
    generator = torch.Generator().manual_seed(4)
    with pytest.raises(ValueError, match="base_kernel has shape"):
        DFlash2BlockConv(
            base_kernel=torch.randn(2, 3, 8, generator=generator),
            kernel_projection_weight=torch.randn(16, 8, generator=generator),
            taps=2,
            group_size=2,
            hidden_size=8,
        )
    with pytest.raises(ValueError, match="kernel_projection has shape"):
        DFlash2BlockConv(
            base_kernel=torch.randn(2, 2, 8, generator=generator),
            kernel_projection_weight=torch.randn(8, 8, generator=generator),
            taps=2,
            group_size=2,
            hidden_size=8,
        )
    with pytest.raises(ValueError, match="conv_group_size=3 must divide"):
        DFlash2BlockConv(
            base_kernel=torch.randn(2, 2, 8, generator=generator),
            kernel_projection_weight=torch.randn(16, 8, generator=generator),
            taps=2,
            group_size=3,
            hidden_size=8,
        )


# ---------------------------------------------------------------------------
# Pairwise candidate selector
# ---------------------------------------------------------------------------

VOCAB, RANK, TOP_K, B, K, HID = 512, 8, 4, 3, 5, 16


def _selector_inputs(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        predecessor=torch.randn(VOCAB, RANK, generator=generator),
        successor=torch.randn(VOCAB, RANK, generator=generator),
        candidate_ids=torch.randint(VOCAB, (B, K, TOP_K), generator=generator),
        unary=torch.randn(B, K, TOP_K, generator=generator),
        gate=torch.randn(B, K, RANK, generator=generator),
        anchors=torch.randint(VOCAB, (B,), generator=generator),
    )


def test_score_edges_matches_sequential_reference():
    """S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>, anchored at position 0."""
    inputs = _selector_inputs()
    actual = dflash2_score_edges(
        inputs.predecessor,
        inputs.successor,
        inputs.candidate_ids,
        inputs.unary,
        inputs.gate,
        inputs.anchors,
    )

    expected = torch.empty_like(actual)
    for step in range(K):
        previous = (
            inputs.anchors[:, None].expand(-1, TOP_K)
            if step == 0
            else inputs.candidate_ids[:, step - 1]
        )
        expected[:, step] = inputs.unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            inputs.predecessor[previous] * inputs.gate[:, step, None],
            inputs.successor[inputs.candidate_ids[:, step]],
        )
    torch.testing.assert_close(actual, expected)


def test_score_edges_accumulates_in_fp32():
    """bf16 rounding moves candidate argmaxes, so scores must not inherit the
    draft dtype."""
    inputs = _selector_inputs()
    scores = dflash2_score_edges(
        inputs.predecessor.bfloat16(),
        inputs.successor.bfloat16(),
        inputs.candidate_ids,
        inputs.unary.bfloat16(),
        inputs.gate.bfloat16(),
        inputs.anchors,
    )
    assert scores.dtype == torch.float32


def test_score_edges_step0_ignores_the_predecessor_axis():
    """All step-0 predecessor slots hold the anchor, so the walk's arbitrary
    entry row is well defined."""
    inputs = _selector_inputs(seed=4)
    first = dflash2_score_edges(
        inputs.predecessor,
        inputs.successor,
        inputs.candidate_ids,
        inputs.unary,
        inputs.gate,
        inputs.anchors,
    )[:, 0]
    for predecessor in range(1, TOP_K):
        torch.testing.assert_close(first[:, predecessor], first[:, 0])


def test_walk_reads_the_row_of_its_own_previous_choice():
    generator = torch.Generator().manual_seed(5)
    candidate_ids = torch.randint(VOCAB, (B, K, TOP_K), generator=generator)
    edge_scores = torch.randn(B, K, TOP_K, TOP_K, generator=generator)

    realized = dflash2_walk_candidate_paths(candidate_ids, edge_scores)

    for request in range(B):
        predecessor = 0
        for step in range(K):
            row = edge_scores[request, step, predecessor]
            torch.testing.assert_close(realized[request, step], row)
            predecessor = int(row.argmax())


def test_walk_recovers_a_coherent_chain_that_marginal_picks_miss():
    """A decoy scores well under every predecessor but slightly worse than the
    planted chain, so only conditioning on the walk's own choice separates
    them."""
    generator = torch.Generator().manual_seed(6)
    candidate_ids = torch.randint(VOCAB, (B, K, TOP_K), generator=generator)
    edge_scores = 0.1 * torch.randn(B, K, TOP_K, TOP_K, generator=generator)
    # The walk enters step 0 at predecessor slot 0.
    chain = torch.cat(
        (torch.zeros(B, 1, dtype=torch.long), torch.randint(TOP_K, (B, K), generator=generator)),
        dim=1,
    )
    for request in range(B):
        for step in range(K):
            successor = int(chain[request, step + 1])
            edge_scores[request, step, :, (successor + 1) % TOP_K] += 8.0
            edge_scores[request, step, chain[request, step], successor] += 10.0

    realized = dflash2_walk_candidate_paths(candidate_ids, edge_scores)

    torch.testing.assert_close(realized.argmax(-1), chain[:, 1:])
    marginal = edge_scores.mean(dim=2).argmax(-1)
    assert not torch.equal(marginal, chain[:, 1:])


def _selector(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return DFlash2CandidateSelector(
        predecessor_codebook=torch.randn(VOCAB, RANK, generator=generator),
        successor_codebook=torch.randn(VOCAB, RANK, generator=generator),
        hidden_projection_weight=torch.randn(RANK, HID, generator=generator),
        top_k=TOP_K,
        vocab_size=VOCAB,
        rank=RANK,
    )


def test_selector_rejects_mismatched_checkpoint_shapes():
    generator = torch.Generator().manual_seed(8)
    with pytest.raises(ValueError, match="predecessor_codebook has shape"):
        DFlash2CandidateSelector(
            predecessor_codebook=torch.randn(VOCAB - 1, RANK, generator=generator),
            successor_codebook=torch.randn(VOCAB, RANK, generator=generator),
            hidden_projection_weight=torch.randn(RANK, HID, generator=generator),
            top_k=TOP_K,
            vocab_size=VOCAB,
            rank=RANK,
        )
    with pytest.raises(ValueError, match="hidden_projection has shape"):
        DFlash2CandidateSelector(
            predecessor_codebook=torch.randn(VOCAB, RANK, generator=generator),
            successor_codebook=torch.randn(VOCAB, RANK, generator=generator),
            hidden_projection_weight=torch.randn(RANK + 1, HID, generator=generator),
            top_k=TOP_K,
            vocab_size=VOCAB,
            rank=RANK,
        )


def _dflash2_wrapper(**attributes):
    """A DFlashForCausalLM with only the attributes under test populated;
    the real constructor needs a CUDA device and a backbone."""
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(wrapper)
    for name, value in attributes.items():
        setattr(wrapper, name, value)
    return wrapper


def test_select_candidate_path_restricts_logits_to_the_walk():
    """The rewritten rows must reproduce the walk under argmax and carry its
    scores, so both acceptance paths read the proposal distribution."""
    selector = _selector()
    wrapper = _dflash2_wrapper(candidate_selector=selector)
    generator = torch.Generator().manual_seed(11)
    block_logits = torch.randn(B, K, VOCAB, generator=generator)
    hidden_states = torch.randn(B, K, HID, generator=generator)
    anchors = torch.randint(VOCAB, (B,), generator=generator)

    unary, candidate_ids = torch.topk(block_logits, TOP_K, dim=-1)
    expected = dflash2_walk_candidate_paths(
        candidate_ids, selector(candidate_ids, unary, hidden_states, anchors)
    )

    rewritten = wrapper.select_candidate_path(
        block_logits, candidate_ids, unary, hidden_states, anchors
    )

    # Only the candidates are reachable, and they carry the walk's scores.
    assert (rewritten > float("-inf")).sum() == B * K * TOP_K
    torch.testing.assert_close(rewritten.gather(-1, candidate_ids), expected)
    torch.testing.assert_close(
        rewritten.argmax(-1),
        candidate_ids.gather(-1, expected.argmax(-1, keepdim=True)).squeeze(-1),
    )


# ---------------------------------------------------------------------------
# Worker-side global top-k
# ---------------------------------------------------------------------------


def _worker(mapping=None, d2t=None):
    worker = DFlashWorker.__new__(DFlashWorker)
    worker.mapping = mapping
    worker._d2t = d2t
    return worker


def test_global_top_k_full_vocab_rewrites_in_place():
    """The single-rank case must not allocate a second full-vocab tensor: a
    [num_gens, K, vocab] copy is hundreds of MB at real vocab sizes."""
    worker = _worker()
    generator = torch.Generator().manual_seed(12)
    gen_logits = torch.randn(B, K, VOCAB, generator=generator)

    candidate_ids, unary, block_logits = worker._dflash2_global_top_k(
        gen_logits, SimpleNamespace(), top_k=TOP_K, full_vocab=VOCAB
    )

    assert block_logits is gen_logits
    expected_unary, expected_ids = torch.topk(gen_logits, TOP_K, dim=-1)
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary, expected_unary)


def test_global_top_k_tp_shards_agree_with_the_full_vocab_result():
    """The selector indexes full-vocab codebooks, so every rank must end up
    with the global candidates the unsharded path would pick."""
    tp_size = 4
    shard_width = VOCAB // tp_size
    generator = torch.Generator().manual_seed(13)
    full_logits = torch.randn(B, K, VOCAB, generator=generator)
    spec_metadata = SimpleNamespace(draft_vocab_size=VOCAB, vocab_size=VOCAB)
    expected_unary, expected_ids = torch.topk(full_logits, TOP_K, dim=-1)

    for tp_rank in range(tp_size):
        mapping = SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, enable_attention_dp=False)
        worker = _worker(mapping)
        shard = full_logits[..., tp_rank * shard_width : (tp_rank + 1) * shard_width]

        # Emulate the collective by replaying the reduction over all shards.
        def fake_allgather(local, _mapping, dim):
            pairs = []
            for rank in range(tp_size):
                rank_shard = full_logits[..., rank * shard_width : (rank + 1) * shard_width]
                values, ids = torch.topk(rank_shard, TOP_K, dim=-1)
                pairs.append(
                    torch.stack(
                        [(ids + rank * shard_width).float(), values.float()], dim=-1
                    ).flatten(-2)
                )
            gathered = torch.cat(pairs, dim=dim)
            torch.testing.assert_close(pairs[tp_rank], local)
            return gathered

        with patch("tensorrt_llm._torch.distributed.ops.allgather", side_effect=fake_allgather):
            candidate_ids, unary, block_logits = worker._dflash2_global_top_k(
                shard, spec_metadata, top_k=TOP_K, full_vocab=VOCAB
            )

        assert block_logits.shape == (B, K, VOCAB)
        assert (block_logits == float("-inf")).all()
        torch.testing.assert_close(candidate_ids, expected_ids)
        torch.testing.assert_close(unary, expected_unary.float())


def test_global_top_k_rejects_a_width_it_cannot_interpret():
    worker = _worker(SimpleNamespace(tp_size=2, tp_rank=0, enable_attention_dp=False))
    with pytest.raises(NotImplementedError, match="plain TP column shard"):
        worker._dflash2_global_top_k(
            torch.randn(B, K, VOCAB // 3),
            SimpleNamespace(draft_vocab_size=VOCAB, vocab_size=VOCAB),
            top_k=TOP_K,
            full_vocab=VOCAB,
        )


def test_selector_rejects_a_remapped_draft_vocab():
    """The codebooks are indexed by draft-vocab id, so a d2t remap would score
    the wrong tokens."""
    worker = _worker(d2t=torch.zeros(VOCAB, dtype=torch.long))
    with pytest.raises(NotImplementedError, match="shared draft/target"):
        worker._apply_dflash2_selector(
            _dflash2_wrapper(candidate_selector=_selector()),
            torch.randn(B, K, VOCAB),
            torch.randn(B, K, HID),
            torch.zeros(B, dtype=torch.long),
            SimpleNamespace(),
        )


# ---------------------------------------------------------------------------
# Config contract of the released drafter
# ---------------------------------------------------------------------------

# incoai/Qwen3.8-27B-DFlash2 config.json, shrunk to a testable size. The
# dflash_config keys and the top-level is_causal are verbatim.
RELEASED_DFLASH_CONFIG = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": VOCAB - 2,
    "selector_rank": RANK,
    "selector_top_k": TOP_K,
    "target_layer_ids": [0, 1],
}


def test_architecture_detection():
    """Matching on the "dflash2" stem picks up per-target label variants."""

    def config(architectures):
        return SimpleNamespace(architectures=architectures)

    assert _is_dflash2_architecture(config(["DFlash2DraftModel"]))
    assert _is_dflash2_architecture(config(["DFlash2Qwen3ForCausalLM"]))
    assert _is_dflash2_architecture(config(["dflash_2_draft_model"]))
    assert not _is_dflash2_architecture(config(["DFlashDraftModel"]))
    assert not _is_dflash2_architecture(config(["Qwen3ForCausalLM"]))
    assert not _is_dflash2_architecture(config(None))


def _validation_wrapper(**overrides):
    values = dict(
        _dflash2_conv_taps=2,
        _dflash2_conv_group_size=16,
        _dflash2_selector_rank=RANK,
        _dflash2_selector_top_k=TOP_K,
    )
    values.update(overrides)
    return _dflash2_wrapper(**values)


def test_validate_dflash2_config_accepts_the_released_recipe():
    _validation_wrapper()._validate_dflash2_config()


@pytest.mark.parametrize(
    "missing",
    [
        "_dflash2_conv_taps",
        "_dflash2_conv_group_size",
        "_dflash2_selector_rank",
        "_dflash2_selector_top_k",
    ],
)
def test_validate_dflash2_config_rejects_a_partial_recipe(missing):
    """A partial recipe would silently degrade to DFlash 1 acceptance."""
    with pytest.raises(ValueError, match="missing required dflash_config"):
        _validation_wrapper(**{missing: 0})._validate_dflash2_config()


def test_validate_dflash2_config_rejects_a_degenerate_top_k():
    with pytest.raises(ValueError, match="no candidates to choose between"):
        _validation_wrapper(_dflash2_selector_top_k=1)._validate_dflash2_config()


def _mask_wrapper(config, is_dflash2=True, sliding_layers_causal=False):
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = config
    wrapper._is_dflash2 = is_dflash2
    wrapper._sliding_layers_causal = sliding_layers_causal
    return wrapper


def test_released_drafter_attention_is_non_causal_and_symmetrically_windowed():
    """The released config declares is_causal=false with all-sliding layers;
    read as causal it would mask the block's own future positions."""
    wrapper = _mask_wrapper(
        SimpleNamespace(
            is_causal=False,
            num_hidden_layers=5,
            layer_types=["sliding_attention"] * 5,
            sliding_window=2048,
            use_sliding_window=True,
            max_window_layers=5,
        )
    )

    for layer_idx in range(5):
        assert wrapper._get_attention_mask_args(layer_idx) == (False, (2047, 2047))


def test_explicit_is_causal_overrides_the_layer_type_default():
    """Without the flag a sliding layer reads as causal; the flag has to win in
    both directions."""
    config = dict(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
        use_sliding_window=True,
    )

    inferred = _mask_wrapper(SimpleNamespace(**config))
    assert inferred._get_attention_mask_args(0) == (True, (511, 0))
    assert inferred._get_attention_mask_args(1) == (False, (-1, -1))

    non_causal = _mask_wrapper(SimpleNamespace(is_causal=False, **config))
    assert non_causal._get_attention_mask_args(0) == (False, (511, 511))
    # A full_attention layer has no window either way.
    assert non_causal._get_attention_mask_args(1) == (False, (-1, -1))

    causal = _mask_wrapper(SimpleNamespace(is_causal=True, **config))
    assert causal._get_attention_mask_args(0) == (True, (511, 0))
    # Causal without a window: unbounded left context, no future.
    assert causal._get_attention_mask_args(1) == (True, (-1, -1))


def test_is_causal_is_ignored_for_a_plain_dflash_drafter():
    """No-regression: an older drafter that happens to carry is_causal must
    keep resolving its windows the old way."""
    config = SimpleNamespace(
        is_causal=False,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
        use_sliding_window=True,
    )
    wrapper = _mask_wrapper(config, is_dflash2=False)
    assert wrapper._get_attention_mask_args(0) == (True, (511, 0))
    assert wrapper._get_attention_mask_args(1) == (False, (-1, -1))


# ---------------------------------------------------------------------------
# Tiny end-to-end drafter: config parsing, weight loading, block-decode
# parity vs an fp32 eager oracle (needs CUDA + flash_attn).
# ---------------------------------------------------------------------------

TINY = dict(
    architectures=["DFlash2DraftModel"],
    model_type="qwen3",
    is_causal=False,
    block_size=8,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    # The fusedQKNormRope kernel used by the bf16 block decode rejects small
    # head dims; 128 is the real drafter's.
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
    num_target_layers=4,
    tie_word_embeddings=False,
    layer_types=["sliding_attention"] * 2,
    # Narrow enough (vs CTX_LEN) that the window demonstrably binds.
    sliding_window=16,
    use_sliding_window=True,
)

CONV_TAPS = RELEASED_DFLASH_CONFIG["conv_kernel_size"]
CONV_GROUP_SIZE = RELEASED_DFLASH_CONFIG["conv_group_size"]
BLOCK = TINY["block_size"]
CTX_LEN = 24
NUM_CAPTURE = 2


def _tiny_config(dflash2: bool):
    from transformers import Qwen3Config

    config = dict(TINY)
    dflash_config = dict(RELEASED_DFLASH_CONFIG)
    if not dflash2:
        for key in ("conv_group_size", "conv_kernel_size", "selector_rank", "selector_top_k"):
            del dflash_config[key]
        config["architectures"] = ["DFlashDraftModel"]
    config["dflash_config"] = dflash_config
    return Qwen3Config.from_dict(config)


def _tiny_weights(dflash2: bool, seed=7):
    generator = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    hidden, inter = TINY["hidden_size"], TINY["intermediate_size"]
    num_heads = TINY["num_attention_heads"]
    num_kv_heads = TINY["num_key_value_heads"]
    head_dim = TINY["head_dim"]
    num_groups = hidden // CONV_GROUP_SIZE
    weights = {
        "fc.weight": rnd(hidden, hidden * NUM_CAPTURE),
        "hidden_norm.weight": rnd(hidden) + 1.0,
        "norm.weight": rnd(hidden) + 1.0,
    }
    if dflash2:
        weights.update(
            {
                "candidate_selector.predecessor_codebook": rnd(VOCAB, RANK),
                "candidate_selector.successor_codebook": rnd(VOCAB, RANK),
                "candidate_selector.hidden_projection.weight": rnd(RANK, hidden),
            }
        )
    for layer_idx in range(TINY["num_hidden_layers"]):
        prefix = f"layers.{layer_idx}."
        weights[prefix + "self_attn.q_proj.weight"] = rnd(num_heads * head_dim, hidden)
        weights[prefix + "self_attn.k_proj.weight"] = rnd(num_kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.v_proj.weight"] = rnd(num_kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.o_proj.weight"] = rnd(hidden, num_heads * head_dim)
        weights[prefix + "self_attn.q_norm.weight"] = rnd(head_dim) + 1.0
        weights[prefix + "self_attn.k_norm.weight"] = rnd(head_dim) + 1.0
        weights[prefix + "input_layernorm.weight"] = rnd(hidden) + 1.0
        weights[prefix + "post_attention_layernorm.weight"] = rnd(hidden) + 1.0
        weights[prefix + "mlp.gate_proj.weight"] = rnd(inter, hidden)
        weights[prefix + "mlp.up_proj.weight"] = rnd(inter, hidden)
        weights[prefix + "mlp.down_proj.weight"] = rnd(hidden, inter)
        if dflash2:
            for name in ("attention_conv", "mlp_conv"):
                weights[prefix + f"{name}.base_kernel"] = rnd(2, CONV_TAPS, hidden)
                weights[prefix + f"{name}.kernel_projection.weight"] = rnd(
                    2 * CONV_TAPS * num_groups, hidden
                )
    return weights


def _build_drafter(dflash2: bool, weights):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(pretrained_config=_tiny_config(dflash2), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(model_config).to("cuda")
    drafter.load_weights(dict(weights))
    return drafter


def _rms(x, weight, eps=1e-6):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


def _rope(x, positions, theta=10000.0):
    # NeoX half-split convention, matching RotaryEmbedding(is_neox=True).
    head_dim = x.shape[-1]
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angle = positions.double().unsqueeze(-1) * inv_freq
    cos = angle.cos().float().unsqueeze(1)
    sin = angle.sin().float().unsqueeze(1)
    left, right = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def _oracle_conv(weights, prefix, side, hidden, coefficients):
    """One side of a two-tap dynamic depthwise convolution over the block."""
    base = weights[prefix + ".base_kernel"][side]
    return _ref_grouped_conv(hidden, coefficients[:, side], base, BLOCK, CONV_GROUP_SIZE)


def _oracle_conv_coefficients(weights, prefix, hidden):
    projection = weights[prefix + ".kernel_projection.weight"]
    num_groups = TINY["hidden_size"] // CONV_GROUP_SIZE
    return (hidden @ projection.T).view(hidden.shape[0], 2, CONV_TAPS, num_groups)


def _oracle_block_decode(weights, captured, noise_embedding, dflash2):
    """fp32 eager port of the DFlash 2 block decode: the DFlash pre-norm
    drafter stack over [context ; draft block], each sublayer wrapped in a
    block-local convolution."""
    w = {k: v.float() for k, v in weights.items()}
    num_heads = TINY["num_attention_heads"]
    num_kv_heads = TINY["num_key_value_heads"]
    head_dim = TINY["head_dim"]
    window = TINY["sliding_window"]
    context_positions = torch.arange(CTX_LEN, dtype=torch.long)
    query_positions = torch.arange(CTX_LEN, CTX_LEN + BLOCK, dtype=torch.long)
    all_positions = torch.cat([context_positions, query_positions])

    # Target features are constant across layers, with no input_layernorm on
    # the context path (generic DFlash).
    context = _rms(captured.float() @ w["fc.weight"].T, w["hidden_norm.weight"])

    hidden = noise_embedding.float()
    for layer_idx in range(TINY["num_hidden_layers"]):
        prefix = f"layers.{layer_idx}."
        normed = _rms(hidden, w[prefix + "input_layernorm.weight"])

        attention_coefficients = None
        if dflash2:
            attention_coefficients = _oracle_conv_coefficients(w, prefix + "attention_conv", normed)
            normed = _oracle_conv(w, prefix + "attention_conv", 0, normed, attention_coefficients)

        q = (normed @ w[prefix + "self_attn.q_proj.weight"].T).view(BLOCK, num_heads, head_dim)
        k_context = (context @ w[prefix + "self_attn.k_proj.weight"].T).view(
            CTX_LEN, num_kv_heads, head_dim
        )
        k_block = (normed @ w[prefix + "self_attn.k_proj.weight"].T).view(
            BLOCK, num_kv_heads, head_dim
        )
        v_context = (context @ w[prefix + "self_attn.v_proj.weight"].T).view(
            CTX_LEN, num_kv_heads, head_dim
        )
        v_block = (normed @ w[prefix + "self_attn.v_proj.weight"].T).view(
            BLOCK, num_kv_heads, head_dim
        )
        k = torch.cat([k_context, k_block], dim=0)
        v = torch.cat([v_context, v_block], dim=0)
        q = _rope(_rms(q, w[prefix + "self_attn.q_norm.weight"]), query_positions)
        k = _rope(_rms(k, w[prefix + "self_attn.k_norm.weight"]), all_positions)
        repeats = num_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim**0.5
        # is_causal=false with a sliding window: symmetric.
        distance = (query_positions.unsqueeze(1) - all_positions.unsqueeze(0)).abs()
        scores = scores.masked_fill(distance.unsqueeze(0) > window - 1, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attention, v).reshape(BLOCK, num_heads * head_dim)
        delta = out @ w[prefix + "self_attn.o_proj.weight"].T
        if dflash2:
            delta = _oracle_conv(w, prefix + "attention_conv", 1, delta, attention_coefficients)
        hidden = hidden + delta

        normed = _rms(hidden, w[prefix + "post_attention_layernorm.weight"])
        if dflash2:
            mlp_coefficients = _oracle_conv_coefficients(w, prefix + "mlp_conv", normed)
            normed = _oracle_conv(w, prefix + "mlp_conv", 0, normed, mlp_coefficients)
        gate = normed @ w[prefix + "mlp.gate_proj.weight"].T
        up = normed @ w[prefix + "mlp.up_proj.weight"].T
        delta = (F.silu(gate) * up) @ w[prefix + "mlp.down_proj.weight"].T
        if dflash2:
            delta = _oracle_conv(w, prefix + "mlp_conv", 1, delta, mlp_coefficients)
        hidden = hidden + delta

    return _rms(hidden, w["norm.weight"])


def _run_block_decode(drafter, captured, noise_embedding):
    projected = drafter.project_target_hidden(captured.to("cuda", torch.bfloat16))
    k, v = drafter.precompute_context_kv(projected, torch.arange(CTX_LEN, device="cuda"))
    num_layers = drafter._num_attn_layers
    pool_k = torch.zeros(
        1,
        num_layers,
        CTX_LEN + BLOCK,
        drafter._num_kv_heads,
        drafter._head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pool_v = torch.zeros_like(pool_k)
    pool_k[0, :, :CTX_LEN] = k.permute(1, 0, 2, 3)
    pool_v[0, :, :CTX_LEN] = v.permute(1, 0, 2, 3)
    out = drafter.dflash_forward(
        noise_embedding=noise_embedding.to("cuda", torch.bfloat16).unsqueeze(0),
        query_positions=torch.arange(CTX_LEN, CTX_LEN + BLOCK, device="cuda").unsqueeze(0),
        num_ctx_per_req=torch.tensor([CTX_LEN], device="cuda"),
        ctx_k_cache=pool_k,
        ctx_v_cache=pool_v,
        ctx_cache_batch_idx=torch.tensor([0], device="cuda"),
    )
    return out.float().cpu()


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_flash_attn(),
    reason="tiny DFlash 2 block-decode parity needs CUDA + flash_attn",
)


@needs_gpu
def test_dflash2_drafter_parses_config_and_loads_checkpoint_weights():
    """Weight names and shapes match the released
    incoai/Qwen3.8-27B-DFlash2 checkpoint."""
    weights = _tiny_weights(True)
    drafter = _build_drafter(True, weights)

    assert drafter._is_dflash2
    assert drafter._dflash2_conv_taps == CONV_TAPS
    assert drafter._dflash2_conv_group_size == CONV_GROUP_SIZE
    assert drafter._dflash2_selector_rank == RANK
    assert drafter._dflash2_selector_top_k == TOP_K
    assert drafter.has_block_conv and drafter.has_candidate_selector
    assert len(drafter.attention_convs) == TINY["num_hidden_layers"]
    assert len(drafter.mlp_convs) == TINY["num_hidden_layers"]

    torch.testing.assert_close(
        drafter.attention_convs[0].base_kernel.cpu(), weights["layers.0.attention_conv.base_kernel"]
    )
    torch.testing.assert_close(
        drafter.mlp_convs[1].kernel_projection.weight.cpu(),
        weights["layers.1.mlp_conv.kernel_projection.weight"],
    )
    torch.testing.assert_close(
        drafter.candidate_selector.successor_codebook.cpu(),
        weights["candidate_selector.successor_codebook"],
    )
    assert drafter.candidate_selector.top_k == TOP_K


@needs_gpu
def test_dflash2_config_without_the_weights_is_rejected():
    """A checkpoint declaring DFlash 2 but shipping no conv weights would
    otherwise load as plain DFlash."""
    weights = {
        k: v
        for k, v in _tiny_weights(True).items()
        if "_conv." not in k and not k.startswith("candidate_selector.")
    }
    with pytest.raises(ValueError, match="missing checkpoint weight"):
        _build_drafter(True, weights)


@needs_gpu
def test_plain_dflash_drafter_keeps_the_old_path():
    """No-regression: a checkpoint without the DFlash 2 fields resolves to
    plain-DFlash behavior."""
    drafter = _build_drafter(False, _tiny_weights(False))
    assert not drafter._is_dflash2
    assert not drafter.has_block_conv and not drafter.has_candidate_selector
    assert drafter.attention_convs is None and drafter.mlp_convs is None
    # is_causal=false is still in the config but only DFlash 2 honors it.
    assert drafter._get_attention_mask_args(0) == (True, (TINY["sliding_window"] - 1, 0))


@needs_gpu
def test_dflash2_block_decode_matches_reference_oracle():
    """The block decode matches the fp32 eager oracle; the conv-free oracle
    serves as a negative control."""
    weights = _tiny_weights(True)
    drafter = _build_drafter(True, weights)

    generator = torch.Generator().manual_seed(42)
    captured = torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=generator) * 0.5
    noise_embedding = torch.randn(BLOCK, TINY["hidden_size"], generator=generator) * 0.5

    out = _run_block_decode(drafter, captured, noise_embedding)

    # The oracle consumes the same bf16-quantized inputs the drafter sees.
    captured_bf16 = captured.to(torch.bfloat16)
    noise_bf16 = noise_embedding.to(torch.bfloat16)
    with_conv = _oracle_block_decode(weights, captured_bf16, noise_bf16, True)
    without_conv = _oracle_block_decode(weights, captured_bf16, noise_bf16, False)

    conv_diff = (out - with_conv).abs().max().item()
    no_conv_diff = (out - without_conv).abs().max().item()
    assert conv_diff < 0.02, f"DFlash 2 parity failed: max abs diff {conv_diff}"
    assert no_conv_diff > 4 * max(conv_diff, 1e-4), (
        f"negative control failed: the conv-free oracle is too close "
        f"({no_conv_diff} vs {conv_diff}) -- the convolutions may not be applied"
    )
