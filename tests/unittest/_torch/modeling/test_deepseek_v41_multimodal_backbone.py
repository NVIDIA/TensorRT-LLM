# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal backbone, Engram history, and image/text routing contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.modules.engram import engram
from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod

pytestmark = pytest.mark.cpu_only


class _RecordingLayer(nn.Module):
    def __init__(self, engram: SimpleNamespace | None = None) -> None:
        super().__init__()
        self.engram = engram
        self.record = Mock()

    def forward(self, hc_state: HCState, **kwargs: object) -> HCState:
        self.record(hc_state=hc_state, **kwargs)
        return hc_state


@pytest.fixture
def backbone() -> tuple[v41.DeepseekV41Model, SimpleNamespace, torch.Tensor, torch.Tensor]:
    model = v41.DeepseekV41Model.__new__(v41.DeepseekV41Model)
    nn.Module.__init__(model)
    model.model_config = SimpleNamespace(
        mapping=SimpleNamespace(has_pp=lambda: False, enable_attention_dp=False)
    )
    model.hc_mult = 4
    model.use_engram = True
    model.engram_layer_ids = [0]
    model._engram_projection_schedule = {}
    model.engram_hash_provider = SimpleNamespace(
        compute_hashes=Mock(return_value={0: torch.tensor([[11], [12], [13], [14]])})
    )
    engram = SimpleNamespace(precompute=Mock(return_value=torch.ones(4, 2)), sync_event=None)
    model.layers = nn.ModuleList([_RecordingLayer(engram), _RecordingLayer(), _RecordingLayer()])
    model.num_hidden_layers = len(model.layers)
    model.ced_kv_precompute = False
    model.disagg_context_only = False
    model.disagg_remote_tail_replay = False
    model._plan_bounded_replay = Mock(return_value=(None, None))
    embeddings = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
    model.embed_tokens = Mock(return_value=embeddings)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata = SimpleNamespace(
        begin_model_forward=Mock(),
        num_tokens=4,
        num_contexts=1,
        all_rank_num_tokens=[4],
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens,
        request_ids=[17],
        kv_cache_manager=SimpleNamespace(max_seq_len=128),
        csa2_precomputed_kv_layers=set(),
    )
    return model, metadata, torch.tensor([5, 101, 101, 7], dtype=torch.int32), embeddings


@pytest.mark.parametrize("fused_embeddings", [False, True])
def test_backbone_preserves_original_ids_and_image_embeddings(
    backbone: tuple[v41.DeepseekV41Model, SimpleNamespace, torch.Tensor, torch.Tensor],
    fused_embeddings: bool,
) -> None:
    model, metadata, token_ids, embeddings = backbone
    image_mask = torch.tensor([False, True, True, False]) if fused_embeddings else None
    output = model(
        attn_metadata=metadata,
        input_ids=None if fused_embeddings else token_ids,
        inputs_embeds=embeddings if fused_embeddings else None,
        orig_input_ids=token_ids if fused_embeddings else token_ids + 1,
        position_ids=torch.arange(4).unsqueeze(0),
        image_mask=image_mask,
    )
    if fused_embeddings:
        model.embed_tokens.assert_not_called()
        torch.testing.assert_close(
            model.engram_hash_provider.compute_hashes.call_args.kwargs["token_mask"], ~image_mask
        )
    else:
        model.embed_tokens.assert_called_once_with(token_ids)
    torch.testing.assert_close(
        model.engram_hash_provider.compute_hashes.call_args.args[0], token_ids
    )
    precomputed = model.layers[0].engram.precompute.return_value
    for index, layer in enumerate(model.layers):
        args = layer.record.call_args.kwargs
        assert args["input_ids"] is token_ids
        assert args.get("image_mask") is image_mask
        assert args["engram_embeddings"] is (precomputed if index == 0 else None)
        torch.testing.assert_close(args["hc_state"].residual, embeddings[:, None].repeat(1, 4, 1))
    torch.testing.assert_close(output, embeddings, rtol=0, atol=0)


def test_replay_gathers_image_masks_ids_and_embedding_rows(
    backbone: tuple[v41.DeepseekV41Model, SimpleNamespace, torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, metadata, token_ids, embeddings = backbone
    positions = torch.arange(4).unsqueeze(0)
    rows = torch.tensor([2, 3])
    plan = SimpleNamespace(
        rows=rows,
        num_encoder_tokens=4,
        swa_floors=[0],
        updates_local_metadata=True,
        replays_local_tokens=True,
        replay_seq_lens=torch.tensor([2]),
    )
    model._plan_bounded_replay.return_value = (1, plan)
    enter, exit_replay = Mock(), Mock()
    monkeypatch.setattr(v41, "enter_decoder_replay", enter)
    monkeypatch.setattr(v41, "exit_decoder_replay", exit_replay)
    image_mask = torch.tensor([False, True, True, False])
    output = model(
        attn_metadata=metadata,
        inputs_embeds=embeddings,
        orig_input_ids=token_ids,
        position_ids=positions,
        all_token_states_required=False,
        image_mask=image_mask,
    )
    enter.assert_called_once_with(metadata, plan, None)
    exit_replay.assert_called_once_with(metadata, plan)
    assert model.layers[0].record.call_args.kwargs["image_mask"] is image_mask
    for layer in model.layers[1:]:
        args = layer.record.call_args.kwargs
        torch.testing.assert_close(args["input_ids"], token_ids[rows])
        torch.testing.assert_close(args["position_ids"], positions[:, rows])
        torch.testing.assert_close(args["image_mask"], image_mask[rows])
        torch.testing.assert_close(
            args["hc_state"].residual, embeddings[rows, None].repeat(1, 4, 1)
        )
    torch.testing.assert_close(output[rows], embeddings[rows], rtol=0, atol=0)
    torch.testing.assert_close(output[:2], torch.zeros_like(embeddings[:2]), rtol=0, atol=0)


_TOKEN_MAP = [3, 5, 2, 1, 1, 4, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14]


def _reference_hashes(
    provider: engram.EngramHashProvider,
    input_ids: list[int],
    token_mask: list[bool],
    seq_lens: list[int],
) -> dict[int, torch.Tensor]:
    """Hash text suffixes without crossing image or packed-sequence boundaries."""
    expected = {}
    for layer_id in provider.layer_ids:
        multipliers = provider._multipliers[layer_id].tolist()
        rows, start = [], 0
        for length in seq_lens:
            boundary = start - 1
            for position in range(start, start + length):
                if not token_mask[position]:
                    boundary = position
                window = [
                    _TOKEN_MAP[input_ids[position - shift]]
                    if position - shift > boundary
                    else _TOKEN_MAP[provider.config.pad_id]
                    for shift in range(provider.config.max_ngram_size)
                ]
                mixed, hashes = window[0] * multipliers[0], []
                for shift in range(1, provider.config.max_ngram_size):
                    mixed ^= window[shift] * multipliers[shift]
                    hashes.extend(
                        mixed % prime
                        for prime in provider.vocab_size_across_layers[layer_id][shift - 1]
                    )
                rows.append(hashes)
            start += length
        expected[layer_id] = torch.tensor(rows, dtype=torch.int64)
    return expected


def _assert_hashes(actual: dict[int, torch.Tensor], expected: dict[int, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for layer, hashes in actual.items():
        torch.testing.assert_close(hashes, expected[layer], rtol=0, atol=0)


def test_engram_image_boundaries_survive_request_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = engram.CompressedTokenizer(
        "unused", lookup_table=torch.tensor(_TOKEN_MAP), num_new_token=max(_TOKEN_MAP) + 1
    )
    monkeypatch.setattr(engram, "CompressedTokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    provider = engram.EngramHashProvider(
        engram.EngramConfig(
            tokenizer_name_or_path="unused",
            engram_vocab_size=[31, 31, 31],
            max_ngram_size=4,
            n_embed_per_ngram=8,
            n_head_per_ngram=2,
            layer_ids=[0, 2],
            pad_id=2,
        )
    )
    ids = [1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 1]
    mask = [True, True, False, True, True, False, True, True, True, False, True, True, True]
    expected = _reference_hashes(provider, ids, mask, [len(ids)])
    packed_ids, packed_mask = ids[:3] + [14, 15], mask[:3] + [True, True]
    actual = provider.compute_hashes(
        torch.tensor(packed_ids),
        seq_lens=torch.tensor([3, 2]),
        position_ids=torch.tensor([0, 1, 2, 0, 1]),
        request_ids=[19, 99],
        seq_lens_host=torch.tensor([3, 2]),
        max_seq_len=32,
        token_mask=torch.tensor(packed_mask),
    )
    _assert_hashes(actual, _reference_hashes(provider, packed_ids, packed_mask, [3, 2]))

    # A resumed request recycles the old rows and restores each context chunk's lookback.
    for start, stop in ((3, 5), (5, 7)):
        provider.seed_context_history(
            [17],
            {17: (start, ids[start - 3 : start])},
            max_seq_len=32,
            device=torch.device("cpu"),
            token_masks={17: mask[start - 3 : start]},
        )
        actual = provider.compute_hashes(
            torch.tensor(ids[start:stop]),
            position_ids=torch.arange(start, stop),
            request_ids=[17],
            seq_lens_host=torch.tensor([stop - start]),
            max_seq_len=32,
            token_mask=torch.tensor(mask[start:stop]),
        )
        _assert_hashes(actual, {layer: hashes[start:stop] for layer, hashes in expected.items()})

    # Disagg restores a bounded seed; refresh inserts an image, then decodes without a mask.
    provider.queue_history_seed(29, 4, ids[4:7], token_mask=mask[4:7])
    for start in (7, 9, 11):
        compute = provider.compute_hashes if start == 7 else provider.refresh_captured_hashes
        chunk_mask = torch.tensor(mask[start : start + 2])
        actual = compute(
            torch.tensor(ids[start : start + 2]),
            position_ids=torch.arange(start, start + 2),
            request_ids=[29],
            seq_lens_host=torch.tensor([2]),
            max_seq_len=32,
            token_mask=chunk_mask if not chunk_mask.all() else None,
        )
        _assert_hashes(
            actual, {layer: hashes[start : start + 2] for layer, hashes in expected.items()}
        )
        if start == 7:
            addresses = {layer: hashes.data_ptr() for layer, hashes in actual.items()}
        else:
            assert {layer: hashes.data_ptr() for layer, hashes in actual.items()} == addresses


def test_mixed_image_text_routing_uses_separate_biases_and_unbiased_weights() -> None:
    routing = DeepSeekV4MoeRoutingMethod(
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: torch.tensor([10.0, 9.0, 0.0, 0.0]),
        callable_tid2eid=lambda: None,
        is_hashed=False,
        callable_e_score_correction_bias_vl=lambda: torch.tensor([0.0, 0.0, 8.0, 7.0]),
    )
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(3, -1)
    indices, weights = routing.apply_with_aux(logits, None, torch.tensor([False, True, False]))
    expected_indices = torch.tensor([[0, 1], [2, 3], [0, 1]], dtype=torch.int32)
    torch.testing.assert_close(indices, expected_indices)
    scores = F.softplus(logits).sqrt().gather(1, expected_indices.long())
    torch.testing.assert_close(weights, scores / scores.sum(-1, keepdim=True) * 2.5)
    assert weights.dtype == torch.float32
