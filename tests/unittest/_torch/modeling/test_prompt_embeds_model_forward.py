# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real forwards of tiny models whose layers read token ids after the embedding.

A prompt-embedding row goes through the same overlay as in the engine
(``embed_prompt_tokens`` + ``prompt_embeds_model_inputs``); plain requests in
the same batch must be unaffected.
"""

from copy import deepcopy
from typing import Optional

import test_modeling_gemma4 as gemma4_tests
import torch

from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams


def _gemma4_context_logits(
    model: torch.nn.Module,
    config: object,
    input_ids: torch.Tensor,
    seq_lens: list[int],
    embed_rows: Optional[torch.Tensor] = None,
    ple_input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one context pass; overlay ``embed_rows`` as the engine does."""
    kv_cache_manager = gemma4_tests._build_gemma4_kv_cache_manager(
        config, num_blocks=2, batch_size=len(seq_lens)
    )
    request_ids = list(range(1, len(seq_lens) + 1))
    kv_cache_manager.add_dummy_requests(request_ids, seq_lens)
    positions = torch.cat([torch.arange(n, dtype=torch.int32) for n in seq_lens]).cuda()
    attn_metadata = get_attention_backend("FLASHINFER").Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=len(seq_lens),
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[0] * len(seq_lens)
        ),
        max_num_requests=len(seq_lens),
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=seq_lens,
    )
    with torch.inference_mode():
        attn_metadata.prepare()
        try:
            return model.forward(
                position_ids=positions.unsqueeze(0),
                attn_metadata=attn_metadata,
                **_forward_inputs(model, input_ids, embed_rows, ple_input_ids),
            ).float()
        finally:
            kv_cache_manager.shutdown()


def _forward_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    embed_rows: Optional[torch.Tensor],
    ple_input_ids: Optional[torch.Tensor],
) -> dict[str, Optional[torch.Tensor]]:
    """Return forward kwargs, overlaying ``embed_rows`` as the engine does."""
    if embed_rows is not None:
        embeds = model.embed_prompt_tokens(input_ids)
        # Rows equal to the token embedding: only the side channel can differ.
        embeds.index_copy_(0, embed_rows, model.model.embed_tokens(input_ids[embed_rows]))
        return {
            "input_ids": None,
            "inputs_embeds": embeds,
            **model.prompt_embeds_model_inputs(input_ids, embed_rows),
        }
    kwargs = {"input_ids": input_ids}
    if ple_input_ids is not None:
        kwargs["ple_input_ids"] = ple_input_ids
    return kwargs


@torch.no_grad()
def test_gemma4_ple_mixed_batch() -> None:
    torch.random.manual_seed(0)
    _, model, config = gemma4_tests.TestGemma4HFComparison()._make_hf_and_trt_models(
        deepcopy(gemma4_tests.GEMMA4_PLE_CONFIG)
    )
    assert model.model.hidden_size_per_layer_input
    plain = torch.tensor([100, 200, 300, 400, 500, 600], dtype=torch.int32)
    embedded = torch.tensor([110, 210, 310, 410, 510, 610, 710], dtype=torch.int32)
    ids = torch.cat([plain, embedded]).cuda()
    seq_lens = [len(plain), len(embedded)]
    rows = torch.tensor([7, 9, 12], dtype=torch.int64).cuda()

    reference_ids = ids.clone()
    reference_ids[rows] = config.pad_token_id
    expected = _gemma4_context_logits(model, config, ids, seq_lens, ple_input_ids=reference_ids)
    got = _gemma4_context_logits(model, config, ids, seq_lens, embed_rows=rows)
    plain_only = _gemma4_context_logits(model, config, ids, seq_lens)

    # Embedding rows see pad in the per-layer embedding, as image rows do.
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
    # The plain request batched alongside keeps its own per-layer embeddings.
    torch.testing.assert_close(got[0], plain_only[0], atol=1e-2, rtol=1e-2)
    # Dropping the per-layer embedding would change the embedded request.
    assert not torch.allclose(got[1], plain_only[1], atol=1e-2, rtol=1e-2)
