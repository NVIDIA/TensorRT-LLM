# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global producer parity with real mHC, indexer, attention and MoE kernels."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from _deepseek_v41_test_utils import cache_case, capture_layers, explicit_global_production
from _deepseek_v41_test_utils import cache_config as shared_cache_config
from _deepseek_v41_test_utils import model_and_config as shared_model_and_config

from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

cache_config = shared_cache_config
model_and_config = shared_model_and_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "case_spec",
    [
        ([640, 17], [0, 0], 2),
        ([129, 7, 1], [129, 260, 193], 2),
        ([1, 1], [641, 194], 0),
    ],
    ids=["cold", "mixed", "decode"],
)
@pytest.mark.parametrize("overlap", [False, True])
def test_explicit_global_production_matches_original(model_and_config, overlap, case_spec):
    model, config, sparse, dtype = model_and_config
    owner = model.model.layers[5]
    lengths, past, num_contexts = case_spec
    # Model/backend and manager share a lifetime, as in the executor. Each case
    # gets a fresh model; native runners may retain manager-specific state.
    with torch.inference_mode(), with_multi_stream(overlap):
        with cache_case(config, sparse, dtype, lengths, past, num_contexts) as case:
            metadata, ids, positions, buffers = case
            print(f"parity case lengths={lengths} past={past}", flush=True)
            if any(past):
                # Populate real prefix history. Zero index K would create
                # ties at the Top-K boundary and arbitrary selected sets.
                query_prompt_lens = metadata.prompt_lens
                metadata.prompt_lens = list(past)
                metadata.num_contexts = len(past)
                metadata.seq_lens = torch.tensor(past, dtype=torch.int32)
                metadata.kv_cache_params.num_cached_tokens_per_seq[:] = [0] * len(past)
                metadata.prepare()
                prefix_ids = torch.randint(
                    0,
                    config.pretrained_config.vocab_size,
                    (sum(past),),
                    device="cuda",
                    dtype=torch.int32,
                    generator=torch.Generator(device="cuda").manual_seed(1),
                )
                prefix_positions = torch.cat(
                    [torch.arange(p, device="cuda", dtype=torch.int32) for p in past]
                ).unsqueeze(0)
                model(
                    input_ids=prefix_ids,
                    position_ids=prefix_positions,
                    attn_metadata=metadata,
                    return_context_logits=True,
                )
                metadata.prompt_lens = query_prompt_lens
                metadata.num_contexts = num_contexts
                metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
                metadata.kv_cache_params.num_cached_tokens_per_seq[:] = past
                metadata.prepare()
            initial = [buffer.clone() for buffer in buffers]
            runs = []
            for split in (False, True):
                print(f"parity forward split={split}", flush=True)
                for buffer, old in zip(buffers, initial):
                    buffer.copy_(old)
                metadata.prepare()
                with (
                    explicit_global_production(owner, metadata) if split else nullcontext(),
                    capture_layers(model) as values,
                    patch.object(
                        owner.self_attn.compressor,
                        "forward",
                        wraps=owner.self_attn.compressor.forward,
                    ) as main_write,
                    patch.object(
                        owner.self_attn.index_wk,
                        "forward",
                        wraps=owner.self_attn.index_wk.forward,
                    ) as index_write,
                ):
                    logits = model(
                        input_ids=ids,
                        position_ids=positions,
                        attn_metadata=metadata,
                        return_context_logits=True,
                    )
                assert {"u20", "wkv", "index_wk", "q_swa", "layer5"} <= values.keys()
                assert main_write.call_count == index_write.call_count == 1
                assert main_write.call_args.args[0].shape[0] == sum(lengths)
                assert index_write.call_args.args[0].shape[0] == sum(lengths)
                torch.cuda.synchronize()
                assert torch.isfinite(logits).all()
                values["logits"] = (logits.clone(),)
                values.update(
                    {f"cache{i}": (v.clone().view(torch.uint8),) for i, v in enumerate(buffers)}
                )
                assert not metadata.csa2_precomputed_kv_layers
                runs.append(values)
            differences = []
            for run_id, run in enumerate(runs[1:], 1):
                assert runs[0].keys() == run.keys()
                for key in runs[0]:
                    for original, actual in zip(runs[0][key], run[key]):
                        if not torch.equal(actual, original):
                            delta = (actual.float() - original.float()).abs()
                            differences.append(
                                f"run={run_id} {key}: unequal={(actual != original).sum().item()} "
                                f"max={delta.max().item()} mean={delta.mean().item()}"
                            )
            assert not differences, f"case={lengths, past}: " + "; ".join(differences)
