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
"""Check Kimi K3 MLA head layouts, KV ownership and prefill/decode handoff.

Head-only cases replicate KV history to isolate projection/layout correctness.
Storage cases use Helix pages, check exact ownership and unchanged storage,
and decode directly from prefill, including multiple context chunks.
"""

import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modeling.test_kimi_k3_mla import _assert_matches_reference, _checkpoint_weights
from mpi4py import MPI

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@dataclass(frozen=True)
class _Scenario:
    tp_size: int = 1
    cp_size: int = 2
    num_heads: int = 32
    fp8: bool = False
    gate: bool = True
    check_kv_ownership: bool = False
    fp8_cache: bool = False
    compact_cache: bool = False
    chunked_history: bool = False


@torch.no_grad()
def _check_context_heads(rank, world_size, scenario, use_nccl_for_alltoall, fifo_version):
    torch.manual_seed(42)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=scenario.tp_size,
        cp_size=scenario.cp_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=scenario.gate,
        max_position_embeddings=2048 if scenario.check_kv_ownership else 512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, scenario.fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8 if scenario.fp8_cache else None),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        }
        if scenario.fp8
        else None,
        allreduce_strategy=AllReduceStrategy.NCCL,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    mixer = runtime.mixer
    expected_heads = config.num_attention_heads // world_size
    assert mixer.mha.num_heads == expected_heads, (
        f"context backend has {mixer.mha.num_heads} heads, expected {expected_heads}"
    )
    assert mixer.mqa.num_heads == config.num_attention_heads // scenario.tp_size

    reference.cuda()
    if scenario.check_kv_ownership:
        return _check_owner_only_kv(runtime, mixer, reference, model_config, mapping, scenario)
    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    mask = torch.full((context_length + 1, context_length + 1), float("-inf"), device="cuda").triu(
        1
    )[None, None]
    expected = reference(hidden, attention_mask=mask)
    cache_scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    # Complete local caches isolate head-layout checks from distributed KV
    # ownership and gathering.
    manager, metadata = setup_kv_and_metadata(
        cache_scenario,
        Mapping(world_size=1, rank=0, tp_size=1),
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    full_q_projection_rows = []
    handle = mixer.q_b_proj.register_forward_hook(
        lambda module, inputs, output: full_q_projection_rows.append(output.shape[0])
    )
    extra_attrs = {
        "attention_metadata": weakref.ref(metadata),
        "mla_layers": {mixer.layer_idx_str: weakref.ref(mixer)},
    }
    try:
        context = hidden[:, :context_length].reshape(-1, config.hidden_size).contiguous()
        with model_extra_attrs(extra_attrs):
            actual = runtime(context, metadata).view(batch, context_length, -1)
        _assert_matches_reference(actual, expected[:, :context_length], scenario.fp8)
        if scenario.cp_size > 1:
            assert full_q_projection_rows == [], (
                "prefill must not run the full TP-local Q projection"
            )

        # Revisit the second half using a complete local prefix. This checks
        # head layouts in both cached-context implementations without needing
        # distributed KV gathering. Rewriting the same tokens is intentional.
        half = context_length // 2
        for chunked in (False, True):
            print(f"Checking cached context: CP={scenario.cp_size}, chunked={chunked}", flush=True)
            metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([half] * batch, dtype=torch.int),
                request_ids=[0, 1],
                max_num_requests=batch,
                num_contexts=batch,
                # Context runtime lengths describe this query chunk. The
                # cached lengths separately account for its 64-token prefix.
                prompt_lens=[half] * batch,
                max_num_tokens=half * batch,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[half] * batch
                ),
                runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=chunked,
                    chunk_size=half,
                    chunked_prefill_buffer_batch_size=1,
                ),
                enable_context_mla_with_cached_kv=True,
            )
            metadata.prepare()
            extra_attrs["attention_metadata"] = weakref.ref(metadata)
            suffix = hidden[:, half:context_length].reshape(-1, config.hidden_size).contiguous()
            with model_extra_attrs(extra_attrs):
                actual = runtime(suffix, metadata).view(batch, half, -1)
            _assert_matches_reference(actual, expected[:, half:context_length], scenario.fp8)
        if scenario.cp_size > 1:
            assert full_q_projection_rows == [], "cached prefill must retain the CP head split"

        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([context_length, 1], dtype=torch.int),
            request_ids=[0, 1],
            max_num_requests=batch,
            num_contexts=1,
            prompt_lens=[context_length, context_length],
            max_num_tokens=context_length + 1,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0, context_length]
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        # The generation kernel emits CP softmax statistics only when its
        # Helix fields are present. Every rank appends to its full replica in
        # this fixture; owner-only writes are checked separately.
        metadata.update_helix_param([context_length], [False])
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        mixed = torch.cat((hidden[0, :context_length], hidden[1, -1:]), dim=0)
        with model_extra_attrs(extra_attrs):
            actual = runtime(mixed, metadata)
        _assert_matches_reference(
            actual[:context_length], expected[0, :context_length], scenario.fp8
        )
        _assert_matches_reference(actual[-1], expected[1, -1], scenario.fp8)
        if scenario.cp_size > 1:
            assert full_q_projection_rows == [1], (
                "only decode tokens use the full TP-local Q projection"
            )
        return 0.0
    finally:
        handle.remove()
        manager.shutdown()


def _check_owner_only_kv(runtime, mixer, reference, model_config, mapping, scenario):
    config = model_config.pretrained_config
    # Default cases overallocate so the old writer can fail safely. Compact
    # cases reserve only the ledger pages needed by the global maximum.
    lengths = [31, 64, 65, 127, 128, 129, 191, 193]
    if scenario.chunked_history:
        lengths = [31, 64, 65, 127, 128, 257, 383, 577]
    if scenario.compact_cache:
        # Amortize the VMM slab even for one-byte FP8 cache elements.
        lengths *= 4
    batch = len(lengths)
    max_length = max(lengths) + (5 if scenario.chunked_history else 4)
    generator = torch.Generator(device="cuda").manual_seed(5678)
    hidden = torch.randn(
        batch,
        max_length,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    mask = torch.full((max_length, max_length), float("-inf"), device="cuda").triu(1)[None, None]
    expected = reference(hidden, attention_mask=mask)
    cache_context = max_length
    if scenario.compact_cache or scenario.chunked_history:
        # The helper reserves (context_per_rank + 1) * CP global tokens.
        # Reserve only the physical pages needed by the global maximum.
        cache_context = (max_length + mapping.cp_size) // mapping.cp_size - 1
    manager, _ = setup_kv_and_metadata(
        SimpleNamespace(
            ctx_len=cache_context * mapping.world_size,
            batch=batch,
            kv_cache_tokens_per_block=64,
            num_layers=1,
            kv_cache_dtype=torch.float8_e4m3fn if scenario.fp8_cache else torch.bfloat16,
        ),
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    cache = manager.get_buffers(0)
    if scenario.compact_cache:
        page = manager.tokens_per_block
        # VMM rounds the backing pool to a 2 MiB slab. Compare against
        # page-rounded unsharded storage and check assigned pages separately.
        unsharded_pages = batch * ((max_length + page) // page)
        assert cache.shape[0] < unsharded_pages
        assert all(manager.kv_cache_map[req].num_blocks == 1 for req in range(batch))
        print(
            f"CP={mapping.cp_size} compact KV bytes per rank: "
            f"{cache.numel() * cache.element_size()}",
            flush=True,
        )
    cache.fill_(-2)
    captured = []
    generated = []
    original_context = mixer.forward_context
    original_generation = mixer.mqa.mla_rope_generation
    prefill_lengths = list(lengths)

    def remember_context(*args, **kwargs):
        latent_cache = kwargs["latent_cache"] if "latent_cache" in kwargs else args[6]
        captured.append(latent_cache.clone())
        return original_context(*args, **kwargs)

    mixer.forward_context = remember_context

    def remember_generation(*args, **kwargs):
        generated.append(args[2].clone())
        return original_generation(*args, **kwargs)

    mixer.mqa.mla_rope_generation = remember_generation
    attrs = {"mla_layers": {mixer.layer_idx_str: weakref.ref(mixer)}}
    page = manager.tokens_per_block

    def metadata_for(
        query_lengths,
        cached,
        num_contexts,
        global_positions=None,
        *,
        enable_context_mla_with_cached_kv=None,
    ):
        metadata = build_attention_metadata(
            model_config,
            max_batch_size=batch,
            max_num_tokens=sum(query_lengths),
            max_beam_width=1,
            attention_backend=get_attention_backend("TRTLLM"),
            attention_runtime_features=AttentionRuntimeFeatures(
                cache_reuse=False,
                chunked_prefill=scenario.chunked_history,
                chunk_size=64,
                chunked_prefill_buffer_batch_size=2,
            ),
            kv_cache_manager=manager,
            mapping=mapping,
            cache_indirection=None,
            enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
        )
        metadata.seq_lens = torch.tensor(query_lengths, dtype=torch.int)
        metadata.request_ids = list(range(batch))
        metadata.num_contexts = num_contexts
        metadata.prompt_lens = [
            query_lengths[i] if i < num_contexts else lengths[i] for i in range(batch)
        ]
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached)
        if global_positions is not None:
            metadata.update_helix_param(
                global_positions,
                [(g // page) % mapping.cp_size != mapping.cp_rank for g in global_positions],
            )
        else:
            # Initialize the legacy context-position view as well so the
            # unchanged writer is safe to run as a negative control.
            metadata.update_helix_param(
                [g for length in query_lengths for g in range(length)], [False] * batch
            )
        metadata.prepare()
        attrs["attention_metadata"] = weakref.ref(metadata)
        return metadata

    def owned_positions(length):
        return [g for g in range(length) if (g // page) % mapping.cp_size == mapping.cp_rank]

    def check_cache(metadata, history):
        occupied = torch.zeros(cache.shape[0], page, dtype=torch.bool, device="cuda")
        for req, values in enumerate(history):
            positions = owned_positions(values.shape[0])
            if not positions:
                continue
            local = torch.arange(len(positions), device="cuda")
            blocks = metadata.kv_cache_block_offsets[0, req, 0][local // page].long()
            actual = cache[blocks, 0, local % page, 0]
            target = values[positions].to(cache.dtype)
            torch.testing.assert_close(actual.float(), target.float(), atol=0, rtol=0)
            occupied[blocks, local % page] = True
        unused = cache[:, 0, :, 0][~occupied]
        assert torch.equal(
            unused.view(torch.uint8), torch.full_like(unused, -2).view(torch.uint8)
        ), f"CP rank {mapping.cp_rank} wrote a non-owned KV slot"

    try:
        history = [
            hidden.new_empty((0, config.kv_lora_rank + config.qk_rope_head_dim)) for _ in lengths
        ]
        prefixes = [0] * batch
        rounds = 3 if scenario.chunked_history else 1
        for part in range(rounds):
            query_lengths = [
                length - prefix if part == rounds - 1 else length // rounds
                for length, prefix in zip(lengths, prefixes)
            ]
            metadata = metadata_for(query_lengths, prefixes, batch)
            if scenario.chunked_history and part > 0:
                assert metadata.chunked_loop_num > 1
                plans = MPI.COMM_WORLD.allgather(metadata.chunked_loop_num)
                assert plans == [metadata.chunked_loop_num] * mapping.world_size
            context = torch.cat(
                [
                    hidden[i, prefix : prefix + count]
                    for i, (prefix, count) in enumerate(zip(prefixes, query_lengths))
                ]
            )
            original_lengths = metadata.kv_lens_runtime.clone()
            original_total = metadata.host_total_kv_lens.clone()
            with model_extra_attrs(attrs):
                actual = runtime(context, metadata)
            torch.testing.assert_close(metadata.kv_lens_runtime, original_lengths)
            torch.testing.assert_close(metadata.host_total_kv_lens, original_total)
            target = torch.cat(
                [
                    expected[i, prefix : prefix + count]
                    for i, (prefix, count) in enumerate(zip(prefixes, query_lengths))
                ]
            )
            _assert_matches_reference(actual, target, scenario.fp8, scenario.fp8_cache)
            history = [
                torch.cat((old, new))
                for old, new in zip(history, captured[-1].split(query_lengths))
            ]
            check_cache(metadata, history)
            prefixes = [prefix + count for prefix, count in zip(prefixes, query_lengths)]
        if scenario.chunked_history:
            # Cached context and generation share one batch and their metadata
            # must retain global context lengths and local decode lengths.
            context_query = 3
            metadata = metadata_for(
                [context_query] + [1] * (batch - 1),
                [lengths[0]] + [len(owned_positions(g)) for g in lengths[1:]],
                1,
                lengths[1:],
            )
            context_start = lengths[0]
            mixed = torch.cat(
                (
                    hidden[0, context_start : context_start + context_query],
                    torch.stack([hidden[i, g] for i, g in enumerate(lengths) if i > 0]),
                )
            )
            original_total = metadata.host_total_kv_lens.clone()
            with model_extra_attrs(attrs):
                actual = runtime(mixed, metadata)
            torch.testing.assert_close(metadata.host_total_kv_lens, original_total)
            target = torch.cat(
                (
                    expected[0, context_start : context_start + context_query],
                    torch.stack([expected[i, g] for i, g in enumerate(lengths) if i > 0]),
                )
            )
            _assert_matches_reference(actual, target, scenario.fp8, scenario.fp8_cache)
            history[0] = torch.cat((history[0], captured[-1]))
            for req in range(1, batch):
                history[req] = torch.cat((history[req], generated[-1][req - 1 : req]))
            check_cache(metadata, history)
            lengths = [lengths[0] + context_query] + [g + 1 for g in lengths[1:]]

        # No cache injection or repair: these are the pages written by prefill.
        metadata = metadata_for(
            [lengths[0]] + [1] * (batch - 1),
            [0] + [len(owned_positions(g)) for g in lengths[1:]],
            1,
            lengths[1:],
        )
        mixed = torch.cat(
            (
                hidden[0, : lengths[0]],
                torch.stack([hidden[i, g] for i, g in enumerate(lengths) if i > 0]),
            )
        )
        with model_extra_attrs(attrs):
            actual = runtime(mixed, metadata)
        target = torch.cat(
            (
                expected[0, : lengths[0]],
                torch.stack([expected[i, g] for i, g in enumerate(lengths) if i > 0]),
            )
        )
        _assert_matches_reference(actual, target, scenario.fp8, scenario.fp8_cache)
        history[0] = captured[-1]
        for req in range(1, batch):
            history[req] = torch.cat((history[req], generated[-1][req - 1 : req]))
        check_cache(metadata, history)
        lengths = [lengths[0]] + [g + 1 for g in lengths[1:]]
        for _ in range(3):
            metadata = metadata_for(
                [1] * batch, [len(owned_positions(g)) for g in lengths], 0, lengths
            )
            inputs = torch.stack([hidden[i, g] for i, g in enumerate(lengths)])
            with model_extra_attrs(attrs):
                actual = runtime(inputs, metadata)
            target = torch.stack([expected[i, g] for i, g in enumerate(lengths)])
            _assert_matches_reference(actual, target, scenario.fp8, scenario.fp8_cache)
            history = [
                torch.cat((values, generated[-1][req : req + 1]))
                for req, values in enumerate(history)
            ]
            check_cache(metadata, history)
            lengths = [g + 1 for g in lengths]

        # Exercise the context writer with nonzero global prefixes and real
        # RoPE, independently of historical-attention gathering.
        # This happens after the direct prefill/decode handoff checks above.
        # The unsharded control also needs cached-context indptrs for this probe.
        chunk = 7
        metadata = metadata_for(
            [chunk] * batch,
            [g - chunk for g in prefill_lengths],
            batch,
            enable_context_mla_with_cached_kv=True,
        )
        latent = torch.cat(
            [values[g - chunk : g] for values, g in zip(history, prefill_lengths)]
        ).clone()
        q = torch.randn(
            batch * chunk,
            mixer.num_context_heads * mixer.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if mapping.cp_size > 1 and not scenario.chunked_history:
            saved_cache = cache.clone()
            with pytest.raises(NotImplementedError, match="requires chunked prefill"):
                original_context(
                    q=q,
                    compressed_kv=latent[:, : mixer.kv_lora_rank],
                    k_pe=latent[:, mixer.kv_lora_rank :],
                    position_ids=None,
                    attn_metadata=metadata,
                    output=torch.empty_like(q),
                    latent_cache=latent,
                )
            torch.testing.assert_close(cache.float(), saved_cache.float(), atol=0, rtol=0)
        table = mixer.mha.rotary_cos_sin
        saved_table = table.clone()
        positions = torch.arange(table.numel() // (2 * mixer.qk_rope_head_dim), device=table.device)
        pair_indices = torch.arange(mixer.qk_rope_head_dim // 2, device=table.device)
        # Include the page number: position % 4 aliases 64-token compaction.
        # Signed quarter turns keep BF16 rotations and FP8 cache checks exact.
        turns = (positions[:, None] // page + positions[:, None] % page + pair_indices) % 4
        cos_sin = torch.stack(
            (
                (turns == 0).float() - (turns == 2).float(),
                (turns == 1).float() - (turns == 3).float(),
            ),
            dim=-1,
        )
        global_positions = torch.tensor(
            [g for length in prefill_lengths for g in range(length - chunk, length)],
            device=table.device,
        )

        def rope_reference(values: torch.Tensor) -> torch.Tensor:
            # Use request-global positions and signed swaps, not native/table lookup.
            token_turns = (
                (global_positions // page + global_positions % page)[:, None] + pair_indices
            ) % 4
            if values.ndim == 3:
                token_turns = token_turns[:, None]
            even, odd = values.unflatten(-1, (-1, 2)).unbind(-1)
            rotated_even = torch.where(token_turns % 2 == 0, even, -odd)
            rotated_odd = torch.where(token_turns % 2 == 0, odd, even)
            return torch.stack(
                (
                    torch.where(token_turns < 2, rotated_even, -rotated_even),
                    torch.where(token_turns < 2, rotated_odd, -rotated_odd),
                ),
                dim=-1,
            ).flatten(-2)

        expected_q = q.clone().view(-1, mixer.num_context_heads, mixer.qk_head_dim)
        expected_q[..., mixer.qk_nope_head_dim :] = rope_reference(
            expected_q[..., mixer.qk_nope_head_dim :]
        )
        expected_latent = latent.clone()
        expected_latent[:, mixer.kv_lora_rank :] = rope_reference(latent[:, mixer.kv_lora_rank :])
        expected_cache = cache.clone()
        for req, length in enumerate(prefill_lengths):
            own = owned_positions(length)
            for g in range(length - chunk, length):
                if g not in own:
                    continue
                local = own.index(g)
                block = metadata.kv_cache_block_offsets[0, req, 0, local // page].item()
                expected_cache[block, 0, local % page, 0] = expected_latent[
                    req * chunk + g - (length - chunk)
                ].to(cache.dtype)
        try:
            # MLA stores two copies of the interleaved (cos, sin) pairs per position.
            table.view(-1, mixer.qk_rope_head_dim, 2).copy_(cos_sin.repeat(1, 2, 1))
            mixer.mha.mla_rope_append_paged_kv_assign_q(q, latent, metadata)
            torch.testing.assert_close(q.view_as(expected_q), expected_q, atol=0, rtol=0)
            torch.testing.assert_close(latent, expected_latent, atol=0, rtol=0)
            torch.testing.assert_close(cache.float(), expected_cache.float(), atol=0, rtol=0)
        finally:
            table.copy_(saved_table)
        return 0.0
    finally:
        mixer.forward_context = original_context
        mixer.mqa.mla_rope_generation = original_generation
        manager.shutdown()
