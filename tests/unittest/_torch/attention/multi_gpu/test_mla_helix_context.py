# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Kimi K3 MLA head layouts, KV ownership and prefill/decode handoff.

Head-only cases replicate KV history to isolate projection/layout correctness.
Storage cases use Helix pages, check exact ownership and unchanged storage,
and decode directly from prefill. Historical attention gathering is separate.
"""

import pickle
import sys
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import (
    CACHE_TYPE_SELFKONLY,
    run_helix_test,
    setup_kv_and_metadata,
)
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modeling.test_kimi_k3_mla import _assert_matches_reference, _checkpoint_weights
from mpi4py import MPI
from utils.util import skip_pre_blackwell

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

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


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
    # A complete local cache on every rank intentionally isolates item 2 from
    # owner-only writes (item 3) and remote cached-KV gathering (item 4).
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
        # this item 2 fixture; ownership is tested separately in item 3.
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
    if scenario.compact_cache:
        # Amortize the VMM slab even for one-byte FP8 cache elements.
        lengths *= 4
    batch, max_length = len(lengths), max(lengths) + 4
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
    if scenario.compact_cache:
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
                cache_reuse=False, chunked_prefill=False
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
        metadata = metadata_for(lengths, [0] * batch, batch)
        context = torch.cat([hidden[i, :length] for i, length in enumerate(lengths)])
        with model_extra_attrs(attrs):
            actual = runtime(context, metadata)
        target = torch.cat([expected[i, :length] for i, length in enumerate(lengths)])
        _assert_matches_reference(actual, target, scenario.fp8, scenario.fp8_cache)
        history = list(captured[0].split(lengths))
        check_cache(metadata, history)
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
        if mapping.cp_size > 1:
            saved_cache = cache.clone()
            with pytest.raises(NotImplementedError, match="distributed KV gathering"):
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
        expected_q = q.clone().view(-1, mixer.num_context_heads, mixer.qk_head_dim)

        def quarter_turn(values):
            pairs = values.unflatten(-1, (-1, 2))
            return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

        expected_q[..., mixer.qk_nope_head_dim :] = quarter_turn(
            expected_q[..., mixer.qk_nope_head_dim :]
        )
        expected_latent = latent.clone()
        expected_latent[:, mixer.kv_lora_rank :] = quarter_turn(latent[:, mixer.kv_lora_rank :])
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
        table = mixer.mha.rotary_cos_sin
        saved_table = table.clone()
        try:
            table.view(-1, 2)[:, 0] = 0
            table.view(-1, 2)[:, 1] = 1
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-weights", "fp8-weights"])
@pytest.mark.parametrize("num_heads,gate", [(32, True), (6, False), (6, True)])
def test_k3_mla_fp8_cache_reference(fp8, num_heads, gate):
    # Keep an unsharded numerical/control baseline for the FP8 KV cache.
    run_helix_test(
        _check_context_heads,
        _Scenario(
            cp_size=1,
            num_heads=num_heads,
            gate=gate,
            fp8=fp8,
            fp8_cache=True,
            check_kv_ownership=True,
        ),
        "nccl",
        world_size=1,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
def test_k3_mla_helix_compact_cache(fp8_cache):
    run_helix_test(
        _check_context_heads,
        _Scenario(
            cp_size=4, fp8=True, fp8_cache=fp8_cache, check_kv_ownership=True, compact_cache=True
        ),
        "nccl",
        world_size=4,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-weights", "fp8-weights"])
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
@pytest.mark.parametrize(
    "tp_size,cp_size,num_heads",
    [
        (1, 2, 32),
        pytest.param(
            2,
            2,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        pytest.param(
            1,
            4,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        (1, 2, 6),
    ],
    ids=["tp1-cp2", "tp2-cp2", "tp1-cp4", "tp1-cp2-unaligned"],
)
def test_k3_mla_helix_owned_prefill(fp8, fp8_cache, gate, tp_size, cp_size, num_heads):
    run_helix_test(
        _check_context_heads,
        _Scenario(
            tp_size=tp_size,
            cp_size=cp_size,
            num_heads=num_heads,
            fp8=fp8,
            gate=gate,
            check_kv_ownership=True,
            fp8_cache=fp8_cache,
        ),
        "nccl",
        world_size=tp_size * cp_size,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "fp8,gate,tp_size,cp_size,num_heads",
    [
        pytest.param(False, True, 1, 1, 32, id="bf16-control-tp1-cp1"),
        pytest.param(False, False, 1, 2, 32, id="bf16-no-gate-tp1-cp2"),
        pytest.param(True, True, 1, 2, 32, id="fp8-gate-tp1-cp2"),
        pytest.param(
            True,
            True,
            2,
            2,
            32,
            id="fp8-gate-tp2-cp2",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        pytest.param(True, False, 1, 2, 6, id="fp8-no-gate-tp1-cp2-unaligned-q-scale-block"),
    ],
)
def test_k3_mla_helix_context_heads(fp8, gate, tp_size, cp_size, num_heads):
    scenario = _Scenario(tp_size=tp_size, cp_size=cp_size, num_heads=num_heads, fp8=fp8, gate=gate)
    run_helix_test(
        _check_context_heads,
        scenario,
        "nccl",
        world_size=scenario.tp_size * scenario.cp_size,
        max_mismatch_ratio=0.0,
    )
