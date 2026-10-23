# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Helix's phase-dependent head split with complete Kimi K3 MLA outputs.

These tests deliberately replicate the KV history to isolate projection and
head-layout correctness. Owner-only persistence and distributed cached-context
loading need separate tests; this fixture does not validate their handoff.
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
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, scenario.fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(),
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
