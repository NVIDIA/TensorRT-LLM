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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import flashinfer_trtllm_gen as trtllm_gen_module
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.phased import PhasedFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import DEFAULT_FMHA_LIBS
from tensorrt_llm._torch.attention_backend.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import PositionEmbeddingType


def test_triton_custom_mask_precedes_general_fmha_libraries() -> None:
    assert DEFAULT_FMHA_LIBS == (
        "triton_custom_mask",
        "flashinfer_trtllm_gen",
        "fallback",
    )


def test_triton_custom_mask_implements_only_context_phase() -> None:
    assert TritonCustomMaskFmha.__bases__ == (PhasedFmha,)
    assert "run_generation" not in TritonCustomMaskFmha.__dict__
    assert "_run_preprocessed_context" not in TritonCustomMaskFmha.__dict__


def test_trtllm_metadata_does_not_add_custom_mask_buffers() -> None:
    fields = TrtllmAttentionMetadata.__dataclass_fields__
    assert "custom_mask_qo_indptr" not in fields
    assert "custom_mask_cached_token_lens" not in fields


def test_custom_mask_context_skips_trtllm_gen_context_checks() -> None:
    attn = SimpleNamespace(is_mla_enable=False, quant_mode=0)
    fmha = object.__new__(TritonCustomMaskFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=0,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((2, 12), dtype=torch.float16)
    output = torch.empty((2, 4), dtype=torch.float16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        is_fused_qkv=True,
    )

    assert fmha.is_context_supported(q, None, None, metadata, forward_args)


def test_custom_mask_mixed_batch_reuses_generation_checks(monkeypatch) -> None:
    attn = SimpleNamespace(is_mla_enable=False, quant_mode=0)
    fmha = object.__new__(TritonCustomMaskFmha)
    fmha._attn_ref = lambda: attn
    fmha._followup_fmhas = ()
    generation_fmha = object.__new__(FlashInferTrtllmGenFmha)
    generation_fmha._attn_ref = lambda: attn
    generation_fmha._followup_fmhas = ()
    fmha.set_followup_fmhas((generation_fmha,))
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((3, 12), dtype=torch.float16)
    output = torch.empty((3, 4), dtype=torch.float16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=True,
    )
    checked_args = None

    def _accept_generation_request(self, q, k, v, attn, metadata, forward_args):
        nonlocal checked_args
        checked_args = forward_args
        return True, ""

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "_is_supported_with_reason",
        _accept_generation_request,
    )
    assert fmha.is_supported(q, None, None, metadata, forward_args)
    assert checked_args.attention_mask == PredefinedAttentionMask.CAUSAL
    assert checked_args.attention_mask_data is None
    assert checked_args.attention_input_type == AttentionInputType.generation_only
    assert forward_args.attention_mask == CustomAttentionMask.CUSTOM


def test_mixed_batch_runs_context_and_generation_on_different_fmhas(monkeypatch) -> None:
    attn = SimpleNamespace(
        is_mla_enable=False,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        v_head_dim=None,
        kv_lora_rank=None,
        predicted_tokens_per_seq=1,
    )
    context_fmha = object.__new__(TritonCustomMaskFmha)
    context_fmha._attn_ref = lambda: attn
    context_fmha.kv_factor = 2
    context_fmha.context_out_head_size = 4
    context_fmha.generation_out_head_size = 4
    generation_fmha = object.__new__(FlashInferTrtllmGenFmha)
    generation_fmha._attn_ref = lambda: attn
    generation_fmha._followup_fmhas = ()
    context_fmha.set_followup_fmhas((generation_fmha,))

    monkeypatch.setattr(
        TritonCustomMaskFmha,
        "is_context_supported",
        lambda *args: True,
    )
    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_generation_supported",
        lambda *args: True,
    )
    monkeypatch.setattr(TritonCustomMaskFmha, "prepare_workspace", lambda *args: None)
    monkeypatch.setattr(FlashInferTrtllmGenFmha, "prepare_workspace", lambda *args: None)

    called_phases = []

    def _run_context(self, params):
        called_phases.append(("context", self, params.fwd))

    def _run_generation(self, params):
        called_phases.append(("generation", self, params.fwd))

    monkeypatch.setattr(TritonCustomMaskFmha, "run_context", _run_context)
    monkeypatch.setattr(FlashInferTrtllmGenFmha, "run_generation", _run_generation)

    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.int8),
        num_contexts=1,
        num_ctx_tokens=2,
        num_generations=1,
        kv_lens_cuda_runtime=torch.tensor([2, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 4], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 4], dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    q = torch.empty((3, 12), dtype=torch.float16)
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 4), dtype=torch.float16),
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones(4, dtype=torch.bool),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
        is_fused_qkv=True,
    )

    assert context_fmha.try_forward(q, None, None, metadata, forward_args)

    assert [(phase, fmha) for phase, fmha, _ in called_phases] == [
        ("context", context_fmha),
        ("generation", generation_fmha),
    ]
    assert called_phases[0][2].attention_mask == CustomAttentionMask.CUSTOM
    assert called_phases[1][2].attention_mask == PredefinedAttentionMask.CAUSAL
    assert called_phases[1][2].attention_mask_data is None


def test_large_head_generation_support_is_owned_by_trtllm_gen() -> None:
    attn = SimpleNamespace(
        head_dim=512,
        is_mla_enable=False,
        quant_mode=0,
        position_embedding_type=int(PositionEmbeddingType.learned_absolute),
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=1,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((1, 8), dtype=torch.bfloat16)
    k = torch.empty((1, 4), dtype=torch.bfloat16)
    v = torch.empty((1, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_input_type=AttentionInputType.generation_only,
        is_fused_qkv=False,
    )

    supported, reason = fmha._check_preprocessed_generation_with_reason(
        q,
        k,
        v,
        metadata,
        forward_args,
    )

    assert supported, reason


def test_preprocessed_generation_launches_trtllm_gen_directly(monkeypatch) -> None:
    attn = SimpleNamespace(
        num_heads=2,
        num_kv_heads=1,
        head_dim=512,
        quant_mode=0,
        local_layer_idx=0,
        q_scaling=1.0,
        attention_chunk_size=None,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        num_generations=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[4]),
        beam_width=1,
        max_num_requests=1,
        host_kv_cache_pool_pointers=torch.empty(0),
        host_kv_cache_pool_mapping=torch.empty(0),
        kv_cache_block_offsets=torch.empty(0),
    )
    params = SimpleNamespace(
        attn=attn,
        meta=metadata,
        fwd=AttentionForwardArgs(),
        qkv_input=torch.randn(1, 2 * 512, dtype=torch.bfloat16),
        key_input=torch.randn(1, 512, dtype=torch.bfloat16),
        value_input=torch.randn(1, 512, dtype=torch.bfloat16),
        context_buf=torch.empty(1, 2, 512, dtype=torch.bfloat16),
        sequence_lengths=torch.tensor([5], dtype=torch.int32),
        workspace=torch.empty(1024, dtype=torch.int8),
        num_tokens=1,
        seq_offset=0,
        num_requests=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=4,
        input_seq_length=1,
        max_past_kv_length=5,
        cyclic_attention_window_size=5,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    fmha._enable_pdl = False
    fmha._multi_processor_count = 1

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "_append_preprocessed_kv",
        staticmethod(lambda *args, **kwargs: (object(), object(), object(), object(), [5])),
    )
    kv_pool = torch.empty(1)
    block_tables = torch.zeros(1, 2, 4, dtype=torch.int32)
    monkeypatch.setattr(
        thop,
        "build_trtllm_gen_kv_cache_metadata",
        lambda *args, **kwargs: (kv_pool, block_tables),
    )
    monkeypatch.setattr(
        trtllm_gen_module,
        "_clear_multi_ctas_kv_counter_workspace",
        lambda *args, **kwargs: None,
    )
    decode_args = None

    def _capture_decode(*args):
        nonlocal decode_args
        decode_args = args

    monkeypatch.setattr(
        trtllm_gen_module,
        "_trtllm_gen_batch_decode_with_kv_cache",
        _capture_decode,
    )

    fmha._run_preprocessed_generation(params)

    assert decode_args is not None
    assert decode_args[0].shape == (1, 2, 512)
    assert decode_args[1] is kv_pool
    assert decode_args[3] is block_tables
    assert decode_args[5] == 5


def test_large_head_context_requires_module_side_rope() -> None:
    attn = SimpleNamespace(
        head_dim=512,
        is_mla_enable=False,
        quant_mode=0,
        position_embedding_type=int(PositionEmbeddingType.rope_gpt_neox),
    )
    fmha = object.__new__(TritonCustomMaskFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=0,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        attention_input_type=AttentionInputType.context_only,
        is_fused_qkv=True,
    )

    assert not fmha.is_context_supported(q, None, None, metadata, forward_args)


def test_triton_prefill_accepts_separate_kv_page_tables() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernel test.")

    from tensorrt_llm._torch.attention_backend.triton_prefill import triton_prefill_with_custom_mask

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    q_len = 2
    prefix_len = 2
    head_dim = 64
    page_size = 2
    q = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    k_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    v_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)

    pool = torch.zeros(2, 1, page_size, head_dim, device=device, dtype=dtype)
    pool[0].copy_(k_prefix.transpose(0, 1))
    pool[1].copy_(v_prefix.transpose(0, 1))
    output = torch.empty_like(q)
    custom_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ],
        device=device,
        dtype=torch.bool,
    )

    triton_prefill_with_custom_mask(
        q=q,
        k=k,
        v=v,
        output=output,
        qo_indptr=torch.tensor([0, q_len], device=device, dtype=torch.int32),
        kv_cache=None,
        prefix_lens=torch.tensor([prefix_len], device=device, dtype=torch.int32),
        page_table_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        page_table_indices=torch.tensor([0], device=device, dtype=torch.int32),
        page_size=page_size,
        custom_mask=custom_mask.flatten(),
        sm_scale=head_dim**-0.5,
        k_cache=pool,
        v_cache=pool,
        v_page_table_indices=torch.tensor([1], device=device, dtype=torch.int32),
    )

    keys = torch.cat([k_prefix, k], dim=0).squeeze(1).float()
    values = torch.cat([v_prefix, v], dim=0).squeeze(1).float()
    scores = q.squeeze(1).float() @ keys.T * head_dim**-0.5
    scores.masked_fill_(~custom_mask, float("-inf"))
    reference = (scores.softmax(dim=-1) @ values).to(dtype).unsqueeze(1)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_triton_prefill_causal_generation_with_large_head() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernel test.")

    from tensorrt_llm._torch.attention_backend.triton_prefill import triton_prefill_with_custom_mask

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 512
    prefix_len = 3
    page_size = 4
    q = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    k_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    v_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    kv_cache = torch.zeros(
        1,
        2,
        1,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache[0, 0, 0, :prefix_len].copy_(k_prefix.squeeze(1))
    kv_cache[0, 1, 0, :prefix_len].copy_(v_prefix.squeeze(1))
    output = torch.empty_like(q)

    triton_prefill_with_custom_mask(
        q=q,
        k=k,
        v=v,
        output=output,
        qo_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        kv_cache=kv_cache,
        prefix_lens=torch.tensor([prefix_len], device=device, dtype=torch.int32),
        page_table_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        page_table_indices=torch.tensor([0], device=device, dtype=torch.int32),
        page_size=page_size,
        custom_mask=None,
        sm_scale=head_dim**-0.5,
    )

    keys = torch.cat([k_prefix, k], dim=0).squeeze(1).float()
    values = torch.cat([v_prefix, v], dim=0).squeeze(1).float()
    reference = ((q.squeeze(1).float() @ keys.T * head_dim**-0.5).softmax(dim=-1) @ values).to(
        dtype
    )
    torch.testing.assert_close(output.squeeze(1), reference, atol=3e-2, rtol=3e-2)


def test_gemma4_large_head_context_and_generation_match_flashinfer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Gemma4 attention test.")

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("The installed transformers version does not support Gemma4.")

    import tensorrt_llm
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4Attention
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = transformers.Gemma4TextConfig(
        model_type="gemma4_text",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=64,
        attention_k_eq_v=True,
        use_bidirectional_attention="vision",
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
        },
        torch_dtype="bfloat16",
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    seq_len = 4
    hidden = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    generation_hidden = torch.randn(
        1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    generation_position_ids = torch.tensor([[seq_len]], dtype=torch.int32, device="cuda")
    custom_mask = torch.ones(seq_len * seq_len, dtype=torch.bool, device="cuda")

    mixed_hidden = torch.cat([hidden, generation_hidden], dim=0)
    mixed_position_ids = torch.cat([position_ids, generation_position_ids], dim=1)

    def run_backend(backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            attn_backend=backend,
        )
        attention = (
            Gemma4Attention(
                model_config,
                layer_idx=0,
                is_sliding=False,
            )
            .cuda()
            .eval()
        )
        torch.manual_seed(7)
        with torch.no_grad():
            for parameter in attention.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.02)

        manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=64, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=2,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        assert manager.add_dummy_requests([1, 2], [seq_len + 1, seq_len]) is not None
        metadata_cls = get_attention_backend(backend).Metadata
        context_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=2,
            max_num_tokens=64,
            kv_cache_manager=manager,
            request_ids=[1],
            prompt_lens=[seq_len],
        )
        context_metadata.prepare()
        with torch.inference_mode():
            context_output = attention(
                position_ids,
                hidden,
                context_metadata,
                attention_mask=CustomAttentionMask.CUSTOM,
                attention_mask_data=custom_mask,
            ).clone()

        mixed_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len, 1], dtype=torch.int32),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0, seq_len],
            ),
            max_num_requests=2,
            max_num_tokens=64,
            kv_cache_manager=manager,
            request_ids=[2, 1],
            prompt_lens=[seq_len, seq_len],
        )
        mixed_metadata.prepare()
        with torch.inference_mode():
            mixed_output = attention(
                mixed_position_ids,
                mixed_hidden,
                mixed_metadata,
                attention_mask=CustomAttentionMask.CUSTOM,
                attention_mask_data=custom_mask,
            ).clone()
        manager.shutdown()
        return context_output, mixed_output

    flashinfer_context, flashinfer_mixed = run_backend("FLASHINFER")
    trtllm_context, trtllm_mixed = run_backend("TRTLLM")

    torch.testing.assert_close(trtllm_context, flashinfer_context, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(trtllm_mixed, flashinfer_mixed, atol=2e-2, rtol=2e-2)
