# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM configuration, loading and vision ownership regressions without checkpoint files."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.glm5_next_weight_mapper import (
    audit_glm5_next_checkpoint,
)
from tensorrt_llm._torch.models.modeling_glm5_next import (
    SPARSE_MLP,
    Glm5NextAllReduce,
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextLinearAttention,
    Glm5NextMTP,
    Glm5NextRuntimeContext,
    Glm5NextSparseAttention,
    build_glm5_next_runtime_context,
    glm5_next_tp_reduces,
)
from tensorrt_llm._torch.models.modeling_glm5_next_vision import (
    Glm5NextVisionModelBase,
    Glm5NextVLM,
)
from tensorrt_llm._torch.pyexecutor.config_utils import get_glm5_next_layer_masks, is_glm5_next
from tensorrt_llm.mapping import Mapping


def _config():
    config = PretrainedConfig(
        model_type="glm5_next",
        text_config=PretrainedConfig(
            model_type="glm5_next_text",
            dtype="bfloat16",
            rms_norm_eps=1e-05,
            num_attention_heads=64,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            v_head_dim=256,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            hidden_size=256,
            num_hidden_layers=2,
            mlp_layer_types=["dense", "dense"],
            first_k_dense_replace=2,
            num_nextn_predict_layers=1,
            max_position_embeddings=1024,
            linear_attn_config={
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        ),
        vision_config=PretrainedConfig(
            attention_bias=True,
            rms_norm_eps=1e-05,
            swiglu_limit=7.0,
            in_channels=3,
            depth=1,
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
            out_hidden_size=256,
            projection_intermediate_size=512,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
    )
    # The shared Transformers version predates this layer name. Attach the
    # synthetic schedule after its constructor validates the base config.
    config.text_config.layer_types = ["linear_attention", "deepseek_sparse_attention"]
    return config


@pytest.mark.cpu_only
@pytest.mark.parametrize("override", [None, "NCCL", "TWOSHOT"])
def test_allreduce_defaults_honor_user_strategy(override):
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    args = TorchLlmArgs(
        model="/tmp/dummy_model", **({"allreduce_strategy": override} if override else {})
    )
    apply_model_defaults_to_llm_args(args, Glm5NextVLM.get_model_defaults(args))
    assert args.allreduce_strategy == (override or "ONESHOT")
    strategy = AllReduceStrategy[args.allreduce_strategy]
    with patch(
        "tensorrt_llm._torch.distributed.AllReduce", side_effect=lambda **kw: Mock()
    ) as create:
        reduction = Glm5NextAllReduce(Mapping(world_size=4, tp_size=4), strategy=strategy)
    assert create.call_args_list[0].kwargs["strategy"] == strategy
    reduction(torch.zeros(512, 8))
    reduction(torch.zeros(513, 8))
    if override:
        assert reduction.small is reduction.large
        assert reduction.small.call_count == 2
    else:
        assert create.call_args_list[1].kwargs["strategy"] == AllReduceStrategy.NCCL
        reduction.small.assert_called_once()
        reduction.large.assert_called_once()


@pytest.mark.cpu_only
def test_mtp_rejects_unsupported_attention_dp_lm_head_sharding():
    from tensorrt_llm.llmapi import MTPDecodingConfig

    mapping = Mapping(
        world_size=4, tp_size=4, enable_attention_dp=True, enable_lm_head_tp_in_adp=True
    )
    with pytest.raises(NotImplementedError, match="tensor-parallel LM head"):
        Glm5NextForCausalLM(
            ModelConfig(
                pretrained_config=_config(),
                mapping=mapping,
                spec_config=MTPDecodingConfig(max_draft_len=3),
            )
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_model_type_recognition_does_not_hide_missing_schedule(model_type):
    config = SimpleNamespace(model_type=model_type, num_hidden_layers=1)
    assert is_glm5_next(config)
    with pytest.raises(ValueError, match="layer_types"):
        get_glm5_next_layer_masks(config)


@pytest.mark.cpu_only
def test_layer_masks_accept_composite_and_text_configs():
    config = _config()
    expected = ([False, True], [True, False])
    assert get_glm5_next_layer_masks(config) == expected
    assert get_glm5_next_layer_masks(config.text_config) == expected
    config.text_config.layer_types.pop()
    with pytest.raises(ValueError, match="num_hidden_layers"):
        get_glm5_next_layer_masks(config)
    config.text_config.layer_types = ["linear_attention", "unknown"]
    with pytest.raises(ValueError, match="must be exactly one"):
        get_glm5_next_layer_masks(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("is_cuda_graph", [False, True])
def test_runtime_context_uses_prepared_schedules_and_live_lengths(is_cuda_graph):
    live_lengths = torch.tensor([6, 9], dtype=torch.int32)
    prepared = SimpleNamespace(
        glm_block_tables=torch.tensor([[0, 1], [2, 3]]),
        glm_ctx_cu_seqlens=[0, 3],
        glm_cached_lens_host=[3, 5],
    )
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        mamba_metadata=prepared,
        seq_lens=torch.tensor([3, 4]),
        num_contexts=1,
        num_ctx_tokens=3,
        num_tokens=7,
        kv_lens_cuda=live_lengths,
        is_cuda_graph=is_cuda_graph,
    )
    context = build_glm5_next_runtime_context(metadata)
    assert context.ctx_cu_seqlens is prepared.glm_ctx_cu_seqlens
    assert context.cached_lens is prepared.glm_cached_lens_host
    assert context.gen_tokens_per_request == 4
    torch.testing.assert_close(context.kv_lens, live_lengths)
    assert context.kv_lens.data_ptr() == live_lengths.data_ptr()
    # MTP rewinds the device lengths without changing the host prefill schedule.
    live_lengths[1] -= 2
    torch.testing.assert_close(
        build_glm5_next_runtime_context(metadata).kv_lens, torch.tensor([6, 7], dtype=torch.int32)
    )
    prepared.glm_block_tables = None
    with pytest.raises(RuntimeError, match="requires prepared glm_block_tables"):
        build_glm5_next_runtime_context(metadata)
    prepared.glm_block_tables = torch.tensor([[0, 1], [2, 3]])
    metadata.kv_lens_cuda = None
    with pytest.raises(ValueError, match="kv_lens_cuda"):
        build_glm5_next_runtime_context(metadata)


@pytest.mark.cpu_only
def test_checkpoint_routes_vision_and_optional_mtp_separately():
    config = _config()
    keys = [
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.0.self_attn.A_log",
        "model.language_model.layers.2.eh_proj.weight",
        "lm_head.weight",
        "unexpected.weight",
    ]
    plain = audit_glm5_next_checkpoint(keys, config)
    mtp = audit_glm5_next_checkpoint(keys, config, num_mtp_layers=1)
    assert keys[0] in plain.ignored
    assert keys[2] in plain.ignored
    assert mtp.destinations["model.layers.2.eh_proj.weight"] == keys[2]
    assert mtp.destinations["model.layers.0.self_attn.A_log"] == keys[1]
    assert mtp.unresolved == ["unexpected.weight"]
    for invalid in (-1, 2):
        with pytest.raises(ValueError, match=f"cannot load {invalid} MTP layers"):
            audit_glm5_next_checkpoint(keys, config, num_mtp_layers=invalid)


@pytest.mark.cpu_only
@pytest.mark.parametrize("route", ["valid", "py_mamba", "manager_preference", "v1", "cpp", "ucx"])
def test_cache_manager_routing_guards(route, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextCacheManager
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
    from tensorrt_llm.llmapi import CacheTransceiverConfig, KvCacheConfig

    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    if route == "py_mamba":
        monkeypatch.setenv("TRTLLM_USE_PY_MAMBA", "1")
    if route == "manager_preference":
        monkeypatch.setenv("TLLM_MAMBA_MANAGER_PREFERENCE", "MIXED")
    config = ModelConfig(pretrained_config=_config())
    kv = KvCacheConfig(use_kv_cache_manager_v2=route != "v1")
    transceiver = CacheTransceiverConfig(
        backend="UCX" if route == "ucx" else "NIXL",
        transceiver_runtime="CPP" if route == "cpp" else "PYTHON",
    )
    if route == "valid":
        assert get_kv_cache_manager_cls(config, kv) is Glm5NextCacheManager
        assert get_kv_cache_manager_cls(config, kv, True, transceiver) is Glm5NextCacheManager
    else:
        with pytest.raises(ValueError, match="glm5_next"):
            get_kv_cache_manager_cls(config, kv, True, transceiver)


@pytest.mark.cpu_only
@pytest.mark.parametrize("fp8_kv_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize(
    "verify_kernel", [None, True, False], ids=["no-mtp", "fused-mtp", "sequential-mtp"]
)
def test_kv_cache_dtype_reaches_manager_construction(fp8_kv_cache, verify_kernel):
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextCacheManager
    from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    spec_config = MTPDecodingConfig(max_draft_len=3) if verify_kernel is not None else None
    config = ModelConfig(
        pretrained_config=_config().text_config,
        spec_config=spec_config,
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
            kv_cache_quant_algo=QuantAlgo.FP8 if fp8_kv_cache else None,
        ),
    )

    class AllocationReached(Exception):
        pass

    allocate = Mock(side_effect=AllocationReached)

    class RecordingManager(Glm5NextCacheManager):
        def __init__(self, *args, **kwargs):
            allocate(*args, **kwargs)

    with patch(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.is_kda_mtp_verify_available",
        return_value=verify_kernel is True,
    ):
        with pytest.raises(AllocationReached):
            _create_kv_cache_manager(
                model_engine=None,
                kv_cache_manager_cls=RecordingManager,
                model_config=config,
                mapping=Mapping(),
                kv_cache_config=KvCacheConfig(
                    use_kv_cache_manager_v2=True, enable_block_reuse=False
                ),
                tokens_per_block=32,
                max_seq_len=128,
                max_batch_size=1,
                spec_config=spec_config,
                sparse_attention_config=None,
                max_num_tokens=64,
                max_beam_width=1,
                kv_connector_manager=None,
                dtype=torch.bfloat16,
                is_draft=False,
            )
    allocate.assert_called_once()
    assert allocate.call_args.kwargs["dtype"] == (DataType.FP8 if fp8_kv_cache else DataType.BF16)
    if verify_kernel is True:
        assert allocate.call_args.kwargs["kda_replay_num_spec"] == 3
    else:
        assert "kda_replay_num_spec" not in allocate.call_args.kwargs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("deferred", [False, True])
def test_encoder_only_factory_materializes_attention_weights(deferred):
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVisionModel

    config = _config()
    config.architectures = ["Glm5NextForConditionalGeneration"]
    model_config = ModelConfig(
        pretrained_config=config, mm_encoder_only=True, skip_create_weights_in_init=deferred
    )
    with torch.device("meta"):
        encoder = AutoModelForCausalLM.from_config(model_config)
    assert isinstance(encoder, Glm5NextVisionModelBase)
    assert isinstance(encoder.visual, Glm5NextVisionModel)
    for block in encoder.visual.blocks:
        for projection in (block.attn.qkv_proj, block.attn.o_proj, block.mlp.down_proj):
            assert projection._weights_created
            assert projection.weight is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kda_shards_preserve_fp32_gate_parameters():
    config = _config().text_config
    full_bias = torch.arange(64 * 128, dtype=torch.float32)
    full_log = torch.arange(64, dtype=torch.float32)
    biases, logs = [], []
    with patch("tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer.AllReduce"):
        for rank in range(4):
            layer = Glm5NextLinearAttention(
                config, 0, mapping=Mapping(world_size=4, tp_size=4, rank=rank)
            )
            assert not layer.use_full_rank_gate
            assert layer.A_log.shape == (16,)
            assert layer.dt_bias.shape == (16 * 128,)
            assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
            biases.append(layer.shard_checkpoint_tensor("dt_bias", full_bias))
            logs.append(layer.shard_checkpoint_tensor("A_log", full_log))
    assert "use_full_rank_gate" not in config.linear_attn_config
    torch.testing.assert_close(torch.cat(biases), full_bias)
    torch.testing.assert_close(torch.cat(logs), full_log)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_vision_attention_dp_is_local_and_matches_single_rank():
    config = _config()
    reference = Glm5NextVisionModelBase(ModelConfig(pretrained_config=config)).cuda().eval()
    reference.visual.setup_attn_metadata(max_num_tokens=128)
    torch.manual_seed(7)
    for parameter in reference.parameters():
        parameter.normal_(0, 0.05)

    # Rank 0 has no image; the other ranks see different token counts and
    # pixels. Any TP collective on the encoder path is an error, even when
    # another rank has no encoder work at all.
    def local_allreduce(**kwargs):
        assert kwargs["mapping"].tp_size == 1, "vision must not reduce across DP ranks"
        return AllReduce(**kwargs)

    with patch("tensorrt_llm._torch.distributed.AllReduce", side_effect=local_allreduce):
        for rank, width in enumerate((0, 4, 6, 8)):
            mapping = Mapping(world_size=4, tp_size=4, rank=rank, enable_attention_dp=True)
            local = (
                Glm5NextVisionModelBase(ModelConfig(pretrained_config=config, mapping=mapping))
                .cuda()
                .eval()
            )
            local.load_state_dict(reference.state_dict())
            local.visual.setup_attn_metadata(max_num_tokens=128)
            if not width:
                continue
            pixels = torch.randn(4 * width, 3 * 2 * 2 * 2, device="cuda", dtype=torch.bfloat16)
            grid = torch.tensor([[1, 4, width]])
            expected = reference.encode_batched(pixels, grid)
            actual = local.encode_batched(pixels, grid)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("temporal_groups", [1, 3], ids=["image", "video"])
@torch.inference_mode()
def test_vision_weights_and_features_match_hf(temporal_groups):
    hf_module = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    Glm5NextVisionModel = hf_module.Glm5NextVisionModel

    from tensorrt_llm._torch.models.modeling_glm5_next_vision import _flatten_video_grid_thw

    torch.manual_seed(11)
    config = _config()
    config.vision_config = hf_module.Glm5NextVisionConfig.from_dict(config.vision_config.to_dict())
    hf = Glm5NextVisionModel(config.vision_config).to(device="cuda", dtype=torch.bfloat16).eval()
    runtime = Glm5NextVisionModelBase(ModelConfig(pretrained_config=config)).cuda().eval()
    runtime.visual.setup_attn_metadata(max_num_tokens=128)
    weights = {"model.visual." + name: tensor for name, tensor in hf.state_dict().items()}
    runtime.load_weights(weights)
    pixels = torch.randn(
        temporal_groups * 4 * 6, 3 * 2 * 2 * 2, device="cuda", dtype=torch.bfloat16
    )
    grid = torch.tensor([[temporal_groups, 4, 6]])
    flat_grid = _flatten_video_grid_thw(grid)
    expected = hf(pixels, grid_thw=flat_grid.cuda()).pooler_output
    actual = runtime.encode_batched(pixels, flat_grid)
    assert actual.shape == (temporal_groups * 6, config.vision_config.out_hidden_size)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-12
    )
    assert cosine > 0.999
    assert relative_l2 < 0.03


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10,
    reason="requires Blackwell sparse MLA",
)
@torch.no_grad()
def test_sparse_prefill_continuation_preserves_partial_pools():
    """A chunk boundary inside a k-pool preserves both cached state and attention outputs."""
    torch.manual_seed(17)
    layer = Glm5NextSparseAttention(_config().text_config, 1).cuda().eval()
    for parameter in layer.parameters():
        parameter.normal_(0, 0.05)
    hidden = torch.randn(13, 256, device="cuda", dtype=torch.bfloat16)
    hidden[:5] += 1
    tables = torch.zeros(1, 256, device="cuda", dtype=torch.long)
    tables[0, :2] = torch.tensor([2, 0], device="cuda")

    def run(lengths, discard_prefix=False):
        latent = torch.zeros(4, 8, 1, layer.kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        index = torch.zeros(
            4, 8, 1, layer.indexer.cache_state_dim, device="cuda", dtype=torch.bfloat16
        )
        manager = SimpleNamespace(
            tokens_per_block=8,
            get_latent_state_buffer=lambda _: latent,
            get_index_state_buffer=lambda _: index,
        )
        outputs = []
        cached = 0
        for length in lengths:
            if cached and discard_prefix:
                latent.zero_()
                index.zero_()
            metadata = SimpleNamespace(
                kv_cache_manager=manager,
                seq_lens=torch.tensor([length]),
                num_contexts=1,
                kv_lens_cuda=torch.tensor([cached + length], device="cuda", dtype=torch.int32),
                mamba_metadata=SimpleNamespace(glm_block_tables=tables),
            )
            outputs.append(
                layer.forward_prefill(
                    hidden[cached : cached + length], [0, length], [cached], metadata
                )
            )
            cached += length
        return torch.cat(outputs), latent, index

    reference, latent, index = run([13])
    actual, chunked_latent, chunked_index = run([5, 8])
    torch.testing.assert_close(chunked_latent, latent, rtol=0.02, atol=1e-3)
    torch.testing.assert_close(chunked_index, index, rtol=0.02, atol=1e-3)
    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.03
    # Negative control: this gate must detect losing the previous chunk.
    broken, _, _ = run([5, 8], discard_prefix=True)
    broken_l2 = (broken.float() - reference.float()).norm() / reference.float().norm()
    assert broken_l2 > 0.03


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_request", [1, 4], ids=["decode", "verify"])
@pytest.mark.parametrize("attention_dp", [False, True], ids=["tp", "attention-dp"])
@pytest.mark.parametrize("layer_kind", ["decoder", "mtp"])
def test_mixed_batches_keep_one_full_batch_moe_call(
    tokens_per_request: int, attention_dp: bool, layer_kind: str
) -> None:
    # ADP ranks may have different phase mixes, but must issue one MoE call
    # per layer. TP attention must reduce its output once for the full batch.
    batches = [(3, 1), (3, 0), (0, 1), (2, 1)]
    counts = [
        context_tokens + generations * tokens_per_request for context_tokens, generations in batches
    ]
    for rank, (context_tokens, generations) in enumerate(batches):
        mapping = Mapping(
            world_size=4, tp_size=4, moe_ep_size=4, rank=rank, enable_attention_dp=attention_dp
        )
        contexts = int(context_tokens > 0)
        metadata = SimpleNamespace(all_rank_num_tokens=counts)
        context = Glm5NextRuntimeContext(
            num_contexts=contexts,
            num_ctx_tokens=context_tokens,
            num_generations=generations,
            ctx_cu_seqlens=[0, context_tokens] if contexts else [0],
            cached_lens=[0] * contexts + [1] * generations,
            kv_lens=torch.tensor(
                ([context_tokens] if contexts else [])
                + ([1 + tokens_per_request] if generations else [])
            ),
            metadata=metadata,
            gen_tokens_per_request=tokens_per_request,
        )
        hidden = torch.arange(counts[rank] * 4 * 8, dtype=torch.float32).view(-1, 4, 8)
        reduction = Mock(side_effect=lambda value: value)
        attention = torch.nn.Module()
        attention.tp_all_reduce = reduction if glm5_next_tp_reduces(mapping) else None

        def attend(x, *args, reduce=True, **kwargs):
            return attention.tp_all_reduce(x) if reduce and attention.tp_all_reduce else x

        for phase in ("prefill", "decode", "verify"):
            implementation = MethodType(
                getattr(Glm5NextSparseAttention, f"forward_{phase}"), attention
            )
            setattr(
                attention,
                f"forward_{phase}",
                create_autospec(implementation, side_effect=attend),
            )
        attention.forward = MethodType(Glm5NextSparseAttention.forward, attention)
        connection = SimpleNamespace(
            pre_mapping=lambda streams: (None, None, streams.mean(dim=1)),
            post_mapping=lambda output, residual, post, comb: residual + output.unsqueeze(1),
        )
        layer = SimpleNamespace(
            attention_type="deepseek_sparse_attention",
            layer_idx=1,
            mlp_type=SPARSE_MLP,
            self_attn=attention,
            hc_attn=connection,
            hc_ffn=connection,
            input_layernorm=torch.nn.Identity(),
            post_attention_layernorm=torch.nn.Identity(),
            mlp=Mock(side_effect=lambda x, all_rank_num_tokens: x),
        )
        if layer_kind == "decoder":
            attn_input = hidden.mean(dim=1)
            layer.run_mlp = MethodType(Glm5NextDecoderLayer.run_mlp, layer)
            result = Glm5NextDecoderLayer.forward(layer, hidden_states=hidden, runtime_ctx=context)
            expected = hidden + attn_input.unsqueeze(1)
            expected = expected + expected.mean(dim=1, keepdim=True)
        else:
            hidden = hidden.mean(dim=1)
            input_ids = torch.arange(hidden.shape[0])
            embed_tokens = torch.nn.Embedding(hidden.shape[0], 8)
            layer.enorm = layer.hnorm = torch.nn.Identity()
            layer.model_config = SimpleNamespace(mapping=mapping)
            layer.eh_proj = torch.nn.Linear(16 if attention_dp else 4, 8, bias=False)
            layer.shared_head = SimpleNamespace(norm=torch.nn.Identity())
            projected_input = torch.cat([embed_tokens(input_ids), hidden], dim=-1)
            if not attention_dp:
                projected_input = projected_input.chunk(mapping.tp_size, dim=-1)[mapping.tp_rank]
            attn_input = layer.eh_proj(projected_input)
            with patch(
                "tensorrt_llm._torch.models.modeling_glm5_next.build_glm5_next_runtime_context",
                return_value=context,
            ):
                result = Glm5NextMTP.forward(layer, input_ids, None, hidden, embed_tokens, metadata)
            expected = 4 * attn_input
        torch.testing.assert_close(result, expected)
        if attention_dp:
            reduction.assert_not_called()
        else:
            reduction.assert_called_once()
            torch.testing.assert_close(reduction.call_args.args[0], attn_input)
        layer.mlp.assert_called_once()
        mlp_tokens, all_rank_num_tokens = layer.mlp.call_args.args
        assert mlp_tokens.shape[0] == counts[rank]
        assert all_rank_num_tokens is counts
        if contexts:
            attention.forward_prefill.assert_called_once()
            torch.testing.assert_close(
                attention.forward_prefill.call_args.args[0], attn_input[:context_tokens]
            )
        else:
            attention.forward_prefill.assert_not_called()
        generation = (
            attention.forward_decode if tokens_per_request == 1 else attention.forward_verify
        )
        if generations:
            generation.assert_called_once()
            assert generation.call_args.kwargs["metadata"] is metadata
            torch.testing.assert_close(generation.call_args.args[0], attn_input[context_tokens:])
            if contexts:
                assert generation.call_args.kwargs["reduce"] is False
            if tokens_per_request > 1:
                assert generation.call_args.kwargs["tokens_per_request"] == tokens_per_request
        else:
            generation.assert_not_called()
        other = attention.forward_decode if tokens_per_request > 1 else attention.forward_verify
        other.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("generation_phase", ["decode", "verify"])
@pytest.mark.parametrize("reduce_output", [False, True], ids=["attention-dp", "tp"])
def test_packed_attention_splits_phases_and_reduces_once(generation_phase, reduce_output):
    gen_tokens = 4 if generation_phase == "verify" else 1
    hidden = torch.randn(3 + gen_tokens, 8)
    ctx = torch.zeros(3, 8)
    gen = torch.ones(gen_tokens, 8)
    reduction = Mock(side_effect=lambda value: value + 10) if reduce_output else None
    attention = SimpleNamespace(
        forward_prefill=Mock(return_value=ctx),
        forward_decode=Mock(return_value=gen),
        forward_verify=Mock(return_value=gen),
        tp_all_reduce=reduction,
    )
    metadata = SimpleNamespace()
    context = Glm5NextRuntimeContext(
        num_contexts=1,
        num_ctx_tokens=3,
        num_generations=1,
        ctx_cu_seqlens=[0, 3],
        cached_lens=[0, 8],
        kv_lens=torch.tensor([3, 8 + gen_tokens]),
        metadata=metadata,
        gen_tokens_per_request=gen_tokens,
    )
    result = Glm5NextSparseAttention.forward(attention, hidden, metadata, runtime_ctx=context)
    expected = torch.cat([ctx, gen]) + (10 if reduce_output else 0)
    torch.testing.assert_close(result, expected)
    attention.forward_prefill.assert_called_once()
    assert attention.forward_prefill.call_args.kwargs["reduce"] is False
    selected = getattr(attention, "forward_" + generation_phase)
    selected.assert_called_once()
    assert selected.call_args.kwargs["reduce"] is False
    torch.testing.assert_close(selected.call_args.args[0], hidden[3:])
    other = attention.forward_decode if generation_phase == "verify" else attention.forward_verify
    other.assert_not_called()
    if reduction is not None:
        reduction.assert_called_once()
