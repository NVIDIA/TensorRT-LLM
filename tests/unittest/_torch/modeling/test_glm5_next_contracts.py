# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM configuration, loading and vision ownership regressions without checkpoint files."""

from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig

from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextMamba2Metadata
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
from tensorrt_llm._torch.pyexecutor.engine.multimodal import setup_mm_encoder_cuda_graphs
from tensorrt_llm.llmapi.llm_args import MultimodalConfig, MultimodalEncoderCudaGraphConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.cpu_only
def test_config_fallback_round_trip(tmp_path):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from tensorrt_llm._torch.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    fields = _config().to_dict()
    fields["architectures"] = ["Glm5NextForConditionalGeneration"]
    fields["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config = Glm5NextConfig.from_dict(fields)
    config.save_pretrained(tmp_path)
    loaded = load_pretrained_config(str(tmp_path))
    assert isinstance(loaded, CONFIG_MAPPING["glm5_next"])
    assert isinstance(AutoConfig.from_pretrained(tmp_path), type(loaded))
    assert loaded.text_config.layer_types == fields["text_config"]["layer_types"]
    assert loaded.text_config.linear_attn_config == fields["text_config"]["linear_attn_config"]
    assert loaded.text_config.qk_head_dim == 256
    assert loaded.text_config.head_dim == 0
    assert loaded.vision_config.spatial_merge_size == 2
    assert loaded.quantization_config == fields["quantization_config"]
    assert get_glm5_next_layer_masks(loaded) == ([False, True], [True, False])
    # Registration must prefer native classes when the installed Transformers has them.
    if "transformers.models.glm5_next" in type(loaded).__module__:
        assert type(loaded) is not Glm5NextConfig
    else:
        assert type(loaded.text_config) is Glm5NextTextConfig


@pytest.mark.cpu_only
def test_config_fallback_matches_native():
    from tensorrt_llm._torch.configs.glm5_next import Glm5NextConfig

    native = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    fields = _config().to_dict()
    fallback = Glm5NextConfig.from_dict(fields)
    reference = native.Glm5NextConfig.from_dict(fields)
    for key in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "head_dim",
        "v_head_dim",
        "index_topk",
        "index_kpool",
        "indexer_types",
        "linear_attn_config",
        "linear_head_dim",
        "linear_num_heads",
        "linear_conv_kernel_dim",
        "linear_lower_bound",
        "layer_types",
        "mlp_layer_types",
        "hc_mult",
        "hc_eps",
        "hc_sinkhorn_iters",
        "swiglu_limit",
    ):
        assert getattr(fallback.text_config, key) == getattr(reference.text_config, key), key
    for key in ("depth", "num_heads", "hidden_size", "spatial_merge_size", "patch_size"):
        assert getattr(fallback.vision_config, key) == getattr(reference.vision_config, key), key


@pytest.mark.cpu_only
def test_text_processing_without_native_processor(monkeypatch):
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES

    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextInputProcessor
    from tensorrt_llm.llmapi import SamplingParams

    monkeypatch.delitem(PROCESSOR_MAPPING_NAMES, "glm5_next", raising=False)
    # A tokenizer called on one string without return_tensors returns a flat list.
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=[1, 2, 3]))
    with patch(
        "tensorrt_llm._torch.models.modeling_glm5_next_vision.AutoProcessor.from_pretrained"
    ) as load_processor:
        processor = Glm5NextInputProcessor("unused", _config(), tokenizer=tokenizer)
        tokens, extra = processor.call_with_text_prompt({"prompt": "hello"}, SamplingParams())
        assert tokens == [1, 2, 3]
        assert extra is None
        assert "return_tensors" not in tokenizer.call_args.kwargs
        load_processor.assert_not_called()
        with pytest.raises(RuntimeError, match="transformers==5.17.0"):
            processor.get_mm_max_tokens_per_item()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides,image_rescale,video_rescale,sample_frames",
    [
        ({}, True, False, False),
        ({"do_rescale": True, "do_sample_frames": True}, True, True, True),
        (
            {
                "images_kwargs": {"do_rescale": False},
                "videos_kwargs": {"do_rescale": True, "do_sample_frames": True},
            },
            False,
            True,
            True,
        ),
    ],
    ids=["defaults", "flat-overrides", "modality-overrides"],
)
def test_mixed_processor_kwargs(overrides, image_rescale, video_rescale, sample_frames):
    from PIL import Image

    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextInputProcessor
    from tensorrt_llm.inputs.multimodal_data import VideoData
    from tensorrt_llm.llmapi import SamplingParams

    processor = Glm5NextInputProcessor("unused", _config(), tokenizer=Mock())
    processor._processor = Mock(
        image_token="<|image|>",
        video_token="<|video|>",
        return_value={"input_ids": torch.tensor([[1, 2]])},
    )
    original = deepcopy(overrides)
    processor.call_with_text_prompt(
        {
            "prompt": "<|image|><|video|>",
            "multi_modal_data": {
                "image": [Image.new("RGB", (28, 28))],
                "video": [VideoData(frames=[torch.zeros(3, 28, 28)] * 2, metadata={})],
            },
            "mm_processor_kwargs": overrides,
        },
        SamplingParams(),
    )
    kwargs = processor._processor.call_args.kwargs
    assert kwargs["images_kwargs"].get("do_rescale", kwargs.get("do_rescale")) == image_rescale
    assert kwargs["videos_kwargs"].get("do_rescale", kwargs.get("do_rescale")) == video_rescale
    assert (
        kwargs["videos_kwargs"].get("do_sample_frames", kwargs.get("do_sample_frames"))
        == sample_frames
    )
    assert overrides == original
    assert kwargs["images_kwargs"] is not overrides.get("images_kwargs")
    assert kwargs["videos_kwargs"] is not overrides.get("videos_kwargs")


@pytest.mark.cpu_only
@pytest.mark.parametrize("tensor_images", [False, True], ids=["pil-image", "tensor-image"])
def test_mixed_processor_matches_separate_inputs(tensor_images):
    native = pytest.importorskip("transformers.models.glm5_next.processing_glm5_next")
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Glm5NextImageProcessor, Glm5NextVideoProcessor, PreTrainedTokenizerFast

    from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextInputProcessor
    from tensorrt_llm.inputs.multimodal_data import VideoData
    from tensorrt_llm.llmapi import SamplingParams

    special_tokens = ["[UNK]", "<|image|>", "<|video|>", "<|begin_of_video|>", "<|end_of_video|>"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(special_tokens, range(5))), unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        additional_special_tokens=special_tokens[1:],
    )
    processor = Glm5NextInputProcessor("unused", _config(), tokenizer=tokenizer)
    processor._processor = native.Glm5NextProcessor(
        image_processor=Glm5NextImageProcessor(do_resize=False),
        video_processor=Glm5NextVideoProcessor(do_resize=False),
        tokenizer=tokenizer,
    )
    pil_image = Image.new("RGB", (28, 28), (64, 128, 192))
    tensor_image = (
        torch.tensor([64, 128, 192], dtype=torch.float32)[:, None, None].expand(3, 28, 28) / 255
    )
    media = {
        "image": [tensor_image if tensor_images else pil_image],
        "video": [
            VideoData(
                frames=[pil_image if tensor_images else tensor_image] * 2,
                metadata={"fps": 2, "frames_indices": [0, 1]},
            )
        ],
    }
    _, mixed = processor.call_with_text_prompt(
        {"prompt": "<|image|><|video|>", "multi_modal_data": media}, SamplingParams()
    )
    for modality in ("image", "video"):
        _, separate = processor.call_with_text_prompt(
            {"prompt": f"<|{modality}|>", "multi_modal_data": {modality: media[modality]}},
            SamplingParams(),
        )
        for key, expected in separate["multimodal_data"][modality].items():
            torch.testing.assert_close(
                mixed["multimodal_data"][modality][key], expected, rtol=0, atol=0
            )


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_images", [1, 2])
def test_profiling_dummy_has_encoder_item_metadata(num_images: int) -> None:
    from tensorrt_llm._torch.models.modeling_glm5_next_vision import (
        Glm5NextInputProcessor,
        _glm5_next_build_batched_input,
    )
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
        EncoderGroup,
        MultimodalModelMixin,
        encode_multimodal_by_groups,
    )
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    processor = Glm5NextInputProcessor("unused", _config(), tokenizer=Mock())
    image_processor = Mock(
        return_value={
            "pixel_values": torch.zeros(num_images * 16, 12),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * num_images),
        },
        max_image_tokens=4,
        merge_size=2,
    )
    processor._processor = SimpleNamespace(image_processor=image_processor)
    model = Mock(spec=MultimodalModelMixin, dtype=torch.float16)
    profiler = object.__new__(KvCacheCreator)
    profiler._llm_args = SimpleNamespace(disable_mm_encoder=False)
    profiler._profiling_stage_data = {"enable_mm_reqs": True}
    profiler._model_engine = SimpleNamespace(
        model=model,
        input_processor=processor,
        encoder_max_num_tokens=num_images * 16,
        encoder_batch_size=num_images,
    )

    params = profiler._create_dummy_encoder_inputs()

    def encode(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        assert grid_thw.shape == (num_images, 3)
        return pixel_values.new_zeros((pixel_values.shape[0] // 4, 2))

    output = encode_multimodal_by_groups(
        (EncoderGroup(("image", "video"), encode, _glm5_next_build_batched_input),), params
    )
    assert output.shape == (num_images * 4, 2)
    assert output.dtype == torch.float16
    metadata = params[0].item_metadata
    assert metadata is not None
    metadata.validate()
    assert metadata.item_refs == tuple(("image", i) for i in range(num_images))
    assert metadata.encoder_token_lengths == (16,) * num_images
    assert metadata.output_embedding_lengths == (4,) * num_images


def _materialize_meta(module: torch.nn.Module) -> None:
    """Allocate meta tensors on CUDA as ModelLoader does after MetaInitMode construction."""
    module._apply(lambda t: torch.empty_like(t, device="cuda") if t.is_meta else t)


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
@pytest.mark.parametrize(
    "architecture", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize("sm_100f, expected", [(True, "TRTLLM"), (False, "CUTLASS")])
def test_moe_auto_backend_prefers_trtllm_gen_on_blackwell(
    monkeypatch, architecture, sm_100f, expected
):
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr("tensorrt_llm._torch.model_config.is_sm_100f", lambda: sm_100f)
    nvfp4 = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    assert ModelConfig.resolve_moe_backend("AUTO", architecture) == expected
    assert ModelConfig.resolve_moe_backend("AUTO", architecture, quant_config=nvfp4) == expected
    assert ModelConfig.resolve_moe_backend("CUTLASS", architecture, quant_config=nvfp4) == "CUTLASS"


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
@pytest.mark.parametrize(
    ("model_config", "separate"),
    [
        (_config(), False),
        (_config().text_config, False),
        (PretrainedConfig(model_type="deepseek_v3", num_nextn_predict_layers=1), True),
    ],
    ids=["glm5_next", "glm5_next_text", "deepseek_v3"],
)
def test_mtp_draft_layers_share_the_target_kv_cache(model_config, separate):
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
    from tensorrt_llm.llmapi import MTPDecodingConfig

    spec_config = MTPDecodingConfig(max_draft_len=3)
    update_spec_config_from_model_config(spec_config, model_config)
    assert should_use_separate_draft_kv_cache(spec_config) is separate


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
    prepared = object.__new__(Glm5NextMamba2Metadata)
    prepared.glm_block_tables = torch.tensor([[0, 1], [2, 3]])
    prepared.glm_ctx_cu_seqlens = [0, 3]
    prepared.glm_cached_lens_host = [3, 5]
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
    metadata.kv_lens_cuda = live_lengths
    for invalid in (None, False, SimpleNamespace(glm_block_tables=prepared.glm_block_tables)):
        metadata.mamba_metadata = invalid
        with pytest.raises(AssertionError, match="Glm5NextMamba2Metadata"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nvfp4_loader_preserves_calibrated_scales():
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._utils import get_sm_version
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 comparison requires SM100/103")
    config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16))

    def build():
        return GatedMLP(
            hidden_size=128,
            intermediate_size=256,
            bias=False,
            dtype=torch.bfloat16,
            config=config,
            swiglu_limit=7.0,
        )

    with torch.device("cuda"):
        reference = build()
        # The model loader path: meta construction, then uninitialized allocation.
        with MetaInitMode():
            actual = build()
    _materialize_meta(actual)
    generator = torch.Generator().manual_seed(19136)
    groups = {}
    for name, rows, cols in (
        ("gate_proj", 256, 128),
        ("up_proj", 256, 128),
        ("down_proj", 128, 256),
    ):
        groups[name] = {
            "weight": torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
            ),
            "weight_scale": torch.ones(rows, cols // 16, dtype=torch.float8_e4m3fn),
            "input_scale": torch.tensor(0.01),
            "weight_scale_2": torch.tensor(0.02),
        }
    reference.gate_up_proj.load_weights([groups["gate_proj"], groups["up_proj"]])
    reference.down_proj.load_weights([groups["down_proj"]])
    weights = {
        f"{name}.{label}": tensor
        for name, group in groups.items()
        for label, tensor in group.items()
    }
    Glm5NextForCausalLM._fill_module(
        None, actual, "dense", "", [(key, key) for key in weights], weights, torch.device("cuda")
    )
    for name in ("gate_up_proj", "down_proj"):
        loaded, expected = getattr(actual, name), getattr(reference, name)
        for scale in ("input_scale", "weight_scale_2", "alpha"):
            torch.testing.assert_close(getattr(loaded, scale), getattr(expected, scale))
            assert torch.all(getattr(loaded, scale) > 0)
        # No KV scales in the checkpoint: they keep their construction value.
        for scale in ("kv_scales", "inv_kv_scales"):
            torch.testing.assert_close(
                getattr(loaded, scale), torch.ones_like(getattr(loaded, scale))
            )
    x = torch.randn(8, 128, generator=generator, dtype=torch.bfloat16).cuda()
    torch.testing.assert_close(actual(x), reference(x), atol=0, rtol=0)


@pytest.mark.cpu_only
def test_checkpoint_lazy_scalar_materialization(tmp_path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    expected = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "input_scale": torch.tensor(0.01),
    }
    path = tmp_path / "scalar.safetensors"
    save_file(expected, path)
    module = torch.nn.Module()
    for name, tensor in expected.items():
        module.register_parameter(
            name, torch.nn.Parameter(torch.empty_like(tensor), requires_grad=False)
        )
    with safe_open(path, framework="pt") as checkpoint:
        weights = {key: checkpoint.get_slice(key) for key in checkpoint.keys()}
        Glm5NextForCausalLM._fill_module(
            None,
            module,
            "scalar",
            "",
            [(key, key) for key in weights],
            weights,
            torch.device("cpu"),
        )
    for name, tensor in expected.items():
        torch.testing.assert_close(getattr(module, name), tensor, rtol=0, atol=0)


@pytest.mark.cpu_only
def test_nvfp4_moe_loader_collects_expert_tensors() -> None:
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    owner = torch.nn.Module()
    owner.mlp = torch.nn.Module()
    owner.mlp.experts = experts = torch.nn.Module()
    # This EP rank owns experts 0 and 2 of 4.
    experts.num_experts = 4
    experts.initial_local_expert_ids = [0, 2]
    experts.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    experts.load_weights = Mock()
    labels = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    prefix = "model.layers.3."
    weights = {
        f"{prefix}mlp.experts.{expert}.{proj}.{label}": torch.tensor(float(expert))
        for expert in range(4)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for label in labels
    }
    entries = [(key, key) for key in weights]
    Glm5NextForCausalLM._fill_module(None, owner, 3, prefix, entries, weights, torch.device("cpu"))
    (loaded,), _ = experts.load_weights.call_args
    proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    # Weights come from local experts only; input_scale comes from every
    # expert so each rank reduces the same layer-wide activation scale.
    expected = {
        f"{expert}.{proj_to_w[proj]}.{label}": weights[
            f"{prefix}mlp.experts.{expert}.{proj}.{label}"
        ]
        for expert in range(4)
        for proj in proj_to_w
        for label in labels
        if expert in (0, 2) or label == "input_scale"
    }
    assert loaded == [expected]

    # A missing expert tensor fails with the collected and expected counts.
    del weights[f"{prefix}mlp.experts.1.down_proj.input_scale"]
    entries = [(key, key) for key in weights]
    with pytest.raises(ValueError, match="collected 29 expert tensors, expected 30"):
        Glm5NextForCausalLM._fill_module(
            None, owner, 3, prefix, entries, weights, torch.device("cpu")
        )


@pytest.mark.cpu_only
def test_nvfp4_excluded_layer_keeps_experts_unquantized(monkeypatch) -> None:
    import importlib

    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextMoE
    from tensorrt_llm._torch.modules import linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    # The package re-exports the create_moe function under the submodule's name.
    create_moe_module = importlib.import_module("tensorrt_llm._torch.moe.fused_moe.create_moe")
    monkeypatch.setattr(linear, "get_sm_version", lambda: 100)
    expert_quant = {}

    def fake_create_moe(*, layer_idx, override_quant_config, **_):
        expert_quant[layer_idx] = override_quant_config
        return torch.nn.Module()

    monkeypatch.setattr(create_moe_module, "create_moe", fake_create_moe)
    text = PretrainedConfig(
        hidden_size=128,
        moe_intermediate_size=256,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        swiglu_limit=10.0,
    )
    # Layer 1 is excluded with the text-runtime name the model rewrites
    # checkpoint exclusions to.
    config = ModelConfig(
        pretrained_config=text,
        moe_backend="CUTLASS",
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4, group_size=16, exclude_modules=["model.layers.1.*"]
        ),
    )
    for layer_idx in (0, 1):
        Glm5NextMoE(text, config, layer_idx)
    assert expert_quant[0].quant_algo == QuantAlgo.NVFP4
    assert expert_quant[1].quant_algo is None


@pytest.mark.cpu_only
def test_text_quant_config_renames_checkpoint_exclusions() -> None:
    from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_text_quant_config
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    checkpoint_exclusions = [
        "lm_head",
        "model.language_model.layers.3.self_attn*",
        "model.language_model.layers.45*",
        "model.visual*",
    ]
    config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4, group_size=16, exclude_modules=list(checkpoint_exclusions)
        )
    )
    quant = glm5_next_text_quant_config(config).quant_config
    assert quant.exclude_modules == [
        "lm_head",
        "model.layers.3.self_attn*",
        "model.layers.45*",
        "model.visual*",
    ]
    assert config.quant_config.exclude_modules == checkpoint_exclusions
    unquantized = ModelConfig(quant_config=QuantConfig())
    assert glm5_next_text_quant_config(unquantized) is unquantized


class _ReplicatedFusionOwner(torch.nn.Module):
    """An owner whose ``fused`` Linear concatenates two checkpoint projections."""

    FUSED_LINEARS = (("fused", ("first", "second")),)


@pytest.mark.cpu_only
def test_nvfp4_replicated_fusion_is_rejected(monkeypatch) -> None:
    from tensorrt_llm._torch.modules import linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(linear, "get_sm_version", lambda: 100)
    owner = _ReplicatedFusionOwner()
    owner.fused = linear.Linear(
        128,
        256,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        skip_create_weights_in_init=True,
        reduce_output=False,
    )
    weights = {}
    for name in ("first", "second"):
        weights[f"{name}.weight"] = torch.zeros(128, 64, dtype=torch.uint8)
        weights[f"{name}.weight_scale"] = torch.ones(128, 8, dtype=torch.float8_e4m3fn)
        weights[f"{name}.weight_scale_2"] = torch.tensor(0.02)
        weights[f"{name}.input_scale"] = torch.tensor(0.01)
    with patch.object(owner.fused, "load_weights") as load:
        with pytest.raises(ValueError, match="cannot fuse NVFP4 projections into 'fused'"):
            Glm5NextForCausalLM._fill_module(
                None,
                owner,
                "replicated",
                "",
                [(key, key) for key in weights],
                weights,
                torch.device("cpu"),
            )
        load.assert_not_called()


@pytest.mark.cpu_only
def test_nvfp4_loader_rejects_missing_scales(monkeypatch) -> None:
    from tensorrt_llm._torch.modules import linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(linear, "get_sm_version", lambda: 100)
    projection = linear.Linear(
        128,
        128,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        skip_create_weights_in_init=True,
        reduce_output=False,
    )
    weights = {
        "weight": torch.zeros(128, 64, dtype=torch.uint8),
        "weight_scale": torch.ones(128, 8, dtype=torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.02),
        "input_scale": torch.tensor(0.01),
    }
    missing = "weight_scale_2"
    del weights[missing]
    with patch.object(projection, "load_weights") as load:
        with pytest.raises(ValueError, match=f"NVFP4 projection .*missing.*{missing}"):
            Glm5NextForCausalLM._fill_module(
                None,
                projection,
                "projection",
                "",
                [(key, key) for key in weights],
                weights,
                torch.device("cpu"),
            )
        load.assert_not_called()


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
    assert allocate.call_args.kwargs["conv_state_layout"] == "q_k_v"
    assert "kda_replay_num_spec" not in allocate.call_args.kwargs
    manager = object.__new__(Glm5NextCacheManager)
    manager._requested_num_spec = None
    manager._requested_draft_records = False
    manager._kda_replay = None
    manager._mamba_ssm_stochastic_rounding = False
    manager.spec_config = spec_config
    manager._state_layout = SimpleNamespace(
        spec_config=spec_config,
        mamba_pp_layers=(),
        conv_state_layout="q_k_v",
        conv_section_dims=(),
    )
    with patch(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.is_kda_mtp_verify_available",
        return_value=verify_kernel is True,
    ):
        state = manager._initialize_spec_state()
    assert manager.use_kda_replay_update is (verify_kernel is True)
    if verify_kernel is True:
        assert state.num_speculative_tokens == 3


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
@pytest.mark.parametrize("encoder_cuda_graph", [False, True], ids=["eager", "cuda-graph"])
@torch.inference_mode()
def test_vision_attention_dp_is_local_and_matches_single_rank(encoder_cuda_graph: bool) -> None:
    config = _config()
    reference = Glm5NextVisionModelBase(ModelConfig(pretrained_config=config)).cuda().eval()
    reference.visual.setup_attn_metadata(max_num_tokens=128)
    torch.manual_seed(7)
    for parameter in reference.parameters():
        parameter.normal_(0, 0.05)
    multimodal_config = (
        MultimodalConfig(
            encoder_cuda_graph={
                "image": MultimodalEncoderCudaGraphConfig(buckets=[(16, 1), (32, 1)])
            }
        )
        if encoder_cuda_graph
        else None
    )

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
                Glm5NextVisionModelBase(
                    ModelConfig(
                        pretrained_config=config,
                        mapping=mapping,
                        multimodal_config=multimodal_config,
                    )
                )
                .cuda()
                .eval()
            )
            local.load_state_dict(reference.state_dict())
            local.visual.setup_attn_metadata(max_num_tokens=128)
            setup_mm_encoder_cuda_graphs(local)
            if not width:
                continue
            pixels = torch.randn(4 * width, 3 * 2 * 2 * 2, device="cuda", dtype=torch.bfloat16)
            grid = torch.tensor([[1, 4, width]])
            expected = reference.encode_batched(pixels, grid)
            with patch.object(
                local.visual, "_run_blocks_eager", wraps=local.visual._run_blocks_eager
            ) as eager_blocks:
                actual = local.encode_batched(pixels, grid)
            if encoder_cuda_graph:
                eager_blocks.assert_not_called()
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
        prepared = object.__new__(Glm5NextMamba2Metadata)
        prepared.glm_block_tables = tables
        for length in lengths:
            if cached and discard_prefix:
                latent.zero_()
                index.zero_()
            metadata = SimpleNamespace(
                kv_cache_manager=manager,
                seq_lens=torch.tensor([length]),
                num_contexts=1,
                kv_lens_cuda=torch.tensor([cached + length], device="cuda", dtype=torch.int32),
                mamba_metadata=prepared,
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
def test_sparse_prefill_bounds_pool_scoring_by_largest_visible_length():
    """forward_prefill bounds pool scoring by the largest cached + new length.

    Request 0 continues a cached prefix with a shorter chunk than request 1, so a
    bound that ignored cached tokens would hide visible pools from top-k.
    """
    cu_seqlens, cached_lens = [0, 10, 70], [100, 0]
    tokens, heads, head_dim = cu_seqlens[-1], 2, 4
    select = Mock(return_value=torch.zeros(tokens, 8))
    attention = SimpleNamespace(
        indexer=SimpleNamespace(index_kpool=4, index_kpool_compress_ape=None),
        attn_backend=Mock(),
        _project_attention_and_indexer=Mock(
            return_value=(torch.zeros(tokens, 3), None, None, None)
        ),
        q_b_proj=lambda x: torch.zeros(x.shape[0], heads * head_dim),
        num_heads=heads,
        qk_head_dim=head_dim,
        _select_and_attend_paged=select,
    )
    Glm5NextSparseAttention.forward_prefill(
        attention, torch.zeros(tokens, 8), cu_seqlens, cached_lens, SimpleNamespace()
    )
    select.assert_called_once()
    visible = select.call_args.args[3]
    max_visible = select.call_args.kwargs["max_visible"]
    assert isinstance(max_visible, int)
    assert max_visible == int(visible.max()) == 110


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


@pytest.mark.cpu_only
def test_indexer_topk_without_cute_dsl(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch import cute_dsl_utils
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextIndexer

    monkeypatch.setattr(cute_dsl_utils, "IS_CUTLASS_DSL_AVAILABLE", False)
    with torch.device("meta"):
        indexer = Glm5NextIndexer(_config().text_config, layer_idx=0)
    scores = torch.arange(520, dtype=torch.float32).repeat(3, 1)
    lengths = torch.tensor([0, 3, 513], dtype=torch.int32)
    selected = torch.empty(3, indexer.select_k, dtype=torch.int32)
    # GLM calls this entry for both packed prefill rows and generation rows.
    indexer.pool_top_k(
        scores, selected, is_prefill=False, sequence_lengths=lengths, scan_lengths=lengths
    )
    for row, count in enumerate(lengths.tolist()):
        valid = selected[row][selected[row] >= 0].sort().values
        expected = torch.arange(max(0, count - indexer.select_k), count, dtype=torch.int32)
        assert torch.equal(valid, expected)
        assert int((selected[row] == -1).sum()) == indexer.select_k - min(count, indexer.select_k)
