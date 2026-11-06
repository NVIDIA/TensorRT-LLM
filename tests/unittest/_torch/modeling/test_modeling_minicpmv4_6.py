# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for MiniCPM-V 4.6 configuration and multimodal forward.

The configuration tests do not require a GPU, model weights or a native
``minicpmv4_6`` transformers release. The forward test uses a small CUDA model
with locally initialized weights. Both work with the repo's currently-pinned
``transformers==5.5.4`` as well as ``>=5.7.0``:

* ``_resolve_composite_torch_dtype`` dtype resolution.
* ``_build_minicpmv4_6_config`` / ``load_pretrained_config`` composite-config
  construction, including the regression guard for the checkpoint that declares
  **no** ``torch_dtype`` (which used to crash the hybrid mamba KV-cache paths
  with ``None.itemsize``).
* The self-contained ``MiniCPMV4_6VisionConfig`` window helpers.
* The runtime ``transformers>=5.7.0`` guard used by the input processor.

A single ``transformers>=5.7.0``-gated test asserts the native config is present
once the pin is bumped (at which point the local shim can be removed).
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers
from packaging.version import Version

from tensorrt_llm._torch.configs.minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    _build_minicpmv4_6_config,
    _resolve_composite_torch_dtype,
    load_pretrained_config,
)

_MIN_TRANSFORMERS = "5.7.0"

requires_native_minicpmv4_6 = pytest.mark.skipif(
    Version(transformers.__version__) < Version(_MIN_TRANSFORMERS),
    reason=f"native minicpmv4_6 requires transformers>={_MIN_TRANSFORMERS}",
)


def _minicpmv4_6_config_dict() -> dict:
    """A faithful, minimal ``config.json`` for openbmb/MiniCPM-V-4.6.

    Mirrors the real checkpoint: composite ``minicpmv4_6`` with a SigLIP2 vision
    tower and a Qwen3.5 dense hybrid text tower, and crucially **no**
    ``torch_dtype``/``dtype`` at any level.
    """
    return copy.deepcopy(
        {
            "architectures": ["MiniCPMV4_6ForConditionalGeneration"],
            "model_type": "minicpmv4_6",
            "drop_vision_last_layer": False,
            "image_size": 1120,
            "insert_layer_id": 6,
            "image_token_id": 248056,
            "video_token_id": 248057,
            "tie_word_embeddings": True,
            "vision_config": {
                "model_type": "minicpmv4_6_vision",
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "image_size": 980,
                "intermediate_size": 4304,
                "layer_norm_eps": 1e-06,
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 27,
                "patch_size": 14,
            },
            "text_config": {
                "model_type": "qwen3_5_text",
                "attention_bias": False,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
                "vocab_size": 248094,
                "tie_word_embeddings": True,
            },
        }
    )


# ---------------------------------------------------------------------------
# _resolve_composite_torch_dtype
# ---------------------------------------------------------------------------
class TestResolveCompositeTorchDtype:
    def test_defaults_to_bfloat16_when_absent(self):
        assert _resolve_composite_torch_dtype({}, {}) is torch.bfloat16

    def test_respects_explicit_string(self):
        assert _resolve_composite_torch_dtype({"torch_dtype": "float16"}) is torch.float16

    def test_respects_dtype_alias_key(self):
        assert _resolve_composite_torch_dtype({"dtype": "float32"}) is torch.float32

    def test_respects_torch_dtype_object(self):
        assert _resolve_composite_torch_dtype({"torch_dtype": torch.float16}) is torch.float16

    def test_auto_and_none_are_skipped(self):
        assert (
            _resolve_composite_torch_dtype(
                {"torch_dtype": "auto"},
                {"dtype": None},
                {"torch_dtype": "float16"},
            )
            is torch.float16
        )

    def test_first_declaration_wins(self):
        assert (
            _resolve_composite_torch_dtype(
                {"torch_dtype": "bfloat16"},
                {"torch_dtype": "float16"},
            )
            is torch.bfloat16
        )


# ---------------------------------------------------------------------------
# _build_minicpmv4_6_config
# ---------------------------------------------------------------------------
class TestBuildMiniCPMV46Config:
    def test_pins_bfloat16_when_checkpoint_declares_none(self):
        # Regression: the real checkpoint declares no torch_dtype anywhere, which
        # crashed the hybrid mamba KV-cache byte sizing (None.itemsize). Both the
        # composite config and the inner text tower must carry a concrete dtype.
        cfg = _build_minicpmv4_6_config(_minicpmv4_6_config_dict())
        assert cfg.torch_dtype is torch.bfloat16
        assert cfg.text_config.torch_dtype is torch.bfloat16

    def test_respects_explicit_dtype(self):
        raw = _minicpmv4_6_config_dict()
        raw["torch_dtype"] = "float16"
        cfg = _build_minicpmv4_6_config(raw)
        assert cfg.torch_dtype is torch.float16
        assert cfg.text_config.torch_dtype is torch.float16

    def test_keeps_top_level_model_type(self):
        # Needed for MULTIMODAL_PLACEHOLDER_REGISTRY lookup; otherwise requests
        # fail with "Unknown modality".
        cfg = _build_minicpmv4_6_config(_minicpmv4_6_config_dict())
        assert cfg.model_type == "minicpmv4_6"

    def test_normalizes_text_tower_to_qwen3next(self):
        cfg = _build_minicpmv4_6_config(_minicpmv4_6_config_dict())
        assert isinstance(cfg.text_config, transformers.Qwen3NextConfig)
        assert cfg.text_config.architectures == ["Qwen3_5ForCausalLM"]
        assert cfg.text_config.num_hidden_layers == 24

    def test_preserves_multimodal_token_ids(self):
        cfg = _build_minicpmv4_6_config(_minicpmv4_6_config_dict())
        assert cfg.image_token_id == 248056
        assert cfg.video_token_id == 248057

    def test_propagates_insert_layer_id_to_vision(self):
        cfg = _build_minicpmv4_6_config(_minicpmv4_6_config_dict())
        assert cfg.insert_layer_id == 6
        assert cfg.vision_config.insert_layer_id == 6
        assert cfg.vision_config.hidden_size == 1152
        assert cfg.vision_config.num_hidden_layers == 27


# ---------------------------------------------------------------------------
# load_pretrained_config routing (offline, via a temp config.json)
# ---------------------------------------------------------------------------
class TestLoadPretrainedConfigRouting:
    def _write_config(self, tmp_path) -> str:
        (tmp_path / "config.json").write_text(json.dumps(_minicpmv4_6_config_dict()))
        return str(tmp_path)

    def test_routes_minicpmv4_6_by_model_type(self, tmp_path):
        cfg = load_pretrained_config(self._write_config(tmp_path))
        assert isinstance(cfg, MiniCPMV4_6Config)
        assert cfg.model_type == "minicpmv4_6"
        assert cfg.torch_dtype is torch.bfloat16

    def test_routes_when_only_architecture_present(self, tmp_path):
        raw = _minicpmv4_6_config_dict()
        raw.pop("model_type")
        (tmp_path / "config.json").write_text(json.dumps(raw))
        cfg = load_pretrained_config(str(tmp_path))
        assert isinstance(cfg, MiniCPMV4_6Config)


# ---------------------------------------------------------------------------
# MiniCPMV4_6VisionConfig helpers
# ---------------------------------------------------------------------------
class TestVisionConfig:
    def test_window_helpers_scale_by_kernel(self):
        vc = MiniCPMV4_6VisionConfig(
            hidden_size=1152, intermediate_size=4304, window_kernel_size=(2, 2)
        )
        assert vc.window_kernel_size == (2, 2)
        assert vc.window_hidden_size == 1152 * 2 * 2
        assert vc.window_intermediate_size == 4304 * 2 * 2

    def test_window_kernel_size_is_tuple(self):
        # Accept list from JSON but expose a tuple.
        vc = MiniCPMV4_6VisionConfig(window_kernel_size=[2, 2])
        assert vc.window_kernel_size == (2, 2)


# ---------------------------------------------------------------------------
# transformers>=5.7.0 runtime guard (used by the input processor)
# ---------------------------------------------------------------------------
class TestTransformersGuard:
    @pytest.mark.parametrize("version", ["5.5.4", "5.6.9"])
    def test_raises_on_old_transformers(self, monkeypatch, version):
        from tensorrt_llm._torch.models import modeling_minicpmv4_6 as mod

        monkeypatch.setattr(transformers, "__version__", version)
        with pytest.raises(RuntimeError, match="transformers>="):
            mod._ensure_transformers_supports_minicpmv4_6()

    @pytest.mark.parametrize("version", ["5.7.0", "5.8.1", "5.12.1"])
    def test_passes_on_supported_transformers(self, monkeypatch, version):
        from tensorrt_llm._torch.models import modeling_minicpmv4_6 as mod

        monkeypatch.setattr(transformers, "__version__", version)
        mod._ensure_transformers_supports_minicpmv4_6()


# ---------------------------------------------------------------------------
# Native transformers config (only once the pin is bumped to >=5.7.0)
# ---------------------------------------------------------------------------
@requires_native_minicpmv4_6
def test_native_minicpmv4_6_config_available():
    # Once transformers>=5.7.0 is pinned, the local MiniCPMV4_6Config shim can
    # be dropped in favor of this native config.
    from transformers import MiniCPMV4_6Config as HFMiniCPMV4_6Config

    assert HFMiniCPMV4_6Config.model_type == "minicpmv4_6"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Model forward requires CUDA")
def test_minicpmv46_multimodal_forward():
    from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import MiniCPMV4_6Model
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalParams,
    )

    raw = _minicpmv4_6_config_dict()
    raw.update(
        torch_dtype="bfloat16",
        image_size=8,
        insert_layer_id=0,
        image_token_id=62,
        video_token_id=63,
    )
    raw["vision_config"].update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    raw["text_config"].update(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=32,
        full_attention_interval=1,
        layer_types=["full_attention"],
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    )
    config = _build_minicpmv4_6_config(raw)
    model_cls = get_registered_model_class("MiniCPMV4_6ForConditionalGeneration")
    assert model_cls is MiniCPMV4_6Model

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        model = (
            model_cls(
                ModelConfig(pretrained_config=config, attn_backend="VANILLA", max_num_tokens=16)
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.ndim > 1:
                    parameter.normal_(mean=0.0, std=0.05)
                elif name.startswith("mm_encoder.") and name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    # Qwen3.5 uses Gemma RMSNorm: zero weights mean unit scale.
                    parameter.zero_()
        model.llm.setup_aliases()
        assert isinstance(model, MiniCPMV4_6Model)
        assert len(model.mm_encoder.encoder.layers) == 1
        assert len(model.llm.model.layers) == 1

        metadata = VanillaAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=4,
            kv_cache_manager=None,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            num_contexts=1,
            request_ids=[1],
            kv_cache_params=KVCacheParams(use_cache=False, num_cached_tokens_per_seq=[0]),
            mamba_metadata=Mamba2Metadata(1, chunk_size=128, max_num_tokens=4),
        )
        metadata.prepare()
        input_ids = torch.tensor([65, 3, 7, 11], dtype=torch.int32, device="cuda")
        position_ids = torch.arange(4, dtype=torch.int32, device="cuda")
        pixels = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16, device="cuda")
        target_sizes = torch.tensor([[4, 4]], dtype=torch.int32)
        image = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": pixels.squeeze(0).permute(2, 0, 1),
                    "target_sizes": target_sizes,
                },
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[1]
                ),
            }
        )
        changed_image = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": -pixels.squeeze(0).permute(2, 0, 1),
                    "target_sizes": target_sizes,
                },
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[1]
                ),
            }
        )
        forward_kwargs = dict(
            attn_metadata=metadata,
            position_ids=position_ids,
            return_context_logits=True,
        )
        with torch.inference_mode():
            logits = model(input_ids=input_ids, multimodal_params=[image], **forward_kwargs)
            embeddings = model.mm_encoder([image])[0]
            attached = MultimodalParams(
                multimodal_data={
                    "image": changed_image.multimodal_data["image"],
                    "multimodal_embedding": embeddings,
                }
            )
            other_ids = input_ids.clone()
            other_ids[0] = 66
            attached_logits = model(
                input_ids=other_ids, multimodal_params=[attached], **forward_kwargs
            )
            changed_logits = model(
                input_ids=input_ids, multimodal_params=[changed_image], **forward_kwargs
            )
            text_embedding = model.llm.model.embed_tokens(input_ids[-1:])
            bypassed_logits = model.llm.lm_head(model.llm.model.norm(text_embedding)).float()

        assert logits.shape == (4, 64)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert torch.isfinite(attached_logits).all()
        assert torch.isfinite(changed_logits).all()
        torch.testing.assert_close(logits, attached_logits, rtol=0.02, atol=0.02)
        assert not torch.allclose(logits[-1:], changed_logits[-1:], rtol=1e-3, atol=1e-3)
        assert not torch.allclose(logits[-1:], bypassed_logits, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TRTLLM attention metadata requires CUDA")
def test_minicpmv46_vision_attn_capacity_is_stable_across_grids():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import MiniCPMV4_6VisionModel
    from tensorrt_llm._torch.pyexecutor.engine.multimodal import setup_mm_encoder_attn_metadata

    raw = _minicpmv4_6_config_dict()
    raw.update(torch_dtype="bfloat16", image_size=8, insert_layer_id=0)
    raw["vision_config"].update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    vision = MiniCPMV4_6VisionModel(
        ModelConfig(
            pretrained_config=_build_minicpmv4_6_config(raw),
            attn_backend="TRTLLM",
            max_num_tokens=64,
        )
    )

    # Grids of different resolutions share the configured capacity, not their own lengths.
    small = vision._make_attn_metadata([16, 4])
    large = vision._make_attn_metadata([64])
    assert small.max_seq_len == large.max_seq_len == 64
    assert small.seq_lens.tolist() == [16, 4]
    assert large.seq_lens.tolist() == [64]
    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        vision._make_attn_metadata([65])

    # The engine's encoder token budget, not the LLM limit, bounds later grids.
    setup_mm_encoder_attn_metadata(
        vision, input_processor=None, encoder_max_num_tokens=128, attention_metadata_capacity=None
    )
    grown = vision._make_attn_metadata([100])
    assert grown.max_seq_len == 128
    assert grown.seq_lens.tolist() == [100]


def _main_minicpm_forward(self, multimodal_params):
    pixel_values_list = []
    target_sizes_list = []
    for param in multimodal_params:
        # Image and video share the NaViT-packed vision path. A request
        # carries at most one modality (the input processor rejects mixed
        # image+video requests), so the fixed image-then-video iteration
        # order here is unambiguous and matches the placeholder order in
        # `input_ids`.
        for modality in ("image", "video"):
            modality_data = param.multimodal_data.get(modality)
            if modality_data is None:
                continue
            pixel_values_list.append(modality_data["pixel_values"])
            target_sizes_list.append(modality_data["target_sizes"])

    if not pixel_values_list:
        return []

    # NaViT packing: concat along the packed patch axis; target grids stack.
    pixel_values = (
        torch.cat(pixel_values_list, dim=-1) if len(pixel_values_list) > 1 else pixel_values_list[0]
    )
    target_sizes = (
        torch.cat(target_sizes_list, dim=0) if len(target_sizes_list) > 1 else target_sizes_list[0]
    )
    target_sizes = target_sizes.to("cpu")

    features = self._get_image_features(pixel_values, target_sizes)
    fused = torch.cat(features, dim=0)

    # Contract-4 invariants (this forward returns ONE pre-concatenated
    # tensor that find_input_mm_embeds / _cache_multimodal_embeddings slice
    # back per request):
    #   I.  Order -- `features` are concatenated in `multimodal_params`
    #       order; the loop above and the per-grid merger both preserve it,
    #       which is exactly the order the downstream slicers assume. (Not
    #       assertable cheaply -- covered by the multi-request E2E test.)
    #   II. Count -- the produced row count must equal the sum of each
    #       request's placeholder count (`total_embeds_in_request`). A
    #       mismatch means the vision output and the fused `input_ids` will
    #       misalign, so fail loudly here instead of silently scattering
    #       wrong rows downstream. Only checked when every request carries
    #       runtime counts (absent on some warmup / JIT paths).
    runtimes = [p.multimodal_runtime for p in multimodal_params]
    if all(r is not None and r.total_embeds_in_request is not None for r in runtimes):
        expected = sum(r.total_embeds_in_request for r in runtimes)
        assert fused.shape[0] == expected, (
            f"MiniCPMV4_6 vision produced {fused.shape[0]} embedding rows "
            f"but the batch's per-request placeholder counts sum to "
            f"{expected}; vision output would misalign with input_ids."
        )
    return [fused]


class _MiniCpmCpuEncoder:
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import MiniCPMV4_6VisionModel

    mm_encoder_groups = MiniCPMV4_6VisionModel.mm_encoder_groups
    encode_batched = MiniCPMV4_6VisionModel.encode_batched
    forward = MiniCPMV4_6VisionModel.forward

    def __init__(self):
        self.calls = []

    def _get_image_features(self, pixel_values, target_sizes):
        self.calls.append((pixel_values, target_sizes))
        packed = pixel_values.squeeze(0).permute(2, 0, 1).reshape(-1, 6)
        result = []
        offset = 0
        for height, width in target_sizes:
            count = int(height * width)
            values = packed[offset : offset + count * 2]
            result.append(values.reshape(count // 16, 32, 6).sum(dim=1))
            offset += count * 2
        return result


def _minicpm_cpu_request(modality, grids, slices_per_item):
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import (
        MiniCPMV4_6InputProcessor,
        _MiniCPMItemLengthsProcessor,
    )
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalParams,
    )

    processor = object.__new__(MiniCPMV4_6InputProcessor)
    processor._config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=32), image_token_id=21, video_token_id=22
    )
    processor._dtype = torch.float32
    processor._tokenizer = Mock()
    processor.tllm_multimodal_token_id = 33
    processor.image_token_str, processor.video_token_str = "<i>", "<v>"
    target_sizes = torch.tensor(grids, dtype=torch.int32)
    total = int(target_sizes.prod(dim=-1).sum())
    pixels = torch.arange(total * 12, dtype=torch.float32).reshape(1, 3, 2, total * 2)
    lengths = []
    start = 0
    for count in slices_per_item:
        lengths.append(int(target_sizes[start : start + count].prod(dim=-1).sum()) // 16)
        start += count
    token = processor.image_token_str if modality == "image" else processor.video_token_str
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    size_key = "target_sizes" if modality == "image" else "target_sizes_videos"
    media_inputs = {pixel_key: pixels, size_key: target_sizes}
    if modality == "image":
        media_inputs["num_patches_per_image"] = slices_per_item
    else:
        media_inputs["num_frames_per_video"] = slices_per_item
        media_inputs["num_patches_per_frame"] = [1] * len(grids)
    native_media = Mock(return_value=media_inputs, downsample_mode="16x")
    media_processor = _MiniCPMItemLengthsProcessor(native_media, modality)

    def preprocess(**kwargs):
        result = dict(media_processor(**kwargs))
        result.pop("num_patches_per_image", None)
        result.pop("num_frames_per_video", None)
        result.pop("num_patches_per_frame", None)
        result["input_ids"] = torch.tensor([[21 if modality == "image" else 22] * sum(lengths)])
        return result

    processor._processor = Mock(side_effect=preprocess)
    raw = (
        {"image": [object()] * len(lengths)}
        if modality == "image"
        else {"video": [SimpleNamespace(frames=[object()], metadata={})] * len(lengths)}
    )
    _, extra = processor.call_with_text_prompt(
        {"prompt": token * len(lengths), "multi_modal_data": raw}, None
    )
    assert "return_text_replacement_offsets" not in processor._processor.call_args.kwargs
    data = extra["multimodal_data"]
    order = [{"modality": modality, "index": index} for index in range(len(lengths))]
    metadata = processor.get_mm_item_metadata([], data, order)
    assert metadata.output_embedding_lengths == tuple(lengths)
    assert processor.get_mm_item_metadata([], data).item_refs == metadata.item_refs
    data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    param = MultimodalParams(multimodal_data=data)
    original = MultimodalParams(
        multimodal_data={modality: {"pixel_values": pixels, "target_sizes": target_sizes}}
    )
    return param, original


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["single", "images", "videos", "mixed-batch"])
def test_minicpm_declared_inputs_match_main(case):
    if case == "single":
        specs = [("image", [(4, 4)], [1])]
    elif case == "images":
        specs = [
            ("image", [(4, 4), (4, 8), (8, 4)], [2, 1]),
            ("image", [(8, 8)], [1]),
        ]
    elif case == "videos":
        specs = [
            ("video", [(4, 4), (4, 8), (8, 4)], [2, 1]),
            ("video", [(8, 8)], [1]),
        ]
    else:
        specs = [
            ("video", [(4, 4), (4, 8)], [2]),
            ("image", [(8, 4), (4, 4)], [1, 1]),
            ("video", [(8, 8)], [1]),
        ]
    pairs = [_minicpm_cpu_request(*spec) for spec in specs]
    encoder = _MiniCpmCpuEncoder()
    actual = encoder.forward([param for param, _ in pairs])
    assert len(actual) == 1
    assert len(encoder.calls) == 1
    reference_encoder = _MiniCpmCpuEncoder()
    expected = torch.cat(
        [_main_minicpm_forward(reference_encoder, [original])[0] for _, original in pairs]
    )
    assert torch.equal(actual[0], expected)
    modality_order = [
        original
        for modality in ("image", "video")
        for _, original in pairs
        if modality in original.multimodal_data
    ]
    packed_reference = _MiniCpmCpuEncoder()
    _main_minicpm_forward(packed_reference, modality_order)
    for captured, oracle in zip(encoder.calls[0], packed_reference.calls[0]):
        assert torch.equal(captured, oracle)
        assert captured.dtype == oracle.dtype
        assert captured.device == oracle.device


@pytest.mark.cpu_only
@pytest.mark.parametrize("tensor_counts", [False, True], ids=["hf57-list", "latest-tensor"])
@pytest.mark.parametrize("downsample_mode", ["4x", "16x"])
def test_minicpm_retains_video_item_lengths_before_hf_pops(tensor_counts, downsample_mode):
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import _MiniCPMItemLengthsProcessor

    inputs = {
        "target_sizes_videos": torch.tensor([[4, 4], [4, 8], [8, 4], [8, 8]]),
        "num_patches_per_frame": torch.tensor([2, 1, 1]) if tensor_counts else [2, 1, 1],
        "num_frames_per_video": [2, 1],
    }
    native = Mock(return_value=inputs, downsample_mode="16x")
    processor = _MiniCPMItemLengthsProcessor(native, "video")
    result = processor(downsample_mode=downsample_mode)
    assert processor.downsample_mode == "16x"
    result.pop("num_patches_per_frame")
    result.pop("num_frames_per_video")
    divisor = 4 if downsample_mode == "4x" else 16
    lengths = result["video_output_embedding_lengths"]
    assert torch.equal(lengths, torch.tensor([80 // divisor, 64 // divisor]))
    assert lengths.dtype == torch.int64
    assert lengths.device.type == "cpu"


@pytest.mark.cpu_only
def test_minicpm_processor_proxy_unpickles_without_recursive_attribute_lookup():
    from tensorrt_llm._torch.models.modeling_minicpmv4_6 import _MiniCPMItemLengthsProcessor

    uninitialized = object.__new__(_MiniCPMItemLengthsProcessor)
    with pytest.raises(AttributeError):
        getattr(uninitialized, "__setstate__")
