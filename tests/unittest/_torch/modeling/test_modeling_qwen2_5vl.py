# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Type

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen2_5_VLConfig
from transformers import \
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalLM
from transformers import Qwen2VLConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen2vl_weight_mapper import \
    Qwen2VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5_VLModel, Qwen2VisionModelBase, Qwen2VLInputProcessorBase,
    Qwen2VLModel, _prepare_qwen_vl_mrope_config)
from tensorrt_llm._torch.models.modeling_qwen3vl import \
    Qwen3VLInputProcessorBase
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs.multimodal import MultimodalParams

QWEN2_5_VL_7B_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "vision_token_id": 151654,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 128000,
    "max_window_layers": 28,
    "model_type": "qwen2_5_vl",
    "num_attention_heads": 28,
    "num_hidden_layers":
    2,  # NOTE: Only 1 layer for testing, 28 layers for full model
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": True,
    "use_sliding_window": False,
    "vision_config": {
        "depth":
        2,  # NOTE: Only 8 layers for testing, 32 layers for full model. At least 8 layer needed for global Attention
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_chans": 3,
        "out_hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [0],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    },
    "rope_scaling": {
        "type": "mrope",
        "mrope_section": [16, 24, 24]
    },
    "vocab_size": 152064,
    # "_attn_implementation": "flash_attention_2",
    "_name_or_path":
    str(os.path.join(llm_models_root(), "Qwen2.5-VL-7B-Instruct"))
}


@dataclass(repr=False)
class TestQwen2_5_VLScenario(MultimodalScenario):

    disable_fuse_rope: bool = False

    def __repr__(self) -> str:
        """Generate a human-readable string representation of the scenario."""
        features = []
        features.append(f"modality:{self.modality.lower()}")
        if self.use_cuda_graph:
            features.append("cuda_graph")
        if self.disable_fuse_rope:
            features.append("no_fuse_rope")
        if self.chunked_prefill:
            features.append("chunked_prefill")
        if self.kv_cache_reuse:
            features.append("kv_cache_reuse")
        return "-".join(features)


class TestQwen2_5_VL(TestModelingMultimodal):

    def get_model_config(self):
        """Return the model configuration dictionary."""
        return QWEN2_5_VL_7B_CONFIG

    def get_trtllm_model_class(self):
        return Qwen2_5_VLModel

    def get_hf_model_class(self):
        return HFQwen2_5_VLForConditionalLM

    def get_weight_mapper_class(self):
        return Qwen2VLHfWeightMapper

    def get_model_type(self):
        return "qwen2_5_vl"

    def get_model_config_class(self):
        return Qwen2_5_VLConfig

    def get_trtllm_inputs(self,
                          input_ids,
                          multimodal_params_list,
                          is_gen: bool = False,
                          num_cached_tokens_per_seq: List[int] = None,
                          total_prompt_len: Optional[int] = None):

        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len)

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]
                    ["mrope_position_deltas"])
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids,
                                               dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (trtllm_inputs["position_ids"] +
                                             mrope_gen_position_ids).expand(
                                                 3, -1, 1).cuda()
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"])
                gen_multimodal_params_list.append(multimodal_param)
            trtllm_inputs["multimodal_params"] = gen_multimodal_params_list
            trtllm_inputs["mrope_delta_read_seq_slots"] = torch.arange(
                len(multimodal_params_list),
                device=self.device,
                dtype=torch.long)
        else:
            # Mrope position ids. For chunked prefill / KV cache reuse we must
            # mirror production `PyTorchModelEngine` behavior and slice the
            # request's full `mrope_position_ids` to the current chunk's
            # range -- the model now indexes mrope cos/sin by batch-flat
            # per-token index, so the position_ids tensor must contain only
            # the tokens of the current forward (chunk-local) with their
            # correct (T, H, W) values.
            chunk_len = input_ids.shape[-1]
            if num_cached_tokens_per_seq is None:
                begin_offsets = [0] * len(multimodal_params_list)
            elif isinstance(num_cached_tokens_per_seq, int):
                begin_offsets = [num_cached_tokens_per_seq
                                 ] * len(multimodal_params_list)
            else:
                begin_offsets = list(num_cached_tokens_per_seq)
            mrope_position_ids = []
            for multimodal_param, begin in zip(multimodal_params_list,
                                               begin_offsets):
                full_mrope = multimodal_param.multimodal_data["mrope_config"][
                    "mrope_position_ids"]
                mrope_position_ids.append(full_mrope[:, :,
                                                     begin:begin + chunk_len])
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids
            trtllm_inputs["mrope_delta_write_seq_slots"] = torch.arange(
                len(multimodal_params_list),
                device=self.device,
                dtype=torch.long)

        return trtllm_inputs

    def init_kv_cache_manager(self, scenario: TestQwen2_5_VLScenario):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen2.5-VL model."""
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config['tokens_per_block']
        max_seq_len = cache_config['max_seq_len']
        batch_size = cache_config['batch_size']

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        self.kv_cache_manager.add_dummy_requests(
            request_ids=[1],
            token_nums=[max_seq_len],
            # NOTE: Qwen2.5-VL model uses mrope
            use_mrope=True)

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen2.5-VL model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            # NOTE: Qwen2.5-VL model uses mrope
            graph_runner = create_mock_cuda_graph_runner(1, use_mrope=True)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs)
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key,
                                             current_inputs=trtllm_inputs)
            return logits.clone()

    def get_scenarios(self) -> List[TestQwen2_5_VLScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
            TestQwen2_5_VLScenario(modality="video",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
            TestQwen2_5_VLScenario(modality="multiple_image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),

            # ==== CUDA Graph Scenarios ====
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=True,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
        ]
        # Paged context FMHA (triggered by chunked_prefill / kv_cache_reuse)
        # is forced on for correctness on Hopper (SM90). On Blackwell (SM100)
        # the trtllm-gen kernel set falls back to an unfused MHA path whose
        # output diverges from the non-paged context kernel; gate those
        # scenarios to SM90 until the Blackwell fallback matches.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.extend([
                # ==== Chunked Prefill Scenarios ====
                TestQwen2_5_VLScenario(modality="image",
                                       use_cuda_graph=False,
                                       disable_fuse_rope=False,
                                       chunked_prefill=True,
                                       kv_cache_reuse=False),
                # ==== KV Cache Reuse Scenarios ====
                TestQwen2_5_VLScenario(modality="image",
                                       use_cuda_graph=False,
                                       disable_fuse_rope=False,
                                       chunked_prefill=False,
                                       kv_cache_reuse=True),
            ])
        # ==== Disable fuse rope scenarios ====
        # Run last: setup_scenario rebuilds trtllm_model with
        # disable_fuse_rope=True for this scenario, and the rebuild is not
        # undone afterwards. Keeping it at the tail prevents the rebuilt
        # model from leaking into chunked-prefill / kv-cache-reuse
        # scenarios (where it surfaces as a cos/sin vs. q/k seq-len
        # mismatch in MRotaryEmbedding.forward).
        scenarios.append(
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=True,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False))
        return scenarios

    def get_hf_inputs(self, modality: str, prompt, media):
        processor_inputs = super().get_hf_inputs(modality, prompt, media)

        # HF transformers 5.x uses a different get_rope_index algorithm for Qwen2.5-VL:
        # it multiplies temporal positions by tokens_per_second, which diverges from
        # TRT-LLM's algorithm. Compute position IDs using TRT-LLM's get_rope_index and
        # pass them explicitly so both models use the same position IDs.
        has_vision = "image_grid_thw" in processor_inputs or "video_grid_thw" in processor_inputs
        if has_vision:
            position_ids, _ = Qwen2VLInputProcessorBase.get_rope_index(
                self.hf_config,
                processor_inputs["input_ids"],
                image_grid_thw=processor_inputs.get("image_grid_thw"),
                video_grid_thw=processor_inputs.get("video_grid_thw"),
                attention_mask=processor_inputs.get("attention_mask"),
                second_per_grid_ts=processor_inputs.get("second_per_grid_ts"),
            )
            processor_inputs["position_ids"] = position_ids.to(
                processor_inputs["input_ids"].device)
            processor_inputs.pop("mm_token_type_ids", None)

        return processor_inputs

    def setup_scenario(self, scenario: TestQwen2_5_VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True)


class _FakeRotaryEmbedding:

    def __init__(self, rotary_dim: int):
        self.rotary_dim = rotary_dim

    def get_cos_sin(self, position_ids):
        num_tokens = position_ids.shape[-1]
        cos = torch.arange(num_tokens * self.rotary_dim,
                           device=position_ids.device,
                           dtype=torch.float32).reshape(1, num_tokens,
                                                        self.rotary_dim)
        return cos, cos + 1000


def test_qwen2_vision_adapter_extracts_pooler_output():
    """The real Transformers 5.x Qwen2 encoder wraps LLM features in pooler_output."""
    config = Qwen2VLConfig(dtype="float32",
                           vision_config={
                               "depth": 1,
                               "embed_dim": 32,
                               "hidden_size": 64,
                               "num_heads": 4,
                               "mlp_ratio": 2,
                               "patch_size": 2,
                               "spatial_merge_size": 2,
                               "temporal_patch_size": 1,
                               "in_channels": 3,
                           })
    vision_model = Qwen2VisionModelBase(
        ModelConfig(pretrained_config=config),
        Qwen2VisionTransformerPretrainedModel).eval()
    # Set to eager to make the test CPU-only.
    vision_model.visual.config._attn_implementation = "eager"
    multimodal_params = [
        MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": torch.randn(16, 12),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }
            })
    ]
    image = multimodal_params[0].multimodal_data["image"]
    encoder_output = vision_model.visual(image["pixel_values"],
                                         grid_thw=image["image_grid_thw"])

    result = vision_model(multimodal_params)

    assert isinstance(encoder_output, BaseModelOutputWithPooling)
    assert len(result) == 1
    torch.testing.assert_close(result[0], encoder_output.pooler_output)


def test_qwen2_vl_device_paths_move_mrope_tensors_to_cuda():
    multimodal_params = MultimodalParams(
        multimodal_data={
            "mrope_config": {
                "mrope_position_ids":
                torch.arange(6, dtype=torch.int32).reshape(3, 1, 2),
                "mrope_position_deltas":
                torch.tensor([[2]], dtype=torch.int32),
            }
        })
    device_paths = Qwen2VLModel.multimodal_data_device_paths.fget(None)
    multimodal_params.to_device("multimodal_data",
                                "cuda",
                                target_keywords=device_paths)

    mrope_config = multimodal_params.multimodal_data["mrope_config"]

    assert mrope_config["mrope_position_ids"].is_cuda
    assert mrope_config["mrope_position_deltas"].is_cuda


def _mrope_param(delta: int) -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={
            "mrope_config": {
                "mrope_position_deltas":
                torch.tensor([delta], device="cuda", dtype=torch.int32)
            }
        })


def test_prepare_qwen_vl_mrope_config_mixed_context_generation():
    rotary_dim = 2
    num_tokens = 5
    position_ids = torch.arange(3 * num_tokens,
                                device="cuda",
                                dtype=torch.int32).reshape(3, 1, num_tokens)
    mrope_position_deltas_cache = torch.zeros(8,
                                              device="cuda",
                                              dtype=torch.int32)
    mrope_rotary_cos_sin_workspace = torch.empty((1, 8 * rotary_dim * 2),
                                                 device="cuda",
                                                 dtype=torch.float32)

    config = _prepare_qwen_vl_mrope_config(
        multimodal_params=[_mrope_param(11), _mrope_param(22)],
        num_generation_requests=2,
        position_ids=position_ids,
        rotary_emb=_FakeRotaryEmbedding(rotary_dim),
        mrope_position_deltas_cache=mrope_position_deltas_cache,
        mrope_rotary_cos_sin_workspace=mrope_rotary_cos_sin_workspace,
        mrope_delta_write_seq_slots=torch.tensor([2, 5],
                                                 device="cuda",
                                                 dtype=torch.long),
        mrope_delta_read_seq_slots=torch.tensor([2, 5],
                                                device="cuda",
                                                dtype=torch.long))

    torch.testing.assert_close(
        mrope_position_deltas_cache[[2, 5]],
        torch.tensor([11, 22], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(
        config["mrope_position_deltas"],
        torch.tensor([[11], [22]], device="cuda", dtype=torch.int32))

    packed = config["mrope_rotary_cos_sin"].view(1, num_tokens, rotary_dim, 2)
    cos, sin = _FakeRotaryEmbedding(rotary_dim).get_cos_sin(position_ids)
    torch.testing.assert_close(packed[..., 0], cos)
    torch.testing.assert_close(packed[..., 1], sin)


def test_prepare_qwen_vl_mrope_config_pure_generation_reads_deltas_only():
    rotary_dim = 2
    position_ids = torch.zeros((3, 1, 2), device="cuda", dtype=torch.int32)
    mrope_position_deltas_cache = torch.zeros(8,
                                              device="cuda",
                                              dtype=torch.int32)
    mrope_position_deltas_cache[[2, 5]] = torch.tensor([11, 22],
                                                       device="cuda",
                                                       dtype=torch.int32)
    mrope_rotary_cos_sin_workspace = torch.empty((1, 8 * rotary_dim * 2),
                                                 device="cuda",
                                                 dtype=torch.float32)

    config = _prepare_qwen_vl_mrope_config(
        multimodal_params=[],
        num_generation_requests=2,
        position_ids=position_ids,
        rotary_emb=_FakeRotaryEmbedding(rotary_dim),
        mrope_position_deltas_cache=mrope_position_deltas_cache,
        mrope_rotary_cos_sin_workspace=mrope_rotary_cos_sin_workspace,
        mrope_delta_read_seq_slots=torch.tensor([2, 5],
                                                device="cuda",
                                                dtype=torch.long))

    assert "mrope_rotary_cos_sin" not in config
    torch.testing.assert_close(
        config["mrope_position_deltas"],
        torch.tensor([[11], [22]], device="cuda", dtype=torch.int32))


# ---------------------------------------------------------------------------
# Deterministic dummy-input sizing (Qwen2/2.5/3-VL input processors).
#
# CPU-only unit tests that reach into the InputProcessorBase classes directly
# (no model load) and stub just enough of the HF config so the encoder-side
# ``_num_vision_tokens`` / ``get_size_for_max_tokens`` / ``get_dummy_mm_data_*``
# math runs. Token unit is pre-merger encoder attention, matching
# ``encoder_max_num_tokens``.
# ---------------------------------------------------------------------------
def _make_dummy_processor(
    processor_cls: Type,
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 1 << 30,
):
    """Construct a processor stub with stubbed vision_config attrs.

    Bypasses the real ``__init__`` (which loads tokenizers/processors) and pins
    just the fields the deterministic math reads. ``min_pixels`` / ``max_pixels``
    stub the HF image processor's ``size`` config that ``_num_vision_tokens``
    reads for its ``smart_resize`` clamp; the default ``max_pixels`` is generous
    so the factor-pair tests aren't clamped (the clamp itself is covered by
    ``test_dummy_size_capped_at_max_pixels``).
    """
    instance = processor_cls.__new__(processor_cls)
    vision_config = SimpleNamespace(
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=3,
    )
    instance._config = SimpleNamespace(vision_config=vision_config)
    instance._processor = SimpleNamespace(image_processor=SimpleNamespace(
        size={
            "shortest_edge": min_pixels,
            "longest_edge": max_pixels
        }))
    return instance


_DUMMY_PROCESSORS = [Qwen2VLInputProcessorBase, Qwen3VLInputProcessorBase]


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
@pytest.mark.parametrize("spatial_merge_size, expected_spatial_merge_unit",
                         [(2, 4), (3, 9)])
def test_spatial_merge_unit_is_merge_size_squared(processor_cls,
                                                  spatial_merge_size,
                                                  expected_spatial_merge_unit):
    proc = _make_dummy_processor(processor_cls,
                                 spatial_merge_size=spatial_merge_size)
    assert proc.spatial_merge_unit == expected_spatial_merge_unit


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_num_vision_tokens_matches_manual_grid(processor_cls):
    proc = _make_dummy_processor(processor_cls,
                                 patch_size=16,
                                 spatial_merge_size=2,
                                 temporal_patch_size=2)
    # 224x224 image, patch=16, merge=2 -> resize-to-multiple-of-(16*2=32)
    # 224/32 -> rounds to 224 (already a multiple), grid 14*14 = 196.
    assert proc._num_vision_tokens(width=224, height=224) == 14 * 14


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_num_vision_tokens_rounds_to_unit(processor_cls):
    proc = _make_dummy_processor(processor_cls,
                                 patch_size=16,
                                 spatial_merge_size=2,
                                 temporal_patch_size=2)
    # 220x220 -> rounds half-up to nearest multiple of 32 = 224 -> 14*14.
    assert proc._num_vision_tokens(width=220, height=220) == 14 * 14


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_num_vision_tokens_temporal_padding(processor_cls):
    proc = _make_dummy_processor(processor_cls,
                                 patch_size=16,
                                 spatial_merge_size=2,
                                 temporal_patch_size=2)
    # Single frame still pads to temporal_patch_size, grid_t = 1.
    single = proc._num_vision_tokens(width=224, height=224, num_frames=1)
    three = proc._num_vision_tokens(width=224, height=224, num_frames=3)
    # 3 frames -> pad to 4 -> grid_t = 2.
    assert three == 2 * single


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
@pytest.mark.parametrize("budget", [256, 1024, 4096, 8192, 16384, 32768])
def test_invertibility_fits_within_budget(processor_cls, budget):
    proc = _make_dummy_processor(processor_cls)
    size = proc.get_size_for_max_tokens(max_tokens=budget)
    actual = proc._num_vision_tokens(width=size["width"],
                                     height=size["height"],
                                     num_frames=size["num_frames"])
    assert actual <= budget, f"Got {actual} tokens for size {size}, budget was {budget}"


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
@pytest.mark.parametrize("budget", [256, 1024, 4096, 8192, 16384, 32768])
def test_invertibility_saturates_budget(processor_cls, budget):
    """Power-of-2 budgets fit the encoder's unit grid exactly."""
    proc = _make_dummy_processor(processor_cls)
    size = proc.get_size_for_max_tokens(max_tokens=budget)
    actual = proc._num_vision_tokens(width=size["width"],
                                     height=size["height"],
                                     num_frames=size["num_frames"])
    # Budgets above are all powers of 2 with merge_size=2 squared as a factor,
    # so saturation should be exact.
    assert actual == budget


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_aspect_ratio_is_bounded(processor_cls):
    """The returned size stays within the model's aspect bound (200x)."""
    proc = _make_dummy_processor(processor_cls)
    size = proc.get_size_for_max_tokens(max_tokens=1_000_000)
    long_edge = max(size["width"], size["height"])
    short_edge = min(size["width"], size["height"])
    assert long_edge / short_edge <= 200, size


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_dummy_size_capped_at_max_pixels(processor_cls):
    """A single image cannot exceed the processor's ``max_pixels``."""
    max_pixels = 512 * 512
    proc = _make_dummy_processor(processor_cls, max_pixels=max_pixels)
    size = proc.get_size_for_max_tokens(max_tokens=1_000_000)

    # The chosen image must fit within max_pixels and round-trip exactly, so the
    # realized token count is the single-image max, below the huge budget.
    assert size["width"] * size["height"] <= max_pixels, size
    actual = proc._num_vision_tokens(width=size["width"],
                                     height=size["height"],
                                     num_frames=size["num_frames"])
    assert 0 < actual < 1_000_000


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_get_size_rejects_non_positive_budget(processor_cls):
    proc = _make_dummy_processor(processor_cls)
    with pytest.raises(ValueError, match=r"max_tokens must be positive"):
        proc.get_size_for_max_tokens(max_tokens=0)
    with pytest.raises(ValueError, match=r"max_tokens must be positive"):
        proc.get_size_for_max_tokens(max_tokens=-1)


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_get_dummy_mm_data_shapes_match_token_count(processor_cls):
    """Direct tensor build: pixel_values rows and the grid_thw product match ``_num_vision_tokens`` per image."""
    proc = _make_dummy_processor(processor_cls)
    cfg = proc._config.vision_config
    width = height = 224
    per_image = proc._num_vision_tokens(width=width, height=height)
    in_dim = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

    data = proc.get_dummy_mm_data_for_size(width=width,
                                           height=height,
                                           num_images=3,
                                           dtype=torch.float32)
    image = data["image"]
    assert image["pixel_values"].shape == (3 * per_image, in_dim)
    assert image["pixel_values"].dtype == torch.float32
    assert image["image_grid_thw"].shape == (3, 3)
    # Each grid row's product equals the per-image token count.
    grid = image["image_grid_thw"]
    assert int(grid[0].prod().item()) == per_image
    assert torch.equal(grid[0], grid[1]) and torch.equal(grid[1], grid[2])


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_get_dummy_mm_data_single_image_default(processor_cls):
    """Defaults to a single image; grid_thw is ``[1, 3]``."""
    proc = _make_dummy_processor(processor_cls)
    data = proc.get_dummy_mm_data_for_size(width=224,
                                           height=224,
                                           dtype=torch.float16)
    assert data["image"]["image_grid_thw"].shape == (1, 3)
    assert data["image"]["pixel_values"].dtype == torch.float16


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
def test_mm_max_tokens_per_item_is_image_only(processor_cls):
    """Qwen-VL declares only ``image`` (image+video share one ViT), valued at the max single-image token count."""
    proc = _make_dummy_processor(processor_cls, max_pixels=512 * 512)
    demand = proc.get_mm_max_tokens_per_item()
    assert set(demand) == {"image"}
    # The declared per-item demand is exactly the max single-image token count.
    cap_size = proc.get_size_for_max_tokens(max_tokens=10**9)
    assert demand["image"] == proc._num_vision_tokens(width=cap_size["width"],
                                                      height=cap_size["height"])
    assert demand["image"] > 0


@pytest.mark.parametrize("processor_cls", _DUMMY_PROCESSORS)
@pytest.mark.parametrize("budget", [1024, 4096, 16384])
def test_get_dummy_mm_data_for_tokens_saturates_budget(processor_cls, budget):
    """Agnostic entry: total pre-merger patches are ``<= budget`` and within one image of it (saturates the budget)."""
    proc = _make_dummy_processor(processor_cls)
    data = proc.get_dummy_mm_data_for_tokens(
        max_tokens_per_modality={"image": budget}, dtype=torch.float32)
    grid = data["image"]["image_grid_thw"]
    total_patches = int(grid.prod(dim=1).sum().item())
    per_image = int(grid[0].prod().item())
    # pixel_values rows == total patches across all batched images.
    assert data["image"]["pixel_values"].shape[0] == total_patches
    # Saturates: within the budget, and adding one more image would exceed it.
    assert total_patches <= budget
    assert total_patches + per_image > budget
