import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModel,
    _qwen3vl_extract_items,
)
from tensorrt_llm._torch.models.multimodal_encoding import MixedModalityAssembly, ModalityItem
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs.multimodal import MultimodalParams

QWEN3_VL_8B_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        # NOTE: Only 4 layers for testing, 36 layers for full model.
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "_attn_implementation": "flash_attention_2",
    "_name_or_path": str(os.path.join(llm_models_root(), "Qwen3", "Qwen3-VL-8B-Instruct")),
}


class _FakeQwenVisual(torch.nn.Module):
    """Minimal stand-in for the real Qwen3VL visual encoder.

    `Qwen3VisionModelBase._encode_visual_inputs` calls `self.visual(pixel_values,
    grid_thw=...)` and concatenates `[embeds] + deepstack_embeds` along dim=1.
    This fake returns the pixel values verbatim as `embeds` plus a single
    deepstack level (`embeds + 100`), so the post-merger hidden width is 2.
    It carries a dummy parameter so `_plan_device` (which reads
    `next(self.visual.parameters()).device`) resolves to a real device.
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so `_plan_device()` can read a device off the encoder.
        self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, pixel_values, grid_thw=None):
        del grid_thw
        base_embeds = pixel_values.to(torch.float32)
        return base_embeds, [base_embeds + 100]


def _make_qwen_vision_model_for_mixed_tests():
    model = Qwen3VisionModelBase.__new__(Qwen3VisionModelBase)
    torch.nn.Module.__init__(model)
    model.model_dtype = torch.float32
    model.visual = _FakeQwenVisual()
    # `_plan_hidden_dim` = out_hidden_size * (1 + num_deepstack_levels). The fake
    # emits 1 base column + 1 deepstack level, so out_hidden_size=1 and a single
    # deepstack index gives the expected concatenated width of 2.
    model.config = SimpleNamespace(
        spatial_merge_size=2,
        out_hidden_size=1,
        deepstack_visual_indexes=[0],
    )
    return model


def _qwen_image_video_param(include_order=True, include_lengths=True):
    """A mixed image+video param exercising the assembly-based encode path.

    The assembly extractor (`_qwen3vl_extract_items`) emits one item per modality and
    orders them by `multimodal_item_order` (tuple form `(modality, index)`, which
    is what `MultimodalPromptOrder` normalizes to). Per-item token counts come from the
    explicit `num_tokens` payload key (test convention; mirrors the Task 13
    extractor tests) so the assembled output is deterministic.
    """
    multimodal_data = {
        "image": {
            "pixel_values": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            # ONE image sub-grid whose post-merge row count is the source of
            # truth: t*(h//m)*(w//m) = 1*(4//2)*(4//2) = 4, matching the 4-row
            # pixel_values and num_tokens. (A 2-row grid would describe TWO
            # images, but this param carries a single `(image, 0)` item.)
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "num_tokens": 4,
        },
        "video": {
            "pixel_values_videos": torch.tensor([[10.0], [11.0], [12.0]]),
            "video_grid_thw": torch.tensor([[3, 2, 2]]),
            "num_tokens": 3,
        },
    }
    if include_order:
        # Prompt order: video item first, then image item.
        multimodal_data["multimodal_item_order"] = [("video", 0), ("image", 0)]
    if not include_lengths:
        # Drop the explicit per-item counts so the token-count fallback chain has
        # nothing to resolve (no num_tokens, no embedding lengths, no runtime).
        del multimodal_data["image"]["num_tokens"]
        del multimodal_data["video"]["num_tokens"]
        multimodal_data["multimodal_embedding_lengths"] = None
    return MultimodalParams(multimodal_data=multimodal_data)


def test_qwen3vl_mixed_image_video_encoder_returns_single_tensor_in_prompt_order():
    model = _make_qwen_vision_model_for_mixed_tests()

    embeddings = model.forward([_qwen_image_video_param()])

    # `forward` returns a single assembled tensor (the cache contract) whose rows
    # are laid out in MultimodalPromptOrder order: the video item (3 rows) precedes the
    # image item (4 rows), even though the extractor walks modalities image-first.
    assert len(embeddings) == 1
    expected = torch.tensor(
        [
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
        ]
    )
    torch.testing.assert_close(embeddings[0], expected)


def test_qwen3vl_mixed_image_video_requires_item_order_metadata():
    """A mixed request needs ordering metadata that disambiguates item slots.

    The extractor reads `multimodal_item_order` as `(modality, index)` pairs to
    assign each modality item a distinct `prompt_pos`. When the ordering does
    not disambiguate the items - so the image and video items collapse onto the
    same prompt slot - the assembly build in `MixedModalityAssembly.from_params` rejects it with
    a duplicate-`prompt_pos` error rather than silently overlapping their embedding
    ranges. This pins the invariant that a valid per-item ordering is required for
    a mixed request; the encoder does not fabricate one.
    """

    def _collapsing_extract(param_idx, p):
        # Model an order that fails to disambiguate the two mixed items by forcing
        # both onto prompt slot 0 (what an absent/degenerate order would yield).
        # `mm_idx_per_modality` stays distinct per item so the collision trips the
        # per-param `prompt_pos` uniqueness check rather than the modality-group one.
        for item in _qwen3vl_extract_items(param_idx, p):
            yield ModalityItem(
                src_param_idx=item.src_param_idx,
                modality=item.modality,
                mm_idx_per_modality=item.mm_idx_per_modality,
                prompt_pos=0,
                rows=item.rows,
                payload=item.payload,
            )

    with pytest.raises(ValueError, match="duplicate prompt_pos"):
        MixedModalityAssembly.from_params([_qwen_image_video_param()], _collapsing_extract)


def test_qwen3vl_mixed_image_video_requires_embedding_length_metadata():
    """Without any token-count source the assembly build cannot size a mixed item.

    Token counts come from `num_tokens`, then `multimodal_embedding_lengths`
    indexed by prompt position, then `multimodal_runtime.total_embeds_in_request`.
    When none is available the extractor's fallback raises, surfacing during assembly
    build (here exercised directly via `MixedModalityAssembly.from_params`). The message
    enumerates `multimodal_embedding_lengths` as one of the missing sources.
    """
    param = _qwen_image_video_param(include_lengths=False)

    with pytest.raises(ValueError, match="multimodal_embedding_lengths"):
        MixedModalityAssembly.from_params([param], _qwen3vl_extract_items)


def test_qwen3vl_video_token_count_sums_multirow_grid():
    processor = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))

    num_tokens = processor.get_num_tokens_per_video(
        video=[],
        video_grid_thw=torch.tensor([[1, 4, 4], [2, 4, 4]]),
    )

    assert num_tokens == 12


def test_qwen3vl_item_order_from_prompt_handles_image_video_interleave():
    processor = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)

    item_order = processor._get_mm_item_order_from_text(
        "v <|vision_start|><|video_pad|><|vision_end|> i <|vision_start|><|image_pad|><|vision_end|>",
        {"image": [object()], "video": [object()]},
    )

    assert item_order == [("video", 0), ("image", 0)]


@dataclass(repr=False)
class TestQwen3VLScenario(MultimodalScenario):
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


class TestQwen3VL(TestModelingMultimodal):
    def get_model_config(self):
        """Return the model configuration dictionary."""
        return QWEN3_VL_8B_CONFIG

    def get_trtllm_model_class(self):
        return Qwen3VLModel

    def get_hf_model_class(self):
        return HFQwen3VLForConditionalLM

    def get_weight_mapper_class(self):
        return Qwen3VLHfWeightMapper

    def get_model_type(self):
        return "qwen3_vl"

    def get_model_config_class(self):
        return Qwen3VLConfig

    def get_trtllm_inputs(
        self,
        input_ids,
        multimodal_params_list,
        is_gen: bool = False,
        num_cached_tokens_per_seq: List[int] = None,
        total_prompt_len: Optional[int] = None,
    ):
        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len,
        )

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                )
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids, dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (
                (trtllm_inputs["position_ids"] + mrope_gen_position_ids).expand(3, -1, 1).cuda()
            )
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"],
                )
                gen_multimodal_params_list.append(multimodal_param)
            trtllm_inputs["multimodal_params"] = gen_multimodal_params_list
        else:
            # Mrope position ids
            mrope_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                )
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids

        return trtllm_inputs

    def init_kv_cache_manager(self, scenario: TestQwen3VLScenario):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config["tokens_per_block"]
        max_seq_len = cache_config["max_seq_len"]
        batch_size = cache_config["batch_size"]

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        self.kv_cache_manager.add_dummy_requests(
            request_ids=[1],
            token_nums=[max_seq_len],
            # NOTE: Qwen3-VL model uses mrope
            use_mrope=True,
        )

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            # NOTE: Qwen3-VL model uses mrope
            graph_runner = create_mock_cuda_graph_runner(1, True)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"
            ].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs,
            )
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key, current_inputs=trtllm_inputs)
            return logits.clone()

    def get_scenarios(self) -> List[TestQwen3VLScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="video",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="multiple_image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # ==== CUDA Graph Scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=True,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
        ]
        # Paged context FMHA (triggered by chunked_prefill / kv_cache_reuse)
        # is forced on for correctness on Hopper (SM90); the trtllm-gen
        # kernel set on Blackwell (SM100) falls back to an unfused MHA path
        # whose output diverges from the non-paged context kernel. Gate
        # those scenarios to SM90 until the Blackwell fallback matches.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.extend(
                [
                    # ==== Chunked Prefill Scenarios ====
                    TestQwen3VLScenario(
                        modality="image",
                        use_cuda_graph=False,
                        disable_fuse_rope=False,
                        chunked_prefill=True,
                        kv_cache_reuse=False,
                    ),
                    # ==== KV Cache Reuse Scenarios ====
                    TestQwen3VLScenario(
                        modality="image",
                        use_cuda_graph=False,
                        disable_fuse_rope=False,
                        chunked_prefill=False,
                        kv_cache_reuse=True,
                    ),
                ]
            )
        # ==== Disable fuse rope scenarios ====
        # Run last: setup_scenario rebuilds trtllm_model with
        # disable_fuse_rope=True for this scenario, and the rebuild is not
        # undone afterwards. Keeping it at the tail prevents the rebuilt
        # model from leaking into chunked-prefill / kv-cache-reuse
        # scenarios (where it surfaces as a cos/sin vs. q/k seq-len
        # mismatch in MRotaryEmbedding.forward).
        scenarios.append(
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            )
        )
        return scenarios

    def get_hf_inputs(self, modality: str, prompt, media):
        processor_inputs = super().get_hf_inputs(modality, prompt, media)

        # For video: the parent class already deleted mm_token_type_ids, which
        # causes HF to fall back to 1D position IDs (no MRope). Qwen3VL encodes
        # video frames as separate timestamp-separated vision segments, so the
        # correct approach is to compute 3D MRope position IDs via TRT-LLM's
        # get_rope_index (which expands video_grid_thw per-frame) and pass them
        # explicitly to HF's forward(), bypassing compute_3d_position_ids.
        if modality == "video" and "video_grid_thw" in processor_inputs:
            position_ids, _ = Qwen3VLInputProcessorBase.get_rope_index(
                self.hf_config,
                processor_inputs["input_ids"],
                image_grid_thw=processor_inputs.get("image_grid_thw"),
                video_grid_thw=processor_inputs["video_grid_thw"],
                attention_mask=processor_inputs.get("attention_mask"),
            )
            processor_inputs["position_ids"] = position_ids.to(processor_inputs["input_ids"].device)

        return processor_inputs

    def setup_scenario(self, scenario: TestQwen3VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True,
            )
