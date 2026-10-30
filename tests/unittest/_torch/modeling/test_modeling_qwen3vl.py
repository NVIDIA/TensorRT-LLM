# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from PIL import Image
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from test_modeling_qwen2vl import _make_vision_attention
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_multimodal_utils, modeling_qwen2vl, modeling_qwen3vl
from tensorrt_llm._torch.models.checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    VisionNormalization,
    normalize_vision_pixels,
    vision_normalization_from_processor,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VLInputProcessorBase,
    Qwen3VLModel,
    _triton_pos_embed_interpolate,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import MultimodalConfig, MultimodalEncoderCudaGraphConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

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
            trtllm_inputs["mrope_delta_read_seq_slots"] = torch.arange(
                len(multimodal_params_list), device=self.device, dtype=torch.long
            )
        else:
            # Mrope position ids. For chunked prefill / KV cache reuse we must
            # mirror production `ModelEngine` behavior and slice the
            # request's full `mrope_position_ids` to the current chunk's
            # range -- the model now indexes mrope cos/sin by batch-flat
            # per-token index, so the position_ids tensor must contain only
            # the tokens of the current forward (chunk-local) with their
            # correct (T, H, W) values.
            chunk_len = input_ids.shape[-1]
            if num_cached_tokens_per_seq is None:
                begin_offsets = [0] * len(multimodal_params_list)
            elif isinstance(num_cached_tokens_per_seq, int):
                begin_offsets = [num_cached_tokens_per_seq] * len(multimodal_params_list)
            else:
                begin_offsets = list(num_cached_tokens_per_seq)
            mrope_position_ids = []
            for multimodal_param, begin in zip(multimodal_params_list, begin_offsets):
                full_mrope = multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                mrope_position_ids.append(full_mrope[:, :, begin : begin + chunk_len])
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids
            trtllm_inputs["mrope_delta_write_seq_slots"] = torch.arange(
                len(multimodal_params_list), device=self.device, dtype=torch.long
            )

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

            key = KeyType(batch_size=1, draft_len=0)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen3vl_deepstack_fusion_reuses_registered_buffer():
    hidden_size = 4
    num_tokens = 5
    mm_token_id = 7
    config_dict = copy.deepcopy(QWEN3_VL_8B_CONFIG)
    config_dict.update(
        image_token_id=mm_token_id,
        video_token_id=8,
        vision_start_token_id=9,
        vision_end_token_id=10,
    )
    config_dict["text_config"].update(
        hidden_size=hidden_size,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden_size,
        vocab_size=32,
    )
    config_dict["vision_config"].update(
        deepstack_visual_indexes=[0, 1],
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=1,
        out_hidden_size=hidden_size,
    )
    model_config = ModelConfig(
        pretrained_config=Qwen3VLConfig.from_dict(config_dict),
        disable_mm_encoder=True,
        max_num_tokens=num_tokens,
    )
    model = Qwen3VLModel(model_config, disable_fuse_rope=True).to("cuda")

    input_ids = torch.tensor([1, mm_token_id, 2, mm_token_id, 3], device="cuda")
    text_token_indices = torch.tensor([0, 2, 4], device="cuda")
    mm_token_indices = torch.tensor([1, 3], device="cuda")
    primary_embeds = [
        torch.full(
            (2, hidden_size),
            10.0,
            dtype=model.embedding_dtype,
            device="cuda",
        )
    ]
    deepstack_embeds = [
        torch.full(
            (2, hidden_size),
            value,
            dtype=model.embedding_dtype,
            device="cuda",
        )
        for value in (20.0, 30.0)
    ]

    _, _, fused_deepstack = model._fuse_multimodal_embeddings(
        input_ids=input_ids,
        multimodal_embeddings=primary_embeds,
        mm_token_ids=model.multimodal_token_ids,
        embedding_layer=model.text_embedding_layer,
        extra_embeds=deepstack_embeds,
        text_token_indices=text_token_indices,
        mm_token_indices=mm_token_indices,
    )

    assert len(fused_deepstack) == 2
    # The returned per-level tensors must be views into the model-owned scratch
    # buffer, not newly allocated full-sequence tensors.
    assert (
        fused_deepstack[0].untyped_storage().data_ptr()
        == model.deepstack_input_embeds.untyped_storage().data_ptr()
    )
    # Sharing storage alone does not prove that fusion scattered each level to
    # the right token rows, so verify the multimodal positions independently.
    torch.testing.assert_close(
        fused_deepstack[0][mm_token_indices],
        deepstack_embeds[0],
    )
    torch.testing.assert_close(
        fused_deepstack[1][mm_token_indices],
        deepstack_embeds[1],
    )
    # The scratch buffer is reused across forwards; non-multimodal positions
    # must be cleared before every scatter to avoid leaking stale features.
    torch.testing.assert_close(
        model.deepstack_input_embeds[:, text_token_indices],
        torch.zeros(
            2,
            len(text_token_indices),
            hidden_size,
            dtype=model.embedding_dtype,
            device="cuda",
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen3vl_init_preserves_caller_quant_config():
    """Building Qwen3VLModel must not mutate the caller's quant_config."""
    hf_config = Qwen3VLConfig.from_dict(copy.deepcopy(QWEN3_VL_8B_CONFIG))
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        quant_config=quant_config,
        skip_create_weights_in_init=True,
    )

    model = Qwen3VLModel(model_config)

    # Outer LLM keeps the caller's quant_config unchanged (same object, same FP8 KV-cache setting).
    assert model.model_config.quant_config is quant_config
    assert model.model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

    # Vision encoder operates on an independent copy whose quant settings have been reset to the
    # defaults (no quantization).
    assert model.mm_encoder.model_config.quant_config is not quant_config
    assert model.mm_encoder.model_config.quant_config.kv_cache_quant_algo is None
    assert model.mm_encoder.model_config.quant_config.quant_algo is None


def test_qwen3_vision_prepare_metadata_passes_fixed_max_seq_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_max_seq_len = 65_536
    seq_lens = [256, 1024]
    metadata = object()
    calls = []

    def capture_prepare_attn_metadata(
        actual_seq_lens: List[int],
        actual_metadata: object,
        *,
        max_seq_len: int,
    ) -> object:
        calls.append((actual_seq_lens, actual_metadata, max_seq_len))
        return actual_metadata

    monkeypatch.setattr(
        modeling_qwen3vl,
        "prepare_encoder_attn_metadata",
        capture_prepare_attn_metadata,
    )
    model = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(model)
    model.set_attn_max_seq_len(fixed_max_seq_len)

    result = model.prepare_attn_metadata(seq_lens, metadata)

    assert result is metadata
    assert calls == [(seq_lens, metadata, fixed_max_seq_len)]


@pytest.mark.cpu_only
def test_qwen3_vision_patch_projection_matches_conv3d() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    config = SimpleNamespace(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
    )
    patch_embed = Qwen3VLVisionPatchEmbed(config)
    pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
    expected = patch_embed(pixel_values)
    # The model must reuse the Conv3d parameters as a GEMM, not run the Conv3d.
    patch_embed.proj.register_forward_pre_hook(
        lambda module, args: pytest.fail("patch projection ran the Conv3d")
    )

    vision = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(vision)
    vision.patch_embed = patch_embed
    vision.attn_metadata = object()
    vision._blocks_graph_runner = None
    vision._rope_position_ids_buffer = None
    vision.rot_pos_emb = lambda grid: (
        torch.empty(6, 0),
        torch.empty(6, 0),
    )
    vision.fast_pos_embed_interpolate = lambda grid: torch.zeros_like(expected)
    vision.prepare_attn_metadata = lambda seq_lens, metadata: metadata
    vision.blocks = torch.nn.ModuleList()
    vision._deepstack_layer_to_merger_idx = {}
    vision.merger = torch.nn.Identity()

    actual, deepstack = vision(pixel_values, torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(actual, expected)
    assert deepstack == []


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("deepstack_visual_indexes", "expected_features"), [([0], [3.0]), ([1, 0], [6.0, 4.0])]
)
def test_qwen3_vision_run_blocks_uses_encoder_graph(
    deepstack_visual_indexes: list[int], expected_features: list[float]
) -> None:
    class FakeGraphRunner:
        def __init__(self) -> None:
            self.calls = []

        def maybe_run(self, **kwargs: object) -> dict[str, torch.Tensor]:
            self.calls.append(kwargs)
            return {
                "hidden_states": torch.full((4, 8), 2.0),
                "deepstack_0": torch.full((4, 8), 3.0),
                "deepstack_1": torch.full((4, 8), 4.0),
            }

    vision = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(vision)
    vision._blocks_graph_runner = FakeGraphRunner()
    vision.deepstack_visual_indexes = deepstack_visual_indexes
    vision._deepstack_layer_to_merger_idx = {
        layer_idx: idx for idx, layer_idx in enumerate(deepstack_visual_indexes)
    }
    vision.deepstack_merger_list = torch.nn.ModuleList(
        [torch.nn.Linear(8, 8, bias=False) for _ in deepstack_visual_indexes]
    )
    with torch.no_grad():
        for idx, merger in enumerate(vision.deepstack_merger_list):
            merger.weight.copy_(torch.eye(8) * (idx + 1))
    vision.merger = torch.nn.Identity()
    metadata = SimpleNamespace(seq_lens=torch.tensor([4]))
    hidden_states = torch.ones(4, 8)
    cos = torch.ones(4, 2)
    sin = torch.zeros(4, 2)
    position_ids = torch.arange(4, dtype=torch.int32)

    output, deepstack = vision._run_blocks(hidden_states, cos, sin, position_ids, metadata)

    assert torch.equal(output, torch.full((4, 8), 2.0))
    assert len(deepstack) == len(expected_features)
    for features, expected in zip(deepstack, expected_features):
        assert torch.equal(features, torch.full((4, 8), expected))
    assert vision._blocks_graph_runner.calls == [
        {
            "seq_lengths": [4],
            "inputs": {
                "hidden_states": hidden_states,
                "cos": cos,
                "sin": sin,
                "rope_position_ids": position_ids,
            },
        }
    ]


@pytest.mark.cpu_only
def test_qwen3_vision_run_blocks_preserves_eager_merger_order() -> None:
    events = []

    class RecordingBlock(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
            events.append(self.name)
            return hidden_states

    class RecordingMerger(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append(self.name)
            return hidden_states

    vision = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(vision)
    vision._blocks_graph_runner = None
    vision.blocks = torch.nn.ModuleList([RecordingBlock("block_0"), RecordingBlock("block_1")])
    vision._deepstack_layer_to_merger_idx = {0: 0}
    vision.deepstack_merger_list = torch.nn.ModuleList([RecordingMerger("deepstack_merger")])
    vision.merger = RecordingMerger("final_merger")
    hidden_states = torch.ones(4, 8)

    vision._run_blocks(
        hidden_states,
        torch.ones(4, 2),
        torch.zeros(4, 2),
        torch.arange(4, dtype=torch.int32),
        SimpleNamespace(seq_lens=torch.tensor([4])),
    )

    assert events == ["block_0", "deepstack_merger", "block_1", "final_merger"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("attn_backend", ["TRTLLM", "FLASHINFER"])
@pytest.mark.parametrize("deepstack_visual_indexes", [[0], [1, 0]])
@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
@pytest.mark.parametrize("num_tokens", [192, 256])
def test_qwen3_vision_cuda_graph_replay_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
    attn_backend: str,
    deepstack_visual_indexes: list[int],
    automatic: bool,
    num_tokens: int,
) -> None:
    from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
        setup_mm_encoder_attn_metadata,
        setup_mm_encoder_cuda_graphs,
    )

    config_dict = copy.deepcopy(QWEN3_VL_8B_CONFIG)
    config_dict["vision_config"].update(depth=2, deepstack_visual_indexes=deepstack_visual_indexes)
    model_config = ModelConfig(
        pretrained_config=Qwen3VLConfig.from_dict(config_dict),
        attn_backend=attn_backend,
        multimodal_config=MultimodalConfig(
            encoder_cuda_graph={
                "vision": (
                    MultimodalEncoderCudaGraphConfig(max_batch_size=2)
                    if automatic
                    else MultimodalEncoderCudaGraphConfig(buckets=[(256, 1)])
                )
            }
        ),
    )
    torch.manual_seed(0)
    encoder = modeling_qwen3vl.Qwen3VisionModelBase(model_config, Qwen3VisionModel).cuda().eval()
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.0)
            else:
                param.normal_(std=0.02)
    vision = encoder.visual
    setup_mm_encoder_attn_metadata(encoder, object(), 256, {"attention": 1})
    grid_thw = torch.tensor([[1, num_tokens // 16, 16]])
    pixel_values = torch.randn(num_tokens, 3 * 2 * 16 * 16, device="cuda")
    eager = encoder.encode_batched(pixel_values, grid_thw)

    if automatic:
        setup_mm_encoder_cuda_graphs(encoder, max_num_tokens=256)
    else:
        vision.enable_blocks_cuda_graph()

    def fail_eager_fallback(*args: object, **kwargs: object) -> None:
        raise AssertionError("Qwen3-VL vision CUDA graph replay fell back to eager.")

    monkeypatch.setattr(vision, "_run_encoder_blocks", fail_eager_fallback)
    graph = encoder.encode_batched(pixel_values, grid_thw)

    torch.testing.assert_close(graph, eager, rtol=5e-2, atol=5e-2)
    captured = vision._blocks_graph_runner._captured[(257, 2)]
    # Replay must refresh the device lengths read by attention, not only the host copy.
    assert captured.metadata.seq_lens_cuda.tolist() == [num_tokens, 257 - num_tokens]
    assert vision.attn_metadata.max_num_tokens == 256
    assert vision.attn_metadata.max_num_requests == 1
    assert captured.metadata.max_num_tokens == 257
    assert captured.metadata.max_num_requests == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("support_fused_qkv", [True, False])
@pytest.mark.parametrize("rope_impl", ["torch", "flash_attn", "flashinfer"])
def test_qwen2_5_vision_attention_reuses_fused_qkv(
    monkeypatch: pytest.MonkeyPatch, support_fused_qkv: bool, rope_impl: str
) -> None:
    # CPU stubs exercise kernel routing and mutation contracts, not CUDA kernels.
    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)
    monkeypatch.setattr(modeling_qwen2vl, "IS_FLASHINFER_AVAILABLE", rope_impl == "flashinfer")
    head_dim = 64 if rope_impl == "flashinfer" else 4
    q_size = 2 * head_dim
    fused_qkv = torch.randn(3, 3 * q_size)
    attention, forwarded = _make_vision_attention(
        fused_qkv, head_dim=head_dim, support_fused_qkv=support_fused_qkv
    )
    q, k, original_v = attention.split_qkv(fused_qkv)
    original_q = q.reshape(3, 2, head_dim).clone()
    original_k = k.reshape(3, 2, head_dim).clone()
    original_v = original_v.clone()
    cos = torch.randn(3, head_dim // 2)
    sin = torch.randn(3, head_dim // 2)
    position_ids = torch.arange(3, dtype=torch.int32)
    kernel_calls = []

    def flash_attn_rope(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        interleaved: bool,
        inplace: bool,
    ) -> torch.Tensor:
        assert not interleaved
        assert inplace
        kernel_calls.append(tensor)
        tensor.copy_(
            modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(tensor, cos, sin, unsqueeze_dim=1)
        )
        return tensor

    def flashinfer_rope(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None:
        assert positions is position_ids
        assert head_size == 64
        assert is_neox
        assert cos_sin_cache.dtype == torch.float32
        assert cos_sin_cache.is_contiguous()
        torch.testing.assert_close(cos_sin_cache, torch.cat((cos, sin), dim=-1))
        kernel_calls.append(query)
        for tensor in (query, key):
            tensor.copy_(
                modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
                    tensor.view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
                ).reshape_as(tensor)
            )

    if rope_impl == "flash_attn":
        monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", flash_attn_rope)
    elif rope_impl == "flashinfer":
        monkeypatch.setattr(
            modeling_qwen2vl,
            "flashinfer_apply_rope_with_cos_sin_cache_inplace",
            flashinfer_rope,
            raising=False,
        )

    expected_q = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_q.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    expected_k = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_k.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    output = attention.forward(
        torch.empty(3, q_size),
        attn_metadata=object(),
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    actual_q, actual_k, actual_v = attention.split_qkv(fused_qkv)
    torch.testing.assert_close(actual_q.reshape_as(expected_q), expected_q)
    torch.testing.assert_close(actual_k.reshape_as(expected_k), expected_k)
    torch.testing.assert_close(actual_v, original_v)
    assert len(kernel_calls) == {"torch": 0, "flash_attn": 1, "flashinfer": 1}[rope_impl]
    if support_fused_qkv:
        assert forwarded["q"] is fused_qkv
        assert forwarded["k"] is None
        assert forwarded["v"] is None
        assert output is fused_qkv
    else:
        assert forwarded["q"].is_set_to(actual_q)
        assert forwarded["k"].is_set_to(actual_k)
        assert forwarded["v"].is_set_to(actual_v)
        assert output.is_set_to(actual_q)


@pytest.fixture(
    params=[modeling_qwen3vl.Qwen3VLVisionAttention, modeling_qwen2vl.Qwen2_5_VLVisionAttention]
)
def vision_attention(request: pytest.FixtureRequest) -> modeling_qwen2vl.Qwen2_5_VLVisionAttention:
    attention = request.param.__new__(request.param)
    torch.nn.Module.__init__(attention)
    return attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(modeling_qwen2vl._flash_attn_apply_rotary is None, reason="flash_attn required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1, 72), (7, 4, 80), (128, 16, 72), (513, 16, 80)])
@pytest.mark.parametrize("row_step", [1, 2])
def test_qwen_vl_vision_packed_rope_cuda(
    vision_attention: modeling_qwen2vl.Qwen2_5_VLVisionAttention,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    row_step: int,
) -> None:
    seq_len, num_heads, head_dim = shape
    # Include a nonzero storage offset and padding between projection rows.
    storage = torch.randn(
        seq_len * row_step + 1, 3 * num_heads * head_dim, device="cuda", dtype=dtype
    )
    original = storage.clone()
    fused_qkv = storage[1::row_step]
    expected_qkv = original[1::row_step]
    vision_attention.head_dim = head_dim
    cos = torch.randn(seq_len, head_dim // 2, device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    rotary = modeling_qwen2vl._flash_attn_apply_rotary
    for tensor in expected_qkv.split(num_heads * head_dim, dim=-1)[:2]:
        rotary(
            tensor.view(1, seq_len, num_heads, head_dim), cos, sin, interleaved=False, inplace=True
        )

    with patch.object(modeling_qwen2vl, "_flash_attn_apply_rotary", wraps=rotary) as calls:
        vision_attention.apply_rope(
            *fused_qkv.split(num_heads * head_dim, dim=-1),
            position_embeddings=(cos, sin),
            packed_qk=fused_qkv[:, : 2 * num_heads * head_dim],
        )
    assert torch.equal(storage, original)
    assert calls.call_count == 1
    assert calls.call_args.args[0].shape == (1, seq_len, 2 * num_heads, head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(modeling_qwen2vl._flash_attn_apply_rotary is None, reason="flash_attn required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["separate", "gap", "dim_stride", "different_stride"])
def test_qwen_vl_vision_rope_layout_fallback_cuda(
    vision_attention: modeling_qwen2vl.Qwen2_5_VLVisionAttention,
    dtype: torch.dtype,
    layout: str,
) -> None:
    seq_len, num_heads, head_dim = 7, 4, 80
    fused_qkv = torch.randn(seq_len, 3, num_heads, head_dim, device="cuda", dtype=dtype)
    if layout == "dim_stride":
        fused_qkv = fused_qkv[..., ::2]
    q, k, v = fused_qkv.unbind(dim=1)
    if layout == "separate":
        q, k = (
            torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=t.dtype).copy_(t)
            for t in (q, k)
        )
    elif layout == "gap":
        k, v = v, k
    elif layout == "different_stride":
        k = k.clone()
    original_v = v.clone()
    expected_q, expected_k = q.clone(), k.clone()
    cos = torch.randn(seq_len, q.shape[-1] // 2, device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    rotary = modeling_qwen2vl._flash_attn_apply_rotary
    for tensor in (expected_q, expected_k):
        rotary(tensor.unsqueeze(0), cos, sin, interleaved=False, inplace=True)
    vision_attention.head_dim = q.shape[-1]

    with patch.object(modeling_qwen2vl, "_flash_attn_apply_rotary", wraps=rotary) as calls:
        actual = vision_attention.apply_rope(
            q.flatten(1), k.flatten(1), v.flatten(1), position_embeddings=(cos, sin)
        )
    assert calls.call_count == 2
    for tensor, expected in zip(actual, (expected_q, expected_k, original_v)):
        assert torch.equal(tensor, expected.flatten(1))
    assert torch.equal(v, original_v)


def test_qwen3_processor_max_pixels_maps_to_fixed_attention_capacity() -> None:
    max_pixels = 16_777_216
    expected_max_tokens_per_item = {"image": 65_536}
    processor = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)
    processor._config = Qwen3VLConfig.from_dict(copy.deepcopy(QWEN3_VL_8B_CONFIG))
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            size={
                "shortest_edge": 3_136,
                "longest_edge": max_pixels,
            }
        )
    )

    assert processor.get_mm_max_tokens_per_item() == expected_max_tokens_per_item


# ---------------------------------------------------------------------------
# Accuracy tests for the fused Triton bilinear position-embedding kernel used
# by the Qwen3-VL vision tower. Mirrors vLLM's
# `tests/kernels/core/test_vit_bilinear_pos_embed.py` so the two
# implementations stay aligned.
# ---------------------------------------------------------------------------

_VIT_POS_EMBED_DTYPES = [torch.float32, torch.bfloat16]
# Qwen3-VL defaults
_VIT_POS_EMBED_NUM_GRID_PER_SIDE = 48
_VIT_POS_EMBED_SPATIAL_MERGE_SIZE = 2
_VIT_POS_EMBED_HIDDEN_DIM = 1152

_VIT_POS_EMBED_GRIDS = [
    (1, 4, 4),
    (1, 16, 16),
    (1, 32, 32),
    (1, 48, 48),
    (1, 8, 16),
    (1, 14, 20),
    (1, 32, 48),
    (1, 60, 80),
]


def _vit_pos_embed_native_reference(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch reference for one (t, h, w) grid — the pre-Triton
    implementation, kept here only as the test oracle for
    `_triton_pos_embed_interpolate`.
    """
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    return combined.expand(t, -1, -1).reshape(-1, hidden_dim).to(dtype)


@pytest.mark.parametrize("dtype", _VIT_POS_EMBED_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "grid_thw",
    _VIT_POS_EMBED_GRIDS,
    ids=[f"{t}x{h}x{w}" for t, h, w in _VIT_POS_EMBED_GRIDS],
)
def test_vit_pos_embed_triton_matches_native(grid_thw, dtype):
    """Triton kernel output must match the PyTorch reference within
    bf16 / fp32 ULP-level rounding."""
    t, h, w = grid_thw
    device = "cuda"

    # Scale to match the real Qwen3-VL pos_embed weight distribution
    # (std ~ 0.23). Larger scales magnify the bf16 rounding gap between
    # the two op orderings.
    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            _VIT_POS_EMBED_NUM_GRID_PER_SIDE * _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
            _VIT_POS_EMBED_HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    native_out = _vit_pos_embed_native_reference(
        embed_weight,
        t,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )
    triton_out = _triton_pos_embed_interpolate(
        embed_weight,
        t,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )

    assert native_out.shape == triton_out.shape

    # Single-ULP differences come from the precomputed scalar h/w_scale
    # in the Triton kernel vs `torch.linspace` in the reference. Match
    # vLLM's tolerances for the same kernel.
    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", _VIT_POS_EMBED_DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_vit_pos_embed_temporal_repeat(dtype):
    """t > 1 must repeat the (h, w) spatial pattern verbatim."""
    device = "cuda"
    h, w = 16, 16
    t_single, t_multi = 1, 3

    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            _VIT_POS_EMBED_NUM_GRID_PER_SIDE * _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
            _VIT_POS_EMBED_HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    out_single = _triton_pos_embed_interpolate(
        embed_weight,
        t_single,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )
    out_multi = _triton_pos_embed_interpolate(
        embed_weight,
        t_multi,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )

    expected = out_single.repeat(t_multi, 1)
    torch.testing.assert_close(out_multi, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Accuracy tests for `Qwen3VisionModel.rot_pos_emb`'s pos_ids construction.
# The new implementation lifts pos_ids generation to CPU (numpy + lru_cache)
# and ships a single H->D copy at the end. Previously pos_ids were built on
# device via a per-image torch.arange/expand/stack chain. The freq_table
# lookup is identical in both paths, so we only verify pos_ids equality —
# that guarantees the final `freq_table[pos_ids]` output matches bit-for-bit.
# ---------------------------------------------------------------------------


def _rot_pos_ids_gpu_reference(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Pre-vectorization GPU reference for pos_ids.

    Mirrors the original on-device implementation: per-image torch.arange
    + broadcast/expand/stack/repeat, written into a contiguous (total, 2)
    buffer on the same device as `grid_thw`.
    """
    device = grid_thw.device
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // spatial_merge_size, width // spatial_merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(spatial_merge_size, device=device)
        intra_col = torch.arange(spatial_merge_size, device=device)

        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return pos_ids


_ROT_POS_IDS_GRIDS = [
    # Single image, varying (h, w)
    [(1, 4, 4)],
    [(1, 16, 16)],
    [(1, 32, 48)],
    [(1, 14, 20)],
    # Multi-frame video (t > 1)
    [(3, 16, 16)],
    [(5, 8, 12)],
    # Multi-image batch (mixed sizes)
    [(1, 16, 16), (1, 32, 32)],
    [(1, 8, 12), (1, 14, 20), (1, 32, 32)],
    # Mix of images and a video clip
    [(1, 16, 16), (3, 8, 12), (1, 14, 20)],
]


@pytest.mark.parametrize(
    "grid_thw_list",
    _ROT_POS_IDS_GRIDS,
    ids=["_".join(f"{t}x{h}x{w}" for t, h, w in cfg) for cfg in _ROT_POS_IDS_GRIDS],
)
def test_rot_pos_ids_matches_gpu_reference(grid_thw_list):
    """`rot_pos_ids` + cat (new path) must produce the same pos_ids tensor
    as the pre-vectorization on-device implementation."""
    spatial_merge_size = _VIT_POS_EMBED_SPATIAL_MERGE_SIZE
    device = "cuda"

    grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)
    expected = _rot_pos_ids_gpu_reference(grid_thw, spatial_merge_size)

    # New path: per-(h, w) CPU build (lru_cache), per-image repeat, then
    # cat + single H->D transfer.
    pieces = [
        Qwen3VisionModel.rot_pos_ids(h, w, spatial_merge_size)
        if t == 1
        else Qwen3VisionModel.rot_pos_ids(h, w, spatial_merge_size).repeat(t, 1)
        for t, h, w in grid_thw_list
    ]
    actual = torch.cat(pieces, dim=0).to(device, non_blocking=True)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.cpu_only
def test_rot_pos_ids_lru_cache_hit():
    """Repeated (h, w, spatial_merge_size) keys must hit the lru_cache and
    return the same underlying tensor object (no recompute)."""
    Qwen3VisionModel.rot_pos_ids.cache_clear()
    a = Qwen3VisionModel.rot_pos_ids(16, 16, 2)
    b = Qwen3VisionModel.rot_pos_ids(16, 16, 2)
    assert a is b, "lru_cache should return the cached tensor object on hit"
    info = Qwen3VisionModel.rot_pos_ids.cache_info()
    assert info.hits >= 1 and info.misses >= 1


# ---------------------------------------------------------------------------
# L2 cache: per-(t, h, w) GPU rotary embeddings via _rotary_pos_emb_thw.
#
# Mirrors vLLM's get_rope_by_thw pattern: identical (t, h, w) returns the
# same on-device tensor object so subsequent rot_pos_emb calls only do a
# device-side torch.cat with no H->D transfer.
# ---------------------------------------------------------------------------


def _make_qwen3_vision_model_for_l2():
    """Build a minimally-initialized Qwen3VisionModel that exposes
    `_rotary_pos_emb_thw` and `rot_pos_emb` without loading any weights.

    Bypasses the heavy `__init__` (blocks, mergers, attn metadata buffers)
    by going through `__new__` so the class-level decorators (`@staticmethod`
    on `rot_pos_ids` and `@lru_cache` on `_rotary_pos_emb_thw`) stay intact.
    The mock wires only the attrs those cached methods touch:
    `spatial_merge_size`, `rotary_pos_emb` (a stub exposing the
    `rotary_cos_sin` cache that `_rotary_pos_emb_thw` indexes into),
    and `patch_embed.proj.weight` (read by `device`).
    """
    obj = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(obj)
    obj.spatial_merge_size = 2

    # Mirror the real init: rope_dim = head_dim // 2 = 36, theta = 10000.
    rope_dim = 36
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    max_rope_seqlen = 8192
    seq = torch.arange(max_rope_seqlen, dtype=torch.float32)
    freqs = torch.outer(seq, inv_freq)
    # `_rotary_pos_emb_thw` slices `rotary_pos_emb.rotary_cos_sin` and
    # indexes with `pos_ids`. Stack cos/sin along dim 1 to match the
    # real `RotaryEmbedding` buffer layout `(max_pos, 2, freq_dim)`.
    rotary_cos_sin = torch.stack([freqs.cos(), freqs.sin()], dim=1).to("cuda")
    rotary_pos_emb = torch.nn.Module()
    rotary_pos_emb.rotary_cos_sin = rotary_cos_sin
    obj.rotary_pos_emb = rotary_pos_emb

    # patch_embed.proj.weight is what `Qwen3VisionModel.device` reads.
    obj.patch_embed = torch.nn.Module()
    obj.patch_embed.proj = torch.nn.Linear(1, 1, bias=False).to("cuda")
    return obj


def test_rotary_pos_emb_thw_returns_device_tensor():
    """Cached per-(t, h, w) rotary (cos, sin) pair must live on CUDA so
    subsequent rot_pos_emb calls do device-side cats (no H->D transfer)."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    cos, sin = vm._rotary_pos_emb_thw(1, 16, 16)
    assert cos.is_cuda and sin.is_cuda, "L2 cache must store cos/sin on device"
    assert cos.shape == sin.shape
    assert cos.shape[0] == 1 * 16 * 16
    # After gather + flatten the per-token dim is 2 * freq_dim (= 36).
    assert cos.shape[1] % 2 == 0


def test_rotary_pos_emb_thw_lru_cache_hit():
    """Same (t, h, w) must return the same cached (cos, sin) tuple
    (object identity, not equality)."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()
    a_cos, a_sin = vm._rotary_pos_emb_thw(1, 16, 16)
    b_cos, b_sin = vm._rotary_pos_emb_thw(1, 16, 16)
    assert a_cos is b_cos and a_sin is b_sin, (
        "L2 lru_cache must return the cached device tensors on hit"
    )
    info = vm._rotary_pos_emb_thw.cache_info()
    assert info.hits >= 1 and info.misses >= 1


def test_rot_pos_emb_l2_matches_per_tile():
    """rot_pos_emb on a multi-image grid must equal the per-tile (cos, sin)
    tensors concatenated along dim 0, bit-exact."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid_thw_list = [(1, 16, 16), (1, 32, 32), (1, 16, 16), (3, 8, 12)]
    cos_out, sin_out = vm.rot_pos_emb(grid_thw_list)

    cos_exp = torch.cat(
        [vm._rotary_pos_emb_thw(t, h, w)[0] for t, h, w in grid_thw_list],
        dim=0,
    )
    sin_exp = torch.cat(
        [vm._rotary_pos_emb_thw(t, h, w)[1] for t, h, w in grid_thw_list],
        dim=0,
    )
    torch.testing.assert_close(cos_out, cos_exp, atol=0, rtol=0)
    torch.testing.assert_close(sin_out, sin_exp, atol=0, rtol=0)


def test_rot_pos_emb_l2_no_device_transfer_on_hit():
    """After the first call warms the L2 cache for a (t, h, w), a second
    rot_pos_emb call on the same grid must not trigger any new miss."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid = [(1, 16, 16), (1, 32, 32), (1, 16, 16)]
    vm.rot_pos_emb(grid)  # warm
    miss_count = vm._rotary_pos_emb_thw.cache_info().misses

    vm.rot_pos_emb(grid)  # all hits
    info = vm._rotary_pos_emb_thw.cache_info()
    assert info.misses == miss_count, (
        f"second rot_pos_emb call should be all cache hits; got "
        f"{info.misses - miss_count} extra miss(es)"
    )
    # Per-tile cos/sin must be on CUDA so forward's torch.cat is device-side.
    for t, h, w in grid:
        cos, sin = vm._rotary_pos_emb_thw(t, h, w)
        assert cos.is_cuda and sin.is_cuda


def test_rot_pos_emb_cos_sin_matches_old_repeat_chain():
    """The new pre-computed-cos/sin path must produce the same per-token
    (cos, sin) after the forward's `.repeat(1, 2)` as the prior
    `freqs.repeat(1, 2).cos()/.sin()` chain."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid_thw_list = [(1, 16, 16), (1, 32, 32), (3, 8, 12)]

    # New path: gathered cos/sin then forward's `.repeat(1, 2)`.
    cos_new, sin_new = vm.rot_pos_emb(grid_thw_list)
    cos_new = cos_new.repeat(1, 2)
    sin_new = sin_new.repeat(1, 2)

    # Old path replica: build freqs via outer on the fly, then
    # `flatten(1).repeat(1, 2).cos()/.sin()`.
    rope_dim = 36
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    inv_freq = inv_freq.to("cuda")
    cos_pieces, sin_pieces = [], []
    for t, h, w in grid_thw_list:
        pos_ids = Qwen3VisionModel.rot_pos_ids(h, w, vm.spatial_merge_size)
        if t > 1:
            pos_ids = pos_ids.repeat(t, 1)
        pos_ids = pos_ids.to("cuda")
        max_hw = max(h, w)
        seq = torch.arange(max_hw, device="cuda", dtype=torch.float32)
        freq_table = torch.outer(seq, inv_freq)  # (max_hw, 18)
        freqs = freq_table[pos_ids].flatten(1)  # (total, 36)
        emb = freqs.repeat(1, 2)  # (total, 72)
        cos_pieces.append(emb.cos())
        sin_pieces.append(emb.sin())
    cos_old = torch.cat(cos_pieces, dim=0)
    sin_old = torch.cat(sin_pieces, dim=0)

    # The cos/sin formulas are identical; the only difference is that
    # the new path computes the outer product host-side at init, while
    # the old path computes it on-device per call. That's a single-ULP
    # fp32 gap (~1e-7).
    torch.testing.assert_close(cos_new, cos_old, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sin_new, sin_old, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Pre-computed cos/sin buffer (RotaryEmbedding.rotary_cos_sin) used by
# `_rotary_pos_emb_thw` to gather without firing per-forward
# `.cos()` / `.sin()` kernels.
# ---------------------------------------------------------------------------


def test_rope_cos_sin_buffers_match_hf_rotary():
    """The cos/sin cache exposed via `rotary_pos_emb.rotary_cos_sin` must
    bit-match HF's `Qwen3VLVisionRotaryEmbedding(dim=36)(max_hw).cos()`
    / `.sin()` output."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding as HFQwen3VLVisionRotaryEmbedding,
    )

    vm = _make_qwen3_vision_model_for_l2()
    hf = HFQwen3VLVisionRotaryEmbedding(36).to("cuda")
    atol = rtol = 1e-6
    # `rotary_cos_sin` is stacked along dim 1 as (max_pos, 2, freq_dim);
    # index 0 holds cos, index 1 holds sin.
    cos_cache = vm.rotary_pos_emb.rotary_cos_sin[:, 0, :]
    sin_cache = vm.rotary_pos_emb.rotary_cos_sin[:, 1, :]
    for max_hw in [16, 32, 48, 64, 100, 128]:
        ref_freqs = hf(max_hw)
        torch.testing.assert_close(cos_cache[:max_hw], ref_freqs.cos(), atol=atol, rtol=rtol)
        torch.testing.assert_close(sin_cache[:max_hw], ref_freqs.sin(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# argsort(window_index) equivalence with the prior scatter-assign inverse.
#
# Qwen2.5-VL's vision tower swaps
#   reverse_indices = torch.empty_like(window_index)
#   reverse_indices[window_index] = torch.arange(N, device=..., dtype=...)
# for
#   reverse_indices = torch.argsort(window_index)
# The two are mathematically the inverse permutation; argsort is one
# fused op vs the prior alloc + scatter pair.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# async_tensor_h2d helper (tensorrt_llm._utils).
# ---------------------------------------------------------------------------


def test_async_tensor_h2d_sequence_input():
    """List/tuple input -> device tensor with correct dtype and values."""
    from tensorrt_llm._utils import async_tensor_h2d

    out = async_tensor_h2d([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    assert out.is_cuda and out.dtype == torch.int32
    torch.testing.assert_close(out.cpu(), torch.tensor([1, 2, 3, 4], dtype=torch.int32))


def test_async_tensor_h2d_tensor_input():
    """CPU tensor input -> device tensor with cast applied and values
    preserved."""
    from tensorrt_llm._utils import async_tensor_h2d

    src = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
    out = async_tensor_h2d(src, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    assert out.is_cuda and out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), src.to(torch.float32))


def test_maybe_pin_memory_idempotent_on_already_pinned():
    """A second call on an already-pinned tensor must return it unchanged
    (same Python object) -- the new `is_pinned()` guard skips the
    redundant `.pin_memory()` dispatch."""
    from tensorrt_llm._utils import maybe_pin_memory

    t = torch.tensor([1.0, 2.0, 3.0])
    pinned = maybe_pin_memory(t)
    if not pinned.is_pinned():
        pytest.skip("prefer_pinned() is False in this environment (confidential compute mode)")
    again = maybe_pin_memory(pinned)
    assert again is pinned


@pytest.mark.parametrize("n", [16, 256, 1024])
def test_argsort_matches_scatter_inverse(n):
    """argsort(perm) must equal the scatter-assign inverse permutation."""
    torch.manual_seed(n)
    window_index = torch.randperm(n, device="cuda", dtype=torch.long)

    # Old scatter-assign path.
    old = torch.empty_like(window_index)
    old[window_index] = torch.arange(n, device="cuda", dtype=window_index.dtype)
    # New argsort path.
    new = torch.argsort(window_index)

    torch.testing.assert_close(new, old, atol=0, rtol=0)
    # Functional check: applying `new` to `window_index` is identity.
    identity = window_index[new]
    torch.testing.assert_close(
        identity, torch.arange(n, device="cuda", dtype=window_index.dtype), atol=0, rtol=0
    )


# ---------------------------------------------------------------------------
# End-to-end Qwen3VLVisionBlock forward path.
#
# The pre-EXAONE-4.5 `apply_rotary_pos_emb_vision` (HF) accepted
# `cos / sin` of shape `(seq, head_dim)`; the post-EXAONE-4.5
# `RotaryEmbedding.apply_rotary_pos_emb` (TRT-LLM) expects
# `(seq, head_dim // 2)` and broadcasts. This test pushes one block
# through with the post-EXAONE shape so any future regression (e.g.
# re-introducing a stray `.repeat(1, 2)` on cos/sin) surfaces here
# rather than only at full-model CI.
# ---------------------------------------------------------------------------


def test_qwen3vl_vision_block_forward_end_to_end():
    """One `Qwen3VLVisionBlock.forward` call with random weights and the
    correct (post-EXAONE) `cos/sin` shape must not raise a broadcasting
    error in `apply_rope`."""
    from transformers import Qwen3VLConfig

    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLVisionBlock

    hf_config = Qwen3VLConfig(
        text_config={
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "head_dim": 64,
            "rope_scaling": {
                "mrope_section": [8, 8, 8],
                "mrope_interleaved": True,
                "rope_type": "default",
            },
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "num_key_value_heads": 4,
            "vocab_size": 1024,
            "dtype": "bfloat16",
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "model_type": "qwen3_vl_text",
        },
        vision_config={
            "hidden_size": 1152,
            "num_heads": 16,  # head_dim = 72
            "depth": 1,
            "intermediate_size": 4304,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "out_hidden_size": 256,
            "deepstack_visual_indexes": [],
            "hidden_act": "gelu_pytorch_tanh",
            "model_type": "qwen3_vl",
            "initializer_range": 0.02,
        },
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )

    mc = ModelConfig(pretrained_config=hf_config, attn_backend="TRTLLM")
    block = Qwen3VLVisionBlock(mc, layer_idx=0).to("cuda", dtype=torch.bfloat16)

    seq_len = 256
    head_dim = 72
    freq_dim = head_dim // 2  # 36 -- the (post-EXAONE) cos/sin last-dim

    hidden = torch.randn(seq_len, 1152, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq_len, freq_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn(seq_len, freq_dim, device="cuda", dtype=torch.float32)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda")

    metadata_cls = get_attention_backend("TRTLLM").Metadata
    attn_metadata = metadata_cls(
        max_num_requests=64,
        max_num_tokens=seq_len,
        kv_cache_manager=None,
    )
    attn_metadata.num_contexts = 1
    attn_metadata.request_ids = [1]
    attn_metadata.prompt_lens = [seq_len]
    attn_metadata.seq_lens = torch.tensor([seq_len], dtype=torch.int)
    attn_metadata.max_seq_len = seq_len
    attn_metadata.prepare()

    out = block(
        hidden,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        attn_metadata=attn_metadata,
    )
    assert out.shape == hidden.shape


def _metadata_fixture(dtype: torch.dtype, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=SimpleNamespace(dtype=dtype))),
        metadata_cls=lambda **kwargs: SimpleNamespace(**kwargs),
        set_attn_max_seq_len=lambda _: None,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalization_startup_follows_encoder_dtype(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, dtype: torch.dtype
) -> None:
    seen = []
    monkeypatch.setattr(
        modeling_qwen3vl,
        "normalize_vision_pixels",
        lambda pixels, normalization: seen.append((pixels, normalization)),
    )
    model = _metadata_fixture(dtype, torch.device("cpu"))
    model.model_config = SimpleNamespace(
        multimodal_config=MultimodalConfig(vision_gpu_normalize=enabled)
    )
    modeling_qwen3vl.Qwen3VisionModel.setup_attn_metadata(model, 16, {"attention": 4})
    assert len(seen) == int(enabled)
    if seen:
        pixels, spec = seen[0]
        assert pixels.dtype == torch.uint8
        assert pixels.is_contiguous()
        assert spec.out_dtype == dtype
        assert spec.compute_dtype is None
    assert model.attn_metadata.max_num_tokens == 16


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "input_kind",
    [
        "rgb",
        "array",
        "grayscale",
        "rgba",
        "float",
        "channel_first",
        "video",
        "mixed",
        "video_float",
        "custom_mean",
        "custom_std",
        "resize",
        "nested",
        "no_rescale",
        "no_normalize",
        "no_elementwise",
        "output_float",
        "disable_rescale",
        "disable_normalize",
        "rescale_factor",
        "convert_rgb",
        "default_float64",
    ],
)
def test_normalization_processor_uses_config(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool | None,
    dtype: torch.dtype,
    input_kind: str,
) -> None:
    config = Qwen3VLConfig.from_dict(QWEN3_VL_8B_CONFIG)
    config.text_config.dtype = dtype
    hf_processor = MagicMock(
        return_value={
            "pixel_values": torch.zeros(1, 1536, dtype=torch.uint8),
            "pixel_values_videos": torch.zeros(1, 1536, dtype=torch.uint8),
        }
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    hf_processor.image_processor = Qwen2VLImageProcessor(
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        rescale_factor=1 / 255,
        patch_size=16,
        temporal_patch_size=2,
    )
    hf_processor.video_processor = copy.deepcopy(hf_processor.image_processor)
    if input_kind == "output_float":
        hf_processor.return_value["pixel_values"] = torch.zeros(1, 1536, dtype=torch.float32)
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_qwen2vl.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: hf_processor,
    )
    options = {} if enabled is None else {"vision_gpu_normalize": enabled}
    processor = Qwen3VLInputProcessorBase("unused", config, MagicMock(), **options)
    image = Image.new("RGB", (32, 32))
    if input_kind == "array":
        image = np.zeros((32, 32, 3), dtype=np.uint8)
    elif input_kind in ("grayscale", "rgba"):
        image = Image.new("L" if input_kind == "grayscale" else "RGBA", (32, 32))
    elif input_kind == "float":
        image = np.zeros((32, 32, 3), dtype=np.float32)
    elif input_kind == "channel_first":
        image = np.zeros((3, 32, 32), dtype=np.uint8)
    mm_data = {"image": [image]}
    if input_kind in ("video", "mixed", "video_float"):
        frame = np.zeros((32, 32, 3), dtype=np.float32) if input_kind == "video_float" else image
        mm_data["video"] = [SimpleNamespace(frames=[frame, frame], metadata={})]
        if input_kind == "video":
            mm_data.pop("image")
    overrides = {
        "custom_mean": {"image_mean": [0.4, 0.5, 0.6]},
        "custom_std": {"image_std": [0.2, 0.3, 0.4]},
        "resize": {"do_resize": False},
        "nested": {"images_kwargs": {"do_normalize": False}},
        "disable_rescale": {"do_rescale": False},
        "disable_normalize": {"do_normalize": False},
        "rescale_factor": {"rescale_factor": 1 / 128},
        "convert_rgb": {"do_convert_rgb": False},
    }.get(input_kind, {})
    if input_kind == "no_rescale":
        hf_processor.image_processor.do_rescale = False
    elif input_kind == "no_normalize":
        hf_processor.image_processor.do_normalize = False
    elif input_kind == "no_elementwise":
        hf_processor.image_processor = SimpleNamespace(do_rescale=True, do_normalize=True)
    elif input_kind == "default_float64":
        monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    processed = processor._preprocess("prompt", mm_data, overrides)
    expected = enabled is True and input_kind in (
        "rgb",
        "array",
        "video",
        "mixed",
        "no_normalize",
        "default_float64",
    )
    assert bool(processed.get("vision_normalization")) is expected
    assert hf_processor.call_args.kwargs["do_rescale"] is overrides.get("do_rescale", not expected)
    assert hf_processor.call_args.kwargs.get("do_normalize") is (
        False if expected else overrides.get("do_normalize")
    )
    if expected:
        assert set(processed["vision_normalization"]) == set(mm_data)
        for spec in processed["vision_normalization"].values():
            assert spec.out_dtype == dtype
            assert isinstance(spec, VisionNormalization)
            assert spec.image_mean == (None if input_kind == "no_normalize" else (0.5,) * 3)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalization_metadata_survives_processor_output(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    config = Qwen3VLConfig.from_dict(QWEN3_VL_8B_CONFIG)
    config.text_config.dtype = dtype
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_qwen2vl.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: MagicMock(),
    )
    processor = Qwen3VLInputProcessorBase("unused", config, MagicMock())
    spec = VisionNormalization(1 / 255, (0.5,) * 3, (0.5,) * 3, dtype)
    processed = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "pixel_values": torch.zeros((4, 1536), dtype=torch.uint8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "pixel_values_videos": torch.zeros((4, 1536), dtype=torch.uint8),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "vision_normalization": {"image": spec, "video": spec},
    }
    monkeypatch.setattr(processor, "_preprocess", lambda *args: processed)
    monkeypatch.setattr(processor, "get_mrope_config", lambda *args: {})
    monkeypatch.setattr(processor, "_postprocess", lambda input_ids: input_ids)
    _, extra = processor.call_with_text_prompt(
        {"prompt": "prompt", "multi_modal_data": {"image": [None], "video": [None]}}, MagicMock()
    )
    for modality, pixel_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
        bucket = extra["multimodal_data"][modality]
        assert bucket[pixel_key].dtype == torch.uint8
        assert bucket["vision_normalization"] == spec


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("worker_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_statistics", [False, True])
@pytest.mark.parametrize("do_rescale,do_normalize", [(True, True), (True, False), (False, True)])
def test_normalization_enabled_operations(
    dtype, default_dtype, worker_dtype, scalar_statistics, do_rescale, do_normalize
):
    import pickle

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(
        do_rescale=do_rescale,
        do_normalize=do_normalize,
        rescale_factor=1 / 127 if do_rescale else None,
        image_mean=(0.2 if scalar_statistics else [0.2, 0.4, 0.6]) if do_normalize else None,
        image_std=(0.1 if scalar_statistics else [0.1, 0.3, 0.7]) if do_normalize else None,
    )
    values = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256, 1), (2, 1, 3))
    pixels = torch.from_numpy(values).permute(2, 0, 1).contiguous()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        spec = vision_normalization_from_processor([values], processor, dtype)
        assert isinstance(spec, VisionNormalization)
        assert spec.rescale_factor == (processor.rescale_factor if do_rescale else None)
        assert spec.image_mean == (
            ((processor.image_mean,) * 3 if scalar_statistics else tuple(processor.image_mean))
            if do_normalize
            else None
        )
        assert spec.compute_dtype == default_dtype
        expected = processor.rescale_and_normalize(
            pixels,
            do_rescale=do_rescale,
            rescale_factor=processor.rescale_factor,
            do_normalize=do_normalize,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
        ).to(dtype)
        spec = pickle.loads(pickle.dumps(spec))
        torch.set_default_dtype(worker_dtype)
        assert torch.equal(
            normalize_vision_pixels(pixels.reshape(3, -1), spec), expected.reshape(3, -1)
        )
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case", ["empty", "noop", "no_elementwise", "custom_rescale", "custom_normalize"]
)
def test_normalization_builder_fallback(case):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    frames = [Image.new("RGB", (16, 16))]
    processor = Qwen2VLImageProcessor()
    if case == "empty":
        frames = []
    elif case == "noop":
        processor.do_rescale = processor.do_normalize = False
    elif case == "no_elementwise":
        processor = SimpleNamespace()
    else:
        setattr(processor, case.removeprefix("custom_"), lambda *args, **kwargs: None)
    assert vision_normalization_from_processor(frames, processor, torch.bfloat16) is None


def _normalization_processor_outputs(
    sample: str, modality: str, dtype: torch.dtype, custom_constants: bool
) -> tuple[torch.Tensor, torch.Tensor, VisionNormalization]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    if sample == "all_values":
        values = torch.arange(1024, dtype=torch.int32).remainder(256).reshape(32, 32)
        pixels = torch.stack([values, values.roll(17), values.roll(71)]).to(torch.uint8)
        assert all(channel.unique().numel() == 256 for channel in pixels)
    else:
        default_path = Path(__file__).parents[3] / "integration/test_input_files/merlion.png"
        path = Path(os.environ.get("TRTLLM_TEST_IMAGE", default_path))
        if not path.is_file() or path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/"
        ):
            pytest.skip("Real-image fixture is not materialized")
        with Image.open(path) as image:
            pixels = torch.from_numpy(np.asarray(image.convert("RGB").resize((53, 79))).copy())
        pixels = pixels.permute(2, 0, 1)
    mean = (0.4, 0.5, 0.6) if custom_constants else (0.5,) * 3
    std = (0.2, 0.3, 0.4) if custom_constants else (0.5,) * 3
    if modality == "image":
        processor = Qwen2VLImageProcessor(
            image_mean=list(mean),
            image_std=list(std),
            patch_size=16,
            temporal_patch_size=2,
            size={"shortest_edge": 4096, "longest_edge": 16384},
        )
        kwargs = {"images": [pixels]}
        pixel_key = "pixel_values"
    else:
        processor = Qwen3VLVideoProcessor(
            image_mean=list(mean),
            image_std=list(std),
            size={"shortest_edge": 4096, "longest_edge": 16384},
        )
        kwargs = {"videos": [torch.stack([pixels, pixels.flip(-1)])], "do_sample_frames": False}
        pixel_key = "pixel_values_videos"
    do_resize = sample == "real_image"
    expected = processor(**kwargs, do_resize=do_resize, return_tensors="pt")[pixel_key].to(dtype)
    raw = processor(
        **kwargs, do_resize=do_resize, do_rescale=False, do_normalize=False, return_tensors="pt"
    )[pixel_key]
    spec = vision_normalization_from_processor(
        [Image.fromarray(pixels.permute(1, 2, 0).numpy())], processor, dtype
    )
    assert spec is not None
    return raw, expected, spec


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sample", ["all_values", "real_image"])
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("custom_constants", [False, True])
def test_normalization_matches_cpu_processor(
    dtype: torch.dtype, sample: str, modality: str, custom_constants: bool
) -> None:
    raw, expected, spec = _normalization_processor_outputs(
        sample, modality, dtype, custom_constants
    )
    assert raw.dtype == torch.uint8
    assert torch.equal(
        normalize_vision_pixels(raw.view(-1, 3, raw.shape[-1] // 3), spec).view(raw.shape),
        expected,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sample", ["all_values", "real_image"])
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("custom_constants", [False, True])
@pytest.mark.parametrize("producer_dtype", [torch.float32, torch.float64])
def test_normalization_matches_gpu_processor(
    dtype: torch.dtype,
    sample: str,
    modality: str,
    custom_constants: bool,
    producer_dtype: torch.dtype,
) -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(producer_dtype)
        raw, expected, spec = _normalization_processor_outputs(
            sample, modality, dtype, custom_constants
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.device("cuda"):
        actual = normalize_vision_pixels(raw.cuda().view(-1, 3, raw.shape[-1] // 3), spec).view(
            raw.shape
        )
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalization_mixed_raw_and_float_batches(dtype: torch.dtype) -> None:
    raw, expected, spec = _normalization_processor_outputs("all_values", "image", dtype, True)
    raw_video, expected_video, video_spec = _normalization_processor_outputs(
        "all_values", "video", dtype, False
    )
    grid = torch.tensor([[1, 2, 2]])
    params = [
        MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": raw,
                    "image_grid_thw": grid,
                    "vision_normalization": spec,
                },
                "video": {"pixel_values_videos": expected_video, "video_grid_thw": grid},
            }
        ),
        MultimodalParams(
            multimodal_data={
                "image": {"pixel_values": expected, "image_grid_thw": grid},
                "video": {
                    "pixel_values_videos": raw_video,
                    "video_grid_thw": grid,
                    "vision_normalization": video_spec,
                },
            }
        ),
    ]
    batched = modeling_qwen3vl._qwen3vl_build_batched_input(params)
    assert batched["pixel_values"].dtype == dtype
    assert torch.equal(
        batched["pixel_values"], torch.cat([expected, expected, expected_video, expected_video])
    )
    assert torch.equal(batched["grid_thw"], grid.repeat(4, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalization_does_not_compile_after_startup(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    model = _metadata_fixture(dtype, torch.device("cuda"))
    model.model_config = SimpleNamespace(
        multimodal_config=MultimodalConfig(vision_gpu_normalize=True)
    )
    modeling_qwen3vl.Qwen3VisionModel.setup_attn_metadata(model, 16, {"attention": 4})

    def unexpected_compile(*args, **kwargs) -> None:
        pytest.fail("Raw-pixel normalization compiled after encoder startup")

    monkeypatch.setattr(
        modeling_multimodal_utils._normalize_vision_pixels_kernel, "_do_compile", unexpected_compile
    )
    for count in (256, 1536, 4096 * 1536):
        pixels = torch.arange(count, device="cuda", dtype=torch.int32).to(torch.uint8)
        spec = VisionNormalization(1 / 255, (0.5,), (0.5,), dtype)
        expected = ((pixels.cpu().float() - 127.5) / 127.5).to(dtype)
        assert torch.equal(
            normalize_vision_pixels(pixels.view(1, 1, -1), spec).view(pixels.shape).cpu(),
            expected,
        )


def _normalization_transport_spec():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    return vision_normalization_from_processor(
        [Image.new("RGB", (16, 16))],
        Qwen2VLImageProcessor(image_mean=[0.25, 0.5, 0.75], image_std=[0.5] * 3),
        torch.bfloat16,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_normalization_transport_item_selection(declared: bool, modality: str) -> None:
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalItemPartition,
    )

    spec = _normalization_transport_spec()
    pixel_key, grid_key = (
        ("pixel_values", "image_grid_thw")
        if modality == "image"
        else ("pixel_values_videos", "video_grid_thw")
    )
    count = len(spec)
    pixels = torch.arange(count * 3, dtype=torch.uint8).view(count, 3)
    # Matching the tuple's field count must not turn shared metadata into item data.
    param = MultimodalParams(
        multimodal_data={
            modality: {
                pixel_key: pixels,
                grid_key: torch.ones(count, 3, dtype=torch.int64),
                "vision_normalization": spec,
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [(modality, i) for i in range(count)],
                output_embedding_lengths=[1] * count,
                input_partitions={
                    modality: {
                        pixel_key: MultimodalItemPartition(),
                        grid_key: MultimodalItemPartition(),
                    }
                },
            ),
        }
    )
    copied_spec = copy.deepcopy(param).multimodal_data[modality]["vision_normalization"]
    assert type(copied_spec) is type(spec) and copied_spec == spec
    model = MultimodalModelMixin()
    if not declared:
        model.get_multimodal_item_partitions = lambda param, modality: None
    selected, lengths, selected_modality = model.prepare_multimodal_encoder_inputs(
        [(param, 3), (param, 1)]
    )[0]
    assert selected_modality == modality and lengths == (1, 1)
    assert selected.multimodal_data[modality]["vision_normalization"] is spec
    assert torch.equal(selected.multimodal_data[modality][pixel_key], pixels[[3, 1]])
    assert param.multimodal_data[modality]["vision_normalization"] is spec


@pytest.mark.cpu_only
@pytest.mark.parametrize("target_keywords", [None, ["image"], ["image.vision_normalization"]])
def test_normalization_transport_device(target_keywords) -> None:
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope

    spec = _normalization_transport_spec()
    pixels = torch.arange(12, dtype=torch.uint8)
    param = MultimodalParams(
        multimodal_data={"image": {"pixel_values": pixels, "vision_normalization": spec}}
    )
    # Exercise the CUDA dispatch without allocating a GPU or replacing metadata.
    with patch.object(torch.Tensor, "to", autospec=True, return_value=pixels) as move:
        param.to_device("multimodal_data", "cuda", target_keywords=target_keywords)
    assert move.call_count == (0 if target_keywords == ["image.vision_normalization"] else 1)
    assert param.multimodal_data["image"]["vision_normalization"] is spec
    param.to_device("multimodal_data", "cpu")
    assert param.multimodal_data["image"]["vision_normalization"] is spec
    with nccl_window_tensor_scope([param]) as scope:
        assert scope.escape(param) is param
    assert param.multimodal_data["image"]["vision_normalization"] is spec


def _normalization_transport_process(connection, payload) -> None:
    import pickle

    request = pickle.loads(payload)
    request.multimodal_params.to_tensor("multimodal_data")
    data = request.multimodal_params.multimodal_data["image"]
    normalized = normalize_vision_pixels(
        data["pixel_values"].reshape(3, -1), data["vision_normalization"]
    )
    connection.send(
        (data["vision_normalization"], data["pixel_values"].tolist(), normalized.tolist())
    )
    connection.close()


@pytest.mark.cpu_only
@pytest.mark.parametrize("producer_dtype", [torch.float32, torch.float64])
def test_normalization_transport_shared_handles_and_process(producer_dtype) -> None:
    import multiprocessing
    import pickle

    from tensorrt_llm.executor.ipc import ZeroMqQueue
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.sampling_params import SamplingParams

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(producer_dtype)
        spec = _normalization_transport_spec()
        assert spec.compute_dtype == producer_dtype
    finally:
        torch.set_default_dtype(previous_dtype)
    pixels = torch.arange(12, dtype=torch.uint8)
    params = MultimodalParams(
        multimodal_data={"image": {"pixel_values": pixels, "vision_normalization": spec}}
    )
    params.to_handle("multimodal_data")
    assert params.multimodal_data["image"]["vision_normalization"] is spec
    assert isinstance(params.multimodal_data["image"]["pixel_values"], dict)
    request = GenerationRequest([1, 2], SamplingParams(), multimodal_params=params)
    # Use the IPC wire serializer without opening sockets.
    queue = ZeroMqQueue.__new__(ZeroMqQueue)
    queue.socket = queue.context = None
    queue.use_hmac_encryption = True
    queue.hmac_key = b"normalization-transport-test"
    wire = queue._prepare_data(request)
    restored = queue._parse_data(wire)
    restored_spec = restored.multimodal_params.multimodal_data["image"]["vision_normalization"]
    assert type(restored_spec) is type(spec) and restored_spec == spec
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_normalization_transport_process, args=(child, pickle.dumps(restored))
    )
    process.start()
    child.close()
    try:
        assert parent.poll(60), "Multimodal transport child did not respond"
        received_spec, received_pixels, received_normalized = parent.recv()
        process.join(60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
    assert type(received_spec) is type(spec) and received_spec == spec
    assert received_pixels == pixels.tolist()
    assert received_normalized == normalize_vision_pixels(pixels.reshape(3, -1), spec).tolist()
    unused = MultimodalParams(
        multimodal_data={"image": {"pixel_values": pixels, "vision_normalization": spec}}
    )
    unused.to_handle("multimodal_data")
    unused.release_handles()
    assert unused.multimodal_data["image"]["vision_normalization"] is spec


@pytest.mark.cpu_only
def test_normalization_transport_partial_encoder_cache() -> None:
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
    from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalItemPartition,
    )

    spec = _normalization_transport_spec()
    params = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(6, dtype=torch.uint8).view(2, 3),
                "vision_normalization": spec,
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", 0), ("image", 1)],
                output_embedding_lengths=[1, 1],
                input_partitions={"image": {"pixel_values": MultimodalItemPartition()}},
            ),
        }
    )
    model = MultimodalModelMixin()
    cache = TensorLRUCache(64)
    cache.put("hit", torch.ones(1, 2))
    partition = model.partition_encoder_cache(params, cache, keys=["hit", "miss"])
    assert partition.miss_indices == [1]

    def encode(residuals):
        assert len(residuals) == 1
        image = residuals[0].multimodal_data["image"]
        assert image["vision_normalization"] is spec
        assert image["pixel_values"].tolist() == [[3, 4, 5]]
        return torch.full((1, 2), 2.0)

    model.encode_multimodal_inputs = encode
    model._encode_with_partial_cache([(params, partition)], cache)
    assert params.multimodal_data["image"]["vision_normalization"] is spec
    assert torch.equal(
        params.multimodal_data["multimodal_embedding"], torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    )
    assert torch.equal(cache.get("miss"), torch.full((1, 2), 2.0))
    encoder = MagicMock(side_effect=AssertionError("Request-local cached output must be reused"))
    attached = modeling_multimodal_utils.get_multimodal_embeddings(encoder, [params])
    encoder.assert_not_called()
    assert torch.equal(attached[0], params.multimodal_data["multimodal_embedding"])
    assert params.multimodal_data["image"]["vision_normalization"] is spec
