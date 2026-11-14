# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import test_modeling_qwen3vl as _qwen3vl
import torch
from test_modeling_qwen3vl import TestQwen3VLScenario as _TestQwen3VLScenario
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.rotary_embedding import MRotaryEmbedding
from tensorrt_llm._torch.configs import Cosmos3EdgeConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.cosmos3_edge_weight_mapper import (
    Cosmos3EdgeHfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_cosmos3_edge import (
    Cosmos3EdgeAttention,
    Cosmos3EdgeImageProcessor,
    Cosmos3EdgeInputProcessor,
    Cosmos3EdgeModel,
    _edge_positional_embedding_params,
    merge_spatial_patches,
    resize_positional_embeddings_packed,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLVisionPatchMerger
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def cosmos3_edge_checkpoint() -> str:
    """Resolve the Edge checkpoint under LLM_MODELS_ROOT.

    The CI cache stores it as ``Cosmos3-Edge``. A ``nvidia/Cosmos3-Edge`` layout
    is also accepted.
    """
    root = llm_models_root()
    if root is None:
        return ""
    for relative in ("Cosmos3-Edge", os.path.join("nvidia", "Cosmos3-Edge")):
        candidate = os.path.join(root, relative)
        if os.path.isdir(os.path.join(candidate, "transformer")):
            return candidate
    return os.path.join(root, "Cosmos3-Edge")


@pytest.mark.cpu_only
def test_edge_mapper_drops_generator_weights_and_renames_reasoner_keys():
    mapper = Cosmos3EdgeHfWeightMapper()
    weights = {
        "layers.0.self_attn.to_q.weight": torch.empty(1),
        "layers.0.self_attn.to_k.weight": torch.empty(1),
        "layers.0.self_attn.to_v.weight": torch.empty(1),
        "layers.0.self_attn.to_out.weight": torch.empty(1),
        "layers.0.self_attn.k_norm_und_for_gen.weight": torch.empty(1),
        "layers.0.self_attn.add_q_proj.weight": torch.empty(1),
        "layers.0.mlp_moe_gen.up_proj.weight": torch.empty(1),
        "layers.0.mlp.up_proj.weight": torch.empty(1),
        "embed_tokens.weight": torch.empty(1),
        "norm.weight": torch.empty(1),
        "norm_moe_gen.weight": torch.empty(1),
        "lm_head.weight": torch.empty(1),
        "model.visual.encoder.layers.0.self_attn.out_proj.weight": torch.empty(1),
        "model.visual.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
        "model.visual.encoder.layers.0.mlp.fc2.weight": torch.empty(1),
        "model.projector.linear_fc1.weight": torch.empty(1),
        "time_embedder.linear_1.weight": torch.empty(1),
    }
    mapped = mapper.preprocess_weights(weights)
    assert "model.layers.0.self_attn.q_proj.weight" in mapped
    assert "model.layers.0.self_attn.k_proj.weight" in mapped
    assert "model.layers.0.self_attn.v_proj.weight" in mapped
    assert "model.layers.0.self_attn.o_proj.weight" in mapped
    assert "model.layers.0.mlp.up_proj.weight" in mapped
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "lm_head.weight" in mapped
    assert "model.visual.encoder.layers.0.self_attn.o_proj.weight" in mapped
    assert "model.visual.encoder.layers.0.mlp.up_proj.weight" in mapped
    assert "model.visual.encoder.layers.0.mlp.down_proj.weight" in mapped
    assert (
        mapped["model.projector.linear_fc1.weight"] is weights["model.projector.linear_fc1.weight"]
    )
    dropped = [
        "layers.0.self_attn.k_norm_und_for_gen.weight",
        "layers.0.self_attn.add_q_proj.weight",
        "layers.0.mlp_moe_gen.up_proj.weight",
        "norm_moe_gen.weight",
        "time_embedder.linear_1.weight",
        "model.layers.0.self_attn.k_norm_und_for_gen.weight",
        "model.layers.0.self_attn.add_q_proj.weight",
        "model.layers.0.mlp_moe_gen.up_proj.weight",
        "model.norm_moe_gen.weight",
    ]
    assert all(key not in mapped for key in dropped)
    renamed_away = [
        "model.visual.encoder.layers.0.mlp.fc1.weight",
        "model.visual.encoder.layers.0.self_attn.out_proj.weight",
    ]
    assert all(key not in mapped for key in renamed_away)
    assert len(mapped) == 12


@pytest.mark.cpu_only
def test_siglip_position_interpolation_keeps_packed_token_count():
    grid = 4
    hidden = 8
    positional = torch.arange(grid * grid * hidden, dtype=torch.float32).reshape(grid, grid, hidden)
    shapes = torch.tensor([[2, 2], [8, 4]], dtype=torch.int64)
    lengths = [4, 32]
    packed = resize_positional_embeddings_packed(positional, shapes, lengths)
    assert packed.shape == (36, hidden)
    assert packed.dtype == positional.dtype


@pytest.mark.cpu_only
def test_spatial_merge_packs_two_by_two_blocks():
    frames, height, width, hidden = 2, 4, 4, 3
    embeds = torch.arange(frames * height * width * hidden, dtype=torch.float32).reshape(
        frames * height * width, hidden
    )
    grid = torch.tensor([[frames, height, width]], dtype=torch.int64)
    merged = merge_spatial_patches(embeds, grid, merge_size=2)
    assert merged.shape == (frames * (height // 2) * (width // 2), 2 * 2 * hidden)


class _FakeTokenizer:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __call__(self, prompt, return_tensors=None):
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.input_ids]))


@pytest.mark.cpu_only
def test_edge_disagg_handoff_without_deepstack():
    image_token_id = 19
    vision_start = 20
    vision_end = 21
    processor = object.__new__(Cosmos3EdgeInputProcessor)
    processor._config = SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=18,
        vision_start_token_id=vision_start,
        text_config=SimpleNamespace(hidden_size=16),
        vision_config=SimpleNamespace(),
    )
    processor._tokenizer = _FakeTokenizer([11, vision_start, image_token_id, vision_end, 12])

    prompt_token_ids, params = processor.build_disagg_prefill_multimodal_inputs(
        {"prompt": "image prompt"},
        [{"tensor_size": (2, 16)}],
    )

    assert prompt_token_ids == [
        11,
        vision_start,
        image_token_id,
        image_token_id,
        vision_end,
        12,
    ]
    assert params.item_metadata.output_embedding_lengths == (2,)


@pytest.mark.cpu_only
def test_edge_grid_stays_off_the_device_path_and_encoder_cache_is_enabled():
    paths = Cosmos3EdgeModel.multimodal_data_device_paths.fget(None)
    assert "image.pixel_values" in paths
    assert "image.image_grid_thw" not in paths
    assert "video.video_grid_thw" not in paths
    assert Cosmos3EdgeModel.supports_encoder_cache is True


@pytest.mark.cpu_only
def test_edge_rope_scale_follows_rope_type():
    from tensorrt_llm._torch.configs.cosmos3_edge import Cosmos3EdgeTextConfig

    config = Cosmos3EdgeTextConfig(
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1e8,
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
    )
    params = _edge_positional_embedding_params(config)
    assert params.type == PositionEmbeddingType.mrope
    assert params.mrope_interleaved is True
    assert params.rope.scale_type == RotaryScalingType.yarn
    assert params.rope.scale == 2.0


def _resize_size(size):
    height = getattr(size, "height", None)
    width = getattr(size, "width", None)
    if height is None:
        height = size["height"]
        width = size["width"]
    return int(height), int(width)


@pytest.mark.cpu_only
def test_edge_resize_keeps_a_factor_on_each_side():
    processor = object.__new__(Cosmos3EdgeImageProcessor)
    processor.merge_size = 2
    processor.patch_size = 16
    processor.size = {"shortest_edge": 1024, "longest_edge": 65536}
    captured = {}

    def capture_resize(image, size, resample):
        del image, resample
        captured["size"] = size
        return torch.empty(1)

    processor.resize = capture_resize
    processor._resize_image(
        torch.empty(3, 40, 8000),
        max_ratio=200,
        resample=None,
        min_pixels=1024,
        max_pixels=65536,
    )
    height, width = _resize_size(captured["size"])
    assert height >= 32
    assert width >= 32
    assert height % 32 == 0
    assert width % 32 == 0


def _enable_encoder_cache(model, max_bytes: int):
    model.model_config._frozen = False
    model.model_config.multimodal_config = MultimodalConfig(encoder_cache_max_bytes=max_bytes)
    assert model.encoder_cache_active
    cache = model._initialize_multimodal_encoder_cache(max_bytes)
    assert cache is not None
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_edge_to_device_keeps_grids_on_cpu():
    paths = Cosmos3EdgeModel.multimodal_data_device_paths.fget(None)
    params = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.randn(4, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            "video": {
                "pixel_values_videos": torch.randn(4, 8),
                "video_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            "mrope_config": {
                "mrope_position_ids": torch.zeros(3, 1, 4),
                "mrope_position_deltas": torch.zeros(1, 1),
            },
        }
    )
    params.to_device("multimodal_data", device="cuda", pin_memory=True, target_keywords=paths)
    data = params.multimodal_data
    assert data["image"]["pixel_values"].device.type == "cuda"
    assert data["video"]["pixel_values_videos"].device.type == "cuda"
    assert data["mrope_config"]["mrope_position_ids"].device.type == "cuda"
    assert data["mrope_config"]["mrope_position_deltas"].device.type == "cuda"
    assert data["image"]["image_grid_thw"].device.type == "cpu"
    assert data["video"]["video_grid_thw"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_edge_yarn_fused_cache_matches_decoder():
    from tensorrt_llm._torch.configs.cosmos3_edge import Cosmos3EdgeTextConfig

    text = Cosmos3EdgeTextConfig(
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1e8,
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
    )
    text.disable_fuse_rope = True
    attention = Cosmos3EdgeAttention(ModelConfig(pretrained_config=text), layer_idx=0)
    holder = SimpleNamespace()
    Cosmos3EdgeModel.init_mrope_embedding(
        holder,
        SimpleNamespace(pretrained_config=SimpleNamespace(text_config=text)),
    )
    assert attention.rotary_emb is not None
    decoder_table = attention.rotary_emb.rotary_cos_sin.cuda()
    fused_table = holder.rotary_emb.rotary_cos_sin.cuda()
    torch.testing.assert_close(decoder_table, fused_table)
    uncorrected = MRotaryEmbedding(
        RopeParams.from_config(text),
        head_dim=attention.head_dim,
        mrope_section=attention.pos_embd_params.mrope_section,
        mrope_interleaved=attention.pos_embd_params.mrope_interleaved,
    )
    assert not torch.allclose(uncorrected.rotary_cos_sin.cuda().float(), fused_table.float())


@pytest.mark.cpu_only
def test_patch_merger_intermediate_width_is_optional():
    assert Qwen3VLVisionPatchMerger.__init__.__defaults__ is not None
    # (use_postshuffle_norm, merger_intermediate_size, norm_eps)
    assert Qwen3VLVisionPatchMerger.__init__.__defaults__[-2:] == (None, None)


# TestQwen3VL is a unittest.TestCase. Reference it through the module so pytest
# does not collect it again from this file.
class TestCosmos3Edge(_qwen3vl.TestQwen3VL):
    def setUp(self):
        checkpoint = cosmos3_edge_checkpoint()
        if not os.path.isdir(os.path.join(checkpoint, "transformer")):
            self.skipTest(f"Cosmos3-Edge checkpoint not found at {checkpoint}")
        self.edge_checkpoint = checkpoint
        super().setUp()
        self._load_trtllm_checkpoint_weights(self.trtllm_model)

    def _load_trtllm_checkpoint_weights(self, model: Cosmos3EdgeModel) -> None:
        weight_mapper = Cosmos3EdgeHfWeightMapper()
        weight_mapper.init_model_and_config(model, self.hf_config)
        weights = HfWeightLoader().load_weights(
            model.llm_checkpoint_dir,
            model.model_config.mapping,
        )
        model.load_weights(weights, weight_mapper)
        for module in model.modules():
            if hasattr(module, "post_load_weights") and not getattr(
                module, "_weights_removed", False
            ):
                module.post_load_weights()

    def setup_scenario(self, scenario: _TestQwen3VLScenario):
        # Bypass TestQwen3VL.setup_scenario: with skip_hf_inference the dummy
        # HF module has no weights, so disable_fuse_rope must not reload them.
        super(_qwen3vl.TestQwen3VL, self).setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=False,
                disable_fuse_rope=True,
            )
            self._load_trtllm_checkpoint_weights(self.trtllm_model)

    def assert_outputs_finite(self, outputs: torch.Tensor) -> None:
        self.assertIsNotNone(outputs)
        self.assertTrue(
            torch.isfinite(outputs).all().item(),
            "forward outputs contain non-finite values",
        )

    def test_encoder_output_reaches_prefill_through_cache(self):
        model = self.trtllm_model
        _enable_encoder_cache(model, 1 << 20)
        vision = model.mm_encoder.visual.config
        merge = int(vision.spatial_merge_size)
        height = width = merge * 2
        patch_dim = int(vision.num_channels) * int(vision.patch_size) ** 2
        rows = (height // merge) * (width // merge)
        pixels = torch.randn(height * width, patch_dim, device="cuda")
        grid = torch.tensor([[1, height, width]], dtype=torch.int64)
        model.mm_encoder.setup_attn_metadata(max_num_tokens=max(64, height * width))

        calls = {"n": 0}
        visual_forward = model.mm_encoder.visual.forward

        def counted_forward(*args, **kwargs):
            calls["n"] += 1
            return visual_forward(*args, **kwargs)

        model.mm_encoder.visual.forward = counted_forward
        try:

            def image_param():
                return MultimodalParams(
                    multimodal_input=MultimodalInput(
                        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                        multimodal_positions=[1],
                        multimodal_lengths=[rows],
                    ),
                    multimodal_data={
                        "image": {"pixel_values": pixels, "image_grid_thw": grid},
                        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                            item_refs=[("image", 0)], output_embedding_lengths=[rows]
                        ),
                        "mm_processor_kwargs_hash": "edge-review",
                    },
                    multimodal_runtime=MultimodalRuntimeData(
                        embed_mask_cumsum=torch.arange(1, rows + 1, dtype=torch.int64),
                        past_seen_token_num=0,
                        chunk_end_pos=rows,
                    ),
                )

            first = model._get_or_encode_multimodal_embeddings([image_param()])
            self.assertEqual(tuple(first.shape), (rows, model.embedding_dim))
            self.assertEqual(calls["n"], 1)
            second = model._get_or_encode_multimodal_embeddings([image_param()])
            self.assertEqual(calls["n"], 1)
            torch.testing.assert_close(second, first)
        finally:
            model.mm_encoder.visual.forward = visual_forward

        image_token_id = self.hf_config.image_token_id
        vision_start = self.hf_config.vision_start_token_id
        vision_end = self.hf_config.vision_end_token_id
        processor = object.__new__(Cosmos3EdgeInputProcessor)
        processor._config = self.hf_config
        processor._tokenizer = _FakeTokenizer([11, vision_start, image_token_id, vision_end, 12])
        prompt_token_ids, params = processor.build_disagg_prefill_multimodal_inputs(
            {"prompt": "image"},
            [{"tensor_size": tuple(first.shape)}],
        )
        self.assertEqual(params.item_metadata.output_embedding_lengths, (rows,))
        params.multimodal_data["multimodal_embedding"] = first
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device="cuda")
        image_positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
        text_positions = torch.nonzero(input_ids != image_token_id, as_tuple=False).flatten()
        self.assertEqual(image_positions.numel(), rows)
        prepared = model.prepare_multimodal_inputs(
            input_ids=input_ids,
            positions=None,
            multimodal_params=[params],
            num_context_requests=1,
            text_token_indices=text_positions,
            mm_token_indices=image_positions,
        )
        self.assertIsNotNone(prepared.inputs_embeds)
        torch.testing.assert_close(
            prepared.inputs_embeds[image_positions],
            first.to(dtype=prepared.inputs_embeds.dtype),
        )

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        logits = super().run_trtllm_forward(trtllm_inputs, use_cuda_graph=use_cuda_graph)
        self.assert_outputs_finite(logits)
        return logits

    def create_hf_config(self):
        return Cosmos3EdgeConfig.from_pretrained(self.edge_checkpoint)

    def get_model_config(self):
        return {"torch_dtype": "bfloat16", "_name_or_path": self.edge_checkpoint}

    def get_trtllm_model_class(self):
        return Cosmos3EdgeModel

    def get_hf_model_class(self):
        return None

    def get_weight_mapper_class(self):
        return Cosmos3EdgeHfWeightMapper

    def get_model_type(self):
        return "cosmos3_edge"

    def get_model_config_class(self):
        return Cosmos3EdgeConfig

    @property
    def skip_hf_inference(self) -> bool:
        # The pinned transformers release has no Cosmos3-Edge class.
        return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cosmos3_edge_init_preserves_caller_quant_config():
    """Building Cosmos3EdgeModel must not mutate the caller's quant_config."""
    checkpoint = cosmos3_edge_checkpoint()
    if not os.path.isdir(os.path.join(checkpoint, "transformer")):
        pytest.skip(f"Cosmos3-Edge checkpoint not found at {checkpoint}")
    hf_config = Cosmos3EdgeConfig.from_pretrained(checkpoint)
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        quant_config=quant_config,
        skip_create_weights_in_init=True,
    )

    model = Cosmos3EdgeModel(model_config)

    assert model.model_config.quant_config is quant_config
    assert model.model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert model.mm_encoder.model_config.quant_config is not quant_config
    assert model.mm_encoder.model_config.quant_config.kv_cache_quant_algo is None
    assert model.mm_encoder.model_config.quant_config.quant_algo is None
