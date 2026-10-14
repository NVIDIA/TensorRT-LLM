# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import test_modeling_qwen3vl as _qwen3vl
import torch
from test_modeling_qwen3vl import TestQwen3VLScenario as _TestQwen3VLScenario
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.configs import Cosmos3EdgeConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.cosmos3_edge_weight_mapper import (
    Cosmos3EdgeHfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_cosmos3_edge import (
    Cosmos3EdgeModel,
    merge_spatial_patches,
    resize_positional_embeddings_packed,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLVisionPatchMerger
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


def test_siglip_position_interpolation_keeps_packed_token_count():
    grid = 4
    hidden = 8
    positional = torch.arange(grid * grid * hidden, dtype=torch.float32).reshape(grid, grid, hidden)
    shapes = torch.tensor([[2, 2], [8, 4]], dtype=torch.int64)
    lengths = [4, 32]
    packed = resize_positional_embeddings_packed(positional, shapes, lengths)
    assert packed.shape == (36, hidden)
    assert packed.dtype == positional.dtype


def test_spatial_merge_packs_two_by_two_blocks():
    frames, height, width, hidden = 2, 4, 4, 3
    embeds = torch.arange(frames * height * width * hidden, dtype=torch.float32).reshape(
        frames * height * width, hidden
    )
    grid = torch.tensor([[frames, height, width]], dtype=torch.int64)
    merged = merge_spatial_patches(embeds, grid, merge_size=2)
    assert merged.shape == (frames * (height // 2) * (width // 2), 2 * 2 * hidden)


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
