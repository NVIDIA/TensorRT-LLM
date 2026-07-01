# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import pytest
import test_modeling_qwen3vl as _qwen3vl
import torch
from test_modeling_qwen3vl import QWEN3_VL_8B_CONFIG
from test_modeling_qwen3vl import TestQwen3VLScenario as _TestQwen3VLScenario
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.configs import Cosmos3Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.cosmos3_weight_mapper import Cosmos3HfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_cosmos3 import Cosmos3Model
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

COSMOS3_NANO_PATH = os.path.join(llm_models_root(), "nvidia", "Cosmos3-Nano")

COSMOS3_TEST_CONFIG = copy.deepcopy(QWEN3_VL_8B_CONFIG)
COSMOS3_TEST_CONFIG.update(
    {
        "architectures": ["Cosmos3ForConditionalGeneration"],
        "model_type": "cosmos3",
        "_name_or_path": COSMOS3_NANO_PATH,
    }
)


# TestQwen3VL is a unittest.TestCase subclass, which pytest collects
# regardless of the ``Test*`` name filter. To prevent that we reference it only via
# through _qwen3vl module.
class TestCosmos3(_qwen3vl.TestQwen3VL):
    def setUp(self):
        if not os.path.isdir(os.path.join(COSMOS3_NANO_PATH, "transformer")):
            self.skipTest(f"Cosmos3-Nano checkpoint not found at {COSMOS3_NANO_PATH}")
        super().setUp()
        self._load_trtllm_checkpoint_weights(self.trtllm_model)

    def _load_trtllm_checkpoint_weights(self, model: Cosmos3Model) -> None:
        weight_mapper = Cosmos3HfWeightMapper()
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

    def get_model_config(self):
        return COSMOS3_TEST_CONFIG

    def get_trtllm_model_class(self):
        return Cosmos3Model

    def get_hf_model_class(self):
        return None

    def get_weight_mapper_class(self):
        return Cosmos3HfWeightMapper

    def get_model_type(self):
        return "cosmos3"

    def get_model_config_class(self):
        return Cosmos3Config

    @property
    def skip_hf_inference(self) -> bool:
        # The pinned transformers (5.5.x) does not register the Cosmos3
        # architecture, so there is no upstream HF model class to diff
        # against; we exercise the TRT-LLM path only here.
        return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cosmos3_init_preserves_caller_quant_config():
    """Building Cosmos3Model must not mutate the caller's quant_config."""
    hf_config = Cosmos3Config.from_dict(copy.deepcopy(COSMOS3_TEST_CONFIG))
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        quant_config=quant_config,
        skip_create_weights_in_init=True,
    )

    model = Cosmos3Model(model_config)

    # Outer LLM keeps the caller's quant_config unchanged (same object, same FP8 KV-cache setting).
    assert model.model_config.quant_config is quant_config
    assert model.model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

    # Vision encoder operates on an independent copy whose quant settings have been reset to the
    # defaults (no quantization).
    assert model.mm_encoder.model_config.quant_config is not quant_config
    assert model.mm_encoder.model_config.quant_config.kv_cache_quant_algo is None
    assert model.mm_encoder.model_config.quant_config.quant_algo is None
