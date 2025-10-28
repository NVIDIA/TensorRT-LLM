import os
from dataclasses import dataclass
from typing import List

import torch
from _torch.helpers import create_mock_engine
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen2_5_VLConfig
from transformers import \
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.checkpoints.hf.qwen2vl_weight_mapper import \
    Qwen2VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5_VLModel
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

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
                          num_cached_tokens_per_seq: List[int] = None):

        trtllm_inputs = super().get_trtllm_inputs(input_ids,
                                                  multimodal_params_list,
                                                  is_gen,
                                                  num_cached_tokens_per_seq)

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
        else:
            # Mrope position ids
            mrope_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]
                    ["mrope_position_ids"])
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids

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
            mock_engine = create_mock_engine(1)
            # NOTE: Qwen2.5-VL model uses mrope
            mock_engine.use_mrope = True
            graph_runner = CUDAGraphRunner(mock_engine)
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

            # ==== Disable fuse rope scenarios ====
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=True,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),

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
        ]
        return scenarios

    def setup_scenario(self, scenario: TestQwen2_5_VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True)
