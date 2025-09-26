import unittest
from copy import deepcopy
from typing import Any

import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_vila import (VilaConfig, VilaModel,
                                                      fuse_input_embeds)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

VILA_1_5_3B_CONFIG = {
    "_name_or_path": "Efficient-Large-Model/VILA1.5-3b",
    "architectures": ["LlavaLlamaModel"],
    "drop_path_rate": 0.0,
    "hidden_size": 2560,
    "image_aspect_ratio": "resize",
    "interpolate_mode": "linear",
    "llm_cfg": {
        "_name_or_path": "./llm",
        "add_cross_attention": False,
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": 1,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 6912,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 4096,
        "min_length": 0,
        "model_max_length": 4096,
        "model_type": "llama",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 20,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 32,
        "num_key_value_heads": 20,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 0,
        "prefix": None,
        "pretraining_tp": 1,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": False,
        "tokenizer_class": None,
        "tokenizer_model_max_length": 4096,
        "tokenizer_padding_side": "right",
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "use_cache": True,
        "vocab_size": 32000
    },
    "mm_hidden_size": 1152,
    "mm_projector_cfg": {
        "_name_or_path": "./mm_projector",
        "add_cross_attention": False,
        "architectures": ["MultimodalProjector"],
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "mm_projector_type": "mlp_downsample",
        "model_type": "v2l_projector",
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False
    },
    "mm_projector_lr": None,
    "mm_use_im_patch_token": False,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "cls_patch",
    "mm_vision_select_layer": -2,
    "model_dtype": "torch.bfloat16",
    "model_type": "llava_llama",
    "num_video_frames": 8,
    "resume_path": "./vlm",
    "s2": False,
    "s2_max_split_size": 336,
    "s2_scales": "336,672,1008",
    "transformers_version": "4.36.2",
    "tune_language_model": True,
    "tune_mm_projector": True,
    "tune_vision_tower": True,
    "vision_resolution": -1,
    "vision_tower_cfg": {
        "_name_or_path": "./vision_tower",
        "add_cross_attention": False,
        "architectures": ["SiglipVisionModel"],
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "image_size": 384,
        "intermediate_size": 4304,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "layer_norm_eps": 1e-06,
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "model_type": "siglip_vision_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 16,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "patch_size": 14,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False
    }
}

NVILA_8B_CONFIG = {
    "_attn_implementation_autoset": True,
    "_name_or_path": "Efficient-Large-Model/NVILA-8B",
    "architectures": ["LlavaLlamaModel"],
    "chat_template": None,
    "drop_path_rate": 0.0,
    "dynamic_s2": True,
    "fps": 0.0,
    "hidden_size": 3584,
    "image_aspect_ratio": "dynamic_s2",
    "interpolate_mode": "linear",
    "llm_cfg": {
        "_attn_implementation_autoset": False,
        "_name_or_path":
        "runs/train/qwen25-8B-dynamic_s2-stage3-20241126000711/model/llm",
        "add_cross_attention": False,
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": 151643,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 151645,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "silu",
        "hidden_size": 3584,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 18944,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 32768,
        "max_window_layers": 28,
        "min_length": 0,
        "model_max_length": 8192,
        "model_type": "qwen2",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 28,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sep_token_id": None,
        "sliding_window": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": False,
        "tokenizer_class": None,
        "tokenizer_model_max_length": 8192,
        "tokenizer_padding_side": "right",
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151648
    },
    "mm_hidden_size": 3456,
    "mm_projector_cfg": {
        "_attn_implementation_autoset": False,
        "_name_or_path":
        "runs/train/qwen25-8B-dynamic_s2-stage3-20241126000711/model/mm_projector",
        "add_cross_attention": False,
        "architectures": ["MultimodalProjector"],
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "mm_projector_type": "mlp_downsample",
        "model_type": "v2l_projector",
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False
    },
    "mm_projector_lr": None,
    "mm_use_im_patch_token": True,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "cls_patch",
    "mm_vision_select_layer": -2,
    "model_dtype": "torch.bfloat16",
    "model_type": "llava_llama",
    "num_time_tokens": 0,
    "num_video_frames": 8,
    "resume_path":
    "runs/train/qwen25-8B-dynamic_s2-stage3-20241126000711/model",
    "s2": False,
    "s2_max_split_size": 448,
    "s2_resize_output_to_scale_idx": -1,
    "s2_scales": "448,896,1344",
    "soft_ce_std": 1.0,
    "time_token_format": "<t{t}>",
    "time_token_ids": [],
    "transformers_version": "4.46.0",
    "tune_language_model": True,
    "tune_mm_projector": True,
    "tune_vision_tower": True,
    "vision_resolution": -1,
    "vision_tower_cfg": {
        "_attn_implementation_autoset": False,
        "_name_or_path":
        "runs/train/qwen25-8B-dynamic_s2-stage3-20241126000711/model/vision_tower",
        "add_cross_attention": False,
        "architectures": ["SiglipVisionModel"],
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "image_size": 448,
        "intermediate_size": 4304,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "layer_norm_eps": 1e-06,
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "model_type": "siglip_vision_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 16,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "num_image_tokens": 256,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "patch_size": 14,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 2048,
        "projector_hidden_act": "gelu_fast",
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "vision_use_head": False,
    }
}


def reduce_vila_config(mem_for_full_model: int, config_dict: dict[str, Any]):
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict['llm_cfg']["num_hidden_layers"] *
                         model_fraction)
        num_layers = min(num_layers, 32)
        config_dict['llm_cfg']["num_hidden_layers"] = num_layers


class TestVila(unittest.TestCase):

    @parameterized.expand([(VILA_1_5_3B_CONFIG, 3), (NVILA_8B_CONFIG, 8)])
    def test_vila_sanity(self, config_dict, param_cnt):
        model, input_ids, position_ids, past_seen_tokens, attn_metadata, kv_cache_manager = \
            self._prepare_sanity_test(config_dict, param_cnt)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata,
                                   return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    def test_vila_fuse_input_embeds(self):
        config_dict = deepcopy(VILA_1_5_3B_CONFIG)
        model, _, _, _, _, _ = self._prepare_sanity_test(config_dict, 3)

        device = torch.device('cuda')
        dtype = model.model_dtype

        input_ids = torch.tensor([
            1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082,
            20255, 29889, 12968, 29901, 29871, 32000, 32001, 32002, 32003,
            32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012,
            32013, 32014, 32015, 32016, 32017, 32018, 32019, 32020, 32021,
            32022, 32023, 32024, 32025, 32026, 32027, 32028, 32029, 32030,
            32031, 32032, 32033, 32034, 32035, 32036, 32037, 32038, 32039,
            32040, 32041, 32042, 32043, 32044, 32045, 32046, 32047, 32048,
            32049, 32050, 32051, 32052, 32053, 32054, 32055, 32056, 32057,
            32058, 32059, 32060, 32061, 32062, 32063, 32064, 32065, 32066,
            32067, 32068, 32069, 32070, 32071, 32072, 32073, 32074, 32075,
            32076, 32077, 32078, 32079, 32080, 32081, 32082, 32083, 32084,
            32085, 32086, 32087, 32088, 32089, 32090, 32091, 32092, 32093,
            32094, 32095, 32096, 32097, 32098, 32099, 32100, 32101, 32102,
            32103, 32104, 32105, 32106, 32107, 32108, 32109, 32110, 32111,
            32112, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120,
            32121, 32122, 32123, 32124, 32125, 32126, 32127, 32128, 32129,
            32130, 32131, 32132, 32133, 32134, 32135, 32136, 32137, 32138,
            32139, 32140, 32141, 32142, 32143, 32144, 32145, 32146, 32147,
            32148, 32149, 32150, 32151, 32152, 32153, 32154, 32155, 32156,
            32157, 32158, 32159, 32160, 32161, 32162, 32163, 32164, 32165,
            32166, 32167, 32168, 32169, 32170, 32171, 32172, 32173, 32174,
            32175, 32176, 32177, 32178, 32179, 32180, 32181, 32182, 32183,
            32184, 32185, 32186, 32187, 32188, 32189, 32190, 32191, 32192,
            32193, 32194, 32195, 29871, 320, 29876, 20355, 915, 278, 1203, 322,
            278, 14826, 4195, 297, 278, 1967, 29889, 2277, 29937, 7900, 22137,
            29901, 450
        ],
                                 device=device,
                                 dtype=torch.int)
        images = [torch.rand(196, 2560, dtype=dtype, device=device)]
        input_ids, input_embeds = fuse_input_embeds(
            model.llm.model.embed_tokens, input_ids, images)
        self.assertIsNone(input_ids)
        self.assertEqual(list(input_embeds.shape), [233, 2560])

    def _prepare_sanity_test(self, config_dict, param_cnt):
        config_dict = deepcopy(config_dict)
        # (param_cnt)B * sizeof(float16) plus some extra for activations
        mem_for_full_model = (2 + 1) * param_cnt * 2**(30)
        reduce_vila_config(mem_for_full_model, config_dict)
        if config_dict['llm_cfg']["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Llava layer")
        vila_config = VilaConfig.from_dict(config_dict)
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=vila_config,
                                   quant_config=None)
        model = VilaModel(model_config).to(device)

        dtype = model.model_dtype

        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        batch_size = len(sequence_lengths)
        request_ids = list(range(batch_size))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        num_kv_heads = model.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=batch_size,
            max_num_tokens=8192,
        )

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_lengths[i] if i < len(
                context_sequence_lengths) else 1
            position_id = torch.arange(tokens,
                                       tokens + seq_len,
                                       device=input_ids.device)
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)
        return model, input_ids, position_ids, past_seen_tokens, attn_metadata, kv_cache_manager
