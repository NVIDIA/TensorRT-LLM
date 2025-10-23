import os
import unittest
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
import torch
from _torch.helpers import create_mock_engine
from parameterized import parameterized
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLConfig
from transformers import \
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalLM

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen2vl_weight_mapper import \
    Qwen2VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5_VLModel
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.inputs import (create_input_processor,
                                 default_multimodal_input_loader, prompt_inputs)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.mapping import Mapping


def llm_models_root() -> str:
    '''return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path
    '''
    DEFAULT_LLM_MODEL_ROOT = os.path.join("/scratch.trt_llm_data", "llm-models")
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", DEFAULT_LLM_MODEL_ROOT)

    return LLM_MODELS_ROOT


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
class Scenario:
    modality: str = "image"
    use_cuda_graph: bool = False
    disable_fuse_rope: bool = False

    def __repr__(self) -> str:
        return f"modality:{self.modality.lower()}-use_cuda_graph:{self.use_cuda_graph}-disable_fuse_rope:{self.disable_fuse_rope}"


class TestQwen2_5_VL(unittest.TestCase):

    def get_test_inputs(self, modality: str):

        test_data_root = Path(
            os.path.join(llm_models_root(), "multimodals", "test_data"))

        if modality == "image":
            return ["Describe the natural environment in the image."], \
                [str(test_data_root / "seashore.png")]
        elif modality == "multiple_image":
            return ["Describe the difference between the two images."], \
                [str(test_data_root / "inpaint.png"),
                 str(test_data_root / "61.jpg")]
        elif modality == "video":
            return ["Tell me what you see in the video briefly."], \
                [str(test_data_root / "OAI-sora-tokyo-walk.mp4")]
        elif modality == "mixture_text_image":
            return ["Describe the scene in the image briefly.",
                    "Who invented the internet?"], \
                [str(test_data_root / "inpaint.png"),
                 ""]
        elif modality == "text":
            return ["Who invented the internet?"], []
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_kv_cache_manager(self, dtype: torch.dtype, config: Qwen2_5_VLConfig,
                             tokens_per_block: int, max_seq_len: int,
                             batch_size: int, num_blocks: int):
        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        return kv_cache_manager

    def get_inputprocessor(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return create_input_processor(model_path, tokenizer=tokenizer)

    def get_trtllm_inputs(self, model, modality: str, device: torch.device,
                          prompt: List[str], media: List[str]):
        processor = self.get_inputprocessor(
            model.model_config.pretrained_config._name_or_path)
        inputs = default_multimodal_input_loader(
            tokenizer=processor.tokenizer,
            model_dir=model.model_config.pretrained_config._name_or_path,
            model_type="qwen2_5_vl",
            modality=modality,
            prompts=prompt,
            media=media,
            image_data_format="pt",
            num_frames=8,
            device="cpu")
        inputs = [prompt_inputs(i) for i in inputs]

        input_ids = []
        context_sequence_lengths = []
        multimodal_params_list = []
        for input in inputs:
            prompt_token_ids, extra_processed_inputs = processor(
                input, sampling_params=None)
            input_ids.extend(prompt_token_ids)
            context_sequence_lengths.append(len(prompt_token_ids))
            multimodal_params = MultimodalParams(
                multimodal_data=extra_processed_inputs.get('multimodal_data'))
            multimodal_params.to_device(
                "multimodal_data",
                device,
                pin_memory=True,
                target_keywords=getattr(model, "multimodal_data_device_paths",
                                        None))
            multimodal_params_list.append(multimodal_params)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        return input_ids, context_sequence_lengths, multimodal_params_list

    def get_hf_inputs(self, model, modality: str, device: torch.device,
                      prompt: List[str], media: List[str]):
        processor = AutoProcessor.from_pretrained(
            model.model_config.pretrained_config._name_or_path)
        inputs = default_multimodal_input_loader(
            tokenizer=processor.tokenizer,
            model_dir=model.model_config.pretrained_config._name_or_path,
            model_type="qwen2_5_vl",
            modality=modality,
            prompts=prompt,
            media=media,
            image_data_format="pt",
            num_frames=8,
            device="cpu")
        inputs = [prompt_inputs(i) for i in inputs]

        images = None
        videos = None

        if modality in ["image", "multiple_image", "mixture_text_image"]:
            images = [input['multi_modal_data']['image'] for input in inputs]
        elif modality == "video":
            videos = [
                input['multi_modal_data'][f'{modality}'] for input in inputs
            ]
        elif modality == "text":
            # For text-only modality, no images or videos needed
            pass
        else:
            raise ValueError(f"Invalid modality: {modality}")

        processor_inputs = processor(
            text=[input['prompt'] for input in inputs],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        ).to(device)
        return processor_inputs

    @pytest.mark.skip(reason="https://nvbugs/5550722")
    def test_qwen2_5_vl_sanity(self):

        config_dict = deepcopy(QWEN2_5_VL_7B_CONFIG)
        qwen2_5_vl_config = Qwen2_5_VLConfig.from_dict(config_dict)
        dtype = qwen2_5_vl_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=qwen2_5_vl_config)
        qwen2_5_vl = Qwen2_5_VLModel(model_config,
                                     disable_fuse_rope=False).to(device)

        prompt, media = self.get_test_inputs("image")
        prompt, media = prompt * 2, media * 2
        input_ids, context_sequence_lengths, multimodal_params_list = self.get_trtllm_inputs(
            qwen2_5_vl, "image", device, prompt, media)

        # Add generation input_ids
        input_ids = torch.cat([
            input_ids,
            torch.tensor([900, 1000], dtype=torch.int32, device=device)
        ],
                              dim=0)
        multimodal_params_list.extend([
            MultimodalParams(
                multimodal_data={
                    "mrope_config": {
                        "mrope_position_deltas":
                        torch.zeros(1, dtype=torch.int32, device='cuda')
                    }
                })
        ] * 2)

        generation_sequence_lengths = [1, 1]
        sequence_lengths = context_sequence_lengths + generation_sequence_lengths
        past_seen_tokens = [0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:2] + past_seen_tokens[2:]

        num_blocks = 100
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2
        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=qwen2_5_vl_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)
        kv_cache_manager.add_dummy_requests(request_ids,
                                            token_nums,
                                            use_mrope=True)

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
            max_num_requests=len(context_sequence_lengths) + 2,
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

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen2_5_vl.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                multimodal_params=multimodal_params_list)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen2_5_vl.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
                multimodal_params=multimodal_params_list)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand([
        Scenario(modality="image",
                 use_cuda_graph=False,
                 disable_fuse_rope=False),
        Scenario(modality="image", use_cuda_graph=True,
                 disable_fuse_rope=False),
        Scenario(modality="image", use_cuda_graph=False,
                 disable_fuse_rope=True),
        Scenario(modality="multiple_image",
                 use_cuda_graph=False,
                 disable_fuse_rope=False),
        Scenario(modality="video",
                 use_cuda_graph=False,
                 disable_fuse_rope=False),
    ])
    @pytest.mark.skip(reason="https://nvbugs/5550722")
    @torch.no_grad()
    def test_qwen2_5_vl_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF.
        """
        torch.random.manual_seed(0)
        config_dict = deepcopy(QWEN2_5_VL_7B_CONFIG)
        qwen2_5_vl_config = Qwen2_5_VLConfig.from_dict(config_dict)

        dtype = qwen2_5_vl_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=qwen2_5_vl_config)
        qwen2_5_vl = Qwen2_5_VLModel(
            model_config,
            disable_fuse_rope=scenario.disable_fuse_rope).to(device)

        num_blocks = 5
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        hf_qwen2_5_vl = HFQwen2_5_VLForConditionalLM(qwen2_5_vl_config).to(
            dtype).to(device).eval()

        weight_mapper = Qwen2VLHfWeightMapper()
        weight_mapper.init_model_and_config(qwen2_5_vl, model_config)
        qwen2_5_vl.load_weights(hf_qwen2_5_vl.state_dict(), weight_mapper)

        prompt, media = self.get_test_inputs(scenario.modality)
        # Context phase.
        input_ids, context_sequence_lengths, multimodal_params_list = self.get_trtllm_inputs(
            qwen2_5_vl, scenario.modality, device, prompt, media)

        max_seq_len = max(input_ids.size(-1) + 1, max_seq_len)
        num_blocks = max_seq_len // tokens_per_block + 1
        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=qwen2_5_vl_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids,
                                            token_nums,
                                            use_mrope=True)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        seq_lens = torch.tensor([input_ids.size(-1)], dtype=torch.int)
        attn_metadata = metadata_cls(
            seq_lens=seq_lens,
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192 if not scenario.modality == "video" else 16384,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_seq_len=seq_lens + 1,
        )

        # Mrope position ids
        position_ids = multimodal_params_list[0].multimodal_data[
            "mrope_config"]["mrope_position_ids"]
        position_ids = position_ids.cuda()

        hf_inputs = self.get_hf_inputs(qwen2_5_vl, scenario.modality, device,
                                       prompt, media)
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen2_5_vl.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                multimodal_params=multimodal_params_list)
            # NOTE: HF creates its own position_ids inside forward() call
            ref = hf_qwen2_5_vl.forward(**hf_inputs, use_cache=True)
            torch.testing.assert_close(logits,
                                       ref.logits[:, -1].float(),
                                       atol=0.4,
                                       rtol=0.4)

        # Generation phase.
        gen_input_ids = torch.tensor([900], dtype=torch.int, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )
        mrope_gen_position_ids = multimodal_params_list[0].multimodal_data[
            "mrope_config"]["mrope_position_deltas"]
        gen_position_ids = [
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1),
                         dtype=torch.int32)
        ]
        gen_position_ids = torch.cat(gen_position_ids)
        gen_position_ids = (gen_position_ids + mrope_gen_position_ids).expand(
            3, 1, 1).cuda()
        gen_multimodal_params_list = []
        for multimodal_param in multimodal_params_list:
            multimodal_param.to_device(
                "multimodal_data",
                device,
                pin_memory=True,
                target_keywords=["mrope_config.mrope_position_deltas"])
            gen_multimodal_params_list.append(multimodal_param)

        graph_runner = None
        if scenario.use_cuda_graph:
            mock_engine = create_mock_engine(1)
            mock_engine.use_mrope = True
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata,
                        multimodal_params):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return qwen2_5_vl.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          multimodal_params=multimodal_params)
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                    "multimodal_params": multimodal_params,
                }
                key = (1, 0, False)
                graph_runner.capture(
                    key=key,
                    forward_fn=lambda inputs: qwen2_5_vl.forward(**inputs),
                    initial_inputs=inputs)

                for _ in range(2):
                    # Run it twice. This helps us catch problems if buffers are accidentally reallocated
                    # in prepare().
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key=key, current_inputs=inputs)
                return logits

        if scenario.use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = run_forward(input_ids=gen_input_ids,
                                 position_ids=gen_position_ids,
                                 attn_metadata=attn_metadata,
                                 multimodal_params=gen_multimodal_params_list)
            ref = hf_qwen2_5_vl.forward(input_ids=gen_input_ids.unsqueeze(0),
                                        position_ids=gen_position_ids,
                                        past_key_values=ref.past_key_values,
                                        use_cache=True)
            torch.testing.assert_close(logits,
                                       ref.logits[:, -1].float(),
                                       atol=0.4,
                                       rtol=0.4)

        kv_cache_manager.shutdown()
