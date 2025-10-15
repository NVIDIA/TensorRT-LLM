import re
import unittest
from copy import deepcopy

import pytest
import torch
from _torch.helpers import create_mock_engine
from parameterized import parameterized
from test_modeling_llama import Scenario, reduce_llama_config
from transformers import MllamaConfig
from transformers import \
    MllamaForConditionalGeneration as HFMllamaForConditionalGeneration

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_mllama import \
    MllamaForConditionalGeneration
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

LLAMA_3_2_11B_VISION_CONFIG = {
    'architectures': ['MllamaForConditionalGeneration'],
    'image_token_index': 128256,
    'model_type': 'mllama',
    'text_config': {
        '_name_or_path': '',
        'add_cross_attention': False,
        'architectures': None,
        'bad_words_ids': None,
        'begin_suppress_tokens': None,
        'bos_token_id': 128000,
        'chunk_size_feed_forward': 0,
        'cross_attention_hidden_size': None,
        'cross_attention_layers': [3, 8, 13, 18, 23, 28, 33, 38],
        'decoder_start_token_id': None,
        'diversity_penalty': 0.0,
        'do_sample': False,
        'dropout': 0,
        'early_stopping': False,
        'encoder_no_repeat_ngram_size': 0,
        'eos_token_id': 128001,
        'exponential_decay_length_penalty': None,
        'finetuning_task': None,
        'forced_bos_token_id': None,
        'forced_eos_token_id': None,
        'hidden_act': 'silu',
        'hidden_size': 4096,
        'id2label': {
            '0': 'LABEL_0',
            '1': 'LABEL_1'
        },
        'initializer_range': 0.02,
        'intermediate_size': 14336,
        'is_decoder': False,
        'is_encoder_decoder': False,
        'label2id': {
            'LABEL_0': 0,
            'LABEL_1': 1
        },
        'length_penalty': 1.0,
        'max_length': 20,
        'max_position_embeddings': 131072,
        'min_length': 0,
        'model_type': 'mllama_text_model',
        'no_repeat_ngram_size': 0,
        'num_attention_heads': 32,
        'num_beam_groups': 1,
        'num_beams': 1,
        'num_hidden_layers': 40,
        'num_key_value_heads': 8,
        'num_return_sequences': 1,
        'output_attentions': False,
        'output_hidden_states': False,
        'output_scores': False,
        'pad_token_id': 128004,
        'prefix': None,
        'problem_type': None,
        'pruned_heads': {},
        'remove_invalid_values': False,
        'repetition_penalty': 1.0,
        'return_dict': True,
        'return_dict_in_generate': False,
        'rms_norm_eps': 1e-05,
        'rope_scaling': {
            'factor': 8.0,
            'high_freq_factor': 4.0,
            'low_freq_factor': 1.0,
            'original_max_position_embeddings': 8192,
            'rope_type': 'llama3'
        },
        'rope_theta': 500000.0,
        'sep_token_id': None,
        'suppress_tokens': None,
        'task_specific_params': None,
        'temperature': 1.0,
        'tf_legacy_loss': False,
        'tie_encoder_decoder': False,
        'tie_word_embeddings': False,
        'tokenizer_class': None,
        'top_k': 50,
        'top_p': 1.0,
        'torch_dtype': 'bfloat16',
        'torchscript': False,
        'typical_p': 1.0,
        'use_bfloat16': False,
        'use_cache': True,
        'vocab_size': 128256
    },
    'torch_dtype': 'bfloat16',
    'transformers_version': '4.45.0.dev0',
    'vision_config': {
        '_name_or_path':
        '',
        'add_cross_attention':
        False,
        'architectures':
        None,
        'attention_heads':
        16,
        'bad_words_ids':
        None,
        'begin_suppress_tokens':
        None,
        'bos_token_id':
        None,
        'chunk_size_feed_forward':
        0,
        'cross_attention_hidden_size':
        None,
        'decoder_start_token_id':
        None,
        'diversity_penalty':
        0.0,
        'do_sample':
        False,
        'early_stopping':
        False,
        'encoder_no_repeat_ngram_size':
        0,
        'eos_token_id':
        None,
        'exponential_decay_length_penalty':
        None,
        'finetuning_task':
        None,
        'forced_bos_token_id':
        None,
        'forced_eos_token_id':
        None,
        'hidden_act':
        'gelu',
        'hidden_size':
        1280,
        'id2label': {
            '0': 'LABEL_0',
            '1': 'LABEL_1'
        },
        'image_size':
        448,
        'intermediate_layers_indices': [3, 7, 15, 23, 30],
        'intermediate_size':
        5120,
        'is_decoder':
        False,
        'is_encoder_decoder':
        False,
        'label2id': {
            'LABEL_0': 0,
            'LABEL_1': 1
        },
        'length_penalty':
        1.0,
        'max_length':
        20,
        'max_num_tiles':
        4,
        'min_length':
        0,
        'model_type':
        'mllama_vision_model',
        'no_repeat_ngram_size':
        0,
        'norm_eps':
        1e-05,
        'num_beam_groups':
        1,
        'num_beams':
        1,
        'num_channels':
        3,
        'num_global_layers':
        8,
        'num_hidden_layers':
        32,
        'num_return_sequences':
        1,
        'output_attentions':
        False,
        'output_hidden_states':
        False,
        'output_scores':
        False,
        'pad_token_id':
        None,
        'patch_size':
        14,
        'prefix':
        None,
        'problem_type':
        None,
        'pruned_heads': {},
        'remove_invalid_values':
        False,
        'repetition_penalty':
        1.0,
        'return_dict':
        True,
        'return_dict_in_generate':
        False,
        'sep_token_id':
        None,
        'supported_aspect_ratios': [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1],
                                    [2, 2], [3, 1], [4, 1]],
        'suppress_tokens':
        None,
        'task_specific_params':
        None,
        'temperature':
        1.0,
        'tf_legacy_loss':
        False,
        'tie_encoder_decoder':
        False,
        'tie_word_embeddings':
        True,
        'tokenizer_class':
        None,
        'top_k':
        50,
        'top_p':
        1.0,
        'torch_dtype':
        'bfloat16',
        'torchscript':
        False,
        'typical_p':
        1.0,
        'use_bfloat16':
        False,
        'vision_output_dim':
        7680
    }
}


def convert_weights_names(weights: dict) -> dict:
    # Since transformers version >= 4.52.0, the default model architecture is changed.
    # We need to convert the weight names accordingly to match TRTLLM naming.
    _checkpoint_conversion_mapping = {
        "^model.language_model": "language_model.model",
        "^model.vision_model": "vision_model",
        "^model.multi_modal_projector": "multi_modal_projector",
        "^lm_head": "language_model.lm_head",
    }
    converted_weights = {}
    for weight_name, weight_value in weights.items():
        new_name = weight_name
        for pattern, replacement in _checkpoint_conversion_mapping.items():
            new_name = re.sub(pattern, replacement, new_name)
        converted_weights[new_name] = weight_value
    return converted_weights


class TestMLlama(unittest.TestCase):

    @parameterized.expand([
        Scenario(backend="VANILLA"),
        Scenario(backend="FLASHINFER"),
        Scenario(backend="FLASHINFER", use_cuda_graph=True),
        Scenario(backend="TRTLLM"),
        Scenario(backend="TRTLLM", use_cuda_graph=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_mllama_allclose_to_hf_text_only(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        if scenario.backend == "FLASHINFER":
            pytest.skip("https://nvbugspro.nvidia.com/bug/5458945")
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(LLAMA_3_2_11B_VISION_CONFIG)
        dtype = MllamaConfig.from_dict(config_dict['text_config']).torch_dtype

        dtype_bytes = dtype.itemsize

        # 11B * sizeof(float16) plus some extra for activations (1.3x approx).
        # MLllama also have vision encoder part. Just use 11B as upper bound.
        activation_factor = 1.3
        model_params = 11 * (10**9)
        mem_for_full_model = 2 * model_params * dtype_bytes * activation_factor

        reduce_llama_config(mem_for_full_model, config_dict['text_config'], 8)
        if config_dict['text_config']['num_hidden_layers'] <= 0:
            self.skipTest('Insufficient memory for a single Llama layer')
        mllama_config = MllamaConfig.from_dict(config_dict)

        # For text path only, downscale vision encoder to only 1 layer.
        config_dict['vision_config']['num_hidden_layers'] = 1

        device = torch.device('cuda')

        hf_mllama = HFMllamaForConditionalGeneration(mllama_config).to(
            dtype).to(device).eval()

        mllama = MllamaForConditionalGeneration(
            ModelConfig(pretrained_config=mllama_config,
                        attn_backend=backend)).to(dtype).to(device)
        weights = convert_weights_names(hf_mllama.state_dict())
        mllama.load_weights(weights)

        # KV cache setup
        num_blocks = 1
        tokens_per_block = 128
        head_dim = mllama.config.hidden_size // mllama.config.num_attention_heads
        num_layers = mllama.config.num_hidden_layers
        num_kv_heads = mllama.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

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
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        # context
        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)
        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens)

        # Note: no CUDA graphs for prefill, the graph runner is built for
        # decoding only.
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = mllama.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attn_metadata=attn_metadata)
            ref = hf_mllama.forward(input_ids=input_ids.unsqueeze(0),
                                    position_ids=position_ids,
                                    use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.3,
                                   rtol=0.3)

        # gen
        gen_input_ids = torch.tensor([600], dtype=torch.int, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens)

        gen_position_ids = [
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        graph_runner = None
        if scenario.use_cuda_graph:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return mllama.forward(input_ids=input_ids,
                                      position_ids=position_ids,
                                      attn_metadata=attn_metadata)
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(key,
                                     lambda inputs: mllama.forward(**inputs),
                                     inputs)

                for _ in range(2):
                    # Run it twice. This helps us catch problems if buffers are accidentally reallocated
                    # in prepare().
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key, inputs)
                return logits

        if scenario.use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(input_ids=gen_input_ids,
                                 position_ids=gen_position_ids,
                                 attn_metadata=attn_metadata)
            ref = hf_mllama.forward(input_ids=gen_input_ids.unsqueeze(0),
                                    position_ids=gen_position_ids,
                                    past_key_values=ref.past_key_values,
                                    use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.3,
                                   rtol=0.3)
        if graph_runner is not None:
            graph_runner.clear()
        kv_cache_manager.shutdown()
