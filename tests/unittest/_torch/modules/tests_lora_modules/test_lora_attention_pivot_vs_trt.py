import unittest

import numpy as np
import torch
from transformers import LlamaConfig

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaAttention
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.layers import (Attention, AttentionMaskType, AttentionParams,
                                 KeyValueCacheParams)
from tensorrt_llm.layers.lora import Lora, LoraParams
from tensorrt_llm.mapping import Mapping


class TestLoraAttentionPivotVsTRT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.seq_len = 8
        cls.hidden_size = 64
        cls.head_num = 1
        cls.num_hidden_layers = 1
        cls.dtype = 'float16'
        cls.torch_dtype = str_dtype_to_torch(cls.dtype)
        cls.device = torch.device('cuda')

        # KV cache parameters
        cls.num_blocks = 4
        cls.tokens_per_block = 32

        cls.llama_config = LlamaConfig(hidden_size=cls.hidden_size,
                                       num_attention_heads=cls.head_num,
                                       num_hidden_layers=cls.num_hidden_layers,
                                       intermediate_size=256,
                                       max_position_embeddings=512,
                                       rms_norm_eps=1e-5,
                                       vocab_size=32000,
                                       num_key_value_heads=cls.head_num,
                                       torch_dtype=cls.torch_dtype)

        # Create KV cache manager
        head_dim = cls.llama_config.hidden_size // cls.llama_config.num_attention_heads
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=cls.num_blocks *
                                        cls.tokens_per_block)
        cls.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.
            CacheType.SELF,
            num_layers=cls.llama_config.num_hidden_layers,
            num_kv_heads=cls.llama_config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=cls.tokens_per_block,
            max_seq_len=cls.num_blocks * cls.tokens_per_block,
            max_batch_size=cls.batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF)

    @classmethod
    def tearDownClass(cls):
        cls.kv_cache_manager.shutdown()

    def _create_attention_inputs(self):
        hidden_states = torch.empty(
            size=[self.batch_size, self.seq_len, self.hidden_size],
            dtype=self.torch_dtype,
            device='cuda')
        hidden_states.normal_(0.0, 0.02)

        # Create weights
        q_weight = torch.empty(size=[self.hidden_size, self.hidden_size],
                               dtype=self.torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # Set K and V and O weights to identity matrix
        eye_weight = torch.eye(self.hidden_size, dtype=self.torch_dtype)
        qkv_weight = torch.cat([q_weight, eye_weight, eye_weight], dim=-1)
        out_weight = eye_weight

        return hidden_states, qkv_weight, out_weight

    def _create_lora_params(self):
        lora_ranks_list = [8]

        host_context_lengths = torch.Tensor(
            [self.seq_len for _ in range(self.batch_size)]).to(torch.int32)
        lora_ranks = torch.Tensor(lora_ranks_list * self.batch_size).to(
            torch.int32)
        host_request_types = torch.zeros_like(host_context_lengths,
                                              device='cpu').int()

        lora_weight_ins = [
            torch.randn(self.hidden_size, lora_rank, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]
        lora_weight_outs = [
            torch.randn(lora_rank, self.hidden_size, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]

        lora_weight_ins = [tmp.contiguous() for tmp in lora_weight_ins]
        lora_weight_outs = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs
        ]

        # Create weight pointers for TensorRT
        lora_weights_pointers = []
        for in_ptr, out_ptr in zip(lora_weight_ins, lora_weight_outs):
            lora_weights_pointers.append(in_ptr.data_ptr())
            lora_weights_pointers.append(out_ptr.data_ptr())

        lora_weights_pointers = torch.LongTensor(lora_weights_pointers).to(
            torch.int64).reshape([self.batch_size, 2])

        return {
            'lora_ranks': lora_ranks,
            'host_context_lengths': host_context_lengths,
            'host_request_types': host_request_types,
            'lora_weights_pointers': lora_weights_pointers,
            'lora_weight_ins': lora_weight_ins,
            'lora_weight_outs': lora_weight_outs
        }

    def _setup_attention_module(self, qkv_weight, out_weight):
        """Set up the attention module with weights."""
        model_config = ModelConfig(pretrained_config=self.llama_config,
                                   attn_backend="VANILLA")
        layer_idx = 0
        attention_module = LlamaAttention(model_config, layer_idx=layer_idx).to(
            self.device).to(self.torch_dtype)

        # Set weights
        attention_module.qkv_proj.weight.data = torch.from_numpy(
            np.ascontiguousarray(qkv_weight.cpu().numpy().transpose(
                [1, 0]))).to(self.device)
        attention_module.o_proj.weight.data = torch.from_numpy(
            np.ascontiguousarray(out_weight.cpu().numpy().transpose(
                [1, 0]))).to(self.device)

        return attention_module, model_config

    def _create_attention_metadata(self, model_config):
        sequence_lengths = [self.seq_len]
        past_seen_tokens = [0]
        request_ids = [0]
        token_nums = [self.seq_len]
        prompt_lens = token_nums

        self.kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        return metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int32),
            num_contexts=len(sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=self.kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=self.batch_size,
            max_num_tokens=self.batch_size * self.seq_len,
        )

    def _setup_trt_network(self, hidden_states, lora_params, attention_module):
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.gpt_attention_plugin = self.dtype
        net.plugin_config.lora_plugin = self.dtype
        net.plugin_config.remove_input_padding = True
        net.plugin_config.paged_kv_cache = True

        with tensorrt_llm.net_guard(net):
            # Create LoRA tensors
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=[lora_params['host_request_types'].shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_lengths_tensor = Tensor(
                name='host_context_lengths',
                shape=[lora_params['host_context_lengths'].shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_ranks_tensor = Tensor(
                name='lora_ranks',
                shape=(lora_params['lora_ranks'].shape[0], ),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_weights_pointers_tensor = Tensor(
                name='lora_weights_pointers',
                shape=lora_params['lora_weights_pointers'].shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            # Create tensors for GPT Attention Plugin
            sequence_length_tensor = Tensor(
                name='sequence_length',
                shape=[self.batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            context_lengths_tensor = Tensor(
                name='context_lengths',
                shape=[self.batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_past_key_value_lengths_tensor = Tensor(
                name='host_past_key_value_lengths',
                shape=[self.batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_max_attention_window_sizes_tensor = Tensor(
                name='host_max_attention_window_sizes',
                shape=[self.num_hidden_layers],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_sink_token_length_tensor = Tensor(
                name='host_sink_token_length',
                shape=[self.num_hidden_layers],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            cache_indirection_tensor = Tensor(
                name='cache_indirection',
                shape=[self.batch_size, 1, 1],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            # Use dummy block offsets as we are in context phase and don't actually need KV cache values yet
            # Shape: [num_layers, batch_size, 2, max_blocks_per_seq]
            max_blocks_per_seq = (self.seq_len + self.tokens_per_block -
                                  1) // self.tokens_per_block
            kv_cache_block_offsets_tensor = Tensor(
                name='kv_cache_block_offsets',
                shape=[
                    self.num_hidden_layers, self.batch_size, 2,
                    max_blocks_per_seq
                ],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_kv_cache_block_offsets_tensor = Tensor(
                name='host_kv_cache_block_offsets',
                shape=[
                    self.num_hidden_layers, self.batch_size, 2,
                    max_blocks_per_seq
                ],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            # Add tensors for perf knobs and context progress
            host_runtime_perf_knobs_tensor = Tensor(
                name='host_runtime_perf_knobs',
                shape=[1],  # Typically a single int64 value
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))
            host_context_progress_tensor = Tensor(
                name='host_context_progress',
                shape=[self.batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            # Add tensors for paged kv cache pool management
            host_kv_cache_pool_pointers_tensor = Tensor(
                name='host_kv_cache_pool_pointers',
                shape=[2],  # Pointers to K and V pools
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))
            host_kv_cache_pool_mapping_tensor = Tensor(
                name='host_kv_cache_pool_mapping',
                shape=[self.batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            # Create LoRA parameters object
            lora_layer_params = LoraParams(
                lora_ranks=[{
                    "attn_q_lora_ranks": lora_ranks_tensor,
                    "attn_k_lora_ranks": lora_ranks_tensor,
                    "attn_v_lora_ranks": lora_ranks_tensor,
                    "attn_dense_lora_ranks": lora_ranks_tensor,
                }],
                lora_weights_pointers=[{
                    "attn_q_lora_weights_pointers":
                    lora_weights_pointers_tensor,
                    "attn_k_lora_weights_pointers":
                    lora_weights_pointers_tensor,
                    "attn_v_lora_weights_pointers":
                    lora_weights_pointers_tensor,
                    "attn_dense_lora_weights_pointers":
                    lora_weights_pointers_tensor,
                }],
                host_context_lengths=host_context_lengths_tensor,
                host_request_types=host_request_types_tensor,
            )

            # Create AttentionParams and KeyValueCacheParams
            attention_params = AttentionParams(
                sequence_length=sequence_length_tensor,
                context_lengths=context_lengths_tensor,
                host_context_lengths=
                host_context_lengths_tensor,  # Use the same tensor on host
                max_context_length=self.
                seq_len,  # Use current seq_len as max for context phase
                host_request_types=
                host_request_types_tensor,  # Use the same tensor on host
                host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
                host_context_progress=host_context_progress_tensor)

            kv_cache_params = KeyValueCacheParams(
                host_past_key_value_lengths=host_past_key_value_lengths_tensor,
                host_max_attention_window_sizes=
                host_max_attention_window_sizes_tensor,
                host_sink_token_length=host_sink_token_length_tensor,
                kv_cache_block_offsets=kv_cache_block_offsets_tensor,
                host_kv_cache_block_offsets=host_kv_cache_block_offsets_tensor,
                cache_indirection=cache_indirection_tensor,
                # past_key_value needs to be None for context phase
                past_key_value=None,
                # Add pool pointers and mapping
                host_kv_cache_pool_pointers=host_kv_cache_pool_pointers_tensor,
                host_kv_cache_pool_mapping=host_kv_cache_pool_mapping_tensor)

            attn_layer = Attention(
                local_layer_idx=0,
                hidden_size=hidden_states.shape[-1],
                num_attention_heads=1,
                num_kv_heads=1,  # Added num_kv_heads
                max_position_embeddings=self.llama_config.
                max_position_embeddings,  # Use config value
                attention_mask_type=AttentionMaskType.causal,
                bias=False)

            attn_layer.qkv_lora = Lora(
                in_hidden_size=attn_layer.hidden_size,
                out_hidden_sizes=[
                    attn_layer.num_attention_heads *
                    attn_layer.attention_head_size,
                    attn_layer.num_attention_kv_heads *
                    attn_layer.attention_head_size,
                    attn_layer.num_attention_kv_heads *
                    attn_layer.attention_head_size
                ],
                max_low_rank=8,
            )

            attn_layer.dense.lora = Lora(
                in_hidden_size=attn_layer.dense.in_features,
                out_hidden_sizes=[attn_layer.dense.out_features],
                max_low_rank=8,
            )

            # Set attention layer weights
            attn_layer.qkv.weight.value = attention_module.qkv_proj.weight.data
            attn_layer.dense.weight.value = attention_module.o_proj.weight.data

            # Create input tensor - already flattened to [numToken, dim]
            trt_hidden_states = Tensor(
                name='hidden_states',
                shape=hidden_states.reshape(-1, hidden_states.shape[-1]).shape,
                dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))

            # Update forward call for GPT Attention Plugin
            output, _ = attn_layer(  # GPT Attention Plugin returns a tuple (context, past_key_value)
                hidden_states=trt_hidden_states,
                lora_layer_params=lora_layer_params,  # Use the renamed object
                attention_params=attention_params,
                kv_cache_params=kv_cache_params,
                use_cache=True  # Must be True for GPT Attention Plugin
            )
            output.mark_output('output',
                               tensorrt_llm.str_dtype_to_trt(self.dtype))

        return builder, net

    def _run_trt_inference(self, builder, net, hidden_states, lora_params):
        builder_config = builder.create_builder_config(name='attention',
                                                       precision=self.dtype)
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

        stream = torch.cuda.current_stream().cuda_stream

        # Prepare inputs for GPT Attention Plugin
        sequence_length_tensor = torch.tensor([self.seq_len] * self.batch_size,
                                              dtype=torch.int32,
                                              device='cuda')
        context_lengths_tensor = torch.tensor([self.seq_len] * self.batch_size,
                                              dtype=torch.int32,
                                              device='cuda')
        host_past_key_value_lengths_tensor = torch.tensor(
            [0] * self.batch_size,
            dtype=torch.int32)  # Start from 0 for context phase
        max_seq_len = self.num_blocks * self.tokens_per_block
        host_max_attention_window_sizes_tensor = torch.tensor(
            [max_seq_len] * self.num_hidden_layers, dtype=torch.int32)
        host_sink_token_length_tensor = torch.tensor([0] *
                                                     self.num_hidden_layers,
                                                     dtype=torch.int32)
        cache_indirection_tensor = torch.arange(self.batch_size,
                                                dtype=torch.int32,
                                                device='cuda').reshape(
                                                    self.batch_size, 1, 1)
        # Create dummy block offsets for context phase
        max_blocks_per_seq = (self.seq_len + self.tokens_per_block -
                              1) // self.tokens_per_block
        shape = (self.num_hidden_layers, self.batch_size, 2, max_blocks_per_seq)
        kv_cache_block_offsets_tensor = torch.zeros(shape,
                                                    dtype=torch.int32,
                                                    device='cuda')
        host_kv_cache_block_offsets_tensor = torch.zeros(
            shape, dtype=torch.int32)  # Host copy
        # Add tensors for paged kv cache pool management (dummy values for context phase)
        # Get the actual pointers from the cache manager if needed for generation phase
        dummy_pool_pointers = torch.tensor([0, 0],
                                           dtype=torch.int64)  # Dummy pointers
        host_kv_cache_pool_pointers_tensor = dummy_pool_pointers
        host_kv_cache_pool_mapping_tensor = torch.zeros(
            [self.batch_size], dtype=torch.int32)  # Map all to pool 0
        host_runtime_perf_knobs_tensor = torch.tensor(
            [0], dtype=torch.int64)  # Default value
        host_context_progress_tensor = torch.zeros(
            [self.batch_size],
            dtype=torch.int32)  # Default value for context phase

        inputs = {
            'hidden_states': hidden_states.reshape(-1, hidden_states.shape[-1]),
            'host_request_types': lora_params['host_request_types'],
            'host_context_lengths': lora_params['host_context_lengths'],
            'lora_ranks': lora_params['lora_ranks'],
            'lora_weights_pointers': lora_params['lora_weights_pointers'],
            # Inputs for GPT Attention Plugin
            'sequence_length': sequence_length_tensor,
            'context_lengths': context_lengths_tensor,
            'host_past_key_value_lengths': host_past_key_value_lengths_tensor,
            'host_max_attention_window_sizes':
            host_max_attention_window_sizes_tensor,
            'host_sink_token_length': host_sink_token_length_tensor,
            'cache_indirection': cache_indirection_tensor,
            'kv_cache_block_offsets': kv_cache_block_offsets_tensor,
            'host_kv_cache_block_offsets': host_kv_cache_block_offsets_tensor,
            'host_runtime_perf_knobs': host_runtime_perf_knobs_tensor,
            'host_context_progress': host_context_progress_tensor,
            # Add pool pointers and mapping to inputs
            'host_kv_cache_pool_pointers': host_kv_cache_pool_pointers_tensor,
            'host_kv_cache_pool_mapping': host_kv_cache_pool_mapping_tensor,
        }

        outputs = {
            'output':
            # Output shape is [num_tokens, hidden_size] when remove_input_padding is True
            torch.empty(
                hidden_states.reshape(-1, hidden_states.shape[-1]).shape,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(self.dtype),
                device='cuda'),
        }

        session.run(inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Reshape output back to [batch_size, seq_len, hidden_size] for comparison
        return outputs['output'].reshape(hidden_states.shape)

    def test_attention_with_lora(self):
        hidden_states, qkv_weight, out_weight = self._create_attention_inputs()

        lora_params = self._create_lora_params()

        attention_module, model_config = self._setup_attention_module(
            qkv_weight, out_weight)

        attn_metadata = self._create_attention_metadata(model_config)
        builder, net = self._setup_trt_network(hidden_states, lora_params,
                                               attention_module)
        trt_output = self._run_trt_inference(builder, net, hidden_states,
                                             lora_params)

        lora_params_pivot = {
            'num_seqs': self.batch_size,
            'host_request_types': lora_params['host_request_types'],
            'prompt_lens_cpu': lora_params['host_context_lengths'],
            0: {  # layer_idx
                LoraModuleType.ATTENTION_Q: {  # Module type
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                    lora_params['lora_weights_pointers'],
                    'is_dora':
                    False,
                },
                LoraModuleType.ATTENTION_K: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers': lora_params['lora_weights_pointers'],
                    'is_dora':
                    False,
                },
                LoraModuleType.ATTENTION_V: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                     lora_params['lora_weights_pointers'],
                    'is_dora':
                    False,
                },
                LoraModuleType.ATTENTION_DENSE: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                    lora_params['lora_weights_pointers'],
                    'is_dora':
                    False,
                }
            }
        }

        with torch.inference_mode():
            attn_metadata.prepare()
            hidden_states_reshaped = hidden_states.reshape(
                -1, hidden_states.shape[-1])

            pivot_output = attention_module(
                position_ids=None,
                hidden_states=hidden_states_reshaped,
                attn_metadata=attn_metadata,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                lora_params=lora_params_pivot)

        torch.testing.assert_close(pivot_output, trt_output, atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
