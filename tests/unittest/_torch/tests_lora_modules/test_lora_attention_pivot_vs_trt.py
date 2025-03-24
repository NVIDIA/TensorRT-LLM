import os
import sys

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
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.layers import Attention
from tensorrt_llm.layers.lora import Lora, LoraParams
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), '../../functional'))



class TestLoraAttention:
    """Test class for LoRA attention implementation."""

    def setup_method(self, method):
        """Set up test parameters and resources."""
        # Test parameters
        self.batch_size = 1
        self.seq_len = 16
        self.hidden_size = 64
        self.head_num = 1
        self.num_hidden_layers = 1
        self.dtype = 'float16'
        self.torch_dtype = str_dtype_to_torch(self.dtype)
        self.device = torch.device('cuda')

        # KV cache parameters
        self.num_blocks = 4
        self.tokens_per_block = 32

        # Create model config
        self.llama_config = LlamaConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.head_num,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=256,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=self.head_num,
            torch_dtype=self.torch_dtype)

        # Create KV cache manager
        head_dim = self.llama_config.hidden_size // self.llama_config.num_attention_heads
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=self.num_blocks *
                                        self.tokens_per_block)
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.
            CacheType.SELF,
            num_layers=self.llama_config.num_hidden_layers,
            num_kv_heads=self.llama_config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=self.tokens_per_block,
            max_seq_len=self.num_blocks * self.tokens_per_block,
            max_batch_size=self.batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF)

    def teardown_method(self, method):
        """Clean up resources after each test."""
        if hasattr(self, 'kv_cache_manager'):
            self.kv_cache_manager.shutdown()

    def _create_attention_inputs(self):
        """Create input tensors and weights for attention."""
        # Create hidden states
        hidden_states = torch.empty(
            size=[self.batch_size, self.seq_len, self.hidden_size],
            dtype=self.torch_dtype,
            device='cuda')
        hidden_states.normal_(0.0, 0.02)

        # Create weights
        q_weight = torch.empty(size=[self.hidden_size, self.hidden_size],
                               dtype=self.torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # Set K and V weights to identity matrix
        eye_weight = torch.eye(self.hidden_size, dtype=self.torch_dtype)
        qkv_weight = torch.cat([q_weight, eye_weight, eye_weight], dim=-1)
        out_weight = eye_weight

        return hidden_states, qkv_weight, out_weight

    def _create_lora_params(self):
        """Create LoRA parameters and tensors."""
        lora_ranks_list = [8]

        host_context_lengths = torch.Tensor(
            [self.seq_len for _ in range(self.batch_size)]).to(torch.int32)
        lora_ranks = torch.Tensor(lora_ranks_list).to(torch.int32)
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

        lora_weight_ins = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_ins
        ]
        lora_weight_outs = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs
        ]

        lora_weights_pointers = []
        for in_ptr, out_ptr in zip(lora_weight_ins, lora_weight_outs):
            lora_weights_pointers.append(in_ptr.data_ptr())
            lora_weights_pointers.append(out_ptr.data_ptr())
            lora_weights_pointers.append(0)  # null dora scale

        lora_weights_pointers = torch.LongTensor(lora_weights_pointers).to(
            torch.int64).reshape([self.batch_size, 3])

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
        """Create attention metadata for the test."""
        sequence_lengths = [self.seq_len]
        past_seen_tokens = [0]
        request_ids = [0]
        token_nums = [self.seq_len]
        prompt_lens = token_nums

        # Add requests to KV cache manager
        self.kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Create attention metadata
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
        """Set up TensorRT network with attention layer."""
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()
        net.plugin_config.lora_plugin = self.dtype

        with tensorrt_llm.net_guard(net):
            # Create input tensor
            trt_hidden_states = Tensor(name='hidden_states',
                                       shape=hidden_states.shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           self.dtype))

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
                shape=[lora_params['lora_ranks'].shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_weights_pointers_tensor = Tensor(
                name='lora_weights_pointers',
                shape=lora_params['lora_weights_pointers'].shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            # Create LoRA parameters
            lora_params = LoraParams(
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
                host_request_types=host_request_types_tensor)

            # Create attention layer
            attn_layer = Attention(
                local_layer_idx=0,
                hidden_size=hidden_states.shape[-1],
                num_attention_heads=1,
                max_position_embeddings=hidden_states.shape[1],
                attention_mask_type=tensorrt_llm.layers.AttentionMaskType.
                causal,
                bias=False)

            # Add LoRA to attention layer
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

            # Run attention
            output = attn_layer(hidden_states=trt_hidden_states,
                                lora_layer_params=lora_params)
            output.mark_output('output',
                               tensorrt_llm.str_dtype_to_trt(self.dtype))

        return builder, net

    def _run_trt_inference(self, builder, net, hidden_states, lora_params):
        """Run TensorRT inference."""
        builder_config = builder.create_builder_config(name='attention',
                                                       precision=self.dtype)
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

        stream = torch.cuda.current_stream().cuda_stream
        inputs = {
            'hidden_states': hidden_states,
            'host_request_types': lora_params['host_request_types'],
            'host_context_lengths': lora_params['host_context_lengths'],
            'lora_ranks': lora_params['lora_ranks'],
            'lora_weights_pointers': lora_params['lora_weights_pointers'],
        }

        outputs = {
            'output':
            torch.empty(hidden_states.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(
                            self.dtype),
                        device='cuda'),
        }

        session.run(inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        return outputs['output'].squeeze(0)

    def test_attention_with_lora(self):
        """Test attention with LoRA weights."""
        # Create inputs and weights
        hidden_states, qkv_weight, out_weight = self._create_attention_inputs()

        # Create LoRA parameters
        lora_params = self._create_lora_params()

        # Set up attention module
        attention_module, model_config = self._setup_attention_module(
            qkv_weight, out_weight)

        # Create attention metadata
        attn_metadata = self._create_attention_metadata(model_config)

        # Create LoRA parameters for pivot attention
        lora_params_dict = {
            "lora_weight_ins_q": lora_params['lora_weight_ins'][0],
            "lora_weight_outs_q": lora_params['lora_weight_outs'][0],
            "lora_weight_ins_k": lora_params['lora_weight_ins'][0],
            "lora_weight_outs_k": lora_params['lora_weight_outs'][0],
            "lora_weight_ins_v": lora_params['lora_weight_ins'][0],
            "lora_weight_outs_v": lora_params['lora_weight_outs'][0],
            "lora_weight_ins_o": lora_params['lora_weight_ins'][0],
            "lora_weight_outs_o": lora_params['lora_weight_outs'][0]
        }

        # Run pivot attention
        with torch.inference_mode():
            attn_metadata.prepare()
            hidden_states_pivot = hidden_states.squeeze(0)
            pivot_output = attention_module(
                position_ids=None,
                hidden_states=hidden_states_pivot,
                attn_metadata=attn_metadata,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                lora_params=lora_params_dict)

        # Set up and run TensorRT network
        builder, net = self._setup_trt_network(hidden_states, lora_params,
                                               attention_module)
        trt_output = self._run_trt_inference(builder, net, hidden_states,
                                             lora_params)

        # Compare outputs
        a_tol = 5e-5 if (self.dtype == "float32") else 2e-3
        np.testing.assert_allclose(pivot_output.cpu().numpy(),
                                   trt_output.cpu().numpy(),
                                   atol=a_tol,
                                   verbose=True)
        print("Test passed!")


if __name__ == "__main__":
    test = TestLoraAttention()
    test.setup_method(None)  # None is passed as method parameter
    test.test_attention_with_lora()
    test.teardown_method(None)  # None is passed as method parameter
