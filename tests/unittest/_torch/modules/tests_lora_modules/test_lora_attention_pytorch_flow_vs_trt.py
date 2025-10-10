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
# LoRA Imports
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.layers import AttentionParams, KeyValueCacheParams
from tensorrt_llm.layers.lora import Lora, LoraParams
from tensorrt_llm.mapping import Mapping


class TestLoraAttentionPytorchFlowVsTRT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.seq_len = 16
        cls.head_num = 1
        cls.head_size = 64
        cls.hidden_size = cls.head_num * cls.head_size
        cls.dtype = 'float16'
        cls.torch_dtype = str_dtype_to_torch(cls.dtype)
        cls.device = torch.device('cuda')
        cls.pos_emb_type = PositionEmbeddingType.learned_absolute
        cls.causal_mask = True

    def _create_lora_params(self, ):
        lora_ranks_list = [8 for _ in range(self.batch_size)]

        lora_ranks = torch.Tensor(lora_ranks_list).to(torch.int32)

        lora_weight_ins = [
            torch.randn(self.hidden_size, lora_rank, device="cuda").to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]
        lora_weight_outs = [
            torch.randn(lora_rank, self.hidden_size, device="cuda").to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]

        lora_weight_ins = [tmp.contiguous() for tmp in lora_weight_ins]
        lora_weight_outs = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs
        ]

        lora_weights_pointers = []
        for in_ptr, out_ptr in zip(lora_weight_ins, lora_weight_outs):
            lora_weights_pointers.append(in_ptr.data_ptr())
            lora_weights_pointers.append(out_ptr.data_ptr())

        lora_weights_pointers = torch.LongTensor(lora_weights_pointers).to(
            torch.int64).reshape([self.batch_size, 2])

        return {
            'lora_ranks': lora_ranks,
            'lora_weights_pointers': lora_weights_pointers,
            'lora_weight_ins': lora_weight_ins,
            'lora_weight_outs': lora_weight_outs
        }

    def test_lora_attention(self):

        mean = 0.0
        std_dev = 0.02 if self.dtype == "float32" else 0.005

        hidden_states = torch.concat([
            torch.empty(size=[self.seq_len, self.hidden_size],
                        dtype=self.torch_dtype,
                        device=self.device).normal_(mean, std_dev)
            for _ in range(self.batch_size)
        ])

        context_lengths = torch.full([self.batch_size],
                                     self.seq_len,
                                     dtype=torch.int32,
                                     device=self.device)

        # Plugin specific setup - only generate 1 step
        max_seq_len = self.seq_len + 1

        # zero means "valid" token, one means invalid.
        host_past_key_value_lengths = torch.tensor([0] * self.batch_size,
                                                   dtype=torch.int32)

        # the max kv cache length for each layer. single tensor since we only have 1 layer here.
        host_max_attention_window_sizes = torch.tensor([max_seq_len],
                                                       dtype=torch.int32)
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)

        sequence_length = torch.full([self.batch_size],
                                     self.seq_len,
                                     dtype=torch.int32,
                                     device=self.device)

        # even in the the context phase, kv cache tensors can not be empty tensor for plugin, the actual shape info
        # otherwise, there will be cublas execution error.
        # are passed to plugin by the `sequence_length` tensor
        kv_shape = (self.batch_size, 2, self.head_num, max_seq_len,
                    self.head_size)
        past_key_value = torch.randn(kv_shape,
                                     dtype=self.torch_dtype,
                                     device=self.device)
        cache_indirection = torch.full((
            self.batch_size,
            1,
            max_seq_len,
        ),
                                       0,
                                       dtype=torch.int32,
                                       device=self.device)

        host_request_types = torch.tensor([0] * self.batch_size,
                                          dtype=torch.int32,
                                          device='cpu')

        perf_knob_tensor_size = 16
        host_runtime_perf_knobs_tensor = torch.tensor([-1] *
                                                      perf_knob_tensor_size,
                                                      dtype=torch.int64,
                                                      device='cpu')
        host_context_progress = torch.tensor([0],
                                             dtype=torch.int64,
                                             device='cpu')

        host_context_lengths = torch.Tensor(
            [self.seq_len for _ in range(self.batch_size)]).to(torch.int32)

        q_weight = torch.empty(size=[self.hidden_size, self.hidden_size],
                               dtype=self.torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # The initialization here is chosen to minimize computation after the
        # QKV BMMs in order to reduce the amount of differences from FP accumulation.
        # We set K and V weights to the identity matrix so that the input is copied
        # without doing any accumulation. Additionally, we set the output projection
        # to the identity for the same reason.
        # The main purpose of these tests is to check the QK^T BMM + Softmax + SV BMM for LoRA.
        eye_weight = torch.eye(self.hidden_size, dtype=self.torch_dtype)
        qkv_weight = torch.cat([q_weight, eye_weight, eye_weight], dim=-1)

        out_weight = eye_weight

        lora_params = self._create_lora_params()

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()
        net.plugin_config.gpt_attention_plugin = self.dtype  # for ragged input we use this plugin with remove_input_padding
        net.plugin_config.remove_input_padding = True
        net.plugin_config.lora_plugin = "float16"
        with tensorrt_llm.net_guard(net):
            trt_hidden_states = Tensor(name='hidden_states',
                                       shape=hidden_states.shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           self.dtype))
            context_lengths_tensor = Tensor(
                name='context_lengths',
                shape=context_lengths.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            host_context_lengths_tensor = Tensor(
                name='host_context_lengths',
                shape=[host_context_lengths.shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            lora_ranks_tensor = Tensor(
                name='lora_ranks',
                shape=(lora_params['lora_ranks'].shape[0], ),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            lora_weights_pointers_tensor = Tensor(
                name='lora_weights_pointers',
                shape=lora_params['lora_weights_pointers'].shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=host_request_types.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            past_key_value_tensor = Tensor(name='past_key_value',
                                           shape=tuple(past_key_value.shape),
                                           dtype=tensorrt_llm.str_dtype_to_trt(
                                               self.dtype))
            sequence_length_tensor = Tensor(
                name='sequence_length',
                shape=tuple(sequence_length.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_past_key_value_lengths_tensor = Tensor(
                name='host_past_key_value_lengths',
                shape=tuple(host_past_key_value_lengths.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_max_attention_window_sizes_tensor = Tensor(
                name='host_max_attention_window_sizes',
                shape=tuple(host_max_attention_window_sizes.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_sink_token_length_tensor = Tensor(
                name='host_sink_token_length',
                shape=tuple(host_sink_token_length.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            cache_indirection_tensor = Tensor(
                name='cache_indirection',
                shape=tuple(cache_indirection.shape),
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_runtime_perf_knobs = Tensor(
                name='host_runtime_perf_knobs',
                shape=[16],
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))
            host_context_progress_tensor = Tensor(
                name='host_context_progress',
                shape=[1],
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

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

            attn_layer = tensorrt_llm.layers.Attention(
                local_layer_idx=0,
                hidden_size=self.hidden_size,
                num_attention_heads=self.head_num,
                max_position_embeddings=self.seq_len,
                attention_mask_type=tensorrt_llm.layers.AttentionMaskType.
                causal,
                position_embedding_type=self.pos_emb_type,
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

            attn_layer.qkv.weight.value = np.ascontiguousarray(
                qkv_weight.cpu().numpy().transpose([1, 0]))
            attn_layer.dense.weight.value = np.ascontiguousarray(
                out_weight.cpu().numpy().transpose([1, 0]))

            output, present_key_value = attn_layer(
                trt_hidden_states,
                use_cache=True,
                lora_layer_params=
                lora_layer_params,  # Always use cache for plugin path in this test
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past_key_value_tensor],
                    host_past_key_value_lengths=
                    host_past_key_value_lengths_tensor,
                    host_max_attention_window_sizes=
                    host_max_attention_window_sizes_tensor,
                    host_sink_token_length=host_sink_token_length_tensor,
                    cache_indirection=cache_indirection_tensor),
                attention_params=AttentionParams(
                    sequence_length=sequence_length_tensor,
                    context_lengths=context_lengths_tensor,
                    host_request_types=host_request_types_tensor,
                    max_context_length=self.seq_len,
                    host_runtime_perf_knobs=host_runtime_perf_knobs,
                    host_context_progress=host_context_progress_tensor,
                    host_context_lengths=host_context_lengths_tensor,
                ))

            assert isinstance(output, Tensor)
            output.mark_output('output',
                               tensorrt_llm.str_dtype_to_trt(self.dtype))
            present_key_value.mark_output(
                'present_key_value', tensorrt_llm.str_dtype_to_trt(self.dtype))

        builder_config = builder.create_builder_config(name='attention_plugin',
                                                       precision=self.dtype)

        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

        stream = torch.cuda.current_stream().cuda_stream

        inputs = {
            'hidden_states': hidden_states,
            'past_key_value': past_key_value,
            'sequence_length': sequence_length,
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'host_max_attention_window_sizes': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'context_lengths': context_lengths,
            'host_request_types': host_request_types,
            'cache_indirection': cache_indirection,
            'host_runtime_perf_knobs': host_runtime_perf_knobs_tensor,
            'host_context_progress': host_context_progress,
            'host_context_lengths': host_context_lengths,
            'lora_ranks': lora_params['lora_ranks'],
            'lora_weights_pointers': lora_params['lora_weights_pointers'],
        }

        outputs = {
            'output':
            torch.empty(hidden_states.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(
                            self.dtype),
                        device=self.device),
            'present_key_value':
            past_key_value,
        }

        session.run(inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Pytorch flow
        llama_config = LlamaConfig(hidden_size=self.hidden_size,
                                   num_attention_heads=self.head_num,
                                   num_hidden_layers=1,
                                   intermediate_size=256,
                                   max_position_embeddings=512,
                                   rms_norm_eps=1e-5,
                                   vocab_size=32000,
                                   num_key_value_heads=self.head_num,
                                   dtype=self.torch_dtype)

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=max_seq_len)
        head_dim = llama_config.hidden_size // llama_config.num_attention_heads
        kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.
            CacheType.SELF,
            num_layers=llama_config.num_hidden_layers,
            num_kv_heads=llama_config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=128,
            max_seq_len=max_seq_len,
            max_batch_size=self.batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF)

        model_config = ModelConfig(pretrained_config=llama_config,
                                   attn_backend="VANILLA")
        attention_module = LlamaAttention(model_config, layer_idx=0).to(
            self.device).to(self.torch_dtype)

        attention_module.qkv_proj.weight.data = torch.from_numpy(
            np.ascontiguousarray(qkv_weight.cpu().numpy().transpose(
                [1, 0]))).to(self.device)
        attention_module.o_proj.weight.data = torch.from_numpy(
            np.ascontiguousarray(out_weight.cpu().numpy().transpose(
                [1, 0]))).to(self.device)

        request_ids = [0]

        kv_cache_manager.add_dummy_requests(request_ids=request_ids,
                                            token_nums=[self.seq_len])
        sequence_lengths = [self.seq_len]
        past_seen_tokens = [0]
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int32),
            num_contexts=len(sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=sequence_lengths,
            max_num_requests=self.batch_size,
            max_num_tokens=self.batch_size * self.seq_len,
        )


        lora_params_pytorch_flow = {
            'num_seqs': self.batch_size,
            'host_request_types':host_request_types,
            'prompt_lens_cpu': host_context_lengths,
            'remove_input_padding': True,
            0: {  # layer_idx
                LoraModuleType.ATTENTION_Q: {  # Module type
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                    lora_params['lora_weights_pointers'],
                },
                LoraModuleType.ATTENTION_K: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers': lora_params['lora_weights_pointers'],
                },
                LoraModuleType.ATTENTION_V: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                     lora_params['lora_weights_pointers'],
                },
                LoraModuleType.ATTENTION_DENSE: {
                    'adapter_size':
                    lora_params['lora_ranks'],
                    'weight_pointers':
                    lora_params['lora_weights_pointers'],
                }
            }
        }

        with torch.inference_mode():
            attn_metadata.prepare()

            pytorch_flow_output = attention_module(
                position_ids=None,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                lora_params=lora_params_pytorch_flow)

        trt_output = outputs['output']

        torch.testing.assert_close(pytorch_flow_output,
                                   trt_output,
                                   atol=2e-3,
                                   rtol=0)
