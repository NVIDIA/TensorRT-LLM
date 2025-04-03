import os
import sys
import unittest

import numpy as np
import torch
from transformers import LlamaConfig

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaAttention
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils'))
from torch_ref import attention_qkvpacked_ref


class TestLoraAttentionPivotVsVanilla(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.seq_len = 16
        cls.hidden_size = 64
        cls.head_num = 1
        cls.num_hidden_layers = 1
        cls.dtype = torch.float16
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
                                       torch_dtype=cls.dtype)

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

    def _get_lora_params(self, in_dim, out_dim):
        lora_rank = 8
        lora_weight_ins = torch.randn(in_dim,
                                      lora_rank,
                                      device=self.device,
                                      dtype=self.dtype)
        lora_weight_outs = torch.randn(lora_rank,
                                       out_dim,
                                       device=self.device,
                                       dtype=self.dtype)
        return lora_weight_ins, lora_weight_outs

    def _create_attention_inputs(self):
        hidden_states = torch.randn(self.seq_len,
                                    self.llama_config.hidden_size,
                                    dtype=self.dtype,
                                    device=self.device)

        q_weight = torch.empty(size=[self.hidden_size, self.hidden_size],
                               dtype=self.dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # Set K and V weights to identity matrix
        eye_weight = torch.eye(self.hidden_size, dtype=self.dtype)
        qkv_weight = torch.cat([q_weight, eye_weight, eye_weight], dim=-1)
        out_weight = eye_weight

        return hidden_states, qkv_weight, out_weight

    def _setup_attention_module(self, qkv_weight, out_weight):
        """Set up the attention module with weights."""
        model_config = ModelConfig(pretrained_config=self.llama_config,
                                   attn_backend="VANILLA")
        attention_module = LlamaAttention(model_config, layer_idx=0).to(
            self.device).to(self.dtype)

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

        # Add requests to KV cache manager
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

    def _run_vanilla_attention(self,
                               hidden_states,
                               qkv_weight,
                               lora_params=None):
        head_dim = self.hidden_size // self.head_num

        # Base QKV computation
        packed_torch_qkv = hidden_states.to("cuda") @ qkv_weight.to("cuda")

        if lora_params:
            # Get the LoRA weights from the new structure
            dense_params = lora_params[0][
                LoraModuleType.
                ATTENTION_DENSE]  # TODO (dafrimi) 0 is the layer_idx, needs to pass it here somehow
            Q_params = lora_params[0][LoraModuleType.ATTENTION_Q]
            K_params = lora_params[0][LoraModuleType.ATTENTION_K]
            V_params = lora_params[0][LoraModuleType.ATTENTION_V]

            A_q, B_q = Q_params['weight_tensors'][0], Q_params[
                'weight_tensors'][1]
            A_k, B_k = K_params['weight_tensors'][0], K_params[
                'weight_tensors'][1]
            A_v, B_v = V_params['weight_tensors'][0], V_params[
                'weight_tensors'][1]
            A_o, B_o = dense_params['weight_tensors'][0], dense_params[
                'weight_tensors'][1]

            # Apply LoRA
            lora_output_q = (hidden_states @ B_q.T) @ A_q.T
            lora_output_k = (hidden_states @ B_k.T) @ A_k.T
            lora_output_v = (hidden_states @ B_v.T) @ A_v.T

            packed_lora_torch_qkv = torch.cat(
                [lora_output_q, lora_output_k, lora_output_v], dim=-1)
            packed_lora_torch_qkv = packed_torch_qkv + packed_lora_torch_qkv

            packed_lora_torch_qkv = packed_lora_torch_qkv.reshape(
                [self.batch_size, self.seq_len, 3, self.head_num, head_dim])

            mha_out_lora, _ = attention_qkvpacked_ref(packed_lora_torch_qkv,
                                                      causal=True,
                                                      upcast=False,
                                                      bias=None)

            torch_out = mha_out_lora.reshape(
                [self.batch_size, self.seq_len, self.hidden_size])
            torch_out = torch_out.squeeze(0)

            # Apply output LoRA and skip projection of O matrix since it's identity
            lora_o = (torch_out @ B_o.T) @ A_o.T
            torch_out = torch_out + lora_o
        else:
            # Run vanilla attention without LoRA
            packed_torch_qkv = packed_torch_qkv.reshape(
                [self.batch_size, self.seq_len, 3, self.head_num, head_dim])

            mha_out, _ = attention_qkvpacked_ref(packed_torch_qkv,
                                                 causal=True,
                                                 upcast=False,
                                                 bias=None)

            torch_out = mha_out.reshape(
                [self.batch_size, self.seq_len, self.hidden_size])
            torch_out = torch_out.squeeze(0)

        return torch_out

    def _run_pivot_attention(self, attention_module, hidden_states,
                             attn_metadata, lora_params):
        with torch.inference_mode():
            attn_metadata.prepare()
            return attention_module(
                position_ids=None,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                lora_params=lora_params)

    def test_attention_with_lora(self):
        hidden_states, qkv_weight, out_weight = self._create_attention_inputs()

        # Create LoRA parameters
        A_q, B_q = self._get_lora_params(self.hidden_size, self.hidden_size)
        A_k, B_k = self._get_lora_params(self.hidden_size, self.hidden_size)
        A_v, B_v = self._get_lora_params(self.hidden_size, self.hidden_size)
        A_o, B_o = self._get_lora_params(self.hidden_size, self.hidden_size)

        attention_module, model_config = self._setup_attention_module(
            qkv_weight, out_weight)

        attn_metadata = self._create_attention_metadata(model_config)

        # Verify QKV projection
        self.assertTrue(
            torch.allclose(attention_module.qkv_proj.forward(hidden_states),
                           hidden_states.to("cuda") @ qkv_weight.to("cuda"),
                           atol=2e-1))

        # Create lora_params in the new format
        lora_params = {
            'num_seqs': self.batch_size,
            'host_request_types': torch.zeros(self.batch_size,
                                              dtype=torch.int32),
            'prompt_lens_cpu': torch.tensor([self.seq_len] * self.batch_size),
            0: {  # layer_idx
                LoraModuleType.ATTENTION_Q: {  # Q module
                    'adapter_size':
                    torch.tensor([8]),  # lora_rank
                    'weight_pointers':
                    torch.tensor([[A_q.data_ptr(),
                                   B_q.data_ptr()]]),
                    'is_dora':
                    False,
                    'weight_tensors': [A_q, B_q]
                },
                LoraModuleType.ATTENTION_K: {  # K module
                    'adapter_size':
                    torch.tensor([8]),  # lora_rank
                    'weight_pointers':
                    torch.tensor([[A_k.data_ptr(),
                                   B_k.data_ptr()]]),
                    'is_dora':
                    False,
                    'weight_tensors': [A_k, B_k]
                },
                LoraModuleType.ATTENTION_V: {  # V module
                    'adapter_size':
                    torch.tensor([8]),  # lora_rank
                    'weight_pointers':
                    torch.tensor([[A_v.data_ptr(),
                                   B_v.data_ptr()]]),
                    'is_dora':
                    False,
                    'weight_tensors': [A_v, B_v]
                },
                LoraModuleType.ATTENTION_DENSE: {  # Output projection module
                    'adapter_size':
                    torch.tensor([8]),  # lora_rank
                    'weight_pointers':
                    torch.tensor([[A_o.data_ptr(),
                                   B_o.data_ptr()]]),
                    'is_dora':
                    False,
                    'weight_tensors': [A_o, B_o]
                }
            }
        }

        # Run vanilla attention with LoRA
        vanilla_output = self._run_vanilla_attention(hidden_states, qkv_weight,
                                                     lora_params)

        # Run pivot attention with LoRA
        pivot_output = self._run_pivot_attention(attention_module,
                                                 hidden_states, attn_metadata,
                                                 lora_params)

        torch.testing.assert_close(pivot_output,
                                   vanilla_output,
                                   atol=2e-2,
                                   rtol=0)
