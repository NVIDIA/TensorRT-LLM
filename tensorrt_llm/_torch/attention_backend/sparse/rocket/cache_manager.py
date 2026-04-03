import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import torch
from triton import next_power_of_2

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import BlockManager, KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig


class RocketKVCacheManager(KVCacheManager):
    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ) -> None:
        assert not kv_cache_config.enable_block_reuse, (
            "RocketKV cache requires block reuse to be disabled in KV cache config"
        )
        self.kt_tokens_per_block = next_power_of_2(
            math.ceil(tokens_per_block / sparse_attn_config.page_size)
        )
        self.kt_cache_dtype = (
            torch.bfloat16 if sparse_attn_config.kt_cache_dtype == "bfloat16" else torch.float8_e5m2
        )

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )
        self.page_size = sparse_attn_config.page_size
        self.prompt_budget = sparse_attn_config.prompt_budget
        self.max_batch_size = max_batch_size

        # Per layer KT cache pool
        # Use the same number of blocks as the paged kv cache. In this way, the scheduler can use the same number of
        # blocks to schedule requests.
        # Use kt_tokens_per_block to make sure the KT cache is large enough to hold the kt tokens,
        # since kt_tokens_per_block * num_blocks * page_size >= tokens_per_block * num_blocks.
        self.num_blocks = self.blocks_in_primary_pool
        self.kt_cache_pool_per_layer = [
            torch.empty(
                (self.num_blocks, self.kt_tokens_per_block, num_kv_heads, head_dim * 2),
                device="cuda",
                dtype=self.kt_cache_dtype,
            )
            for _ in range(self.num_local_layers)
        ]
        self.max_kt_blocks_per_seq = self.num_blocks

        # Block manager to manage the KT cache blocks for each request. Different layers share the
        # same block ids.
        self.kt_cache_manager = BlockManager(self.num_blocks, self.kt_tokens_per_block)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager=None,
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            num_extra_decoding_steps=num_extra_decoding_steps,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )
        if prepare_resource:
            for req in requests:
                request_id = req.py_request_id
                kt_token_num = math.ceil(req.max_beam_num_tokens / self.page_size)
                self.kt_cache_manager.add_tokens(request_id, kt_token_num)
        return requests

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache_pool_per_layer[layer_idx]

    def copy_kt_block_offsets(
        self, request_ids: List[int], block_offsets: torch.Tensor
    ) -> torch.Tensor:
        self.kt_cache_manager.copy_block_offsets(request_ids, block_offsets)

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)
        for req in scheduled_batch.context_requests:
            request_id = req.py_request_id
            num_tokens = req.prompt_len
            kt_token_num = math.ceil(num_tokens / self.page_size)
            self.kt_cache_manager.add_tokens(request_id, kt_token_num)

        for req in scheduled_batch.generation_requests:
            request_id = req.py_request_id
            num_tokens = req.max_beam_num_tokens + 1
            if num_tokens % self.page_size == 1:
                self.kt_cache_manager.add_tokens(request_id, 1)

    def update_resources(
        self,
        scheduled_batch,
        attn_metadata: AttentionMetadata = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)
                # get the rewind length for kt cache
                num_tokens = request.max_beam_num_tokens
                updated_kt_token_num = num_tokens - rewind_len
                rewind_len = math.ceil(num_tokens / self.page_size) - math.ceil(
                    updated_kt_token_num / self.page_size
                )
                self.kt_cache_manager.rewind_cache(request, rewind_len)

    def free_resources(self, request):
        super().free_resources(request)
        self.kt_cache_manager.free_resources(request)

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfig, mapping: Mapping, num_layers: Optional[int] = None, **kwargs
    ):
        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
            mem_per_token = 1

        # get num key value heads
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(num_key_value_heads)

        # get head dim
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = config.hidden_size // config.num_attention_heads
        head_dim = head_dim * num_key_value_heads // tp_size

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        mem_per_token *= num_attention_layers * head_dim

        # K and V
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        tokens_per_block = kwargs["tokens_per_block"]
        sparse_attn_config = model_config.sparse_attention_config
        kt_tokens_per_block = next_power_of_2(
            math.ceil(tokens_per_block / sparse_attn_config.page_size)
        )
        kt_factor = 2
        if sparse_attn_config.kt_cache_dtype == "float8_e5m2":
            kt_factor = 1
        kv_factor = 2 + kt_factor * kt_tokens_per_block / tokens_per_block
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        kt_factor = 2
        if self.kt_cache_dtype == torch.float8_e5m2:
            kt_factor = 1
        kv_factor = self.kv_factor + kt_factor * self.kt_tokens_per_block / self.tokens_per_block
        cache_size_per_token = math.ceil(
            kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim
        )

        if self.dtype not in (
            DataType.FP8,
            DataType.HALF,
            DataType.BF16,
            DataType.FLOAT,
            DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token, quant_vector_size=16, scaling_factor_dtype=DataType.FP8
            )
        return cache_size_bytes_per_token
