import contextlib
import os
import weakref
from abc import ABC, abstractmethod
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import WideEPMoE
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla, is_qwen3_next
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import local_mpi_size, mpi_rank, mpi_world_size, torch_dtype_to_binding
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .runner_interface import BalanceMethod


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b


class RunnerMixin(ABC):
    @staticmethod
    @abstractmethod
    def has_mamba_metadata() -> bool:
        pass

    @staticmethod
    @contextlib.contextmanager
    def scaled_from_ctx(scaled_from, mapping, pretrained_config):
        if scaled_from is None:
            yield
            return
        # To run the problem size of $B$ GPUs on $A$ GPUs, we need:
        # (1) Attention: If TP, reduce the number of attention heads; If DP, nothing to change.
        # (2) MoE: If EP, reduce the number of experts; If TP, reduce head size.
        #     Maintain the result of AllToAll method selection because it is affected by EP size.
        if not mapping.enable_attention_dp:
            if hasattr(pretrained_config, "index_n_heads"):
                raise NotImplementedError("Not support Indexer TP for weak scaling")
            pretrained_config.num_attention_heads = (
                pretrained_config.num_attention_heads // scaled_from * mapping.tp_size
            )
            pretrained_config.num_key_value_heads = (
                pretrained_config.num_key_value_heads // scaled_from * mapping.tp_size
            )
        if mapping.moe_ep_size != mapping.world_size:
            raise NotImplementedError("Not support MoE TP for weak scaling")
        pretrained_config.n_routed_experts = (
            pretrained_config.n_routed_experts // scaled_from * mapping.moe_ep_size
        )
        select_alltoall_method_type_orig = WideEPMoE.select_alltoall_method_type

        def select_alltoall_method_type(cls: type, mapping: Mapping, top_k: int, *args, **kwargs):
            # Replace the condition `mapping.moe_ep_size <= top_k` with `scaled_from <= top_k`
            # by replacing `top_k` with `fake_top_k`
            if scaled_from <= top_k:
                fake_top_k = mapping.moe_ep_size + 1
            else:
                fake_top_k = mapping.moe_ep_size - 1
            assert (mapping.moe_ep_size <= fake_top_k) == (scaled_from <= top_k)
            return select_alltoall_method_type_orig(mapping, fake_top_k, *args, **kwargs)

        WideEPMoE.select_alltoall_method_type = select_alltoall_method_type
        try:
            yield
        finally:
            WideEPMoE.select_alltoall_method_type = select_alltoall_method_type_orig

    @staticmethod
    def apply_quant_config_exclude_modules(layers, quant_config):
        # Please refer to tensorrt_llm/_torch/models/modeling_utils.py
        new_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for layer in layers:
            for name, module in layer.named_modules():
                name = f"model.layers.{layer.layer_idx}.{name}"
                candidates = [name]
                if isinstance(module, Linear):
                    weight_mode = module.weights_loading_config.weight_mode
                    if weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
                        # sometimes gate and up proj are not packed in the checkpoint,
                        # but they still share the same exclusion rule
                        candidates += [
                            name.replace("gate_up_proj", "gate_proj"),
                            name.replace("gate_up_proj", "up_proj"),
                        ]
                    elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                        # sometimes q_proj, k_proj and v_proj are not packed in the checkpoint,
                        # but they still share the same exclusion rule
                        candidates += [
                            name.replace("qkv_proj", "q_proj"),
                            name.replace("qkv_proj", "k_proj"),
                            name.replace("qkv_proj", "v_proj"),
                        ]
                is_excluded = any(
                    quant_config.is_module_excluded_from_quantization(n) for n in candidates
                )
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = new_quant_config

    def create_run_pack(
        self,
        run_type: str,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv_cache: int,
        kv_cache_manager: KVCacheManager,
        attn_workspace: Optional[torch.Tensor] = None,
    ):
        if self.model_config.moe_backend == "TRTLLM" and os.getenv("TRTLLM_ENABLE_PDL") != "1":
            raise ValueError("Suggest to set TRTLLM_ENABLE_PDL=1 when moe_backend is TRTLLM")
        world_size = mpi_world_size()
        AttentionCls = get_attention_backend(
            self.model_config.attn_backend, self.model_config.sparse_attention_config
        )
        attn_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int),
            request_ids=list(range(batch_size)),
            max_num_requests=kv_cache_manager.max_batch_size,
            num_contexts={
                "CTX": batch_size,
                "GEN": 0,
            }[run_type],
            prompt_lens=[
                {
                    "CTX": seq_len_q,
                    "GEN": seq_len_kv_cache,
                }[run_type]
            ]
            * batch_size,
            max_num_tokens=batch_size * seq_len_q,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[seq_len_kv_cache] * batch_size,
            ),
            workspace=attn_workspace,
            mapping=self.model_config.mapping,
            sparse_attention_config=self.model_config.sparse_attention_config,
        )
        attn_metadata.all_rank_num_tokens = [batch_size * seq_len_q] * world_size
        attn_metadata.prepare()
        with model_extra_attrs(self.model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
        hidden_size = self.model_config.pretrained_config.hidden_size
        position_ids = torch.tensor(
            [list(range(seq_len_kv_cache, seq_len_kv_cache + seq_len_q)) * batch_size],
            dtype=torch.int32,
            device="cuda",
        )
        hidden_states = torch.rand(
            (batch_size * seq_len_q, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        residual = torch.rand(
            (batch_size * seq_len_q, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        kwargs = {}

        if self.has_mamba_metadata():
            # Please refer to `tensorrt_llm/_torch/models/modeling_qwen3_next.py` for `mamba_metadata`
            mamba_metadata = Mamba2Metadata(attn_metadata.max_num_requests, chunk_size=128)
            mamba_metadata.prepare(attn_metadata)
            kwargs["mamba_metadata"] = mamba_metadata

        def run_pack():
            output = hidden_states, residual
            with model_extra_attrs(self.model_config.extra_attrs):
                with torch.inference_mode():
                    for layer in self.layers:
                        output = layer(position_ids, output[0], attn_metadata, output[1], **kwargs)
            return output

        return run_pack

    def replace_routing_method(self, balance_method: BalanceMethod, balance_ratio: float):
        if balance_method != BalanceMethod.NotModified:
            raise NotImplementedError("not support replacing routing method for this runner")

    @staticmethod
    def create_kv_cache_manager(
        pretrained_model_name_or_path,
        mapping,
        tokens_per_block,
        max_batch_size,
        max_seq_len,
        layer_indices,
    ):
        # Please refer to `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` for `tokens_per_block`
        model_config = ModelConfig.from_pretrained(pretrained_model_name_or_path)
        if model_config.enable_flash_mla:
            assert tokens_per_block == 64

        # Please refer to `tensorrt_llm/_torch/pyexecutor/_util.py` for `kv_cache_manager`
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config)
        config = model_config.pretrained_config
        kv_cache_config = KvCacheConfig(
            max_tokens=max_batch_size * round_up(max_seq_len, tokens_per_block),
            enable_block_reuse=False,
        )
        kv_cache_dtype = torch_dtype_to_binding(
            {
                None: torch.bfloat16,
                "FP8": torch.float8_e4m3fn,
            }[model_config.quant_config.kv_cache_quant_algo]
        )
        if is_mla(config):
            layer_mask = [i in layer_indices for i in range(config.num_hidden_layers)]
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                kv_cache_config,
                CacheType.SELFKONLY,
                num_layers=sum(layer_mask),
                num_kv_heads=1,
                head_dim=model_config.pretrained_config.kv_lora_rank
                + model_config.pretrained_config.qk_rope_head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                layer_mask=layer_mask,
                sparse_attn_config=model_config.sparse_attention_config,
            )
        elif is_qwen3_next(config):
            mamba_layer_mask = [
                i in layer_indices
                if i % config.full_attention_interval != config.full_attention_interval - 1
                else False
                for i in range(config.num_hidden_layers)
            ]
            layer_mask = [
                False
                if i % config.full_attention_interval != config.full_attention_interval - 1
                else i in layer_indices
                for i in range(config.num_hidden_layers)
            ]
            num_mamba_layers = sum(mamba_layer_mask)
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                # mamba cache parameters
                config.linear_key_head_dim,
                config.linear_conv_kernel_dim,
                config.linear_num_value_heads,
                config.linear_num_key_heads,
                config.linear_value_head_dim,
                num_mamba_layers,
                mamba_layer_mask,
                config.torch_dtype,
                model_config.quant_config.mamba_ssm_cache_dtype,
                # kv cache parameters
                kv_cache_config,
                CacheType.SELF,
                num_layers=num_layers,
                layer_mask=layer_mask,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=None,
            )
        else:
            raise NotImplementedError("Unsupported config")
        kv_cache_manager.add_dummy_requests(
            list(range(max_batch_size)), [max_seq_len] * max_batch_size
        )
        return kv_cache_manager

    @staticmethod
    def create_mapping(enable_attention_dp: bool):
        world_size = mpi_world_size()
        rank = mpi_rank()
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=local_mpi_size(),
            cp_size=1,
            tp_size=world_size,
            pp_size=1,
            moe_cluster_size=1,
            moe_tp_size=1,
            moe_ep_size=world_size,
            attn_tp_size=world_size,
            attn_cp_size=1,
            enable_attention_dp=enable_attention_dp,
        )
        return mapping
