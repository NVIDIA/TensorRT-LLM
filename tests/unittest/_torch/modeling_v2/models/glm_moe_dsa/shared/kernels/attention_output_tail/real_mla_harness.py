# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real GLM-5.2 DSA attention layers for attention_output_tail end-to-end tests and benchmarks.

``AttentionStack`` builds, for one TP8 rank, the attention module the model
builds for ``glm_moe_dsa`` checkpoints (``DeepseekV32Attention``) for a run of
consecutive layers. It loads every attention weight of the rank's shard from
the checkpoint the way ``DeepseekV3WeightLoader`` does, fills an FP8
``DSACacheManager`` with a real prefill forward, and prepares GEN-only
attention metadata for one request with ``rows`` query tokens (MTP target
verification with ``rows - 1`` draft tokens) the way the decoder runner does:
CUDA-graph metadata, ``update_spec_dec_param``, then ``prepare``.

Each module is adopted as the target's ``GlmMLA`` right after construction, as the target's
model adopts it, and the GEN forward runs as a decode step (``scope()``), so ``GlmMLA.forward``
then runs exactly as in the model: ``register_to_config`` is set, so the GEN attention runs
inside the target's DSA attention op and ``o_proj`` runs in ``GlmMLA.forward``. With a plan
running ``attention_output_tail`` installed at construction (``fast_paths``, or ``select``),
every target layer owns an attention_output_tail ``AttnTail``; ``set_attn_tail`` swaps it in and out to compare
against the native tail on the same module, cache and inputs. The prefill runs as a prefill
step (the built-in attention).

Fixture labels: the weights (all attention projections, norms and indexer
weights of the rank's shard) are real. The activations are synthetic: the
prefill and GEN inputs are the layer's real ``input_layernorm`` applied to
real token embeddings, not the residual stream of a full forward.

The cache manager and the attention metadata use a one-rank mapping: an MLA
cache holds one latent head, so it is not TP-sharded, and a one-rank mapping
keeps their constructors free of collectives in a process that stands in for
one of eight TP ranks.
"""

import contextlib
import os
import sys
import weakref
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402
import tail_fixtures  # noqa: E402

TP8 = 8
REQUEST_ID = 7
TOKEN_OFFSET = 1000


def load_pretrained(ckpt: str):
    """Load the checkpoint's config as the model does for ``glm_moe_dsa``."""
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    cfg = load_pretrained_config(ckpt, trust_remote_code=True)
    # DeepseekV3ForCausalLM renames glm_moe_dsa to deepseek_v32, which makes
    # the decoder layers build DeepseekV32Attention.
    cfg.model_type = "deepseek_v32"
    dtype = getattr(cfg, "torch_dtype", None)
    if not isinstance(dtype, torch.dtype):
        resolved = getattr(torch, str(dtype or "bfloat16").replace("torch.", ""))
        try:
            cfg.torch_dtype = resolved
        except AttributeError:
            cfg.dtype = resolved
    return cfg


def make_model_config(pretrained, tp_rank: int, max_draft_len: int):
    """``ModelConfig`` of the served GEN role for TP8 rank ``tp_rank``.

    Attention weights of this checkpoint are excluded from NVFP4, so the
    module is built unquantized with an FP8 KV cache, as the model's
    exclusion handling leaves it.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant = QuantConfig()
    quant.kv_cache_quant_algo = QuantAlgo.FP8
    sparse = DeepSeekSparseAttentionConfig(
        enable_heuristic_topk=True, use_cute_dsl_paged_mqa_logits=True, use_cute_dsl_topk=True
    )
    spec = MTPDecodingConfig(max_draft_len=max_draft_len) if max_draft_len > 0 else None
    model_config = ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(world_size=TP8, tp_size=TP8, rank=tp_rank),
        quant_config=quant,
        sparse_attention_config=sparse,
        spec_config=spec,
    )
    model_config.extra_attrs["kv_cache_dtype"] = "fp8"
    return model_config


def build_attention(model_config, layer_idx: int, aux_stream, reduce_output: bool):
    """Build ``DeepseekV32Attention`` for one layer on the current device, adopted as ``GlmMLA``.

    The module has TP8 shapes, but this process may run without the other
    ranks, so the all-reduce modules its ``Linear``s would create are
    replaced by ``None`` during construction; callers that reduce attach a
    real ``AllReduce`` afterwards.
    """
    import tensorrt_llm._torch.distributed as distributed
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        GlmMLA,
    )
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
    from tensorrt_llm._torch.utils import AuxStreamType

    saved = distributed.AllReduce
    distributed.AllReduce = lambda *args, **kwargs: None
    try:
        with torch.device("cuda"):
            attn = DeepseekV32Attention(
                model_config,
                layer_idx=layer_idx,
                aux_stream_dict={AuxStreamType.Attention: aux_stream},
                reduce_output=reduce_output,
            )
            GlmMLA.adopt(attn)
            # The model builds the FMHA libraries in ``create_weights`` after adoption; a module
            # built alone rebuilds them for the library adoption declared.
            for backend in (attn.mha, attn.mqa):
                if backend is not None and backend.prepended_fmha_lib_classes:
                    backend.update_quant_config(backend.quant_config)
    finally:
        distributed.AllReduce = saved
    return attn.cuda()


def load_attention_weights(attn, ckpt: str, layer: int, tp_rank: int, tp_size: int = TP8) -> None:
    """Load the attention weights of ``layer`` like ``DeepseekV3WeightLoader``."""
    from tensorrt_llm._torch.models.modeling_utils import filter_weights

    prefix = f"model.layers.{layer}.self_attn"
    weights = tail_fixtures.read_prefix(ckpt, prefix + ".")
    heads = attn.num_heads
    nope, v_dim, kv_lora = (attn.qk_nope_head_dim, attn.v_head_dim, attn.kv_lora_rank)
    with torch.no_grad():
        for name, module in attn.named_modules():
            if not name or len(module._parameters) == 0:
                continue
            full = f"{prefix}.{name}"
            leaf = name.split(".")[-1]
            if leaf == "kv_b_proj":
                kv_b = weights[f"{full}.weight"].unflatten(0, [heads, nope + v_dim])
                kv_b = torch.chunk(kv_b, tp_size, dim=0)[tp_rank]
                k_nope, v_w = kv_b.split([nope, v_dim], dim=1)
                local = kv_b.shape[0]
                module.weight.data.copy_(
                    torch.cat(
                        [k_nope.reshape(local * nope, kv_lora), v_w.reshape(local * v_dim, kv_lora)]
                    )
                )
                attn.v_b_proj = torch.nn.Parameter(
                    module.weight.data[local * nope :].view(local, v_dim, kv_lora),
                    requires_grad=False,
                )
                attn.k_b_proj_trans.data.copy_(
                    k_nope.transpose(2, 1).contiguous().reshape(attn.k_b_proj_trans.shape)
                )
            elif leaf == "kv_a_proj_with_mqa":
                fused = torch.cat([weights[f"{prefix}.q_a_proj.weight"], weights[f"{full}.weight"]])
                module.weight.data[: fused.shape[0]].copy_(fused)
            else:
                module_weights = filter_weights(full, weights)
                if hasattr(module, "load_weights"):
                    module.load_weights(weights=[module_weights])
                else:
                    for n, p in module.named_parameters():
                        p.data.copy_(module_weights[n][:])
        for m in attn.modules():
            post = getattr(m, "post_load_weights", None)
            if post is not None and not getattr(m, "_weights_removed", False):
                post()


def layer_inputs(ckpt: str, layer: int, first_token: int, num_tokens: int) -> torch.Tensor:
    """Real ``input_layernorm`` of ``layer`` over real token embeddings."""
    return tail_fixtures.normed_embeddings(ckpt, layer, first_token, num_tokens)


def select(fast_paths: str) -> None:
    """Install, for this process, the plan that runs exactly ``fast_paths`` (comma-separated)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        dispatch,
    )

    plan_testlib.select(dispatch, fast_paths)


class AttentionStack:
    """Consecutive real DSA attention layers of one TP8 rank with a filled cache.

    Args:
        ckpt: checkpoint directory.
        layers: consecutive layer indices; the first must run a full indexer.
        tp_rank: TP8 rank whose weight shard is loaded.
        ctx: cached context length per layer (filled by a real prefill).
        rows: GEN query rows of the single request (``1 + max_draft_len``).
        reduce_output: build ``o_proj`` with ``reduce_output`` (dense layers).
        fast_paths: when given, the plan that runs exactly these fast paths is installed
            (``select``) before the layers are built; otherwise the installed plan is used.
    """

    def __init__(
        self,
        ckpt: str,
        layers: List[int],
        tp_rank: int,
        ctx: int = 4096,
        rows: int = 6,
        reduce_output: bool = False,
        fast_paths: Optional[str] = None,
    ):
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend

        if fast_paths is not None:
            select(fast_paths)

        self.ckpt = ckpt
        self.layers = list(layers)
        self.tp_rank = tp_rank
        self.ctx = ctx
        self.rows = rows
        self.max_draft_len = rows - 1
        self.pretrained = load_pretrained(ckpt)
        self.model_config = make_model_config(self.pretrained, tp_rank, self.max_draft_len)
        sparse = self.model_config.sparse_attention_config
        self.sparse_params = sparse.to_sparse_params(pretrained_config=self.pretrained)
        self.sparse_metadata_params = sparse.to_sparse_metadata_params(
            pretrained_config=self.pretrained
        )
        self.backend = get_attention_backend("TRTLLM", sparse_params=self.sparse_params)
        self.aux_stream = torch.cuda.Stream()
        self.attn: Dict[int, torch.nn.Module] = {}
        for layer in self.layers:
            attn = build_attention(self.model_config, layer, self.aux_stream, reduce_output)
            load_attention_weights(attn, ckpt, layer, tp_rank)
            self.attn[layer] = attn
        self.attn_tail_tails = {
            layer: getattr(attn, "_attn_tail", None) for layer, attn in self.attn.items()
        }
        self.kv_cache_manager = self._make_cache()
        self.kv_cache_manager.add_dummy_requests(
            request_ids=[REQUEST_ID], token_nums=[ctx + 64], is_gen=False, prepare_resource=True
        )
        self._prefill()
        self.gen_metadata = self._gen_metadata()

    # -- construction --------------------------------------------------------
    def _one_rank_mapping(self):
        from tensorrt_llm.mapping import Mapping

        return Mapping(world_size=1, tp_size=1, rank=0)

    def _make_cache(self):
        import tensorrt_llm.bindings
        from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
        from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        num_layers = max(self.layers) + 1
        layer_mask = [i in self.layers for i in range(num_layers)]
        max_tokens = self.ctx + 256
        return DSACacheManager(
            KvCacheConfig(
                dtype="fp8",
                max_tokens=max_tokens,
                enable_block_reuse=False,
                event_buffer_max_size=0,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=len(self.layers),
            layer_mask=layer_mask,
            num_kv_heads=1,
            head_dim=self.pretrained.kv_lora_rank + self.pretrained.qk_rope_head_dim,
            tokens_per_block=64,
            max_seq_len=max_tokens,
            max_batch_size=1,
            mapping=self._one_rank_mapping(),
            dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
            max_num_tokens=max(8192, self.ctx),
            sparse_attention_config=self.model_config.sparse_attention_config,
            model_config=self.model_config,
            pretrained_config=self.pretrained,
            vocab_size=self.pretrained.vocab_size,
        )

    def _prefill(self) -> None:
        from tensorrt_llm._torch.metadata import KVCacheParams

        meta = self.backend.Metadata(
            seq_lens=torch.tensor([self.ctx], dtype=torch.int),
            request_ids=[REQUEST_ID],
            max_num_requests=1,
            num_contexts=1,
            prompt_lens=[self.ctx],
            max_num_tokens=self.ctx,
            kv_cache_manager=self.kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            mapping=self._one_rank_mapping(),
            sparse_metadata_params=self.sparse_metadata_params,
        )
        meta.prepare()
        position_ids = torch.arange(self.ctx, dtype=torch.int32, device="cuda")[None]
        from tensorrt_llm._torch.distributed import AllReduceParams

        no_ar = AllReduceParams(enable_allreduce=False)
        with self.scope(meta):
            for layer in self.layers:
                hidden = layer_inputs(self.ckpt, layer, TOKEN_OFFSET, self.ctx)
                self.set_attn_tail(False)
                self.attn[layer](
                    position_ids=position_ids,
                    hidden_states=hidden,
                    attn_metadata=meta,
                    all_reduce_params=no_ar,
                )
        torch.cuda.synchronize()

    def _gen_metadata(self):
        from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
        from tensorrt_llm._torch.metadata import KVCacheParams

        base = self.backend.Metadata(
            max_num_requests=1,
            max_num_tokens=128,
            max_num_sequences=1,
            kv_cache_manager=self.kv_cache_manager,
            mapping=self._one_rank_mapping(),
            runtime_features=AttentionRuntimeFeatures(
                has_speculative_draft_tokens=self.max_draft_len > 0
            ),
            enable_flash_mla=False,
            enable_context_mla_with_cached_kv=False,
            num_heads_per_kv=1,
            sparse_metadata_params=self.sparse_metadata_params,
        )
        meta = base.create_cuda_graph_metadata(1, False, self.max_draft_len)
        meta.beam_width = 1
        meta.request_ids = [REQUEST_ID]
        meta.prompt_lens = [self.ctx]
        meta.num_contexts = 0
        meta.num_chunked_ctx_requests = 0
        meta.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[self.ctx],
            num_extra_kv_tokens=max(self.max_draft_len - 1, 0),
        )
        meta.kv_cache_manager = self.kv_cache_manager
        if self.max_draft_len > 0:
            meta.update_spec_dec_param(
                batch_size=1,
                is_spec_decoding_enabled=True,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=self.max_draft_len,
                max_total_draft_tokens=self.max_draft_len,
                num_contexts=0,
            )
        meta.prepare()
        return meta

    # -- running -------------------------------------------------------------
    @contextlib.contextmanager
    def scope(self, meta=None):
        """Model extra attrs, multi-stream mode and the step's phase, as the engine forward
        sets them: the GEN metadata (``meta`` None) is a decode step, ``meta`` a prefill."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
            dispatch,
        )
        from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
        from tensorrt_llm._torch.utils import model_extra_attrs

        phase = dispatch.Phase.DECODE if meta is None else dispatch.Phase.PREFILL
        attrs = dict(self.model_config.extra_attrs)
        attrs["attention_metadata"] = weakref.ref(meta or self.gen_metadata)
        with model_extra_attrs(attrs), with_multi_stream(True), dispatch.step(phase):
            yield

    def set_attn_tail(self, enabled: bool, config=None) -> None:
        """Attach (or detach) each layer's attention_output_tail; optionally set its config."""
        for layer, attn in self.attn.items():
            tail = self.attn_tail_tails[layer]
            if enabled and tail is None:
                raise RuntimeError(
                    f"layer {layer} has no attention_output_tail tail "
                    "(the installed plan does not run attention_output_tail)"
                )
            if tail is not None and config is not None:
                tail.config = config
            attn._attn_tail = tail if enabled else None

    def gen_inputs(self, layer: int, step: int = 0) -> torch.Tensor:
        """GEN hidden states ``[rows, hidden]`` for ``layer`` at ``step``."""
        first = TOKEN_OFFSET + self.ctx + step * self.rows
        return layer_inputs(self.ckpt, layer, first, self.rows)

    def position_ids(self) -> torch.Tensor:
        return torch.arange(self.ctx, self.ctx + self.rows, dtype=torch.int32, device="cuda")[None]

    def forward(
        self, layer: int, hidden: torch.Tensor, position_ids, all_reduce_params
    ) -> torch.Tensor:
        """One GEN ``GlmMLA.forward`` of ``layer``; call inside ``scope()``."""
        return self.attn[layer](
            position_ids=position_ids,
            hidden_states=hidden,
            attn_metadata=self.gen_metadata,
            all_reduce_params=all_reduce_params,
        )

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


class LatentRecorder:
    """Record the latent each layer's GEN tail receives (eager runs only).

    Wraps the native W_UV entry (``MLA._bmm_bf16_out``) and the attention_output_tail entry
    (``AttnTail.defer_w_uv``) on the given modules, so the test can build the
    FP64 oracle from the latent the real attention produced.
    """

    def __init__(self, stack: AttentionStack):
        self.stack = stack
        self.latents: Dict[int, torch.Tensor] = {}
        self._saved: List = []

    def __enter__(self):
        for layer, attn in self.stack.attn.items():
            native = attn._bmm_bf16_out

            def bmm(a, b, b_t, out, *rest, _layer=layer, _native=native):
                # a: latent transposed to [H, R, C].
                self.latents[_layer] = a.transpose(0, 1).contiguous().clone()
                return _native(a, b, b_t, out, *rest)

            attn._bmm_bf16_out = bmm
            self._saved.append((attn, "_bmm_bf16_out", None))
            tail = self.stack.attn_tail_tails[layer]
            if tail is not None:
                defer = tail.defer_w_uv

                def wrapped(mla, meta, latent, output, _layer=layer, _defer=defer):
                    self.latents[_layer] = latent.contiguous().clone()
                    return _defer(mla, meta, latent, output)

                tail.defer_w_uv = wrapped
                self._saved.append((tail, "defer_w_uv", None))
        return self

    def __exit__(self, *exc):
        for obj, name, _ in self._saved:
            delattr(obj, name)
        self._saved.clear()
        return False


def native_reference(attn, latent: torch.Tensor) -> Optional[torch.Tensor]:
    """FP64 oracle of the tail on a recorded latent (native BF16 ``v``)."""
    import tail_reference

    ref, _ = tail_reference.oracle_attn_tail(
        latent, attn.v_b_proj, attn.o_proj.weight, latent.shape[0]
    )
    return ref
