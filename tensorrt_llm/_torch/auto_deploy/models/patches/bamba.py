# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A patch for the Bamba model to make it compatible with torch.export."""

from typing import Optional

import torch
import torch.utils._pytree as _pytree
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.bamba.modeling_bamba import (
    ALL_ATTENTION_FUNCTIONS,
    BambaAttention,
    BambaMixer,
    BambaModel,
    BambaPreTrainedModel,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ...custom_ops.attention_interface import BatchInfo
from ...export.interface import BaseExportPatch, ExportPatchRegistry

# transformers>=5.5 removed ``HybridMambaAttentionDynamicCache`` (replaced by the
# generic ``Cache`` base class with per-layer ``layers[i].conv_states`` /
# ``layers[i].recurrent_states`` fields). Importing it lazily keeps the patch
# resilient if a future release drops the old name entirely but reintroduces a
# bool quirk on the unified cache class.
try:
    from transformers.models.bamba.modeling_bamba import (
        HybridMambaAttentionDynamicCache as _HybridMambaAttentionDynamicCache,
    )
except ImportError:
    _HybridMambaAttentionDynamicCache = None

# transformers>=5.5 returns the unified ``DynamicCache`` (with per-layer
# ``CacheLayerMixin`` subclasses) as ``past_key_values`` in ``ModelOutput``
# subclasses. ``torch.export``'s aot_autograd ``flat_fn`` flattens the output via
# pytree, and chokes when it encounters ``DynamicCache`` because the class is
# not registered as a known pytree node. Registering it at module import keeps
# the export-time flatten/unflatten contract intact (so the patched-vs-original
# comparison in the export tests still compares like-for-like structures).
try:
    from transformers.cache_utils import DynamicCache as _DynamicCache
except ImportError:
    _DynamicCache = None

_LAYER_TENSOR_ATTRS = ("keys", "values", "conv_states", "recurrent_states")


def _flatten_dynamic_cache(cache):
    leaves = []
    spec = []
    for layer in cache.layers:
        present = []
        for attr in _LAYER_TENSOR_ATTRS:
            val = getattr(layer, attr, None)
            if torch.is_tensor(val):
                leaves.append(val)
                present.append(attr)
        spec.append((type(layer), tuple(present)))
    return leaves, tuple(spec)


def _unflatten_dynamic_cache(leaves, context):
    cache = _DynamicCache.__new__(_DynamicCache)
    cache.layers = []
    idx = 0
    for layer_cls, present in context:
        layer = layer_cls.__new__(layer_cls)
        for attr in _LAYER_TENSOR_ATTRS:
            setattr(layer, attr, None)
        for attr in present:
            setattr(layer, attr, leaves[idx])
            idx += 1
        cache.layers.append(layer)
    return cache


if _DynamicCache is not None and _DynamicCache not in _pytree.SUPPORTED_NODES:
    _pytree.register_pytree_node(
        _DynamicCache,
        _flatten_dynamic_cache,
        _unflatten_dynamic_cache,
    )


# Original implementation:
# https://github.com/huggingface/transformers/blob/06f8004e5cd9d06cfbffc3f47afb6c2b43bcb3d2/
# src/transformers/models/bamba/modeling_bamba.py#L719
# NOTE: we remove the cache-related code paths.
def _bamba_mixer_torch_forward(
    self,
    input_states,
    cache_params: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    # `input_states` is of shape `[B, T, hidden_dim]`, and during generate, each successive call
    # will have T = T + 1.
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # 1. Gated MLP's linear projection
    input_states = apply_mask_to_padding_states(input_states, attention_mask)
    projected_states = self.in_proj(input_states)
    gate, hidden_states_B_C, dt = projected_states.split(
        [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
    )

    use_caching = cache_params is not None

    if use_caching:
        # transformers>=5.5 unified ``DynamicCache`` lazily allocates per-layer
        # ``conv_states`` / ``recurrent_states`` (5.3.x ``HybridMambaAttentionDynamicCache``
        # pre-allocated them). The cached custom ops below need real buffers, so trigger
        # lazy_initialization with correctly-shaped zero tensors before first use.
        _layer = cache_params.layers[self.layer_idx]
        if getattr(_layer, "conv_states", None) is None and hasattr(_layer, "lazy_initialization"):
            _conv_dim = self.conv1d.weight.shape[0]
            _conv_kernel_size = self.conv1d.weight.shape[-1]
            _layer.lazy_initialization(
                conv_states=torch.zeros(
                    batch_size,
                    _conv_dim,
                    _conv_kernel_size,
                    device=input_states.device,
                    dtype=hidden_states_B_C.dtype,
                )
            )
        if getattr(_layer, "recurrent_states", None) is None and hasattr(
            _layer, "lazy_initialization"
        ):
            _layer.lazy_initialization(
                recurrent_states=torch.zeros(
                    batch_size,
                    self.num_heads,
                    self.head_dim,
                    self.ssm_state_size,
                    device=input_states.device,
                    dtype=hidden_states_B_C.dtype,
                )
            )

    # 2. Convolution sequence transformation (cached/uncached handled inside the op)
    if use_caching:
        # Prepare dense metadata for cached flattened op
        seq_len_t = torch.full((batch_size,), seq_len, device=input_states.device, dtype=torch.int)
        cu_seqlen_t = torch.arange(
            0, batch_size * seq_len, seq_len, device=input_states.device, dtype=torch.int
        )
        slot_idx_t = torch.arange(batch_size, device=input_states.device, dtype=torch.long)
        use_initial_states_t = torch.zeros(batch_size, device=input_states.device, dtype=torch.bool)
        # ``BatchInfo()`` default-constructs the host tensor with
        # ``pin_memory=prefer_pinned()`` (=True) and no explicit device. Under
        # torch.export the tracer captures the implicit default device as cuda,
        # producing ``aten.empty.memory_format([13], device=cuda, pin_memory=True)``
        # which is illegal at gm execution. Pre-allocate the host buffer on CPU
        # without pinning so the captured op is well-formed.
        _batch_info = BatchInfo(
            batch_info_host=torch.zeros(BatchInfo._NUM_ELEMENTS, dtype=torch.int, device="cpu")
        )
        if seq_len == 1:
            _batch_info.update([0, 0, 0, 0, batch_size, batch_size])
        else:
            _batch_info.update([batch_size, batch_size * seq_len, 0, 0, 0, 0])
        batch_info_host_t = _batch_info.serialize()
    if use_caching:
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_cached_causal_conv1d(
                # INPUTS
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                # STANDARD METADATA
                batch_info_host_t,
                seq_len_t,
                cu_seqlen_t,
                slot_idx_t,
                use_initial_states_t,
                # CACHES
                cache_params.layers[self.layer_idx].conv_states,
                # CONSTANTS
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
        )
    else:
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
        )

    hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
    hidden_states, B, C = torch.split(
        hidden_states_B_C,
        [
            self.intermediate_size,
            self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size,
        ],
        dim=-1,
    )

    # 3. SSM transformation
    A = -torch.exp(self.A_log.float())  # [num_heads]

    if use_caching:
        # Use new flattened cached op for both cache updates and outputs
        y = torch.ops.auto_deploy.torch_cached_ssm(
            # INPUTS
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            # STANDARD METADATA
            batch_info_host=batch_info_host_t,
            seq_len=seq_len_t,
            cu_seqlen=cu_seqlen_t,
            slot_idx=slot_idx_t,
            use_initial_states=use_initial_states_t,
            # CACHES
            ssm_state_cache=cache_params.layers[self.layer_idx].recurrent_states,
            # CONSTANTS
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )
    else:
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )

    y = y.view(batch_size, seq_len, -1)

    scan_output = self.norm(y, gate)

    # end ssd naive
    # !! This is the end of the uncached code path.

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
    return contextualized_states


# Original 5.3.x signature was ``(self, attention_mask, cache_position)``; transformers>=5.5
# renamed the second argument to ``past_key_values`` (mask handling unified under the new cache).
# Accept the new name and ignore both — we just want to disable the mask for export.
def _bamba_model_update_mamba_mask(self, attention_mask, past_key_values):
    return None


# transformers>=5.5 rewrote `DynamicLayer.update` to lazily allocate the cache by calling
# `torch.tensor([], device=...)`, which torch.export's fake-tensor tracer rejects on the
# `meta` device. The original 5.3.x flow exposed a typed `HybridMambaAttentionDynamicCache`
# whose attention layers carried pre-allocated key/value tensors so this didn't trigger.
# During export we don't actually need to maintain the KV cache — the AutoDeploy runtime
# substitutes its own cached attention kernel later — so just skip the cache.update call.
def _bamba_attention_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _bamba_model_update_causal_mask(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
    output_attentions,
):
    # Force attention to use causal mode without explicit masks
    return None


# NOTE: this would need to be applied earlier than other patches, since the `_init_weights` (which
# is called by `post_init`) is called before we run `forward`.
def _bamba_pretrained_model_init_weights(self, module):
    # ARGH python does not like this.
    super(BambaPreTrainedModel, self)._init_weights(module)
    if isinstance(module, BambaMixer):
        module.dt_bias.data.fill_(1.0)
        # ? Why is this even necessary? `BambaMixer.__init__` already sets
        A = torch.arange(1, module.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        # module.A_log.data = torch.log(torch.arange(1, module.num_heads + 1))
        module.D.data.fill_(1.0)


def _cache_bool(self) -> bool:
    # This is a workaround for the bug on this line:
    # https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/bamba/modeling_bamba.py#L1211
    # The base `Cache` class from which `HybridMambaAttentionDynamicCache` inherits has a `__len__`
    # special method implement that returns `False` when the instance is initialized, but stores no
    # cache.
    # The original line should really be checking for `if past_key_values is not None` but adding
    # a patch for `BambaModel.forward` just for that reason seems overly intrusive.
    return True


@ExportPatchRegistry.register("bamba")
class BambaModelPatch(BaseExportPatch):
    """Patch for `BambaMixer`."""

    def _apply_patch(self):
        self.original_values["BambaMixer.torch_forward"] = BambaMixer.torch_forward
        self.original_values["BambaModel._update_mamba_mask"] = BambaModel._update_mamba_mask
        self.original_values["BambaAttention.forward"] = BambaAttention.forward
        # Older transformers expose both; newer releases dropped `_update_causal_mask` on `BambaModel`
        # (mask handling consolidated under `_update_mamba_mask`).
        if hasattr(BambaModel, "_update_causal_mask"):
            self.original_values["BambaModel._update_causal_mask"] = BambaModel._update_causal_mask
        # NOTE: there is `HybridMambaAttentionDynamicCache.__bool__` to save.
        # self.original_values["BambaPreTrainedModel._init_weights"] = BambaPreTrainedModel._init_weights

        BambaMixer.torch_forward = _bamba_mixer_torch_forward
        BambaModel._update_mamba_mask = _bamba_model_update_mamba_mask
        BambaAttention.forward = _bamba_attention_forward
        if hasattr(BambaModel, "_update_causal_mask"):
            BambaModel._update_causal_mask = _bamba_model_update_causal_mask
        if _HybridMambaAttentionDynamicCache is not None:
            self.original_values["HybridMambaAttentionDynamicCache.__bool__"] = (
                _HybridMambaAttentionDynamicCache.__bool__
            )
            _HybridMambaAttentionDynamicCache.__bool__ = _cache_bool

    def _revert_patch(self):
        BambaMixer.torch_forward = self.original_values["BambaMixer.torch_forward"]
        BambaModel._update_mamba_mask = self.original_values["BambaModel._update_mamba_mask"]
        BambaAttention.forward = self.original_values["BambaAttention.forward"]
        if "BambaModel._update_causal_mask" in self.original_values:
            BambaModel._update_causal_mask = self.original_values["BambaModel._update_causal_mask"]
        if "HybridMambaAttentionDynamicCache.__bool__" in self.original_values:
            _HybridMambaAttentionDynamicCache.__bool__ = self.original_values[
                "HybridMambaAttentionDynamicCache.__bool__"
            ]


# NOTE: patch that is used during build model time
BambaPreTrainedModel._init_weights = _bamba_pretrained_model_init_weights
