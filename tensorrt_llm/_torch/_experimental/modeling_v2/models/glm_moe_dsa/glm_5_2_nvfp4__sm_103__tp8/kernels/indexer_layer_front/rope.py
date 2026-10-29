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
"""indexer_layer_front generation rope: rope_kv_append's fused rope with the trigger after the last read.

``native_ops.indexer_layer_rope_kv_append``
(``glm_5_2_nvfp4__sm_103__tp8/csrc/indexerLayerRopeKvAppend.cu``) takes the
arguments of ``native_ops.rope_kv_append`` and writes the same bytes (FP8 query,
latent and indexer K caches, FMHA scheduler buffers). Its kernel executes
``griddepcontrol.launch_dependents`` right after its last global read instead
of at its end (a block counts as triggered at its first thread's
invocation), so the next kernel starts while the rope's stores are in
flight: the indexer_layer_front CP logits kernel on full-indexer target layers and draft
step 0, sparse_mla_decode's sparse MLA decode on shared-indexer layers, rope_kv_append's top-k
bridge on draft steps 1-4, or their native fallbacks. Every such dependent reads the
rope's outputs only after its own ``griddepcontrol.wait`` and writes no
global memory before it, as rope_kv_append's dependents already must.

The caller (the adapter's rope_kv_append branch of ``_mla_gen_rope``) checks rope_kv_append's
``rope_eligible`` first and uses this op when indexer_layer_front is enabled; otherwise
rope_kv_append's op runs.
"""

from typing import List, Optional

import torch

from ... import native_ops
from ..rope_kv_append.ops import IndexerKScatter, _consume


def mla_rope_generation(
    mqa,
    fused_q,
    q_pe,
    latent_cache,
    metadata,
    cu_q_seqlens,
    cu_kv_seqlens,
    fmha_scheduler_counter,
    mla_bmm1_scale,
    mla_bmm2_scale,
    quant_q_buffer,
    indexer_k: Optional[IndexerKScatter] = None,
    early_trigger: bool = True,
) -> None:
    """rope_kv_append's ``mla_rope_generation`` (same arguments and bytes) with the indexer_layer_front kernel."""
    kvm = metadata.kv_cache_manager
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    idx_args: List[Optional[torch.Tensor]] = [None, None, None, None, None]
    idx_tokens = 0
    if indexer_k is not None:
        _consume(indexer_k, "front_rope")
        if indexer_k.num_tokens > 0:
            idx_args = [
                indexer_k.k_fp8,
                indexer_k.k_scale,
                indexer_k.k_cache,
                indexer_k.slot_mapping_fp8,
                indexer_k.slot_mapping_scale,
            ]
            idx_tokens = indexer_k.num_tokens
    native_ops.indexer_layer_rope_kv_append(
        fused_q,
        q_pe,
        latent_cache,
        mqa.rotary_cos_sin,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
        metadata.kv_lens_cuda_runtime,
        metadata.prompt_lens_cpu_runtime,
        metadata.num_contexts,
        metadata.kv_cache_block_offsets,
        kvm.kv_cache_pool_pointers,
        kvm.kv_cache_pool_mapping,
        getattr(mqa, "_nvfp4_mla_kv_scale_orig_quant", None),
        mqa.get_local_layer_idx(metadata),
        mqa.num_heads,
        kvm.tokens_per_block,
        metadata.max_seq_len,
        mqa.quant_mode,
        mqa.q_scaling,
        mqa.qk_nope_head_dim,
        mqa.qk_rope_head_dim,
        mqa.kv_lora_rank,
        *idx_args,
        idx_tokens,
        early_trigger,
    )
