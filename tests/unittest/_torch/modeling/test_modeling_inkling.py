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
"""CPU-only structural tests for the Inkling text tower.

The config-and-geometry tests run anywhere; the weight-accounting and tensor-shape
tests read the checkpoint's safetensors *index* only (a few hundred KB of JSON, no
weights, no GPU) and skip when the checkpoint is not available. Together they
guarantee no required text tensor (q/k norm, relative bias, short conv,
route/global scale, unpadded logits) can silently go missing.
"""

import json
import os
import struct
from types import SimpleNamespace

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.configs.inkling import InklingConfig, InklingTextConfig
from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
    inkling_account_checkpoint,
    inkling_nvfp4_expert_layers,
)

_models_root = llm_models_root()
CHECKPOINT = os.environ.get(
    "INKLING_CHECKPOINT", str(_models_root / "Inkling-NVFP4") if _models_root else ""
)

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT or not os.path.isdir(CHECKPOINT), reason="Inkling checkpoint not available"
)

# The checkpoint's hybrid attention pattern: every 6th layer at offset 5 is a
# global (full-causal) layer, the rest are local (sliding-window).
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
LOCAL_LAYER_IDS = [n for n in range(66) if n not in GLOBAL_LAYERS]


def _index(ckpt: str) -> dict:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _exclude_modules(ckpt: str) -> set:
    with open(os.path.join(ckpt, "hf_quant_config.json")) as f:
        return set(json.load(f)["quantization"].get("exclude_modules", []))


def _safetensors_shape(ckpt: str, key: str):
    """Read one tensor's shape/dtype from its shard header (no tensor data)."""
    shard = _index(ckpt)[key]
    with open(os.path.join(ckpt, shard), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header[key]["shape"], header[key]["dtype"]


@pytest.fixture(scope="module")
def ckpt_config() -> InklingTextConfig:
    """The real checkpoint's text sub-config (checkpoint-gated tests only)."""
    return InklingConfig.from_pretrained(CHECKPOINT).text_config


def test_model_and_config_are_registered():
    # Importing the model module registers the auto-model and the weight mapper.
    import tensorrt_llm._torch.models.modeling_inkling  # noqa: F401
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    assert "InklingForConditionalGeneration" in MODEL_CLASS_MAPPING
    cfg = InklingConfig()
    assert cfg.model_type == "inkling_mm_model"
    assert isinstance(cfg.text_config, InklingTextConfig)
    assert cfg.text_config.model_type == "inkling_text"


def test_model_defaults_pin_v2_and_disable_block_reuse():
    """Both defaults are correctness requirements, not tuning.

    V2 because the per-layer KV-head split (local 16 / global 8) needs
    per-layer geometry V1 cannot represent; block reuse off because the four
    depthwise short convs per layer hold a ``kernel_size - 1`` window outside
    the KV cache, and ``InklingConvRuntime.build`` seeds every context request
    with ``has_initial_state=False`` -- a reused prefix would restart the
    convolutions from zeros while attention resumed from real history.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration

    defaults = InklingForConditionalGeneration.get_model_defaults(None)

    assert defaults["kv_cache_config"]["use_kv_cache_manager_v2"] is True
    assert defaults["kv_cache_config"]["enable_block_reuse"] is False


def test_block_reuse_default_departs_from_the_framework_default():
    """The framework enables block reuse by default, so the model default is
    load-bearing. If KvCacheConfig ever flips its default, this test still
    documents which direction Inkling needs."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    assert KvCacheConfig().enable_block_reuse is True
    defaults = InklingForConditionalGeneration.get_model_defaults(None)
    assert defaults["kv_cache_config"]["enable_block_reuse"] is False


def test_text_geometry():
    """The config defaults are the checkpoint's text-tower geometry."""
    tc = InklingTextConfig()
    assert (tc.num_hidden_layers, tc.hidden_size, tc.head_dim) == (66, 6144, 128)
    assert (tc.num_attention_heads, tc.num_key_value_heads) == (64, 8)
    assert (tc.vocab_size, tc.unpadded_vocab_size) == (201024, 200058)
    assert tc.logits_mup_width_multiplier == 24.0
    assert tc.use_embed_norm is True
    assert (tc.n_routed_experts, tc.num_experts_per_tok, tc.n_shared_experts) == (256, 6, 2)
    assert tc.dense_mlp_idx == 2
    assert (tc.sliding_window_size, tc.swa_num_key_value_heads) == (512, 16)


def test_layer_classification():
    """Dense = {0, 1}; local/global geometry follows ``local_layer_ids``."""
    tc = InklingTextConfig(local_layer_ids=LOCAL_LAYER_IDS)
    assert [n for n in range(tc.num_hidden_layers) if tc.is_dense_layer(n)] == [0, 1]
    assert [n for n in range(tc.num_hidden_layers) if not tc.is_local_layer(n)] == GLOBAL_LAYERS
    # local: 16 kv-heads behind a 512 window; global: 8 kv-heads, no window.
    assert tc.layer_num_kv_heads(0) == 16 and tc.layer_window(0) == 512
    assert tc.layer_num_kv_heads(5) == 8 and tc.layer_window(5) is None
    # The paged KV cache is sized per layer from exactly this hybrid split.
    assert tc.num_kv_heads_per_layer() == [
        tc.layer_num_kv_heads(n) for n in range(tc.num_hidden_layers)
    ]


@requires_checkpoint
def test_checkpoint_layer_pattern_matches_config_defaults(ckpt_config):
    """The checkpoint declares the hybrid pattern the CPU tests assume."""
    assert ckpt_config.local_layer_ids == LOCAL_LAYER_IDS


@requires_checkpoint
def test_text_weight_accounting(ckpt_config):
    """Every checkpoint key is consumed-text or intentionally deferred; the text
    tower is exactly and fully covered (nothing missing, nothing unaccounted)."""
    exclude = _exclude_modules(CHECKPOINT)
    all_keys = set(_index(CHECKPOINT))

    acct = inkling_account_checkpoint(all_keys, ckpt_config, exclude)
    assert not acct["unaccounted"], sorted(acct["unaccounted"])[:10]
    assert not acct["missing"], sorted(acct["missing"])[:10]
    assert all(
        k.startswith(("model.audio.", "model.visual.", "model.mtp.")) for k in acct["deferred"]
    )
    assert len(acct["consumed_text"]) + len(acct["deferred"]) == len(all_keys)

    # NVFP4 routed experts are exactly layers 3..65 (layer-2 experts are bf16).
    assert inkling_nvfp4_expert_layers(ckpt_config, exclude) == list(range(3, 66))
    assert "model.llm.layers.2.mlp.experts" in exclude
    assert "model.llm.layers.3.mlp.experts" not in exclude


@requires_checkpoint
def test_checkpoint_tensor_shapes_match_geometry(ckpt_config):
    """Sampled checkpoint tensors have the shapes the modules construct."""
    tc = ckpt_config
    hd, hidden = tc.head_dim, tc.hidden_size

    # q/k/v/r projection out-dims (layer 0 is local: 16 kv-heads).
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wq_du.weight")[0] == [
        tc.num_attention_heads * hd,
        hidden,
    ]
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wk_dv.weight")[0] == [
        tc.swa_num_key_value_heads * hd,
        hidden,
    ]
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.5.attn.wk_dv.weight")[0] == [
        tc.num_key_value_heads * hd,
        hidden,
    ]  # global layer
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wr_du.weight")[0] == [
        tc.num_attention_heads * tc.d_rel,
        hidden,
    ]

    # short-conv depthwise weight: [channels, 1, kernel].
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.k_sconv.weight")[0] == [
        tc.swa_num_key_value_heads * hd,
        1,
        tc.sconv_kernel_size,
    ]

    # NVFP4 routed experts: [E, 2*inter, hidden/2] packed uint8 + block scale.
    shape, dtype = _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight")
    assert shape == [tc.n_routed_experts, 2 * tc.intermediate_size, hidden // 2]
    assert dtype in ("U8", "UINT8")
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight.scale")[0] == [
        tc.n_routed_experts,
        2 * tc.intermediate_size,
        hidden // 16,
    ]

    # The router covers routed + shared experts.
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.gate.weight")[0] == [
        tc.n_routed_experts + tc.n_shared_experts,
        hidden,
    ]


# ---------------------------------------------------------------------------
# InklingAttentionMetadata: the per-step decode publish
# ---------------------------------------------------------------------------
class _FakeKvManager:
    """Minimal stand-in for KVCacheManagerV2's per-layer block-table API."""

    def __init__(self, pp_layers, blocks_by_layer, max_blocks_per_seq=4):
        self.pp_layers = pp_layers
        self.max_blocks_per_seq = max_blocks_per_seq
        self._blocks = blocks_by_layer
        self.calls = []

    def get_batch_cache_indices(self, request_ids, layer_idx):
        self.calls.append((tuple(request_ids), layer_idx))
        return self._blocks[layer_idx]


def _ink_metadata(
    num_contexts=0, request_ids=(7, 9), num_cached=(3, 130), pp_layers=(0, 1), max_blocks_per_seq=4
):
    """An InklingAttentionMetadata with prepare()'s inputs stubbed in.

    Builds the object without running AttentionMetadata's dataclass __init__ so
    the test stays CPU-only and independent of the KV-cache stack; only the
    fields _prepare_inkling_decode reads are populated.
    """
    import torch

    from tensorrt_llm._torch.attention_backend.inkling_triton import InklingAttentionMetadata

    md = object.__new__(InklingAttentionMetadata)
    md.ink_num_gen = 0
    md.ink_max_pages = None
    md.ink_cap = 0
    md.ink_seq_lens = None
    md.ink_page_table = {}
    md._ink_sl_host = None
    md._ink_pt_host = None
    md.is_cuda_graph = False
    md.request_ids = list(request_ids)
    md._num_contexts = num_contexts
    md.kv_cache_params = SimpleNamespace(num_cached_tokens_per_seq=list(num_cached))
    blocks = {
        layer: [[layer * 10 + 1, -1, -1, -1], [layer * 10 + 2, layer * 10 + 3, -1, -1]]
        for layer in pp_layers
    }
    md.kv_cache_manager = _FakeKvManager(list(pp_layers), blocks, max_blocks_per_seq)
    md.seq_lens_cuda = torch.zeros(1, dtype=torch.int32)
    return md


def test_metadata_publishes_total_kv_lengths_and_per_layer_page_table():
    """seq_lens is num_cached + 1 and layer-independent; the page table is
    per-layer because get_batch_cache_indices is per pool_id."""
    md = _ink_metadata()

    md._prepare_inkling_decode()

    assert md.ink_num_gen == 2
    assert md.ink_seq_lens[:2].tolist() == [4, 131]
    # One block-table fetch per layer this rank owns, all on the generation ids.
    assert md.kv_cache_manager.calls == [((7, 9), 0), ((7, 9), 1)]
    assert md.ink_page_table[0][:2].tolist() == [[1, 0, 0, 0], [2, 3, 0, 0]]
    assert md.ink_page_table[1][:2].tolist() == [[11, 0, 0, 0], [12, 13, 0, 0]]


def test_metadata_skips_the_context_slice():
    """Only generation rows get decode metadata; context rows run the prefill
    kernel and are excluded by num_contexts."""
    md = _ink_metadata(num_contexts=1, request_ids=(7, 9), num_cached=(0, 130))

    md._prepare_inkling_decode()

    assert md.ink_num_gen == 1
    assert md.ink_seq_lens[:1].tolist() == [131]
    assert md.kv_cache_manager.calls == [((9,), 0), ((9,), 1)]


def test_metadata_reports_nothing_published_for_a_context_only_batch():
    """A prefill-only step must not leave the previous step's page table
    advertised as current -- that is what the old epoch counter guarded."""
    md = _ink_metadata()
    md._prepare_inkling_decode()
    assert md.ink_num_gen == 2

    md._num_contexts = 2  # same object, now a context-only batch
    md._prepare_inkling_decode()

    assert md.ink_num_gen == 0


def test_metadata_refuses_to_grow_its_buffers_under_cuda_graph():
    """Growth would strand the captured pointer, so it must raise rather than
    silently reallocate."""
    md = _ink_metadata()
    md._prepare_inkling_decode()
    md.is_cuda_graph = True
    md.request_ids = [7, 9, 11, 13]
    md.kv_cache_params = SimpleNamespace(num_cached_tokens_per_seq=[3, 130, 5, 5])

    with pytest.raises(RuntimeError, match="CUDA graph"):
        md._prepare_inkling_decode()


def test_metadata_clamps_a_row_to_max_pages():
    """A row longer than max_blocks_per_seq is truncated, never written past
    the stable buffer's width."""
    md = _ink_metadata(max_blocks_per_seq=2)
    md.kv_cache_manager._blocks = {0: [[1, 2, 3, 4], [5, 6, 7, 8]], 1: [[1, 2, 3, 4], [5, 6, 7, 8]]}

    md._prepare_inkling_decode()

    assert md.ink_max_pages == 2
    assert md.ink_page_table[0][:2].tolist() == [[1, 2], [5, 6]]


def test_inkling_backend_is_registered_and_carries_the_metadata():
    from tensorrt_llm._torch.attention_backend.inkling_triton import (
        InklingAttentionMetadata,
        InklingTritonAttention,
    )
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    assert get_attention_backend("INKLING") is InklingTritonAttention
    assert InklingTritonAttention.Metadata is InklingAttentionMetadata


def test_model_defaults_select_the_inkling_backend():
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration

    assert InklingForConditionalGeneration.get_model_defaults(None)["attn_backend"] == "INKLING"


def test_conv_state_manager_implements_the_capability_protocol():
    """The CONV_STATE_MANAGER resource type must stand behind a capability
    interface, not a model-specific branch, so a consumer can isinstance-test
    it the way _prepare_mamba_metadata tests BaseMambaCacheManager."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingConvStateManager
    from tensorrt_llm._torch.pyexecutor.conv_state_manager import BaseConvStateManager

    assert issubclass(InklingConvStateManager, BaseConvStateManager)
    # No abstract method left unimplemented -- an incomplete implementation
    # would only surface at instantiation time on a GPU node otherwise.
    assert not getattr(InklingConvStateManager, "__abstractmethods__", frozenset())


def test_conv_state_protocol_is_not_the_mamba_one():
    """Kept separate on purpose: BaseMambaCacheManager mandates get_ssm_states /
    is_speculative / mamba_layer_cache, and Inkling has no selective-scan state
    to back them with."""
    from tensorrt_llm._torch.pyexecutor.conv_state_manager import BaseConvStateManager
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import BaseMambaCacheManager

    assert not issubclass(BaseConvStateManager, BaseMambaCacheManager)
    assert set(BaseConvStateManager.__abstractmethods__) == {
        "get_conv_state_cache",
        "write_conv_state_indices",
        "free_conv_state",
    }


def test_attn_backend_override_fails_loudly():
    """An attn_backend override silently removes the decode publish and the run
    then dies inside CUDA-graph capture with a message that names neither
    Inkling nor the setting. Catch it at load instead."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    check = InklingForCausalLM._assert_inkling_attn_backend
    check(SimpleNamespace(attn_backend="INKLING"))  # no raise
    check(SimpleNamespace(attn_backend="inkling"))  # case-insensitive
    check(SimpleNamespace(attn_backend=None))  # nothing to check

    for bad in ("TRTLLM", "FLASHINFER", "VANILLA"):
        with pytest.raises(ValueError, match="attn_backend='INKLING'"):
            check(SimpleNamespace(attn_backend=bad))
