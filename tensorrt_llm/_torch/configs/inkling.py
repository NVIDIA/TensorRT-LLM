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
"""Config classes for the Inkling multimodal checkpoint (text bring-up).

The in-scope checkpoint (``Inkling-NVFP4-full``) publishes ``config.json`` with
``model_type == "inkling_mm_model"`` and ``architectures ==
["InklingForConditionalGeneration"]``. The installed transformers pin does not
ship Inkling, so these classes reconstruct the config from the checkpoint's
nested dicts without any transformers shim (reference-test policy: no installed
``transformers`` Inkling remote-code as pass evidence).

Only the text tower drives the GSM8K/MMLU accuracy gates, so the audio, vision,
and MTP sub-configs are kept verbatim (as ``PretrainedConfig`` blobs) but are
not otherwise interpreted here.

Field names mirror the checkpoint ``text_config`` and the SGLang / HF reference
(``codes/sglang/.../models/inkling*`` and
``codes/transformers/.../models/inkling/``). All numeric defaults are the real
checkpoint values, but a checkpoint ``config.json`` that spells a field out
overrides the default via ``from_dict``.
"""

from transformers.configuration_utils import PretrainedConfig


class InklingTextConfig(PretrainedConfig):
    """Text-tower sub-config (``InklingCausalLLM``).

    A RoPE-free hybrid-attention decoder: per-head q/k RMSNorm, learned
    relative-position bias, four short convolutions per layer, sigmoid-gated MoE
    with two shared experts, muP logit scaling, and an unpadded vocab slice.
    """

    model_type = "inkling_text"

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        hidden_size: int = 6144,
        num_hidden_layers: int = 66,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        model_max_length: int = 1048576,
        logits_mup_width_multiplier: float = 24.0,
        use_embed_norm: bool = True,
        tie_word_embeddings: bool = False,
        # hybrid attention geometry
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 512,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 16,
        swa_head_dim: int = 128,
        # relative-bias / log-scaling
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 128000,
        log_scaling_alpha: float = 0.1,
        # short conv
        use_sconv: bool = True,
        sconv_kernel_size: int = 4,
        # dense MLP / MoE
        dense_mlp_idx: int = 2,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 24576,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 2,
        shared_expert_sink: bool = True,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        gate_activation: str = "sigmoid",
        norm_after_topk: bool = True,
        use_global_scale: bool = True,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.model_max_length = model_max_length
        # `max_position_embeddings` is read by several TRT-LLM code paths
        # (Attention, RopeParams probing); Inkling has no RoPE but keep the 1M
        # context window available so nothing clamps sequence length.
        self.max_position_embeddings = kwargs.get("max_position_embeddings", model_max_length)
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.use_embed_norm = use_embed_norm

        self.local_layer_ids = list(local_layer_ids) if local_layer_ids else []
        self.sliding_window_size = sliding_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim

        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha

        self.use_sconv = use_sconv
        self.sconv_kernel_size = sconv_kernel_size

        self.dense_mlp_idx = dense_mlp_idx
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.shared_expert_sink = shared_expert_sink
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.gate_activation = gate_activation
        self.norm_after_topk = norm_after_topk
        self.use_global_scale = use_global_scale
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout

    # ---- per-layer classification helpers (single source of truth) ----
    @property
    def _local_ids(self) -> set:
        return set(self.local_layer_ids)

    def is_dense_layer(self, layer_idx: int) -> bool:
        """Dense MLP layers are the ones with index < ``dense_mlp_idx``."""
        return layer_idx < self.dense_mlp_idx

    def is_local_layer(self, layer_idx: int) -> bool:
        """Local (sliding-window) layers are listed in ``local_layer_ids``."""
        return layer_idx in self._local_ids

    def layer_num_kv_heads(self, layer_idx: int) -> int:
        return (
            self.swa_num_key_value_heads
            if self.is_local_layer(layer_idx)
            else self.num_key_value_heads
        )

    def layer_num_heads(self, layer_idx: int) -> int:
        return (
            self.swa_num_attention_heads
            if self.is_local_layer(layer_idx)
            else self.num_attention_heads
        )

    def layer_head_dim(self, layer_idx: int) -> int:
        return self.swa_head_dim if self.is_local_layer(layer_idx) else self.head_dim

    def layer_window(self, layer_idx: int) -> int | None:
        """Sliding-window size for local layers; ``None`` for global layers."""
        return self.sliding_window_size if self.is_local_layer(layer_idx) else None

    def num_kv_heads_per_layer(self) -> list[int]:
        """Per-layer KV-head counts for the hybrid attention geometry.

        Local (sliding-window) layers use ``swa_num_key_value_heads`` (16) and
        global layers use ``num_key_value_heads`` (8). ``KVCacheManagerV2``
        accepts this ``List[int]`` as ``num_kv_heads`` (it divides each by
        ``tp_size``), so the paged KV cache allocates the right per-layer head
        count instead of a single uniform value. ``head_dim`` is uniform (128)
        across local and global layers, so only the KV-head count varies.
        """
        return [self.layer_num_kv_heads(i) for i in range(self.num_hidden_layers)]


class InklingMTPConfig(PretrainedConfig):
    """MTP (multi-token-prediction / next-N EAGLE draft) sub-config.

    The checkpoint ships ``mtp_config = {num_nextn_predict_layers,
    chain_hidden_post_norm, local_layer_ids}``. Each of the ``N`` draft depths is
    one ``model.mtp.layers.{d}`` block: two RMSNorms (``embed_norm`` +
    ``hidden_norm``), an ``input_proj`` Linear on ``concat(hidden_norm(hidden),
    embed_norm(embed))`` (hidden ``2*hidden_size -> hidden_size``), and a
    ``transformer_block`` that is an Inkling decoder layer forced to the DENSE MLP
    (``w13_dn``/``w2_md``, never MoE).

    Per-depth banded attention mirrors the trunk (SGLang ``mtp_swa_* ==`` trunk
    ``swa_*``): draft depths listed in ``local_layer_ids`` are sliding-window
    (SWA head geometry, window = ``mtp_local_extent`` default = trunk
    ``sliding_window_size``); the other depths are full/global attention. All 160
    MTP tensors are BF16 in the checkpoint and MUST be built UNQUANTIZED -- they
    are absent from ``hf_quant_config.json`` ``exclude_modules`` only because the
    quantizer never touched them, so quantization state must NOT be inferred from
    the exclude list (mirrors SGLang forcing ``quant_config=None`` on the MTP
    block). Geometry fields are injected from the sibling ``text_config`` by
    :meth:`InklingConfig._as_mtp_config`.
    """

    model_type = "inkling_mtp"

    def __init__(
        self,
        num_nextn_predict_layers: int = 8,
        chain_hidden_post_norm: bool = False,
        local_layer_ids: list[int] | None = None,
        # geometry mirrored from the text tower (defaults are the real checkpoint
        # values; InklingConfig injects the actual text_config geometry).
        hidden_size: int = 6144,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 16,
        swa_head_dim: int = 128,
        sliding_window_size: int = 512,
        mtp_local_extent: int | None = None,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_nextn_predict_layers = int(num_nextn_predict_layers)
        self.chain_hidden_post_norm = bool(chain_hidden_post_norm)
        self.local_layer_ids = list(local_layer_ids) if local_layer_ids else []
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        self.mtp_local_extent = (
            mtp_local_extent if mtp_local_extent is not None else sliding_window_size
        )
        self.rms_norm_eps = rms_norm_eps

    # ---- per-depth banded-attention helpers (mirror InklingTextConfig) ----
    @property
    def _local_ids(self) -> set:
        return set(self.local_layer_ids)

    def is_local_depth(self, depth: int) -> bool:
        """Draft depths in ``local_layer_ids`` are sliding-window (SWA)."""
        return depth in self._local_ids

    def depth_num_kv_heads(self, depth: int) -> int:
        return (
            self.swa_num_key_value_heads
            if self.is_local_depth(depth)
            else self.num_key_value_heads
        )

    def depth_num_heads(self, depth: int) -> int:
        return (
            self.swa_num_attention_heads
            if self.is_local_depth(depth)
            else self.num_attention_heads
        )

    def depth_head_dim(self, depth: int) -> int:
        return self.swa_head_dim if self.is_local_depth(depth) else self.head_dim

    def depth_window(self, depth: int) -> int | None:
        """Sliding-window size for local draft depths; ``None`` for global depths."""
        return self.mtp_local_extent if self.is_local_depth(depth) else None


class InklingConfig(PretrainedConfig):
    """Top-level Inkling multimodal config (``inkling_mm_model``).

    Reconstructs ``text_config`` with :class:`InklingTextConfig`; ``audio_config``,
    ``vision_config`` and ``mtp_config`` are retained as plain
    ``PretrainedConfig`` blobs so the multimodal checkpoint round-trips, but only
    the text tower is built for the GSM8K/MMLU bring-up.
    """

    model_type = "inkling_mm_model"
    sub_configs = {"text_config": InklingTextConfig}

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        vision_config=None,
        mtp_config=None,
        eos_token_id: int = 200006,
        # Multimodal placeholder ids. One appears per media item in the
        # pre-rendered token stream; the input processor expands it to one token
        # per patch (image) / audio frame and the vision/audio fusion overwrites
        # those positions. Both placeholders are IN-VOCAB chat-template tokens:
        # image ``<|unused_200054|>`` (id 200054) and audio ``<|unused_200053|>``
        # (id 200053) -- exactly the tokens the Inkling chat template renders for
        # an image / audio content part
        # (``<|content_image|><|unused_200054|>...`` and
        # ``<|content_audio_input|><|unused_200053|><|audio_end|>...``).
        # They MUST be in-vocab: TensorRT-LLM's executor validates request token
        # ids and rejects an out-of-range id, so the SGLang-internal -101/-102
        # sentinels (parser/inkling_tokenizer.py) raise ``RequestError: Token ID
        # out of range`` at ``llm.generate``. The sentinel and the in-vocab id
        # are interchangeable for parity (both are replaced by the same tower
        # embeddings); the SGLang reference is POSTed with -101/-102, TRT runs
        # 200054/200053. The in-scope checkpoint's config.json omits these, so the
        # defaults are authoritative unless a config spells them out.
        image_token_id: int = 200054,
        audio_token_id: int = 200053,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id

        if text_config is None:
            self.text_config = InklingTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = InklingTextConfig(**text_config)
        else:
            self.text_config = text_config

        # Retained verbatim; interpreted only in the Phase-3 multimodal stage.
        self.audio_config = self._as_config(audio_config)
        self.vision_config = self._as_config(vision_config)
        # MTP is surfaced as a typed InklingMTPConfig (Stage 9): the 3 checkpoint
        # fields plus the per-depth banded-attention geometry pulled from the text
        # tower (SGLang mtp_swa_* == trunk swa_*). ``None`` for a checkpoint with
        # no mtp_config.
        self.mtp_config = self._as_mtp_config(mtp_config, self.text_config)

    @property
    def has_mtp(self) -> bool:
        """True iff the checkpoint ships MTP/next-N draft layers to load."""
        return (
            self.mtp_config is not None
            and int(getattr(self.mtp_config, "num_nextn_predict_layers", 0)) > 0
        )

    @staticmethod
    def _as_config(value):
        if value is None or isinstance(value, PretrainedConfig):
            return value
        if isinstance(value, dict):
            cfg = PretrainedConfig()
            for k, v in value.items():
                setattr(cfg, k, v)
            return cfg
        return value

    @staticmethod
    def _as_mtp_config(value, text_config):
        """Build a typed :class:`InklingMTPConfig`, injecting the text-tower
        geometry the checkpoint's ``mtp_config`` blob omits (hidden size, head
        counts, SWA geometry, window). Accepts the checkpoint dict, an existing
        InklingMTPConfig (round-trip), or a plain blob; ``None`` stays ``None``.
        """
        if value is None or isinstance(value, InklingMTPConfig):
            return value
        if isinstance(value, dict):
            blob = dict(value)
        else:  # a PretrainedConfig / object blob
            blob = {
                k: v
                for k, v in getattr(value, "__dict__", {}).items()
                if not k.startswith("_")
            }
        geom = dict(
            hidden_size=text_config.hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            swa_num_attention_heads=text_config.swa_num_attention_heads,
            swa_num_key_value_heads=text_config.swa_num_key_value_heads,
            swa_head_dim=text_config.swa_head_dim,
            sliding_window_size=text_config.sliding_window_size,
            rms_norm_eps=text_config.rms_norm_eps,
        )
        # Only pass the MTP-defining fields from the blob; geometry always comes
        # from the text tower so a partial round-trip cannot desync them.
        mtp_fields = {
            k: blob[k]
            for k in ("num_nextn_predict_layers", "chain_hidden_post_norm",
                      "local_layer_ids", "mtp_local_extent")
            if k in blob
        }
        return InklingMTPConfig(**geom, **mtp_fields)
