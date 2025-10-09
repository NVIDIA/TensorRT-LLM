from typing import List

import torch
import transformers

from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.attention_backend import interface as attention_interface
from tensorrt_llm._torch.attention_backend import utils as attention_utils
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.modules import attention as trtllm_attention
from tensorrt_llm._torch.modules import gated_mlp as trtllm_gated_mlp
from tensorrt_llm._torch.modules import rms_norm as trtllm_rmsnorm


class PixtralAttention(trtllm_attention.Attention):
    def __init__(
        self,
        model_config: model_config_lib.ModelConfig[transformers.PixtralVisionConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        pos_embd_params = None
        max_position_embeddings = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=getattr(config, "torch_dtype", torch.float32),
            config=model_config,
            # Pixtral first needs to compute positional embeddings using its own
            # `PixtralRotaryEmbedding`.
            rope_fusion=False,
        )


class PixtralAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        config: model_config_lib.ModelConfig[transformers.PixtralVisionConfig],
        layer_idx: int,
    ):
        super().__init__()
        hidden_size = config.pretrained_config.hidden_size
        dtype = config.pretrained_config.torch_dtype
        self.attention_norm = trtllm_rmsnorm.RMSNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=dtype,
        )
        pretrained_config = config.pretrained_config

        if pretrained_config.hidden_act != "silu":
            raise ValueError(
                "Only 'silu' is accepted as the activation function for the MLP in "
                f"{self.__class__.__name__}. Got: {pretrained_config.hidden_act}."
            )
        self.feed_forward = trtllm_gated_mlp.GatedMLP(
            hidden_size=pretrained_config.hidden_size,
            intermediate_size=pretrained_config.intermediate_size,
            bias=False,
            activation=torch.nn.functional.silu,
            dtype=pretrained_config.torch_dtype,
            config=config,
        )
        self.attention = PixtralAttention(config, layer_idx)
        self.ffn_norm = trtllm_rmsnorm.RMSNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: attention_interface.AttentionMetadata,
        position_ids: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            # NOTE: although we do not need the `position_ids` to compute ROPE (since it has already
            # been pre-computed), internally, the cos / sin vectors will not be applied to the
            # query / key tensors if `position_ids=None` in `RotaryEmbedding`.
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_interface.PredefinedAttentionMask.FULL,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# Original implementation:
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/pixtral/modeling_pixtral.py#L279
class PixtralTransformer(torch.nn.Module):
    def __init__(self, config: model_config_lib.ModelConfig[transformers.PixtralVisionConfig]):
        super().__init__()
        tp_size = config.mapping.tp_size
        num_heads = config.pretrained_config.num_attention_heads
        if (num_heads % tp_size) > 0:
            raise ValueError(f"{tp_size=} must divide {num_heads=}.")
        num_heads //= tp_size

        self._head_dim = config.pretrained_config.head_dim
        self._num_heads = num_heads

        self.layers = torch.nn.ModuleList()
        for i in range(config.pretrained_config.num_hidden_layers):
            self.layers.append(PixtralAttentionLayer(config=config, layer_idx=i))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: attention_interface.AttentionMetadata,
    ):
        if inputs_embeds.ndim == 3:
            batch_size, patches, _ = inputs_embeds.shape
        elif inputs_embeds.ndim == 2:
            batch_size = 1
            patches = inputs_embeds.size(0)
        rope_function = _RopeFunction(
            batch_size=batch_size,
            patches=patches,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
            position_embeddings=position_embeddings,
        )
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            # The way pixtral applies rope is by:
            # 1. Computing the `position_ids` using the `patch_embeds` (which are essentially a
            #    sliced + concat'ed output of the conv2d layer), using their positions in the
            #    a meshgrid.
            # 2. Computing `position_embeddings` once for the entire transformer portion.
            #    See: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/
            #         models/pixtral/modeling_pixtral.py#L494
            # 3. These `position_embeddings` are then the ones used to apply rope in each attention
            #    layer.
            # By substituting the `encoder_layer.attention.rotary_emb` to use `_RopeFunction`, which
            # has these `position_embeddings` as an attribute, we can reuse the embeddings + application
            # logic for each encoder layer.
            encoder_layer.attention.rotary_emb = rope_function
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs

        return hidden_states


# Original implementation:
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/pixtral/modeling_pixtral.py#L440
@modeling_utils.register_auto_model("PixtralVisionModel")
class PixtralVisionModel(torch.nn.Module):
    def __init__(
        self, model_config: model_config_lib.ModelConfig[transformers.PixtralVisionConfig]
    ):
        super().__init__()
        # Both the below are needed in order to use `_load_weights_impl`.
        self.model_config = model_config
        self.config: transformers.PixtralVisionConfig = model_config.pretrained_config
        self.patch_conv = torch.nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )
        self._patch_size = self.config.patch_size
        self.ln_pre = trtllm_rmsnorm.RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=1e-5,
            dtype=self.config.torch_dtype,
        )
        self.transformer = PixtralTransformer(model_config)
        self._patch_positional_embedding = (
            transformers.models.pixtral.modeling_pixtral.PixtralRotaryEmbedding(self.config)
        )

        self._metadata_cls = attention_utils.get_attention_backend(
            model_config.attn_backend
        ).Metadata

    @torch.inference_mode()
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        with torch.autocast(device_type="cuda", dtype=self.config.torch_dtype):
            patch_embeds = self.patch_conv(pixel_values)

        patch_embeds_list = [
            embed[..., : (size[0] // self._patch_size), : (size[1] // self._patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        flattened_embeds = [p.flatten(1).T for p in patch_embeds_list]
        patch_embeds = torch.cat(flattened_embeds, dim=0)
        patch_embeds = self.ln_pre(patch_embeds)

        # The `position_ids_in_meshgrid` code does not look at the inputs' device to create the
        # `position_ids`, so it ends up defaulting to CPU, which will incur an H2D copy later down
        # the line. We therefore use this `torch.device` context manager here.
        with torch.device(device=pixel_values.device):
            position_ids = transformers.models.pixtral.modeling_pixtral.position_ids_in_meshgrid(
                patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
            )
        position_embeddings = self._patch_positional_embedding(patch_embeds, position_ids)

        attn_metadata = self._prepare_attn_metadata(
            batch_size=pixel_values.size(0),
            seq_lengths=[x.size(0) for x in flattened_embeds],
        )
        out = self.transformer(
            patch_embeds,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
        )

        return out

    def load_weights(self, weights):
        modeling_utils._load_weights_impl(self, weights)

    def _prepare_attn_metadata(self, batch_size: int, seq_lengths: List[int]):
        request_ids = list(range(1, batch_size + 1))
        attn_metadata = self._metadata_cls(
            seq_lens=torch.tensor(seq_lengths, dtype=torch.int),
            num_contexts=batch_size,
            max_num_requests=batch_size,
            max_num_tokens=sum(seq_lengths),
            kv_cache_manager=None,
            request_ids=request_ids,
            prompt_lens=seq_lengths,
        )
        attn_metadata.max_seq_len = max(seq_lengths)
        attn_metadata.prepare()
        return attn_metadata


class _RopeFunction:
    def __init__(
        self,
        batch_size: int,
        patches: int,
        num_heads: int,
        head_dim: int,
        position_embeddings: torch.Tensor,
    ):
        self._batch_size = batch_size
        self._patches = patches
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._cos, self._sin = position_embeddings

    # This signature matches that of
    # `tensorrt_llm/_torch/modules/rotary_embedding.py::RotaryEmbedding.forward` so that we are
    # able to override the `PixtralAttentionLayer.rotary_embed` attribute.
    @torch.no_grad()
    def __call__(
        self,
        # Unused.
        position_ids: torch.Tensor,
        # Assumed to be in the order `[q, k]`.
        targets: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(targets) != 2:
            raise ValueError("Expected exactly two targets [q, k].")

        # TODO: see if we can reuse `RotaryEmbedding.apply_rotary_pos_emb`.
        orig_shape = targets[0].shape
        q_embed, k_embed = transformers.models.pixtral.modeling_pixtral.apply_rotary_pos_emb(
            q=targets[0]
            .view(
                self._batch_size,
                self._patches,
                self._num_heads,
                self._head_dim,
            )
            .transpose(1, 2),
            k=targets[1]
            .view(
                self._batch_size,
                self._patches,
                self._num_heads,
                self._head_dim,
            )
            .transpose(1, 2),
            cos=self._cos,
            sin=self._sin,
            unsqueeze_dim=0,
        )

        q_embed = q_embed.transpose(2, 1).reshape(orig_shape)
        k_embed = k_embed.transpose(2, 1).reshape(orig_shape)

        return [q_embed, k_embed]
