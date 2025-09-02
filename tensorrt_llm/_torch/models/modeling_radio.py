import math
from collections import namedtuple
from typing import (Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple,
                    Type, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.attention_backend import \
    interface as attention_interface
from tensorrt_llm._torch.attention_backend import utils as attention_utils
from tensorrt_llm._torch.modules import attention as trtllm_attention
from tensorrt_llm._torch.modules import mlp as trtllm_mlp

input_dim_t = Union[int, Tuple[int, int]]

DEFAULT_VERSION = "radio_v2.5-h"


class Resolution(NamedTuple):
    height: int
    width: int


class RADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = DEFAULT_VERSION,
        patch_size: Optional[int] = None,
        max_resolution: Optional[int] = None,
        preferred_resolution: Optional[Resolution] = None,
        adaptor_names: Union[str, List[str]] = None,
        adaptor_configs: Dict[str, Dict[str, int]] = None,
        vitdet_window_size: Optional[int] = None,
        feature_normalizer_config: Optional[dict] = None,
        inter_feature_normalizer_config: Optional[dict] = None,
        **kwargs,
    ):
        self.args = args
        for field in ["dtype", "amp_dtype"]:
            if self.args is not None and field in self.args:
                # Convert to a string in order to make it serializable.
                # For example for torch.float32 we will store "float32",
                # for "bfloat16" we will store "bfloat16".
                self.args[field] = str(args[field]).split(".")[-1]
        self.version = version
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        self.preferred_resolution = preferred_resolution

        self.adaptor_names = adaptor_names
        self.adaptor_configs = adaptor_configs
        self.vitdet_window_size = vitdet_window_size
        self.feature_normalizer_config = feature_normalizer_config
        self.inter_feature_normalizer_config = inter_feature_normalizer_config
        self.num_key_value_heads = 16  # Hardcoded for model weight loading
        self.num_attention_heads = 16  # Hardcoded for model weight loading
        super().__init__(**kwargs)


class ClsToken(nn.Module):

    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
    ):
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (num_tokens %
                                                          register_multiple)

            scale = ndim**-0.5
            self.token = nn.Parameter(
                torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def disable(self):
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([
            token,
            x,
        ], dim=1)

        return x


class ViTPatchGenerator(nn.Module):

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: input_dim_t,
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[input_dim_t] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
        patch_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)

        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims)

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc

        factory = dict(device=device, dtype=dtype)

        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim

        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size,
                                       embed_dim,
                                       bias=patch_bias,
                                       **factory)

        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, **factory) * scale)

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )

        self.patch_normalizer = nn.LayerNorm(
            embed_dim) if normalize_patches else nn.Identity()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.im_to_patches(x)
        patches = self.embedder(patches)
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)
        return patches + pos_enc, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(
            -1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(pos_embed.expand(patch_idxs.shape[0], -1, -1),
                                 dim=1,
                                 index=exp_patch_idxs)
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols,
                                           -1).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., :input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, :input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            max_dim = max(input_dims)
            pos_embed = F.interpolate(pos_embed.float(),
                                      size=(max_dim, max_dim),
                                      align_corners=True,
                                      mode='bilinear').to(pos_embed.dtype)

            pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(pos_embed.float(),
                                      size=input_dims,
                                      align_corners=True,
                                      mode='bilinear').to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            'b c (py yy) (px xx) -> b (py px) (c yy xx)',
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):

    def __init__(self,
                 patch_size: int,
                 embed_dim: int,
                 bias: bool = False,
                 **factory):
        super().__init__(3 * (patch_size**2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        layer_idx: Optional[int] = None,
        model_config: model_config_lib.ModelConfig[RADIOConfig] = None,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.model_config = model_config
        self.norm1 = norm_layer(dim)

        if qk_norm:
            raise IOError("Block does not support qk_norm for now.")
        self.attn = trtllm_attention.Attention(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            bias=qkv_bias,
            dense_bias=proj_bias,
            dtype=self.model_config.torch_dtype,
            layer_idx=layer_idx,
            pos_embd_params=None,
            rope_fusion=None,
            q_scaling=1.0,
            attention_chunk_size=None,
            config=self.model_config,
            max_position_embeddings=None,
        )
        if init_values:
            raise IOError("Block does not support LayerScale for now.")
        self.ls1 = nn.Identity()
        if drop_path > 0.:
            raise IOError("Block does not support DropPath for now.")
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        if scale_mlp_norm:
            raise IOError("Block does not support scale_mlp_norm for now.")
        if proj_drop > 0.:
            raise IOError("Block does not support proj_drop for now.")
        self.mlp = trtllm_mlp.MLP(
            hidden_size=dim,
            intermediate_size=int(dim * mlp_ratio),
            bias=proj_bias,
            activation=nn.GELU(),
            dtype=self.model_config.torch_dtype,
            config=self.model_config,
            layer_idx=layer_idx,
        )
        if init_values:
            raise IOError("Block does not support LayerScale for now.")
        self.ls2 = nn.Identity()
        if drop_path > 0.:
            raise IOError("Block does not support DropPath for now.")
        self.drop_path2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_metadata: Optional[attention_interface.AttentionMetadata] = None
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(
            position_ids=None,
            hidden_states=x,
            attn_metadata=attn_metadata,
            attention_mask=attention_interface.PredefinedAttentionMask.
            FULL  # Always FULL for Vision
        )
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = self.drop_path2(x)
        x = residual + x
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token',
                             'map'] = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = 'learn',
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
        fix_init: bool = False,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        cpe_max_size: Optional[int] = None,
        special_args: Optional[dict] = None,
        model_config: model_config_lib.ModelConfig[RADIOConfig] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax',
                                      'max') if fc_norm is None else fc_norm

        if norm_layer is not None or act_layer is not None:
            raise ValueError(
                f"We assume to use default norm_layer and act_layer, while getting {norm_layer=} {act_layer=}."
            )
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.model_config = model_config
        if self.model_config is not None:
            self.config = model_config.pretrained_config
            self.config.num_key_value_heads = num_heads

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix

        self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_attn_norm=scale_attn_norm,
                scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                model_config=model_config,
                layer_idx=i,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(
            embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        self.attn_pool = None
        self.fc_norm = norm_layer(
            embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        uq_teachers = set(t['name'] for t in special_args.teachers)
        patch_size = patch_size
        embed_dim = self.embed_dim
        input_dims = img_size
        normalize_patches = False

        max_img_size = int(
            round(special_args.cpe_max_size / patch_size) * patch_size)
        pos_dropout = 0.1
        num_cls_tokens = len(
            uq_teachers) if special_args.cls_token_per_teacher else 1
        register_multiple = getattr(special_args, 'register_multiple', None)
        num_registers = getattr(special_args, 'cpe_num_registers', None)

        self.patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            input_dims=input_dims,
            normalize_patches=normalize_patches,
            cls_token=class_token,
            max_input_dims=max_img_size,
            pos_dropout=pos_dropout,
            num_cls_tokens=num_cls_tokens,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )
        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.pos_drop = None
        self.patch_size = patch_size
        self.num_cls_tokens = num_cls_tokens
        self.num_registers = self.patch_generator.num_registers
        if self.model_config is not None:
            self.metadata_cls = attention_utils.get_attention_backend(
                model_config.attn_backend).Metadata
        else:
            self.metadata_cls = None

    def prepare_attn_metadata(self, batch_size: int, seq_lengths: List[int]):
        """
        To simplify the usage of the model, this function aims to fill the metadata for Attention
        Call this function before forward pass
        """
        request_ids = list(range(1, batch_size + 1))
        attn_metadata = self.metadata_cls(
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_generator(x)
        batch_size, seq_len, hidden_size = x.shape
        if self.model_config is not None:
            seq_lengths = [seq_len] * batch_size
            attn_metadata = self.prepare_attn_metadata(batch_size, seq_lengths)
            x = x.reshape(
                batch_size * seq_len,
                hidden_size)  # Need flatten batch/seq_len for trtllm attention.
        else:
            attn_metadata = None
        for block in self.blocks:
            x = block(x, attn_metadata=attn_metadata)
        x = x.reshape(batch_size, seq_len, hidden_size)
        x = self.norm(x)
        return x


class RADIOVisionModelBase(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        input_conditioner: nn.Module,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        window_size: int = None,
        adaptors: Optional[Dict[str, nn.Module]] = None,
        feature_normalizer: Optional[nn.Module] = None,
        inter_feature_normalizer: Optional[nn.Module] = None,
        model_config: model_config_lib.ModelConfig[RADIOConfig] = None,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner

        self._preferred_resolution = preferred_resolution
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._window_size = window_size

        adaptors = adaptors or dict()
        self.adaptors = nn.ModuleDict(adaptors)

        if feature_normalizer is None:
            feature_normalizer = nn.Identity()
        self.feature_normalizer = feature_normalizer
        self.inter_feature_normalizer = inter_feature_normalizer
        self.model_config = model_config

    @property
    def num_summary_tokens(self) -> int:
        if hasattr(self.model, 'num_summary_tokens'):
            return self.model.num_summary_tokens

        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_skip
        elif getattr(self.model, 'global_pool', None) == 'avg':
            return 0
        return 1

    @property
    def num_cls_tokens(self) -> int:
        if hasattr(self.model, 'num_cls_tokens'):
            return self.model.num_cls_tokens

        patch_gen = getattr(self.model, 'patch_generator', None)
        if patch_gen is not None:
            return patch_gen.num_cls_tokens
        elif getattr(self.model, 'global_pool', None) == 'avg':
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        if self._patch_size is not None:
            return self._patch_size
        if hasattr(self.model, "patch_size"):
            return self.model.patch_size
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.patch_size
        return None

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_resolution_step(self) -> int:
        res = self.patch_size
        if self.window_size is not None:
            res *= self.window_size
        return res

    @property
    def blocks(self) -> Iterable[nn.Module]:
        blocks = getattr(self.model, 'blocks', None)
        if blocks is not None:
            return blocks
        return None

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def get_nearest_supported_resolution(self, height: int,
                                         width: int) -> Resolution:
        height = int(
            round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(
            round(width / self.min_resolution_step) * self.min_resolution_step)

        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)

        return Resolution(height=height, width=width)

    def forward(
        self,
        x: torch.Tensor,
        feature_fmt: str = 'NLC'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        '''
        res_step = self.min_resolution_step
        if res_step is not None and (x.shape[-2] % res_step != 0
                                     or x.shape[-1] % res_step != 0):
            raise ValueError(
                'The input resolution must be a multiple of `self.min_resolution_step`. '
                '`self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. '
                f'Input: {x.shape[-2:]}, Nearest: {self.get_nearest_supported_resolution(*x.shape[-2:])}'
            )
        x = self.input_conditioner(x)
        y = self.model.forward_features(x)
        ret = self._extract_final(x, y, feature_fmt=feature_fmt)
        return ret

    @torch.compile
    def _extract_final(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       feature_fmt: str = 'NLC'):
        if isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                all_feat = y[:, patch_gen.num_skip:]
            elif self.model.global_pool == "avg":
                all_feat = y
            else:
                all_feat = y[:, 1:]
        else:
            raise ValueError(f'Unsupported model type: {type(self.model)}')

        all_feat = self.feature_normalizer(all_feat)

        if feature_fmt == 'NCHW':
            fmt_feat = (all_feat.reshape(all_feat.shape[0],
                                         x.shape[-2] // self.patch_size,
                                         x.shape[-1] // self.patch_size,
                                         all_feat.shape[2]).permute(0, 3, 1, 2))
        elif feature_fmt == 'NLC':
            fmt_feat = all_feat
        else:
            raise ValueError(
                f'Unsupported feature_fmt: {feature_fmt}. Must be one of ["NLC", "NCHW"]'
            )

        ret = fmt_feat

        if self.adaptors:
            raise ValueError("Adaptors are not supported for RADIO models.")

        return ret


class RADIOVisionModel(PreTrainedModel):

    def __init__(self, model_config: model_config_lib.ModelConfig):
        config = model_config.pretrained_config
        super().__init__(config)
        self.model_config = model_config
        self.config = config

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)

        # Build the model.
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        elif args.input_size is not None:
            in_chans = args.input_size[0]
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            in_chans=in_chans,
            drop_rate=args.drop,
            weight_init=args.model_kwargs.pop("weight_init", "skip"),
            special_args=args,
            model_config=model_config,
        )
        if hasattr(model, 'norm') and not getattr(args, 'model_norm', False):
            model.norm = nn.Identity()
        model.head = nn.Identity()
        model.to(dtype=config.torch_dtype)

        input_conditioner = nn.Identity()
        input_conditioner.dtype = config.torch_dtype

        adaptor_names = config.adaptor_names or []
        if len(adaptor_names) > 0:
            raise ValueError(
                "Adaptor names are not supported for RADIO models.")
        adaptors = dict()

        feature_normalizer = None
        if config.feature_normalizer_config is not None:
            raise ValueError(
                "Feature normalizer is not supported for RADIO models.")

        inter_feature_normalizer = None
        if config.inter_feature_normalizer_config is not None:
            raise ValueError(
                "Intermediate feature normalizer is not supported for RADIO models."
            )

        self.radio_model = RADIOVisionModelBase(
            model,
            input_conditioner,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
            feature_normalizer=feature_normalizer,
            inter_feature_normalizer=inter_feature_normalizer,
            model_config=model_config,
        )

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)
