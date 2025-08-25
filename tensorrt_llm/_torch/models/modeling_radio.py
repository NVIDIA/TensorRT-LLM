import math
from collections import namedtuple
from typing import (Dict, Iterable, List, NamedTuple, Optional, Tuple, Type,
                    Union)

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import (Attention, LayerNorm, LayerType, Mlp, get_act_layer,
                         get_norm_layer)
from torch import nn

norm_t = Union[Tuple[float, float, float], torch.Tensor]
input_dim_t = Union[int, Tuple[int, int]]

DEFAULT_VERSION = "radio_v2.5-h"


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)


class Resolution(NamedTuple):
    height: int
    width: int


class InputConditioner(nn.Module):

    def __init__(
        self,
        input_scale: float,
        norm_mean: norm_t,
        norm_std: norm_t,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


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

    def no_weight_decay(self):
        return [
            'token',
        ]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def apply_cls_token(self):
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_cls_patches(self):
        return self.cls_token.num_patches

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self):
        return [
            'pos_embed',
        ]

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))

            assert src_size**2 == src_embed.shape[
                1], 'Unable to interpolate non-square embedding'

            src_embed = rearrange(src_embed,
                                  'b (h w) c -> b c h w',
                                  h=src_size,
                                  w=src_size)
            src_embed = F.interpolate(src_embed,
                                      size=(self.num_rows, self.num_cols),
                                      mode='bicubic',
                                      align_corners=True,
                                      antialias=False)
            src_embed = rearrange(src_embed, 'b c h w -> b (h w) c')
        targ_embed.data.copy_(src_embed)

    def _load_projection(self, src_proj_weight: torch.Tensor,
                         targ_proj_weight: torch.Tensor):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1], 'Unable to interpolate non-square patch size'

            src_proj_weight = rearrange(src_proj_weight,
                                        'b (c h w) -> b c h w',
                                        c=3,
                                        h=src_patch_size,
                                        w=src_patch_size)
            src_proj_weight = F.interpolate(src_proj_weight,
                                            size=(self.patch_size,
                                                  self.patch_size),
                                            mode='bicubic',
                                            align_corners=True,
                                            antialias=False)
            src_proj_weight = rearrange(src_proj_weight, 'b c h w -> b (c h w)')
        targ_proj_weight.data.copy_(src_proj_weight)

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

        if self.training and self.pos_dropout > 0:
            keeps = torch.rand(patches.shape[0],
                               1,
                               1,
                               dtype=pos_enc.dtype,
                               device=pos_enc.device) > self.pos_dropout
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

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
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = torch.rand(batch_size, 1, 1, device=pos_embed.device
                                   ) * (1 - min_scale) + min_scale
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device) *
                    (aspect_max - aspect_min) + aspect_min)

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = torch.rand(
                    batch_size, 1, 1, 2,
                    device=pos_embed.device) * (1 - scale_xy)

                lin_x = torch.linspace(
                    0, 1, steps=input_dims[1],
                    device=pos_embed.device)[None, None].expand(
                        batch_size, input_dims[0], -1)
                lin_y = torch.linspace(
                    0, 1, steps=input_dims[0],
                    device=pos_embed.device)[None, :, None].expand(
                        batch_size, -1, input_dims[1])

                lin_xy = torch.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                # i_rows, i_cols = input_dims
                # p_rows, p_cols = pos_embed.shape[2:]
                # if i_rows <= p_rows and i_cols <= p_cols:
                #     left = (p_cols - i_cols) // 2
                #     top = (p_rows - i_rows) // 2
                #     pos_embed = pos_embed[..., top:top+i_rows, left:left+i_cols]
                # else:
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
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
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
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        if init_values:
            raise IOError("Block does not support LayerScale for now.")
        self.ls1 = nn.Identity()
        if drop_path > 0.:
            raise IOError("Block does not support DropPath for now.")
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        if init_values:
            raise IOError("Block does not support LayerScale for now.")
        self.ls2 = nn.Identity()
        if drop_path > 0.:
            raise IOError("Block does not support DropPath for now.")
        self.drop_path2 = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

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
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        cpe_max_size: Optional[int] = None,
        special_args: Optional[dict] = None,
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
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        act_layer = get_act_layer(act_layer) or nn.GELU

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
        self.blocks = nn.Sequential(*[
            block_fn(
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
                mlp_layer=mlp_layer,
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

    def forward_features(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_generator(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


def create_model_from_args(args) -> nn.Module:
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
    )

    if hasattr(model, 'norm') and not getattr(args, 'model_norm', False):
        model.norm = nn.Identity()

    model.head = nn.Identity()

    return model


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
        super().__init__(**kwargs)


class RadioOutput(NamedTuple):
    summary: torch.Tensor
    features: torch.Tensor

    def to(self, *args, **kwargs):
        return RadioOutput(
            self.summary.to(*args, **kwargs)
            if self.summary is not None else None,
            self.features.to(*args, **kwargs)
            if self.features is not None else None,
        )


class RADIOModelBase(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        input_conditioner: InputConditioner,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: Optional[torch.Tensor] = None,
        window_size: int = None,
        adaptors: Optional[Dict[str, nn.Module]] = None,
        feature_normalizer: Optional[nn.Module] = None,
        inter_feature_normalizer: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        if summary_idxs is not None:
            self.register_buffer('summary_idxs', summary_idxs)
        else:
            self.summary_idxs = None

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

    def _extract_final(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       feature_fmt: str = 'NLC'):
        if isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                all_summary = y[:, :patch_gen.num_cls_tokens]
                if self.summary_idxs is not None:
                    bb_summary = all_summary[:, self.summary_idxs]
                else:
                    bb_summary = all_summary
                all_feat = y[:, patch_gen.num_skip:]
            elif self.model.global_pool == "avg":
                all_summary = y[:, self.model.num_prefix_tokens:].mean(dim=1)
                bb_summary = all_summary
                all_feat = y
            else:
                all_summary = y[:, 0]
                bb_summary = all_summary
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

        ret = RadioOutput(bb_summary.flatten(1), fmt_feat)

        if self.adaptors:
            raise ValueError("Adaptors are not supported for RADIO models.")

        return ret


class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig

    def __init__(self, config: RADIOConfig):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config

        model = create_model_from_args(args)
        input_conditioner = InputConditioner(
            input_scale=1.0,
            norm_mean=OPENAI_CLIP_MEAN,
            norm_std=OPENAI_CLIP_STD,
        )

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        summary_idxs = torch.tensor(
            [
                i for i, t in enumerate(args.teachers)
                if t.get("use_summary", True)
            ],
            dtype=torch.int64,
        )

        config.adaptor_configs
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

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=summary_idxs,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
            feature_normalizer=feature_normalizer,
            inter_feature_normalizer=inter_feature_normalizer,
        )

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)
