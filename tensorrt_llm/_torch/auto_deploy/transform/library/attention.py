"""Pattern matching for detecting repeat_kv, eager, grouped attention patterns from Huggingface models."""

from typing import Any, Callable, Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from pydantic import Field
from torch.fx import GraphModule

from ...custom_ops.attention_interface import AttentionDescriptor
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _apply_pattern(
    gm: GraphModule,
    pattern_name: str,
    register_fn: Callable[[ADPatternMatcherPass], None],
) -> int:
    """Utility to register and apply a pattern."""
    patterns = ADPatternMatcherPass()
    register_fn(patterns)
    num_matches = patterns.apply(gm.graph)
    return num_matches


def _repeat_kv_pattern(hidden_states, n_rep) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = torch.unsqueeze(hidden_states, 2)
    hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _repeat_kv_repl(hidden_states, n_rep) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_attention_repeat_kv(hidden_states, n_rep)


# with causal_mask, no division
def _sfdp_pattern_1(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_1(query, key, value, attention_mask, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True,
        scale=scaling,
    )


# no causal_mask, no division
def _sfdp_pattern_2(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_2(query, key, value, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# with causal_mask, with division
def _sfdp_pattern_3(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_3(query, key, value, attention_mask, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True,
        scale=scaling,
    )


# no causal_mask, with division
def _sfdp_pattern_4(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_4(query, key, value, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# no causal_mask, with division, explicit casting model
def _sfdp_pattern_5(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights.to(torch.float32)
    attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_5(query, key, value, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# with causal_mask, with division, explicit casting model
def _sfdp_pattern_6(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = attn_weights.to(torch.float32)
    attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_6(query, key, value, attention_mask, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True,
        scale=scaling,
    )


# Only pass in causal attention mask in downstream standardized pipeline
def _sfdp_pattern_7(query, key, value, attention_mask, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


def _sfdp_replacement_7(query, key, value, attention_mask, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True if attention_mask is not None else False,
        scale=scaling,
    )


# with causal_mask, no division, does not cast to fp32 for softmax
def _sfdp_pattern_8(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_8(query, key, value, attention_mask, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True,
        scale=scaling,
    )


def _get_sfdp_patterns() -> List[Dict[str, Any]]:
    bs, seq_len, n_heads, hidden_size = 8, 16, 8, 512
    head_dim = hidden_size // n_heads

    def common_tensor():
        return torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

    def causal_mask():
        return torch.randn(bs, 1, 1, seq_len, device="cuda", dtype=torch.bfloat16)

    configs = [
        (_sfdp_pattern_1, _sfdp_replacement_1, True, 0.1234743, 0.85849734),
        (_sfdp_pattern_2, _sfdp_replacement_2, False, 0.234743, 0.5849734),
        (_sfdp_pattern_3, _sfdp_replacement_3, True, 0.34743, 0.849734),
        (_sfdp_pattern_4, _sfdp_replacement_4, False, 0.74321, 0.9734),
        (_sfdp_pattern_5, _sfdp_replacement_5, False, 0.874321, 0.89734),
        (_sfdp_pattern_6, _sfdp_replacement_6, True, 0.634743, 0.6849734),
        (_sfdp_pattern_7, _sfdp_replacement_7, True, 0.34743, 0.849734),
        (_sfdp_pattern_8, _sfdp_replacement_8, True, 0.2234743, 0.95849734),
    ]

    patterns = []
    for search_fn, replace_fn, has_mask, scale, dropout in configs:
        dummy_args = [common_tensor(), common_tensor(), common_tensor()]
        if has_mask:
            dummy_args.append(causal_mask())
        dummy_args.extend([scale, dropout])

        patterns.append(
            {
                "search_fn": search_fn,
                "replace_fn": replace_fn,
                "dummy_args": dummy_args,
                "scalar_workaround": {"scaling": scale, "dropout": dropout},
                "op_ignore_types": {torch.ops.aten.to.dtype: (torch.dtype,)},
            }
        )

    return patterns


def _grouped_attn_pattern_1(q, k, v, n_rep, attn_mask, dropout_p, scale):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_replacement_1(q, k, v, n_rep, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


# Only expose torch_attention after the transformation
def _grouped_attn_pattern_2(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_replacement_2(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_pattern_3(q, k, v, n_rep, attn_mask, dropout_p, scale):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_replacement_3(q, k, v, n_rep, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


# Only expose torch_attention after the transformation
def _grouped_attn_pattern_4(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_replacement_4(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_pattern_5(q, k, v, n_rep, attn_mask):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(q, k, v, attn_mask)


def _grouped_attn_replacement_5(q, k, v, n_rep, attn_mask):
    return torch.ops.auto_deploy.torch_attention.default(q, k, v, attn_mask)


def _grouped_attn_pattern_6(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )


def _grouped_attn_replacement_6(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_pattern_7(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )


def _grouped_attn_replacement_7(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_pattern_8(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )


def _grouped_attn_replacement_8(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_pattern_9(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )


def _grouped_attn_replacement_9(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_pattern_10(q, k, v, n_rep, dropout_p):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
    )


def _grouped_attn_replacement_10(q, k, v, n_rep, dropout_p):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
    )


@TransformRegistry.register("match_repeat_kv")
class MatchRepeatKV(BaseTransform):
    """
    Match and replace the repeat_kv pattern with torch.ops.auto_deploy.torch_attention_repeat_kv.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        def register_repeat_kv(patterns: ADPatternMatcherPass):
            dummy_args = [
                torch.randn(8, 8, 16, 64, device="cuda", dtype=torch.float16),
                7,
            ]
            register_ad_pattern(
                search_fn=_repeat_kv_pattern,
                replace_fn=_repeat_kv_repl,
                patterns=patterns,
                dummy_args=dummy_args,
                op_ignore_types={
                    torch.ops.aten.reshape.default: (int,),
                    torch.ops.aten.expand.default: (int,),
                },
                scalar_workaround={"n_rep": dummy_args[1]},
            )

        num_kv_patterns = _apply_pattern(gm, "Repeat KV", register_repeat_kv)

        if num_kv_patterns > 0:
            self.config.run_shape_prop = True

        info = TransformInfo(
            skipped=False,
            num_matches=num_kv_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


@TransformRegistry.register("match_eager_attention")
class MatchEagerAttention(BaseTransform):
    """
    Match and replace the eager attention pattern with torch.ops.auto_deploy.torch_attention_sdpa.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        def register_eager_attention(patterns: ADPatternMatcherPass):
            for pattern_config in _get_sfdp_patterns():
                register_ad_pattern(**pattern_config, patterns=patterns)

        num_eager_patterns = _apply_pattern(gm, "Eager Attention", register_eager_attention)

        info = TransformInfo(
            skipped=False,
            num_matches=num_eager_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


@TransformRegistry.register("match_grouped_attention")
class MatchGroupedAttention(BaseTransform):
    """
    Match and replace the grouped attention pattern with
    torch.ops.auto_deploy.torch_attention.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        def register_grouped_attention(patterns: ADPatternMatcherPass):
            q = torch.randn(8, 8, 16, 64, device="cuda", dtype=torch.float16)
            k1 = torch.randn(8, 1, 16, 64, device="cuda", dtype=torch.float16)
            v1 = torch.randn(8, 1, 16, 64, device="cuda", dtype=torch.float16)
            attn_mask = torch.randn(8, 1, 1, 16, device="cuda", dtype=torch.float16)
            dropout = 0.12345
            scale = 0.56789
            n_rep = 7

            dummy_args_1 = [q, k1, v1, n_rep, attn_mask, dropout, scale]
            dummy_args_2 = [q, k1, v1, attn_mask, dropout, scale]
            dummy_args_3 = [q, k1, v1, n_rep, attn_mask]
            dummy_args_4 = [q, k1, v1, dropout, scale]
            dummy_args_5 = [q, k1, v1, n_rep, dropout]

            register_ad_pattern(
                search_fn=_grouped_attn_pattern_1,
                replace_fn=_grouped_attn_replacement_1,
                patterns=patterns,
                dummy_args=dummy_args_1,
                scalar_workaround={"scale": scale, "dropout_p": dropout, "n_rep": n_rep},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_2,
                replace_fn=_grouped_attn_replacement_2,
                patterns=patterns,
                dummy_args=dummy_args_2,
                scalar_workaround={
                    "scale": scale,
                    "dropout_p": dropout,
                },
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_3,
                replace_fn=_grouped_attn_replacement_3,
                patterns=patterns,
                dummy_args=dummy_args_1,
                scalar_workaround={"scale": scale, "dropout_p": dropout, "n_rep": n_rep},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_4,
                replace_fn=_grouped_attn_replacement_4,
                patterns=patterns,
                dummy_args=dummy_args_2,
                scalar_workaround={
                    "scale": scale,
                    "dropout_p": dropout,
                },
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_5,
                replace_fn=_grouped_attn_replacement_5,
                patterns=patterns,
                dummy_args=dummy_args_3,
                scalar_workaround={"n_rep": n_rep},
            )

            register_ad_pattern(
                search_fn=_grouped_attn_pattern_6,
                replace_fn=_grouped_attn_replacement_6,
                patterns=patterns,
                dummy_args=dummy_args_2,
                scalar_workaround={"scale": scale, "dropout_p": dropout},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_7,
                replace_fn=_grouped_attn_replacement_7,
                patterns=patterns,
                dummy_args=dummy_args_2,
                scalar_workaround={"scale": scale, "dropout_p": dropout},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_8,
                replace_fn=_grouped_attn_replacement_8,
                patterns=patterns,
                dummy_args=dummy_args_4,
                scalar_workaround={"scale": scale, "dropout_p": dropout},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_9,
                replace_fn=_grouped_attn_replacement_9,
                patterns=patterns,
                dummy_args=dummy_args_4,
                scalar_workaround={"scale": scale, "dropout_p": dropout},
            )
            register_ad_pattern(
                search_fn=_grouped_attn_pattern_10,
                replace_fn=_grouped_attn_replacement_10,
                patterns=patterns,
                dummy_args=dummy_args_5,
                scalar_workaround={"dropout_p": dropout, "n_rep": n_rep},
            )

        num_grouped_patterns = _apply_pattern(gm, "Grouped Attention", register_grouped_attention)
        if num_grouped_patterns == 0:
            ad_logger.warning(
                "Fail to find any Group Attention Pattern, output or performance may be incorrect"
            )

        info = TransformInfo(
            skipped=False,
            num_matches=num_grouped_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


def _attn_bnsd_pattern_1(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        layout="bnsd",
    )


def _attn_bnsd_pattern_2(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        layout="bnsd",
    )


def _attn_bnsd_pattern_3(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale,
        layout="bnsd",
    )


def _attn_bnsd_pattern_4(q, k, v, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale,
        layout="bnsd",
    )


def _attn_bnsd_pattern_5(q, k, v, attn_mask, dropout_p):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=None,
        layout="bnsd",
    )


def _attn_bnsd_pattern_6(q, k, v, dropout_p):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=None,
        layout="bnsd",
    )


def _attn_bnsd_pattern_7(q, k, v, attn_mask, dropout_p):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=True,
        scale=None,
        layout="bnsd",
    )


def _attn_bnsd_pattern_8(q, k, v, dropout_p):
    return torch.ops.auto_deploy.torch_attention.default(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
        scale=None,
        layout="bnsd",
    )


def _attn_bnsd_to_bnsd_via_bsnd(q, k, v, *, attn_mask, dropout_p, is_causal, scale):
    q_bsnd = torch.ops.aten.transpose.int(q, 1, 2)
    k_bsnd = torch.ops.aten.transpose.int(k, 1, 2)
    v_bsnd = torch.ops.aten.transpose.int(v, 1, 2)

    out_bsnd = torch.ops.auto_deploy.torch_attention.default(
        q_bsnd,
        k_bsnd,
        v_bsnd,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        layout="bsnd",
    )
    return torch.ops.aten.transpose.int(out_bsnd, 1, 2)


# 1) is_causal=False, mask present, scale present
def _attn_bnsd_replacement_1(q, k, v, attn_mask, dropout_p, scale):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


# 2) is_causal=False, mask None, scale present
def _attn_bnsd_replacement_2(q, k, v, dropout_p, scale):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=scale
    )


# 3) is_causal=True, mask present, scale present
def _attn_bnsd_replacement_3(q, k, v, attn_mask, dropout_p, scale):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


# 4) is_causal=True, mask None, scale present
def _attn_bnsd_replacement_4(q, k, v, dropout_p, scale):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True, scale=scale
    )


# 5) is_causal=False, mask present, scale=None
def _attn_bnsd_replacement_5(q, k, v, attn_mask, dropout_p):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=None
    )


# 6) is_causal=False, mask None, scale=None
def _attn_bnsd_replacement_6(q, k, v, dropout_p):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=None
    )


# 7) is_causal=True, mask present, scale=None
def _attn_bnsd_replacement_7(q, k, v, attn_mask, dropout_p):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=None
    )


# 8) is_causal=True, mask None, scale=None
def _attn_bnsd_replacement_8(q, k, v, dropout_p):
    return _attn_bnsd_to_bnsd_via_bsnd(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True, scale=None
    )


def register_match_attn_layout(patterns: ADPatternMatcherPass):
    # Dummy tensors in BNSD (we match bnsd calls)
    bs, n_heads, s_q, head_dim = 8, 8, 16, 64
    q = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    attn_mask = torch.randn(bs, n_heads, 1, s_q, device="cuda", dtype=torch.float16)

    dropout_p = 0.12345
    scale_val = 0.56789

    # 1..4 (scale present)
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_1,
        replace_fn=_attn_bnsd_replacement_1,
        patterns=patterns,
        dummy_args=[q, k, v, attn_mask, dropout_p, scale_val],
        scalar_workaround={"dropout_p": dropout_p, "scale": scale_val},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_2,
        replace_fn=_attn_bnsd_replacement_2,
        patterns=patterns,
        dummy_args=[q, k, v, dropout_p, scale_val],
        scalar_workaround={"dropout_p": dropout_p, "scale": scale_val},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_3,
        replace_fn=_attn_bnsd_replacement_3,
        patterns=patterns,
        dummy_args=[q, k, v, attn_mask, dropout_p, scale_val],
        scalar_workaround={"dropout_p": dropout_p, "scale": scale_val},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_4,
        replace_fn=_attn_bnsd_replacement_4,
        patterns=patterns,
        dummy_args=[q, k, v, dropout_p, scale_val],
        scalar_workaround={"dropout_p": dropout_p, "scale": scale_val},
    )

    # 5..8 (scale None)
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_5,
        replace_fn=_attn_bnsd_replacement_5,
        patterns=patterns,
        dummy_args=[q, k, v, attn_mask, dropout_p],
        scalar_workaround={"dropout_p": dropout_p},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_6,
        replace_fn=_attn_bnsd_replacement_6,
        patterns=patterns,
        dummy_args=[q, k, v, dropout_p],
        scalar_workaround={"dropout_p": dropout_p},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_7,
        replace_fn=_attn_bnsd_replacement_7,
        patterns=patterns,
        dummy_args=[q, k, v, attn_mask, dropout_p],
        scalar_workaround={"dropout_p": dropout_p},
    )
    register_ad_pattern(
        search_fn=_attn_bnsd_pattern_8,
        replace_fn=_attn_bnsd_replacement_8,
        patterns=patterns,
        dummy_args=[q, k, v, dropout_p],
        scalar_workaround={"dropout_p": dropout_p},
    )


class MatchAttentionLayoutConfig(TransformConfig):
    """Configuration for the insert cached attention transform."""

    attention_op: Type[AttentionDescriptor] = Field(description="The attention descriptor to use.")


@TransformRegistry.register("match_attention_layout")
class MatchAttentionLayout(BaseTransform):
    """
    Convert unified torch_attention calls from layout='bnsd' (explicit, positional or default)
    into layout='bsnd' + correct Q/K/V transposes, and transpose the output back to bnsd.
    """

    config: MatchAttentionLayoutConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MatchAttentionLayoutConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        attention_layout = self.config.attention_op.get_attention_layout()

        if attention_layout not in ("bnsd", "bsnd"):
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # If backend expects bnsd, nothing to do.
        if attention_layout == "bnsd":
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=False, has_valid_shapes=False
            )

        num_matches = _apply_pattern(
            gm, "MatchAttentionLayout(bnsd→bsnd)", register_match_attn_layout
        )

        # If we changed any attention calls, the shapes may change around the transposes; flag for shape prop.
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info
