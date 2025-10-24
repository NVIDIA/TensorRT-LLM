"""Pattern matching for detecting repeat_kv, eager, grouped attention patterns from Huggingface models."""

from inspect import Parameter, Signature
from itertools import product
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

import torch
import torch.nn.functional as F
from pydantic import Field
from torch.fx import GraphModule

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


def _attach_signature(fn: Callable, argnames: List[str]) -> Callable:
    # Make FX "see" q,k,v[,attn_mask][,dropout_p][,scale] even though fn(*args) internally
    params = [Parameter(n, kind=Parameter.POSITIONAL_OR_KEYWORD) for n in argnames]
    fn.__signature__ = Signature(parameters=params)
    return fn


def _call_sdpa(
    q, k, v, *, is_causal: bool, enable_gqa: bool, attn_mask=None, dropout_p=None, scale=None
):
    kwargs = {"is_causal": is_causal}
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    if dropout_p is not None:
        kwargs["dropout_p"] = dropout_p
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa:
        kwargs["enable_gqa"] = True
    return torch.ops.auto_deploy.torch_attention_sdpa.default(q, k, v, **kwargs)


def _call_attn(q, k, v, *, is_causal: bool, attn_mask=None, dropout_p=None, scale=None):
    kwargs = {"is_causal": is_causal}
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    if dropout_p is not None:
        kwargs["dropout_p"] = dropout_p
    if scale is not None:
        kwargs["scale"] = scale
    return torch.ops.auto_deploy.torch_attention.default(q, k, v, **kwargs)


def make_grouped_attn_pair(
    *,
    repeat_kv: bool,
    is_causal: bool,
    has_scale: bool,
    enable_gqa: bool,
    has_attn_mask: bool,
    has_dropout: bool,
) -> Tuple[Callable, Callable, List[str]]:
    """
    Returns (pattern_fn, replacement_fn, argnames) with exact positional parity.

    Arg order rules:
      Base: (q, k, v)
      +repeat_kv -> insert n_rep after (q, k, v)
      +attn_mask -> include attn_mask after n_rep if repeat_kv else after (q, k, v)
      +dropout   -> include dropout_p after attn_mask or after n_rep/base if no attn_mask
      +scale     -> include scale last
    """
    argnames: List[str] = ["q", "k", "v"]
    if repeat_kv:
        argnames.append("n_rep")
    if has_attn_mask:
        argnames.append("attn_mask")
    if has_dropout:
        argnames.append("dropout_p")
    if has_scale:
        argnames.append("scale")

    def pattern_fn(*args):
        if len(args) != len(argnames):
            raise TypeError(f"Expected {len(argnames)} args {tuple(argnames)}, got {len(args)}")
        m = dict(zip(argnames, args))

        q = m["q"]
        k = m["k"]
        v = m["v"]

        if repeat_kv:
            n_rep = m["n_rep"]
            k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
            v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)

        return _call_sdpa(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
            attn_mask=m.get("attn_mask"),
            dropout_p=m.get("dropout_p"),
            scale=m.get("scale"),
        )

    # Replacement: torch_attention.default mirroring the positional signature exactly.
    # We do NOT pass enable_gqa here (it’s SDPA-only). We accept n_rep to mirror signature,
    # but we don’t need to use it in the replacement graph.
    def replacement_fn(*args):
        if len(args) != len(argnames):
            raise TypeError(f"Expected {len(argnames)} args {tuple(argnames)}, got {len(args)}")
        m = dict(zip(argnames, args))
        return _call_attn(
            m["q"],
            m["k"],
            m["v"],
            is_causal=is_causal,
            attn_mask=m.get("attn_mask"),
            dropout_p=m.get("dropout_p"),
            scale=m.get("scale"),
        )

    # Pattern matcher needs to see explicit arg names
    _attach_signature(pattern_fn, argnames)
    _attach_signature(replacement_fn, argnames)

    return pattern_fn, replacement_fn, argnames


def generate_and_register_grouped_attn_patterns(
    patterns, register_ad_pattern: Callable, only_repeat_kv: bool = None
):
    """
    Auto-generate all grouped attention patterns across these axes:
      1) repeat_kv:        [False, True]
      2) is_causal:        [False, True]
      3) has_scale:        [False, True]
      4) enable_gqa:       [False, True]   (only a kwarg to SDPA side)
      5) has_attn_mask:    [False, True]
      6) has_dropout:      [False, True]

    Args:
        patterns: The ADPatternMatcherPass instance to register patterns to
        register_ad_pattern: The function to call to register each pattern
        only_repeat_kv: If True, only register patterns with repeat_kv=True.
                        If False, only register patterns with repeat_kv=False.
                        If None, register all patterns.

    For each valid combo, we:
      - build pattern/replacement functions with exact-arg parity
      - build dummy args matching the signature (with CUDA fp16 tensors etc.)
      - build scalar_workaround dict for any scalars/n_rep present
      - call register_ad_pattern(...)
    """
    q = torch.randn(8, 8, 16, 64, device="cuda", dtype=torch.float16)
    k1 = torch.randn(8, 1, 16, 64, device="cuda", dtype=torch.float16)
    v1 = torch.randn(8, 1, 16, 64, device="cuda", dtype=torch.float16)
    attn_mask_tensor = torch.randn(8, 1, 1, 16, device="cuda", dtype=torch.float16)

    dropout_val = 0.12345
    scale_val = 0.56789
    n_rep_val = 7

    total = 0
    axes = ((False, True),) * 6
    for repeat_kv, is_causal, has_scale, enable_gqa, has_attn_mask, has_dropout in product(*axes):
        if only_repeat_kv is not None:
            if only_repeat_kv and not repeat_kv:
                continue  # Skip patterns without repeat_kv
            if not only_repeat_kv and repeat_kv:
                continue  # Skip patterns with repeat_kv

        pat_fn, rep_fn, argnames = make_grouped_attn_pair(
            repeat_kv=repeat_kv,
            is_causal=is_causal,
            has_scale=has_scale,
            enable_gqa=enable_gqa,
            has_attn_mask=has_attn_mask,
            has_dropout=has_dropout,
        )

        # Build dummy args in the same positional order
        value_map = {
            "q": q,
            "k": k1,
            "v": v1,
            "n_rep": n_rep_val,
            "attn_mask": attn_mask_tensor,
            "dropout_p": dropout_val,
            "scale": scale_val,
        }
        dummy_args: List[object] = []
        for name in argnames:
            try:
                dummy_args.append(value_map[name])
            except KeyError:
                raise RuntimeError(f"Unexpected arg name: {name}")

        scalar_names = {"n_rep", "dropout_p", "scale"}
        scalar_workaround: Dict[str, object] = {
            n: value_map[n] for n in argnames if n in scalar_names
        }
        if not scalar_workaround:
            scalar_workaround = None

        register_ad_pattern(
            search_fn=pat_fn,
            replace_fn=rep_fn,
            patterns=patterns,
            dummy_args=dummy_args,
            scalar_workaround=scalar_workaround,
        )
        total += 1
    return total


@TransformRegistry.register("match_grouped_attention_with_repeat_kv")
class MatchGroupedAttentionWithRepeatKV(BaseTransform):
    """
    Match and replace grouped attention patterns WITH repeat_kv to
    torch.ops.auto_deploy.torch_attention.

    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        def register_grouped_attention_with_repeat_kv(patterns: ADPatternMatcherPass):
            return generate_and_register_grouped_attn_patterns(
                patterns, register_ad_pattern, only_repeat_kv=True
            )

        num_grouped_patterns = _apply_pattern(
            gm, "Grouped Attention (with repeat_kv)", register_grouped_attention_with_repeat_kv
        )

        info = TransformInfo(
            skipped=False,
            num_matches=num_grouped_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info


@TransformRegistry.register("match_grouped_attention_without_repeat_kv")
class MatchGroupedAttentionWithoutRepeatKV(BaseTransform):
    """
    Match and replace grouped attention patterns WITHOUT repeat_kv to
    torch.ops.auto_deploy.torch_attention.

    This transform should run AFTER match_grouped_attention_with_repeat_kv
    to avoid incorrectly matching patterns that should have repeat_kv.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        def register_grouped_attention_without_repeat_kv(patterns: ADPatternMatcherPass):
            return generate_and_register_grouped_attn_patterns(
                patterns, register_ad_pattern, only_repeat_kv=False
            )

        num_grouped_patterns = _apply_pattern(
            gm,
            "Grouped Attention (without repeat_kv)",
            register_grouped_attention_without_repeat_kv,
        )

        if num_grouped_patterns == 0:
            ad_logger.warning(
                "Fail to find any Group Attention Pattern (without repeat_kv), "
                "output or performance may be incorrect"
            )

        info = TransformInfo(
            skipped=False,
            num_matches=num_grouped_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info


def _call_torch_attention(
    q, k, v, *, is_causal, layout, attn_mask=None, dropout_p=None, scale=None
):
    kwargs = {"is_causal": is_causal, "layout": layout}
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    if dropout_p is not None:
        kwargs["dropout_p"] = dropout_p
    if scale is not None:
        kwargs["scale"] = scale
    return torch.ops.auto_deploy.torch_attention.default(q, k, v, **kwargs)


def make_attn_bnsd_pair(
    *,
    has_attn_mask: bool,
    has_dropout: bool,
    is_causal: bool,
    has_scale: bool,
) -> Tuple[Callable, Callable, List[str], str, str]:
    argnames: List[str] = ["q", "k", "v"]
    if has_attn_mask:
        argnames.append("attn_mask")
    if has_dropout:
        argnames.append("dropout_p")
    if has_scale:
        argnames.append("scale")

    def pattern_fn(*args):
        if len(args) != len(argnames):
            raise TypeError(f"Expected {len(argnames)} args {tuple(argnames)}, got {len(args)}")
        m = dict(zip(argnames, args))
        return _call_torch_attention(
            m["q"],
            m["k"],
            m["v"],
            is_causal=is_causal,
            layout="bnsd",
            attn_mask=m.get("attn_mask"),
            dropout_p=m.get("dropout_p"),
            scale=m.get("scale"),
        )

    def replacement_fn(*args):
        if len(args) != len(argnames):
            raise TypeError(f"Expected {len(argnames)} args {tuple(argnames)}, got {len(args)}")
        m = dict(zip(argnames, args))
        q_b = torch.ops.aten.transpose.int(m["q"], 1, 2)
        k_b = torch.ops.aten.transpose.int(m["k"], 1, 2)
        v_b = torch.ops.aten.transpose.int(m["v"], 1, 2)
        out_b = _call_torch_attention(
            q_b,
            k_b,
            v_b,
            is_causal=is_causal,
            layout="bsnd",
            attn_mask=m.get("attn_mask"),
            dropout_p=m.get("dropout_p"),
            scale=m.get("scale"),
        )
        return torch.ops.aten.transpose.int(out_b, 1, 2)

    # Pattern matcher needs to see explicit arg names
    _attach_signature(pattern_fn, argnames)
    _attach_signature(replacement_fn, argnames)

    return pattern_fn, replacement_fn, argnames


def generate_and_register_attn_layout_patterns(patterns, register_ad_pattern: Callable):
    """
    Enumerate all combinations across:
      - has_attn_mask in {False, True}
      - has_dropout   in {False, True}
      - is_causal     in {False, True}
      - has_scale     in {False, True}
    Register each pattern/replacement with appropriate dummy args and scalar workarounds.
    """
    # Dummy tensors in BNSD
    bs, n_heads, s_q, head_dim = 8, 8, 16, 64
    q = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, n_heads, s_q, head_dim, device="cuda", dtype=torch.float16)
    attn_mask = torch.randn(bs, n_heads, 1, s_q, device="cuda", dtype=torch.float16)

    dropout_p = 0.12345
    scale_val = 0.56789

    total = 0
    axes = ((False, True),) * 4
    for has_attn_mask, has_dropout, is_causal, has_scale in product(*axes):
        pat_fn, rep_fn, argnames = make_attn_bnsd_pair(
            has_attn_mask=has_attn_mask,
            has_dropout=has_dropout,
            is_causal=is_causal,
            has_scale=has_scale,
        )

        # Build dummy args following positional signature
        value_map = {
            "q": q,
            "k": k,
            "v": v,
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "scale": scale_val,
        }
        dummy_args: List[object] = []
        for name in argnames:
            try:
                dummy_args.append(value_map[name])
            except KeyError:
                raise RuntimeError(f"Unexpected arg name: {name}")

        # Scalar workaround for present scalars only
        scalar_names = {"dropout_p", "scale"}
        scalar_workaround: Dict[str, object] = {
            n: value_map[n] for n in argnames if n in scalar_names
        }
        if not scalar_workaround:
            scalar_workaround = None

        register_ad_pattern(
            search_fn=pat_fn,
            replace_fn=rep_fn,
            patterns=patterns,
            dummy_args=dummy_args,
            scalar_workaround=scalar_workaround,
        )
        total += 1
    return total


def register_match_attn_layout(patterns: ADPatternMatcherPass):
    return generate_and_register_attn_layout_patterns(patterns, register_ad_pattern)


class MatchAttentionLayoutConfig(TransformConfig):
    """Configuration for the match attention layout transform."""

    attn_layout: Literal["bsnd", "bnsd"] = Field(
        description="Layout expected by the attention backend."
    )


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
        # If backend expects bnsd, nothing to do.
        if self.config.attn_layout == "bnsd":
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
