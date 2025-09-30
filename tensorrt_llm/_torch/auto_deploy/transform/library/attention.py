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


def make_grouped_attn_pair(
    *,
    repeat_kv: bool,
    is_causal: bool,
    has_scale: bool,
    enable_gqa: bool,
    has_attn_mask: bool,
    has_dropout: bool,
):
    """
    Returns (pattern_fn, replacement_fn, argnames) such that:
      - pattern_fn(*args) calls torch_attention_sdpa.default with the specified knobs
        and optional pre-repeat kv
      - replacement_fn(*args) calls torch_attention.default mirroring signature exactly
      - argnames is the ordered arg list for constructing dummy args

    Arg order rules:
      - Base: (q, k, v)
      - +repeat_kv -> insert n_rep after (q, k, v)
      - +attn_mask  -> include attn_mask after n_rep if repeat_kv else after (q, k, v)
      - +dropout    -> include dropout_p after attn_mask or after n_rep / base if no attn_mask
      - +scale      -> include scale last (after dropout_p if present, else after attn_mask/n_rep/base)
    """
    # build signature
    argnames: list[str] = ["q", "k", "v"]
    if repeat_kv:
        argnames.append("n_rep")
    if has_attn_mask:
        argnames.append("attn_mask")
    if has_dropout:
        argnames.append("dropout_p")
    if has_scale:
        argnames.append("scale")

    # helper to build call kwargs source strings
    def _build_sdpa_kw_src():
        parts = []
        if has_attn_mask:
            parts.append("attn_mask=attn_mask")
        if has_dropout:
            parts.append("dropout_p=dropout_p")
        if has_scale:
            parts.append("scale=scale")
        parts.append(f"is_causal={str(is_causal)}")
        if enable_gqa:
            parts.append("enable_gqa=True")
        return ", ".join(parts)

    def _build_attn_kw_src():
        parts = []
        if has_attn_mask:
            parts.append("attn_mask=attn_mask")
        if has_dropout:
            parts.append("dropout_p=dropout_p")
        if has_scale:
            parts.append("scale=scale")
        parts.append(f"is_causal={str(is_causal)}")
        return ", ".join(parts)

    sdpa_kw_src = _build_sdpa_kw_src()
    attn_kw_src = _build_attn_kw_src()

    # factories that also return the source we exec
    def pattern_factory(argnames=tuple(argnames), repeat_kv=repeat_kv):
        args_sig = ", ".join(argnames)
        fn_name = (
            f"ga_pat_r{int(repeat_kv)}_c{int(is_causal)}_s{int(has_scale)}_"
            f"g{int(enable_gqa)}_m{int(has_attn_mask)}_d{int(has_dropout)}"
        )
        body_lines = [f"def {fn_name}({args_sig}):"]
        if repeat_kv:
            body_lines.append("    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)")
            body_lines.append("    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)")
        call_line = (
            "    return torch.ops.auto_deploy.torch_attention_sdpa.default("
            f"q, k, v{', ' if sdpa_kw_src else ''}{sdpa_kw_src})"
        )
        body_lines.append(call_line)
        src = "\n".join(body_lines)
        scope = {"torch": torch}
        exec(src, scope)
        return scope[fn_name]

    def replacement_factory(argnames=tuple(argnames)):
        args_sig = ", ".join(argnames)
        fn_name = (
            f"ga_rep_r{int(repeat_kv)}_c{int(is_causal)}_s{int(has_scale)}_"
            f"g{int(enable_gqa)}_m{int(has_attn_mask)}_d{int(has_dropout)}"
        )
        body = [f"def {fn_name}({args_sig}):"]
        call_line = (
            "    return torch.ops.auto_deploy.torch_attention.default("
            f"q, k, v{', ' if attn_kw_src else ''}{attn_kw_src})"
        )
        body.append(call_line)
        src = "\n".join(body)
        scope = {"torch": torch}
        exec(src, scope)
        return scope[fn_name]

    pat_fn = pattern_factory()
    rep_fn = replacement_factory()

    return pat_fn, rep_fn, argnames


def generate_and_register_grouped_attn_patterns(patterns, register_ad_pattern: Callable):
    """
    Auto-generate all grouped attention patterns across these axes:
      1) repeat_kv:        [False, True]
      2) is_causal:        [False, True]
      3) has_scale:        [False, True]
      4) enable_gqa:       [False, True]   (only a kwarg to SDPA side)
      5) has_attn_mask:    [False, True]
      6) has_dropout:      [False, True]

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
    for repeat_kv in (False, True):
        for is_causal in (False, True):
            for has_scale in (False, True):
                for enable_gqa in (False, True):
                    for has_attn_mask in (False, True):
                        for has_dropout in (False, True):
                            # Build functions
                            pat_fn, rep_fn, argnames = make_grouped_attn_pair(
                                repeat_kv=repeat_kv,
                                is_causal=is_causal,
                                has_scale=has_scale,
                                enable_gqa=enable_gqa,
                                has_attn_mask=has_attn_mask,
                                has_dropout=has_dropout,
                            )

                            # Build dummy args in the same positional order
                            dummy_args: List[object] = []
                            for name in argnames:
                                if name == "q":
                                    dummy_args.append(q)
                                elif name == "k":
                                    dummy_args.append(k1)
                                elif name == "v":
                                    dummy_args.append(v1)
                                elif name == "n_rep":
                                    dummy_args.append(n_rep_val)
                                elif name == "attn_mask":
                                    dummy_args.append(attn_mask_tensor)
                                elif name == "dropout_p":
                                    dummy_args.append(dropout_val)
                                elif name == "scale":
                                    dummy_args.append(scale_val)
                                else:
                                    raise RuntimeError(f"Unexpected arg name: {name}")

                            # scalar_workaround mirrors only the scalar args present by name
                            scalar_workaround: Dict[str, object] = {}
                            if "n_rep" in argnames:
                                scalar_workaround["n_rep"] = n_rep_val
                            if "dropout_p" in argnames:
                                scalar_workaround["dropout_p"] = dropout_val
                            if "scale" in argnames:
                                scalar_workaround["scale"] = scale_val

                            # Register
                            register_ad_pattern(
                                search_fn=pat_fn,
                                replace_fn=rep_fn,
                                patterns=patterns,
                                dummy_args=dummy_args,
                                scalar_workaround=scalar_workaround if scalar_workaround else None,
                            )
                            total += 1
    return total


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
            return generate_and_register_grouped_attn_patterns(patterns, register_ad_pattern)

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


def make_attn_bnsd_pair(
    *,
    has_attn_mask: bool,
    has_dropout: bool,
    is_causal: bool,
    has_scale: bool,
) -> Tuple[Callable, Callable, List[str], str, str]:
    """
    Returns (pattern_fn, replacement_fn, argnames, pat_src, rep_src)
      - pattern_fn(*args) matches torch_attention.default(..., layout="bnsd")
      - replacement_fn(*args) transposes to BSND, runs torch_attention.default(..., layout="bsnd"), transposes back
      - argnames is the ordered arg list: (q, k, v [, attn_mask] [, dropout_p] [, scale])
      - pat_src / rep_src are the exact function bodies (for debug logging)
    """
    # signature in positional order
    argnames: List[str] = ["q", "k", "v"]
    if has_attn_mask:
        argnames.append("attn_mask")
    if has_dropout:
        argnames.append("dropout_p")
    if has_scale:
        argnames.append("scale")

    # build kw parts (omit anything not present; always include is_causal; set layout explicitly
    def _build_kw_src(layout_value: str) -> str:
        parts = []
        if has_attn_mask:
            parts.append("attn_mask=attn_mask")
        if has_dropout:
            parts.append("dropout_p=dropout_p")
        if has_scale:
            parts.append("scale=scale")
        parts.append(f"is_causal={str(is_causal)}")
        parts.append(f'layout="{layout_value}"')
        return ", ".join(parts)

    bnsd_kw_src = _build_kw_src("bnsd")
    bsnd_kw_src = _build_kw_src("bsnd")

    # factories: generate functions with explicit kwargs
    def pattern_factory(argnames=tuple(argnames)):
        args_sig = ", ".join(argnames)
        fn_name = (
            f"attn_bnsd_pat_m{int(has_attn_mask)}_d{int(has_dropout)}_"
            f"c{int(is_causal)}_s{int(has_scale)}"
        )
        body = [f"def {fn_name}({args_sig}):"]
        call = (
            "    return torch.ops.auto_deploy.torch_attention.default("
            f"q, k, v{', ' if bnsd_kw_src else ''}{bnsd_kw_src})"
        )
        body.append(call)
        src = "\n".join(body)
        scope = {"torch": torch}
        exec(src, scope)
        return scope[fn_name]

    def replacement_factory(argnames=tuple(argnames)):
        args_sig = ", ".join(argnames)
        fn_name = (
            f"attn_bnsd_rep_m{int(has_attn_mask)}_d{int(has_dropout)}_"
            f"c{int(is_causal)}_s{int(has_scale)}"
        )
        body = [f"def {fn_name}({args_sig}):"]
        body.append("    q_bsnd = torch.ops.aten.transpose.int(q, 1, 2)")
        body.append("    k_bsnd = torch.ops.aten.transpose.int(k, 1, 2)")
        body.append("    v_bsnd = torch.ops.aten.transpose.int(v, 1, 2)")
        call = (
            "    out_bsnd = torch.ops.auto_deploy.torch_attention.default("
            f"q_bsnd, k_bsnd, v_bsnd{', ' if bsnd_kw_src else ''}{bsnd_kw_src})"
        )
        body.append(call)
        body.append("    return torch.ops.aten.transpose.int(out_bsnd, 1, 2)")
        src = "\n".join(body)
        scope = {"torch": torch}
        exec(src, scope)
        return scope[fn_name]

    pat_fn = pattern_factory()
    rep_fn = replacement_factory()

    return pat_fn, rep_fn, argnames


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
    for has_attn_mask in (False, True):
        for has_dropout in (False, True):
            for is_causal in (False, True):
                for has_scale in (False, True):
                    pat_fn, rep_fn, argnames = make_attn_bnsd_pair(
                        has_attn_mask=has_attn_mask,
                        has_dropout=has_dropout,
                        is_causal=is_causal,
                        has_scale=has_scale,
                    )

                    # Build dummy args following positional signature
                    dummy_args: List[object] = []
                    for name in argnames:
                        if name == "q":
                            dummy_args.append(q)
                        elif name == "k":
                            dummy_args.append(k)
                        elif name == "v":
                            dummy_args.append(v)
                        elif name == "attn_mask":
                            dummy_args.append(attn_mask)
                        elif name == "dropout_p":
                            dummy_args.append(dropout_p)
                        elif name == "scale":
                            dummy_args.append(scale_val)
                        else:
                            raise RuntimeError(f"Unexpected arg name: {name}")

                    # Scalar workaround for present scalars only
                    scalar_workaround = {}
                    if has_dropout:
                        scalar_workaround["dropout_p"] = dropout_p
                    if has_scale:
                        scalar_workaround["scale"] = scale_val
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
