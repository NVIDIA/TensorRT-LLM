"""Pattern matching for detecting eager and grouped attention pattern from Huggingface models."""

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Type

import torch
import torch.nn.functional as F
from torch.fx import GraphModule

from ...custom_ops.attention_interface import AttentionDescriptor
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from .._graph import canonicalize_graph, lift_to_meta


def _apply_pattern(
    gm: GraphModule,
    pattern_name: str,
    register_fn: Callable[[ADPatternMatcherPass], None],
    shape_prop: bool = False,
) -> None:
    """Utility to register and apply a pattern."""
    patterns = ADPatternMatcherPass()
    register_fn(patterns)
    num_matches = patterns.apply(gm.graph)

    if num_matches > 0:
        with lift_to_meta(gm) if shape_prop else nullcontext():
            canonicalize_graph(gm, shape_prop=shape_prop)
    ad_logger.info(f"Found and matched {num_matches} {pattern_name} pattern(s)")


def match_attention_pattern(gm: GraphModule) -> None:
    """
    Match and replace attention patterns in the graph.

    This transformation detects both eager (ungrouped) and grouped attention patterns,
    and replaces them with `torch.ops.auto_deploy.torch_attention_grouped_sdpa`.
    """

    def register_eager_attention(patterns: ADPatternMatcherPass):
        for pattern_config in _get_sfdp_patterns():
            register_ad_pattern(**pattern_config, patterns=patterns)

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

        register_ad_pattern(
            search_fn=_grouped_attn_pattern,
            replace_fn=_grouped_attn_replacement,
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

    _apply_pattern(gm, "Eager Attention", register_eager_attention)
    _apply_pattern(gm, "Grouped Attention", register_grouped_attention)


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
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
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
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
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
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
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


def _grouped_attn_pattern(q, k, v, n_rep, attn_mask, dropout_p, scale):
    # Repeat k and v
    k_rep = (
        torch.unsqueeze(k, 2)
        .expand(k.shape[0], k.shape[1], n_rep, k.shape[2], k.shape[3])
        .reshape(k.shape[0], k.shape[1] * n_rep, k.shape[2], k.shape[3])
    )

    v_rep = (
        torch.unsqueeze(v, 2)
        .expand(v.shape[0], v.shape[1], n_rep, v.shape[2], v.shape[3])
        .reshape(v.shape[0], v.shape[1] * n_rep, v.shape[2], v.shape[3])
    )

    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_replacement(q, k, v, n_rep, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


# Only expose torch_attention_grouped_sdpa after the transformation
def _grouped_attn_pattern_2(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_replacement_2(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def match_attention_layout(gm: GraphModule, attention_op: Type[AttentionDescriptor]) -> None:
    """
    Match and transform attention operations to match the layout expected by the attention backend.

    If the attention backend expects 'bnsd' layout (batch, num_heads, seq_len, head_dim), which
    is the default for SDPA operations, we don't need to transform anything.

    If the backend expects 'bsnd' layout (batch, seq_len, num_heads, head_dim), we insert
    appropriate transposes before and after SDPA operations and replace them with bsnd_grouped_sdpa.
    """
    # Get attention layout from attention_op
    attention_layout = attention_op.get_attention_layout()

    # List of SDPA operations to look for
    sdpa_ops = {
        torch.ops.auto_deploy.torch_attention_sdpa,
        torch.ops.auto_deploy.torch_attention_grouped_sdpa,
    }

    graph = gm.graph
    num_bsnd_patterns = 0

    # Look for SDPA operations
    for sdpa_node in list(graph.nodes):
        if sdpa_node.op != "call_function" or not is_op(sdpa_node, sdpa_ops):
            continue

        ad_logger.debug(f"Found SDPA node to transform for bsnd layout: {sdpa_node}")

        # Extract q, k, v inputs
        q, k, v = sdpa_node.args[:3]

        # Check if we need to transpose the inputs
        if attention_layout == "bsnd":
            # Add transposes before the node (from bnsd to bsnd)
            with graph.inserting_before(sdpa_node):
                q_updated = graph.call_function(torch.ops.aten.transpose.int, args=(q, 1, 2))
                k_updated = graph.call_function(torch.ops.aten.transpose.int, args=(k, 1, 2))
                v_updated = graph.call_function(torch.ops.aten.transpose.int, args=(v, 1, 2))

            # Preserve fake tensor in meta["val"] for the transposed inputs
            q_updated.meta["val"] = q.meta["val"].transpose(1, 2)
            k_updated.meta["val"] = k.meta["val"].transpose(1, 2)
            v_updated.meta["val"] = v.meta["val"].transpose(1, 2)
        elif attention_layout == "bnsd":
            # we don't need to do anything...
            q_updated = q
            k_updated = k
            v_updated = v
        else:
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # Create bsnd_grouped_sdpa node with the same args as the original node
        # but using the transposed inputs
        with graph.inserting_before(sdpa_node):
            source_sdpa_node = graph.call_function(
                attention_op.get_source_attention_op(),
                args=(q_updated, k_updated, v_updated) + sdpa_node.args[3:],
                kwargs=sdpa_node.kwargs,
            )

        # Check if need to update the output node to match the layout
        if attention_layout == "bsnd":
            # Add transpose for the output (from bsnd back to bnsd)
            with graph.inserting_after(source_sdpa_node):
                output_updated = graph.call_function(
                    torch.ops.aten.transpose.int, args=(source_sdpa_node, 1, 2)
                )

            # Preserve fake tensor in meta["val"] for the transposed inputs
            source_sdpa_node.meta["val"] = sdpa_node.meta["val"].transpose(1, 2).contiguous()
            output_updated.meta["val"] = source_sdpa_node.meta["val"].transpose(1, 2)
        elif attention_layout == "bnsd":
            output_updated = source_sdpa_node
        else:
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # Replace the old node with the transposed output
        sdpa_node.replace_all_uses_with(output_updated)

        num_bsnd_patterns += 1

    # Clean up the graph if we made any replacements
    if num_bsnd_patterns:
        canonicalize_graph(gm)
        ad_logger.debug(f"Transformed graph for bsnd layout: {gm}")

    ad_logger.info(f"Found and matched {num_bsnd_patterns} attention layouts")
