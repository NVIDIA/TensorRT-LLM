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
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, TransformConfig, TransformInfo, TransformRegistry


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


def _grouped_attn_pattern_3(q, k, v, n_rep, attn_mask, dropout_p, scale):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_replacement_3(q, k, v, n_rep, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


# Only expose torch_attention_grouped_sdpa after the transformation
def _grouped_attn_pattern_4(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


def _grouped_attn_replacement_4(q, k, v, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True, scale=scale
    )


@TransformRegistry.register("match_repeat_kv")
class MatchRepeatKV(BaseTransform):
    """
    Match and replace the repeat_kv pattern with torch.ops.auto_deploy.torch_attention_repeat_kv.
    """

    def _apply(
        self, gm: GraphModule, cm: CachedSequenceInterface, factory: ModelFactory
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
        self, gm: GraphModule, cm: CachedSequenceInterface, factory: ModelFactory
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
    torch.ops.auto_deploy.torch_attention_grouped_sdpa.
    """

    def _apply(
        self, gm: GraphModule, cm: CachedSequenceInterface, factory: ModelFactory
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

        num_grouped_patterns = _apply_pattern(gm, "Grouped Attention", register_grouped_attention)

        info = TransformInfo(
            skipped=False,
            num_matches=num_grouped_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


class MatchAttentionLayoutConfig(TransformConfig):
    """Configuration for the insert cached attention transform."""

    attention_op: Type[AttentionDescriptor] = Field(description="The attention descriptor to use.")


@TransformRegistry.register("match_attention_layout")
class MatchAttentionLayout(BaseTransform):
    """
    Match and transform attention operations to match the layout expected by the attention backend.

    If the attention backend expects 'bnsd' layout (batch, num_heads, seq_len, head_dim), which
    is the default for SDPA operations, we don't need to transform anything.

    If the backend expects 'bsnd' layout (batch, seq_len, num_heads, head_dim), we insert
    appropriate transposes before and after SDPA operations and replace them with bsnd_grouped_sdpa.
    """

    config: MatchAttentionLayoutConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MatchAttentionLayoutConfig

    def _apply(
        self, gm: GraphModule, cm: CachedSequenceInterface, factory: ModelFactory
    ) -> Tuple[GraphModule, TransformInfo]:
        # Get attention layout from attention_op
        attention_layout = self.config.attention_op.get_attention_layout()

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
                    self.config.attention_op.get_source_attention_op(),
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

        info = TransformInfo(
            skipped=False,
            num_matches=num_bsnd_patterns,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info
