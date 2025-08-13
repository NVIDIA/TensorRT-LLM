"""
This transformation defines two main RoPE (Rotary Positional Embedding) pattern matchers used
to identify and replace RoPE subgraphs with a custom op (`torch.ops.auto_deploy.flashinfer_rope`).

Supported RoPE variants:

1. Explicit Cos/Sin Multiplication (HF-style, e.g., LLaMA, Mixtral, Qwen)
   - Input layout: non-interleaved, [B, N, S, D] with unsqueeze_dim=1 and
        [B, S, N, D] with unsqueeze_dim=2, default is [B, N, S, D]
   - Frequencies are provided as separate `cos` and `sin` tensors of shape [B, S, head_dim].
   - Source code:
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

2. Complex Multiplication (GPTJ/Llama-stack-style, interleaved)
   - Input layout: [B, S, N, D] (interleaved)
   - Frequencies are combined into a single complex-valued tensor `freqs_cis` of shape [B, S, head_dim // 2].
   - Source code:
        def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,  # Expected shape: (B, seq, head_dim//2)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

Supported Minor variants:
- DeepSeekV3:   reshape + transpose before applying RoPE.
                dynamic position-based updates to frequency cache.

TODO: Support other variants:
- Phi-4: rotary applied only to part of the hidden dimension (q_rot, q_pass split).
- LLaMA4 Vision: 2D rotary frequencies constructed from image patches.
"""

import operator
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Sequence, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import extract_op_args, extract_output_tuple, is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, Match, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _explicit_rope_pattern(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _explicit_rope_repl(q, k, cos, sin, unsqueeze_dim):
    return torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default(
        q, k, cos, sin, unsqueeze_dim
    )


def _interleaved_rope_pattern(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _interleaved_rope_repl(q, k, cos, sin, unsqueeze_dim):
    return torch.ops.auto_deploy.torch_rope_with_qk_interleaving.default(
        q, k, cos, sin, unsqueeze_dim
    )


# exporting with {"unsqueeze_dim": 2},
# would confuse the pattern matcher since '2' is arg of other ops
def _complex_rope_pattern(xq, xk, freqs_cis, unsqueeze_dim=1):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_q = freqs_cis.unsqueeze(unsqueeze_dim)
    freqs_k = freqs_cis.unsqueeze(unsqueeze_dim)
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _complex_rope_repl(q, k, freqs_cis, unsqueeze_dim):
    return torch.ops.auto_deploy.torch_rope_with_complex_freqs.default(
        q, k, freqs_cis, unsqueeze_dim
    )


def _explicit_not_interleaved(match: Match) -> bool:
    q, k = match.kwargs.get("q"), match.kwargs.get("k")
    return not any(isinstance(n, Node) and _match_input_interleave_pattern(n) for n in (q, k))


@TransformRegistry.register("match_rope_pattern")
class MatchRopePattern(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        # dummy shapes: can be arbitrary
        batch_size = 8
        seq_len = 16
        num_heads = 8
        hidden_size = 512
        head_dim = hidden_size // num_heads

        dummy_explicit = [
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float16
            ),
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float16
            ),
            torch.randn(batch_size, seq_len, head_dim, device="meta", dtype=torch.float16),
            torch.randn(batch_size, seq_len, head_dim, device="meta", dtype=torch.float16),
        ]
        dummy_complex = [
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float16
            ),
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float16
            ),
            torch.randn(batch_size, seq_len, head_dim // 2, device="meta", dtype=torch.float16),
        ]
        dummy_complex_2 = [
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float32
            ),
            torch.randn(
                batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float32
            ),
            torch.randn(batch_size, seq_len, head_dim // 2, device="meta", dtype=torch.float32),
        ]
        register_ad_pattern(
            search_fn=_explicit_rope_pattern,
            replace_fn=_explicit_rope_repl,
            patterns=patterns,
            dummy_args=dummy_explicit,
            op_ignore_types={torch.ops.aten.slice.Tensor: (int,)},
            scalar_workaround={"unsqueeze_dim": 1},
            extra_check=_explicit_not_interleaved,
        )
        register_ad_pattern(
            search_fn=_interleaved_rope_pattern,
            replace_fn=_interleaved_rope_repl,
            patterns=patterns,
            dummy_args=dummy_explicit,
            op_ignore_types={
                torch.ops.aten.slice.Tensor: (int,),
                torch.ops.aten.reshape.default: (int,),
                torch.ops.aten.view.default: (int,),
            },
            scalar_workaround={"unsqueeze_dim": 1},
        )
        register_ad_pattern(
            search_fn=_complex_rope_pattern,
            replace_fn=_complex_rope_repl,
            patterns=patterns,
            dummy_args=dummy_complex,
            op_ignore_types={
                torch.ops.aten.reshape.default: (int,),
            },
            scalar_workaround={"unsqueeze_dim": 1},
        )
        register_ad_pattern(
            search_fn=_complex_rope_pattern,
            replace_fn=_complex_rope_repl,
            patterns=patterns,
            dummy_args=dummy_complex_2,
            op_ignore_types={
                torch.ops.aten.reshape.default: (int,),
            },
            scalar_workaround={"unsqueeze_dim": 1},
        )

        num_matches = patterns.apply(graph)

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )

        return gm, info


class MatchRopeLayoutConfig(TransformConfig):
    """Configuration for the match rope layout transform."""

    expected_layout: str = Field(
        default="bsnd",
        description="The expected layout of the rope operation. Must be one of 'bsnd' or 'bnsd'.",
    )


@TransformRegistry.register("match_rope_layout")
class MatchRopeLayout(BaseTransform):
    """
    Match and transform input and output of rope ops to the layout specified to meet requirements of optimized ops.
    Supported layout is 'bsnd' (batch, seq, head, dim).
    """

    config: MatchRopeLayoutConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MatchRopeLayoutConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        supported = {"bsnd", "bnsd"}
        if self.config.expected_layout.lower() not in supported:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        graph = gm.graph
        rope_ops = {
            torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin,
            torch.ops.auto_deploy.torch_rope_with_qk_interleaving,
            torch.ops.auto_deploy.torch_rope_with_complex_freqs,
        }

        need_transpose = False
        num_rope_layout_matches = 0
        for node in graph.nodes:
            if not is_op(node, rope_ops):
                continue

            if is_op(node, torch.ops.auto_deploy.torch_rope_with_complex_freqs):
                q_node, k_node, freqs_node, unsq = extract_op_args(
                    node,
                    "xq",  # argument name in schema
                    "xk",
                    "freqs_cis",
                    "unsqueeze_dim",
                )
            else:
                q_node, k_node, cos_node, sin_node, unsq = extract_op_args(
                    node, "q", "k", "cos", "sin", "unsqueeze_dim"
                )

            if unsq == 2:
                current_layout = "bsnd"
            elif unsq == 1:
                current_layout = "bnsd"
            else:
                continue

            need_transpose = self.config.expected_layout.lower() != current_layout

            if not need_transpose:
                continue

            num_rope_layout_matches += 1
            # retrieve q and k output node from node
            q_rope_old, k_rope_old = extract_output_tuple(node, 2)
            if q_rope_old is None or k_rope_old is None:
                continue

            with graph.inserting_before(node):
                q_for_op = graph.call_function(torch.ops.aten.transpose, args=(q_node, 1, 2))
                k_for_op = graph.call_function(torch.ops.aten.transpose, args=(k_node, 1, 2))
                q_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(q_for_op,))
                k_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(k_for_op,))

            q_for_op_contig.meta["val"] = q_node.meta["val"].transpose(1, 2)
            k_for_op_contig.meta["val"] = k_node.meta["val"].transpose(1, 2)

            if is_op(node, torch.ops.auto_deploy.torch_rope_with_complex_freqs):
                new_args = (
                    q_for_op_contig,
                    k_for_op_contig,
                    freqs_node,
                    2 if self.config.expected_layout.lower() == "bsnd" else 1,
                )  # unsqueeze_dim updated
            else:
                new_args = (
                    q_for_op_contig,
                    k_for_op_contig,
                    cos_node,
                    sin_node,
                    2 if self.config.expected_layout.lower() == "bsnd" else 1,
                )  # unsqueeze_dim updated
            node.args = new_args

            with graph.inserting_after(q_rope_old):
                q_rope_new = graph.call_function(torch.ops.aten.transpose, args=(q_rope_old, 1, 2))
            with graph.inserting_after(k_rope_old):
                k_rope_new = graph.call_function(torch.ops.aten.transpose, args=(k_rope_old, 1, 2))

            # Preserve fake tensor in meta["val"] for the transposed inputs
            q_rope_new.meta["val"] = q_rope_old.meta["val"]
            q_rope_old.meta["val"] = q_rope_old.meta["val"].transpose(1, 2)
            k_rope_new.meta["val"] = k_rope_old.meta["val"]
            k_rope_old.meta["val"] = k_rope_old.meta["val"].transpose(1, 2)

            q_rope_old.replace_all_uses_with(q_rope_new)
            k_rope_old.replace_all_uses_with(k_rope_new)
            q_rope_new.args = (q_rope_old, 1, 2)
            k_rope_new.args = (k_rope_old, 1, 2)

        info = TransformInfo(
            skipped=False,
            num_matches=num_rope_layout_matches,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


@TransformRegistry.register("optimize_rope")
class OptimizeRope(BaseTransform):
    """
    Scan the FX graph and replace calls to the torch-reference RoPE ops with
    the optimized `rope::flashinfer` kernel.
    Precomputes positional IDs and the fused cosine-sine cache as explicit nodes,
    and reuses those nodes when possible.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        rope_flash_cache: DefaultDict[Any, Optional[Node]] = defaultdict(lambda: None)
        rope_position_ids_cache: Dict[str, Node] = {}

        num_rope_optimizations = 0
        for node in list(graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
                _optimize_explicit(graph, node, rope_flash_cache, rope_position_ids_cache)
            elif is_op(node, torch.ops.auto_deploy.torch_rope_with_complex_freqs):
                _optimize_complex(graph, node, rope_flash_cache, rope_position_ids_cache)
            else:
                continue
            num_rope_optimizations += 1

        info = TransformInfo(
            skipped=False, num_matches=num_rope_optimizations, is_clean=False, has_valid_shapes=True
        )

        return gm, info


def _optimize_explicit(
    graph: GraphModule, node: Node, cache: Dict[Any, Node], pos_cache: Dict[str, Node]
) -> None:
    # node.args may be (q, k, cos, sin) or (q, k, cos, sin, unsq)
    q_node, k_node, cos_node, sin_node, *rest = node.args
    # retrieve q and k output node from node
    q_rope_old, k_rope_old = extract_output_tuple(node, 2)
    if q_rope_old is None or k_rope_old is None:
        return

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    # Sanity check that input layout is BSND (no transpose needed).
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        if not (isinstance(q_fake.shape[1], torch.SymInt) and isinstance(q_fake.shape[2], int)):
            return
    elif q_fake is not None:
        return

    head_dim = cos_node.meta["val"].shape[-1]
    half_head_dim = head_dim // 2

    cache_key = (cos_node, sin_node)
    if cache_key in cache:
        fused_cos_sin_to = cache[cache_key]
    else:
        with graph.inserting_after(cos_node):
            cos_prefix = graph.call_function(
                torch.ops.aten.slice, args=(cos_node, -1, 0, half_head_dim)
            )
        with graph.inserting_after(sin_node):
            sin_prefix = graph.call_function(
                torch.ops.aten.slice, args=(sin_node, -1, 0, half_head_dim)
            )
        with graph.inserting_after(sin_prefix):
            fused_cos_sin = graph.call_function(
                torch.ops.aten.cat, args=((cos_prefix, sin_prefix), -1)
            )
        with graph.inserting_after(q_node):
            sym_batch = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, 0))
            sym_seq = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, 1))
        with graph.inserting_after(_get_last_node([sym_batch, sym_seq])):
            bs_seq = graph.call_function(operator.mul, args=(sym_batch, sym_seq))
        with graph.inserting_after(_get_last_node([bs_seq, fused_cos_sin])):
            fused_cos_sin_flat = graph.call_function(
                torch.ops.aten.view, args=(fused_cos_sin, (bs_seq, -1))
            )
        with graph.inserting_after(fused_cos_sin_flat):
            fused_cos_sin_to = graph.call_function(
                torch.ops.aten.to, args=(fused_cos_sin_flat, torch.float32)
            )
        cache[cache_key] = fused_cos_sin_to

    with graph.inserting_before(node):
        position_ids = _get_position_ids(
            graph,
            q_node,
            batch_dim=0,
            seq_dim=1,
            rope_position_ids_cache=pos_cache,
        )
        flash_node = graph.call_function(
            torch.ops.auto_deploy.flashinfer_rope,
            args=(q_node, k_node, position_ids, fused_cos_sin_to, True),
        )

    with graph.inserting_after(flash_node):
        q_rope_new = graph.call_function(operator.getitem, args=(flash_node, 0))
        k_rope_new = graph.call_function(operator.getitem, args=(flash_node, 1))

    q_rope_new.meta["val"] = q_rope_old.meta.get("val", None)
    k_rope_new.meta["val"] = k_rope_old.meta.get("val", None)

    q_rope_old.replace_all_uses_with(q_rope_new)
    k_rope_old.replace_all_uses_with(k_rope_new)

    graph.erase_node(q_rope_old)
    graph.erase_node(k_rope_old)


def _optimize_complex(
    graph: GraphModule, node: Node, cache: Dict[Any, Node], pos_cache: Dict[str, Node]
) -> None:
    # q_node, k_node, inv_freq_node = node.args
    q_node, k_node, inv_freq_node = extract_op_args(
        node,
        "xq",  # argument name in schema
        "xk",
        "freqs_cis",
    )

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    # Sanity check that input layout is BSND (no transpose needed).
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        if not (isinstance(q_fake.shape[1], torch.SymInt) and isinstance(q_fake.shape[2], int)):
            return
    elif q_fake is not None:
        return

    # Retrieve or register the lookup table for inv_freq_node -> cos_sin_flash
    if inv_freq_node in cache:
        cos_sin_flash = cache[inv_freq_node]
    else:
        # Compute the fused cosine/sine cache.
        with graph.inserting_after(inv_freq_node):
            real_part = graph.call_function(torch.ops.aten.real, args=(inv_freq_node,))
            imag_part = graph.call_function(torch.ops.aten.imag, args=(inv_freq_node,))
        with graph.inserting_after(real_part):
            cos_sin_flash_3d = graph.call_function(
                torch.ops.aten.cat, args=((real_part, imag_part), -1)
            )
        with graph.inserting_after(q_node):
            sym_batch = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, 0))
            sym_seq = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, 1))
        with graph.inserting_after(_get_last_node([sym_batch, sym_seq])):
            bs_seq = graph.call_function(operator.mul, args=(sym_batch, sym_seq))
        with graph.inserting_after(_get_last_node([bs_seq, cos_sin_flash_3d])):
            fused_cos_sin_flat = graph.call_function(
                torch.ops.aten.view, args=(cos_sin_flash_3d, (bs_seq, -1))
            )
        with graph.inserting_after(fused_cos_sin_flat):
            cos_sin_flash = graph.call_function(
                torch.ops.aten.to, args=(fused_cos_sin_flat, torch.float32)
            )
        cache[inv_freq_node] = cos_sin_flash

    with graph.inserting_before(node):
        position_ids = _get_position_ids(
            graph, q_node, batch_dim=0, seq_dim=1, rope_position_ids_cache=pos_cache
        )
        flash_node = graph.call_function(
            torch.ops.auto_deploy.flashinfer_rope,
            args=(q_node, k_node, position_ids, cos_sin_flash, False),
        )

    flash_node.meta["val"] = node.meta.get("val", None)
    node.replace_all_uses_with(flash_node)
    graph.erase_node(node)


def _match_input_interleave_pattern(node: Node) -> Optional[Dict[str, Node]]:
    """
    Detect DeepSeek-style interleave on Q/K:
      reshape(transpose(view(raw, [b,h,s,d//2,2]), 4, 3), [b,h,s,d])
    Returns:
      {"interleaved": raw_node} if matched, else None.
    """
    if not is_op(node, torch.ops.aten.reshape):
        return None
    transpose_node = node.args[0]
    if not is_op(transpose_node, torch.ops.aten.transpose):
        return None
    view_node = transpose_node.args[0]
    if not is_op(view_node, torch.ops.aten.view):
        return None
    raw_node = view_node.args[0]
    if not isinstance(raw_node, Node):
        return None
    return {"interleaved": raw_node}


def _get_last_node(nodes: Sequence[Node]) -> Node:
    """
    Given a list of FX Nodes,
    return the one that appears last in the graph's execution order.
    """
    if not nodes:
        raise ValueError("`nodes` must be a non-empty sequence of FX Node objects")

    graph = nodes[0].graph
    ordering = list(graph.nodes)

    # Sanity check that all nodes are in same graph
    valid = [n for n in nodes if n in ordering]
    if not valid:
        raise ValueError("None of the provided nodes belong to the same graph")

    last = max(valid, key=lambda n: ordering.index(n))
    return last


def _validate_rope_inputs(q_node: Node, k_node: Node) -> bool:
    """
    Validates that:
    - The last dimension (head_dim) of both q and k is a multiple of 64.
    - The dtype of q and k is half precision (bfloat16 or float16).
    - Layout should be [B,S,N,D] (dim 1 should be symbolic)
    """
    for name, node in [("q", q_node), ("k", k_node)]:
        fake_val = node.meta.get("val", None)
        if fake_val is None:
            return False

        # Check dtype
        if fake_val.dtype not in (torch.float16, torch.bfloat16):
            return False

        # Check head_dim
        if len(fake_val.shape) < 1:
            return False
        head_dim = fake_val.shape[-1]
        if isinstance(head_dim, int) and head_dim % 64 != 0:
            return False

        # Check shape
        if not isinstance(fake_val.shape[1], torch.SymInt):
            return False

    return True


def _get_position_ids(
    graph: GraphModule,
    q_node: Node,
    batch_dim: int = 0,
    seq_dim: int = 1,
    rope_position_ids_cache: Dict[str, Node] = None,
) -> Node:
    """
    Retrieves the cached position_ids from the graph if available, or computes and caches them.
    It uses the symbolic batch and sequence sizes from q_node with the provided dimension indices.
    """
    if rope_position_ids_cache is None:
        rope_position_ids_cache = {}

    if "position_ids" in rope_position_ids_cache:
        return rope_position_ids_cache["position_ids"]

    sym_batch = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, batch_dim))
    sym_seq = graph.call_function(torch.ops.aten.sym_size.int, args=(q_node, seq_dim))
    bs_seq = graph.call_function(operator.mul, args=(sym_batch, sym_seq))

    # Retrieve device information, ensuring it is a torch.device.
    device = q_node.meta.get("device", "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    position_ids = graph.call_function(
        torch.ops.aten.arange,
        args=(bs_seq,),
        kwargs={"dtype": torch.float32, "device": device, "pin_memory": False},
    )
    rope_position_ids_cache["position_ids"] = position_ids
    return position_ids
