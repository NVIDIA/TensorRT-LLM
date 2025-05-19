"""
This transformation defines two main RoPE (Rotary Positional Embedding) pattern matchers used
to identify and replace RoPE subgraphs with a custom op (`torch.ops.rope.flashinfer`).

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
from typing import Any, DefaultDict, Dict, List, Optional, Sequence

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import bfs, extract_output_tuple, identify_regions_between_residuals, is_op
from .._graph import canonicalize_graph


def match_explicit_rope(gm: GraphModule) -> GraphModule:
    """
    Identify and replace RoPE subgraphs (explicit cos/sin multiplication pattern):

      - HF-style: output = (raw * unsqueeze(cos)) + (rotate_half(raw) * unsqueeze(sin))
      - DS-style: requires interleaving Q/K before the cos/sin mul

    If exactly two such branches (query and key) are detected within each region, they're replaced
    by a call to `rope::torch_apply_rope_with_qk_interleaving` or
    `rope::torch_apply_rope_with_explicit_cos_sin` respectively.
    """
    ad_logger.info("Match explicit(HF) style RoPE")
    graph = gm.graph
    boundary_nodes: List[torch.fx.Node] = identify_regions_between_residuals(gm)

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        matches = []  # list of (match_info, is_ds)
        node = start_boundary
        while node != end_boundary:
            if is_op(node, torch.ops.aten.add):
                explicit = _match_explicit_rope_subpattern(node)
                if explicit is not None:
                    raw = explicit["raw_input"]
                    # check if this raw is result of DS interleave
                    inter = _match_input_interleave_pattern(raw)
                    if inter is not None:
                        ds_match = {
                            "raw_input": inter["interleaved"],
                            "unsqueeze_cos": explicit["unsqueeze_cos"],
                            "unsqueeze_sin": explicit["unsqueeze_sin"],
                            "add_node": explicit["add_node"],
                        }
                        matches.append((ds_match, True))
                    else:
                        matches.append((explicit, False))
            node = node.next

        if not matches:
            continue
        if len(matches) != 2:
            ad_logger.warning(
                f"Expected exactly 2 legacy RoPE branches between {start_boundary} and {end_boundary}, "
                f"found {len(matches)}."
            )
            continue

        (q_match, q_is_ds), (k_match, k_is_ds) = matches
        if q_is_ds != k_is_ds:
            ad_logger.warning("Mismatched RoPE types between q and k branches")
            continue

        if q_is_ds:
            _process_input_interleave_rope(graph, q_match, k_match)
        else:
            _process_explicit_rope(graph, q_match, k_match, start_boundary)

    gm = canonicalize_graph(gm)
    return gm


def match_complex_rope(gm: GraphModule) -> GraphModule:
    """
    Identify and replace RoPE subgraphs using complex multiplication pattern:

      output = type_as(flatten(view_as_real(mul(view_as_complex(reshape(to_dtype(x))), unsqueeze(freqs_cis, 2)))), x)

    If exactly two such branches (query and key) are detected within each region, they're replaced
    by a call to `torch.ops.rope.torch_apply_rope_with_complex_freqs`.
    """
    ad_logger.info("Match Complex style RoPE")
    graph = gm.graph
    boundary_nodes: List[torch.fx.Node] = identify_regions_between_residuals(gm)

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        matches = []
        node = start_boundary
        while node != end_boundary:
            if is_op(node, torch.ops.aten.type_as):
                match_info = _match_complex_rope_subpattern(node)
                if match_info:
                    matches.append(match_info)
            node = node.next

        if not matches:
            continue
        if len(matches) != 2:
            ad_logger.warning(
                f"Expected exactly 2 complex RoPE branches between {start_boundary} and {end_boundary}, "
                f"found {len(matches)}."
            )
            continue

        # Assume the first matched branch is query (q), second is key (k).
        # This assumption is based on the default ordering in the exported graph,
        # since node naming conventions don't reliably indicate q/k branches.
        q_match, k_match = matches
        _process_complex_rope(graph, q_match, k_match)

    gm = canonicalize_graph(gm)
    return gm


def _get_default_unsqueeze_dim(op):
    schema = next(iter(op._schemas.values()))
    for a in schema.arguments:
        if a.name == "unsqueeze_dim" and a.has_default_value:
            return a.default_value
    raise RuntimeError(f"No default unsqueeze_dim on {op}")


def match_rope_layout(gm: GraphModule, expected_layout: str = "bsnd") -> GraphModule:
    """
    Match and transform input and output of rope ops to the layout specified to meet requirements of optimized ops.
    Supported layout is 'bsnd' (batch, seq, head, dim).
    """
    supported = {"bsnd", "bnsd"}
    if expected_layout.lower() not in supported:
        ad_logger.warning(
            f"Unsupported RoPE layout '{expected_layout}'; expected '{supported}'. Skipping RoPE layout matching."
        )
        return gm

    ad_logger.info(f"Match RoPE layout to {expected_layout}")

    graph = gm.graph
    rope_ops = {
        torch.ops.rope.torch_apply_rope_with_explicit_cos_sin,
        torch.ops.rope.torch_apply_rope_with_qk_interleaving,
        torch.ops.rope.torch_apply_rope_with_complex_freqs,
    }

    need_transpose = False
    need_canonicalize_graph = False
    for node in graph.nodes:
        if not is_op(node, rope_ops):
            continue

        rope_op = next(op for op in rope_ops if is_op(node, op))
        if is_op(node, torch.ops.rope.torch_apply_rope_with_complex_freqs):
            q_node, k_node, freqs_node, *rest = node.args
            unsq = rest[0] if rest else _get_default_unsqueeze_dim(rope_op)
        else:
            q_node, k_node, cos_node, sin_node, *rest = node.args
            unsq = rest[0] if rest else _get_default_unsqueeze_dim(rope_op)

        if unsq == 2:
            current_layout = "bsnd"
        elif unsq == 1:
            current_layout = "bnsd"
        else:
            ad_logger.warning(
                "Unsqueeze_dim is not one of [1, 2]. "
                "Unable to infer layout of q node. Skip layout matching"
            )
            continue

        need_transpose = expected_layout.lower() != current_layout

        if not need_transpose:
            continue

        need_canonicalize_graph = True
        # retrieve q and k output node from node
        q_rope_old, k_rope_old = extract_output_tuple(node, 2)
        if q_rope_old is None or k_rope_old is None:
            ad_logger.warning(
                f"Failed to extract all two outputs from the explicit op, \
                    get {q_rope_old}, {k_rope_old}, fail to match rope layout with {node} with"
            )
            continue

        ad_logger.debug(
            f"Inferred RoPE input layout: '{current_layout}']Mapping layout to '{expected_layout}']"
        )
        with graph.inserting_before(node):
            q_for_op = graph.call_function(torch.ops.aten.transpose, args=(q_node, 1, 2))
            k_for_op = graph.call_function(torch.ops.aten.transpose, args=(k_node, 1, 2))
            q_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(q_for_op,))
            k_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(k_for_op,))

        q_for_op_contig.meta["val"] = q_node.meta["val"].transpose(1, 2)
        k_for_op_contig.meta["val"] = k_node.meta["val"].transpose(1, 2)

        if is_op(node, torch.ops.rope.torch_apply_rope_with_complex_freqs):
            new_args = (
                q_for_op_contig,
                k_for_op_contig,
                freqs_node,
                2 if expected_layout.lower() == "bsnd" else 1,
            )  # unsqueeze_dim updated
        else:
            new_args = (
                q_for_op_contig,
                k_for_op_contig,
                cos_node,
                sin_node,
                2 if expected_layout.lower() == "bsnd" else 1,
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

    if need_canonicalize_graph:
        gm = canonicalize_graph(gm)
    return gm


def optimize_rope(gm: GraphModule) -> GraphModule:
    """
    Scan the FX graph and replace calls to the torch-reference RoPE ops with
    the optimized `rope::flashinfer` kernel.
    Precomputes positional IDs and the fused cosine-sine cache as explicit nodes,
    and reuses those nodes when possible.
    """
    ad_logger.info("RoPE optimization")
    graph = gm.graph
    rope_flash_cache: DefaultDict[Any, Optional[Node]] = defaultdict(lambda: None)
    rope_position_ids_cache: Dict[str, Node] = {}

    for node in list(graph.nodes):
        if is_op(node, torch.ops.rope.torch_apply_rope_with_explicit_cos_sin):
            _optimize_explicit(graph, node, rope_flash_cache, rope_position_ids_cache)
        elif is_op(node, torch.ops.rope.torch_apply_rope_with_complex_freqs):
            _optimize_complex(graph, node, rope_flash_cache, rope_position_ids_cache)

    gm = canonicalize_graph(gm)
    return gm


def _optimize_explicit(
    graph: GraphModule, node: Node, cache: Dict[Any, Node], pos_cache: Dict[str, Node]
) -> None:
    # node.args may be (q, k, cos, sin) or (q, k, cos, sin, unsq)
    q_node, k_node, cos_node, sin_node, *rest = node.args
    # retrieve q and k output node from node
    q_rope_old, k_rope_old = extract_output_tuple(node, 2)
    if q_rope_old is None or k_rope_old is None:
        ad_logger.warning(
            f"Failed to extract all two outputs from the explicit op, \
                get {q_rope_old}, {k_rope_old}, fail to replace {node} with flashinfer rope"
        )
        return

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    # Sanity check that input layout is BSND (no transpose needed).
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        if not (isinstance(q_fake.shape[1], torch.SymInt) and isinstance(q_fake.shape[2], int)):
            ad_logger.warning(
                f"""Sanity check failed: q_fake should have shape [b, s, n, d],
                s should be symbolic and n should be int, instead got shape {q_fake.shape}"""
            )
            return
    elif q_fake is not None:
        ad_logger.warning(
            f"Sanity check failed: q_fake should be 3D or 4D, but got shape {q_fake.shape}"
        )
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
            torch.ops.rope.flashinfer,
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
    q_node, k_node, inv_freq_node = node.args

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    # Sanity check that input layout is BSND (no transpose needed).
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        if not (isinstance(q_fake.shape[1], torch.SymInt) and isinstance(q_fake.shape[2], int)):
            ad_logger.warning(
                f"""Sanity check failed: q_fake should have shape [b, s, n, d],
                s should be symbolic and n should be int, instead got shape {q_fake.shape}"""
            )
            return
    elif q_fake is not None:
        ad_logger.warning(
            f"Sanity check failed: q_fake should be 3D or 4D, but got shape {q_fake.shape}"
        )
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
            torch.ops.rope.flashinfer,
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


def _match_explicit_rope_subpattern(add_node: Node) -> Optional[Dict[str, Node]]:
    """
    Given an aten.add.Tensor node that is expected to compute:
      output = (raw_input * unsqueeze(cos)) + (rotate_half(raw_input) * unsqueeze(sin))
    where rotate_half is implemented as:
      rotate_half(x) = cat([ -slice(x, second_half), slice(x, first_half) ], dim=-1)
    this function inspects the structure of add_node and returns a dictionary with:
       - "raw_input": the original q/k tensor,
       - "unsqueeze_cos": the unsqueeze node feeding the raw multiplication,
       - "unsqueeze_sin": the unsqueeze node feeding the rotated multiplication,
       - "add_node": the addition node itself.
    Returns None if the pattern does not match.
    """
    # Check that add_node is an add operation with two inputs.
    if not is_op(add_node, torch.ops.aten.add):
        return None
    if not (len(add_node.args) == 2):
        return None

    mul1, mul2 = add_node.args
    # Both inputs to the add should be multiplications.
    if not is_op(mul1, torch.ops.aten.mul):
        return None
    if not is_op(mul2, torch.ops.aten.mul):
        return None

    # One branch should be the raw branch and the other the rotated branch.
    # We decide by checking if one multiplication’s first argument is a cat (i.e. the rotate_half result).
    if is_op(mul1.args[0], torch.ops.aten.cat):
        mul_rot = mul1
        mul_raw = mul2
    elif is_op(mul2.args[0], torch.ops.aten.cat):
        mul_rot = mul2
        mul_raw = mul1
    else:
        return None

    # Verify that both multiplications have an unsqueeze as their second argument.
    unsqueeze_cos = mul_raw.args[1]
    unsqueeze_sin = mul_rot.args[1]
    if not is_op(unsqueeze_cos, torch.ops.aten.unsqueeze):
        return None
    if not is_op(unsqueeze_sin, torch.ops.aten.unsqueeze):
        return None

    # Check that the rotated branch is a cat of two tensors along -1.
    cat_node = mul_rot.args[0]
    if not is_op(cat_node, torch.ops.aten.cat):
        return None
    # Expecting two inputs in a list/tuple.
    cat_inputs = cat_node.args[0]
    if not (isinstance(cat_inputs, (list, tuple)) and len(cat_inputs) == 2):
        return None

    # One of the two inputs should be a negation of a slice, the other should be a slice.
    first_item, second_item = cat_inputs
    if not is_op(first_item, torch.ops.aten.neg):
        return None
    if not is_op(second_item, torch.ops.aten.slice):
        return None

    # The negation node should wrap a slice.
    neg_node = first_item
    if not (len(neg_node.args) >= 1 and is_op(neg_node.args[0], torch.ops.aten.slice)):
        return None

    # For simplicity, require that the two slice operations (the one inside neg and the one used directly)
    # are applied on the same original tensor. This original tensor is the one being rotated.
    slice_in_neg = neg_node.args[0]
    if slice_in_neg.args[0] != second_item.args[0]:
        return None

    # Finally, the raw branch should multiply the original tensor (i.e. q or k) by unsqueeze_cos.
    raw_input = mul_raw.args[0]
    # We also expect that the tensor being sliced (and negated) is the same as raw_input.
    if raw_input != slice_in_neg.args[0]:
        return None

    return {
        "raw_input": raw_input,
        "unsqueeze_cos": unsqueeze_cos,
        "unsqueeze_sin": unsqueeze_sin,
        "add_node": add_node,
    }


def _match_complex_rope_subpattern(type_as_node: Node) -> Optional[Dict[str, Node]]:
    """
    Given a type_as node, this function inspects the graph
    structure and returns a dictionary with:
       - "input": the original xq (or xk) tensor,
       - "inv_freq": the freqs_cis tensor (before unsqueeze),
       - "out": the type_as node corresponding to the branch output.

    Expected branch structure for each output:
        x_out = type_as( flatten( view_as_real( view_as_complex(reshape(to_dtype(x))) * unsqueeze(freqs_cis) ) ) )

    Returns None if the structure does not match.
    """
    if not is_op(type_as_node, torch.ops.aten.type_as):
        return None

    # The type_as node should have at least one argument: its first argument is the flatten op.
    if not (len(type_as_node.args) >= 1):
        return None
    flatten_node = type_as_node.args[0]
    if not is_op(flatten_node, torch.ops.aten.flatten):
        return None

    # The input of the flatten op should be a view_as_real op.
    if not (len(flatten_node.args) >= 1):
        return None
    view_as_real_node = flatten_node.args[0]
    if not is_op(view_as_real_node, torch.ops.aten.view_as_real):
        return None

    # The input of view_as_real should be a multiplication.
    if not (len(view_as_real_node.args) >= 1):
        return None
    mul_node = view_as_real_node.args[0]
    if not is_op(mul_node, torch.ops.aten.mul):
        return None
    if len(mul_node.args) != 2:
        return None

    # In the multiplication, one operand should be an unsqueeze of freqs_cis and
    #    the other operand is the output of view_as_complex.
    if is_op(mul_node.args[0], torch.ops.aten.unsqueeze):
        unsqueeze_node = mul_node.args[0]
        vc_node = mul_node.args[1]
    elif is_op(mul_node.args[1], torch.ops.aten.unsqueeze):
        unsqueeze_node = mul_node.args[1]
        vc_node = mul_node.args[0]
    else:
        return None

    if not (len(unsqueeze_node.args) >= 2):
        return None
    unsqueeze_dim = unsqueeze_node.args[1]

    inv_freq_candidate = unsqueeze_node.args[0]

    # Match the view_as_complex branch.
    if not is_op(vc_node, torch.ops.aten.view_as_complex):
        return None
    if not (len(vc_node.args) >= 1):
        return None
    reshape_node = vc_node.args[0]
    if not is_op(reshape_node, torch.ops.aten.reshape):
        return None

    # The reshape op should get its input from a to(dtype) conversion.
    if not (len(reshape_node.args) >= 1):
        return None
    to_node = reshape_node.args[0]
    if not is_op(to_node, torch.ops.aten.to):
        return None
    if not (len(to_node.args) >= 1):
        return None
    input_tensor = to_node.args[0]

    return {
        "input": input_tensor,
        "inv_freq": inv_freq_candidate,
        "out": type_as_node,
        "unsqueeze_dim": unsqueeze_dim,
    }


def _process_explicit_rope(
    graph: GraphModule.graph,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
    start_boundary: Node,
) -> None:
    """
    Replace matched Explicit RoPE subgraph with `rope::torch_apply_rope_with_explicit_cos_sin`.
    """
    q_node = q_match["raw_input"]
    k_node = k_match["raw_input"]
    cos_unsq = q_match["unsqueeze_cos"]
    sin_unsq = q_match["unsqueeze_sin"]
    cos_node = cos_unsq.args[0]
    sin_node = sin_unsq.args[0]
    unsq_dim = cos_unsq.args[1]
    add_node = q_match["add_node"]

    # Sanity-check: ensure cos/sin nodes trace back to aten.cos/aten.sin.
    bfs(
        cos_node,
        lambda n: is_op(n, torch.ops.aten.cos),
        attr_next="all_input_nodes",
        boundary=start_boundary,
    )
    bfs(
        sin_node,
        lambda n: is_op(n, torch.ops.aten.sin),
        attr_next="all_input_nodes",
        boundary=start_boundary,
    )

    with graph.inserting_before(add_node):
        rope_node = graph.call_function(
            torch.ops.rope.torch_apply_rope_with_explicit_cos_sin,
            args=(q_node, k_node, cos_node, sin_node, unsq_dim),
        )

    with graph.inserting_after(rope_node):
        out_q = graph.call_function(operator.getitem, args=(rope_node, 0))
        out_k = graph.call_function(operator.getitem, args=(rope_node, 1))

    out_q.meta["val"] = add_node.meta.get("val", None)
    out_k.meta["val"] = k_match["add_node"].meta.get("val", None)

    q_match["add_node"].replace_all_uses_with(out_q)
    k_match["add_node"].replace_all_uses_with(out_k)


def _process_complex_rope(
    graph: GraphModule.graph,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
) -> None:
    """
    Replace matched Complex RoPE subgraph with `rope::torch_apply_rope_with_complex_freqs`.
    """
    xq = q_match["input"]
    xk = k_match["input"]
    inv = q_match["inv_freq"]
    usdim = q_match["unsqueeze_dim"]
    out_node = q_match.get("out")

    if inv != k_match["inv_freq"]:
        ad_logger.warning(
            "Mismatch of freqs_cis (inv_freq) between branches. Fail to match complex rope pattern"
        )
        return

    with graph.inserting_before(out_node):
        rope_node = graph.call_function(
            torch.ops.rope.torch_apply_rope_with_complex_freqs,
            args=(xq, xk, inv, usdim),
        )

    with graph.inserting_after(rope_node):
        out_q = graph.call_function(operator.getitem, args=(rope_node, 0))
        out_k = graph.call_function(operator.getitem, args=(rope_node, 1))

    out_q.meta["val"] = out_node.meta.get("val", None)
    out_k.meta["val"] = k_match["out"].meta.get("val", None)

    out_node.replace_all_uses_with(out_q)
    k_match["out"].replace_all_uses_with(out_k)


def _process_input_interleave_rope(
    graph: GraphModule,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
) -> None:
    """
    Replace a matched DS-style RoPE subgraph with a call to rope::torch_apply_rope_with_qk_interleaving.
    Cache the one-time unsqueeze of cos/sin.
    """
    q_node = q_match["raw_input"]
    k_node = k_match["raw_input"]
    cos_node = q_match["unsqueeze_cos"].args[0]
    sin_node = q_match["unsqueeze_sin"].args[0]
    # A patch for the case when q_output appears before k_input in the graph
    # Move q_output down right before its first user so that graph remains in
    # topological order after inserting the apply rope custom op
    q_match["add_node"] = _move_node_before_first_user(q_match["add_node"])

    # Infer unsqueeze_dim from layout
    unsq_dim = 1
    fake = q_node.meta.get("val", None)
    if fake is not None and len(fake.shape) == 4:
        # if shape[1] symbolic, it's [B, S, N, D] => BSND -> head dim is 2
        if isinstance(fake.shape[1], torch.SymInt):
            unsq_dim = 2
        else:
            unsq_dim = 1

    with graph.inserting_after(_get_last_node([q_node, k_node, cos_node, sin_node])):
        ds_node = graph.call_function(
            torch.ops.rope.torch_apply_rope_with_qk_interleaving,
            args=(q_node, k_node, cos_node, sin_node, unsq_dim),
        )

    with graph.inserting_after(ds_node):
        q_out = graph.call_function(operator.getitem, args=(ds_node, 0))
        k_out = graph.call_function(operator.getitem, args=(ds_node, 1))

    q_out.meta["val"] = q_match["add_node"].meta.get("val", None)
    k_out.meta["val"] = k_match["add_node"].meta.get("val", None)

    q_match["add_node"].replace_all_uses_with(q_out)
    k_match["add_node"].replace_all_uses_with(k_out)


def _move_node_before_first_user(node: Node) -> Node:
    """
    Remove `node` from the graph and re-insert a clone of it immediately
    before its earliest user. Returns the new node.

    If `node` has no users, or is already right before its first user,
    this is a no-op and returns the original node.
    """
    graph = node.graph
    ordering = list(graph.nodes)

    users = list(node.users)
    if not users:
        return node

    # locate the earliest user in the current ordering
    first_user = min(users, key=lambda u: ordering.index(u))
    if ordering.index(node) == ordering.index(first_user) - 1:
        return node

    with graph.inserting_before(first_user):
        new_node = graph.node_copy(node, lambda n: n)

    node.replace_all_uses_with(new_node)
    graph.erase_node(node)

    return new_node


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
            ad_logger.warning(
                f"Meta['val'] for {name} not available; skipping RoPE transformation."
            )
            return False

        # Check dtype
        if fake_val.dtype not in (torch.float16, torch.bfloat16):
            ad_logger.warning(
                f"""{name} tensor is {fake_val.dtype},
                expected half precision (float16 or bfloat16). Skipping RoPE transformation."""
            )
            return False

        # Check head_dim
        if len(fake_val.shape) < 1:
            ad_logger.warning(f"{name} tensor has invalid shape {fake_val.shape}.")
            return False
        head_dim = fake_val.shape[-1]
        if isinstance(head_dim, int) and head_dim % 64 != 0:
            ad_logger.warning(
                f"{name} head_dim = {head_dim} is not a multiple of 64. Skipping RoPE transformation."
            )
            return False

        # Check shape
        if not isinstance(fake_val.shape[1], torch.SymInt):
            ad_logger.warning(
                f"{name} has shape {fake_val.shape} that is not supported. Only support [B, S, N, D] layout.\
                Skipping RoPE transformation."
            )
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
