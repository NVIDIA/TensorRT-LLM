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

        def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

TODO: Support Minor variants:
- DeepSeekV3: reshape + transpose before applying RoPE.
- DeepSeekV3: dynamic position-based updates to frequency cache.
- Phi-4: rotary applied only to part of the hidden dimension (q_rot, q_pass split).
- LLaMA4 Vision: 2D rotary frequencies constructed from image patches.
"""

import operator
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import bfs, identify_regions_between_residuals, is_op
from .._graph import canonicalize_graph


def _match_ds_rope_interleave_pattern(node: Node) -> Optional[Dict[str, Node]]:
    """
    Detect DeepSeek-style interleave on Q/K:
      reshape(transpose(view(raw, [b,h,s,d//2,2]), 4, 3), [b,h,s,d])
    Returns:
      {"interleaved": reshape_node} if matched, else None.
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


def match_explicit_rope(gm: GraphModule) -> GraphModule:
    """
    Identify and replace legacy RoPE subgraphs (explicit cos/sin multiplication pattern):

      - HF style: output = (raw * unsqueeze(cos)) + (rotate_half(raw) * unsqueeze(sin))
      - DS-style: requires interleaving Q/K before the cos/sin mul → apply_rope_ds

    If exactly two such branches (query and key) are detected within each region, they're replaced
    by a call to `torch.ops.rope.flashinfer`.
    """
    graph = gm.graph
    boundary_nodes: List[torch.fx.Node] = identify_regions_between_residuals(gm)

    rope_flash_cache: DefaultDict[Any, Optional[Node]] = defaultdict(lambda: None)
    rope_position_ids_cache: Dict[str, Node] = {}

    rope_ds_cache: DefaultDict[Tuple[Node, Node, int], Tuple[Optional[Node], Optional[Node]]] = (
        defaultdict(lambda: (None, None))
    )

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        matches = []  # list of (match_info, is_ds)
        node = start_boundary
        while node != end_boundary:
            if is_op(node, torch.ops.aten.add):
                explicit = _match_explicit_rope_subpattern(node)
                if explicit is not None:
                    raw = explicit["raw_input"]
                    # check if this raw is result of DS interleave
                    inter = _match_ds_rope_interleave_pattern(raw)
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
            raise RuntimeError(
                f"Expected exactly 2 legacy RoPE branches between {start_boundary} and {end_boundary}, "
                f"found {len(matches)}."
            )

        (q_match, q_is_ds), (k_match, k_is_ds) = matches
        if q_is_ds != k_is_ds:
            raise RuntimeError("Mismatched RoPE types between q and k branches")

        if q_is_ds:
            _process_ds_rope(
                graph,
                q_match,
                k_match,
                rope_ds_cache,
            )
        else:
            _process_explicit_rope(
                graph,
                q_match,
                k_match,
                start_boundary,
                rope_flash_cache,
                rope_position_ids_cache,
            )

    gm = canonicalize_graph(gm)
    return gm


def match_complex_rope(gm: GraphModule) -> GraphModule:
    """
    Identify and replace RoPE subgraphs using complex multiplication pattern:

      output = type_as(flatten(view_as_real(mul(view_as_complex(reshape(to_dtype(x))), unsqueeze(freqs_cis, 2)))), x)

    If exactly two such branches (query and key) are detected within each region, they're replaced
    by a call to `torch.ops.rope.flashinfer`.
    """
    graph = gm.graph
    boundary_nodes: List[torch.fx.Node] = identify_regions_between_residuals(gm)

    rope_flash_cache: DefaultDict[Any, Optional[Node]] = defaultdict(lambda: None)
    rope_position_ids_cache: Dict[str, Node] = {}

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
            raise RuntimeError(
                f"Expected exactly 2 complex RoPE branches between {start_boundary} and {end_boundary}, "
                f"found {len(matches)}."
            )

        # Assume the first matched branch is query (q), second is key (k).
        # This assumption is based on the default ordering in the exported graph,
        # since node naming conventions don't reliably indicate q/k branches.
        q_match, k_match = matches
        _process_complex_rope(
            graph,
            q_match,
            k_match,
            rope_flash_cache,
            rope_position_ids_cache,
        )

    gm = canonicalize_graph(gm)
    return gm


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

    # Verify that the unsqueeze is performed along dimension 2.
    if not (len(unsqueeze_node.args) >= 2 and unsqueeze_node.args[1] == 2):
        return None
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
    }


def _process_explicit_rope(
    graph: GraphModule,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
    start_boundary: Node,
    rope_flash_cache: DefaultDict[Any, Optional[Node]],
    rope_position_ids_cache: Dict[str, Node],
) -> None:
    """
    Process a region that matched the legacy RoPE pattern (v1).
    Inserts the custom op (flashinfer) and replaces the original add nodes.
    Precomputes positional IDs and the fused cosine-sine cache as explicit nodes,
    and reuses those nodes when possible.
    """
    q_node = q_match["raw_input"]
    k_node = k_match["raw_input"]
    cos_node = q_match["unsqueeze_cos"].args[0]
    sin_node = q_match["unsqueeze_sin"].args[0]

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

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

    # Infer input layout; default to [b, n, s, d] if inference fails.
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        need_transpose = isinstance(q_fake.shape[1], int)
        ad_logger.debug(
            f"Inferred RoPE input layout: [{'[b, n, s, d]' if need_transpose else '[b, s, n, d]'}]"
        )
        # Additional sanity check for the third dimension
        if need_transpose:
            if not isinstance(q_fake.shape[2], torch.SymInt):
                ad_logger.warning(
                    "Sanity check failed: q_fake.shape[2] should be symbolic. Defaulting to [b, n, s, d]"
                )
                need_transpose = True
        else:
            if not isinstance(q_fake.shape[1], torch.SymInt):
                ad_logger.warning(
                    "Sanity check failed: q_fake.shape[2] should be symbolic. Defaulting to [b, n, s, d]"
                )
                need_transpose = True
    else:
        ad_logger.warning("Unable to infer layout of q node. Defaulting to [b, n, s, d].")
        need_transpose = True

    with graph.inserting_before(q_match["add_node"]):
        if need_transpose:
            q_for_op = graph.call_function(torch.ops.aten.transpose, args=(q_node, 1, 2))
            k_for_op = graph.call_function(torch.ops.aten.transpose, args=(k_node, 1, 2))
            q_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(q_for_op,))
            k_for_op_contig = graph.call_function(torch.ops.aten.contiguous, args=(k_for_op,))
        else:
            q_for_op_contig, k_for_op_contig = q_node, k_node

        head_dim = cos_node.meta["val"].shape[-1]
        half_head_dim = head_dim // 2

    cache = rope_flash_cache
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
        with graph.inserting_after(fused_cos_sin):
            fused_cos_sin_0 = graph.call_function(operator.getitem, args=(fused_cos_sin, 0))
        with graph.inserting_after(fused_cos_sin_0):
            fused_cos_sin_to = graph.call_function(
                torch.ops.aten.to, args=(fused_cos_sin_0, torch.float32)
            )
        cache[cache_key] = fused_cos_sin_to

    with graph.inserting_before(q_match["add_node"]):
        position_ids = _get_position_ids(
            graph,
            q_for_op_contig,
            batch_dim=0,
            seq_dim=1,
            rope_position_ids_cache=rope_position_ids_cache,
        )

        flash_node = graph.call_function(
            torch.ops.rope.flashinfer,
            args=(q_for_op_contig, k_for_op_contig, position_ids, fused_cos_sin_to, True),
        )

    with graph.inserting_after(flash_node):
        raw_q = graph.call_function(operator.getitem, args=(flash_node, 0))
        raw_k = graph.call_function(operator.getitem, args=(flash_node, 1))

    if need_transpose:
        with graph.inserting_after(raw_q):
            new_q = graph.call_function(torch.ops.aten.transpose.int, args=(raw_q, 1, 2))
        with graph.inserting_after(raw_k):
            new_k = graph.call_function(torch.ops.aten.transpose.int, args=(raw_k, 1, 2))
    else:
        new_q, new_k = raw_q, raw_k

    new_q.meta["val"] = q_match["add_node"].meta.get("val", None)
    new_k.meta["val"] = k_match["add_node"].meta.get("val", None)

    q_match["add_node"].replace_all_uses_with(new_q)
    k_match["add_node"].replace_all_uses_with(new_k)


def _process_complex_rope(
    graph: GraphModule,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
    rope_flash_cache: DefaultDict[Any, Optional[Node]],
    rope_position_ids_cache: Dict[str, Node],
) -> None:
    """
    Process a region that matched the complex-multiplication RoPE pattern (v2).
    Inserts the custom op (flashinfer) after extracting frequency information,
    and replaces the original type_as nodes.
    Precomputes positional IDs and the fused cosine-sine cache as explicit nodes,
    and reuses those nodes when possible.
    """
    q_node = q_match["input"]
    k_node = k_match["input"]
    inv_freq_node = q_match["inv_freq"]

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    if inv_freq_node != k_match["inv_freq"]:
        raise RuntimeError("Mismatch of freqs_cis (inv_freq) between branches.")

    # Sanity check that input layout is BSND (no transpose needed).
    q_fake = q_node.meta.get("val", None)
    if q_fake is not None and len(q_fake.shape) > 2:
        if not (isinstance(q_fake.shape[1], torch.SymInt) and isinstance(q_fake.shape[2], int)):
            ad_logger.warning(
                f"""Sanity check failed: q_fake should have shape [b, s, n, d],
                s should be symbolic and n should be int, instead got shape {q_fake.shape}"""
            )
    else:
        ad_logger.warning(
            f"Sanity check failed: q_fake should be 3D or 4D, but got shape {q_fake.shape}"
        )

    # Retrieve or register the lookup table for inv_freq_node -> cos_sin_flash
    cache = rope_flash_cache
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
        with graph.inserting_after(cos_sin_flash_3d):
            cos_sin_flash = graph.call_function(operator.getitem, args=(cos_sin_flash_3d, 0))
        with graph.inserting_after(cos_sin_flash):
            cos_sin_flash = graph.call_function(
                torch.ops.aten.to, args=(cos_sin_flash, torch.float32)
            )
        cache[inv_freq_node] = cos_sin_flash

    with graph.inserting_before(q_match["out"]):
        position_ids = _get_position_ids(
            graph, q_node, batch_dim=0, seq_dim=1, rope_position_ids_cache=rope_position_ids_cache
        )
        flash_node = graph.call_function(
            torch.ops.rope.flashinfer,
            args=(q_node, k_node, position_ids, cos_sin_flash, False),
        )

    with graph.inserting_after(flash_node):
        raw_q = graph.call_function(operator.getitem, args=(flash_node, 0))
        raw_k = graph.call_function(operator.getitem, args=(flash_node, 1))

    raw_q.meta["val"] = q_match["out"].meta.get("val", None)
    raw_k.meta["val"] = k_match["out"].meta.get("val", None)

    q_match["out"].replace_all_uses_with(raw_q)
    k_match["out"].replace_all_uses_with(raw_k)


def _process_ds_rope(
    graph: GraphModule,
    q_match: Dict[str, Node],
    k_match: Dict[str, Node],
    rope_ds_cache: DefaultDict[Tuple[Node, Node, int], Tuple[Optional[Node], Optional[Node]]],
) -> None:
    """
    Replace a matched DS-style RoPE subgraph with a call to rope::apply_rope_ds.
    Cache the one-time unsqueeze of cos/sin.
    """
    q_node = q_match["raw_input"]
    k_node = k_match["raw_input"]
    cos_node = q_match["unsqueeze_cos"].args[0]
    sin_node = q_match["unsqueeze_sin"].args[0]

    # Sanity check on head_dim
    if not _validate_rope_inputs(q_node, k_node):
        return

    # Infer unsqueeze_dim from layout
    unsq_dim = 1
    fake = q_node.meta.get("val", None)
    if fake is not None and len(fake.shape) == 4:
        # if shape[1] symbolic, it's [B, S, N, D] => BSND -> head dim is 2
        if isinstance(fake.shape[1], torch.SymInt):
            unsq_dim = 2
        else:
            unsq_dim = 1

    # look up—or first insert—the unsqueezed cos/sin
    cache_key = (cos_node, sin_node, unsq_dim)
    cos_u, sin_u = rope_ds_cache[cache_key]
    if cos_u is None:
        with graph.inserting_before(q_match["add_node"]):
            cos_u = graph.call_function(torch.ops.aten.unsqueeze, args=(cos_node, unsq_dim))
            sin_u = graph.call_function(torch.ops.aten.unsqueeze, args=(sin_node, unsq_dim))
        rope_ds_cache[cache_key] = (cos_u, sin_u)

    with graph.inserting_before(q_match["add_node"]):
        ds_node = graph.call_function(
            torch.ops.rope.apply_rope_ds,
            args=(q_node, k_node, cos_u, sin_u),
        )

    with graph.inserting_after(ds_node):
        q_out = graph.call_function(operator.getitem, args=(ds_node, 0))
        k_out = graph.call_function(operator.getitem, args=(ds_node, 1))

    q_out.meta["val"] = q_match["add_node"].meta.get("val", None)
    k_out.meta["val"] = k_match["add_node"].meta.get("val", None)

    q_match["add_node"].replace_all_uses_with(q_out)
    k_match["add_node"].replace_all_uses_with(k_out)


def _validate_rope_inputs(q_node: Node, k_node: Node) -> bool:
    """
    Validates that:
    - The last dimension (head_dim) of both q and k is a multiple of 64.
    - The dtype of q and k is half precision (bfloat16 or float16).
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

    # Retrieve device information, ensuring it is a torch.device.
    device = q_node.meta.get("device", "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Build positions: arange(sym_seq) -> view -> expand -> flatten.
    positions_node = graph.call_function(
        torch.ops.aten.arange,
        args=(sym_seq,),
        kwargs={"dtype": torch.float32, "device": device, "pin_memory": False},
    )
    positions_node = graph.call_function(torch.ops.aten.view, args=(positions_node, (1, -1)))
    positions_node = graph.call_function(
        torch.ops.aten.expand, args=(positions_node, (sym_batch, -1))
    )
    position_ids = graph.call_function(torch.ops.aten.flatten, args=(positions_node,))
    rope_position_ids_cache["position_ids"] = position_ids
    return position_ids
