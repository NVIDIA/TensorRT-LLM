"""Pattern matching for detecting repeat_kv pattern from Huggingface models."""

from typing import Dict, Optional, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import AttentionDescriptor
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from .._graph import canonicalize_graph


def match_repeat_kv(gm: GraphModule) -> GraphModule:
    """
    Match and replace the repeat_kv pattern in fx graphs.

    The pattern is:
    unsqueeze -> expand -> reshape -> [optional] contiguous

    This is replaced with torch.ops.auto_deploy.torch_attention_repeat_kv.
    """
    graph = gm.graph

    num_kv_patterns = 0

    # Iterate through nodes in the graph
    for node in list(graph.nodes):
        # Look for reshape nodes that could be the end of our pattern
        if is_op(node, torch.ops.aten.reshape):
            match_info = _match_repeat_kv_pattern(node)
            if match_info:
                ad_logger.debug(f"Found repeat_kv pattern at {node}")
                _replace_with_repeat_kv(graph, match_info)
                num_kv_patterns += 1

    # Clean up the graph if we made any replacements
    if num_kv_patterns:
        gm = canonicalize_graph(gm)
    ad_logger.info(f"Found {num_kv_patterns} repeat_kv patterns")

    return gm


def match_eager_attention(gm: GraphModule) -> GraphModule:
    """
    Match and replace the eager attention pattern in fx graphs.

    The pattern is:
    transpose -> matmul -> mul -> (optional) add -> softmax -> to -> dropout -> matmul

    This is replaced with torch.ops.auto_deploy.torch_attention_sdpa.
    """
    graph = gm.graph

    # Track replacements to avoid processing nodes multiple times
    num_eager_patterns = 0

    # Iterate through nodes in the graph
    for node in list(graph.nodes):
        # Look for the final matmul nodes that could be part of our pattern
        if is_op(node, torch.ops.aten.matmul):
            match_info = _match_eager_attention_pattern(node)
            if match_info:
                ad_logger.debug(f"Found eager attention pattern at {node}")
                _replace_with_sdpa(graph, match_info)
                num_eager_patterns += 1

    # Clean up the graph if we made any replacements
    if num_eager_patterns:
        gm = canonicalize_graph(gm)
    ad_logger.info(f"Found {num_eager_patterns} eager attention patterns")
    return gm


def match_grouped_attention(gm: GraphModule) -> GraphModule:
    """
    Match and replace the grouped attention pattern in fx graphs.

    The pattern is:
    repeat_kv(k, n_rep) ->
    repeat_kv(v, n_rep) ->
    sdpa(q, repeated_k, repeated_v)

    This is replaced with torch.ops.auto_deploy.torch_attention_grouped_sdpa.
    """
    graph = gm.graph

    # Track replacements to avoid processing nodes multiple times
    num_grouped_patterns = 0

    # Iterate through nodes in the graph
    for node in list(graph.nodes):
        # Look for SDPA nodes that could be part of our pattern
        if is_op(node, torch.ops.auto_deploy.torch_attention_sdpa):
            match_info = _match_grouped_attention_pattern(node)
            if match_info:
                ad_logger.debug(f"Found grouped attention pattern at {node}")
                _replace_with_grouped_sdpa(graph, match_info)
                num_grouped_patterns += 1

    # Clean up the graph if we made any replacements
    if num_grouped_patterns:
        gm = canonicalize_graph(gm)
    ad_logger.info(f"Found {num_grouped_patterns} grouped attention patterns")
    return gm


def match_causal_attn_mask(gm: GraphModule) -> GraphModule:
    """
    Match attention operations with causal attention masks and optimize them.

    For operations that use explicit causal masks, this replaces:
    - sdpa(q, k, v, causal_mask, dropout_p, False, scale)
    with:
    - sdpa(q, k, v, None, dropout_p, True, scale)

    This optimization enables more efficient implementations on supported backends.
    """
    graph = gm.graph

    # Track replacements to avoid processing nodes multiple times
    num_causal_patterns = 0

    # Iterate through nodes in the graph
    for node in list(graph.nodes):
        # Look for SDPA nodes or grouped SDPA nodes
        if not (
            is_op(node, torch.ops.auto_deploy.torch_attention_sdpa)
            or is_op(node, torch.ops.auto_deploy.torch_attention_grouped_sdpa)
        ):
            continue

        # Get the attention mask argument (4th argument)
        if len(node.args) < 4 or node.args[3] is None:
            continue

        attn_mask = node.args[3]

        # Check if this mask is a causal mask
        if not _is_causal_mask(attn_mask):
            ad_logger.debug(f"Found non-causal attention mask at {node=}!")
            continue

        ad_logger.debug(f"Found causal attention mask at {node}")

        # construct the new args list with args provided to the node and the default values otherwise
        new_args = []
        for idx, arg in enumerate(node.target._schema.arguments):
            # In case arg is provided to the node, use it
            if idx < len(node.args):
                new_args.append(node.args[idx])
            # In case arg is not provided to the node, use the default value
            elif arg.has_default_value:
                new_args.append(arg.default_value)
            else:
                raise ValueError(f"Missing required argument: {arg.name}")

        # Create new arguments with None mask and is_causal=True
        new_args[3] = None  # Set mask to None
        new_args[5] = True  # Set is_causal to True

        # Create new node with updated arguments
        with graph.inserting_before(node):
            new_node = graph.call_function(node.target, args=tuple(new_args), kwargs=node.kwargs)

        # Preserve metadata
        new_node.meta = node.meta.copy()

        # Replace the old node with the new one
        node.replace_all_uses_with(new_node)

        num_causal_patterns += 1

    # Clean up the graph if we made any replacements
    if num_causal_patterns:
        gm = canonicalize_graph(gm)
    ad_logger.info(f"Found {num_causal_patterns} causal mask attention patterns")
    return gm


def _match_repeat_kv_pattern(reshape_node: Node) -> Optional[Dict[str, Node]]:
    """
    Match the repeat_kv pattern starting from a reshape node.

    The pattern is:
    unsqueeze -> expand -> reshape -> [optional] contiguous

    Returns a dictionary with information about the match or None if no match.
    """
    # Check that reshape_node is a reshape operation
    if not is_op(reshape_node, torch.ops.aten.reshape):
        return None

    # The reshape should have expand as its first argument
    if len(reshape_node.args) < 1:
        return None

    expand_node = reshape_node.args[0]
    if not is_op(expand_node, torch.ops.aten.expand):
        return None

    # The expand should have unsqueeze as its first argument
    if len(expand_node.args) < 1:
        return None

    unsqueeze_node = expand_node.args[0]
    if not is_op(unsqueeze_node, torch.ops.aten.unsqueeze):
        return None

    # The unsqueeze should be inserting a dimension at position 2
    if len(unsqueeze_node.args) < 2 or unsqueeze_node.args[1] != 2:
        return None

    # Get the input tensor to unsqueeze
    if len(unsqueeze_node.args) < 1:
        return None

    input_tensor = unsqueeze_node.args[0]

    # Check input dimensions - should be 4D (batch, num_key_value_heads, seq_len, head_dim)
    input_val = input_tensor.meta.get("val", None)
    if input_val is None or len(input_val.shape) != 4:
        return None

    # Extract batch size, num_kv_heads, seq_len, and head_dim from the input tensor shape
    batch_size, num_kv_heads, seq_len, head_dim = input_val.shape

    # Check reshape args
    if len(reshape_node.args) < 2 or not isinstance(reshape_node.args[1], list):
        return None

    reshape_args = reshape_node.args[1]
    if len(reshape_args) != 4:
        return None

    # Check expand args
    if len(expand_node.args) < 2 or not isinstance(expand_node.args[1], list):
        return None

    expand_args = expand_node.args[1]
    if len(expand_args) != 5:
        return None

    # Determine n_rep by comparing the output and input head dimensions
    # In the expand args, we should have [batch, num_kv_heads, n_rep, seq_len, head_dim]
    # In the reshape args, we should have [batch, num_heads, seq_len, head_dim]
    # where num_heads = num_kv_heads * n_rep
    _, _, n_rep, _, _ = expand_args
    _, reshape_num_heads, _, _ = reshape_args

    # Check that n_rep is an integer
    if not isinstance(n_rep, int):
        return None

    # Check that num_heads = num_kv_heads * n_rep
    # This may be a symbolic expression, so we need to compare with caution
    reshape_out_val = reshape_node.meta.get("val", None)
    if reshape_out_val is None or len(reshape_out_val.shape) != 4:
        return None

    # Ensure output shape is correct
    out_batch, out_heads, out_seq, out_dim = reshape_out_val.shape

    # Check that input batch and seq dimensions match output
    if out_batch != batch_size or out_seq != seq_len or out_dim != head_dim:
        return None

    # Check if reshape is followed by a contiguous node
    contiguous_node = None
    users = list(reshape_node.users)

    # Only consider contiguous if reshape has exactly one user
    if len(users) == 1 and is_op(users[0], torch.ops.aten.contiguous):
        contiguous_node = users[0]

    result = {
        "input_tensor": input_tensor,
        "unsqueeze_node": unsqueeze_node,
        "expand_node": expand_node,
        "reshape_node": reshape_node,
        "n_rep": n_rep,
    }

    if contiguous_node:
        result["contiguous_node"] = contiguous_node

    return result


def _match_eager_attention_pattern(final_matmul_node: Node) -> Optional[Dict[str, Node]]:
    """
    Match the eager attention pattern starting from the final matmul node.

    The pattern is:
    transpose -> matmul -> mul/div -> (optional) add -> (optional) to -> softmax -> (optional) to -> dropout -> matmul

    Returns a dictionary with information about the match or None if no match.
    """
    # Check that final_matmul_node is a matmul operation
    if not is_op(final_matmul_node, torch.ops.aten.matmul):
        return None

    # Check we have two arguments
    if len(final_matmul_node.args) < 2:
        return None

    # The first arg of final matmul should be dropout
    dropout_node = final_matmul_node.args[0]
    if not is_op(dropout_node, torch.ops.aten.dropout):
        return None

    # The second arg of final matmul is the value tensor (possibly repeated/transformed)
    value = final_matmul_node.args[1]

    # The dropout should have a to_dtype node (or directly softmax) as input
    if len(dropout_node.args) < 1:
        return None

    # Allow optional to_dtype node after softmax
    to_dtype_after_softmax = dropout_node.args[0]
    if is_op(to_dtype_after_softmax, torch.ops.aten.to):
        if len(to_dtype_after_softmax.args) < 1:
            return None
        softmax_node = to_dtype_after_softmax.args[0]
    else:
        softmax_node = to_dtype_after_softmax

    # Now we should have a softmax node
    if not is_op(softmax_node, torch.ops.aten.softmax):
        return None

    # The softmax should have dim=-1 (may be specified in different ways)
    if len(softmax_node.args) < 2 or (
        isinstance(softmax_node.args[1], int) and softmax_node.args[1] != -1
    ):
        # Check kwargs if not in args
        if softmax_node.kwargs.get("dim", -1) != -1:
            return None

    # The softmax node's input can be:
    # - direct from add/mul/div
    # - or through a to_dtype node (like to_35 in the example)
    if len(softmax_node.args) < 1:
        return None

    # Handle optional to_dtype node before softmax
    prev_node = softmax_node.args[0]
    if is_op(prev_node, torch.ops.aten.to):
        if len(prev_node.args) < 1:
            return None
        prev_node = prev_node.args[0]

    # Check for attention mask pattern (add node)
    if is_op(prev_node, torch.ops.aten.add):
        add_node = prev_node
        attn_mask = add_node.args[1]  # Second arg is the mask

        # The add should have a mul or div node as its first argument
        if len(add_node.args) < 1:
            return None

        scaling_node = add_node.args[0]
        if not (is_op(scaling_node, torch.ops.aten.mul) or is_op(scaling_node, torch.ops.aten.div)):
            return None
    elif is_op(prev_node, torch.ops.aten.mul) or is_op(prev_node, torch.ops.aten.div):
        # No mask case - the softmax input is directly the mul or div node
        scaling_node = prev_node
        attn_mask = None
    else:
        return None

    # Check the scaling operation and extract the scaling factor
    is_division = is_op(scaling_node, torch.ops.aten.div)

    # The mul/div node should have a matmul node as input
    if len(scaling_node.args) < 2:
        return None

    # Extract the scaling factor, adjusting for division vs multiplication
    scale = scaling_node.args[1]
    # Allow for constant or tensor scale
    if not isinstance(scale, (float, int, Node)):
        return None

    # For division, we need to invert the scaling factor if it's a constant
    if is_division and isinstance(scale, (float, int)):
        scale = 1.0 / scale

    first_matmul_node = scaling_node.args[0]
    if not is_op(first_matmul_node, torch.ops.aten.matmul):
        return None

    # The first matmul should have the query and key transpose as inputs
    if len(first_matmul_node.args) < 2:
        return None

    query = first_matmul_node.args[0]
    transpose_key = first_matmul_node.args[1]

    # Check for transpose, could be any dimensions
    if not is_op(transpose_key, torch.ops.aten.transpose):
        return None

    # The transpose should have the key as input
    if len(transpose_key.args) < 1:
        return None

    key = transpose_key.args[0]

    # Create the match info dictionary
    match_info = {
        "query": query,
        "key": key,
        "value": value,
        "scale": scale,
        "dropout_p": dropout_node.args[1] if len(dropout_node.args) > 1 else 0.0,
        "final_matmul": final_matmul_node,
    }

    # Add the attention mask if it exists
    if attn_mask is not None:
        match_info["attn_mask"] = attn_mask

    return match_info


def _match_grouped_attention_pattern(sdpa_node: Node) -> Optional[Dict[str, Node]]:
    """
    Match the grouped attention pattern starting from an SDPA node.

    The pattern is:
    repeat_kv(k, n_rep) ->
    repeat_kv(v, n_rep) ->
    sdpa(q, repeated_k, repeated_v)

    Returns a dictionary with information about the match or None if no match.
    """
    # Check that sdpa_node is an SDPA operation
    if not is_op(sdpa_node, torch.ops.auto_deploy.torch_attention_sdpa):
        return None

    # SDPA should have query, key, value as its first three arguments
    if len(sdpa_node.args) < 3:
        return None

    query, key_repeated, value_repeated = sdpa_node.args[0:3]

    # Key and value should come from repeat_kv operations
    if not is_op(key_repeated, torch.ops.auto_deploy.torch_attention_repeat_kv) or not is_op(
        value_repeated, torch.ops.auto_deploy.torch_attention_repeat_kv
    ):
        return None

    # Extract the original key, value, and n_rep
    orig_key = key_repeated.args[0]
    orig_value = value_repeated.args[0]
    key_n_rep = key_repeated.args[1]
    value_n_rep = value_repeated.args[1]

    # Both repeat_kv operations should have the same n_rep
    if key_n_rep != value_n_rep:
        return None

    # Return the match information
    return {
        "query": query,
        "key": orig_key,
        "value": orig_value,
        "key_repeated": key_repeated,
        "value_repeated": value_repeated,
        "n_rep": key_n_rep,
        "sdpa_node": sdpa_node,
    }


def _replace_with_repeat_kv(graph, match_info: Dict[str, Node]) -> None:
    """
    Replace the matched repeat_kv pattern with the custom op.
    """
    input_tensor = match_info["input_tensor"]
    reshape_node = match_info["reshape_node"]
    n_rep = match_info["n_rep"]

    # Determine the node to replace (either reshape or contiguous if present)
    node_to_replace = match_info.get("contiguous_node", reshape_node)

    with graph.inserting_before(node_to_replace):
        repeat_kv_node = graph.call_function(
            torch.ops.auto_deploy.torch_attention_repeat_kv, args=(input_tensor, n_rep)
        )

    # Preserve metadata from the original node
    repeat_kv_node.meta = node_to_replace.meta.copy()

    # Replace all uses of the node with the repeat_kv node
    node_to_replace.replace_all_uses_with(repeat_kv_node)


def _replace_with_sdpa(graph, match_info: Dict[str, Node]) -> None:
    """
    Replace the matched eager attention pattern with scaled_dot_product_attention.
    """
    # retrieve the default op for scaled_dot_product_attention
    sdpa_op = torch.ops.auto_deploy.torch_attention_sdpa.default

    # construct the args for the ops based on the match_info and the op's schema
    args = []
    for arg in sdpa_op._schema.arguments:
        if arg.name in match_info:
            args.append(match_info[arg.name])
        elif arg.has_default_value:
            args.append(arg.default_value)
        else:
            raise ValueError(f"Missing required argument: {arg.name}")
    args = tuple(args)

    # retrieve the final matmul node to know where to insert the sdpa node
    final_matmul = match_info["final_matmul"]

    with graph.inserting_before(final_matmul):
        sdpa_node = graph.call_function(sdpa_op, args=args)

    # Preserve metadata from the original node
    sdpa_node.meta = final_matmul.meta.copy()

    # Replace all uses of the final matmul node with the sdpa node
    final_matmul.replace_all_uses_with(sdpa_node)


def _replace_with_grouped_sdpa(graph, match_info: Dict[str, Node]) -> None:
    """
    Replace the matched grouped attention pattern with torch.ops.auto_deploy.torch_attention_grouped_sdpa.
    """
    sdpa_node = match_info["sdpa_node"]
    query = match_info["query"]
    key = match_info["key"]
    value = match_info["value"]

    # Construct the new args and kwargs
    args = (query, key, value) + sdpa_node.args[3:]
    kwargs = sdpa_node.kwargs.copy()

    with graph.inserting_before(sdpa_node):
        grouped_sdpa_node = graph.call_function(
            torch.ops.auto_deploy.torch_attention_grouped_sdpa.default, args=args, kwargs=kwargs
        )

    # Preserve metadata from the original node
    grouped_sdpa_node.meta = sdpa_node.meta.copy()

    # Replace all uses of the SDPA node with the grouped_sdpa node
    sdpa_node.replace_all_uses_with(grouped_sdpa_node)


def _is_causal_mask(mask_node: Node) -> bool:
    """
    Determine if a node represents a causal attention mask.

    Causal masks typically involve:
    1. Creating a matrix with very negative values (e.g., -inf or close to it)
    2. Using triu with offset 1 to create an upper triangular matrix
    3. Usually involves comparison operations (gt, lt) with position indices

    Returns True if the node appears to be a causal mask pattern.
    """
    # Direct pattern from the test case: masked_fill with triu(ones,1) and -inf
    if is_op(mask_node, torch.ops.aten.masked_fill):
        mask_args = mask_node.args
        if len(mask_args) >= 2:
            _ = mask_args[0]  # zero tensor
            mask_tensor = mask_args[1]
            fill_value = mask_args[2] if len(mask_args) > 2 else mask_node.kwargs.get("value", None)

            # Check if fill value is very negative (e.g., -inf)
            if fill_value is not None and (
                fill_value == float("-inf")
                or (isinstance(fill_value, (int, float)) and fill_value < -1e4)
            ):
                # Try to trace back to find a triu pattern
                if _has_triu_ancestor(mask_tensor, offset=1):
                    return True

    # Pattern from negative_fill test case: masked_fill with ~triu(ones,1) and 0.0
    # The negative_fill pattern has a pre-filled tensor with very negative values
    # and zeros in the lower triangle
    if is_op(mask_node, torch.ops.aten.masked_fill):
        mask_args = mask_node.args
        if len(mask_args) >= 2:
            negative_tensor = mask_args[0]
            mask_tensor = mask_args[1]
            fill_value = mask_args[2] if len(mask_args) > 2 else mask_node.kwargs.get("value", None)

            # Check if fill value is zero and the tensor is pre-filled with negative values
            if fill_value == 0.0 or fill_value == 0:
                # Check for the full tensor with negative values
                if is_op(negative_tensor, torch.ops.aten.full):
                    fill_args = negative_tensor.args
                    if (
                        len(fill_args) > 1
                        and isinstance(fill_args[1], (int, float))
                        and fill_args[1] < -1e4
                    ):
                        # This is likely a negative-filled tensor
                        # Now check if the mask is a bitwise_not of triu
                        if is_op(mask_tensor, torch.ops.aten.bitwise_not):
                            if len(mask_tensor.args) > 0 and _has_triu_ancestor(
                                mask_tensor.args[0], offset=1
                            ):
                                return True

    # Pattern for llama-3.1 style causal mask: slice of expand(unsqueeze(unsqueeze(mul_(triu, gt))))
    if is_op(mask_node, torch.ops.aten.slice):
        # Follow the chain backward to the source of the slice
        if len(mask_node.args) == 0:
            return False
        slice_source = mask_node.args[0]

        # Check for typical expand pattern
        if not (slice_source and is_op(slice_source, torch.ops.aten.expand)):
            return False

        # Continue tracing back through the pattern
        if len(slice_source.args) == 0:
            return False
        expand_source = slice_source.args[0]

        # Check for first unsqueeze operation
        if not (expand_source and is_op(expand_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of first unsqueeze
        if len(expand_source.args) == 0:
            return False
        first_unsqueeze_source = expand_source.args[0]

        # Check for second unsqueeze operation
        if not (first_unsqueeze_source and is_op(first_unsqueeze_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of the second unsqueeze
        if len(first_unsqueeze_source.args) == 0:
            return False
        second_unsqueeze_source = first_unsqueeze_source.args[0]

        # Check for mul_ operation
        if is_op(second_unsqueeze_source, torch.ops.aten.mul_):
            # Check if one of the mul_ arguments is a triu operation
            has_triu = False
            for arg in second_unsqueeze_source.args:
                if is_op(arg, torch.ops.aten.triu):
                    if len(arg.args) > 1 and arg.args[1] == 1:
                        has_triu = True
                        break

            if has_triu:
                # Check if one of the mul_ arguments involves a full tensor with negative values
                for arg in second_unsqueeze_source.args:
                    if is_op(arg, torch.ops.aten.full):
                        if (
                            len(arg.args) > 1
                            and isinstance(arg.args[1], (int, float))
                            and arg.args[1] < -1e4
                        ):
                            return True

            return has_triu

    # Original implementation for backward compatibility
    if is_op(mask_node, torch.ops.aten.slice):
        # Follow the chain backward to the source of the slice
        if len(mask_node.args) == 0:
            return False
        slice_source = mask_node.args[0]

        # Check for typical expand pattern
        if not (slice_source and is_op(slice_source, torch.ops.aten.expand)):
            return False

        # Continue tracing back through the pattern
        if len(slice_source.args) == 0:
            return False
        expand_source = slice_source.args[0]

        # Check for unsqueeze operations
        if not (expand_source and is_op(expand_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of the unsqueeze
        if len(expand_source.args) == 0:
            return False
        unsqueeze_source = expand_source.args[0]

        if not unsqueeze_source:
            return False

        # Check for triu pattern which is common in causal masks
        if is_op(unsqueeze_source, torch.ops.aten.mul_):
            for arg in unsqueeze_source.args:
                if not is_op(arg, torch.ops.aten.triu):
                    continue

                if len(arg.args) <= 1:
                    continue

                triu_offset = arg.args[1]
                # Causal masks typically use triu with offset 1
                if triu_offset == 1:
                    return True

            return False

        # Check if we have a full tensor filled with a very negative number
        if not is_op(unsqueeze_source, torch.ops.aten.full):
            return False

        if len(unsqueeze_source.args) <= 1:
            return False

        fill_value = unsqueeze_source.args[1]
        # Check if the fill value is very negative (likely -inf or close)
        if isinstance(fill_value, float) and fill_value < -1e10:
            return True

    # If we can't definitively identify it as causal, return False
    return False


def _has_triu_ancestor(node: Node, offset: int = 1, depth: int = 0, max_depth: int = 5) -> bool:
    """Helper function to find a triu operation in the ancestry of a node."""
    if depth > max_depth:  # Prevent infinite recursion
        return False

    if is_op(node, torch.ops.aten.triu):
        if len(node.args) > 1 and node.args[1] == offset:
            return True

    # Check if any of the arguments has a triu ancestor
    for arg in node.args:
        if isinstance(arg, Node) and _has_triu_ancestor(arg, offset, depth + 1, max_depth):
            return True

    # Check if any of the kwargs has a triu ancestor
    for value in node.kwargs.values():
        if isinstance(value, Node) and _has_triu_ancestor(value, offset, depth + 1, max_depth):
            return True

    return False


def match_attention_layout(gm: GraphModule, attention_op: Type[AttentionDescriptor]) -> GraphModule:
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
        gm = canonicalize_graph(gm)
        ad_logger.debug(f"Transformed graph for bsnd layout: {gm}")

    ad_logger.info(f"Found and matched {num_bsnd_patterns} attention layouts")

    return gm
