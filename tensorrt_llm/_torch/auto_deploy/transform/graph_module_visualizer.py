# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch GraphModule Visualization Tool

This module provides functionality to convert PyTorch GraphModule to Graphviz diagrams.
Supports different node styles and detailed graph annotations.

Key Features:
- Convert FX GraphModule to Graphviz diagrams
- Display tensor shape information on edges
- Adjust edge width based on tensor element count
- Intelligent port assignment for multi-input/output handling
- Color coding based on tensor identity

Usage Example:
    import torch
    import torch.fx as fx
    from graph_module_visualizer import to_dot

    # Trace model
    model = YourModel()
    traced = fx.symbolic_trace(model)

    # Generate visualization
    dot = to_dot(traced, format="svg", include_shapes=True)

Requirements: pip install graphviz
"""

import math
import re
from typing import Any, Dict, Optional

import graphviz
import torch
import torch.fx as fx
from torch.fx import GraphModule

from ..utils.logger import ad_logger


def _calculate_edge_width(val) -> float:
    """
    Calculate edge width based on tensor element count
    Formula: log10(num_elements) + 2

    Args:
        val: FakeTensor, Tensor, or any object with .shape attribute
    """
    min_width = 2.0
    max_width = 10.0
    if not hasattr(val, "shape"):
        return min_width  # Default width

    try:
        # Calculate total number of elements
        num_elements = 1
        for dim in val.shape:
            if isinstance(dim, int):
                num_elements *= dim
            else:
                num_elements *= 1

        if num_elements <= 0:
            return min_width

        # Calculate width: log10(element count) + 2
        width = math.log10(num_elements) + 2

        # Constrain width range (2.0 to 10.0)
        width = max(min_width, min(max_width, width))

        return width

    except (ValueError, TypeError):
        return min_width  # Use default width on error


def _get_edge_color(source_node_name: str, output_index: int) -> str:
    """
    Assign color based on source node and actual output index
    Ensures all edges of the same tensor use the same color
    """
    colors = [
        "#FF6B9D80",  # Pink
        "#4ECDC480",  # Mint green
        "#45B7D180",  # Sky blue
        "#96CEB480",  # Light green
        "#DDA0DD80",  # Light purple
        "#98D8C880",  # Teal
        "#BB8FCE80",  # Violet
        "#85C1E980",  # Light blue
        "#F8C47180",  # Peach
        "#82E0AA80",  # Mint
        "#F7DC6F80",  # Lemon yellow
        "#AED6F180",  # Light sky blue
    ]

    # Use combination of source_node + output_index
    # Ensures multiple edges of the same tensor have the same color
    tensor_id = f"{source_node_name}_{output_index}"
    color_index = hash(tensor_id) % len(colors)
    return colors[color_index]


def _get_port_of_five(input_idx, total_inputs):
    """
    0 xxxxxxxx  a = 7
    1 xxxxxxxx  b = 2
    2 xxxxxxx   x = 2 * 8 = 16
    3 xxxxxxx
    4 xxxxxxx
    """
    K = 5
    a, b = total_inputs // K, total_inputs % K
    x = b * (a + 1)
    if input_idx < x:
        return input_idx // (a + 1)
    else:
        return (input_idx - x) // a + b


def _get_input_port(input_index: int, total_inputs: int) -> str:
    """
    Get input port based on input index and total input count
    """
    if total_inputs <= 1:
        return "n"  # Single input uses default port
    elif total_inputs == 2:
        return "nw" if input_index == 0 else "ne"  # Northwest, northeast
    elif total_inputs == 3:
        ports = ["nw", "n", "ne"]  # Northwest, north, northeast
        return ports[input_index]
    elif total_inputs == 4:
        ports = ["nw", "n", "ne", "e"]
        return ports[input_index]
    else:
        # 5+ inputs: west, northwest, north, northeast, east in order
        ports = ["w", "nw", "n", "ne", "e"]
        # Cycle through ports for more than 5 inputs
        return ports[_get_port_of_five(input_index, total_inputs)]


def _get_output_port(output_index: int, total_outputs: int) -> str:
    """
    Get output port based on output index and total output count (symmetric to input, but on bottom)
    """
    if total_outputs <= 1:
        return "s"  # Single output uses default port
    elif total_outputs == 2:
        return "sw" if output_index == 0 else "se"  # Southwest, southeast
    elif total_outputs == 3:
        ports = ["sw", "s", "se"]  # Southwest, south, southeast
        return ports[output_index]
    else:
        # 4+ outputs: west, southwest, south, southeast, east in order
        ports = ["w", "sw", "s", "se", "e"]
        # Cycle through ports for more than 5 outputs
        return ports[_get_port_of_five(output_index, total_outputs)]


def to_dot(
    graph_module: GraphModule,
    name: str,
    save_path: str,
    format: str = "svg",
    include_shapes: bool = True,
) -> Optional["graphviz.Digraph"]:
    """
    Convert PyTorch GraphModule to Graphviz diagram

    Args:
        graph_module: GraphModule to visualize
        name: Name of the diagram
        save_path: Save path, if None uses name
        format: Output format ('png', 'pdf', 'svg', 'dot', etc.)
        include_shapes: Whether to include tensor shape information

    Returns:
        graphviz.Digraph object
    """
    # Create Graphviz diagram
    dot = graphviz.Digraph(
        name=name, comment=f"PyTorch GraphModule: {graph_module.__class__.__name__}", format=format
    )

    # Set graph attributes
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("node", shape="box", style="rounded,filled", height="0.2")
    # Remove default edge color, let each edge use its own color

    # Node style configuration
    node_styles = {
        "placeholder": {"fillcolor": "lightgreen", "shape": "box"},
        "get_attr": {"fillcolor": "lightcyan", "shape": "box"},
        "call_function": {"fillcolor": "lightblue", "shape": "box"},
        "call_method": {"fillcolor": "lightyellow", "shape": "box"},
        "call_module": {"fillcolor": "lightpink", "shape": "box"},
        "output": {"fillcolor": "lightcoral", "shape": "box"},
    }

    # Analyze graph structure
    graph = graph_module.graph
    nodes = list(graph.nodes)
    node_labels = {}

    # Process each node
    for node in nodes:
        # Get basic node information
        node_name = node.name
        op_type = node.op

        # Create node label
        label = _get_node_label(graph_module, node)
        node_labels[node_name] = label

        # Set node style
        style = node_styles.get(op_type, {"fillcolor": "white", "shape": "box"})

        # Add node to diagram
        node_attrs = {
            "label": label,
            "fillcolor": style["fillcolor"],
            "shape": style["shape"],
            "tooltip": node_name,  # Use node name as tooltip
        }
        # If the node has no value, set the fillcolor to red
        if "val" not in node.meta:
            node_attrs["fillcolor"] = "red"
        elif isinstance(node.meta["val"], torch.Tensor):
            node_attrs["label"] += "\n" + str(node.meta["val"].device)
        elif isinstance(node.meta["val"], (list, tuple)):
            for val in node.meta["val"]:
                if isinstance(val, torch.Tensor):
                    node_attrs["label"] += "\n" + str(val.device)
                else:
                    node_attrs["label"] += "\n" + str(val)

        dot.node(node_name, **node_attrs)

    # First collect all edge information
    edges = []  # Format: (source_node, target_node, val, source_output_index, target_input_index)
    node_inputs = {}  # Input list for each node: [(source_node_name, output_index)]
    node_outputs = {}  # Output list for each node: [(target_node_name, input_index)]

    # Initialize
    for node in nodes:
        node_inputs[node.name] = []
        node_outputs[node.name] = []

    # Collect edge information
    def _add_edge_from_node_with_index(input_idx, source_node: fx.Node, target_node: fx.Node):
        """Add edge from source_node to target_node, automatically determining correct output index"""
        # Extract val (FakeTensor or Tensor) from node.meta["val"]
        # This is more reliable than tensor_meta since FakeTensorProp always populates val
        val = None
        if include_shapes and hasattr(source_node, "meta") and "val" in source_node.meta:
            val = source_node.meta["val"]

        # Calculate indices
        source_output_index = _determine_output_index(source_node, target_node)

        # Add edge and update indices (store tuple containing node name and corresponding index)
        edges.append((source_node.name, target_node.name, val, source_output_index, input_idx))
        node_inputs[target_node.name].append((source_node.name, source_output_index))
        node_outputs[source_node.name].append((target_node.name, input_idx))

    def _determine_output_index(source_node: fx.Node, target_node: fx.Node) -> int:
        """Determine the output index for the edge from source_node to target_node"""
        # Check if target_node is a getitem operation
        if (
            target_node.op == "call_function"
            and hasattr(target_node.target, "__name__")
            and target_node.target.__name__ == "getitem"
            and len(target_node.args) >= 2
        ):
            # target_node.args[0] is source node, target_node.args[1] is index
            if target_node.args[0] == source_node and isinstance(target_node.args[1], int):
                return target_node.args[1]

        # By default, FX nodes have only one output
        return 0

    def _fix_problematic_negative_one(text):
        """Fix the problematic 9223372036854775807 number back to -1"""
        return text.replace("9223372036854775807", "-1")

    def _format_constant_label(value):
        """Format constant value for display as node label"""
        # Special case: direct problematic integer
        if isinstance(value, int) and value == 9223372036854775807:  # 2^63 - 1
            return "-1"

        # Handle different value types
        if isinstance(value, (int, float, bool, str)):
            result = str(value)
        elif hasattr(value, "shape"):
            # Tensor-like object with shape
            if hasattr(value, "numel") and value.numel() > 6:
                return f"shape={tuple(value.shape)}"
            else:
                result = str(value)
        elif isinstance(value, (list, tuple)):
            # Collection of elements
            if len(value) > 6:
                return f"length={len(value)}"
            else:
                result = str(value)
        else:
            # Other types
            result = str(value)[:50] + ("..." if len(str(value)) > 50 else "")

        # Apply the fix to any string representation
        return _fix_problematic_negative_one(result)

    # Store constants to be created later
    constants_to_create = []

    def _process_arg(input_idx, arg, target_node: fx.Node):
        """Process a single argument and create appropriate edges in the graph.

        Handles three types of arguments:
        - fx.Node: Creates an edge from the source node to the target node.
        - Container (list/tuple): Iterates through and creates edges for any fx.Node elements.
        - Constant: Creates a constant node and adds it to the graph with appropriate edges.

        Args:
            input_idx: The index of this argument in the target node's input list.
            arg: The argument to process. Can be an fx.Node, a container (list/tuple),
                or a constant value.
            target_node: The fx.Node that receives this argument as input.

        Note:
            This function modifies external state including `constants_to_create`,
            `node_inputs`, `node_outputs`, and `edges`.
        """
        if isinstance(arg, fx.Node):
            _add_edge_from_node_with_index(input_idx, arg, target_node)
        elif isinstance(arg, (list, tuple)):
            for sub_arg in arg:
                if isinstance(sub_arg, fx.Node):
                    _add_edge_from_node_with_index(input_idx, sub_arg, target_node)
        else:
            # This is a constant value - store for later creation
            const_node_name = f"const_{target_node.name}_{input_idx}"
            constants_to_create.append((const_node_name, arg, target_node.name, input_idx))

            # Add to node tracking immediately
            node_inputs[target_node.name].append(const_node_name)
            if const_node_name not in node_outputs:
                node_outputs[const_node_name] = []
            node_outputs[const_node_name].append((target_node.name, input_idx))

            # Create edge from constant to target
            edges.append((const_node_name, target_node.name, None, 0, input_idx))

    def _determine_node_output_count(node_name: str) -> int:
        """Determine the actual output count of a node (applies to all node types)"""
        # Check if any getitem operations use this node
        max_index = -1
        for n in nodes:
            # Only check getitem operation pattern, don't restrict source node op type
            if (
                n.op == "call_function"
                and hasattr(n.target, "__name__")
                and n.target.__name__ == "getitem"
                and len(n.args) >= 2
                and hasattr(n.args[0], "name")
                and n.args[0].name == node_name
                and isinstance(n.args[1], int)
            ):
                max_index = max(max_index, n.args[1])

        # If getitem operations found, output count is max_index + 1
        if max_index >= 0:
            return max_index + 1

        # By default, nodes have only one output
        return 1

    # Traverse all nodes to collect edges
    for node in nodes:
        if not node.args:
            continue

        for input_idx, arg in enumerate(node.args):
            _process_arg(input_idx, arg, node)

    # Print 10 nodes with most inputs
    # ad_logger.debug("Nodes with most inputs:")
    # node_inputs_sorted = sorted(node_inputs.items(), key=lambda x: len(x[1]), reverse=True)
    # for node_name, input_list in node_inputs_sorted[:10]:
    #     ad_logger.debug(f"  {node_name}: {len(input_list)}")

    # Print 10 nodes with most outputs
    node_outputs_sorted = sorted(node_outputs.items(), key=lambda x: len(x[1]), reverse=True)
    # ad_logger.debug("Nodes with most outputs:")
    large_fanout_nodes: Dict[str, int] = {}
    for node_name, output_list in node_outputs_sorted[:10]:
        if len(output_list) > 10:
            large_fanout_nodes[node_name] = 0
        # ad_logger.debug(f"  {node_name}: {len(output_list)}")

    # Overwrite large fanout nodes style
    for node_name in large_fanout_nodes:
        # Add node to diagram
        dot.node(
            node_name,
            label=node_name,
            fillcolor="#ffdddd80",
            color="#88888880",
            shape="box",
            style="filled,dashed,rounded",
            tooltip=node_name,  # Use node name as tooltip
        )

    node_inputs_sorted = sorted(node_inputs.items(), key=lambda x: len(x[1]), reverse=True)
    large_fanin_nodes: Dict[str, int] = {}
    for node_name, input_list in node_inputs_sorted[:10]:
        if len(input_list) > 12:
            large_fanin_nodes[node_name] = 0

    for node_name in large_fanin_nodes:
        # Add node to diagram
        dot.node(
            node_name,
            label=node_name,
            fillcolor="#ddffdd80",
            color="#88888880",
            shape="box",
            style="filled,dashed,rounded",
            tooltip=node_name,  # Use node name as tooltip
        )

    # Create constant nodes
    for const_node_name, const_value, target_name, input_idx in constants_to_create:
        const_label = _format_constant_label(const_value)

        # Add constant node to dot graph
        const_attrs = {
            "label": const_label,
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#ffffcc",  # Light yellow background
            "color": "#cccccc",  # Light gray border
            "width": "0.2",
            "fontsize": "10",
        }
        dot.node(const_node_name, **const_attrs)

    # Add edges with ports and colors
    for source_name, target_name, val, source_output_index, target_input_index in edges:
        edge_attrs = {}

        # Calculate ports (for graphical display positioning)
        input_list = node_inputs[target_name]

        # Use actual output count, not usage count
        source_output_count = _determine_node_output_count(source_name)

        input_port = _get_input_port(target_input_index, len(input_list))
        output_port = _get_output_port(source_output_index, source_output_count)

        # Build node names with ports
        source_name_port = f"{source_name}:{output_port}" if output_port else source_name
        target_name_port = f"{target_name}:{input_port}" if input_port else target_name

        # Set edge color (based on actual output_index)
        edge_color = _get_edge_color(source_name, source_output_index)
        edge_attrs["color"] = edge_color

        # Add tensor shape and width information
        # val can be FakeTensor, Tensor, or other types with .shape attribute
        if val is not None and include_shapes and hasattr(val, "shape"):
            shape_str = str(tuple(val.shape))
            # Add dtype if available
            dtype_str = ""
            if hasattr(val, "dtype"):
                dtype_str = str(val.dtype).replace("torch.", "")
            edge_attrs["xlabel"] = f"{shape_str}\n{dtype_str}" if dtype_str else shape_str
            edge_attrs["fontsize"] = "10"
            edge_attrs["fontcolor"] = "blue"

            # Calculate edge width based on element count
            width = _calculate_edge_width(val)
            edge_attrs["penwidth"] = str(width)

        # For those large fanout nodes, large fantout nodes will stuck the graphviz layout algorithm.
        # So we need to duplicate the node, so each edge has its own source node.
        # Make the layout algorithm work.
        # So is large fanin nodes.
        if source_name in large_fanout_nodes:
            node_attrs = {
                "fillcolor": "#ddffdd80",
                "color": "#88888880",
                "shape": "box",
                "style": "filled,dashed,rounded",
            }
            large_fanout_nodes[source_name] += 1
            source_name_port = source_name + f"___{large_fanout_nodes[source_name]}"
            dot.node(
                name=source_name_port,
                label=source_name,
                **node_attrs,
            )

        if target_name in large_fanin_nodes:
            node_attrs = {
                "fillcolor": "#ddffdd80",
                "color": "#88888880",
                "shape": "box",
                "style": "filled,dashed,rounded",
            }
            large_fanin_nodes[target_name] += 1
            target_name_port = target_name + f"___{large_fanin_nodes[target_name]}"
            dot.node(name=target_name_port, label=target_name, **node_attrs)

        dot.edge(source_name_port, target_name_port, **edge_attrs)

    # Save diagram
    try:
        dot.render(save_path, cleanup=True)
        ad_logger.info(f"Diagram saved: {save_path}.{format}")
        with open(save_path + ".txt", "w") as f:
            f.write(str(graph_module.graph))
    except Exception as e:
        ad_logger.error(f"Failed to save diagram: {e}")

    return dot


def analyze_graph_structure(graph_module: GraphModule) -> Dict[str, Any]:
    """
    Analyze structural statistics of GraphModule

    Args:
        graph_module: GraphModule to analyze

    Returns:
        Dictionary containing structural statistics
    """
    graph = graph_module.graph
    nodes = list(graph.nodes)

    # Count node types
    op_counts = {}
    for node in nodes:
        op_type = node.op
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    # Analyze connections
    total_connections = 0
    for node in nodes:
        if node.args:
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    total_connections += 1
                elif isinstance(arg, (list, tuple)):
                    for sub_arg in arg:
                        if isinstance(sub_arg, fx.Node):
                            total_connections += 1

    # Calculate graph complexity
    complexity_score = len(nodes) + total_connections

    return {
        "total_nodes": len(nodes),
        "node_types": op_counts,
        "total_connections": total_connections,
        "complexity_score": complexity_score,
        "graph_depth": _calculate_graph_depth(nodes),
    }


def _get_node_label(graph_module: GraphModule, node: fx.Node) -> str:
    """Get node label"""
    if node.op == "call_function":
        func_name = _get_function_name(node.target)
        tokens = func_name.split(".")
        assert len(tokens) <= 2, f"Function name {func_name} has more than 2 tokens"
        label = tokens[0] if tokens[0] != "to" else func_name
    elif node.op == "call_method":
        label = str(node.target)
    elif node.op == "call_module":
        label = _get_module_name(graph_module, node.target)
    elif node.op == "get_attr":
        attr_name = str(node.target).split(".")[-1] if "." in str(node.target) else str(node.target)
        label = attr_name
    elif node.op == "placeholder":
        label = "ph: " + str(node.name)
    elif node.op == "output":
        label = "out: " + str(node.name)
    else:
        label = node.op
    return label


def _get_function_name(func) -> str:
    """Get simplified function name"""
    if hasattr(func, "__name__"):
        return func.__name__

    func_str = str(func)

    # Handle torch functions
    if "torch." in func_str:
        match = re.search(r"torch\.(\w+)\.(\w+)", func_str)
        if match:
            return f"{match.group(1)}.{match.group(2)}"

        match = re.search(r"torch\.(\w+)", func_str)
        if match:
            return match.group(1)

    # Handle built-in functions
    if "built-in" in func_str:
        match = re.search(r"'(\w+)'", func_str)
        if match:
            return match.group(1)

    return str(func).split(".")[-1] if "." in str(func) else str(func)


def _get_module_name(graph_module: GraphModule, target) -> str:
    """Extract module name, handle numeric indices in Sequential"""
    try:
        # Try to get actual module type name
        actual_module = graph_module.get_submodule(str(target))
        module_type = actual_module.__class__.__name__

        # Extract the last part of module name
        module_name = str(target).split(".")[-1] if "." in str(target) else str(target)

        # If it's numeric index (like modules in Sequential), show type name
        if module_name.isdigit():
            return module_type
        else:
            return module_name
    except Exception:
        # If unable to get module, fall back to original logic
        module_name = str(target).split(".")[-1] if "." in str(target) else str(target)
        return module_name


def _calculate_graph_depth(nodes) -> int:
    """Calculate maximum depth of the graph"""
    # Build dependency relationships
    dependencies = {}
    for node in nodes:
        dependencies[node.name] = []
        if node.args:
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    dependencies[node.name].append(arg.name)
                elif isinstance(arg, (list, tuple)):
                    for sub_arg in arg:
                        if isinstance(sub_arg, fx.Node):
                            dependencies[node.name].append(sub_arg.name)

    # Calculate depth of each node
    depths = {}

    def calculate_depth(node_name):
        if node_name in depths:
            return depths[node_name]

        if not dependencies[node_name]:
            depths[node_name] = 0
            return 0

        max_dep_depth = max(calculate_depth(dep) for dep in dependencies[node_name])
        depths[node_name] = max_dep_depth + 1
        return depths[node_name]

    for node in nodes:
        calculate_depth(node.name)

    return max(depths.values()) if depths else 0
