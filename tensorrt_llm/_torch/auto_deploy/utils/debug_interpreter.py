"""Debug Interpreter for FX GraphModule execution with capture and stop support.

This module provides a DebugInterpreter class that extends torch.fx.Interpreter
to support:
- Capturing intermediate tensor values at specified nodes
- Stopping execution at a specified node
- Useful for debugging and comparing graph execution
"""

from typing import Any, Dict, List, Optional, Set, Union

import torch
from torch.fx import GraphModule, Interpreter, Node


class DebugInterpreter(Interpreter):
    """An Interpreter that can capture intermediate values and stop at specific nodes.

    This is useful for:
    - Capturing outputs at module boundaries for comparison with HF
    - Binary search within a graph to find divergence points
    - Debugging graph execution step by step

    Example:
        ```python
        # Capture specific nodes
        interp = DebugInterpreter(gm, capture_nodes={"node_a", "node_b"})
        output = interp.run(input_ids)
        captured = interp.captured  # {"node_a": tensor(...), "node_b": tensor(...)}

        # Stop at a specific node
        interp = DebugInterpreter(gm, stop_at_node="node_c")
        partial_output = interp.run(input_ids)  # Returns output of node_c
        ```
    """

    def __init__(
        self,
        gm: GraphModule,
        capture_nodes: Optional[Union[Set[str], List[str]]] = None,
        stop_at_node: Optional[str] = None,
        garbage_collect_values: bool = True,
    ):
        """Initialize the DebugInterpreter.

        Args:
            gm: The GraphModule to interpret
            capture_nodes: Set of node names whose outputs should be captured
            stop_at_node: If set, execution stops after this node and returns its output
            garbage_collect_values: Whether to garbage collect intermediate values
                                   (set to False if you need to inspect all values)
        """
        super().__init__(gm, garbage_collect_values=garbage_collect_values)
        self.capture_nodes: Set[str] = set(capture_nodes) if capture_nodes else set()
        self.stop_at_node = stop_at_node
        self.captured: Dict[str, Any] = {}
        self.stopped = False
        self.stop_output = None

    def run(self, *args, **kwargs) -> Any:
        """Run the interpreter, possibly stopping early if stop_at_node is set.

        Returns:
            If stop_at_node is set: the output of that node
            Otherwise: the normal graph output
        """
        # Reset state
        self.captured = {}
        self.stopped = False
        self.stop_output = None

        # Map kwargs onto positional args according to the GraphModule placeholders.
        # This lets us call the interpreter the same way the executor calls the model
        # (i.e., with **named_args) while keeping torch.fx.Interpreter.run signature happy.
        if kwargs:
            placeholders = [n for n in self.module.graph.nodes if n.op == "placeholder"]
            args = list(args)

            # Fill remaining positional slots from kwargs following placeholder order.
            for ph in placeholders[len(args) :]:
                name = ph.target
                if name in kwargs:
                    args.append(kwargs.pop(name))
                else:
                    raise KeyError(f"Missing input for placeholder '{name}'")

            if kwargs:
                raise TypeError(f"Unexpected kwargs for DebugInterpreter: {list(kwargs.keys())}")

            args = tuple(args)

        try:
            result = super().run(*args)
        except StopExecution as e:
            # We stopped early
            return e.output

        return result

    def run_node(self, node: Node) -> Any:
        """Execute a single node, with capture and stop support."""
        if self.stopped:
            # If we've already stopped, skip execution
            # Return a dummy value (this shouldn't be reached in normal use)
            return None

        # Execute the node normally
        result = super().run_node(node)

        # Capture if this node is in our capture set
        if node.name in self.capture_nodes:
            if isinstance(result, torch.Tensor):
                self.captured[node.name] = result.detach().clone()
            elif isinstance(result, (tuple, list)):
                # Handle tuple/list outputs (common for attention)
                self.captured[node.name] = tuple(
                    t.detach().clone() if isinstance(t, torch.Tensor) else t for t in result
                )
            else:
                self.captured[node.name] = result

        # Stop if we've reached our stopping point
        if node.name == self.stop_at_node:
            self.stopped = True
            self.stop_output = result
            # Raise to exit early
            raise StopExecution(result)

        return result


class StopExecution(Exception):
    """Exception raised to stop interpreter execution early."""

    def __init__(self, output: Any):
        self.output = output
        super().__init__("Interpreter stopped early")


def run_interpreter_to_node(
    gm: GraphModule,
    target_node: str,
    *args,
    **kwargs,
) -> Any:
    """Convenience function to run a GraphModule and stop at a specific node.

    Args:
        gm: The GraphModule to run
        target_node: The node name to stop at
        *args, **kwargs: Inputs to the graph

    Returns:
        The output of the target node
    """
    interp = DebugInterpreter(gm, stop_at_node=target_node)
    return interp.run(*args, **kwargs)


def run_interpreter_with_captures(
    gm: GraphModule,
    capture_nodes: Union[Set[str], List[str]],
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to run a GraphModule and capture specific node outputs.

    Args:
        gm: The GraphModule to run
        capture_nodes: Set of node names to capture
        *args, **kwargs: Inputs to the graph

    Returns:
        Dict mapping node names to their outputs
    """
    interp = DebugInterpreter(gm, capture_nodes=capture_nodes)
    interp.run(*args, **kwargs)
    return interp.captured
